// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/utilities.h>

#include <immersx/vascularization/global_generation.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "archive_internal.h"
#include "distribution_internal.h"
#include "forest_state.h"
#include "generation_count.h"
#include "generation_pairs.h"
#include "growth_batch.h"
#include "growth_cell_sampling.h"
#include "growth_commit.h"
#include "growth_geometry.h"
#include "growth_system.h"
#include "network_serialization.h"
#include "tissue_integration.h"

namespace ImmersX::Vascularization
{
  GlobalGenerationParameters::GlobalGenerationParameters(
    const std::string &section)
    : dealii::ParameterAcceptor(section)
  {
    add_parameter(
      "Initial terminals",
      initial_terminals,
      "First common prefix; enlarged when needed to seed every selected attachment.");
    add_parameter(
      "Topology trials",
      topology_trials,
      "Bounded subtree-swap trials after each jointly repaired three-role batch.");
    add_parameter("Random seed", random_seed);
    add_parameter("Pool attempt budget", pool_attempt_budget);
    add_parameter(
      "Maximum group records",
      maximum_group_records,
      "Temporary node, vessel and terminal records on one coordinator rank; no global-graph fallback.");
    add_parameter(
      "Initial temperature",
      initial_temperature,
      "Dimensionless factor multiplying the initially optimized batch objective in watts.");
    add_parameter(
      "Cooling factor",
      cooling_factor,
      "Explicit geometric temperature factor per trial; a search control, not a physiological constant.");
    add_parameter(
      "Minimum length",
      geometry.minimum_length_m,
      "Metres. Zero resolves the existing minimum-length and mesh-scale policy.");
    add_parameter(
      "Maximum length",
      geometry.maximum_length_m,
      "Metres. Zero resolves the global background bounding-box diagonal.");
    add_parameter(
      "Clearance",
      geometry.clearance_m,
      "Metres; at least the clearance resolved from the existing mesh-scale policy.");
    add_parameter("Maximum geometry outer iterations",
                  geometry.maximum_outer_iterations);
    add_parameter("Maximum geometry inner iterations",
                  geometry.maximum_inner_iterations);
    add_parameter("Geometry depth", geometry.geometry_depth);
    add_parameter("Geometry work", geometry.geometry_work);
    add_parameter("Constraint tolerance", geometry.constraint_tolerance);
    add_parameter("Gradient tolerance", geometry.gradient_tolerance);
    add_parameter(
      "Arterial metabolic demand",
      geometry.metabolic_demand_W_m3.at(System::arterial),
      "Metabolic volume cost in W/m^3; the default is illustrative, not patient calibration.");
    add_parameter(
      "Portal metabolic demand",
      geometry.metabolic_demand_W_m3.at(System::portal),
      "Metabolic volume cost in W/m^3; the default is illustrative, not patient calibration.");
    add_parameter(
      "Venous metabolic demand",
      geometry.metabolic_demand_W_m3.at(System::venous),
      "Metabolic volume cost in W/m^3; the default is illustrative, not patient calibration.");
  }

  void
  GlobalGenerationParameters::validate() const
  {
    AssertThrow(initial_terminals > 0 && pool_attempt_budget > 0 &&
                  maximum_group_records > 0 &&
                  std::isfinite(initial_temperature) &&
                  initial_temperature >= 0. && std::isfinite(cooling_factor) &&
                  cooling_factor > 0. && cooling_factor <= 1.,
                dealii::ExcMessage(
                  "Invalid bounded global-forest search controls."));
  }

  namespace
  {
    using namespace dealii;
    using namespace internal;

    struct Attachment
    {
      Id           terminal = invalid_id, root = invalid_id;
      Terminal     old_terminal;
      Node         cap;
      Vessel       properties;
      unsigned int coordinator = 0;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &terminal &root &old_terminal &cap &properties &coordinator;
      }
    };

    struct PoolTarget
    {
      Territory               territory;
      std::array<Point<3>, 3> positions;
      std::array<Id, 3>       terminals{}, groups{};
      Id                      ordinal = 0;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &territory.id &territory.position &territory.owner &territory
          .demand &positions &terminals &groups               &ordinal;
      }
    };

    struct TargetEdit
    {
      Id territory = invalid_id, terminal = invalid_id, node = invalid_id,
         junction = invalid_id, continuation = invalid_id, leaf = invalid_id;
      Point<3> position;
      double   demand = 0.;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &territory &terminal &node &junction &continuation &leaf &position
          &demand;
      }
    };

    using GroupViews  = std::map<Id, NetworkData>;
    using GroupPacket = std::map<unsigned int, GroupViews>;

    void
    merge(NetworkData &to, const NetworkData &from)
    {
      to.nodes.insert(from.nodes.begin(), from.nodes.end());
      to.vessels.insert(from.vessels.begin(), from.vessels.end());
      to.terminals.insert(from.terminals.begin(), from.terminals.end());
    }

    bool
    phase(bool               local,
          ValidationReport  &report,
          const std::string &message,
          MPI_Comm           comm)
    {
      const bool valid =
        Utilities::MPI::min(static_cast<unsigned int>(local && report.valid()),
                            comm) == 1;
      if (!valid)
        report.issues.push_back({"global forest", invalid_id, message});
      return valid;
    }

    Id
    maximum_id(const NetworkData &n, unsigned int kind, MPI_Comm comm)
    {
      const auto rank  = Utilities::MPI::this_mpi_process(comm);
      Id         value = 0;
      if (kind == 0)
        for (const auto &[id, x] : n.nodes)
          if (x.owner == rank)
            value = std::max(value, id);
      if (kind == 1)
        for (const auto &[id, x] : n.vessels)
          if (x.owner == rank)
            value = std::max(value, id);
      if (kind == 2)
        for (const auto &[id, x] : n.terminals)
          if (x.owner == rank)
            value = std::max(value, id);
      return Utilities::MPI::max(value, comm);
    }

    std::vector<Attachment>
    attachments(const NetworkData &original,
                const Parameters  &p,
                Id                 first_root,
                MPI_Comm           comm,
                ValidationReport  &report)
    {
      const auto rank = Utilities::MPI::this_mpi_process(comm);
      const auto size = Utilities::MPI::n_mpi_processes(comm);
      std::map<unsigned int, std::vector<std::pair<Id, Vessel>>> send;
      for (const auto &[id, v] : original.vessels)
        if (v.owner == rank)
          send[original.nodes.at(v.nodes[1]).owner].push_back({id, v});
      std::map<Id, std::vector<Vessel>> incoming;
      for (const auto &[sender, values] :
           Utilities::MPI::some_to_some(comm, send))
        for (const auto &[id, v] : values)
          incoming[v.nodes[1]].push_back(v);
      const std::array<const std::vector<Id> *, 3> selected{
        {&p.selected_terminals,
         &p.venous.selected_terminals,
         &p.portal.selected_terminals}};
      bool                    valid = true;
      std::vector<Attachment> local;
      for (const auto &[id, t] : original.terminals)
        if (t.owner == rank)
          {
            const auto s = static_cast<unsigned int>(t.system);
            if (!selected[s]->empty() &&
                std::find(selected[s]->begin(), selected[s]->end(), id) ==
                  selected[s]->end())
              continue;
            const auto node = original.nodes.find(t.node);
            valid           = valid && node != original.nodes.end() &&
                    incoming[t.node].size() == 1 && t.territory == invalid_id;
            if (node == original.nodes.end() || incoming[t.node].size() != 1)
              continue;
            valid = valid && node->second.port == Port::none &&
                    incoming[t.node].front().system == t.system;
            Attachment group;
            group.terminal     = id;
            group.old_terminal = t;
            group.cap          = node->second;
            group.properties   = incoming[t.node].front();
            local.push_back(group);
          }
      if (!phase(valid,
                 report,
                 "Invalid original attachment terminal or incoming vessel.",
                 comm))
        return {};
      std::vector<Attachment> result;
      for (const auto &part : Utilities::MPI::all_gather(comm, local))
        result.insert(result.end(), part.begin(), part.end());
      std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
        return a.terminal < b.terminal;
      });
      std::array<unsigned int, 3> counts{};
      for (std::size_t i = 0; i < result.size(); ++i)
        {
          result[i].root = first_root + i;
          // Cyclic sorted IDs avoid concentrating aligned input IDs on rank 0.
          result[i].coordinator = static_cast<unsigned int>(i % size);
          ++counts[static_cast<unsigned int>(result[i].old_terminal.system)];
        }
      for (unsigned int s = 0; s < 3; ++s)
        valid = valid && counts[s] > 0 &&
                (selected[s]->empty() || counts[s] == selected[s]->size());
      if (
        !phase(
          valid,
          report,
          "All three roles need exactly their selected real attachment groups.",
          comm))
        return {};
      return result;
    }

    std::map<Id, double>
    root_controls(const NetworkData             &network,
                  const std::vector<Attachment> &groups,
                  unsigned int                   rank)
    {
      std::map<Id, double> roots;
      for (const auto &g : groups)
        {
          const auto v = network.vessels.find(g.root);
          if (v != network.vessels.end() && v->second.owner == rank)
            roots[g.root] = g.properties.radii[1];
        }
      return roots;
    }

    std::set<Id>
    mutable_vessels(const NetworkData &network, Id first, unsigned int rank)
    {
      std::set<Id> result;
      for (const auto &[id, v] : network.vessels)
        if (v.owner == rank && id >= first)
          result.insert(id);
      return result;
    }

    ForestState
    normalize(NetworkData                   &network,
              const std::vector<Attachment> &groups,
              Id                             first,
              const Parameters              &p,
              const MeshLengthScale         &scale,
              MPI_Comm                       comm)
    {
      const auto rank  = Utilities::MPI::this_mpi_process(comm);
      auto       state = evaluate_forest(network,
                                   mutable_vessels(network, first, rank),
                                   root_controls(network, groups, rank),
                                   p,
                                   comm,
                                   scale);
      if (state.validation.valid())
        for (const auto &[id, branch] : state.owned)
          network.vessels.at(id).radii = {{branch.radius, branch.radius}};
      return state;
    }

    GroupViews
    group_views(const NetworkData             &network,
                const ForestState             &state,
                const std::vector<Attachment> &groups,
                MPI_Comm                       comm,
                std::uint64_t                  budget,
                ValidationReport              &report,
                Id                             only = invalid_id)
    {
      const auto rank = Utilities::MPI::this_mpi_process(comm);
      std::map<Id, const Attachment *> by_root;
      for (const auto &g : groups)
        by_root[g.root] = &g;
      // Reserve a conservative record bound before transferring any graph.
      std::map<unsigned int, std::uint64_t> bounds;
      for (const auto &[id, b] : state.owned)
        {
          const auto &g = *by_root.at(b.root);
          if (only == invalid_id || only == g.terminal)
            bounds[g.coordinator] += 3 + (b.terminal == invalid_id ? 0 : 2);
        }
      std::uint64_t incoming_bound = 0;
      for (const auto &[sender, value] :
           Utilities::MPI::some_to_some(comm, bounds))
        incoming_bound += value;
      if (
        !phase(
          incoming_bound <= budget,
          report,
          "Temporary group-view allocation bound exceeds Maximum group records.",
          comm))
        return {};
      GroupPacket                                            send;
      std::map<unsigned int, std::vector<std::array<Id, 3>>> requests;
      for (const auto &[id, b] : state.owned)
        {
          const auto &g = *by_root.at(b.root);
          if (only != invalid_id && only != g.terminal)
            continue;
          auto       &packet = send[g.coordinator][g.terminal];
          const auto &v      = network.vessels.at(id);
          packet.vessels[id] = v;
          for (Id node : v.nodes)
            packet.nodes[node] = network.nodes.at(node);
          if (b.terminal != invalid_id)
            requests[b.terminal_owner].push_back(
              {{b.terminal, g.terminal, g.coordinator}});
        }
      for (const auto &[source, values] :
           Utilities::MPI::some_to_some(comm, requests))
        for (const auto &v : values)
          {
            const auto &t = network.terminals.at(v[0]);
            AssertThrow(t.owner == rank,
                        ExcMessage("Missing authoritative group terminal."));
            auto &packet = send[static_cast<unsigned int>(v[2])][v[1]];
            packet.terminals[v[0]] = t;
            packet.nodes[t.node]   = network.nodes.at(t.node);
          }
      GroupViews result;
      for (const auto &[sender, packet] :
           Utilities::MPI::some_to_some(comm, send))
        for (const auto &[group, data] : packet)
          merge(result[group], data);
      return result;
    }

    bool
    view_budget(const GroupViews                 &views,
                const GlobalGenerationParameters &c,
                GlobalGenerationReport           &stats,
                ValidationReport                 &report,
                MPI_Comm                          comm)
    {
      std::uint64_t records = 0;
      for (const auto &[id, n] : views)
        records += n.nodes.size() + n.vessels.size() + n.terminals.size();
      auto all = Utilities::MPI::all_gather(comm, records);
      if (stats.peak_group_records_by_rank.empty())
        stats.peak_group_records_by_rank.resize(all.size(), 0);
      for (unsigned int i = 0; i < all.size(); ++i)
        stats.peak_group_records_by_rank[i] =
          std::max(stats.peak_group_records_by_rank[i], all[i]);
      return phase(
        records <= c.maximum_group_records,
        report,
        "Temporary attachment-group view exceeds Maximum group records; no replicated-forest fallback.",
        comm);
    }

    std::vector<PoolTarget>
    sample_pool(const NetworkData                &original,
                const std::vector<Attachment>    &groups,
                const Parameters                 &p,
                const GlobalGenerationParameters &c,
                const GlobalGeometryParameters   &gc,
                unsigned int                      target,
                unsigned int                      initial,
                Id                                first_terminal,
                BackgroundGeometry               &geometry,
                GlobalGenerationReport           &stats,
                ValidationReport                 &report,
                MPI_Comm                          comm)
    {
      const auto            rank  = Utilities::MPI::this_mpi_process(comm);
      const auto            size  = Utilities::MPI::n_mpi_processes(comm);
      const auto           &scale = *geometry.length_scale;
      std::array<double, 3> radius{};
      for (const auto &g : groups)
        radius[static_cast<unsigned int>(g.old_terminal.system)] =
          std::max(radius[static_cast<unsigned int>(g.old_terminal.system)],
                   g.properties.radii[1]);
      stats.endpoint_radius_bounds_m = radius;
      const auto     &sampler = geometry.sample_cells(p.density_beta, comm);
      std::mt19937_64 selection(p.random_seed + 104729ull * rank);
      std::mt19937    physical(p.random_seed + 65537u * rank);
      std::vector<PoolTarget> owned;
      Id                      count = 0;
      for (unsigned int attempt = 0;
           count < target && attempt < c.pool_attempt_budget;
           ++attempt)
        {
          ++stats.sampled_points;
          const auto producer = attempt % size;
          Point<3>   centre;
          int        status = 0;
          double     h      = 0.;
          if (rank == producer)
            try
              {
                const auto point = sampler.sample(selection, physical);
                if (point)
                  {
                    centre          = *point;
                    const auto cell = geometry.containing(centre);
                    if (cell)
                      {
                        h      = (*cell)->diameter();
                        status = 1;
                      }
                  }
              }
            catch (const std::exception &)
              {
                status = -1;
              }
          status = Utilities::MPI::broadcast(comm, status, producer);
          if (status < 0)
            {
              phase(
                false,
                report,
                "Physical common-pool sampling failed on its producer; no fallback density was used.",
                comm);
              return owned;
            }
          if (status == 0)
            continue;
          centre = Utilities::MPI::broadcast(comm, centre, producer);
          h      = Utilities::MPI::broadcast(comm, h, producer);
          const double safety = 128. * std::numeric_limits<double>::epsilon() *
                                std::max(1., centre.norm());
          const double sepP =
            safety +
            std::max(scale.physical_length(p.portal.terminal_separation, h),
                     radius[0] + radius[2] + gc.clearance_m);
          const double sepH =
            safety +
            std::max(scale.physical_length(p.venous.terminal_separation, h),
                     radius[0] + radius[1] + gc.clearance_m);
          std::array<Point<3>, 3> points{{centre, centre, centre}};
          points[2][0] += sepP;
          points[1][0] -= sepH;
          bool admissible = true;
          for (unsigned int s = 0; s < 3; ++s)
            {
              const bool inside = geometry.containing(points[s]).has_value();
              const bool anywhere =
                Utilities::MPI::max(static_cast<unsigned int>(inside), comm) ==
                1;
              admissible =
                admissible && anywhere &&
                geometry.boundary_separated({points[s], points[s]},
                                            radius[s] + gc.clearance_m,
                                            gc.geometry_depth,
                                            gc.geometry_work);
              for (const auto &[id, v] : original.vessels)
                if (v.owner == rank)
                  {
                    const double reach = radius[s] +
                                         std::max(v.radii[0], v.radii[1]) +
                                         gc.clearance_m;
                    if (point_distance_square(
                          points[s],
                          {original.nodes.at(v.nodes[0]).position,
                           original.nodes.at(v.nodes[1]).position}) <=
                        reach * reach)
                      admissible = false;
                  }
              for (const auto &old : owned)
                for (unsigned int t = 0; t < 3; ++t)
                  {
                    const double reach = radius[s] + radius[t] + gc.clearance_m;
                    if (points[s].distance_square(old.positions[t]) <=
                        reach * reach)
                      admissible = false;
                  }
            }
          if (Utilities::MPI::min(static_cast<unsigned int>(admissible),
                                  comm) == 0)
            continue;
          if (rank == producer)
            {
              PoolTarget t;
              t.ordinal   = count;
              t.territory = {count, centre, rank, 1.};
              t.positions = points;
              for (unsigned int s = 0; s < 3; ++s)
                {
                  t.terminals[s] = first_terminal + 3 * count + s;
                  t.groups[s]    = invalid_id;
                }
              owned.push_back(t);
            }
          ++count;
        }
      if (
        !phase(
          count == target,
          report,
          "Upfront common-pool sampling exhausted its finite budget; fixed endpoints were not moved.",
          comm))
        return owned;
      // Every selected group receives a distinct target within the SAME initial
      // common prefix. Matching in one role never changes another role's IDs.
      for (const auto &g : groups)
        {
          const auto s = static_cast<unsigned int>(g.old_terminal.system);
          std::pair<double, Id> best{std::numeric_limits<double>::infinity(),
                                     invalid_id};
          for (const auto &t : owned)
            if (t.ordinal < initial && t.groups[s] == invalid_id)
              best = std::min(best,
                              std::make_pair(t.positions[s].distance_square(
                                               g.cap.position),
                                             t.ordinal));
          for (const auto &candidate : Utilities::MPI::all_gather(comm, best))
            best = std::min(best, candidate);
          if (!phase(best.second != invalid_id,
                     report,
                     "No unique initial target for a selected attachment.",
                     comm))
            return owned;
          for (auto &t : owned)
            if (t.ordinal == best.second)
              t.groups[s] = g.terminal;
        }
      for (auto &t : owned)
        for (unsigned int s = 0; s < 3; ++s)
          if (t.groups[s] == invalid_id)
            {
              std::pair<double, Id> best{
                std::numeric_limits<double>::infinity(), invalid_id};
              for (const auto &g : groups)
                if (static_cast<unsigned int>(g.old_terminal.system) == s)
                  best = std::min(best,
                                  std::make_pair(t.positions[s].distance_square(
                                                   g.cap.position),
                                                 g.terminal));
              t.groups[s] = best.second;
            }
      std::vector<TissueCentre> centres;
      for (const auto &t : owned)
        centres.push_back(
          {t.territory.id, t.terminals[0], t.territory.position, rank});
      TissueDemandIntegration integration(
        geometry, p.tissue_demand, p.tissue_integration, target, centres, comm);
      auto preview = integration.preview({}, 0);
      if (!phase(preview.validation.valid(),
                 report,
                 "Upfront pool demand integration failed.",
                 comm))
        return owned;
      integration.prepare_adopt(preview);
      integration.adopt();
      std::map<unsigned int, std::vector<std::pair<Id, unsigned int>>>
        registration;
      for (const auto &t : owned)
        registration[t.terminals[0] % size].push_back({t.terminals[0], rank});
      std::map<Id, unsigned int> directory;
      for (const auto &[sender, values] :
           Utilities::MPI::some_to_some(comm, registration))
        for (const auto &v : values)
          directory.emplace(v);
      std::map<unsigned int, std::vector<TerminalDemandUpdate>> updates;
      for (const auto &d : integration.owned_demands())
        updates[directory.at(d.terminal)].push_back(d);
      std::map<Id, double> demand;
      for (const auto &[sender, values] :
           Utilities::MPI::some_to_some(comm, updates))
        for (const auto &d : values)
          demand[d.terminal] = d.demand;
      bool valid = true;
      for (auto &t : owned)
        {
          const auto value = demand.find(t.terminals[0]);
          valid            = valid && value != demand.end() &&
                  std::isfinite(value->second) && value->second > 0.;
          if (value != demand.end())
            t.territory.demand = value->second;
        }
      phase(
        valid,
        report,
        "Every upfront common territory needs a finite positive integrated demand.",
        comm);
      return owned;
    }

    void
    add_target(NetworkData &view, const Attachment &group, const TargetEdit &t)
    {
      const auto writer = group.coordinator;
      for (auto &[id, v] : view.vessels)
        v.owner = writer;
      for (auto &[id, n] : view.nodes)
        if (id != group.old_terminal.node)
          n.owner = writer;
      for (auto &[id, terminal] : view.terminals)
        terminal.owner = writer;
      if (view.vessels.empty())
        {
          view.nodes[group.old_terminal.node] = group.cap;
          view.nodes[t.node]                  = {t.position,
                                                 writer,
                                                 Port::none,
                                                 Provenance::generated};
          auto v                              = group.properties;
          v.nodes = {{group.old_terminal.node, t.node}};
          v.radii = {{group.properties.radii[1], group.properties.radii[1]}};
          v.owner = writer;
          v.provenance               = Provenance::generated;
          view.vessels[group.root]   = v;
          view.terminals[t.terminal] = {
            t.node, t.territory, group.old_terminal.system, writer, t.demand};
          return;
        }
      std::pair<double, Id> nearest{std::numeric_limits<double>::infinity(),
                                    invalid_id};
      for (const auto &[id, v] : view.vessels)
        nearest =
          std::min(nearest,
                   std::make_pair(point_distance_square(
                                    t.position,
                                    {view.nodes.at(v.nodes[0]).position,
                                     view.nodes.at(v.nodes[1]).position}),
                                  id));
      const auto &old   = view.vessels.at(nearest.second);
      const auto  a     = view.nodes.at(old.nodes[0]).position,
                 b      = view.nodes.at(old.nodes[1]).position;
      Point<3> junction = (a + b + t.position) / 3.;
      if (junction.distance(a) == 0. || junction.distance(b) == 0. ||
          junction.distance(t.position) == 0.)
        junction = (a + 2. * b + 4. * t.position) / 7.;
      GrowthSplit split;
      split.vessel            = nearest.second;
      split.owner             = writer;
      split.old_nodes         = old.nodes;
      split.old_points        = {{a, b}};
      split.junction          = t.junction;
      split.tip               = t.node;
      split.junction_position = junction;
      split.tip_position      = t.position;
      split.continuation      = t.continuation;
      split.leaf              = t.leaf;
      split.terminal          = t.terminal;
      split.territory         = t.territory;
      split.demand            = t.demand;
      AssertThrow(apply_growth_split(view, split),
                  ExcMessage("Invalid nearest-edge topology insertion."));
    }

    NetworkData
    assemble_draft(const NetworkData             &original,
                   const GroupViews              &views,
                   const std::vector<Attachment> &groups)
    {
      NetworkData result = original;
      for (const auto &g : groups)
        result.terminals.erase(g.terminal);
      for (const auto &[id, view] : views)
        merge(result, view);
      return result;
    }

    std::map<Id, double>
    flows(const ForestState &state)
    {
      std::map<Id, double> result;
      for (const auto &[id, b] : state.owned)
        result[id] = b.flow;
      return result;
    }

    bool
    propose_swap(const NetworkData &view,
                 const Attachment  &group,
                 std::mt19937_64   &rng,
                 NetworkData       &edits)
    {
      std::map<Id, std::vector<Id>> outgoing;
      std::map<Id, Id>              incoming;
      for (const auto &[id, v] : view.vessels)
        {
          outgoing[v.nodes[0]].push_back(id);
          incoming[v.nodes[1]] = id;
        }
      std::vector<Id> eligible;
      for (const auto &[id, v] : view.vessels)
        if (v.nodes[0] != group.old_terminal.node &&
            outgoing[v.nodes[0]].size() == 2)
          eligible.push_back(id);
      if (eligible.size() < 2)
        return false;
      const auto ancestor = [&](Id a, Id b) {
        Id node = view.vessels.at(b).nodes[0];
        while (incoming.count(node))
          {
            const auto id = incoming.at(node);
            if (id == a)
              return true;
            node = view.vessels.at(id).nodes[0];
          }
        return false;
      };
      std::uniform_int_distribution<std::size_t> choose(0, eligible.size() - 1);
      for (unsigned int attempt = 0; attempt < 16; ++attempt)
        {
          const Id a = eligible[choose(rng)], b = eligible[choose(rng)];
          if (a == b || ancestor(a, b) || ancestor(b, a))
            continue;
          auto va = view.vessels.at(a), vb = view.vessels.at(b);
          if (va.nodes[0] == vb.nodes[0])
            continue;
          const double old_a =
            view.nodes.at(va.nodes[0])
              .position.distance(view.nodes.at(va.nodes[1]).position);
          const double old_b =
            view.nodes.at(vb.nodes[0])
              .position.distance(view.nodes.at(vb.nodes[1]).position);
          std::swap(va.nodes[0], vb.nodes[0]);
          const double new_a =
            view.nodes.at(va.nodes[0])
              .position.distance(view.nodes.at(va.nodes[1]).position);
          const double new_b =
            view.nodes.at(vb.nodes[0])
              .position.distance(view.nodes.at(vb.nodes[1]).position);
          if (!(new_a > 0. && new_b > 0. && new_a < 2 * old_a &&
                new_b < 2 * old_b))
            continue;
          edits.vessels = {{a, va}, {b, vb}};
          for (const auto &[id, v] : edits.vessels)
            for (Id node : v.nodes)
              edits.nodes[node] = view.nodes.at(node);
          return true;
        }
      return false;
    }

    NetworkData
    apply_owned_edits(const NetworkData &snapshot,
                      const NetworkData &edits,
                      MPI_Comm           comm)
    {
      const auto rank = Utilities::MPI::this_mpi_process(comm);
      std::map<unsigned int, NetworkData> send;
      for (const auto &[id, v] : edits.vessels)
        {
          auto &packet       = send[v.owner];
          packet.vessels[id] = v;
          for (Id node : v.nodes)
            packet.nodes[node] = edits.nodes.at(node);
        }
      NetworkData candidate = snapshot;
      for (const auto &[sender, data] :
           Utilities::MPI::some_to_some(comm, send))
        {
          for (const auto &[id, v] : data.vessels)
            {
              AssertThrow(candidate.vessels.count(id) &&
                            candidate.vessels.at(id).owner == rank,
                          ExcMessage(
                            "Topology edit lost its native writing owner."));
              candidate.vessels.at(id) = v;
            }
          candidate.nodes.insert(data.nodes.begin(), data.nodes.end());
        }
      return candidate;
    }

    bool
    certify_native(const NetworkData  &network,
                   Id                  first,
                   const Parameters   &p,
                   BackgroundGeometry &geometry,
                   ValidationReport   &report,
                   MPI_Comm            comm)
    {
      const auto                 rank = Utilities::MPI::this_mpi_process(comm);
      std::vector<BatchProposal> checks;
      for (const auto &[id, v] : network.vessels)
        if (id >= first && v.owner == rank)
          {
            BatchProposal check;
            check.id            = id;
            check.owners[id]    = rank;
            check.old_nodes[id] = v.nodes;
            check.changed.push_back({id,
                                     v.nodes,
                                     {{network.nodes.at(v.nodes[0]).position,
                                       network.nodes.at(v.nodes[1]).position}},
                                     std::max(v.radii[0], v.radii[1])});
            for (Id node : v.nodes)
              {
                const auto &n = network.nodes.at(node);
                if (n.inward_port_normal.norm_square() > 0.)
                  {
                    check.port_owners[node] = n.owner;
                    check.normals[node]     = n.inward_port_normal;
                  }
              }
            checks.push_back(std::move(check));
          }
      const auto certified = select_growth_batch(
        geometry,
        network,
        checks,
        p,
        comm,
        [&](Id id) { return network.vessels.at(id).radii; },
        true);
      bool valid = true;
      for (const auto &[id, accepted] : certified.local)
        valid = valid && accepted;
      return phase(
        valid,
        report,
        "Joint forest geometry failed the native generation certificate.",
        comm);
    }

    GlobalGeometryParameters
    resolved_geometry(const Parameters                 &p,
                      const GlobalGenerationParameters &c,
                      const Triangulation<3>           &bg,
                      const MeshLengthScale            &scale,
                      MPI_Comm                          comm)
    {
      auto     result = c.geometry;
      Point<3> lo, hi;
      for (unsigned int d = 0; d < 3; ++d)
        {
          lo[d] = std::numeric_limits<double>::infinity();
          hi[d] = -lo[d];
        }
      for (const auto &cell : bg.active_cell_iterators())
        if (cell->is_locally_owned())
          for (unsigned int v = 0; v < cell->n_vertices(); ++v)
            for (unsigned int d = 0; d < 3; ++d)
              {
                lo[d] = std::min(lo[d], cell->vertex(v)[d]);
                hi[d] = std::max(hi[d], cell->vertex(v)[d]);
              }
      for (unsigned int d = 0; d < 3; ++d)
        {
          lo[d] = Utilities::MPI::min(lo[d], comm);
          hi[d] = Utilities::MPI::max(hi[d], comm);
        }
      if (result.minimum_length_m == 0.)
        result.minimum_length_m =
          scale.physical_length(p.minimum_length, scale.minimum());
      if (result.maximum_length_m == 0.)
        result.maximum_length_m = lo.distance(hi);
      result.clearance_m =
        std::max(result.clearance_m,
                 scale.physical_length(p.clearance, scale.maximum()));
      bool valid = true;
      try
        {
          result.validate();
        }
      catch (const std::exception &)
        {
          valid = false;
        }
      AssertThrow(
        Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1,
        ExcMessage(
          "Resolved global geometry SI bounds and metabolic controls are invalid."));
      return result;
    }

    NetworkData
    redistribute_draft(const Triangulation<3>         &bg,
                       const Mapping<3>               &mapping,
                       const NetworkData              &n,
                       BackgroundGeometry             &geometry,
                       const GlobalGeometryParameters &controls,
                       MPI_Comm                        comm)
    {
      return redistribute_with_owners(
        bg, mapping, n, controls.clearance_m, comm, {}, &geometry);
    }

    bool
    certified_physiology(const NetworkData          &network,
                         const ForestState          &state,
                         const std::map<Id, double> &design_flows,
                         ValidationReport           &report,
                         MPI_Comm                    comm)
    {
      bool valid =
        state.validation.valid() && design_flows.size() == state.owned.size();
      for (const auto &[id, b] : state.owned)
        {
          const long double tolerance = 64.L *
                                        std::numeric_limits<double>::epsilon() *
                                        (2.L + state.upward_levels);
          for (double r : network.vessels.at(id).radii)
            valid = valid && std::abs(static_cast<long double>(r) - b.radius) /
                                 std::max<long double>(r, b.radius) <=
                               tolerance;
          const auto design = design_flows.find(id);
          valid             = valid && std::isfinite(b.flow) && b.flow > 0. &&
                  design != design_flows.end();
          if (design != design_flows.end())
            valid =
              valid && std::isfinite(design->second) && design->second > 0. &&
              std::abs(static_cast<long double>(design->second) - b.flow) /
                  std::max<long double>(design->second, b.flow) <=
                tolerance;
        }
      return phase(
        valid,
        report,
        "Geometry changed fixed-flow Murray state or produced a nonpositive design flow.",
        comm);
    }
  } // namespace

  GenerationResult
  generate_global(const dealii::Triangulation<3>   &background,
                  const dealii::Mapping<3>         &mapping,
                  const NetworkData                &input,
                  const Parameters                 &p,
                  const GlobalGenerationParameters &c,
                  MPI_Comm                          comm)
  {
    using namespace dealii;
    using namespace internal;
    const auto rank           = Utilities::MPI::this_mpi_process(comm);
    const auto size           = Utilities::MPI::n_mpi_processes(comm);
    bool       controls_valid = true;
    try
      {
        c.validate();
        p.validate(false);
      }
    catch (const std::exception &)
      {
        controls_valid = false;
      }
    controls_valid = controls_valid && p.venous.enabled && p.portal.enabled &&
                     std::isfinite(p.reference_flow) && p.reference_flow > 0.;
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(controls_valid), comm) == 1,
      ExcMessage(
        "Global forest generation needs valid matching controls, all three roles and positive SI reference flow."));
    const auto own_controls = std::make_pair(
      std::array<std::uint64_t, 7>{
        {c.initial_terminals,
         c.topology_trials,
         c.random_seed,
         c.pool_attempt_budget,
         c.maximum_group_records,
         p.terminal_count,
         static_cast<std::uint64_t>(static_cast<bool>(c.stage_callback))}},
      std::make_pair(
        std::array<double, 9>{{c.initial_temperature,
                               c.cooling_factor,
                               c.geometry.minimum_length_m,
                               c.geometry.maximum_length_m,
                               c.geometry.clearance_m,
                               p.minimum_length,
                               p.clearance,
                               p.reference_flow,
                               p.portal_flow_fraction}},
        std::make_pair(
          std::array<std::vector<Id>, 3>{{p.selected_terminals,
                                          p.venous.selected_terminals,
                                          p.portal.selected_terminals}},
          p.terminal_count_multiplier_factor)));
    const auto geometry_controls = std::make_pair(
      std::array<unsigned int, 4>{{c.geometry.maximum_outer_iterations,
                                   c.geometry.maximum_inner_iterations,
                                   c.geometry.geometry_depth,
                                   c.geometry.geometry_work}},
      std::make_pair(std::array<double, 2>{{c.geometry.constraint_tolerance,
                                            c.geometry.gradient_tolerance}},
                     c.geometry.metabolic_demand_W_m3));
    const auto sampling_controls =
      std::make_pair(std::array<double, 3>{{p.density_beta,
                                            p.portal.terminal_separation,
                                            p.venous.terminal_separation}},
                     p.random_seed);
    bool matching = true;
    for (const auto &other :
         Utilities::MPI::all_gather(comm, sampling_controls))
      matching = matching && other == sampling_controls;
    for (const auto &other : Utilities::MPI::all_gather(comm, own_controls))
      matching = matching && other == own_controls;
    for (const auto &other :
         Utilities::MPI::all_gather(comm, geometry_controls))
      matching = matching && other == geometry_controls;
    AssertThrow(matching,
                ExcMessage("Global forest controls must match on all ranks."));

    GenerationResult       result;
    GlobalGenerationReport stats;
    result.network                = input;
    result.base_terminal_count    = p.terminal_count;
    const unsigned int target     = resolve_terminal_count(background, p, comm);
    result.initial_terminal_count = target;
    result.requested_terminal_count = target;
    const unsigned int maximum =
      p.calibration.maximum_terminals == 0 ?
        std::max(target, p.calibration.minimum_terminals) :
        p.calibration.maximum_terminals;
    if (!phase(
          target >= p.calibration.minimum_terminals && target <= maximum,
          result.validation,
          "The common-pool count lies outside the absolute terminal limits.",
          comm))
      return result;
    if (!phase(validate_data(input, size).valid(),
               result.validation,
               "Invalid original physical records.",
               comm))
      return result;
    BackgroundGeometry geometry(background, mapping);
    geometry.configure_length_scale(p.length_scale, comm);
    result.length_scale = *geometry.length_scale;
    auto gc = resolved_geometry(p, c, background, result.length_scale, comm);
    stats.minimum_length_m = gc.minimum_length_m;
    stats.maximum_length_m = gc.maximum_length_m;
    const auto original =
      redistribute_draft(background, mapping, input, geometry, gc, comm);
    result.network = original;
    validate_archive_data(original, {}, {}, comm);
    const Id original_node_max     = maximum_id(original, 0, comm);
    const Id original_vessel_max   = maximum_id(original, 1, comm);
    const Id original_terminal_max = maximum_id(original, 2, comm);
    if (!phase(original_node_max < invalid_id - 6ull * target &&
                 original_vessel_max < invalid_id - 9ull * target &&
                 original_terminal_max < invalid_id - 3ull * target,
               result.validation,
               "Insufficient persistent physical ID space.",
               comm))
      return result;
    const Id first_root     = original_vessel_max + 1,
             first_terminal = original_terminal_max + 1;
    const auto groups =
      attachments(original, p, first_root, comm, result.validation);
    if (!result.validation.valid())
      return result;
    std::array<unsigned int, 3>      group_counts{};
    std::map<Id, const Attachment *> group_by;
    for (const auto &g : groups)
      {
        ++group_counts[static_cast<unsigned int>(g.old_terminal.system)];
        group_by[g.terminal] = &g;
      }
    const auto required =
      *std::max_element(group_counts.begin(), group_counts.end());
    if (!phase(target >= required,
               result.validation,
               "More selected attachments than common targets.",
               comm))
      return result;
    const auto first_count =
      std::min(target, std::max(c.initial_terminals, required));
    auto pool             = sample_pool(original,
                            groups,
                            p,
                            c,
                            gc,
                            target,
                            first_count,
                            first_terminal,
                            geometry,
                            stats,
                            result.validation,
                            comm);
    stats.upfront_targets = target;
    result.sampled_points = stats.sampled_points;
    if (!result.validation.valid())
      return result;
    ForestState  state;
    unsigned int previous = 0, count = first_count;
    const auto   refresh_territories = [&]() {
      result.territories.clear();
      // The logical centre follows its fixed arterial endpoint's native owner,
      // including a point on a background partition boundary.
      for (const auto &[id, t] : result.network.terminals)
        if (t.owner == rank && t.system == System::arterial &&
            t.territory != invalid_id)
          result.territories.push_back(
            {t.territory,
             result.network.nodes.at(t.node).position,
             rank,
             t.demand});
    };
    const auto publish = [&](GlobalStage stage) {
      result.global_generation = stats;
      if (c.stage_callback)
        {
          auto diagnostic = result;
          diagnostic.validation.issues.push_back(
            {"global forest",
             invalid_id,
             "Intermediate global-forest stage; full target and final reference/discretization validation are pending."});
          c.stage_callback(diagnostic, stage);
        }
    };
    while (true)
      {
        GroupViews views;
        if (previous > 0)
          views = group_views(result.network,
                              state,
                              groups,
                              comm,
                              c.maximum_group_records,
                              result.validation);
        if (!result.validation.valid() ||
            !view_budget(views, c, stats, result.validation, comm))
          return result;
        const Id node_begin = maximum_id(result.network, 0, comm) + 1;
        const Id vessel_begin =
          std::max(maximum_id(result.network, 1, comm) + 1,
                   first_root + groups.size());
        const Id needed = 6ull * (count - previous);
        if (!phase(node_begin < invalid_id - needed &&
                     vessel_begin < invalid_id - needed,
                   result.validation,
                   "No fresh IDs for a complete common batch.",
                   comm))
          return result;
        std::map<unsigned int, std::map<Id, std::vector<TargetEdit>>> send;
        for (const auto &t : pool)
          if (t.ordinal >= previous && t.ordinal < count)
            for (unsigned int s = 0; s < 3; ++s)
              {
                const auto &g      = *group_by.at(t.groups[s]);
                const Id    offset = 6 * (t.ordinal - previous) + 2 * s;
                TargetEdit  edit;
                edit.territory    = t.territory.id;
                edit.terminal     = t.terminals[s];
                edit.node         = node_begin + offset;
                edit.junction     = node_begin + offset + 1;
                edit.continuation = vessel_begin + offset;
                edit.leaf         = vessel_begin + offset + 1;
                edit.position     = t.positions[s];
                edit.demand       = t.territory.demand;
                send[g.coordinator][g.terminal].push_back(edit);
              }
        std::uint64_t view_records = 0;
        for (const auto &[id, view] : views)
          view_records +=
            view.nodes.size() + view.vessels.size() + view.terminals.size();
        bool within_budget = true;
        for (const auto &[sender, packets] :
             Utilities::MPI::some_to_some(comm, send))
          for (const auto &[group, edits] : packets)
            for (const auto &edit : edits)
              {
                const std::uint64_t increment =
                  views[group].vessels.empty() ? 4 : 5;
                if (view_records > c.maximum_group_records ||
                    increment > c.maximum_group_records - view_records)
                  {
                    within_budget = false;
                    continue;
                  }
                add_target(views[group], *group_by.at(group), edit);
                view_records += increment;
              }
        if (!phase(
              within_budget,
              result.validation,
              "Common batch exceeds the temporary group-view record budget.",
              comm))
          return result;
        if (!result.validation.valid() ||
            !view_budget(views, c, stats, result.validation, comm))
          return result;
        result.network =
          redistribute_draft(background,
                             mapping,
                             assemble_draft(original, views, groups),
                             geometry,
                             gc,
                             comm);
        state = normalize(
          result.network, groups, first_root, p, result.length_scale, comm);
        result.validation = state.validation;
        if (!phase(result.validation.valid(),
                   result.validation,
                   "Draft forest physics failed.",
                   comm))
          return result;
        result.network = redistribute_draft(
          background, mapping, result.network, geometry, gc, comm);
        result.effective_terminal_count = count;
        result.accepted_insertions =
          3 * count - static_cast<unsigned int>(groups.size());
        refresh_territories();
        GlobalGenerationBatch batch;
        batch.terminals = count;
        stats.history.push_back(batch);
        publish(GlobalStage::draft);

        auto optimized = optimize_forest_geometry(background,
                                                  mapping,
                                                  original,
                                                  result.network,
                                                  flows(state),
                                                  gc,
                                                  comm);
        ++stats.history.back().geometry_calls;
        result.global_geometry = optimized.report;
        const bool optimized_ok =
          Utilities::MPI::min(static_cast<unsigned int>(
                                optimized.validation.valid() &&
                                optimized.report.certified &&
                                std::isfinite(optimized.report.objective_W)),
                              comm) == 1;
        if (!optimized_ok)
          {
            const auto owned_count = [&]() {
              std::uint64_t n = 0;
              for (const auto &[id, v] : optimized.network.vessels)
                n += v.owner == rank;
              return Utilities::MPI::sum(n, comm);
            }();
            if (owned_count > 0)
              result.network = std::move(optimized.network);
            result.validation = optimized.validation;
            phase(false,
                  result.validation,
                  "Joint geometry failed; no complete forest was adopted.",
                  comm);
            publish(GlobalStage::draft);
            result.global_generation = stats;
            return result;
          }
        result.network = std::move(optimized.network);
        state =
          evaluate_forest(result.network,
                          mutable_vessels(result.network, first_root, rank),
                          root_controls(result.network, groups, rank),
                          p,
                          comm,
                          result.length_scale);
        if (!certified_physiology(result.network,
                                  state,
                                  optimized.design_flows,
                                  result.validation,
                                  comm) ||
            !certify_native(
              result.network, first_root, p, geometry, result.validation, comm))
          return result;
        double energy                            = optimized.report.objective_W;
        stats.history.back().initial_objective_W = energy;
        NetworkData     best_network             = result.network;
        ForestState     best_state               = state;
        auto            best_report              = optimized.report;
        double          best_energy              = energy;
        std::mt19937_64 rng(c.random_seed + 104729ull * count);
        std::uniform_real_distribution<double> uniform(0., 1.);
        for (unsigned int trial = 0; trial < c.topology_trials; ++trial)
          {
            ++stats.history.back().topology_trials;
            const auto &group = groups[trial % groups.size()];
            auto        local = group_views(result.network,
                                     state,
                                     groups,
                                     comm,
                                     c.maximum_group_records,
                                     result.validation,
                                     group.terminal);
            if (!result.validation.valid() ||
                !view_budget(local, c, stats, result.validation, comm))
              return result;
            NetworkData edits;
            bool        proposed = false;
            if (rank == group.coordinator)
              proposed =
                propose_swap(local.at(group.terminal), group, rng, edits);
            if (Utilities::MPI::sum(static_cast<unsigned int>(proposed),
                                    comm) == 0)
              continue;
            auto candidate =
              redistribute_draft(background,
                                 mapping,
                                 apply_owned_edits(result.network, edits, comm),
                                 geometry,
                                 gc,
                                 comm);
            auto trial_state = normalize(
              candidate, groups, first_root, p, result.length_scale, comm);
            if (Utilities::MPI::min(static_cast<unsigned int>(
                                      trial_state.validation.valid()),
                                    comm) == 0)
              continue;
            candidate = redistribute_draft(
              background, mapping, candidate, geometry, gc, comm);
            auto repaired = optimize_forest_geometry(background,
                                                     mapping,
                                                     original,
                                                     candidate,
                                                     flows(trial_state),
                                                     gc,
                                                     comm);
            ++stats.history.back().geometry_calls;
            bool valid = repaired.validation.valid() &&
                         repaired.report.certified &&
                         std::isfinite(repaired.report.objective_W);
            valid =
              Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1;
            if (!valid)
              continue;
            trial_state = evaluate_forest(
              repaired.network,
              mutable_vessels(repaired.network, first_root, rank),
              root_controls(repaired.network, groups, rank),
              p,
              comm,
              result.length_scale);
            ValidationReport trial_validation;
            if (!certified_physiology(repaired.network,
                                      trial_state,
                                      repaired.design_flows,
                                      trial_validation,
                                      comm) ||
                !certify_native(repaired.network,
                                first_root,
                                p,
                                geometry,
                                trial_validation,
                                comm))
              continue;
            const double next_energy = repaired.report.objective_W;
            const double temperature =
              c.initial_temperature *
              std::max(stats.history.back().initial_objective_W,
                       std::numeric_limits<double>::min()) *
              std::pow(c.cooling_factor, trial);
            double u = 0.;
            if (rank == 0)
              u = uniform(rng);
            u = Utilities::MPI::broadcast(comm, u, 0);
            const bool accepted =
              next_energy <= energy ||
              (temperature > 0. &&
               u < std::exp((energy - next_energy) / temperature));
            if (accepted)
              {
                ++stats.history.back().topology_accepted;
                result.network         = std::move(repaired.network);
                state                  = std::move(trial_state);
                energy                 = next_energy;
                result.global_geometry = repaired.report;
                if (energy < best_energy)
                  {
                    best_network = result.network;
                    best_state   = state;
                    best_report  = repaired.report;
                    best_energy  = energy;
                  }
              }
          }
        result.network                          = std::move(best_network);
        state                                   = std::move(best_state);
        result.global_geometry                  = best_report;
        stats.history.back().best_objective_W   = best_energy;
        stats.history.back().geometry_certified = true;
        refresh_territories();
        publish(GlobalStage::certified);
        if (count == target)
          break;
        previous = count;
        count    = static_cast<unsigned int>(
          std::min<std::uint64_t>(target, 2ull * count));
      }
    result.validation.issues.clear();
    result.target_terminal_flows = state.terminal_flows;
    for (const auto &group : groups)
      {
        if (group.old_terminal.owner == rank)
          result.attachment_roots[group.terminal] = group.root;
        const auto branch = state.owned.find(group.root);
        if (branch != state.owned.end())
          result.target_attachment_flows[group.root] = branch->second.flow;
      }
    pair_territories(result, p, comm);
    if (Utilities::MPI::min(
          static_cast<unsigned int>(result.validation.valid()), comm) == 0)
      {
        phase(false,
              result.validation,
              "Final reference/compartment validation failed.",
              comm);
        return result;
      }
    result.discretization =
      discretize(background, mapping, result.network, p.discretization, comm);
    result.validation.issues.insert(
      result.validation.issues.end(),
      result.discretization.validation.issues.begin(),
      result.discretization.validation.issues.end());
    if (!phase(result.validation.valid(),
               result.validation,
               "Final discretization failed.",
               comm))
      return result;
    result.tissue              = measure_tissue(background,
                                   mapping,
                                   result.network,
                                   result.discretization,
                                   p.coverage,
                                   comm,
                                   result.length_scale);
    result.normalized_geometry = measure_scaled_vessels(
      background, mapping, result.network, result.length_scale, comm);
    const auto  &cal = p.calibration;
    const double error =
      cal.volume_fraction > 0. ?
        std::abs(result.tissue.volume_fraction / cal.volume_fraction - 1.) :
        0.;
    phase(error <= cal.relative_tolerance &&
            (cal.minimum_coverage == 0. ||
             (result.tissue.coverage_evaluated &&
              result.tissue.triple_coverage >= cal.minimum_coverage)),
          result.validation,
          "Fixed common pool failed volume or triple-coverage calibration.",
          comm);
    result.calibration.status         = result.validation.valid() ?
                                          CalibrationStatus::converged :
                                          CalibrationStatus::exhausted;
    result.calibration.sampled_points = result.sampled_points;
    result.calibration.attempts.push_back(
      {target, result.tissue, result.validation, result.sampled_points});
    result.global_generation = std::move(stats);
    return result;
  }
} // namespace ImmersX::Vascularization
