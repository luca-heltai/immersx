// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_start.h"

#include <deal.II/base/function_parser.h>

#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/distribution.h>

#include "growth_batch.h"
namespace ImmersX::Vascularization::internal
{
  std::vector<dealii::Point<3>>
  initial_root_directions(
    const dealii::Point<3>                &position,
    const dealii::Point<3>                &cell_center,
    const std::vector<dealii::Point<3>>   &cell_vertices,
    const dealii::Point<3>                &explicit_direction,
    const std::optional<dealii::Point<3>> &incoming_direction,
    const Territory                       *shared,
    double                                 separation)
  {
    using namespace dealii;
    std::vector<Point<3>> directions;
    if (shared)
      for (unsigned int i = 0; i < 6; ++i)
        {
          auto               end  = shared->position;
          const unsigned int axis = (i + shared->id) % 6;
          end[axis / 2] += (axis % 2 ? 1. : -1.) * separation;
          directions.emplace_back(end - position);
        }
    else if (explicit_direction.norm() > 0.)
      directions.emplace_back(explicit_direction);
    else
      {
        if (incoming_direction)
          directions.emplace_back(*incoming_direction);
        directions.emplace_back(cell_center - position);
        for (const auto &vertex : cell_vertices)
          directions.emplace_back(.5 * (vertex + cell_center) - position);
        for (unsigned int d = 0; d < 3; ++d)
          for (const double sign : {-1., 1.})
            {
              Point<3> direction;
              direction[d] = sign;
              directions.emplace_back(direction);
            }
      }
    return directions;
  }

  namespace
  {
    struct SeedCell
    {
      dealii::CellId                id;
      double                        diameter = 0.;
      dealii::Point<3>              center;
      std::vector<dealii::Point<3>> vertices;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &id &diameter &center &vertices;
      }
    };
  } // namespace

  GrowthCommit
  start_seed(const dealii::Triangulation<3>  &bg,
             const dealii::Mapping<3>        &mapping,
             const Parameters                &p,
             const dealii::FunctionParser<3> &demand,
             MPI_Comm                         comm)
  {
    using namespace dealii;
    const auto rank        = Utilities::MPI::this_mpi_process(comm);
    const auto size        = Utilities::MPI::n_mpi_processes(comm);
    const auto require_all = [&](bool valid, const std::string &message) {
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid), comm) ==
                    1,
                  ExcMessage(message));
    };
    std::vector<double> controls{p.seed_radius,
                                 p.seed_length,
                                 p.viscosity,
                                 p.minimum_radius,
                                 p.minimum_length,
                                 p.minimum_angle,
                                 p.clearance,
                                 static_cast<double>(p.geometry_depth),
                                 static_cast<double>(p.geometry_work),
                                 p.arterial_murray_exponent,
                                 p.reference_flow,
                                 p.dissipation_weight};
    for (const auto &point :
         {p.seed_position, p.seed_direction, p.seed_inward_port_normal})
      for (unsigned int d = 0; d < 3; ++d)
        controls.push_back(point[d]);
    const auto local_controls = std::make_pair(controls, p.tissue_demand);
    bool       valid          = true;
    for (const auto &other : Utilities::MPI::all_gather(comm, local_controls))
      valid = other == local_controls && valid;
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(bg.get_mpi_communicator(), comm, &comparison);
    const auto *q  = dynamic_cast<const MappingQ<3> *>(&mapping);
    const auto *fe = dynamic_cast<const MappingFE<3> *>(&mapping);
    valid = valid && (comparison == MPI_IDENT || comparison == MPI_CONGRUENT) &&
            ((q && q->get_degree() == 1) || (fe && fe->get_degree() == 1));
    for (const auto &cell : bg.active_cell_iterators())
      if (cell->is_locally_owned())
        valid = (cell->reference_cell().is_simplex() ||
                 cell->reference_cell().is_hyper_cube()) &&
                mapping.is_compatible_with(cell->reference_cell()) && valid;
    require_all(
      valid,
      "Seed writers require matching controls, a first-order background mapping and congruent communicators.");
    GrowthCommit result;
    const auto   fail = [&](const std::string &message) {
      result.validation.issues.push_back({"seed", 0, message});
    };
    const auto normal = normalized_port_normal(p.seed_inward_port_normal);
    BackgroundGeometry geometry(bg, mapping);
    geometry.configure_length_scale(p.length_scale, comm);
    const auto  &scale = *geometry.length_scale;
    const double minimum_length =
      scale.physical_length(p.minimum_length, scale.minimum());
    std::vector<SeedCell> local_cells;
    if (const auto cell =
          geometry.containing(p.seed_position,
                              normal.norm_square() > 0. ? 1.e-12 : 0.))
      {
        SeedCell hint;
        hint.id       = (*cell)->id();
        hint.diameter = (*cell)->diameter();
        hint.center   = (*cell)->center();
        for (const auto v : (*cell)->vertex_indices())
          hint.vertices.push_back((*cell)->vertex(v));
        local_cells.push_back(std::move(hint));
      }
    const auto cells = Utilities::MPI::all_gather(comm, local_cells);
    std::optional<SeedCell> best;
    unsigned int            source = size;
    for (unsigned int r = 0; r < size; ++r)
      for (const auto &hint : cells[r])
        {
          source = std::min(source, r);
          if (!best || hint.diameter < best->diameter ||
              (hint.diameter == best->diameter && hint.id < best->id))
            best = hint;
        }
    if (!best)
      {
        fail("Seed position is outside the background.");
        return result;
      }
    const double seed_length =
      scale.physical_length(p.seed_length, best->diameter);
    if (normal.norm_square() > 0. &&
        Utilities::MPI::max(static_cast<unsigned int>(
                              geometry.planar_section(p.seed_position, normal)),
                            comm) == 0)
      {
        fail(
          "The declared planar port does not match a planar background boundary section.");
        return result;
      }
    // Only a real containing process submits the parameter-defined port.
    // Native ParticleHandler ownership then determines its physical writer.
    NetworkData port;
    if (rank == source)
      port.nodes[0] = {p.seed_position,
                       rank,
                       Port::arterial_inlet,
                       Provenance::generated,
                       normal};
    port                = redistribute(bg, mapping, port, 0., comm);
    const bool producer = port.nodes.count(0) && port.nodes.at(0).owner == rank;
    std::vector<Point<3>> directions;
    if (producer)
      directions = initial_root_directions(p.seed_position,
                                           best->center,
                                           best->vertices,
                                           p.seed_direction,
                                           {},
                                           nullptr,
                                           0.);
    const auto count =
      Utilities::MPI::max(static_cast<unsigned int>(directions.size()), comm);
    for (unsigned int i = 0; i < count; ++i)
      {
        std::vector<BatchProposal> proposals;
        Point<3>                   end;
        if (producer)
          {
            const double norm = directions[i].norm();
            if (norm > 0. && std::isfinite(norm))
              {
                end = p.seed_position + seed_length / norm * directions[i];
                if (end.distance(p.seed_position) >= minimum_length)
                  {
                    BatchProposal proposal;
                    proposal.id = 0;
                    proposal.changed.push_back(
                      {0, {{0, 1}}, {{p.seed_position, end}}, p.seed_radius});
                    if (normal.norm_square() > 0.)
                      {
                        proposal.normals[0]     = normal;
                        proposal.port_owners[0] = rank;
                      }
                    proposals.push_back(std::move(proposal));
                  }
              }
          }
        const auto selected =
          select_growth_batch(geometry, port, proposals, p, comm, [](Id) {
            return 0.;
          });
        const bool accepted = selected.local.count(0) && selected.local.at(0);
        if (Utilities::MPI::max(static_cast<unsigned int>(accepted), comm) == 0)
          continue;
        double value        = 0.;
        bool   valid_demand = true;
        if (producer)
          {
            try
              {
                value = demand.value(end);
              }
            catch (const std::exception &)
              {
                valid_demand = false;
              }
            valid_demand = valid_demand && std::isfinite(value) && value > 0.;
          }
        require_all(
          valid_demand,
          "Tissue demand must be positive and finite at every initial terminal.");
        auto                 trial = port;
        std::set<Id>         mutable_owned;
        std::map<Id, double> roots;
        if (producer)
          {
            trial.nodes[1] = {end, rank, Port::none, Provenance::generated};
            Vessel vessel;
            vessel.nodes       = {{0, 1}};
            vessel.radii       = {{p.seed_radius, p.seed_radius}};
            vessel.viscosity   = p.viscosity;
            vessel.provenance  = Provenance::generated;
            vessel.owner       = rank;
            trial.vessels[0]   = vessel;
            trial.terminals[0] = {1, 0, System::arterial, rank, value};
            mutable_owned.insert(0);
            roots[0] = p.seed_radius;
          }
        auto committed = commit_growth_splits(
          bg, mapping, trial, mutable_owned, roots, {}, p, comm, &geometry);
        if (committed.committed)
          return committed;
      }
    fail(
      "No admissible initial subtree direction was found within the deterministic geometry budget.");
    return result;
  }
  AttachmentStartup
  start_attachments(const dealii::Triangulation<3>  &bg,
                    const dealii::Mapping<3>        &mapping,
                    const NetworkData               &initial,
                    const Parameters                &p,
                    const dealii::FunctionParser<3> &demand,
                    unsigned int                     terminal_count,
                    MPI_Comm                         comm)
  {
    using namespace dealii;
    const auto        rank = Utilities::MPI::this_mpi_process(comm);
    AttachmentStartup result;
    const auto        fail = [&](Id id, const std::string &message) {
      result.accepted.validation.issues.push_back({"attachment", id, message});
    };
    const auto valid = [&]() {
      const bool all_valid =
        Utilities::MPI::min(static_cast<unsigned int>(
                              result.accepted.validation.valid()),
                            comm) == 1;
      if (!all_valid && result.accepted.validation.valid())
        fail(invalid_id,
             "An input attachment failed validation on another process.");
      if (!all_valid)
        result.accepted.committed = false;
      return all_valid;
    };
    // Check controls before any rank decides how many geometry rounds to run.
    std::vector<double> controls{p.seed_length,
                                 p.minimum_radius,
                                 p.minimum_length,
                                 p.minimum_angle,
                                 p.clearance,
                                 static_cast<double>(p.geometry_depth),
                                 static_cast<double>(p.geometry_work),
                                 p.arterial_murray_exponent,
                                 p.reference_flow,
                                 p.dissipation_weight,
                                 static_cast<double>(terminal_count)};
    for (unsigned int d = 0; d < 3; ++d)
      controls.push_back(p.seed_direction[d]);
    const auto local_controls = std::make_pair(controls, p.tissue_demand);
    bool       matching       = true;
    for (const auto &other : Utilities::MPI::all_gather(comm, local_controls))
      matching = matching && other == local_controls;
    AssertThrow(matching,
                ExcMessage(
                  "Imported startup requires matching growth controls."));

    const auto   scale = mesh_length_scale(bg, p.length_scale, comm);
    const double halo  = scale.physical_length(p.clearance, scale.maximum());
    const double minimum_radius =
      scale.physical_length(p.minimum_radius, scale.minimum());
    const double minimum_length =
      scale.physical_length(p.minimum_length, scale.minimum());
    scale.physical_length(p.seed_length, scale.minimum());
    scale.physical_length(p.seed_length, scale.maximum());
    // Native ownership and endpoint incidence are established before selecting
    // leaves. Input geometry and properties are copied, never normalized.
    auto snapshot = redistribute(bg, mapping, initial, halo, comm);
    std::map<Id, std::vector<Id>> incoming, outgoing;
    for (const auto &[id, v] : snapshot.vessels)
      {
        incoming[v.nodes[1]].push_back(id);
        outgoing[v.nodes[0]].push_back(id);
      }
    std::set<Id> recorded, original_owned_terminal_ids;
    for (const auto &[id, t] : snapshot.terminals)
      if (t.owner == rank)
        {
          original_owned_terminal_ids.insert(id);
          if (!recorded.insert(t.node).second || incoming[t.node].size() != 1 ||
              !outgoing[t.node].empty())
            fail(
              id,
              "A tissue terminal must uniquely identify an outward leaf of the input network.");
          else if (snapshot.vessels.at(incoming[t.node][0]).system != t.system)
            fail(id, "Terminal system disagrees with its incoming vessel.");
        }
    if (!valid())
      return result;

    // Reserve disjoint integer ranges once per phase, including empty ranks.
    const auto reserve = [&](Id maximum, bool present, Id count) {
      Id local_next = 0;
      if (present)
        {
          if (maximum >= invalid_id - 1)
            fail(maximum, "Vascular identifier space exhausted.");
          else
            local_next = maximum + 1;
        }
      const Id base   = Utilities::MPI::max(local_next, comm);
      const Id total  = Utilities::MPI::sum(count, comm);
      Id       prefix = 0;
      MPI_Exscan(&count, &prefix, 1, MPI_UINT64_T, MPI_SUM, comm);
      if (rank == 0)
        prefix = 0;
      if (total > invalid_id - base)
        fail(invalid_id, "Vascular identifier space exhausted.");
      return base + prefix;
    };
    const auto reserve_after = [&](const auto &data, Id count) {
      return reserve(data.empty() ? 0 : data.rbegin()->first,
                     !data.empty(),
                     count);
    };
    std::vector<Id> unrecorded;
    for (const auto &[id, node] : snapshot.nodes)
      if (node.owner == rank && node.port == Port::none &&
          incoming[id].size() == 1 && outgoing[id].empty() &&
          !recorded.count(id))
        unrecorded.push_back(id);
    Id next_terminal = reserve_after(snapshot.terminals, unrecorded.size());
    if (!valid())
      return result;
    for (Id node : unrecorded)
      snapshot.terminals[next_terminal++] = {
        node,
        invalid_id,
        snapshot.vessels.at(incoming.at(node)[0]).system,
        rank,
        1.};

    // Route all attachment point queries together using coarse background
    // boxes; neither terminal topology nor point coordinates are replicated.
    BackgroundGeometry geometry(bg, mapping);
    geometry.configure_length_scale(p.length_scale, comm);
    const auto &tree =
      geometry.cache.get_locally_owned_cell_bounding_boxes_rtree();
    std::vector<BoundingBox<3>> local_boxes;
    if (!tree.empty())
      local_boxes = extract_rtree_level(tree, 1);
    const auto boxes = Utilities::MPI::all_gather(comm, local_boxes);
    std::vector<std::pair<BoundingBox<3>, unsigned int>> covering;
    for (unsigned int r = 0; r < boxes.size(); ++r)
      for (const auto &box : boxes[r])
        covering.emplace_back(box, r);
    const auto   covering_tree = pack_rtree(covering);
    std::set<Id> explicit_ids(p.selected_terminals.begin(),
                              p.selected_terminals.end());
    if (explicit_ids.size() != p.selected_terminals.size())
      fail(invalid_id, "An input terminal may be selected only once.");
    Id                                                           found = 0;
    std::map<unsigned int, std::vector<std::pair<Id, Point<3>>>> queries;
    for (const auto &[id, t] : snapshot.terminals)
      if (t.owner == rank)
        {
          if (explicit_ids.count(id))
            {
              // Explicit ids refer to the supplied registry, not auto leaves.
              if (!original_owned_terminal_ids.count(id))
                fail(
                  id,
                  "Selected terminal id is absent from the input terminal registry.");
              ++found;
              if (t.system != System::arterial)
                fail(
                  id,
                  "Arterial generation requires an arterial attachment terminal.");
            }
          if (t.system == System::arterial &&
              (explicit_ids.empty() || explicit_ids.count(id)))
            {
              const auto &position = snapshot.nodes.at(t.node).position;
              std::set<unsigned int> targets;
              for (const auto &[box, target] :
                   covering_tree |
                     bgi::adaptors::queried(bgi::intersects(position)))
                targets.insert(target);
              for (const auto target : targets)
                queries[target].emplace_back(id, position);
            }
        }
    if (!explicit_ids.empty() &&
        Utilities::MPI::sum(found, comm) != explicit_ids.size())
      fail(invalid_id,
           "Selected terminal id is absent from the input terminal registry.");
    // The branch above depends only on collectively matching explicit ids.
    if (!valid())
      return result;
    const auto received = Utilities::MPI::some_to_some(comm, queries);
    std::map<unsigned int, std::map<Id, SeedCell>> replies;
    for (const auto &[source, points] : received)
      for (const auto &[id, point] : points)
        if (const auto cell = geometry.containing(point))
          {
            SeedCell hint;
            hint.id       = (*cell)->id();
            hint.diameter = (*cell)->diameter();
            hint.center   = (*cell)->center();
            for (const auto v : (*cell)->vertex_indices())
              hint.vertices.push_back((*cell)->vertex(v));
            replies[source].emplace(id, std::move(hint));
          }
    std::map<Id, SeedCell> hints;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(comm, replies))
      for (const auto &[id, hint] : entries)
        if (!hints.count(id) || hint.diameter < hints.at(id).diameter ||
            (hint.diameter == hints.at(id).diameter &&
             hint.id < hints.at(id).id))
          hints[id] = hint;
    for (const auto &[id, t] : snapshot.terminals)
      if (t.owner == rank && explicit_ids.count(id) && !hints.count(id))
        fail(
          id,
          "An explicitly selected exterior terminal cannot be extended by interior vessels.");
    const Id count = hints.size();
    const Id total = Utilities::MPI::sum(count, comm);
    if (total == 0)
      fail(invalid_id, "No available interior arterial terminal was found.");
    if (terminal_count < total)
      fail(
        invalid_id,
        "Terminal count is smaller than the number of selected attachments.");
    Id next_node           = reserve_after(snapshot.nodes, count);
    Id next_vessel         = reserve_after(snapshot.vessels, count);
    next_terminal          = reserve_after(snapshot.terminals, count);
    Id   maximum_territory = 0;
    bool have_territory    = false;
    for (const auto &[id, t] : snapshot.terminals)
      if (t.territory != invalid_id)
        {
          have_territory    = true;
          maximum_territory = std::max(maximum_territory, t.territory);
        }
    Id next_territory = reserve(maximum_territory, have_territory, count);
    if (!valid())
      return result;

    struct Pending
    {
      Id                    node, vessel, terminal, territory;
      std::vector<Point<3>> directions;
      unsigned int          next_direction = 0;
    };
    std::map<Id, Pending> pending;
    for (const auto &[id, hint] : hints)
      {
        const auto &terminal = snapshot.terminals.at(id);
        const auto &v = snapshot.vessels.at(incoming.at(terminal.node)[0]);
        const auto &position = snapshot.nodes.at(terminal.node).position;
        if (v.radii[1] < minimum_radius)
          fail(
            id,
            "Fixed attachment radius is below the minimum admissible new radius.");
        pending.emplace(
          id,
          Pending{next_node++,
                  next_vessel++,
                  next_terminal++,
                  next_territory++,
                  initial_root_directions(
                    position,
                    hint.center,
                    hint.vertices,
                    p.seed_direction,
                    Point<3>(position - snapshot.nodes.at(v.nodes[0]).position),
                    nullptr,
                    0.)});
      }
    if (!valid())
      return result;
    std::set<Id>         mutable_owned;
    std::map<Id, double> roots;
    while (Utilities::MPI::sum(static_cast<Id>(pending.size()), comm) > 0)
      {
        std::vector<BatchProposal> proposals;
        for (auto &[id, item] : pending)
          {
            if (item.next_direction >= item.directions.size())
              {
                fail(
                  id,
                  "No admissible initial subtree direction was found within the deterministic geometry budget.");
                continue;
              }
            const auto  &t        = snapshot.terminals.at(id);
            const auto  &v        = snapshot.vessels.at(incoming.at(t.node)[0]);
            const auto  &position = snapshot.nodes.at(t.node).position;
            const auto  &direction = item.directions[item.next_direction++];
            const double norm      = direction.norm();
            if (!(norm > 0.) || !std::isfinite(norm))
              continue;
            const Point<3> end =
              position +
              scale.physical_length(p.seed_length, hints.at(id).diameter) /
                norm * direction;
            if (position.distance(end) < minimum_length)
              continue;
            BatchProposal proposal;
            proposal.id = id;
            proposal.changed.push_back({item.vessel,
                                        {{t.node, item.node}},
                                        {{position, end}},
                                        v.radii[1]});
            proposals.push_back(std::move(proposal));
          }
        if (!valid())
          return result;
        const auto selection = select_growth_batch(
          geometry, snapshot, proposals, p, comm, [&](Id id) {
            const auto &v = snapshot.vessels.at(id);
            return std::max(v.radii[0], v.radii[1]);
          });
        std::vector<Id> accepted;
        auto            trial = snapshot;
        for (const auto &proposal : proposals)
          if (selection.local.at(proposal.id))
            {
              const Id    id    = proposal.id;
              const auto &item  = pending.at(id);
              const auto &t     = snapshot.terminals.at(id);
              auto        v     = snapshot.vessels.at(incoming.at(t.node)[0]);
              const auto &end   = proposal.changed[0].points[1];
              double      value = 0.;
              try
                {
                  value = demand.value(end);
                }
              catch (const std::exception &)
                {}
              if (!std::isfinite(value) || !(value > 0.))
                fail(
                  id,
                  "Tissue demand must be positive and finite at every initial terminal.");
              trial.nodes[item.node]     = {end,
                                            rank,
                                            Port::none,
                                            Provenance::generated};
              v.nodes                    = {{t.node, item.node}};
              v.radii                    = {{v.radii[1], v.radii[1]}};
              v.owner                    = rank;
              v.provenance               = Provenance::generated;
              trial.vessels[item.vessel] = v;
              trial.terminals.erase(id);
              trial.terminals[item.terminal] = {
                item.node, item.territory, System::arterial, rank, value};
              mutable_owned.insert(item.vessel);
              roots[item.vessel] = v.radii[0];
              accepted.push_back(id);
            }
        if (!valid())
          return result;
        if (Utilities::MPI::sum(static_cast<Id>(accepted.size()), comm) == 0)
          continue;
        auto committed = commit_growth_splits(
          bg, mapping, trial, mutable_owned, roots, {}, p, comm, &geometry);
        if (!committed.committed)
          {
            result.accepted = std::move(committed);
            return result;
          }
        for (Id id : accepted)
          {
            result.attachment_roots.emplace(id, pending.at(id).vessel);
            pending.erase(id);
          }
        snapshot        = committed.network;
        mutable_owned   = committed.mutable_owned;
        roots           = committed.fixed_owned_roots;
        result.accepted = std::move(committed);
      }
    return result;
  }
} // namespace ImmersX::Vascularization::internal
