// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_loop.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/scope_exit.h>
#include <deal.II/base/timer.h>

#include <deal.II/particles/particle_handler.h>

#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ranked_index.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/generation.h>

#include <deque>
#include <memory>
#include <random>

#include "growth_mpi_scope.h"
#include "growth_paths.h"
#include "growth_selection.h"
#include "growth_work.h"
namespace ImmersX::Vascularization::internal
{
  namespace
  {
    using namespace dealii;
    using Leaf = std::pair<Segment<3>, Id>;
    struct Hit
    {
      Id           id;
      unsigned int owner;
      double       distance;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &id &owner &distance;
      }
    };
    double
    lower_bound(const Point<3> &p, const BoundingBox<3> &box)
    {
      const auto &ends   = box.get_boundary_points();
      double      square = 0.;
      for (unsigned int d = 0; d < 3; ++d)
        {
          const double delta =
            std::max({0., ends.first[d] - p[d], p[d] - ends.second[d]});
          square += delta * delta;
        }
      return square;
    }
    std::vector<Hit>
    nearest(const std::optional<Point<3>> &target,
            const GrowthIndices           &indices,
            unsigned int                   k,
            MPI_Comm                       comm,
            System                         system,
            GrowthWork                    *work)
    {
      const unsigned int rank = Utilities::MPI::this_mpi_process(comm),
                         size = Utilities::MPI::n_mpi_processes(comm);
      const auto &tree        = indices.mutable_tree(system);
      const auto  local_boxes = GrowthIndices::covering(tree);
      const auto  boxes       = Utilities::MPI::all_gather(comm, local_boxes);
      std::set<unsigned int> queried;
      std::vector<Hit>       hits;
      const auto exchange = [&](const std::map<unsigned int, Point<3>> &send) {
        const auto incoming = Utilities::MPI::some_to_some(comm, send);
        std::map<unsigned int, std::vector<Hit>> replies;
        for (const auto &[source, point] : incoming)
          {
            const auto candidates = nearest_centerlines(tree, point, k, work);
            for (const auto &[segment, id] : candidates)
              replies[source].push_back(
                {id, rank, point_distance_square(point, segment)});
          }
        const auto received = Utilities::MPI::some_to_some(comm, replies);
        for (const auto &[owner, values] : received)
          hits.insert(hits.end(), values.begin(), values.end());
        std::sort(hits.begin(), hits.end(), [](const auto &a, const auto &b) {
          return a.distance < b.distance ||
                 (a.distance == b.distance && a.id < b.id);
        });
        if (hits.size() > k)
          hits.resize(k);
      };
      std::map<unsigned int, Point<3>> send;
      if (target)
        {
          unsigned int best     = size;
          double       distance = std::numeric_limits<double>::infinity();
          for (unsigned int owner = 0; owner < size; ++owner)
            if (!boxes[owner].empty())
              {
                const double candidate = lower_bound(*target, boxes[owner][0]);
                if (candidate < distance)
                  {
                    best     = owner;
                    distance = candidate;
                  }
              }
          if (best != size)
            {
              send[best] = *target;
              queried.insert(best);
            }
        }
      exchange(send);
      send.clear();
      if (target)
        {
          const double cutoff = hits.size() == k ?
                                  hits.back().distance :
                                  std::numeric_limits<double>::infinity();
          for (unsigned int owner = 0; owner < size; ++owner)
            if (!queried.count(owner) && !boxes[owner].empty())
              {
                const auto &[lo, hi] = boxes[owner][0].get_boundary_points();
                const double guard   = 128. *
                                     std::numeric_limits<double>::epsilon() *
                                     std::max({1.,
                                               target->norm_square(),
                                               lo.norm_square(),
                                               hi.norm_square()});
                if (lower_bound(*target, boxes[owner][0]) <= cutoff + guard)
                  send[owner] = *target;
              }
        }
      exchange(send);
      return hits;
    }
  } // namespace
  GrowthRun
  grow_forest(const dealii::Triangulation<3> &bg,
              const dealii::Mapping<3>       &mapping,
              const NetworkData              &initial,
              const std::set<Id>             &mutable_owned,
              const std::map<Id, double>     &fixed_owned_roots,
              const Parameters               &p,
              MPI_Comm                        comm,
              unsigned int                    requested_count,
              System                          system,
              const std::vector<Territory>   &owned_territories,
              GrowthDemandCallbacks           demand_callbacks,
              BackgroundGeometry             *shared_geometry)
  {
    using namespace dealii;
    Timer              timer;
    GrowthRun          run;
    auto              *work = &run.work;
    const unsigned int terminal_count =
      requested_count == 0 ? p.terminal_count : requested_count;
    GrowthMpiRun mpi_run(static_cast<unsigned int>(system), terminal_count);
    auto initialization_scope = mpi_run.phase(GrowthMpiPhase::initialization);
    const unsigned int rank   = Utilities::MPI::this_mpi_process(comm),
                       size   = Utilities::MPI::n_mpi_processes(comm);
    const auto phase          = [&](bool valid, const std::string &message) {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1;
      if (!all)
        run.validation.issues.push_back({"growth", invalid_id, message});
      return all;
    };
    const auto &companion = system == System::portal ? p.portal : p.venous;
    bool        valid     = system == System::arterial ||
                 (system == System::venous || system == System::portal);
    double gamma = 0.;
    if (system == System::arterial)
      gamma = p.arterial_murray_exponent;
    else if (system == System::venous || system == System::portal)
      gamma = companion.murray_exponent;
    if (system == System::venous || system == System::portal)
      valid = valid && std::isfinite(companion.terminal_separation) &&
              companion.terminal_separation > 0. && std::isfinite(gamma) &&
              gamma > 0.;
    try
      {
        p.validate(false);
      }
    catch (const std::exception &)
      {
        valid = false;
      }
    for (Id id : mutable_owned)
      valid = valid && initial.vessels.count(id) &&
              initial.vessels.at(id).system == system;
    if (
      !phase(
        valid,
        "Batched growth requires valid controls and real attachment trees of the active system."))
      return run;
    const bool has_demand_callbacks =
      demand_callbacks.bootstrap || demand_callbacks.preview ||
      demand_callbacks.joint || demand_callbacks.prepare_adopt ||
      demand_callbacks.adopt || demand_callbacks.discard;
    if (!phase(!has_demand_callbacks ||
                 (demand_callbacks.bootstrap && demand_callbacks.preview &&
                  demand_callbacks.joint && demand_callbacks.prepare_adopt &&
                  demand_callbacks.adopt && demand_callbacks.discard),
               "Integrated growth requires a complete demand transaction."))
      return run;
    const auto controls = std::make_pair(
      std::array<double, 10>{{static_cast<double>(terminal_count),
                              static_cast<double>(p.random_seed),
                              static_cast<double>(p.sample_budget),
                              static_cast<double>(p.candidate_count),
                              static_cast<double>(p.optimization_steps),
                              p.density_beta,
                              p.terminal_weight_bound,
                              static_cast<double>(system),
                              companion.terminal_separation,
                              has_demand_callbacks ? 1. : 0.}},
      std::array<std::string, 3>{
        {p.terminal_weight, p.tissue_demand, p.terminal_sampling}});
    for (const auto &other : Utilities::MPI::all_gather(comm, controls))
      valid = valid && other == controls;
    if (!phase(valid,
               "All producers must use the same growth and sampling controls."))
      return run;
    std::unique_ptr<BackgroundGeometry> local_geometry;
    if (!shared_geometry)
      local_geometry = std::make_unique<BackgroundGeometry>(bg, mapping, work);
    auto &geometry = shared_geometry ? *shared_geometry : *local_geometry;
    if (!phase(&geometry.cache.get_triangulation() == &bg &&
                 &geometry.mapping == &mapping,
               "Growth geometry must describe its background and mapping."))
      return run;
    auto *original_counts = geometry.counts;
    geometry.counts       = work;
    const ScopeExit restore_counts(
      [&]() { geometry.counts = original_counts; });
    geometry.configure_length_scale(p.length_scale, comm);
    const auto  &scale = *geometry.length_scale;
    const double clearance_lower =
      scale.physical_length(p.clearance, scale.minimum());
    const double broad_clearance =
      scale.physical_length(p.clearance, scale.maximum());
    const double physical_minimum_radius =
      scale.physical_length(p.minimum_radius, scale.minimum());
    // Startup has established native topology and fixed physical radii only.
    // Assign all initial Di before the first Murray/flow/objective evaluation,
    // including a request satisfied by its initial leaves without insertions.
    {
      count_copy(work, initial);
      auto                    weighted_initial = initial;
      GrowthDemandJointUpdate initial_demands;
      if (demand_callbacks.bootstrap)
        {
          const auto leaves =
            collect_growth_leaf_terminals(initial, mutable_owned, comm);
          if (!leaves.validation.valid())
            {
              run.validation = leaves.validation;
              return run;
            }
          initial_demands =
            demand_callbacks.bootstrap(initial, leaves.owned, 0);
          run.validation = initialize_growth_demands(
            weighted_initial, leaves.owned, initial_demands, 0, comm);
          if (!run.validation.valid())
            {
              demand_callbacks.discard();
              return run;
            }
          demand_callbacks.prepare_adopt(
            initial_demands.update.quadrature_revision, 1);
        }
      run.accepted = commit_growth_splits(bg,
                                          mapping,
                                          weighted_initial,
                                          mutable_owned,
                                          fixed_owned_roots,
                                          {},
                                          p,
                                          comm,
                                          &geometry);
    }
    if (!run.accepted.committed)
      {
        if (demand_callbacks.discard)
          demand_callbacks.discard();
        run.validation = run.accepted.validation;
        return run;
      }
    if (demand_callbacks.adopt)
      demand_callbacks.adopt();
    const auto count_leaves = [&]() {
      std::uint64_t local = 0;
      for (const auto &[id, b] : run.accepted.physiology.owned)
        if (b.terminal != invalid_id)
          ++local;
      return Utilities::MPI::sum(local, comm);
    };
    std::uint64_t count = count_leaves();
    if (
      !phase(
        count > 0 && count <= terminal_count,
        "Growth requires real roots and a target not smaller than the existing mutable forest."))
      return run;
    // Keep the sorted-id ordinal stable when served territories are erased.
    boost::multi_index_container<
      Territory,
      boost::multi_index::indexed_by<boost::multi_index::ranked_unique<
        boost::multi_index::member<Territory, Id, &Territory::id>>>>
                         pending;
    std::map<Id, double> territory_separations;
    if (system == System::venous || system == System::portal)
      {
        std::map<unsigned int, std::map<Id, double>> registration;
        for (const auto &territory : owned_territories)
          {
            valid = valid && territory.id != invalid_id &&
                    territory.owner == rank &&
                    std::isfinite(territory.demand) && territory.demand > 0. &&
                    pending.insert(territory).second;
            for (unsigned int d = 0; d < 3; ++d)
              valid = valid && std::isfinite(territory.position[d]);
            registration[territory.id % size][territory.id] = territory.demand;
          }
        const auto n_territories = Utilities::MPI::sum(
          static_cast<std::uint64_t>(owned_territories.size()), comm);
        if (
          !phase(
            valid && n_territories == terminal_count,
            "Venous growth requires exactly the requested number of owned common territories with finite positive demand."))
          return run;
        const auto registered =
          Utilities::MPI::some_to_some(comm, registration);
        std::map<Id, std::pair<unsigned int, double>> directory;
        for (const auto &[owner, values] : registered)
          for (const auto &[id, d] : values)
            valid =
              directory.emplace(id, std::make_pair(owner, d)).second && valid;
        if (!phase(
              valid,
              "A common territory has more than one authoritative producer."))
          return run;
        using Claim = std::pair<Id, std::pair<Id, double>>;
        std::map<unsigned int, std::vector<Claim>> claims;
        for (Id tid : run.accepted.physiology.owned_leaf_terminals)
          {
            const auto &t = run.accepted.network.terminals.at(tid);
            claims[t.territory % size].push_back(
              {t.territory, {tid, t.demand}});
          }
        const auto   claimed = Utilities::MPI::some_to_some(comm, claims);
        std::set<Id> seen;
        std::map<unsigned int, std::map<Id, Id>> served;
        for (const auto &[owner, values] : claimed)
          for (const auto &[territory, terminal] : values)
            {
              const auto entry = directory.find(territory);
              if (entry == directory.end() || !seen.insert(territory).second ||
                  entry->second.second != terminal.second)
                valid = false;
              else
                served[entry->second.first][territory] = terminal.first;
            }
        if (
          !phase(
            valid,
            "Existing venous leaves must claim distinct common territories with exactly their shared demand."))
          return run;
        const auto received = Utilities::MPI::some_to_some(comm, served);
        for (const auto &[shard, values] : received)
          for (const auto &[territory, tid] : values)
            {
              valid = pending.erase(territory) == 1 && valid;
              run.owned_venous_terminals.emplace(territory, tid);
            }
        if (
          !phase(
            valid,
            "An existing venous claim has no authoritative common-territory owner."))
          return run;
      }
    if (system == System::venous || system == System::portal)
      {
        scale.physical_length(companion.terminal_separation, scale.minimum());
        scale.physical_length(companion.terminal_separation, scale.maximum());
        std::map<Id, Point<3>> territory_queries;
        for (const auto &territory : owned_territories)
          territory_queries.emplace(territory.id, territory.position);
        const auto tissue_cells =
          geometry.locate_points(territory_queries, comm);
        bool found_all = true;
        for (const auto &territory : owned_territories)
          {
            const auto found = tissue_cells.find(territory.id);
            found_all        = found_all && found != tissue_cells.end();
            if (found != tissue_cells.end())
              territory_separations[territory.id] =
                scale.physical_length(companion.terminal_separation,
                                      found->second.diameter);
          }
        if (
          !phase(
            found_all,
            "Common territory centres must be inside the prepared background."))
          return run;
      }
    double local_capacity = 0.;
    for (const auto &[id, radius] : run.accepted.fixed_owned_roots)
      local_capacity +=
        physical_minimum_radius == 0. ?
          terminal_count :
          std::min<double>(
            terminal_count,
            std::floor(std::pow(radius / physical_minimum_radius, gamma) *
                       (1. + 32. * std::numeric_limits<double>::epsilon())));
    if (
      !phase(
        Utilities::MPI::sum(local_capacity, comm) >= terminal_count,
        "Requested terminal count is incompatible with Murray's minimum-radius capacity at the real fixed roots."))
      return run;
    FunctionParser<3> weight, demand;
    try
      {
        weight.initialize("x,y,z", p.terminal_weight, {});
        demand.initialize("x,y,z", p.tissue_demand, {});
      }
    catch (const std::exception &)
      {
        valid = false;
      }
    if (!phase(valid, "Invalid spatial-weight or tissue-demand expression."))
      return run;
    const BackgroundCellSampler *cell_sampler = nullptr;
    const bool                   prescribed   = system == System::arterial &&
                            p.terminal_sampling == "prescribed per cell";
    if (system == System::arterial && !prescribed)
      try
        {
          cell_sampler = &geometry.sample_cells(p.density_beta, comm);
        }
      catch (const std::exception &error)
        {
          valid = false;
          run.validation.issues.push_back(
            {"cell sampling", invalid_id, error.what()});
        }
    if (!phase(valid, "Cannot prepare physical background-cell sampling."))
      return run;
    std::seed_seq   seed{p.random_seed, rank, 194837u};
    std::mt19937_64 random(seed);
    std::seed_seq   physical_seed{p.random_seed, rank, 194838u};
    std::mt19937    physical_random(physical_seed);
    std::unique_ptr<Particles::ParticleHandler<3>> prescribed_points;
    std::deque<Particles::ParticleHandler<3>::particle_iterator>
      prescribed_pending;
    if (prescribed)
      {
        // Existing seed leaves occupy slots, so the final forest, rather than
        // just its extra insertions, has exactly k terminal points in each
        // cell.
        std::map<Id, Point<3>> leaves;
        for (Id id : run.accepted.physiology.owned_leaf_terminals)
          leaves.emplace(id,
                         run.accepted.network.nodes
                           .at(run.accepted.network.terminals.at(id).node)
                           .position);
        const auto located = geometry.locate_points(leaves, comm);
        std::map<unsigned int, std::map<CellId, unsigned int>> claims;
        valid = located.size() == leaves.size();
        for (const auto &[id, cell] : located)
          ++claims[cell.cell_owner][cell.id];
        const auto received = Utilities::MPI::some_to_some(comm, claims);
        std::map<CellId, unsigned int> occupied;
        for (const auto &[owner, cells] : received)
          for (const auto &[cell, n] : cells)
            occupied[cell] += n;
        prescribed_points =
          std::make_unique<Particles::ParticleHandler<3>>(bg, mapping);
        types::particle_index next_point = rank;
        try
          {
            for (const auto &cell : bg.active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  const auto n = occupied[cell->id()];
                  valid        = valid && n <= p.terminal_count;
                  std::vector<Point<3>> locations;
                  if (cell->reference_cell().is_hyper_cube())
                    {
                      unsigned int side = 1;
                      while (static_cast<std::uint64_t>(side) * side * side <
                             p.terminal_count)
                        ++side;
                      locations =
                        QIterated<3>(QMidpoint<1>(), side).get_points();
                    }
                  else if (cell->reference_cell().is_simplex())
                    {
                      unsigned int order = 1;
                      while (QGaussSimplex<3>(order).size() < p.terminal_count)
                        ++order;
                      locations = QGaussSimplex<3>(order).get_points();
                    }
                  else
                    AssertThrow(
                      false,
                      ExcMessage(
                        "Prescribed sampling supports hexahedra and tetrahedra."));
                  std::shuffle(locations.begin(), locations.end(), random);
                  for (unsigned int j = n; j < p.terminal_count; ++j)
                    {
                      const auto &reference = locations[j];
                      const auto  point =
                        mapping.transform_unit_to_real_cell(cell, reference);
                      prescribed_points->insert_particle(point,
                                                         reference,
                                                         next_point,
                                                         cell);
                      next_point += size;
                    }
                }
          }
        catch (const std::exception &error)
          {
            valid = false;
            run.validation.issues.push_back(
              {"prescribed sampling", invalid_id, error.what()});
          }
        std::vector<Particles::ParticleHandler<3>::particle_iterator> order;
        for (auto particle = prescribed_points->begin();
             particle != prescribed_points->end();
             ++particle)
          order.push_back(particle);
        std::shuffle(order.begin(), order.end(), random);
        prescribed_pending.assign(order.begin(), order.end());
        const auto remaining = Utilities::MPI::sum(
          static_cast<std::uint64_t>(prescribed_pending.size()), comm);
        valid = valid && remaining + count == terminal_count;
        if (
          !phase(
            valid,
            "Prescribed targets require at most k existing mutable leaves per cell and exactly k terminal slots per global cell."))
          return run;
      }
    const auto uniform = [&]() {
      return static_cast<double>(random() >> 11) * 0x1.0p-53;
    };
    std::array<Id, 4> next{{0, 0, 0, 0}};
    for (const auto &[id, n] : run.accepted.network.nodes)
      if (n.owner == rank)
        next[0] = std::max(next[0], id);
    for (const auto &[id, v] : run.accepted.network.vessels)
      if (v.owner == rank)
        next[1] = std::max(next[1], id);
    for (const auto &[id, t] : run.accepted.network.terminals)
      if (t.owner == rank)
        {
          next[2] = std::max(next[2], id);
          if (t.territory != invalid_id)
            next[3] = std::max(next[3], t.territory);
        }
    for (auto &id : next)
      {
        id = Utilities::MPI::max(id, comm);
        if (id != invalid_id)
          ++id;
      }
    GrowthGeometrySegments initial_segments;
    for (const auto &[id, v] : run.accepted.network.vessels)
      if (v.owner == rank)
        initial_segments.emplace(
          id,
          Segment<3>{run.accepted.network.nodes.at(v.nodes[0]).position,
                     run.accepted.network.nodes.at(v.nodes[1]).position});
    const auto initial_measures =
      geometry.measure_segments(initial_segments, comm);
    GrowthExecution execution(
      run.accepted.network,
      run.accepted.physiology,
      p,
      1,
      geometry,
      initial_measures,
      [&](const GrowthGeometrySegments &segments) {
        return geometry.measure_segments(segments, comm);
      },
      [&](const GrowthGeometrySegments &segments) {
        geometry.release_profiles(segments);
      },
      broad_clearance,
      comm,
      work);
    geometry.release_profiles(initial_segments);
    if (!execution.validation().valid())
      {
        run.validation = execution.validation();
        return run;
      }
    // The execution controller owns the accepted rows and summaries now.
    run.accepted                 = {};
    const auto update_quadrature = [&](GrowthStatistics &statistics) {
      if (demand_callbacks.quadrature_diagnostics)
        {
          const auto [points, peak] = demand_callbacks.quadrature_diagnostics();
          statistics.demand_quadrature_points = points;
          statistics.demand_quadrature_peak_points =
            std::max(statistics.demand_quadrature_peak_points, peak);
        }
    };
    const auto finish = [&]() {
      auto mpi_scope = mpi_run.phase(GrowthMpiPhase::finalization, count);
      const double final_begin = timer.wall_time();
      const auto   final_work  = *work;
      auto         output      = execution.materialize();
      // The independent full transaction runs once at output. Ordinary growth
      // keeps native rows and lazy physiology in the incremental controller.
      auto final = commit_growth_splits(bg,
                                        mapping,
                                        output.network,
                                        output.mutable_owned,
                                        output.fixed_owned_roots,
                                        {},
                                        p,
                                        comm,
                                        &geometry);
      if (!final.committed)
        run.validation.issues.insert(run.validation.issues.end(),
                                     final.validation.issues.begin(),
                                     final.validation.issues.end());
      if (final.committed)
        {
          bool same_radii = true;
          // Positive subtree sums accumulate roundoff along their ancestors.
          // This bound also covers the final power and ratio evaluations.
          const long double tolerance =
            64.L * std::numeric_limits<double>::epsilon() *
            (1.L + (1.L + final.physiology.upward_levels) / gamma);
          count_work(work,
                     GrowthOperation::validation_record_visits,
                     output.network.vessels.size());
          for (const auto &[id, vessel] : output.network.vessels)
            if (vessel.owner == rank)
              {
                const auto after = final.network.vessels.find(id);
                if (after == final.network.vessels.end() ||
                    after->second.owner != rank)
                  {
                    same_radii = false;
                    continue;
                  }
                for (unsigned int end = 0; end < 2; ++end)
                  {
                    const long double before_radius = vessel.radii[end];
                    const long double after_radius  = after->second.radii[end];
                    if (output.mutable_owned.count(id))
                      same_radii = same_radii &&
                                   std::abs(before_radius - after_radius) /
                                       std::max(before_radius, after_radius) <=
                                     tolerance;
                    else
                      same_radii = same_radii && vessel.radii[end] ==
                                                   after->second.radii[end];
                  }
              }
          phase(
            same_radii,
            "The independent final reconstruction disagrees with accepted physical radii or native ownership.");
        }
      run.accepted = std::move(final);
      if (!run.statistics.empty())
        {
          auto &statistics = run.statistics.rbegin()->second;
          statistics.finalization_seconds += timer.wall_time() - final_begin;
          statistics.finalization_work += *work - final_work;
          statistics.work += *work - final_work;
          update_quadrature(statistics);
        }
      return std::move(run);
    };
    const auto bucket = [&](std::uint64_t leaves) -> GrowthStatistics & {
      unsigned int exponent = 0;
      for (auto n = leaves; n > 1; n >>= 1)
        ++exponent;
      auto [entry, inserted] = run.statistics.try_emplace(exponent);
      auto &statistics       = entry->second;
      if (inserted)
        {
          statistics.system              = system;
          statistics.requested_terminals = terminal_count;
          statistics.first_terminals = statistics.last_terminals = leaves;
          statistics.owned_vessels_first = statistics.owned_vessels_last =
            statistics.owned_vessels_maximum =
              execution.accepted_snapshot().owned_mutable_count();
          update_quadrature(statistics);
        }
      return statistics;
    };
    bucket(count).initialization_seconds = timer.wall_time();
    bucket(count).work += *work;
    bucket(count).initialization_work += *work;
    unsigned int failures                     = 0;
    bool         integration_budget_exhausted = false;
    initialization_scope.finish();
    while (count < terminal_count && failures < p.sample_budget)
      {
        auto       mpi_scope   = mpi_run.phase(GrowthMpiPhase::batch, count);
        auto      &statistics  = bucket(count);
        const auto work_before = *work;
        const auto accepted    = execution.accepted_snapshot();
        const auto owned_before =
          execution.accepted_snapshot().owned_mutable_count();
        const auto   samples_before   = run.sampled_points;
        const auto   proposals_before = run.proposed_insertions;
        const auto   accepted_before  = run.accepted_insertions;
        const double batch_begin      = timer.wall_time();
        double       sampling_end = batch_begin, optimization_end = batch_begin,
               selection_end      = batch_begin;
        double     search_seconds = 0., optimization_seconds = 0.;
        const auto record = [&](const GrowthExecutionResult *commit) {
          const double end = timer.wall_time();
          ++statistics.batches;
          statistics.sampled_points += run.sampled_points - samples_before;
          statistics.proposed_insertions +=
            run.proposed_insertions - proposals_before;
          statistics.accepted_insertions +=
            run.accepted_insertions - accepted_before;
          statistics.active_writer_batches += owned_before > 0;
          statistics.last_terminals = count;
          statistics.owned_vessels_last =
            execution.accepted_snapshot().owned_mutable_count();
          statistics.owned_vessels_maximum = std::max<std::uint64_t>(
            statistics.owned_vessels_maximum,
            std::max<std::uint64_t>(owned_before,
                                    statistics.owned_vessels_last));
          statistics.sampling_seconds += sampling_end - batch_begin;
          statistics.search_seconds += search_seconds;
          statistics.optimization_seconds += optimization_seconds;
          statistics.work += *work - work_before;
          update_quadrature(statistics);
          statistics.selection_seconds += selection_end - optimization_end;
          statistics.total_seconds += end - batch_begin;
          if (commit)
            {
              statistics.commit_seconds += end - selection_end;
              statistics.redistribution_seconds +=
                commit->redistribution_seconds;
              statistics.physiology_seconds += commit->physiology_seconds;
              statistics.certification_seconds += commit->certification_seconds;
              statistics.committed_batches += commit->committed;
              statistics.rejected_batches += !commit->committed;
            }
        };
        ++run.batches;
        std::array<Id, 4> allocated;
        for (unsigned int kind = 0; kind < 4; ++kind)
          {
            const Id   stride    = kind < 2 ? 2 : 1;
            const bool available = next[kind] < invalid_id - stride * size;
            valid                = valid && available;
            allocated[kind]      = invalid_id;
            if (available)
              {
                allocated[kind] = next[kind] + stride * rank;
                next[kind] += stride * size;
              }
          }
        if (!phase(valid,
                   "The physical vascular identifier space is exhausted."))
          return finish();
        std::optional<Point<3>> raw;
        Id                      shared_territory = invalid_id;
        if (system == System::arterial)
          {
            try
              {
                if (prescribed)
                  {
                    if (!prescribed_pending.empty())
                      {
                        raw = prescribed_pending.front()->get_location();
                        // Failed targets remain in the cloud. Try the others
                        // before retrying this point against a larger tree.
                        prescribed_pending.push_back(
                          prescribed_pending.front());
                        prescribed_pending.pop_front();
                      }
                  }
                else
                  raw = cell_sampler->sample(random, physical_random);
                run.sampled_points += raw.has_value();
              }
            catch (const std::exception &error)
              {
                valid = false;
                run.validation.issues.push_back(
                  {"cell sampling", invalid_id, error.what()});
              }
          }
        else if ((system == System::venous || system == System::portal) &&
                 !pending.empty())
          {
            const auto entry =
              pending.nth(((run.batches - 1) / 6) % pending.size());
            shared_territory = entry->id;
            raw              = entry->position;
            const unsigned int direction =
              ((run.batches - 1) % 6 + shared_territory % 6) % 6;
            (*raw)[direction / 2] += (direction % 2 ? 1. : -1.) *
                                     territory_separations.at(shared_territory);
            ++run.sampled_points;
          }
        std::map<Id, Point<3>> point_queries;
        if (raw)
          point_queries.emplace(0, *raw);
        const auto located       = geometry.locate_points(point_queries, comm);
        const auto selected_cell = located.find(0);
        const auto point_owner   = selected_cell == located.end() ?
                                     size :
                                     selected_cell->second.point_owner;
        std::optional<Point<3>> target;
        double                  terminal_demand = 0.;
        if ((system == System::venous || system == System::portal) && raw &&
            point_owner < size)
          {
            target          = raw;
            terminal_demand = pending.find(shared_territory)->demand;
          }
        else if (system == System::arterial && raw && point_owner == rank)
          {
            const double w = weight.value(*raw);
            valid          = valid && std::isfinite(w) && w >= 0. &&
                    w <= p.terminal_weight_bound;
            if (valid && uniform() < w / p.terminal_weight_bound)
              {
                target = raw;
                if (!has_demand_callbacks)
                  {
                    terminal_demand = demand.value(*target);
                    valid =
                      std::isfinite(terminal_demand) && terminal_demand > 0.;
                  }
              }
          }
        if (
          !phase(
            valid,
            "Native cell sampling failed, or spatial weight/tissue demand violates its configured range."))
          return finish();
        sampling_end               = timer.wall_time();
        const double nearest_begin = timer.wall_time();
        const auto   nearby        = nearest(
          target, accepted.indices(), p.candidate_count, comm, system, work);
        search_seconds += timer.wall_time() - nearest_begin;
        const auto active_producers =
          Utilities::MPI::all_gather(comm, target.has_value());
        GrowthCandidatePaths                       paths;
        std::optional<GrowthCandidate>             best;
        const GrowthCandidateContext              *chosen = nullptr;
        const std::map<Id, GrowthCandidateContext> no_contexts;
        for (unsigned int producer = 0; producer < size; ++producer)
          if (active_producers[producer])
            {
              const double            search_begin = timer.wall_time();
              const GrowthProposalKey key{producer,
                                          allocated[2] - rank + producer};
              GrowthDemandUpdate      update;
              update.snapshot_revision = accepted.revision();
              double new_demand = rank == producer ? terminal_demand : 0.;
              if (demand_callbacks.preview)
                {
                  auto proposal =
                    demand_callbacks.preview(key,
                                             key.second,
                                             allocated[3],
                                             rank == producer ? *target :
                                                                Point<3>(),
                                             accepted.revision());
                  const bool usable =
                    Utilities::MPI::min(
                      static_cast<unsigned int>(
                        proposal.validation.valid() &&
                        !proposal.integration_budget_exhausted),
                      comm) == 1;
                  if (!usable)
                    {
                      const bool budget =
                        Utilities::MPI::max(
                          static_cast<unsigned int>(
                            proposal.integration_budget_exhausted),
                          comm) == 1;
                      search_seconds += timer.wall_time() - search_begin;
                      if (!budget)
                        {
                          run.validation = proposal.validation;
                          phase(
                            false,
                            "Integrated candidate demand evaluation failed.");
                          demand_callbacks.discard();
                          return finish();
                        }
                      integration_budget_exhausted = true;
                      if (rank == producer)
                        count_work(
                          work, GrowthOperation::integration_budget_rejections);
                      continue;
                    }
                  update     = std::move(proposal.update);
                  new_demand = proposal.new_demand;
                  if (rank == producer)
                    terminal_demand = new_demand;
                }
              auto preview = accepted.demand().preview(update);
              if (!preview.validation.valid())
                {
                  run.validation = preview.validation;
                  if (demand_callbacks.discard)
                    demand_callbacks.discard();
                  return finish();
                }
              std::map<Id, unsigned int> requests;
              if (rank == producer)
                for (const auto &hit : nearby)
                  requests[hit.id] = hit.owner;
              auto fetched =
                accepted.demand().candidate_paths(accepted.network(),
                                                  preview,
                                                  requests,
                                                  allocated[0],
                                                  allocated[1],
                                                  new_demand,
                                                  scale.minimum(),
                                                  work);
              if (!fetched.validation.valid())
                {
                  run.validation = fetched.validation;
                  if (demand_callbacks.discard)
                    demand_callbacks.discard();
                  return finish();
                }
              if (rank == producer)
                paths = std::move(fetched);
              const auto obstacle_reply = execution.candidate_obstacles(
                accepted,
                preview,
                rank == producer ? paths.contexts : no_contexts,
                rank == producer ? target : std::optional<Point<3>>{},
                new_demand,
                producer);
              if (!obstacle_reply.validation.valid())
                {
                  run.validation = obstacle_reply.validation;
                  if (demand_callbacks.discard)
                    demand_callbacks.discard();
                  return finish();
                }
              const auto       &obstacles = obstacle_reply.records;
              std::vector<Leaf> obstacle_leaves;
              double            obstacle_radius = 0.;
              for (const auto &[id, obstacle] : obstacles)
                {
                  obstacle_leaves.push_back(
                    {{obstacle.edge.points[0], obstacle.edge.points[1]}, id});
                  obstacle_radius =
                    std::max(obstacle_radius, obstacle.edge.radius);
                }
              const auto obstacle_tree = pack_rtree(obstacle_leaves);
              count_work(work, GrowthOperation::network_index_builds);
              count_work(work,
                         GrowthOperation::network_index_insertions,
                         obstacle_leaves.size());
              search_seconds += timer.wall_time() - search_begin;
              const double optimization_begin = timer.wall_time();
              if (rank == producer && target)
                for (const auto &hit : nearby)
                  {
                    const auto &x = paths.contexts.at(hit.id);
                    const auto  radii =
                      candidate_radii(x, terminal_demand, work);
                    if (!radii)
                      continue;
                    count_work(work,
                               GrowthOperation::candidate_path_visits,
                               x.path.size());
                    std::map<Id, Point<3>> port_normals;
                    for (const auto &branch : x.path)
                      for (unsigned int end = 0; end < 2; ++end)
                        port_normals[branch.edge.nodes[end]] =
                          branch.port_normals[end];
                    const auto &covered =
                      obstacle_reply.covered_ancestors.at(hit.id);
                    const auto cheap_geometry =
                      [&](const std::vector<Edge> &changed) {
                        std::set<Id> replaced;
                        using TrialLeaf =
                          std::pair<BoundingBox<3>, std::size_t>;
                        std::vector<TrialLeaf> trial_leaves;
                        const auto             envelope = [&](const Edge &edge,
                                                  double      padding) {
                          Point<3> a, b;
                          for (unsigned int d = 0; d < 3; ++d)
                            {
                              a[d] =
                                std::min(edge.points[0][d], edge.points[1][d]) -
                                padding;
                              b[d] =
                                std::max(edge.points[0][d], edge.points[1][d]) +
                                padding;
                            }
                          return BoundingBox<3>(std::make_pair(a, b));
                        };
                        for (std::size_t i = 0; i < changed.size(); ++i)
                          {
                            replaced.insert(changed[i].id);
                            trial_leaves.emplace_back(
                              envelope(changed[i],
                                       changed[i].radius + clearance_lower),
                              i);
                          }
                        const auto trial_tree = pack_rtree(trial_leaves);
                        count_work(work, GrowthOperation::network_index_builds);
                        count_work(work,
                                   GrowthOperation::network_index_insertions,
                                   trial_leaves.size());
                        for (std::size_t i = 0; i < changed.size(); ++i)
                          {
                            const auto &e = changed[i];
                            // The controller proves these unchanged ancestor
                            // envelopes against the current demand overlay. New
                            // geometry still checks its pairs with them through
                            // the trial tree.
                            if (covered.count(e.id))
                              continue;
                            std::array<Point<3>, 2> normals;
                            for (unsigned int end = 0; end < 2; ++end)
                              if (const auto normal =
                                    port_normals.find(e.nodes[end]);
                                  normal != port_normals.end())
                                normals[end] = normal->second;
                            if (!geometry.boundary_separated(
                                  {e.points[0], e.points[1]},
                                  e.radius + clearance_lower,
                                  p.geometry_depth,
                                  p.geometry_work,
                                  normals))
                              return false;
                            const double guard =
                              64. * std::numeric_limits<double>::epsilon() *
                              std::max(
                                {1., e.points[0].norm(), e.points[1].norm()});
                            count_work(work,
                                       GrowthOperation::network_index_queries);
                            for (const auto &[segment, id] :
                                 obstacle_tree |
                                   bgi::adaptors::queried(bgi::intersects(
                                     envelope(e,
                                              e.radius + obstacle_radius +
                                                broad_clearance + guard))))
                              {
                                count_work(work,
                                           GrowthOperation::network_index_hits);
                                if (!replaced.count(id))
                                  {
                                    auto obstacle = obstacles.at(id).edge;
                                    obstacle.radius =
                                      (*radii)(id,
                                               obstacles.at(id).root,
                                               obstacle.radius);
                                    if (!compatible(e,
                                                    obstacle,
                                                    p.minimum_angle,
                                                    clearance_lower,
                                                    work))
                                      return false;
                                  }
                              }
                            count_work(work,
                                       GrowthOperation::network_index_queries);
                            for (const auto &[box, j] :
                                 trial_tree |
                                   bgi::adaptors::queried(
                                     bgi::intersects(trial_leaves[i].first)))
                              {
                                count_work(work,
                                           GrowthOperation::network_index_hits);
                                if (j != i &&
                                    (j > i || covered.count(changed[j].id)) &&
                                    !compatible(e,
                                                changed[j],
                                                p.minimum_angle,
                                                clearance_lower,
                                                work))
                                  return false;
                              }
                          }
                        return true;
                      };
                    if (auto proposal = optimize_growth(
                          x, p, *target, terminal_demand, cheap_geometry, work))
                      if (!best || proposal->cost < best->cost)
                        {
                          best   = std::move(proposal);
                          chosen = &x;
                        }
                  }

              optimization_seconds += timer.wall_time() - optimization_begin;
            }
        optimization_end = timer.wall_time();
        unsigned int has = best ? 1 : 0, prefix = 0;
        MPI_Exscan(&has, &prefix, 1, MPI_UNSIGNED, MPI_SUM, comm);
        if (rank == 0)
          prefix = 0;
        if (prefix >= terminal_count - count)
          best.reset();
        std::vector<BatchProposal> proposals;
        if (best)
          {
            ++run.proposed_insertions;
            BatchProposal proposal;
            proposal.id   = allocated[2];
            proposal.cost = best->cost;
            proposal.changed.assign(best->changed.begin(),
                                    best->changed.begin() + 3);
            const auto &v                 = chosen->path.front();
            proposal.owners[v.edge.id]    = v.owner;
            proposal.old_nodes[v.edge.id] = v.edge.nodes;
            for (const auto &edge : proposal.changed)
              for (Id node : edge.nodes)
                for (unsigned int end = 0; end < 2; ++end)
                  if (node == v.edge.nodes[end] &&
                      normalized_port_normal(v.port_normals[end])
                          .norm_square() > 0.)
                    {
                      proposal.port_owners[node] = v.node_owners[end];
                      proposal.normals[node]     = v.port_normals[end];
                    }
            proposals.push_back(proposal);
          }
        // Trials used proposed Murray radii: unchanged siblings shrink.
        // Current snapshot radii cannot certify those trials. The commit
        // below always certifies the fully normalized joint snapshot.
        const auto selection =
          select_growth_batch(geometry, accepted, proposals, p, comm);
        std::vector<GrowthSplit> splits;
        if (best && selection.local.at(allocated[2]))
          {
            const auto &v = chosen->path.front();
            GrowthSplit split;
            split.vessel            = best->vessel;
            split.owner             = v.owner;
            split.old_nodes         = v.edge.nodes;
            split.old_points        = v.edge.points;
            split.junction          = allocated[0];
            split.tip               = allocated[0] + 1;
            split.junction_position = best->junction;
            split.tip_position      = best->terminal;
            split.continuation      = allocated[1];
            split.leaf              = allocated[1] + 1;
            split.terminal          = allocated[2];
            if (system == System::arterial)
              split.territory = allocated[3];
            else if (system == System::venous || system == System::portal)
              split.territory = shared_territory;
            split.demand = best->terminal_demand;
            splits.push_back(split);
          }
        const auto selected =
          Utilities::MPI::sum(static_cast<unsigned int>(splits.size()), comm);
        selection_end = timer.wall_time();
        if (selected == 0)
          {
            ++failures;
            if (demand_callbacks.discard)
              demand_callbacks.discard();
            record(nullptr);
            continue;
          }
        GrowthDemandJointUpdate joint;
        if (demand_callbacks.joint)
          {
            std::vector<GrowthProposalKey> keys;
            for (const auto &split : splits)
              keys.emplace_back(rank, split.terminal);
            joint = demand_callbacks.joint(keys, accepted.revision());
          }
        else
          {
            joint.update.snapshot_revision = accepted.revision();
            for (const auto &split : splits)
              joint.new_terminals.push_back({split.terminal, split.demand});
          }
        const bool joint_valid =
          Utilities::MPI::min(static_cast<unsigned int>(
                                joint.validation.valid() &&
                                !joint.integration_budget_exhausted),
                              comm) == 1;
        const bool joint_budget =
          Utilities::MPI::max(static_cast<unsigned int>(
                                joint.integration_budget_exhausted),
                              comm) == 1;
        if (!joint_valid)
          {
            if (demand_callbacks.discard)
              demand_callbacks.discard();
            if (!joint_budget)
              {
                run.validation = joint.validation;
                phase(false,
                      "Joint integrated territory demand evaluation failed.");
                return finish();
              }
            integration_budget_exhausted = true;
            count_work(work,
                       GrowthOperation::integration_budget_rejections,
                       splits.size());
            ++run.rejected_batches;
            ++failures;
            const GrowthExecutionResult rejected;
            record(&rejected);
            continue;
          }
        std::map<Id, double> joint_demands;
        for (const auto &item : joint.new_terminals)
          valid =
            joint_demands.emplace(item.terminal, item.demand).second && valid;
        for (auto &split : splits)
          {
            const auto found = joint_demands.find(split.terminal);
            valid            = valid && found != joint_demands.end();
            if (found != joint_demands.end())
              split.demand = found->second;
          }
        if (
          !phase(
            valid,
            "Every selected split requires its true joint integrated demand."))
          {
            if (demand_callbacks.discard)
              demand_callbacks.discard();
            return finish();
          }
        auto committed = execution.commit(splits, joint, demand_callbacks);
        if (!committed.committed)
          {
            ++run.rejected_batches;
            ++failures;
            record(&committed);
            continue;
          }
        if ((system == System::venous || system == System::portal) &&
            !splits.empty())
          {
            run.owned_venous_terminals.emplace(shared_territory,
                                               splits.front().terminal);
            pending.erase(shared_territory);
          }
        run.accepted_insertions += splits.size();
        if (prescribed && !splits.empty())
          prescribed_pending.pop_back();
        run.maximum_batch_insertions =
          std::max(run.maximum_batch_insertions, selected);
        count += selected;
        failures = 0;
        record(&committed);
      }
    std::uint64_t unserved = 0;
    if (prescribed)
      unserved = Utilities::MPI::sum(
        static_cast<std::uint64_t>(prescribed_pending.size()), comm);
    if (system == System::venous || system == System::portal)
      unserved =
        Utilities::MPI::sum(static_cast<std::uint64_t>(pending.size()), comm);
    if (count != terminal_count && integration_budget_exhausted)
      phase(
        false,
        "Tissue integration exhausted its point or refinement budget for rejected territories; no demand floor was applied.");
    phase(
      count == terminal_count && unserved == 0,
      "The finite batched sampling budget was exhausted before reaching the terminal target.");
    return finish();
  }
} // namespace ImmersX::Vascularization::internal
