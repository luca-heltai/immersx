// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_loop.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/timer.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/generation.h>

#include <random>

#include "growth_batch.h"
#include "growth_paths.h"
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
            std::vector<Leaf> candidates;
            count_work(work, GrowthOperation::network_index_queries);
            tree.query(bgi::nearest(point, k), std::back_inserter(candidates));
            count_work(work,
                       GrowthOperation::network_index_hits,
                       candidates.size());
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
            if (!queried.count(owner) && !boxes[owner].empty() &&
                lower_bound(*target, boxes[owner][0]) <= cutoff)
              send[owner] = *target;
        }
      exchange(send);
      return hits;
    }
    struct Obstacle
    {
      Edge edge;
      Id   root = invalid_id;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &edge &root;
      }
    };
    // Fetch geometry covering every possible local-search trial and each fixed
    // ancestral segment. Replies come from physical owners, including owners
    // of imported exterior vessels that have no nearby background cells.
    std::map<Id, Obstacle>
    candidate_obstacles(const NetworkData                          &network,
                        const ForestState                          &state,
                        const GrowthIndices                        &indices,
                        const std::map<Id, GrowthCandidateContext> &contexts,
                        const std::optional<Point<3>>              &target,
                        const Parameters                           &p,
                        MPI_Comm                                    comm,
                        GrowthWork                                 *work)
    {
      const unsigned int rank  = Utilities::MPI::this_mpi_process(comm),
                         size  = Utilities::MPI::n_mpi_processes(comm);
      const double max_radius  = indices.radius_bound;
      const auto  &tree        = indices.owned;
      const auto   local_boxes = GrowthIndices::covering(tree);
      const auto   boxes       = Utilities::MPI::all_gather(comm, local_boxes);
      std::vector<BoundingBox<3>> regions;
      if (target)
        for (const auto &[id, x] : contexts)
          for (std::size_t i = 0; i < x.path.size(); ++i)
            {
              const auto  &edge = x.path[i].edge;
              const double reach =
                i == 0 ? .25 * edge.points[0].distance(edge.points[1]) *
                           p.optimization_steps :
                         0.;
              const double guard = 64. *
                                   std::numeric_limits<double>::epsilon() *
                                   std::max({1.,
                                             target->norm(),
                                             edge.points[0].norm(),
                                             edge.points[1].norm()});
              const double pad =
                reach + x.root_radius + max_radius + p.clearance + guard;
              Point<3> a, b;
              for (unsigned int d = 0; d < 3; ++d)
                {
                  a[d] = std::min(edge.points[0][d], edge.points[1][d]);
                  b[d] = std::max(edge.points[0][d], edge.points[1][d]);
                  if (i == 0)
                    {
                      a[d] = std::min(a[d], (*target)[d]);
                      b[d] = std::max(b[d], (*target)[d]);
                    }
                  a[d] -= pad;
                  b[d] += pad;
                }
              regions.emplace_back(std::make_pair(a, b));
            }
      std::map<unsigned int, std::vector<BoundingBox<3>>> send;
      for (unsigned int owner = 0; owner < size; ++owner)
        for (const auto &box : boxes[owner])
          for (const auto &region : regions)
            if (boost::geometry::intersects(box, region))
              send[owner].push_back(region);
      const auto incoming = Utilities::MPI::some_to_some(comm, send);
      std::map<unsigned int, std::vector<Obstacle>> replies;
      for (const auto &[source, query] : incoming)
        {
          std::set<Id> ids;
          count_work(work,
                     GrowthOperation::network_index_queries,
                     query.size());
          for (const auto &region : query)
            for (const auto &[segment, id] :
                 tree | bgi::adaptors::queried(bgi::intersects(region)))
              {
                count_work(work, GrowthOperation::network_index_hits);
                ids.insert(id);
              }
          for (Id id : ids)
            {
              const auto &v = network.vessels.at(id);
              const auto  b = state.owned.find(id);
              replies[source].push_back(
                {{id,
                  v.nodes,
                  {{network.nodes.at(v.nodes[0]).position,
                    network.nodes.at(v.nodes[1]).position}},
                  std::max(v.radii[0], v.radii[1])},
                 b == state.owned.end() ? invalid_id : b->second.root});
            }
        }
      const auto received = Utilities::MPI::some_to_some(comm, replies);
      std::map<Id, Obstacle> result;
      for (const auto &[owner, values] : received)
        for (const auto &obstacle : values)
          result.emplace(obstacle.edge.id, obstacle);
      return result;
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
              const std::vector<Territory>   &owned_territories)
  {
    using namespace dealii;
    Timer              timer;
    GrowthRun          run;
    auto              *work = &run.work;
    const unsigned int terminal_count =
      requested_count == 0 ? p.terminal_count : requested_count;
    const unsigned int rank = Utilities::MPI::this_mpi_process(comm),
                       size = Utilities::MPI::n_mpi_processes(comm);
    const auto phase        = [&](bool valid, const std::string &message) {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1;
      if (!all)
        run.validation.issues.push_back({"growth", invalid_id, message});
      return all;
    };
    bool   valid = system == System::arterial || system == System::venous;
    double gamma = 0.;
    if (system == System::arterial)
      gamma = p.arterial_murray_exponent;
    else if (system == System::venous)
      gamma = p.venous.murray_exponent;
    if (system == System::venous)
      valid = valid && std::isfinite(p.venous.terminal_separation) &&
              p.venous.terminal_separation > 0. && std::isfinite(gamma) &&
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
    const auto controls = std::make_pair(
      std::array<double, 9>{{static_cast<double>(terminal_count),
                             static_cast<double>(p.random_seed),
                             static_cast<double>(p.sample_budget),
                             static_cast<double>(p.candidate_count),
                             static_cast<double>(p.optimization_steps),
                             p.density_beta,
                             p.terminal_weight_bound,
                             static_cast<double>(system),
                             p.venous.terminal_separation}},
      std::array<std::string, 2>{{p.terminal_weight, p.tissue_demand}});
    for (const auto &other : Utilities::MPI::all_gather(comm, controls))
      valid = valid && other == controls;
    if (!phase(valid,
               "All producers must use the same growth and sampling controls."))
      return run;
    run.accepted = commit_growth_splits(bg,
                                        mapping,
                                        initial,
                                        mutable_owned,
                                        fixed_owned_roots,
                                        {},
                                        p,
                                        comm,
                                        nullptr,
                                        work);
    if (!run.accepted.committed)
      {
        run.validation = run.accepted.validation;
        return run;
      }
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
    std::map<Id, Territory> pending;
    if (system == System::venous)
      {
        std::map<unsigned int, std::map<Id, double>> registration;
        for (const auto &territory : owned_territories)
          {
            valid = valid && territory.id != invalid_id &&
                    territory.owner == rank &&
                    std::isfinite(territory.demand) && territory.demand > 0. &&
                    pending.emplace(territory.id, territory).second;
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
    double local_capacity = 0.;
    for (const auto &[id, radius] : run.accepted.fixed_owned_roots)
      local_capacity +=
        p.minimum_radius == 0. ?
          terminal_count :
          std::min<double>(
            terminal_count,
            std::floor(std::pow(radius / p.minimum_radius, gamma) *
                       (1. + 32. * std::numeric_limits<double>::epsilon())));
    if (
      !phase(
        Utilities::MPI::sum(local_capacity, comm) >= terminal_count,
        "Requested terminal count is incompatible with Murray's minimum-radius capacity at the real fixed roots."))
      return run;
    BackgroundGeometry geometry(bg, mapping, work);
    GrowthIndices      indices(run.accepted.network,
                          run.accepted.mutable_owned,
                          run.accepted.fixed_owned_roots,
                          comm,
                          work);
    FunctionParser<3>  weight, demand;
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
    Point<3> lo, hi;
    bool     first = true;
    double   min_h = std::numeric_limits<double>::infinity(), max_h = 0.;
    for (const auto &cell : bg.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto ends = cell->bounding_box().get_boundary_points();
          for (unsigned int d = 0; d < 3; ++d)
            {
              lo[d] = first ? ends.first[d] : std::min(lo[d], ends.first[d]);
              hi[d] = first ? ends.second[d] : std::max(hi[d], ends.second[d]);
            }
          first = false;
          min_h = std::min(min_h, cell->diameter());
          max_h = std::max(max_h, cell->diameter());
        }
    std::vector<BoundingBox<3>> local_boxes;
    double                      box_volume = 0.;
    if (!first)
      {
        local_boxes.emplace_back(std::make_pair(lo, hi));
        box_volume = 1.;
        for (unsigned int d = 0; d < 3; ++d)
          box_volume *= hi[d] - lo[d];
      }
    const auto   boxes          = Utilities::MPI::all_gather(comm, local_boxes);
    const double maximum_volume = Utilities::MPI::max(box_volume, comm);
    min_h                       = Utilities::MPI::min(min_h, comm);
    max_h                       = Utilities::MPI::max(max_h, comm);
    if (
      !phase(
        std::isfinite(maximum_volume) && maximum_volume > 0. &&
          std::isfinite(min_h) && min_h > 0. && std::isfinite(max_h),
        "Sampling requires finite positive background sizes and covering-box volumes."))
      return run;
    std::seed_seq   seed{p.random_seed, rank, 194837u};
    std::mt19937_64 random(seed);
    const auto      uniform = [&]() {
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
          next[3] = std::max(next[3], t.territory);
        }
    for (auto &id : next)
      {
        id = Utilities::MPI::max(id, comm);
        ++id;
      }
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
              run.accepted.mutable_owned.size();
        }
      return statistics;
    };
    bucket(count).initialization_seconds = timer.wall_time();
    bucket(count).work += *work;
    unsigned int failures = 0;
    while (count < terminal_count && failures < p.sample_budget)
      {
        auto        &statistics       = bucket(count);
        const auto   work_before      = *work;
        const auto   owned_before     = run.accepted.mutable_owned.size();
        const auto   samples_before   = run.sampled_points;
        const auto   proposals_before = run.proposed_insertions;
        const auto   accepted_before  = run.accepted_insertions;
        const double batch_begin      = timer.wall_time();
        double       sampling_end = batch_begin, search_end = batch_begin,
               optimization_end = batch_begin, selection_end = batch_begin;
        const auto record = [&](const GrowthCommit *commit) {
          const double end = timer.wall_time();
          statistics.work += *work - work_before;
          ++statistics.batches;
          statistics.sampled_points += run.sampled_points - samples_before;
          statistics.proposed_insertions +=
            run.proposed_insertions - proposals_before;
          statistics.accepted_insertions +=
            run.accepted_insertions - accepted_before;
          statistics.active_writer_batches += owned_before > 0;
          statistics.last_terminals        = count;
          statistics.owned_vessels_last    = run.accepted.mutable_owned.size();
          statistics.owned_vessels_maximum = std::max<std::uint64_t>(
            statistics.owned_vessels_maximum,
            std::max<std::uint64_t>(owned_before,
                                    statistics.owned_vessels_last));
          statistics.sampling_seconds += sampling_end - batch_begin;
          statistics.search_seconds += search_end - sampling_end;
          statistics.optimization_seconds += optimization_end - search_end;
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
            const Id stride = kind < 2 ? 2 : 1;
            valid           = valid && next[kind] < invalid_id - stride * size;
            allocated[kind] = next[kind] + stride * rank;
            next[kind] += stride * size;
          }
        if (!phase(valid,
                   "The physical vascular identifier space is exhausted."))
          return run;
        std::optional<Point<3>> raw;
        Id                      shared_territory = invalid_id;
        if (system == System::arterial && !first)
          {
            raw.emplace();
            for (unsigned int d = 0; d < 3; ++d)
              (*raw)[d] = lo[d] + uniform() * (hi[d] - lo[d]);
            ++run.sampled_points;
          }
        else if (system == System::venous && !pending.empty())
          {
            auto entry = pending.begin();
            std::advance(entry, ((run.batches - 1) / 6) % pending.size());
            shared_territory = entry->first;
            raw              = entry->second.position;
            const unsigned int direction =
              ((run.batches - 1) % 6 + shared_territory % 6) % 6;
            (*raw)[direction / 2] +=
              (direction % 2 ? 1. : -1.) * p.venous.terminal_separation;
            ++run.sampled_points;
          }
        // Query actual containing cells for h(x), including MPI interfaces.
        // Only coarse covering boxes are replicated, never the vessel graph.
        std::map<unsigned int, Point<3>> sample_send;
        if (raw)
          for (unsigned int owner = 0; owner < size; ++owner)
            for (const auto &box : boxes[owner])
              if (box.point_inside(*raw))
                sample_send[owner] = *raw;
        const auto sample_queries =
          Utilities::MPI::some_to_some(comm, sample_send);
        std::map<unsigned int, double> sample_replies;
        for (const auto &[source, point] : sample_queries)
          if (const auto cell = geometry.containing(point))
            sample_replies[source] = (*cell)->diameter();
        const auto located = Utilities::MPI::some_to_some(comm, sample_replies);
        unsigned int point_owner = size;
        double       h           = std::numeric_limits<double>::infinity();
        for (const auto &[owner, diameter] : located)
          {
            point_owner = std::min(point_owner, owner);
            h           = std::min(h, diameter);
          }
        std::optional<Point<3>> target;
        double                  terminal_demand = 0.;
        if (system == System::venous && raw && point_owner < size)
          {
            target          = raw;
            terminal_demand = pending.at(shared_territory).demand;
          }
        else if (system == System::arterial && raw && point_owner == rank)
          {
            const double w = weight.value(*raw);
            valid          = valid && std::isfinite(w) && w >= 0. &&
                    w <= p.terminal_weight_bound;
            const double scale = p.density_beta >= 0. ? min_h : max_h;
            if (valid && uniform() < box_volume / maximum_volume * w /
                                       p.terminal_weight_bound *
                                       std::pow(scale / h, p.density_beta))
              {
                target          = raw;
                terminal_demand = demand.value(*target);
                valid = std::isfinite(terminal_demand) && terminal_demand > 0.;
              }
          }
        if (
          !phase(
            valid,
            "Spatial weight violates its bound, or tissue demand is not positive and finite."))
          return run;
        sampling_end = timer.wall_time();
        const auto nearby =
          nearest(target, indices, p.candidate_count, comm, system, work);
        std::map<Id, unsigned int> requests;
        for (const auto &hit : nearby)
          requests[hit.id] = hit.owner;
        const auto paths = fetch_growth_candidates(run.accepted.network,
                                                   run.accepted.physiology,
                                                   requests,
                                                   allocated[0],
                                                   allocated[1],
                                                   p,
                                                   comm,
                                                   work);
        if (!paths.validation.valid())
          {
            run.validation = paths.validation;
            return run;
          }
        const auto obstacles = candidate_obstacles(run.accepted.network,
                                                   run.accepted.physiology,
                                                   indices,
                                                   paths.contexts,
                                                   target,
                                                   p,
                                                   comm,
                                                   work);
        search_end           = timer.wall_time();
        std::vector<Leaf> obstacle_leaves;
        double            obstacle_radius = 0.;
        for (const auto &[id, obstacle] : obstacles)
          {
            obstacle_leaves.push_back(
              {{obstacle.edge.points[0], obstacle.edge.points[1]}, id});
            obstacle_radius = std::max(obstacle_radius, obstacle.edge.radius);
          }
        count_work(work, GrowthOperation::network_index_builds);
        count_work(work,
                   GrowthOperation::network_index_insertions,
                   obstacle_leaves.size());
        const auto obstacle_tree = pack_rtree(obstacle_leaves);
        std::optional<GrowthCandidate> best;
        const GrowthCandidateContext  *chosen = nullptr;
        if (target)
          for (const auto &hit : nearby)
            {
              const auto  &x = paths.contexts.at(hit.id);
              const double w = terminal_demand / x.demand_scale;
              const double shrink =
                std::pow(x.root_demand / (x.root_demand + w),
                         1. / x.murray_exponent);
              count_work(work,
                         GrowthOperation::candidate_path_visits,
                         x.path.size());
              std::map<Id, double> path_demands;
              for (const auto &branch : x.path)
                path_demands[branch.edge.id] = branch.demand;
              const auto cheap_geometry = [&](
                                            const std::vector<Edge> &changed) {
                std::set<Id> replaced;
                for (const auto &edge : changed)
                  replaced.insert(edge.id);
                for (std::size_t i = 0; i < changed.size(); ++i)
                  {
                    std::array<Point<3>, 2> normals;
                    count_work(work,
                               GrowthOperation::candidate_path_visits,
                               x.path.size());
                    for (const auto &branch : x.path)
                      for (unsigned int end = 0; end < 2; ++end)
                        for (unsigned int old = 0; old < 2; ++old)
                          if (changed[i].nodes[end] == branch.edge.nodes[old])
                            normals[end] = branch.port_normals[old];
                    if (!geometry.boundary_separated(
                          {changed[i].points[0], changed[i].points[1]},
                          changed[i].radius + p.clearance,
                          p.geometry_depth,
                          p.geometry_work,
                          normals))
                      return false;
                    const auto  &e = changed[i];
                    const double pad =
                      e.radius + std::max(obstacle_radius, x.root_radius) +
                      p.clearance +
                      64. * std::numeric_limits<double>::epsilon() *
                        std::max({1., e.points[0].norm(), e.points[1].norm()});
                    Point<3> a, b;
                    for (unsigned int d = 0; d < 3; ++d)
                      {
                        a[d] = std::min(e.points[0][d], e.points[1][d]) - pad;
                        b[d] = std::max(e.points[0][d], e.points[1][d]) + pad;
                      }
                    count_work(work, GrowthOperation::network_index_queries);
                    for (const auto &[segment, id] :
                         obstacle_tree |
                           bgi::adaptors::queried(bgi::intersects(
                             BoundingBox<3>(std::make_pair(a, b)))))
                      {
                        count_work(work, GrowthOperation::network_index_hits);
                        if (!replaced.count(id))
                          {
                            auto obstacle = obstacles.at(id).edge;
                            if (obstacles.at(id).root == x.root)
                              obstacle.radius =
                                path_demands.count(id) ?
                                  x.root_radius *
                                    std::pow((path_demands.at(id) + w) /
                                               (x.root_demand + w),
                                             1. / x.murray_exponent) :
                                  obstacle.radius * shrink;
                            if (!compatible(e,
                                            obstacle,
                                            p.minimum_angle,
                                            p.clearance,
                                            work))
                              return false;
                          }
                      }
                    for (std::size_t j = 0; j < i; ++j)
                      if (!compatible(changed[i],
                                      changed[j],
                                      p.minimum_angle,
                                      p.clearance,
                                      work))
                        return false;
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
            proposal.id   = 0;
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
        const auto selection = select_growth_batch(
          geometry,
          run.accepted.network,
          proposals,
          p,
          comm,
          [&](Id id) {
            const auto &v = run.accepted.network.vessels.at(id);
            return std::max(v.radii[0], v.radii[1]);
          },
          false,
          work);
        std::vector<GrowthSplit> splits;
        if (best && selection.local.at(0))
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
            else if (system == System::venous)
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
            record(nullptr);
            continue;
          }
        auto committed = commit_growth_splits(bg,
                                              mapping,
                                              run.accepted.network,
                                              run.accepted.mutable_owned,
                                              run.accepted.fixed_owned_roots,
                                              splits,
                                              p,
                                              comm,
                                              &geometry,
                                              work);
        if (!committed.committed)
          {
            ++run.rejected_batches;
            ++failures;
            record(&committed);
            continue;
          }
        if (system == System::venous && !splits.empty())
          {
            run.owned_venous_terminals.emplace(shared_territory,
                                               splits.front().terminal);
            pending.erase(shared_territory);
          }
        indices.update(committed.network,
                       committed.mutable_owned,
                       committed.geometry_changes);
        run.accepted = std::move(committed);
        run.accepted_insertions += splits.size();
        run.maximum_batch_insertions =
          std::max(run.maximum_batch_insertions, selected);
        count += selected;
        failures = 0;
        record(&run.accepted);
      }
    std::uint64_t unserved = 0;
    if (system == System::venous)
      unserved =
        Utilities::MPI::sum(static_cast<std::uint64_t>(pending.size()), comm);
    phase(
      count == terminal_count && unserved == 0,
      "The finite batched sampling budget was exhausted before reaching the terminal target.");
    return run;
  }
} // namespace ImmersX::Vascularization::internal
