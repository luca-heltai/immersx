// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/generation.h>

#include "growth_execution.h"

namespace ImmersX::Vascularization::internal
{
  namespace
  {
    dealii::BoundingBox<3>
    envelope(const Edge &edge, double padding)
    {
      dealii::Point<3> low, high;
      const double     guard =
        64. * std::numeric_limits<double>::epsilon() *
        std::max({1., edge.points[0].norm(), edge.points[1].norm()});
      for (unsigned int d = 0; d < 3; ++d)
        {
          low[d] =
            std::min(edge.points[0][d], edge.points[1][d]) - padding - guard;
          high[d] =
            std::max(edge.points[0][d], edge.points[1][d]) + padding + guard;
        }
      return dealii::BoundingBox<3>({low, high});
    }
  } // namespace

  GrowthCandidateObstacles
  GrowthExecution::candidate_obstacles(
    const AcceptedSnapshot                     &snapshot,
    const GrowthDemandPreview                  &preview,
    const std::map<Id, GrowthCandidateContext> &contexts,
    const std::optional<dealii::Point<3>>      &target,
    double                                      new_demand,
    unsigned int                                producer) const
  {
    using namespace dealii;
    const auto               comm = communicator;
    const auto               rank = Utilities::MPI::this_mpi_process(comm);
    const auto               size = Utilities::MPI::n_mpi_processes(comm);
    GrowthCandidateObstacles result;
    bool valid = snapshot.owner == this && snapshot.current() &&
                 preview.validation.valid() &&
                 preview.snapshot_revision == demand_state.revision &&
                 producer < size && (rank == producer || contexts.empty());
    const auto producers = Utilities::MPI::all_gather(comm, producer);
    for (auto p : producers)
      valid = valid && p == producer;
    if (
      !phase(
        result.validation,
        valid,
        "Obstacle queries require one current accepted snapshot and producer."))
      return result;
    if (!parameters.enforce_geometry)
      {
        for (const auto &[id, context] : contexts)
          result.covered_ancestors.emplace(id, std::set<Id>{});
        return result;
      }
    new_demand = Utilities::MPI::broadcast(comm, new_demand, producer);
    std::map<Id, GrowthCandidateRadii> radii;
    std::map<Id, Id>                   context_roots;
    std::map<Id, unsigned int>         cap_requests;
    if (rank == producer)
      for (const auto &[id, context] : contexts)
        {
          valid = valid && target.has_value() && !context.path.empty() &&
                  std::isfinite(new_demand) && new_demand > 0.;
          if (context.path.empty())
            continue;
          valid = valid && context.path.front().edge.id == id &&
                  context.path.back().edge.id == context.root;
          valid =
            valid && (!context.physiology ||
                      context.physiology->path.size() == context.path.size());
          if (target)
            for (unsigned int d = 0; d < 3; ++d)
              valid = valid && std::isfinite((*target)[d]);
          if (!valid)
            continue;
          const auto value = candidate_radii(context, new_demand, work);
          valid            = valid && value.has_value();
          if (value)
            radii.emplace(id, *value);
          context_roots.emplace(id, context.root);
          for (const auto &branch : context.path)
            {
              const auto inserted =
                cap_requests.emplace(branch.edge.id, branch.owner);
              valid =
                valid && branch.owner < size &&
                (inserted.second || inserted.first->second == branch.owner);
            }
        }
    if (
      !phase(
        result.validation,
        valid,
        "Candidate obstacles require complete paths, native owners and positive demand."))
      return result;
    context_roots = Utilities::MPI::broadcast(comm, context_roots, producer);
    std::set<Id> changed_roots;
    for (const auto &[id, value] : preview.owned)
      changed_roots.insert(demand_state.branches.at(id).topology.root);
    std::set<Id> wanted;
    for (const auto &part : Utilities::MPI::all_gather(comm, changed_roots))
      wanted.insert(part.begin(), part.end());
    for (const auto &[id, root] : context_roots)
      wanted.insert(root);
    const auto current_roots =
      demand_state.root_controls(wanted, preview, result.validation);
    const auto old_roots =
      demand_state.root_controls(wanted, accepted_preview(), result.validation);
    if (!result.validation.valid())
      return result;

    // A normalization change moves every radius by at most this additive
    // amount. Only branches whose own mass increased need extra envelopes.
    double expansion = 0.;
    for (Id root : wanted)
      {
        const auto &now = current_roots.at(root);
        const auto &old = old_roots.at(root);
        if (now.summary.mass < old.summary.mass)
          {
            boost::multiprecision::cpp_rational ratio(old.summary.mass);
            ratio /= now.summary.mass;
            const long double factor =
              std::pow(ratio.convert_to<long double>(), 1.L / now.gamma);
            const double increment =
              std::min<long double>(now.radius, now.radius * (factor - 1.L));
            expansion =
              std::max(expansion,
                       std::nextafter(increment,
                                      std::numeric_limits<double>::infinity()));
          }
      }
    expansion = Utilities::MPI::max(expansion, comm);
    std::vector<ObstacleLeaf> extra;
    for (const auto &[id, value] : preview.owned)
      {
        const auto &cap = certificates.at(id);
        if (value.mass <= cap.mass)
          continue;
        const auto &root = current_roots.at(cap.root);
        auto        edge = cap.cap;
        edge.radius      = radius_at(value.mass, root, root.summary.mass);
        extra.emplace_back(envelope(edge,
                                    edge.radius + parameters.clearance *
                                                    cap.clearance_scale),
                           id);
      }
    const ObstacleTree overlay(extra.begin(), extra.end());
    count_work(work, GrowthOperation::network_index_builds);
    count_work(work, GrowthOperation::network_index_insertions, extra.size());

    // Off-path geometry must remain inside accepted certificates before
    // ancestral checks can be omitted. Active path pairs remain trial checks.
    std::map<Id, bool> covered;
    for (const auto &[candidate, selected_root] : context_roots)
      {
        bool       safe      = true;
        const Mass increment = demand_mass_units(new_demand);
        for (Id root : wanted)
          {
            Mass final_mass = current_roots.at(root).summary.mass;
            if (root == selected_root)
              final_mass += increment;
            const auto events = thresholds.find(root);
            if (events != thresholds.end() && !events->second.empty())
              safe = safe && final_mass >= events->second.rbegin()->first;
          }
        for (const auto &[id, value] : preview.owned)
          {
            const auto &cap = certificates.at(id);
            if (value.mass <= cap.mass)
              continue;
            const auto &root       = current_roots.at(cap.root);
            Mass        final_mass = root.summary.mass;
            if (cap.root == selected_root)
              final_mass += increment;
            safe =
              safe && radius_at(value.mass, root, final_mass) <= cap.cap.radius;
          }
        covered.emplace(candidate, safe);
      }
    for (const auto &part : Utilities::MPI::all_gather(comm, covered))
      for (const auto &[id, safe] : part)
        covered.at(id) = covered.at(id) && safe;

    std::map<unsigned int, std::vector<Id>> cap_send;
    for (const auto &[id, owner] : cap_requests)
      cap_send[owner].push_back(id);
    std::map<unsigned int, std::map<Id, double>> cap_replies;
    for (const auto &[source, ids] :
         Utilities::MPI::some_to_some(comm, cap_send))
      for (Id id : ids)
        {
          const auto cap = certificates.find(id);
          valid          = valid && cap != certificates.end();
          if (cap != certificates.end())
            cap_replies[source].emplace(id, cap->second.cap.radius);
        }
    std::map<Id, double> caps;
    for (const auto &[source, part] :
         Utilities::MPI::some_to_some(comm, cap_replies))
      for (const auto &[id, r] : part)
        valid = valid && cap_requests.count(id) &&
                cap_requests.at(id) == source && caps.emplace(id, r).second;
    valid = valid && caps.size() == cap_requests.size();
    if (!phase(result.validation,
               valid,
               "An ancestral certificate must resolve to its native writer."))
      return result;

    using Box = BoundingBox<3>;
    std::vector<Box> regions;
    if (rank == producer)
      for (const auto &[id, context] : contexts)
        {
          const auto &trial    = radii.at(id);
          const auto &selected = context.path.front().edge;
          Edge        search   = selected;
          for (unsigned int d = 0; d < 3; ++d)
            {
              search.points[0][d] = std::min(
                {selected.points[0][d], selected.points[1][d], (*target)[d]});
              search.points[1][d] = std::max(
                {selected.points[0][d], selected.points[1][d], (*target)[d]});
            }
          const double reach = .25 *
                               selected.points[0].distance(selected.points[1]) *
                               parameters.optimization_steps;
          regions.push_back(
            envelope(search, reach + trial.path_radii.at(id) + ghost_halo));
          auto &mask = result.covered_ancestors[id];
          for (std::size_t i = 1; i < context.path.size(); ++i)
            {
              auto edge   = context.path[i].edge;
              edge.radius = trial.path_radii.at(edge.id);
              if (covered.at(id) && edge.radius <= caps.at(edge.id))
                mask.insert(edge.id);
              else
                regions.push_back(envelope(edge, edge.radius + ghost_halo));
            }
        }
    std::vector<Box> local_cover;
    if (!obstacle_index.empty())
      {
        Box box;
        boost::geometry::convert(obstacle_index.bounds(), box);
        local_cover.push_back(box);
      }
    if (!overlay.empty())
      {
        Box box;
        boost::geometry::convert(overlay.bounds(), box);
        local_cover.push_back(box);
      }
    std::vector<std::pair<Box, unsigned int>> covers;
    const auto cover_parts = Utilities::MPI::all_gather(comm, local_cover);
    for (unsigned int owner = 0; owner < cover_parts.size(); ++owner)
      for (const auto &box : cover_parts[owner])
        covers.emplace_back(box, owner);
    const auto coarse = pack_rtree(covers);
    count_work(work, GrowthOperation::network_index_builds);
    count_work(work, GrowthOperation::network_index_insertions, covers.size());
    std::map<unsigned int, std::vector<Box>> queries;
    for (auto box : regions)
      {
        auto points = box.get_boundary_points();
        for (unsigned int d = 0; d < 3; ++d)
          {
            points.first[d] -= expansion;
            points.second[d] += expansion;
          }
        box = Box(points);
        std::set<unsigned int> owners;
        for (const auto &[cover, owner] :
             coarse | bgi::adaptors::queried(bgi::intersects(box)))
          owners.insert(owner);
        for (auto owner : owners)
          queries[owner].push_back(box);
      }
    std::map<unsigned int, std::set<Id>> replies;
    for (const auto &[source, boxes] :
         Utilities::MPI::some_to_some(comm, queries))
      for (const auto &box : boxes)
        for (const auto *tree : {&obstacle_index, &overlay})
          {
            count_work(work, GrowthOperation::network_index_queries);
            for (const auto &[bounds, id] :
                 *tree | bgi::adaptors::queried(bgi::intersects(box)))
              {
                count_work(work, GrowthOperation::network_index_hits);
                replies[source].insert(id);
              }
          }
    for (const auto &[owner, ids] : Utilities::MPI::some_to_some(comm, replies))
      for (Id id : ids)
        {
          const auto inserted = result.owners.emplace(id, owner);
          valid = valid && (inserted.second || inserted.first->second == owner);
        }
    if (!phase(
          result.validation,
          valid,
          "Candidate obstacle envelopes require one native owner per vessel."))
      return result;
    auto physical =
      demand_state.physical_edges(rows.network(), preview, result.owners);
    result.validation = std::move(physical.validation);
    result.records    = std::move(physical.records);
    return result;
  }
} // namespace ImmersX::Vascularization::internal
