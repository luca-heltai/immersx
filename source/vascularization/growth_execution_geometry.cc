// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/generation.h>

#include <cmath>

#include "growth_execution.h"
#include "growth_wall_certificates.h"
namespace ImmersX::Vascularization::internal
{
  namespace
  {
    struct ObstacleCertificate
    {
      Edge   cap;
      double clearance_scale = 0.;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &cap &clearance_scale;
      }
    };
    dealii::BoundingBox<3>
    query_box(const Edge &edge, double padding)
    {
      dealii::Point<3> lo, hi;
      for (unsigned int d = 0; d < 3; ++d)
        {
          lo[d] = std::min(edge.points[0][d], edge.points[1][d]) - padding;
          hi[d] = std::max(edge.points[0][d], edge.points[1][d]) + padding;
        }
      return dealii::BoundingBox<3>(std::make_pair(lo, hi));
    }
  } // namespace
  dealii::BoundingBox<3>
  GrowthExecution::certificate_box(const Certificate &value) const
  {
    const double guard =
      64. * std::numeric_limits<double>::epsilon() *
      std::max({1., value.cap.points[0].norm(), value.cap.points[1].norm()});
    return query_box(value.cap,
                     value.cap.radius +
                       parameters.clearance * value.clearance_scale + guard);
  }
  GrowthExecution::Certificate
  GrowthExecution::certificate_at(Id                         id,
                                  const NetworkData         &network,
                                  const GrowthDemandPreview &preview,
                                  double                     clearance_scale,
                                  const Mass                &lower) const
  {
    const auto &v      = network.vessels.at(id);
    const auto &branch = demand_state.branches.at(id);
    const auto &root   = controls.at(branch.topology.root);
    Certificate value;
    value.root            = branch.topology.root;
    value.mass            = demand_state.summary(id, preview).mass;
    value.lower_root_mass = lower;
    value.upper_root_mass =
      id == value.root ? Mass(0) : Mass(2 * root.summary.mass);
    value.clearance_scale = clearance_scale;
    const double cap =
      value.mass >= lower ? root.radius : radius_at(value.mass, root, lower);
    value.cap = {id,
                 v.nodes,
                 {{network.nodes.at(v.nodes[0]).position,
                   network.nodes.at(v.nodes[1]).position}},
                 std::min(root.radius, std::max(cap, radius(id, preview)))};
    if (value.cap.radius == root.radius)
      value.lower_root_mass = 0;
    return value;
  }
  void
  GrowthExecution::replace_certificate(Id                         id,
                                       std::optional<Certificate> value,
                                       bool                       remember)
  {
    const auto old = certificates.find(id);
    if (remember && !certificates_before.count(id))
      certificates_before.emplace(id,
                                  old == certificates.end() ?
                                    std::optional<Certificate>{} :
                                    old->second);
    std::optional<ObstacleLeaf> old_leaf, new_leaf;
    if (old != certificates.end())
      old_leaf = ObstacleLeaf{certificate_box(old->second), id};
    if (value)
      new_leaf = ObstacleLeaf{certificate_box(*value), id};
    const bool same_box = old_leaf && new_leaf &&
                          old_leaf->first.get_boundary_points() ==
                            new_leaf->first.get_boundary_points();
    if (old_leaf && !same_box)
      {
        AssertThrow(obstacle_index.remove(*old_leaf) == 1,
                    dealii::ExcInternalError());
        count_work(work, GrowthOperation::network_index_removals);
      }
    if (old != certificates.end())
      {
        if (old->second.root != invalid_id)
          {
            const Id root   = old->second.root;
            auto    &events = thresholds.at(root);
            auto     event  = events.find(old->second.lower_root_mass);
            AssertThrow(event != events.end() && event->second.erase(id) == 1,
                        dealii::ExcInternalError());
            if (event->second.empty())
              events.erase(event);
            if (events.empty())
              thresholds.erase(root);
            if (old->second.upper_root_mass > 0)
              {
                auto &shrinks = shrink_thresholds.at(root);
                auto  high    = shrinks.find(old->second.upper_root_mass);
                AssertThrow(high != shrinks.end() &&
                              high->second.erase(id) == 1,
                            dealii::ExcInternalError());
                if (high->second.empty())
                  shrinks.erase(high);
                if (shrinks.empty())
                  shrink_thresholds.erase(root);
              }
            auto members = root_members.find(root);
            AssertThrow(members != root_members.end() && members->second > 0,
                        dealii::ExcInternalError());
            if (--members->second == 0)
              root_members.erase(members);
            root_members_changed.insert(root);
          }
        certificates.erase(old);
      }
    if (value)
      {
        if (value->root != invalid_id)
          {
            thresholds[value->root][value->lower_root_mass].insert(id);
            if (value->upper_root_mass > 0)
              shrink_thresholds[value->root][value->upper_root_mass].insert(id);
            ++root_members[value->root];
            root_members_changed.insert(value->root);
          }
        certificates.emplace(id, std::move(*value));
      }
    if (new_leaf && !same_box)
      {
        obstacle_index.insert(*new_leaf);
        count_work(work, GrowthOperation::network_index_insertions);
      }
  }
  bool
  GrowthExecution::certify(const GrowthOwnershipDelta   &delta,
                           const GrowthDemandPreview    &preview,
                           const GrowthGeometryMeasures &measures,
                           ValidationReport             &validation)
  {
    using namespace dealii;
    const auto comm = communicator;
    const auto rank = Utilities::MPI::this_mpi_process(comm);
    for (const auto &[id, value] : measures)
      clearance_bound = std::max(clearance_bound, value.clearance_scale);
    clearance_bound        = Utilities::MPI::max(clearance_bound, comm);
    const double clearance = parameters.clearance * clearance_bound;
    if (!phase(validation,
               std::isfinite(clearance),
               "A joint physical geometric envelope is not representable."))
      return false;
    for (Id id : delta.released_owned)
      replace_certificate(id, {});
    std::set<Id> reconsider = delta.changed_owned;
    for (const auto &[id, value] : preview.owned)
      {
        const auto old = certificates.find(id);
        if (old == certificates.end() || old->second.mass != value.mass)
          reconsider.insert(id);
      }
    for (const auto &[root, before] : controls_before)
      {
        const auto &mass   = controls.at(root).summary.mass;
        const auto  events = thresholds.find(root);
        if (events != thresholds.end())
          for (auto it = events->second.upper_bound(mass);
               it != events->second.end();
               ++it)
            reconsider.insert(it->second.begin(), it->second.end());
        const auto shrinks = shrink_thresholds.find(root);
        if (shrinks != shrink_thresholds.end())
          for (auto it = shrinks->second.begin();
               it != shrinks->second.upper_bound(mass);
               ++it)
            reconsider.insert(it->second.begin(), it->second.end());
      }
    std::set<Id> check;
    bool         valid = true;
    // Explicit mass bands avoid inverse floating thresholds. Doubling events
    // shrink untouched envelopes with amortized updates during point growth.
    const Mass denominator = 2;
    for (Id id : reconsider)
      {
        const auto &branch = demand_state.branches.at(id);
        const auto &root   = controls.at(branch.topology.root);
        Mass lower = (root.summary.mass * (denominator - 1)) / denominator;
        if (lower == 0)
          lower = 1;
        auto value = certificate_at(
          id, rows.network(), preview, branch.geometry.clearance_scale, lower);
        const auto old     = certificates.find(id);
        const bool changed = delta.changed_owned.count(id);
        if (old == certificates.end() || changed ||
            value.cap.radius > old->second.cap.radius)
          check.insert(id);
        valid =
          valid && std::isfinite(value.cap.radius) && value.cap.radius > 0.;
        if (changed)
          valid = valid && branch.geometry.normalized_length >=
                             parameters.minimum_length;
        replace_certificate(id, std::move(value));
      }
    if (
      !phase(
        validation,
        valid,
        "Joint geometry needs positive radii and the configured normalized lengths."))
      return false;
    auto wall_request = [&](Id id, double r) {
      const auto       &cap = certificates.at(id);
      GrowthWallRequest request;
      request.edge         = cap.cap;
      request.edge.radius  = r;
      request.padding      = parameters.clearance * cap.clearance_scale;
      request.check_domain = delta.changed_owned.count(id);
      for (unsigned int end = 0; end < 2; ++end)
        {
          const auto &node = rows.network().nodes.at(cap.cap.nodes[end]);
          if (node.port != Port::none)
            request.port_normals[end] = node.inward_port_normal;
        }
      return request;
    };
    std::map<Id, GrowthWallRequest> walls;
    for (Id id : check)
      walls.emplace(id, wall_request(id, certificates.at(id).cap.radius));
    auto wall = certify_growth_walls(geometry, walls, parameters, comm, work);
    if (!wall.validation.valid())
      {
        validation = std::move(wall.validation);
        return false;
      }
    std::map<Id, GrowthWallRequest> actual_walls;
    for (Id id : check)
      if (!wall.local.at(id))
        actual_walls.emplace(id, wall_request(id, radius(id, preview)));
    auto actual_wall =
      certify_growth_walls(geometry, actual_walls, parameters, comm, work);
    if (!actual_wall.validation.valid())
      {
        validation = std::move(actual_wall.validation);
        return false;
      }
    for (const auto &[id, request] : actual_walls)
      {
        valid               = actual_wall.local.at(id) && valid;
        auto cap            = certificates.at(id);
        cap.cap.radius      = request.edge.radius;
        cap.lower_root_mass = cap.cap.radius == controls.at(cap.root).radius ?
                                Mass(0) :
                                controls.at(cap.root).summary.mass;
        replace_certificate(id, std::move(cap));
      }
    if (
      !phase(
        validation,
        valid,
        "A joint generated segment leaves the physical domain or crosses a wall."))
      return false;
    using Box = BoundingBox<3>;
    std::vector<std::pair<Box, unsigned int>> rank_boxes;
    std::vector<Box>                          local_cover;
    if (!obstacle_index.empty())
      {
        Box bounds;
        boost::geometry::convert(obstacle_index.bounds(), bounds);
        local_cover.push_back(bounds);
      }
    const auto covers = Utilities::MPI::all_gather(comm, local_cover);
    for (unsigned int owner = 0; owner < covers.size(); ++owner)
      for (const auto &box : covers[owner])
        rank_boxes.emplace_back(box, owner);
    const auto coarse = pack_rtree(rank_boxes);
    count_work(work, GrowthOperation::network_index_builds);
    count_work(work,
               GrowthOperation::network_index_insertions,
               rank_boxes.size());
    std::map<unsigned int, std::vector<std::pair<Id, Box>>> queries;
    for (Id id : check)
      {
        const auto             box = certificate_box(certificates.at(id));
        std::set<unsigned int> owners;
        for (const auto &[cover, owner] :
             coarse | bgi::adaptors::queried(bgi::intersects(box)))
          owners.insert(owner);
        for (auto owner : owners)
          queries[owner].emplace_back(id, box);
      }
    using Replies = std::map<Id, std::map<Id, ObstacleCertificate>>;
    std::map<unsigned int, Replies> replies;
    for (const auto &[producer, requests] :
         Utilities::MPI::some_to_some(comm, queries))
      for (const auto &[query, box] : requests)
        {
          count_work(work, GrowthOperation::network_index_queries);
          for (const auto &[envelope, id] :
               obstacle_index | bgi::adaptors::queried(bgi::intersects(box)))
            {
              count_work(work, GrowthOperation::network_index_hits);
              if (id == query)
                continue;
              const auto &cap = certificates.at(id);
              replies[producer][query].emplace(
                id, ObstacleCertificate{cap.cap, cap.clearance_scale});
            }
        }
    std::map<Id, unsigned int> requested;
    std::map<Id, std::map<Id, std::pair<unsigned int, ObstacleCertificate>>>
      nearby;
    for (const auto &[owner, records] :
         Utilities::MPI::some_to_some(comm, replies))
      for (const auto &[query, obstacles] : records)
        for (const auto &[id, cap] : obstacles)
          {
            nearby[query].emplace(id, std::make_pair(owner, cap));
            const auto found = requested.emplace(id, owner);
            valid = valid && (found.second || found.first->second == owner);
          }
    if (!phase(validation,
               valid,
               "Each geometric obstacle needs one native writing owner."))
      return false;
    const auto actual =
      demand_state.physical_edges(rows.network(), preview, requested);
    if (!actual.validation.valid())
      {
        validation = actual.validation;
        return false;
      }
    std::map<unsigned int, std::map<Id, double>> tighten;
    for (const auto &[query, obstacles] : nearby)
      for (const auto &[id, obstacle] : obstacles)
        {
          auto         a = certificates.at(query).cap;
          const auto  &b = obstacle.second.cap;
          const double pair_clearance =
            parameters.clearance *
            std::max(certificates.at(query).clearance_scale,
                     obstacle.second.clearance_scale);
          if (!compatible(a, b, parameters.minimum_angle, pair_clearance, work))
            {
              a.radius             = radius(query, preview);
              const auto &physical = actual.records.at(id).edge;
              if (!compatible(a,
                              physical,
                              parameters.minimum_angle,
                              pair_clearance,
                              work))
                valid = false;
              else
                {
                  tighten[rank][query]        = a.radius;
                  tighten[obstacle.first][id] = physical.radius;
                }
            }
        }
    for (const auto &[producer, records] :
         Utilities::MPI::some_to_some(comm, tighten))
      for (const auto &[id, r] : records)
        {
          auto cap = certificates.at(id);
          valid    = valid && std::isfinite(r) && r > 0. && r <= cap.cap.radius;
          cap.cap.radius = r;
          if (cap.root != invalid_id)
            cap.lower_root_mass = r == controls.at(cap.root).radius ?
                                    Mass(0) :
                                    controls.at(cap.root).summary.mass;
          replace_certificate(id, std::move(cap));
        }
    return phase(
      validation,
      valid,
      "Joint generated vessel geometry intersects an active native obstacle.");
  }
  void
  GrowthExecution::discard() noexcept
  {
    if (!staging)
      return;
    rows.discard();
    journal.discard();
    GrowthGeometryChanges restore;
    std::set<Id>          mutable_owned;
    const auto rank = dealii::Utilities::MPI::this_mpi_process(communicator);
    for (Id id : index_touched)
      {
        const auto v = rows.network().vessels.find(id);
        if (v != rows.network().vessels.end() && v->second.owner == rank)
          {
            restore.changed_owned.insert(id);
            if (demand_state.branches.count(id))
              mutable_owned.insert(id);
          }
        else
          restore.released_owned.insert(id);
      }
    spatial.update(rows.network(), mutable_owned, restore);
    for (auto &[id, value] : certificates_before)
      replace_certificate(id, std::move(value), false);
    for (auto &[id, value] : controls_before)
      if (value)
        controls[id] = std::move(*value);
      else
        controls.erase(id);
    if (release)
      release(release_trial);
    certificates_before.clear();
    controls_before.clear();
    root_members_changed.clear();
    index_touched.clear();
    release_old.clear();
    release_trial.clear();
    clearance_bound = old_clearance_bound;
    staging         = false;
  }
  void
  GrowthExecution::adopt()
  {
    rows.adopt();
    journal.adopt();
    for (Id root : root_members_changed)
      if (!root_members.count(root))
        controls.erase(root);
    certificates_before.clear();
    controls_before.clear();
    root_members_changed.clear();
    index_touched.clear();
    staging = false;
  }
} // namespace ImmersX::Vascularization::internal
