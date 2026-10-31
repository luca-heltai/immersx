// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_geometry.h"

#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <set>
namespace ImmersX::Vascularization::internal
{
  BackgroundGeometry::ProfileKey
  BackgroundGeometry::profile_key(const Segment<3> &segment)
  {
    return {{segment.first[0],
             segment.first[1],
             segment.first[2],
             segment.second[0],
             segment.second[1],
             segment.second[2]}};
  }

  void
  BackgroundGeometry::configure_length_scale(const std::string &policy,
                                             MPI_Comm           communicator)
  {
    for (const auto &other : Utilities::MPI::all_gather(communicator, policy))
      AssertThrow(other == policy,
                  ExcMessage("Length scale policies must agree across ranks."));
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(cache.get_triangulation().get_mpi_communicator(),
                     communicator,
                     &comparison);
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(comparison == MPI_IDENT ||
                                                    comparison ==
                                                      MPI_CONGRUENT),
                          communicator) == 1,
      ExcMessage("Geometry and length-scale communicators must be congruent."));
    const bool configure =
      Utilities::MPI::max(static_cast<unsigned int>(!length_scale ||
                                                    cached_policy != policy),
                          communicator) > 0;
    if (configure)
      {
        length_scale =
          mesh_length_scale(cache.get_triangulation(), policy, communicator);
        cached_policy = policy;
      }
  }

  std::map<Id, std::vector<BackgroundInterval>>
  BackgroundGeometry::crossings(const std::map<Id, Segment<3>> &segments,
                                MPI_Comm                        communicator)
  {
    bool                                          valid = true;
    std::map<Id, Segment<3>>                      missing;
    std::map<Id, std::vector<BackgroundInterval>> result;
    for (const auto &[id, segment] : segments)
      {
        const auto   key    = profile_key(segment);
        const double length = segment.first.distance(segment.second);
        valid               = valid && std::isfinite(length) && length > 0.;
        for (const auto coordinate : key)
          valid = valid && std::isfinite(coordinate);
        const auto cached = cached_profiles.find(key);
        if (cached != cached_profiles.end())
          result.emplace(id, cached->second);
        else
          missing.emplace(id, segment);
      }
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid), communicator) == 1,
      ExcMessage("Background profiles require finite positive segments."));
    // The shared exact crossing protocol accepts rank-local opaque keys. Even
    // ranks with only cache hits join the exchange for other geometry writers.
    const auto queried =
      intersect_background(cache, mapping, missing, communicator);
    for (const auto &[id, profile] : queried)
      {
        cached_profiles[profile_key(segments.at(id))] = profile;
        result.emplace(id, profile);
      }
    return result;
  }

  std::map<Id, LocatedCell>
  BackgroundGeometry::locate_points(const std::map<Id, Point<3>> &points,
                                    MPI_Comm                      communicator,
                                    double boundary_tolerance) const
  {
    const auto rank       = Utilities::MPI::this_mpi_process(communicator);
    int        comparison = MPI_UNEQUAL;
    MPI_Comm_compare(cache.get_triangulation().get_mpi_communicator(),
                     communicator,
                     &comparison);
    bool valid = (comparison == MPI_IDENT || comparison == MPI_CONGRUENT) &&
                 std::isfinite(boundary_tolerance) && boundary_tolerance >= 0.;
    for (const auto &[id, point] : points)
      for (unsigned int d = 0; d < 3; ++d)
        valid = valid && std::isfinite(point[d]);
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid), communicator) == 1,
      ExcMessage(
        "Cell queries require finite coordinates and nonnegative tolerances."));
    const auto &covering_tree = routing(communicator).covering_tree;
    using Query               = std::pair<Id, std::pair<Point<3>, double>>;
    std::map<unsigned int, std::vector<Query>> outgoing;
    for (const auto &[id, point] : points)
      {
        Point<3>     lo = point, hi = point;
        const double guard = 64. * std::numeric_limits<double>::epsilon() *
                             std::max(1., point.norm());
        for (unsigned int d = 0; d < 3; ++d)
          {
            lo[d] -= guard;
            hi[d] += guard;
          }
        std::set<unsigned int> recipients;
        for (const auto &[box, owner] :
             covering_tree | bgi::adaptors::queried(bgi::intersects(
                               BoundingBox<3>(std::make_pair(lo, hi)))))
          recipients.insert(owner);
        for (const auto owner : recipients)
          outgoing[owner].push_back({id, {point, boundary_tolerance}});
      }
    const auto incoming = Utilities::MPI::some_to_some(communicator, outgoing);
    std::map<unsigned int, std::map<Id, LocatedCell>> replies;
    for (const auto &[source, queries] : incoming)
      for (const auto &[id, query] : queries)
        if (const auto cell = containing(query.first, query.second))
          {
            LocatedCell found;
            found.id          = (*cell)->id();
            found.diameter    = (*cell)->diameter();
            found.volume      = (*cell)->measure();
            found.center      = (*cell)->center();
            found.point_owner = rank;
            found.cell_owner  = rank;
            for (const auto vertex : (*cell)->vertex_indices())
              found.vertices.push_back((*cell)->vertex(vertex));
            replies[source].emplace(id, std::move(found));
          }
    std::map<Id, LocatedCell> result;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(communicator, replies))
      for (const auto &[id, cell] : entries)
        {
          const auto current = result.find(id);
          const auto owner =
            current == result.end() ?
              cell.point_owner :
              std::min(current->second.point_owner, cell.point_owner);
          if (current == result.end() ||
              cell.diameter < current->second.diameter ||
              (cell.diameter == current->second.diameter &&
               cell.id < current->second.id))
            result[id] = cell;
          result.at(id).point_owner = owner;
        }
    return result;
  }

  GrowthGeometryMeasures
  BackgroundGeometry::measure_segments(const GrowthGeometrySegments &segments,
                                       MPI_Comm communicator)
  {
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(cache.get_triangulation().get_mpi_communicator(),
                     communicator,
                     &comparison);
    bool valid = length_scale.has_value() &&
                 (comparison == MPI_IDENT || comparison == MPI_CONGRUENT);
    if (length_scale)
      try
        {
          length_scale->validate();
        }
      catch (const std::exception &)
        {
          valid = false;
        }
    for (const auto &[id, segment] : segments)
      {
        const double length = segment.first.distance(segment.second);
        valid               = valid && std::isfinite(length) && length > 0.;
        for (double coordinate : profile_key(segment))
          valid = valid && std::isfinite(coordinate);
      }
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid), communicator) == 1,
      ExcMessage(
        "Geometry measures require a configured length policy and finite positive segments."));
    const auto                                   &scale = *length_scale;
    std::map<Id, std::vector<BackgroundInterval>> profiles;
    if (scale.policy == LengthScalePolicy::local_mesh_size)
      profiles = crossings(segments, communicator);
    GrowthGeometryMeasures result;
    for (const auto &[id, segment] : segments)
      {
        GrowthGeometryMeasure measure;
        const double          length = segment.first.distance(segment.second);
        if (scale.policy == LengthScalePolicy::local_mesh_size)
          {
            const auto profile =
              measure_background_profile(length, profiles.at(id), scale);
            measure.normalized_length = profile.normalized_length;
            measure.radius_scale      = measure.clearance_scale =
              profile.maximum_scale;
            measure.complete = profile.complete;
          }
        else if (scale.policy == LengthScalePolicy::meter ||
                 scale.policy == LengthScalePolicy::min_mesh_size ||
                 scale.policy == LengthScalePolicy::max_mesh_size)
          {
            measure.radius_scale = measure.clearance_scale = scale.minimum();
            measure.normalized_length = length / measure.radius_scale;
          }
        valid = valid && std::isfinite(measure.normalized_length) &&
                measure.normalized_length >= 0. &&
                std::isfinite(measure.radius_scale) &&
                measure.radius_scale >= 0. &&
                std::isfinite(measure.clearance_scale) &&
                measure.clearance_scale >= 0.;
        result.emplace(id, measure);
      }
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid), communicator) == 1,
      ExcMessage("Geometry normalization must remain finite and nonnegative."));
    return result;
  }

  void
  BackgroundGeometry::release_profiles(
    const GrowthGeometrySegments &segments) noexcept
  {
    for (const auto &[id, segment] : segments)
      cached_profiles.erase(profile_key(segment));
  }

  void
  BackgroundGeometry::retain_profiles(const NetworkData &snapshot)
  {
    std::set<ProfileKey> retained;
    for (const auto &[id, vessel] : snapshot.vessels)
      retained.insert(
        profile_key({snapshot.nodes.at(vessel.nodes[0]).position,
                     snapshot.nodes.at(vessel.nodes[1]).position}));
    for (auto it = cached_profiles.begin(); it != cached_profiles.end();)
      if (retained.count(it->first))
        ++it;
      else
        it = cached_profiles.erase(it);
  }
} // namespace ImmersX::Vascularization::internal
