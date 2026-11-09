// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_wall_certificates.h"

#include <deal.II/base/scope_exit.h>

#include <deal.II/grid/grid_tools.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/generation.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>
namespace ImmersX::Vascularization::internal
{
  namespace
  {
    struct WallVote
    {
      bool                valid = true;
      std::array<bool, 2> planes{{false, false}};
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &valid &planes;
      }
    };
    dealii::BoundingBox<3>
    wall_envelope(const GrowthWallRequest &request)
    {
      const auto  &e = request.edge;
      const double pad =
        e.radius + request.padding +
        inverse_mapping_tolerance *
          std::max({1., e.points[0].norm(), e.points[1].norm()});
      dealii::Point<3> lo, hi;
      for (unsigned int d = 0; d < 3; ++d)
        {
          lo[d] = std::min(e.points[0][d], e.points[1][d]) - pad;
          hi[d] = std::max(e.points[0][d], e.points[1][d]) + pad;
        }
      return dealii::BoundingBox<3>(std::make_pair(lo, hi));
    }
  } // namespace
  GrowthWallResult
  certify_growth_walls(BackgroundGeometry                    &geometry,
                       const std::map<Id, GrowthWallRequest> &local,
                       unsigned int                           geometry_depth,
                       unsigned int                           geometry_work,
                       MPI_Comm                               comm,
                       GrowthWork                            *work)
  {
    using namespace dealii;
    GrowthWallResult result;
    const auto       previous_counts = geometry.counts;
    ScopeExit        restore_counts(
      [&geometry, previous_counts]() { geometry.counts = previous_counts; });
    if (work)
      geometry.counts = work;
    auto *counts     = geometry.counts;
    int   comparison = MPI_UNEQUAL;
    MPI_Comm_compare(geometry.cache.get_triangulation().get_mpi_communicator(),
                     comm,
                     &comparison);
    bool valid = (comparison == MPI_IDENT || comparison == MPI_CONGRUENT) &&
                 geometry_depth <= 30 && geometry_work > 0;
    count_work(counts, GrowthOperation::validation_record_visits, local.size());
    std::uint64_t query_count = 0;
    for (const auto &[id, request] : local)
      {
        result.local.emplace(id, false);
        const auto &e   = request.edge;
        bool admissible = e.id != invalid_id && e.nodes[0] != invalid_id &&
                          e.nodes[1] != invalid_id &&
                          e.nodes[0] != e.nodes[1] && std::isfinite(e.radius) &&
                          e.radius > 0. && std::isfinite(request.padding) &&
                          request.padding >= 0.;
        for (unsigned int end = 0; end < 2; ++end)
          for (unsigned int d = 0; d < 3; ++d)
            admissible = std::isfinite(e.points[end][d]) &&
                         std::isfinite(request.port_normals[end][d]) &&
                         admissible;
        if (admissible)
          {
            const double length = e.points[0].distance(e.points[1]);
            admissible          = std::isfinite(length) && length > 0.;
            const auto box      = wall_envelope(request);
            for (const auto &p : {box.get_boundary_points().first,
                                  box.get_boundary_points().second})
              for (unsigned int d = 0; d < 3; ++d)
                admissible = std::isfinite(p[d]) && admissible;
          }
        if (request.check_domain)
          query_count += 2;
        valid = valid && admissible;
      }
    valid = valid && query_count <= std::numeric_limits<unsigned int>::max();
    const std::array<std::uint64_t, 3> controls{
      {geometry_depth, geometry_work, query_count}};
    bool queries = false;
    for (const auto &other : Utilities::MPI::all_gather(comm, controls))
      {
        valid   = valid && other[0] == controls[0] && other[1] == controls[1];
        queries = queries || other[2] > 0;
      }
    if (Utilities::MPI::min(static_cast<unsigned int>(valid), comm) != 1)
      {
        result.validation.issues.push_back(
          {"growth wall",
           invalid_id,
           "Wall requests require finite geometry, nonnegative physical padding, matching geometry budgets and congruent communicators."});
        return result;
      }
    const auto &routing = geometry.routing(comm, counts);
    if (routing.covering_tree.empty())
      {
        result.validation.issues.push_back(
          {"growth wall",
           invalid_id,
           "Wall certification requires a nonempty global background."});
        return result;
      }
    std::vector<Point<3>>                                   positions;
    std::vector<bool>                                       port_points;
    std::map<Id, std::size_t>                               ranges;
    std::map<Id, std::array<bool, 2>>                       planes;
    std::map<unsigned int, std::map<Id, GrowthWallRequest>> send;
    for (const auto &[id, request] : local)
      {
        result.local[id] = true;
        count_work(counts, GrowthOperation::certified_vessels);
        std::array<bool, 2> ports;
        for (unsigned int end = 0; end < 2; ++end)
          ports[end] =
            normalized_port_normal(request.port_normals[end]).norm_square() >
            0.;
        planes[id] = {{!ports[0], !ports[1]}};
        if (request.check_domain)
          {
            ranges.emplace(id, positions.size());
            for (unsigned int end = 0; end < 2; ++end)
              {
                positions.push_back(request.edge.points[end]);
                port_points.push_back(ports[end]);
              }
          }
        std::set<unsigned int> recipients;
        for (const auto &[box, rank] :
             routing.covering_tree |
               bgi::adaptors::queried(bgi::intersects(wall_envelope(request))))
          recipients.insert(rank);
        for (auto rank : recipients)
          {
            send[rank].emplace(id, request);
            count_work(counts, GrowthOperation::metadata_copy_attempts);
          }
      }
    if (queries)
      {
        count_work(counts,
                   GrowthOperation::point_cell_queries,
                   positions.size());
        const auto matches =
          GridTools::distributed_compute_point_locations(geometry.cache,
                                                         positions,
                                                         routing.boxes);
        // Native unit points can be clamped. Recompute from the original
        // physical points, retaining both strict and declared-port tolerances.
        std::map<unsigned int,
                 std::vector<std::pair<std::size_t, unsigned char>>>
          strict_found;
        for (std::size_t c = 0; c < std::get<0>(matches).size(); ++c)
          for (std::size_t i = 0; i < std::get<2>(matches)[c].size(); ++i)
            {
              count_work(counts, GrowthOperation::point_cell_hits);
              const auto &cell = std::get<0>(matches)[c];
              try
                {
                  const auto unit =
                    geometry.mapping.transform_real_to_unit_cell(
                      cell, std::get<3>(matches)[c][i]);
                  const bool strict = reference_contains(cell, unit, 0.);
                  const bool port =
                    strict || reference_contains(cell, unit, 1.e-12);
                  const unsigned char mask = (strict ? 1 : 0) | (port ? 2 : 0);
                  if (mask)
                    strict_found[std::get<4>(matches)[c][i]].emplace_back(
                      std::get<2>(matches)[c][i], mask);
                }
              catch (const Mapping<3>::ExcTransformationFailed &)
                {}
            }
        std::vector<bool> found(positions.size(), false);
        for (const auto &[owner, indices] :
             Utilities::MPI::some_to_some(comm, strict_found))
          for (const auto &[index, mask] : indices)
            found.at(index) =
              found.at(index) || (mask & (port_points.at(index) ? 2 : 1));
        for (const auto &[id, begin] : ranges)
          result.local[id] = found[begin] && found[begin + 1];
      }
    const auto incoming = Utilities::MPI::some_to_some(comm, send);
    std::map<unsigned int, std::map<Id, WallVote>> replies;
    for (const auto &[source, requests] : incoming)
      for (const auto &[id, request] : requests)
        {
          auto &vote = replies[source][id];
          for (unsigned int end = 0; end < 2; ++end)
            if (normalized_port_normal(request.port_normals[end])
                  .norm_square() > 0.)
              vote.planes[end] =
                geometry.planar_section(request.edge.points[end],
                                        normalized_port_normal(
                                          request.port_normals[end]));
          vote.valid =
            geometry.boundary_separated({request.edge.points[0],
                                         request.edge.points[1]},
                                        request.edge.radius + request.padding,
                                        geometry_depth,
                                        geometry_work,
                                        request.port_normals);
        }
    for (const auto &[source, votes] :
         Utilities::MPI::some_to_some(comm, replies))
      for (const auto &[id, vote] : votes)
        {
          result.local.at(id) = result.local.at(id) && vote.valid;
          for (unsigned int end = 0; end < 2; ++end)
            planes.at(id)[end] = planes.at(id)[end] || vote.planes[end];
        }
    for (const auto &[id, plane] : planes)
      result.local[id] = result.local[id] && plane[0] && plane[1];
    return result;
  }
  GrowthWallResult
  certify_growth_walls(BackgroundGeometry                    &geometry,
                       const std::map<Id, GrowthWallRequest> &local,
                       const Parameters                      &parameters,
                       MPI_Comm                               comm,
                       GrowthWork                            *work)
  {
    return certify_growth_walls(geometry,
                                local,
                                parameters.geometry_depth,
                                parameters.geometry_work,
                                comm,
                                work);
  }
} // namespace ImmersX::Vascularization::internal
