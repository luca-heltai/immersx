// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <boost/serialization/map.hpp>

#include <gtest/gtest.h>
#include <immersx/vascularization/distribution.h>
#include <immersx/vascularization/generation.h>

#include <array>
#include <limits>
#include <set>

#include "../source/vascularization/growth_batch.h"
#include "../source/vascularization/growth_wall_certificates.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
using ImmersX::Vascularization::internal::BackgroundGeometry;
using ImmersX::Vascularization::internal::BatchProposal;
using ImmersX::Vascularization::internal::certify_growth_walls;
using ImmersX::Vascularization::internal::GrowthWallRequest;
namespace
{
  GrowthWallRequest
  wall_request(Id       id,
               Point<3> a,
               Point<3> b,
               double   radius,
               double   padding = 0.)
  {
    GrowthWallRequest r;
    r.edge    = {id, {{2 * id, 2 * id + 1}}, {{a, b}}, radius};
    r.padding = padding;
    return r;
  }
  void
  compare_wall_batches(parallel::distributed::Triangulation<3> &bg,
                       const std::map<Id, GrowthWallRequest>   &cases,
                       const std::map<Id, bool>                &expected)
  {
    const auto   rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto   size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    MappingQ1<3> mapping;
    NetworkData  initial;
    if (rank == 0)
      for (const auto &[id, r] : cases)
        for (unsigned int end = 0; end < 2; ++end)
          if (ImmersX::Vascularization::internal::normalized_port_normal(
                r.port_normals[end])
                .norm_square() > 0.)
            initial.nodes[r.edge.nodes[end]] = {r.edge.points[end],
                                                rank,
                                                end == 0 ?
                                                  Port::arterial_inlet :
                                                  Port::arterial_outlet,
                                                Provenance::imported,
                                                r.port_normals[end]};
    const auto network = redistribute(bg, mapping, initial, 0., MPI_COMM_WORLD);
    std::map<Id, unsigned int> local_owners, owners;
    for (const auto &[id, n] : network.nodes)
      if (n.owner == rank)
        local_owners.emplace(id, rank);
    for (const auto &part :
         Utilities::MPI::all_gather(MPI_COMM_WORLD, local_owners))
      owners.insert(part.begin(), part.end());
    GrowthWork         original, work;
    BackgroundGeometry geometry(bg, mapping, &original);
    const auto         saved = original.counts;
    Parameters         p("/Sparse wall checks/");
    p.clearance = 10.;
    std::map<Id, GrowthWallRequest> local;
    if (rank == size - 1)
      local = cases;
    const auto result =
      certify_growth_walls(geometry, local, p, MPI_COMM_WORLD, &work);
    ASSERT_TRUE(result.validation.valid());
    EXPECT_EQ(geometry.counts, &original);
    EXPECT_EQ(original.counts, saved);
    EXPECT_EQ(result.local.size(), local.size());
    if (rank == size - 1)
      for (const auto &[id, wanted] : expected)
        {
          EXPECT_EQ(result.local.at(id), wanted) << id;
        }
    EXPECT_EQ(Utilities::MPI::sum(work[GrowthOperation::certified_vessels],
                                  MPI_COMM_WORLD),
              cases.size());
    EXPECT_EQ(work[GrowthOperation::background_cache_constructions], 0u);
    EXPECT_EQ(work[GrowthOperation::network_index_builds], 0u);
    EXPECT_EQ(work[GrowthOperation::pair_checks], 0u);
    Parameters oracle("/Full wall oracle/");
    oracle.minimum_length = 0.;
    oracle.minimum_radius = 0.;
    oracle.geometry_depth = p.geometry_depth;
    oracle.geometry_work  = p.geometry_work;
    for (const auto &[id, r] : cases)
      {
        oracle.clearance = r.padding;
        std::vector<BatchProposal> proposals;
        if (rank == size - 1)
          {
            BatchProposal proposal;
            proposal.id = id;
            proposal.changed.push_back(r.edge);
            for (unsigned int end = 0; end < 2; ++end)
              if (ImmersX::Vascularization::internal::normalized_port_normal(
                    r.port_normals[end])
                    .norm_square() > 0.)
                {
                  proposal.normals[r.edge.nodes[end]] = r.port_normals[end];
                  proposal.port_owners[r.edge.nodes[end]] =
                    owners.at(r.edge.nodes[end]);
                }
            proposals.push_back(proposal);
          }
        const auto full =
          ImmersX::Vascularization::internal::select_growth_batch(
            geometry, network, proposals, oracle, MPI_COMM_WORLD, [](Id) {
              return 0.;
            });
        if (rank == size - 1)
          {
            EXPECT_EQ(result.local.at(id), full.local.at(id)) << id;
          }
      }
  }
} // namespace
TEST(VascularGeneration, BOTH_WallCertificatesMatchNativeBatchWithoutPairWork)
{
  for (unsigned int refinement : {0u, 1u})
    {
      parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
      GridGenerator::hyper_cube(bg);
      bg.refine_global(refinement);
      std::map<Id, GrowthWallRequest> cases;
      cases[1] = wall_request(101, {.2, .5, .5}, {.8, .5, .5}, .01);
      cases[2] = wall_request(102, {.2, .04, .5}, {.8, .04, .5}, .01, .02);
      cases[3] = wall_request(103, {.2, .04, .5}, {.8, .04, .5}, .01, .04);
      cases[4] =
        wall_request(104, {.2, -1.e-13, .5}, {.8, -1.e-13, .5}, 1.e-15);
      cases[5] = wall_request(105, {0., .5, .5}, {.4, .5, .5}, .02, .01);
      cases[5].port_normals[0] = {1., 0., 0.};
      cases[6] = wall_request(106, {0., .5, .5}, {.4, .5, .5}, .02);
      cases[7] = wall_request(107, {0., .5, .5}, {.4, .5, .5}, .02);
      cases[7].port_normals[0] = {-1., 0., 0.};
      cases[8] = wall_request(108, {0., .5, .5}, {.01, .5, .5}, .02);
      cases[8].port_normals[0] = {1., 0., 0.};
      cases[9]  = wall_request(109, {.2, .5, .5}, {.8, .5, .5}, .6);
      cases[10] = wall_request(110, {1.2, .5, .5}, {1.8, .5, .5}, .01);
      cases[11] = wall_request(111, {0., .5, .5}, {1., .5, .5}, .02);
      cases[11].port_normals = {{Point<3>(1., 0., 0.), Point<3>(-1., 0., 0.)}};
      cases[12] = wall_request(112, {0., .5, .5}, {.4, .5, .5}, .02);
      cases[12].port_normals[0] = {1.e-250, 0., 0.};
      compare_wall_batches(bg,
                           cases,
                           {{1, true},
                            {2, true},
                            {3, false},
                            {4, false},
                            {5, true},
                            {6, false},
                            {7, false},
                            {8, false},
                            {9, false},
                            {10, false},
                            {11, true},
                            {12, true}});
    }
}
TEST(VascularGeneration,
     BOTH_WallCertificatesRejectWarpedPortsAndNonconvexCrossings)
{
  {
    parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
    GridGenerator::hyper_cube(bg);
    GridTools::transform(
      [](const Point<3> &p) {
        return Point<3>(p[0] + .2 * p[1] * p[2], p[1], p[2]);
      },
      bg);
    auto port            = wall_request(201, {.05, .5, .5}, {.55, .5, .5}, .01);
    port.port_normals[0] = {1., 0., 0.};
    compare_wall_batches(
      bg,
      {{1, port}, {2, wall_request(202, {.2, .5, .5}, {.8, .5, .5}, .01)}},
      {{1, false}, {2, true}});
  }
  {
    Triangulation<3>                        full;
    parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
    GridGenerator::subdivided_hyper_rectangle(full,
                                              {3, 3, 1},
                                              Point<3>(),
                                              Point<3>(3., 3., 1.));
    std::set<Triangulation<3>::active_cell_iterator> removed;
    for (const auto &cell : full.active_cell_iterators())
      if (cell->center()[0] > 1. && cell->center()[0] < 2. &&
          cell->center()[1] > 1.)
        removed.insert(cell);
    GridGenerator::create_triangulation_with_removed_cells(full, removed, bg);
    compare_wall_batches(
      bg,
      {{1, wall_request(301, {.5, 2.5, .5}, {2.5, 2.5, .5}, .01)},
       {2, wall_request(302, {.2, 2.5, .5}, {.8, 2.5, .5}, .01)}},
      {{1, false}, {2, true}});
  }
}
TEST(VascularGeneration,
     BOTH_WallCertificatesReuseDomainAndRejectMalformedRequestsCollectively)
{
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  MappingQ1<3>                    mapping;
  GrowthWork                      previous, work;
  BackgroundGeometry              geometry(bg, mapping, &previous);
  Parameters                      p("/Wall certificate reuse/");
  std::map<Id, GrowthWallRequest> local;
  if (rank == size - 1)
    local.emplace(7, wall_request(401, {.2, .5, .5}, {.8, .5, .5}, .01));
  const auto initial =
    certify_growth_walls(geometry, local, p, MPI_COMM_WORLD, &work);
  ASSERT_TRUE(initial.validation.valid());
  const bool certified =
    rank != size - 1 || (initial.local.count(7) && initial.local.at(7));
  ASSERT_EQ(Utilities::MPI::min(static_cast<unsigned int>(certified),
                                MPI_COMM_WORLD),
            1u);
  if (rank == size - 1)
    {
      local.at(7).check_domain = false;
      local.emplace(8, local.at(7));
      local.at(8).edge.radius = .6;
    }
  work = {};
  const auto cached =
    certify_growth_walls(geometry, local, p, MPI_COMM_WORLD, &work);
  ASSERT_TRUE(cached.validation.valid());
  if (rank == size - 1)
    {
      EXPECT_TRUE(cached.local.at(7));
      EXPECT_FALSE(cached.local.at(8));
    }
  EXPECT_EQ(work[GrowthOperation::point_cell_queries], 0u);
  EXPECT_EQ(geometry.counts, &previous);
  for (unsigned int kind : {0u, 1u})
    {
      if (kind == 0 && rank == size - 1)
        local.at(7).padding = -1.;
      else if (kind == 1 && rank == size - 1)
        local.at(7).padding = std::numeric_limits<double>::infinity();
      const auto malformed =
        certify_growth_walls(geometry, local, p, MPI_COMM_WORLD, &work);
      EXPECT_FALSE(malformed.validation.valid());
      EXPECT_EQ(geometry.counts, &previous);
    }
  const auto empty =
    certify_growth_walls(geometry, {}, p, MPI_COMM_WORLD, &work);
  EXPECT_TRUE(empty.validation.valid());
  EXPECT_TRUE(empty.local.empty());
}
