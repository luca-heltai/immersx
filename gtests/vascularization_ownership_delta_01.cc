// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/distribution.h>

#include "../source/vascularization/growth_geometry.h"
#include "../source/vascularization/growth_ownership_delta.h"
#include "../source/vascularization/network_serialization.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
using ImmersX::Vascularization::internal::BackgroundGeometry;
using ImmersX::Vascularization::internal::GrowthSplit;
using ImmersX::Vascularization::internal::prepare_growth_ownership;
namespace
{
  constexpr Id base = Id(1) << 54;
  NetworkData
  ownership_seed(double extent, unsigned int extra)
  {
    NetworkData n;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
      return n;
    n.nodes[base + 10] = {{.1, .5, .5}, 0, Port::none, Provenance::imported};
    n.nodes[base + 11] = {{extent - .1, .5, .5},
                          0,
                          Port::none,
                          Provenance::generated};
    Vessel v;
    v.nodes                                       = {{base + 10, base + 11}};
    v.radii                                       = {{.01, .015}};
    v.viscosity                                   = .004;
    v.physical_properties["retained SI property"] = 17.25;
    n.vessels[base]                               = v;
    for (unsigned int i = 0; i < extra; ++i)
      {
        const Id     a = base + 1000 + 2 * i, b = a + 1;
        const double y            = .05 + .1 * i / std::max(1u, extra);
        n.nodes[a].position       = {.2, y, .2};
        n.nodes[b].position       = {.3, y, .2};
        auto unrelated            = v;
        unrelated.nodes           = {{a, b}};
        n.vessels[base + 100 + i] = unrelated;
      }
    return n;
  }
  GrowthSplit
  ownership_split(double extent, unsigned int owner)
  {
    GrowthSplit s;
    s.vessel     = base;
    s.owner      = owner;
    s.old_nodes  = {{base + 10, base + 11}};
    s.old_points = {{Point<3>(.1, .5, .5), Point<3>(extent - .1, .5, .5)}};
    // Physical IDs occupy separate node, vessel and terminal namespaces.
    s.junction          = base;
    s.tip               = base + 1;
    s.continuation      = base + 1;
    s.leaf              = base + 2;
    s.terminal          = base + 2;
    s.territory         = base + 3;
    s.junction_position = {.25 * extent, .5, .5};
    s.tip_position      = {.8 * extent, .7, .5};
    s.demand            = 1.7;
    return s;
  }
  NetworkData
  collect_owned(const NetworkData &n)
  {
    const auto  rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    NetworkData local, result;
    for (const auto &[id, e] : n.nodes)
      if (e.owner == rank)
        local.nodes.emplace(id, e);
    for (const auto &[id, e] : n.vessels)
      if (e.owner == rank)
        local.vessels.emplace(id, e);
    for (const auto &[id, e] : n.terminals)
      if (e.owner == rank)
        local.terminals.emplace(id, e);
    for (const auto &part : Utilities::MPI::all_gather(MPI_COMM_WORLD, local))
      {
        result.nodes.insert(part.nodes.begin(), part.nodes.end());
        result.vessels.insert(part.vessels.begin(), part.vessels.end());
        result.terminals.insert(part.terminals.begin(), part.terminals.end());
      }
    return result;
  }
} // namespace
TEST(VascularForest, BOTH_SparseOwnershipMatchesNativeFullRedistribution)
{
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  for (unsigned int cells : {1u, 4u})
    for (unsigned int unrelated : {0u, 64u})
      {
        parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
        GridGenerator::subdivided_hyper_rectangle(bg,
                                                  {cells, 1, 1},
                                                  Point<3>(),
                                                  Point<3>(cells, 1., 1.));
        MappingQ1<3> mapping;
        const auto   accepted = redistribute(
          bg, mapping, ownership_seed(cells, unrelated), 0., MPI_COMM_WORLD);
        const auto original = collect_owned(accepted);
        const auto split =
          ownership_split(cells, original.vessels.at(base).owner);
        std::vector<GrowthSplit> local;
        if (rank == size - 1)
          local.push_back(split);
        const auto         bytes = Utilities::pack(accepted, false);
        BackgroundGeometry geometry(bg, mapping);
        GrowthWork         work;
        const auto         delta = prepare_growth_ownership(
          accepted, local, geometry, MPI_COMM_WORLD, &work);
        ASSERT_TRUE(delta.validation.valid());
        EXPECT_EQ(bytes, Utilities::pack(accepted, false));
        EXPECT_EQ(delta.global_splits.size(), 1u);
        EXPECT_EQ(delta.written_splits.size(), rank == split.owner ? 1u : 0u);
        EXPECT_EQ(delta.node_owners.size(), 2u);
        EXPECT_EQ(delta.vessel_owners.size(), 3u);
        auto full    = accepted;
        bool applied = true;
        if (rank == split.owner)
          applied =
            ImmersX::Vascularization::internal::apply_growth_split(full, split);
        ASSERT_EQ(Utilities::MPI::min(static_cast<unsigned int>(applied),
                                      MPI_COMM_WORLD),
                  1u);
        full = redistribute(bg, mapping, full, 0., MPI_COMM_WORLD);
        const auto expected = collect_owned(full);
        bool       complete = expected.terminals.count(split.terminal);
        for (Id id : {split.junction, split.tip})
          complete =
            expected.nodes.count(id) && delta.node_owners.count(id) && complete;
        for (Id id : {split.vessel, split.continuation, split.leaf})
          complete = expected.vessels.count(id) &&
                     delta.vessel_owners.count(id) && complete;
        ASSERT_EQ(Utilities::MPI::min(static_cast<unsigned int>(complete),
                                      MPI_COMM_WORLD),
                  1u);
        for (Id id : {split.junction, split.tip})
          EXPECT_EQ(delta.node_owners.at(id), expected.nodes.at(id).owner);
        for (Id id : {split.vessel, split.continuation, split.leaf})
          {
            const auto &v = expected.vessels.at(id);
            EXPECT_EQ(delta.vessel_owners.at(id), v.owner);
            EXPECT_EQ(delta.changed_owned.count(id), v.owner == rank ? 1u : 0u);
            EXPECT_EQ(delta.owned.vessels.count(id), v.owner == rank ? 1u : 0u);
            if (delta.owned.vessels.count(id))
              EXPECT_EQ(Utilities::pack(delta.owned.vessels.at(id), false),
                        Utilities::pack(v, false));
          }
        EXPECT_EQ(delta.released_owned.count(split.vessel),
                  rank == split.owner &&
                      expected.vessels.at(base).owner != rank ?
                    1u :
                    0u);
        for (const auto &[id, n] : delta.owned.nodes)
          {
            const auto found = expected.nodes.find(id);
            EXPECT_NE(found, expected.nodes.end());
            if (found != expected.nodes.end())
              EXPECT_EQ(Utilities::pack(n, false),
                        Utilities::pack(found->second, false));
          }
        const auto &terminal = expected.terminals.at(split.terminal);
        EXPECT_EQ(delta.owned.terminals.count(split.terminal),
                  terminal.owner == rank ? 1u : 0u);
        if (delta.owned.terminals.count(split.terminal))
          EXPECT_EQ(Utilities::pack(delta.owned.terminals.at(split.terminal),
                                    false),
                    Utilities::pack(terminal, false));
        EXPECT_EQ(Utilities::MPI::sum(work[GrowthOperation::node_localizations],
                                      MPI_COMM_WORLD),
                  2u);
        EXPECT_EQ(Utilities::MPI::sum(
                    work[GrowthOperation::vessel_midpoint_localizations],
                    MPI_COMM_WORLD),
                  3u);
        EXPECT_EQ(work[GrowthOperation::background_cache_constructions], 0u);
      }
}
TEST(VascularForest, BOTH_SparseOwnershipRejectsStaleAndRemoteIdConflicts)
{
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  MappingQ1<3> mapping;
  const auto   accepted =
    redistribute(bg, mapping, ownership_seed(1., 0), 0., MPI_COMM_WORLD);
  const auto original = collect_owned(accepted);
  const auto good     = ownership_split(1., original.vessels.at(base).owner);
  BackgroundGeometry geometry(bg, mapping);
  for (unsigned int kind : {0u, 1u, 2u, 3u})
    {
      auto                     input = accepted;
      auto                     split = good;
      std::vector<GrowthSplit> local;
      if (kind == 0)
        split.old_points[1][0] -= .01;
      else if (kind == 1 && rank == size - 1)
        input.nodes[split.tip] = {{-.5, .5, .5},
                                  rank,
                                  Port::none,
                                  Provenance::imported};
      else if (kind == 2)
        split.owner = size;
      if (rank == size - 1)
        {
          local.push_back(split);
          if (kind == 3)
            local.push_back(split);
        }
      const auto bytes = Utilities::pack(input, false);
      const auto failed =
        prepare_growth_ownership(input, local, geometry, MPI_COMM_WORLD);
      EXPECT_FALSE(failed.validation.valid());
      EXPECT_TRUE(failed.owned.vessels.empty());
      EXPECT_TRUE(failed.written.vessels.empty());
      EXPECT_EQ(Utilities::pack(input, false), bytes);
    }
  std::vector<GrowthSplit> local;
  if (rank == size - 1)
    local.push_back(good);
  const auto retry =
    prepare_growth_ownership(accepted, local, geometry, MPI_COMM_WORLD);
  EXPECT_TRUE(retry.validation.valid());
}
TEST(VascularForest, BOTH_SparseOwnershipRoutesConcurrentAdjacentSplits)
{
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 1, 1},
                                            Point<3>(),
                                            Point<3>(4., 1., 1.));
  MappingQ1<3> mapping;
  auto         source = ownership_seed(4., 0);
  if (rank == 0)
    {
      source.nodes[base + 12]          = {{2., .5, .5},
                                          0,
                                          Port::none,
                                          Provenance::generated};
      auto child                       = source.vessels.at(base);
      source.vessels.at(base).nodes[1] = base + 12;
      child.nodes[0]                   = base + 12;
      source.vessels.emplace(base + 4, child);
    }
  const auto accepted = redistribute(bg, mapping, source, 0., MPI_COMM_WORLD);
  const auto original = collect_owned(accepted);
  auto       first    = ownership_split(4., original.vessels.at(base).owner);
  first.old_nodes[1]  = base + 12;
  first.old_points[1] = {2., .5, .5};
  first.junction_position = {.75, .5, .5};
  first.tip_position      = {1.25, .75, .5};
  auto second   = ownership_split(4., original.vessels.at(base + 4).owner);
  second.vessel = base + 4;
  second.old_nodes[0]      = base + 12;
  second.old_points[0]     = {2., .5, .5};
  second.junction          = base + 20;
  second.tip               = base + 21;
  second.continuation      = base + 22;
  second.leaf              = base + 23;
  second.terminal          = base + 24;
  second.territory         = base + 25;
  second.junction_position = {2.75, .5, .5};
  second.tip_position      = {3.25, .75, .5};
  std::vector<GrowthSplit> local;
  if (rank == 0)
    local.push_back(first);
  if (rank == size - 1)
    local.push_back(second);
  const auto         bytes = Utilities::pack(accepted, false);
  BackgroundGeometry geometry(bg, mapping);
  GrowthWork         work;
  const auto         delta =
    prepare_growth_ownership(accepted, local, geometry, MPI_COMM_WORLD, &work);
  ASSERT_TRUE(delta.validation.valid());
  EXPECT_EQ(Utilities::pack(accepted, false), bytes);
  EXPECT_EQ(delta.global_splits.size(), 2u);
  EXPECT_EQ(delta.node_owners.size(), 4u);
  EXPECT_EQ(delta.vessel_owners.size(), 6u);
  auto full    = accepted;
  bool applied = true;
  for (const auto &s : {first, second})
    if (rank == s.owner)
      applied =
        ImmersX::Vascularization::internal::apply_growth_split(full, s) &&
        applied;
  ASSERT_EQ(Utilities::MPI::min(static_cast<unsigned int>(applied),
                                MPI_COMM_WORLD),
            1u);
  full                = redistribute(bg, mapping, full, 0., MPI_COMM_WORLD);
  const auto expected = collect_owned(full);
  for (const auto &[id, v] : delta.owned.vessels)
    {
      const auto found = expected.vessels.find(id);
      EXPECT_NE(found, expected.vessels.end());
      if (found != expected.vessels.end())
        EXPECT_EQ(Utilities::pack(v, false),
                  Utilities::pack(found->second, false));
    }
  EXPECT_EQ(Utilities::MPI::sum(delta.owned.vessels.size(), MPI_COMM_WORLD),
            6u);
  EXPECT_EQ(Utilities::MPI::sum(work[GrowthOperation::node_localizations],
                                MPI_COMM_WORLD),
            4u);
  EXPECT_EQ(
    Utilities::MPI::sum(work[GrowthOperation::vessel_midpoint_localizations],
                        MPI_COMM_WORLD),
    6u);
}
