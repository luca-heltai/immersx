// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/distribution.h>

#include "../source/vascularization/growth_start.h"
#include "../source/vascularization/growth_territory_search.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
using namespace ImmersX::Vascularization::internal;
namespace
{
  std::optional<Id>
  nearest(const std::map<Id, Territory> &available,
          const std::set<Id>            &excluded,
          const Point<3>                &position)
  {
    std::optional<std::pair<double, Id>> best;
    for (const auto &[id, point] : available)
      if (!excluded.count(id))
        {
          const auto key =
            std::make_pair(position.distance_square(point.position), id);
          if (!best || key < *best)
            best = key;
        }
    return best ? std::optional<Id>(best->second) : std::nullopt;
  }
} // namespace
TEST(VascularForest,
     BOTH_VenousNearestCursorMatchesExhaustiveSearchWithClaimsAndExclusions)
{
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const Id   base = (Id(1) << 60) + rank * 10000;
  for (unsigned int count : {32u, 128u, 512u})
    {
      std::vector<Territory>  points;
      std::map<Id, Territory> available;
      for (unsigned int i = 0; i < count; ++i)
        {
          Point<3> point(.05 * (i % 16), .03 * (i / 16), .2);
          if (i < 8)
            point = Point<3>(.2, .2, .2);
          if (i == 8)
            point = Point<3>(std::nextafter(.2, 1.), .2, .2);
          Territory territory{base + count - i, point, rank, 1. + i};
          points.push_back(territory);
          available.emplace(territory.id, territory);
        }
      GrowthWork            work;
      VenousTerritorySearch search(points, rank, &work);
      ASSERT_TRUE(search.validation().valid());
      ASSERT_TRUE(search.cover());
      EXPECT_EQ(search.size(), count);
      const Point<3>              origins[] = {Point<3>(.2, .2, .2),
                                               Point<3>(.13, .27, .11)};
      std::array<std::set<Id>, 2> excluded;
      std::array<Id, 2>           rejected{{invalid_id, invalid_id}};
      for (unsigned int iteration = 0; iteration <= count; ++iteration)
        for (unsigned int root = 0; root < 2; ++root)
          {
            const auto expected =
              nearest(available, excluded[root], origins[root]);
            const auto actual =
              search.query(base + 1000 + root, origins[root], rejected[root]);
            rejected[root] = invalid_id;
            ASSERT_EQ(bool(actual), bool(expected));
            if (!expected)
              continue;
            ASSERT_EQ(actual->id, *expected)
              << count << " " << iteration << " " << root;
            EXPECT_EQ(actual->owner, rank);
            EXPECT_DOUBLE_EQ(actual->demand, available.at(*expected).demand);
            const auto visits = work[GrowthOperation::network_index_hits];
            const auto repeated =
              search.query(base + 1000 + root, origins[root]);
            ASSERT_TRUE(repeated);
            EXPECT_EQ(repeated->id, *expected);
            EXPECT_EQ(work[GrowthOperation::network_index_hits], visits);
            if (root == 0 && iteration % 7 == 3)
              {
                EXPECT_TRUE(search.claim(*expected));
                available.erase(*expected);
              }
            else
              {
                excluded[root].insert(*expected);
                rejected[root] = *expected;
              }
          }
      EXPECT_LE(work[GrowthOperation::network_index_hits], 2 * count);
      EXPECT_EQ(work[GrowthOperation::network_index_queries], 2u);
      EXPECT_EQ(work[GrowthOperation::network_index_builds], 1u);
      EXPECT_EQ(work[GrowthOperation::network_index_removals], 0u);
      EXPECT_EQ(search.size(), available.size());
      EXPECT_EQ(search.active_queries(), 2u);
      search.forget(base + 1000);
      search.forget(base + 1001);
      EXPECT_EQ(search.active_queries(), 0u);
      const auto expected = nearest(available, {}, origins[0]);
      const auto fresh    = search.query(base + 2000, origins[0]);
      ASSERT_EQ(bool(fresh), bool(expected));
      if (fresh)
        {
          EXPECT_EQ(fresh->id, *expected);
        }
    }
}
TEST(VascularForest, BOTH_VenousNearestCursorValidatesLifetimeAndEmptyInput)
{
  const auto            rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  VenousTerritorySearch empty({}, rank);
  EXPECT_TRUE(empty.validation().valid());
  EXPECT_FALSE(empty.cover());
  EXPECT_FALSE(empty.query(1, Point<3>()));
  EXPECT_EQ(empty.active_queries(), 1u);
  empty.forget(1);
  EXPECT_EQ(empty.active_queries(), 0u);
  Territory point{5, Point<3>(.2, .3, .4), rank, 2.};
  for (unsigned int failure = 0; failure < 5; ++failure)
    {
      auto bad = point;
      if (failure == 0)
        bad.id = invalid_id;
      else if (failure == 1)
        bad.position[0] = std::numeric_limits<double>::infinity();
      else if (failure == 2)
        bad.demand = 0.;
      else if (failure == 3)
        bad.demand = std::numeric_limits<double>::quiet_NaN();
      else if (failure == 4)
        bad.owner = rank + 1;
      VenousTerritorySearch invalid({bad}, rank);
      EXPECT_FALSE(invalid.validation().valid());
      EXPECT_THROW(invalid.query(1, Point<3>()), ExceptionBase);
    }
  VenousTerritorySearch duplicate({point, point}, rank);
  EXPECT_FALSE(duplicate.validation().valid());
  VenousTerritorySearch search({point}, rank);
  ASSERT_TRUE(search.query(1, Point<3>()));
  EXPECT_THROW(search.query(1, Point<3>(1., 0., 0.)), ExceptionBase);
  EXPECT_THROW(search.query(invalid_id, Point<3>()), ExceptionBase);
  EXPECT_TRUE(search.claim(5));
  EXPECT_FALSE(search.claim(5));
  EXPECT_FALSE(search.query(1, Point<3>()));
}

TEST(VascularAV,
     BOTH_VenousStartupChoosesGlobalNearestTerritoryAcrossNativeOwners)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  const Id   base = Id(1) << 60;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 1, 1},
                                            Point<3>(),
                                            Point<3>(1., 1., 1.));
  MappingQ1<3>   mapping;
  NetworkData    input;
  const Point<3> points[] = {Point<3>(.2, .25, .5),
                             Point<3>(.8, .6, .5),
                             Point<3>(.51, .75, .5)};
  if (rank == size - 1)
    for (unsigned int i = 0; i < 3; ++i)
      {
        const Id node         = base + 10 * i;
        input.nodes[node]     = {Point<3>(points[i][0], points[i][1], .1),
                                 rank,
                                 Port::arterial_inlet};
        input.nodes[node + 1] = {points[i], rank};
        Vessel vessel;
        vessel.nodes     = {{node, node + 1}};
        vessel.radii     = {{2.e-5, 2.e-5}};
        vessel.viscosity = .004;
        vessel.owner     = rank;
        vessel.system    = System::arterial;
        input.vessels.emplace(base + 100 + i, vessel);
        input.terminals.emplace(
          base + 200 + i,
          Terminal{node + 1, base + 500 + i, System::arterial, rank, 1. + i});
      }
  const auto             original = redistribute(bg, mapping, input, 0., comm);
  std::vector<Territory> owned;
  for (const auto &[id, t] : original.terminals)
    if (t.owner == rank)
      owned.push_back(
        {t.territory, original.nodes.at(t.node).position, rank, t.demand});
  Parameters p;
  p.minimum_radius             = 0.;
  p.minimum_length             = 0.;
  p.minimum_angle              = 1.;
  p.seed_length                = .002;
  p.viscosity                  = .004;
  p.venous.seed_position       = Point<3>(.49, .75, .5);
  p.venous.seed_radius         = 5.e-5;
  p.venous.terminal_separation = .005;
  const auto reserved = reserve_venous_outlet(bg, mapping, original, p, comm);
  ASSERT_TRUE(reserved.validation.valid());
  const auto actual = start_venous(
    bg, mapping, reserved.network, original, p, owned, reserved.port, comm);
  ASSERT_TRUE(actual.accepted.validation.valid());
  unsigned int terminals = 0;
  for (const auto &[id, t] : actual.accepted.network.terminals)
    if (t.owner == rank && t.system == System::venous)
      {
        ++terminals;
        EXPECT_EQ(t.territory, base + 502);
        EXPECT_DOUBLE_EQ(t.demand, 3.);
      }
  EXPECT_EQ(Utilities::MPI::sum(terminals, comm), 1u);
  for (const auto &[id, vessel] : original.vessels)
    if (vessel.owner == rank)
      {
        ASSERT_TRUE(actual.accepted.network.vessels.count(id));
        const auto &preserved = actual.accepted.network.vessels.at(id);
        EXPECT_EQ(preserved.nodes, vessel.nodes);
        EXPECT_EQ(preserved.radii, vessel.radii);
        EXPECT_EQ(preserved.owner, vessel.owner);
        EXPECT_EQ(preserved.system, vessel.system);
        EXPECT_DOUBLE_EQ(preserved.viscosity, vessel.viscosity);
      }
}
