// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_description.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/distribution.h>
#include <immersx/vascularization/generation.h>

#include "../source/vascularization/growth_batch.h"

using namespace dealii;
using namespace ImmersX::Vascularization;
using namespace ImmersX::Vascularization::internal;
namespace
{
  unsigned int
  rank()
  {
    return Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  }
  unsigned int
  size()
  {
    return Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  }
  unsigned int
  point_owner(const Triangulation<3> &bg, const Point<3> &p)
  {
    unsigned int owner = size();
    for (const auto &cell : bg.active_cell_iterators())
      if (cell->is_locally_owned() && cell->bounding_box().point_inside(p))
        owner = rank();
    return Utilities::MPI::min(owner, MPI_COMM_WORLD);
  }
  NetworkData
  roots(const Triangulation<3> &bg,
        const Mapping<3>       &mapping,
        bool                    port = false)
  {
    NetworkData n;
    if (rank() == size() - 1)
      for (unsigned int i = 0; i < 2; ++i)
        {
          const Id a = 2 * i, b = a + 1;
          n.nodes[a].position =
            Point<3>(i == 0 && port ? 0. : .2 + i, .5 + i, .5);
          n.nodes[a].owner = rank();
          n.nodes[a].port  = Port::arterial_inlet;
          if (i == 0 && port)
            n.nodes[a].inward_port_normal = Point<3>(1., 0., 0.);
          n.nodes[b].position = Point<3>(.8 + i, .5 + i, .5);
          n.nodes[b].owner    = rank();
          Vessel v;
          v.nodes                = {{a, b}};
          v.radii                = {{.01, .01}};
          v.viscosity            = .004;
          v.owner                = rank();
          n.vessels[10 + 10 * i] = v;
        }
    return redistribute(bg, mapping, n, 3., MPI_COMM_WORLD);
  }
  BatchProposal
  split(const NetworkData &n, unsigned int i, const Point<3> &target)
  {
    BatchProposal p;
    p.id            = i;
    p.cost          = 1. + i;
    const Id    old = 10 + 10 * i, j = 100 + 2 * i, t = j + 1, c = 1000 + 2 * i;
    const auto &v         = n.vessels.at(old);
    const auto  a         = n.nodes.at(v.nodes[0]).position,
               b          = n.nodes.at(v.nodes[1]).position;
    const Point<3> middle = .5 * (a + b);
    const double   r      = .01 * std::cbrt(.5);
    p.changed             = {{old, {{v.nodes[0], j}}, {{a, middle}}, .01},
                             {c, {{j, v.nodes[1]}}, {{middle, b}}, r},
                             {c + 1, {{j, t}}, {{middle, target}}, r}};
    p.owners[old]         = v.owner;
    p.old_nodes[old]      = v.nodes;
    for (Id node : v.nodes)
      if (n.nodes.at(node).inward_port_normal.norm_square() > 0.)
        {
          p.normals[node]     = n.nodes.at(node).inward_port_normal;
          p.port_owners[node] = n.nodes.at(node).owner;
        }
    return p;
  }
  BatchSelection
  select(BackgroundGeometry               &geometry,
         const NetworkData                &n,
         const std::vector<BatchProposal> &p,
         Parameters                       &controls)
  {
    return select_growth_batch(
      geometry, n, p, controls, MPI_COMM_WORLD, [&](Id id) {
        const auto &v = n.vessels.at(id);
        return std::max(v.radii[0], v.radii[1]);
      });
  }
} // namespace
TEST(VascularGrowthBatch,
     BOTH_DisjointBifurcationsAreSelectedTogetherByTheirActualOwners)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 1, 1},
                                            Point<3>(),
                                            Point<3>(2., 2., 2.));
  const MappingQ1<3> mapping;
  const auto         n          = roots(bg, mapping);
  const Point<3>     targets[2] = {{.48, .85, .5}, {1.52, 1.85, .5}};
  const auto a = point_owner(bg, targets[0]), b = point_owner(bg, targets[1]);
  std::vector<BatchProposal> local;
  if (rank() == a)
    local.push_back(split(n, 0, targets[0]));
  if (rank() == b)
    local.push_back(split(n, 1, targets[1]));
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  const auto         result = select(geometry, n, local, p);
  unsigned int       count  = 0;
  for (const auto &[id, accepted] : result.local)
    {
      EXPECT_TRUE(accepted);
      count += accepted;
    }
  EXPECT_EQ(Utilities::MPI::sum(count, MPI_COMM_WORLD), 2u);
  for (const auto &[key, accepted] : result.relevant)
    EXPECT_TRUE(accepted);
  // Selection does not mutate an existing vessel or accepted physical state.
  for (const auto &[id, v] : n.vessels)
    EXPECT_EQ(v.radii, (std::array<double, 2>{{.01, .01}}));
}
TEST(VascularGrowthBatch,
     BOTH_NewRaysCollideAcrossTheInterfaceAndChooseOneDeterministicWinner)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 1, 1},
                                            Point<3>(),
                                            Point<3>(2., 2., 2.));
  MappingQ1<3>   mapping;
  const auto     n          = roots(bg, mapping);
  const Point<3> targets[2] = {{.995, 1., .5}, {1.005, 1., .5}};
  const auto a = point_owner(bg, targets[0]), b = point_owner(bg, targets[1]);
  std::vector<BatchProposal> local;
  if (rank() == a)
    local.push_back(split(n, 0, targets[0]));
  if (rank() == b)
    local.push_back(split(n, 1, targets[1]));
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  const auto         result = select(geometry, n, local, p);
  if (rank() == a)
    EXPECT_TRUE(result.local.at(0));
  if (rank() == b)
    EXPECT_FALSE(result.local.at(1));
  for (const auto &[key, accepted] : result.relevant)
    EXPECT_EQ(accepted, key.second == 0);
  const auto again = select(geometry, n, local, p);
  EXPECT_EQ(again.local, result.local);
  EXPECT_EQ(again.relevant, result.relevant);
}
TEST(VascularGrowthBatch,
     BOTH_AnInvalidLowCostNeighborDoesNotBlockAValidProposal)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 1, 1},
                                            Point<3>(),
                                            Point<3>(2., 2., 2.));
  MappingQ1<3>   mapping;
  const auto     n          = roots(bg, mapping);
  const Point<3> targets[2] = {{.995, 1., .5}, {1.005, 1., .5}};
  const auto a = point_owner(bg, targets[0]), b = point_owner(bg, targets[1]);
  std::vector<BatchProposal> local;
  if (rank() == a)
    {
      auto bad              = split(n, 0, targets[0]);
      bad.changed[0].radius = .6;
      local.push_back(bad);
    }
  if (rank() == b)
    local.push_back(split(n, 1, targets[1]));
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  const auto         result = select(geometry, n, local, p);
  if (rank() == a)
    EXPECT_FALSE(result.local.at(0));
  if (rank() == b)
    EXPECT_TRUE(result.local.at(1));
  for (const auto &[key, accepted] : result.relevant)
    EXPECT_EQ(accepted, key.second == 1);
}
TEST(VascularGrowthBatch, BOTH_ModifiedTopologyAndGeometryNeedTheOwnersConsent)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 1, 1},
                                            Point<3>(),
                                            Point<3>(2., 2., 2.));
  MappingQ1<3>       mapping;
  const auto         n = roots(bg, mapping);
  const Point<3>     target(.48, .85, .5);
  const auto         proposer = point_owner(bg, target);
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  for (unsigned int test = 0; test < 3; ++test)
    {
      std::vector<BatchProposal> local;
      if (rank() == proposer)
        {
          auto candidate = split(n, 0, target);
          if (test == 0)
            candidate.old_nodes.at(10)[1] = 999999;
          else if (test == 1)
            candidate.changed[0].points[0][1] += .03;
          else if (test == 2)
            {
              if (size() > 1)
                candidate.owners[10] = (candidate.owners.at(10) + 1) % size();
              else
                candidate.old_nodes.at(10)[0] = 999999;
            }
          local.push_back(candidate);
        }
      const auto result = select(geometry, n, local, p);
      if (rank() == proposer)
        EXPECT_FALSE(result.local.at(0));
      for (const auto &[key, accepted] : result.relevant)
        EXPECT_FALSE(accepted);
    }
}
TEST(VascularGrowthBatch,
     BOTH_APlanarPortNeedsOneMatchingPhysicalFaceAndGlobalSeparation)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 1, 1},
                                            Point<3>(),
                                            Point<3>(2., 2., 2.));
  MappingQ1<3>               mapping;
  const auto                 n = roots(bg, mapping, true);
  const Point<3>             target(.48, .85, .5);
  const auto                 proposer = point_owner(bg, target);
  std::vector<BatchProposal> local;
  if (rank() == proposer)
    local.push_back(split(n, 0, target));
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  const auto         result = select(geometry, n, local, p);
  if (rank() == proposer)
    EXPECT_TRUE(result.local.at(0));
  for (const auto &[key, accepted] : result.relevant)
    EXPECT_TRUE(accepted);
}
TEST(VascularGrowthBatch,
     BOTH_SimultaneousWritesToOnePhysicalVesselHaveOneWinner)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 1, 1},
                                            Point<3>(),
                                            Point<3>(2., 2., 2.));
  MappingQ1<3>   mapping;
  const auto     n          = roots(bg, mapping);
  const Point<3> targets[2] = {{.48, .85, .5}, {1.52, .85, .8}};
  const auto a = point_owner(bg, targets[0]), b = point_owner(bg, targets[1]);
  std::vector<BatchProposal> local;
  if (rank() == b)
    {
      auto proposal = split(n, 0, targets[1]);
      proposal.id   = 1;
      proposal.cost = 2.;
      for (auto &e : proposal.changed)
        {
          if (e.id >= 1000)
            e.id += 10;
          for (auto &node : e.nodes)
            if (node >= 100)
              node += 10;
        }
      local.push_back(proposal);
    }
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  const auto         alone = select(geometry, n, local, p);
  if (rank() == b)
    EXPECT_TRUE(alone.local.at(1));
  if (rank() == a)
    local.push_back(split(n, 0, targets[0]));
  const auto together = select(geometry, n, local, p);
  if (rank() == a)
    EXPECT_TRUE(together.local.at(0));
  if (rank() == b)
    EXPECT_FALSE(together.local.at(1));
  for (const auto &[key, accepted] : together.relevant)
    EXPECT_EQ(accepted, key.second == 0);
}
TEST(VascularGrowthBatch,
     BOTH_EndpointsInsideDifferentPartitionsDoNotAllowLeavingTheDomain)
{
  Triangulation<3> full, serial;
  GridGenerator::subdivided_hyper_rectangle(full,
                                            {3, 1, 1},
                                            Point<3>(),
                                            Point<3>(3., 2., 2.));
  std::set<Triangulation<3>::active_cell_iterator> remove;
  for (const auto &cell : full.active_cell_iterators())
    if (cell->center()[0] > 1. && cell->center()[0] < 2.)
      remove.insert(cell);
  GridGenerator::create_triangulation_with_removed_cells(full, remove, serial);
  for (const auto &cell : serial.active_cell_iterators())
    cell->set_subdomain_id(cell->center()[0] < 1. ? 0 : size() - 1);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<3> bg(MPI_COMM_WORLD);
  bg.create_triangulation(description);
  MappingQ1<3>               mapping;
  const auto                 n = roots(bg, mapping);
  const Point<3>             target(2.5, .85, .5);
  const auto                 proposer = point_owner(bg, target);
  std::vector<BatchProposal> local;
  if (rank() == proposer)
    local.push_back(split(n, 0, target));
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  const auto         result = select(geometry, n, local, p);
  if (rank() == proposer)
    EXPECT_FALSE(result.local.at(0));
  for (const auto &[key, accepted] : result.relevant)
    EXPECT_FALSE(accepted);
}
TEST(VascularGrowthBatch,
     BOTH_MalformedLocalBatchFailsCollectivelyBeforeCommunicationPhases)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 1, 1},
                                            Point<3>(),
                                            Point<3>(2., 2., 2.));
  MappingQ1<3>               mapping;
  const auto                 n = roots(bg, mapping);
  const Point<3>             target(.48, .85, .5);
  const auto                 proposer = point_owner(bg, target);
  std::vector<BatchProposal> local;
  if (rank() == proposer)
    {
      auto bad = split(n, 0, target);
      bad.cost = std::numeric_limits<double>::quiet_NaN();
      local.push_back(bad);
    }
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  EXPECT_THROW(select(geometry, n, local, p), ExceptionBase);
  if (size() > 1)
    EXPECT_THROW(select_growth_batch(
                   geometry, {}, {}, p, MPI_COMM_SELF, [](Id) { return .01; }),
                 ExceptionBase);
}
TEST(VascularGrowthBatch,
     BOTH_PointSearchToleranceCannotAdmitAnExteriorThinVessel)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 1, 1},
                                            Point<3>(),
                                            Point<3>(2., 2., 2.));
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  for (double x : {-1.e-11, 1.e-11})
    {
      std::vector<BatchProposal> local;
      if (rank() == 0)
        {
          BatchProposal proposal;
          proposal.id      = 0;
          proposal.cost    = 1.;
          proposal.changed = {{100,
                               {{1000, 1001}},
                               {{Point<3>(x, .5, .5), Point<3>(x, 1., .5)}},
                               1.e-12}};
          local.push_back(proposal);
        }
      const auto result = select(geometry, {}, local, p);
      if (rank() == 0)
        EXPECT_EQ(result.local.at(0), x > 0.);
      for (const auto &[key, accepted] : result.relevant)
        EXPECT_EQ(accepted, x > 0.);
    }
}

TEST(VascularGrowthBatch, BOTH_SpatialConflictVotesMatchExhaustivePairs)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 2, 2},
                                            Point<3>(),
                                            Point<3>(4., 4., 4.));
  MappingQ1<3>               mapping;
  BackgroundGeometry         geometry(bg, mapping);
  Parameters                 controls;
  std::vector<BatchProposal> proposals;
  const auto                 edge =
    [](Id id, const Point<3> &a, const Point<3> &b, double radius) {
      return Edge{id, {{2 * id, 2 * id + 1}}, {{a, b}}, radius};
    };
  for (unsigned int i = 0; i < 72; ++i)
    {
      BatchProposal p;
      p.id   = i;
      p.cost = i % 5;
      // Two distinct physical changes per proposal, with varied radii and
      // envelopes that sometimes overlap although the segments do not.
      for (unsigned int j = 0; j < 2; ++j)
        {
          const unsigned int k = 2 * i + j;
          const Point<3>     a(.3 + .4 * (k % 8),
                           .3 + .4 * ((k / 8) % 8),
                           .4 + .65 * (k % 5));
          p.changed.push_back(
            edge(1000 + k,
                 a,
                 a + Point<3>(.25, .11 * (k % 3), -.08 * (k % 2)),
                 .003 + .004 * (k % 4)));
        }
      proposals.push_back(p);
    }
  proposals[0].changed = {
    edge(100, Point<3>(.3, 1., 1.), Point<3>(3.7, 1., 1.), .015)};
  proposals[1].changed = {
    edge(101, Point<3>(2., .3, 1.02), Point<3>(2., 3.7, 1.02), .01)};
  proposals[2].changed = {
    edge(102, Point<3>(.3, 1., 1.09), Point<3>(3.7, 1., 1.09), .004)};
  proposals[3].changed = {
    edge(103, Point<3>(.3, 1., 1.), Point<3>(1., 1.001, 1.), .006)};
  proposals[3].changed[0].nodes[0] = proposals[0].changed[0].nodes[0];
  // Writes to the same physical identifier must be compared even when their
  // provisional envelopes are disjoint.
  proposals[4].changed = {
    edge(100, Point<3>(.3, 3.5, 3.5), Point<3>(1., 3.5, 3.5), .005)};
  proposals[4].changed[0].nodes = {{9990, 9991}};
  const auto producer           = [&](Id id) {
    return id == 4 ? 0u : static_cast<unsigned int>(id % std::min(3u, size()));
  };
  for (double clearance : {0., .08})
    {
      controls.clearance = clearance;
      // This exhaustive oracle has no spatial filtering. The exact segment
      // predicate has independent geometric tests; this checks that broad
      // phase pruning preserves every deterministic vote and tie break.
      std::map<Id, bool> expected;
      for (const auto &a : proposals)
        {
          bool accepted = true;
          for (const auto &b : proposals)
            if (std::make_tuple(b.cost, producer(b.id), b.id) <
                std::make_tuple(a.cost, producer(a.id), a.id))
              for (const auto &e : a.changed)
                for (const auto &f : b.changed)
                  if (e.id == f.id ||
                      !compatible(e, f, controls.minimum_angle, clearance))
                    accepted = false;
          expected[a.id] = accepted;
        }
      std::vector<BatchProposal> local;
      for (const auto &p : proposals)
        if (producer(p.id) == rank())
          local.push_back(p);
      for (unsigned int order = 0; order < 2; ++order)
        {
          const auto   selected = select(geometry, {}, local, controls);
          unsigned int accepted = 0;
          for (const auto &[id, chosen] : selected.local)
            {
              EXPECT_EQ(chosen, expected.at(id))
                << "proposal=" << id << " clearance=" << clearance;
              accepted += chosen;
            }
          for (const auto &[key, chosen] : selected.relevant)
            EXPECT_EQ(chosen, expected.at(key.second));
          const auto count = Utilities::MPI::sum(accepted, MPI_COMM_WORLD);
          EXPECT_GT(count, 0u);
          EXPECT_LT(count, proposals.size());
          std::reverse(local.begin(), local.end());
        }
    }
}
