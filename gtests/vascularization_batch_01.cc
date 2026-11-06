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

#include <algorithm>
#include <cmath>
#include <limits>

#include "../source/vascularization/growth_batch.h"

using namespace dealii;
using namespace ImmersX::Vascularization;
using namespace ImmersX::Vascularization::internal;
namespace
{
  void
  physical_controls(Parameters &p)
  {
    p.length_scale               = "meter";
    p.minimum_length             = 0.;
    p.density_beta               = 0.;
    p.coverage.service_radius    = 0.;
    p.venous.terminal_separation = 0.;
  }
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
        return v.radii;
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
  physical_controls(p);
  const auto   result = select(geometry, n, local, p);
  unsigned int count  = 0;
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
  const Point<3> targets[2] = {{1.005, 1., .5}, {.995, 1., .5}};
  // Both flat-ended rays contain this same strict interior point, so the
  // distributed conflict does not depend on rounded capsule endcaps.
  const Point<3> witness(1., 1., .5);
  for (unsigned int i = 0; i < 2; ++i)
    {
      const Point<3> origin(.5 + i, .5 + i, .5);
      const auto     axis = targets[i] - origin;
      const double   t    = ((witness - origin) * axis) / axis.norm_square();
      EXPECT_GT(t, 0.);
      EXPECT_LT(t, 1.);
      EXPECT_LT((witness - origin - t * axis).norm(), .01 * std::cbrt(.5));
    }
  const auto a = point_owner(bg, targets[0]), b = point_owner(bg, targets[1]);
  std::vector<BatchProposal> local;
  if (rank() == a)
    local.push_back(split(n, 0, targets[0]));
  if (rank() == b)
    local.push_back(split(n, 1, targets[1]));
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  physical_controls(p);
  const auto result = select(geometry, n, local, p);
  if (rank() == a)
    {
      EXPECT_TRUE(result.local.at(0));
    }
  if (rank() == b)
    {
      EXPECT_FALSE(result.local.at(1));
    }
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
  physical_controls(p);
  const auto result = select(geometry, n, local, p);
  if (rank() == a)
    {
      EXPECT_FALSE(result.local.at(0));
    }
  if (rank() == b)
    {
      EXPECT_TRUE(result.local.at(1));
    }
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
  physical_controls(p);
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
        {
          EXPECT_FALSE(result.local.at(0));
        }
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
  physical_controls(p);
  const auto result = select(geometry, n, local, p);
  if (rank() == proposer)
    {
      EXPECT_TRUE(result.local.at(0));
    }
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
  physical_controls(p);
  const auto alone = select(geometry, n, local, p);
  if (rank() == b)
    {
      EXPECT_TRUE(alone.local.at(1));
    }
  if (rank() == a)
    local.push_back(split(n, 0, targets[0]));
  const auto together = select(geometry, n, local, p);
  if (rank() == a)
    {
      EXPECT_TRUE(together.local.at(0));
    }
  if (rank() == b)
    {
      EXPECT_FALSE(together.local.at(1));
    }
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
  physical_controls(p);
  const auto result = select(geometry, n, local, p);
  if (rank() == proposer)
    {
      EXPECT_FALSE(result.local.at(0));
    }
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
  physical_controls(p);
  EXPECT_THROW(select(geometry, n, local, p), ExceptionBase);
  if (size() > 1)
    {
      EXPECT_THROW(
        select_growth_batch(geometry,
                            {},
                            {},
                            p,
                            MPI_COMM_SELF,
                            [](Id) {
                              return std::array<double, 2>{{.01, .01}};
                            }),
        ExceptionBase);
    }
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
  physical_controls(p);
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
        {
          EXPECT_EQ(result.local.at(0), x > 0.);
        }
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
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  Parameters         controls;
  physical_controls(controls);
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
TEST(VascularGrowthBatch,
     BOTH_FlatTubeSeparationIsInvariantUnderDegreeTwoSamples)
{
  const Edge extension{100,
                       {{1, 2}},
                       {{Point<3>(.6, .5, .5), Point<3>(.9, .5, .5)}},
                       .05};
  for (const unsigned int pieces : {1u, 2u, 8u, 32u})
    for (unsigned int i = 0; i < pieces; ++i)
      {
        const Edge fixed{10 + i,
                         {{i == 0 ? Id{0} : Id{1000 + i},
                           i + 1 == pieces ? Id{1} : Id{1001 + i}}},
                         {{Point<3>(.1 + .5 * i / pieces, .5, .5),
                           Point<3>(.1 + .5 * (i + 1) / pieces, .5, .5)}},
                         .05};
        // These flat-ended pieces have disjoint interiors. Only the last
        // piece shares the actual junction; sampling changes no physical tube.
        EXPECT_TRUE(compatible(extension, fixed, 15., 0.))
          << pieces << ':' << i;
        EXPECT_TRUE(compatible(fixed, extension, 15., 0.))
          << pieces << ':' << i;
      }
}

TEST(VascularGrowthBatch,
     BOTH_FlatCylinderSeparatesAnatomicalTipButNotIntersection)
{
  const Point<3> p52(.15715330839, .11075111479, .092861868441);
  const Point<3> p53(.15608678758, .1121147275, .085326336324);
  const Point<3> tip(.15570352972, .11287145317, .084041535854);
  const double   r52 = .0012242651519, r53 = .0012156437614;
  const double   rtip = .0010541478181, length = .009112321167690196;
  const Edge     fixed{52, {{52, 53}}, {{p52, p53}}, r52};
  const auto     direction = (tip - p53) / tip.distance(p53);
  const Edge forward{100, {{54, 55}}, {{tip, tip + length * direction}}, rtip};
  EXPECT_TRUE(compatible(forward, fixed, 15., 0.));
  EXPECT_TRUE(compatible(fixed, forward, 15., 0.));
  const Edge reverse{100,
                     {{54, 55}},
                     {{tip, tip + Point<3>(0., 0., length)}},
                     rtip};
  // This point lies strictly inside both actual linearly tapered frusta,
  // independently of the bounding-cylinder separation implementation.
  const Point<3> witness(.15570352972, .11287145317, .08559063045250732);
  const auto     check_inside = [&](const Edge &edge, double r0, double r1) {
    const auto   axis = edge.points[1] - edge.points[0];
    const double t = ((witness - edge.points[0]) * axis) / axis.norm_square();
    EXPECT_GT(t, 0.);
    EXPECT_LT(t, 1.);
    EXPECT_LT((witness - edge.points[0] - t * axis).norm(),
              (1. - t) * r0 + t * r1);
  };
  check_inside(fixed, r52, r53);
  check_inside(reverse, rtip, rtip);
  EXPECT_FALSE(compatible(reverse, fixed, 15., 0.));
  EXPECT_FALSE(compatible(fixed, reverse, 15., 0.));
}

TEST(VascularGrowthBatch,
     BOTH_FlatCylinderKeepsTouchClearanceAndCrossingsRejected)
{
  const Edge first{0,
                   {{0, 1}},
                   {{Point<3>(0., 0., 0.), Point<3>(1., 0., 0.)}},
                   .1};
  Edge       second{2,
                    {{2, 3}},
                    {{Point<3>(1.05, 0., 0.), Point<3>(2., 0., 0.)}},
              .1};
  EXPECT_TRUE(compatible(first, second, 15., 0.));
  EXPECT_TRUE(compatible(first, second, 15., .04));
  EXPECT_FALSE(compatible(first, second, 15., .05));
  second.points[0][0] = 1.;
  EXPECT_FALSE(compatible(first, second, 15., 0.));
  second.points = {{Point<3>(.5, -1., 0.), Point<3>(.5, 1., 0.)}};
  EXPECT_FALSE(compatible(first, second, 15., 0.));
  second.points = {{Point<3>(0., .2, 0.), Point<3>(1., .2, 0.)}};
  EXPECT_FALSE(compatible(first, second, 15., 0.));
  second.points = {{Point<3>(2., 1.e-10, 0.), Point<3>(1.05, 0., 0.)}};
  EXPECT_TRUE(compatible(first, second, 15., 0.));
  for (const double gap : {1.e-6, .05})
    {
      const Point<3> shift(1.e9, 1.e9, 1.e9);
      Edge           a = first, b = second;
      a.points = {{shift, shift + Point<3>(1., 0., 0.)}};
      b.points = {
        {shift + Point<3>(1. + gap, 0., 0.), shift + Point<3>(2., 1.e-10, 0.)}};
      EXPECT_EQ(compatible(a, b, 15., 0.), gap == .05);
    }
  const Point<3> shift(1.e6, 1.e6, 1.e6);
  const Edge     needle{0,
                        {{0, 1}},
                        {{shift, shift + Point<3>(1.e-6, 0., 0.)}},
                    .1};
  const Edge     near{2,
                      {{2, 3}},
                      {{shift + Point<3>(.000101, 0., 0.),
                        shift + Point<3>(.000102, 0., 0.)}},
                  .1};
  // Normalizing these very short axes amplifies absolute coordinate error;
  // a coordinate-only guard would falsely call this test well resolved.
  EXPECT_FALSE(compatible(needle, near, 15., 0.));
  for (const double extent : {1.e150, 1.e154})
    {
      const Edge huge_a{0,
                        {{0, 1}},
                        {{Point<3>(-extent, 0., 0.), Point<3>(extent, 0., 0.)}},
                        .1};
      const Edge huge_b{2,
                        {{2, 3}},
                        {{Point<3>(0., -extent, 0.), Point<3>(0., extent, 0.)}},
                        .1};
      // Their axes intersect. An overflowing capsule distance is uncertainty,
      // not a positive separation certificate.
      EXPECT_FALSE(compatible(huge_a, huge_b, 15., 0.));
    }
}

TEST(VascularGrowthBatch, BOTH_FlatCylinderVoteKeepsRemoteAndEmptyOwners)
{
  Triangulation<3> serial;
  GridGenerator::subdivided_hyper_rectangle(serial,
                                            {2, 1, 1},
                                            Point<3>(),
                                            Point<3>(2., 1., 1.));
  for (const auto &cell : serial.active_cell_iterators())
    cell->set_subdomain_id(cell->center()[0] < 1. ? 0 : size() - 1);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<3> bg(MPI_COMM_WORLD);
  bg.create_triangulation(description);
  const MappingQ1<3> mapping;
  if (size() > 2)
    {
      EXPECT_EQ(Utilities::MPI::sum(bg.n_locally_owned_active_cells() == 0 ?
                                      1u :
                                      0u,
                                    MPI_COMM_WORLD),
                size() - 2);
    }
  for (unsigned int scenario = 0; scenario < 3; ++scenario)
    {
      NetworkData input;
      if (rank() == size() - 1)
        {
          input.nodes[0] = {{.2, .5, .5}, rank(), Port::arterial_inlet};
          input.nodes[1] = {{.55, .5, .5}, rank()};
          input.nodes[2] = {{.6, .5, .5}, rank()};
          for (Id id : {0u, 1u})
            {
              Vessel v;
              v.nodes           = {{id, id + 1}};
              v.radii           = {{.05, .05}};
              v.viscosity       = .004;
              v.owner           = rank();
              input.vessels[id] = v;
            }
          if (scenario == 1)
            {
              input.nodes[10]   = {{.65, .5, .5}, rank(), Port::portal_inlet};
              input.nodes[11]   = {{.85, .5, .5}, rank()};
              Vessel v          = input.vessels.at(0);
              v.nodes           = {{10, 11}};
              v.system          = System::portal;
              input.vessels[10] = v;
            }
        }
      const auto n = redistribute(bg, mapping, input, .2, MPI_COMM_WORLD);
      std::vector<BatchProposal> local;
      if (rank() == size() - 1)
        {
          BatchProposal proposal;
          proposal.id      = 100;
          proposal.cost    = 1.;
          proposal.changed = {{100,
                               {{2, 100}},
                               {{Point<3>(.6, .5, .5),
                                 Point<3>(.9, scenario == 2 ? 1.1 : .5, .5)}},
                               .05}};
          local.push_back(proposal);
        }
      BackgroundGeometry geometry(bg, mapping);
      Parameters         p;
      physical_controls(p);
      p.minimum_angle     = 15.;
      const auto result   = select(geometry, n, local, p);
      const bool expected = scenario == 0;
      if (rank() == size() - 1)
        {
          EXPECT_EQ(result.local.at(100), expected);
        }
      for (const auto &[key, accepted] : result.relevant)
        EXPECT_EQ(accepted, expected);
    }
}


namespace
{
  // The convex hull of these circumscribed endpoint polygons contains the
  // true linear-taper frustum. Its vertex projections are an independent
  // separation oracle; they do not use the production support function.
  std::pair<double, double>
  outer_frustum_projection(const std::array<Point<3>, 2> &points,
                           const std::array<double, 2>   &radii,
                           const Tensor<1, 3>            &normal,
                           const Point<3>                &origin)
  {
    constexpr unsigned int sides     = 128;
    auto                   direction = points[1] - points[0];
    direction /= direction.norm();
    Tensor<1, 3> reference;
    reference[std::abs(direction[0]) < .9 ? 0 : 1] = 1.;
    auto first = cross_product_3d(direction, reference);
    first /= first.norm();
    const auto second = cross_product_3d(direction, first);
    double     lower  = std::numeric_limits<double>::infinity();
    double     upper  = -lower;
    for (unsigned int end = 0; end < 2; ++end)
      for (unsigned int i = 0; i < sides; ++i)
        {
          const double angle = 2. * numbers::PI * i / sides;
          const auto   vertex =
            points[end] +
            radii[end] / std::cos(numbers::PI / sides) *
              (std::cos(angle) * first + std::sin(angle) * second);
          const double value = normal * (vertex - origin);
          lower              = std::min(lower, value);
          upper              = std::max(upper, value);
        }
    return {lower, upper};
  }
} // namespace

TEST(VascularGrowthBatch, BOTH_NativeTaperedAttachmentKeepsTrueSeparation)
{
  Triangulation<3> serial;
  GridGenerator::subdivided_hyper_rectangle(serial,
                                            {2, 1, 1},
                                            Point<3>(.14, .11, .20),
                                            Point<3>(.18, .15, .24));
  for (const auto &cell : serial.active_cell_iterators())
    cell->set_subdomain_id(cell->center()[0] < .16 ? 0 : size() - 1);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<3> bg(MPI_COMM_WORLD);
  bg.create_triangulation(description);
  MappingQ1<3> mapping;
  if (size() > 2)
    {
      EXPECT_EQ(Utilities::MPI::sum(bg.n_locally_owned_active_cells() == 0 ?
                                      1u :
                                      0u,
                                    MPI_COMM_WORLD),
                size() - 2);
    }

  const Point<3> p121(.16105571389, .12534973025, .21835261583);
  const Point<3> p169(.15844026208, .12357060611, .22041496634);
  const Point<3> original(.15807075799, .12433482707, .22056400776);
  const Point<3> knee(.1526197896255688, .1356087057336848, .22276268454706896);
  const std::array<double, 2> old_radii{{.0013626601474, .00073727190304}};
  constexpr double            tip_radius = .00038278414048;
  const Tensor<1, 3>          normal{
             {-.554320947064179, .7858657958471155, .27412267064847345}};
  const auto old_projection =
    outer_frustum_projection({{p121, p169}}, old_radii, normal, original);
  const auto new_projection = outer_frustum_projection(
    {{original, knee}}, {{tip_radius, tip_radius}}, normal, original);
  EXPECT_GT(new_projection.first - old_projection.second, 4.e-5);

  for (const bool intersecting : {false, true})
    {
      Point<3> start = original;
      if (intersecting)
        {
          auto         axis      = p169 - p121;
          const double parameter = ((start - p121) * axis) / axis.norm_square();
          auto         radial    = start - p121 - parameter * axis;
          radial /= radial.norm();
          start -= .00015 * radial;
          auto ray = knee - start;
          ray /= ray.norm();
          const auto   witness = start + 1.e-5 * ray;
          const double t       = ((witness - p121) * axis) / axis.norm_square();
          EXPECT_GT(t, 0.);
          EXPECT_LT(t, 1.);
          EXPECT_LT((witness - p121 - t * axis).norm(),
                    (1. - t) * old_radii[0] + t * old_radii[1]);
          EXPECT_GT((witness - start) * ray, 0.);
          EXPECT_LT((witness - start) * ray, start.distance(knee));
          EXPECT_LT((witness - start - ((witness - start) * ray) * ray).norm(),
                    tip_radius);
        }
      NetworkData input;
      if (rank() == size() - 1)
        {
          input.nodes[121] = {p121, rank()};
          input.nodes[169] = {p169, rank()};
          input.nodes[170] = {start, rank()};
          Vessel old;
          old.nodes          = {{121, 169}};
          old.radii          = old_radii;
          old.viscosity      = .00371;
          old.system         = System::venous;
          old.owner          = rank();
          input.vessels[151] = old;
          old.nodes          = {{169, 170}};
          old.radii          = {{old_radii[1], tip_radius}};
          input.vessels[168] = old;
        }
      const auto native = redistribute(bg, mapping, input, .02, MPI_COMM_WORLD);
      std::vector<BatchProposal> proposals;
      if (rank() == size() - 1)
        {
          BatchProposal p;
          p.id      = 2020;
          p.changed = {{2020, {{170, 2023}}, {{start, knee}}, tip_radius}};
          proposals.push_back(p);
        }
      BackgroundGeometry geometry(bg, mapping);
      Parameters         p;
      physical_controls(p);
      p.minimum_angle      = 15.;
      const auto selection = select(geometry, native, proposals, p);
      if (rank() == size() - 1)
        {
          EXPECT_EQ(selection.local.at(2020), !intersecting);
        }
      for (const auto &[key, accepted] : selection.relevant)
        EXPECT_EQ(accepted, !intersecting);
      for (const auto &[id, vessel] : native.vessels)
        if (id == 151)
          {
            EXPECT_EQ(vessel.radii, old_radii);
          }
    }
}


TEST(VascularGrowthBatch, BOTH_CurrentRadiusCannotUseNarrowSnapshotProfile)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_cube(bg, 2);
  MappingQ1<3> mapping;
  NetworkData  input;
  if (rank() == size() - 1)
    {
      input.nodes[0] = {{.3, .5, .5}, rank(), Port::arterial_inlet};
      input.nodes[1] = {{.7, .5, .5}, rank()};
      Vessel vessel;
      vessel.nodes     = {{0, 1}};
      vessel.radii     = {{.005, .005}};
      vessel.viscosity = .004;
      vessel.owner     = rank();
      input.vessels[0] = vessel;
    }
  const auto native = redistribute(bg, mapping, input, .1, MPI_COMM_WORLD);
  std::vector<BatchProposal> proposals;
  if (rank() == size() - 1)
    {
      BatchProposal proposal;
      proposal.id      = 10;
      proposal.changed = {{10,
                           {{10, 11}},
                           {{Point<3>(.3, .55, .5), Point<3>(.7, .55, .5)}},
                           .04}};
      proposals.push_back(proposal);
    }
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  physical_controls(p);
  const auto snapshot = select(geometry, native, proposals, p);
  const auto current =
    select_growth_batch(geometry, native, proposals, p, MPI_COMM_WORLD, [](Id) {
      return std::array<double, 2>{{.04, .04}};
    });
  // (.5,.525,.5) is strictly inside both current cylinders. The narrow
  // materialized snapshot would separate them and must not replace the live
  // radius supplied after a demand update.
  EXPECT_LT(.025, .04);
  EXPECT_GT(.05, .04 + .005);
  if (rank() == size() - 1)
    {
      EXPECT_TRUE(snapshot.local.at(10));
      EXPECT_FALSE(current.local.at(10));
    }
  for (const auto &[key, accepted] : current.relevant)
    EXPECT_FALSE(accepted);
  for (const auto &[id, vessel] : native.vessels)
    EXPECT_EQ(vessel.radii, (std::array<double, 2>{{.005, .005}}));
}


TEST(VascularGrowthBatch, BOTH_TaperedClearanceRequiresStrictSeparation)
{
  const Edge a{0, {{0, 1}}, {{Point<3>(), Point<3>(1., 0., 0.)}}, .1};
  Edge b{1, {{2, 3}}, {{Point<3>(0., .15, 0.), Point<3>(1., .15, 0.)}}, .1};
  const std::array<double, 2> ra{{.1, .01}}, rb{{.01, .1}};
  // The facing generatrices are parallel lines y=.1-.09x and
  // y=.14-.09x. These interior points attain their exact separation.
  const Point<3>     first(.5, .1 - .09 * .5, 0.);
  const Tensor<1, 3> normal{{.09, 1., 0.}};
  const Point<3>     second = first + normal * (.04 / (1. + .09 * .09));
  const double       gap    = first.distance(second);
  EXPECT_NEAR(second[1], .14 - .09 * second[0], 1.e-16);
  EXPECT_GT(second[0], 0.);
  EXPECT_LT(second[0], 1.);
  EXPECT_FALSE(compatible(a, b, 15., 0.));
  EXPECT_TRUE(compatible(a, b, 15., gap - 1.e-8, ra, rb));
  EXPECT_TRUE(compatible(b, a, 15., gap - 1.e-8, rb, ra));
  EXPECT_FALSE(compatible(a, b, 15., gap + 1.e-8, ra, rb));
  EXPECT_FALSE(compatible(b, a, 15., gap + 1.e-8, rb, ra));
  for (const double invalid : {0.,
                               -.1,
                               .10001,
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()})
    EXPECT_FALSE(compatible(a, b, 15., 0., {{invalid, ra[1]}}, rb));
  b.points = {{Point<3>(0., .11, 0.), Point<3>(1., .11, 0.)}};
  EXPECT_FALSE(compatible(a, b, 15., 0., ra, rb));
}
