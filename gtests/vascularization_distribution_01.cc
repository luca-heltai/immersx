// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_description.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/distribution.h>

#include <cmath>
#include <limits>

using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  constexpr Id base = (Id{1} << 60) + 37;
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
  void
  stripes(parallel::distributed::Triangulation<3> &background,
          unsigned int                             cells = 8)
  {
    GridGenerator::subdivided_hyper_rectangle(background,
                                              {cells, 1, 1},
                                              Point<3>(0., 0., 0.),
                                              Point<3>(cells, 1., 1.));
  }
  unsigned int
  owner(const Triangulation<3> &background,
        const Point<3>         &point,
        unsigned int            outside)
  {
    unsigned int candidate = size();
    for (const auto &cell : background.active_cell_iterators())
      if (cell->is_locally_owned() && cell->bounding_box().point_inside(point))
        candidate = rank();
    const auto found = Utilities::MPI::min(candidate, MPI_COMM_WORLD);
    return found == size() ? outside : found;
  }
  NetworkData
  segment(double a, double b, unsigned int source, bool terminal = true)
  {
    NetworkData n;
    if (rank() != source)
      return n;
    n.nodes[base].position     = Point<3>(a, .5, .5);
    n.nodes[base].owner        = source;
    n.nodes[base].port         = Port::arterial_inlet;
    n.nodes[base + 1].position = Point<3>(b, .5, .5);
    n.nodes[base + 1].owner    = source;
    Vessel v;
    v.nodes                          = {{base, base + 1}};
    v.radii                          = {{.01, .02}};
    v.viscosity                      = .004;
    v.owner                          = source;
    v.physical_properties["density"] = 1050.;
    n.vessels[base]                  = v;
    if (terminal)
      n.terminals[base + 2] = {
        base + 1, base + 3, System::arterial, source, 7.};
    return n;
  }
  std::uint64_t
  owned_vessels(const NetworkData &n)
  {
    std::uint64_t count = 0;
    for (const auto &[id, v] : n.vessels)
      if (v.owner == rank())
        ++count;
    return Utilities::MPI::sum(count, MPI_COMM_WORLD);
  }
} // namespace
TEST(VascularDistribution,
     BOTH_SparseMigrationRetainsLargeIdsAndPhysicalProperties)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  stripes(bg);
  NetworkData input;
  const auto  source = size() - 1;
  if (rank() == source)
    for (unsigned int i = 0; i < 8; ++i)
      {
        const Id a = base + 4 * i, b = a + 1;
        input.nodes[a].position = Point<3>(i + .35, .5, .5);
        input.nodes[a].owner    = source;
        input.nodes[a].port     = Port::arterial_inlet;
        input.nodes[b].position = Point<3>(i + .65, .5, .5);
        input.nodes[b].owner    = source;
        Vessel v;
        v.nodes                          = {{a, b}};
        v.radii                          = {{.01, .02}};
        v.viscosity                      = .004;
        v.owner                          = source;
        v.physical_properties["density"] = 1050. + i;
        input.vessels[base + 100 + i]    = v;
        input.terminals[base + 200 + i]  = {
          b, base + 300 + i, System::arterial, source, 1. + i};
      }
  const auto output =
    redistribute(bg, MappingQ1<3>(), input, 0., MPI_COMM_WORLD);
  EXPECT_TRUE(validate_data(output, size()).valid());
  EXPECT_EQ(owned_vessels(output), 8u);
  for (unsigned int i = 0; i < 8; ++i)
    {
      const auto expected = owner(bg, Point<3>(i + .5, .5, .5), source);
      if (expected == rank())
        EXPECT_EQ(output.vessels.count(base + 100 + i), 1u);
      if (output.vessels.count(base + 100 + i))
        {
          const auto &v = output.vessels.at(base + 100 + i);
          EXPECT_EQ(v.owner, expected);
          EXPECT_EQ(v.nodes,
                    (std::array<Id, 2>{{base + 4 * i, base + 4 * i + 1}}));
          EXPECT_EQ(v.radii, (std::array<double, 2>{{.01, .02}}));
          EXPECT_DOUBLE_EQ(v.viscosity, .004);
          EXPECT_EQ(v.provenance, Provenance::imported);
          EXPECT_EQ(v.system, System::arterial);
          EXPECT_DOUBLE_EQ(v.physical_properties.at("density"), 1050. + i);
          EXPECT_EQ(output.nodes.at(base + 4 * i).position,
                    Point<3>(i + .35, .5, .5));
          EXPECT_EQ(output.nodes.at(base + 4 * i).port, Port::arterial_inlet);
          EXPECT_DOUBLE_EQ(output.terminals.at(base + 200 + i).demand, 1. + i);
          EXPECT_EQ(output.terminals.at(base + 200 + i).territory,
                    base + 300 + i);
        }
    }
  if (size() > 1)
    EXPECT_LT(output.vessels.size(), 8u);
  if (rank() == source)
    {
      EXPECT_EQ(input.vessels.size(), 8u);
      for (const auto &[id, v] : input.vessels)
        EXPECT_EQ(v.owner, source);
    }
  else
    EXPECT_TRUE(input.nodes.empty());
}
TEST(VascularDistribution,
     BOTH_AcrossPartitionsKeepsOnePhysicalVesselAndEndpointGhosts)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  stripes(bg);
  const auto source = size() - 1;
  auto       input  = segment(.1, 7.7, source);
  const auto output =
    redistribute(bg, MappingQ1<3>(), input, 0., MPI_COMM_WORLD);
  const auto expected_v = owner(bg, Point<3>(3.9, .5, .5), source),
             expected_a = owner(bg, Point<3>(.1, .5, .5), source),
             expected_b = owner(bg, Point<3>(7.7, .5, .5), source);
  EXPECT_EQ(owned_vessels(output), 1u);
  ASSERT_EQ(output.vessels.size(), 1u);
  ASSERT_EQ(output.nodes.size(), 2u);
  ASSERT_EQ(output.terminals.size(), 1u);
  EXPECT_EQ(output.vessels.at(base).owner, expected_v);
  EXPECT_EQ(output.nodes.at(base).owner, expected_a);
  EXPECT_EQ(output.nodes.at(base + 1).owner, expected_b);
  EXPECT_EQ(output.terminals.at(base + 2).owner, expected_b);
  EXPECT_EQ(output.vessels.at(base).nodes,
            (std::array<Id, 2>{{base, base + 1}}));
  EXPECT_NEAR(length(output, base), 7.6, 1.e-13);
  const double expected_volume =
    numbers::PI * 7.6 * (.0001 + .0002 + .0004) / 3.;
  EXPECT_NEAR(volume(output, base) / expected_volume, 1., 1.e-13);
}
TEST(VascularDistribution,
     BOTH_ExteriorInputKeepsItsOwnerAndInteriorCrossingGhosts)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  stripes(bg);
  const auto source = size() - 1;
  auto       input  = segment(-1., 3., source);
  if (rank() == source)
    {
      input.nodes.at(base).inward_port_normal = Point<3>(1., 0., 0.);
      auto extra                              = segment(-3., -2., source);
      Node a = extra.nodes.at(base), b = extra.nodes.at(base + 1);
      input.nodes[base + 10]   = a;
      input.nodes[base + 11]   = b;
      auto v                   = extra.vessels.at(base);
      v.nodes                  = {{base + 10, base + 11}};
      input.vessels[base + 10] = v;
    }
  const auto output =
    redistribute(bg, MappingQ1<3>(), input, 0., MPI_COMM_WORLD);
  EXPECT_EQ(owned_vessels(output), 2u);
  if (rank() == source)
    {
      ASSERT_EQ(output.vessels.count(base + 10), 1u);
      EXPECT_EQ(output.vessels.at(base + 10).owner, source);
      EXPECT_EQ(output.nodes.at(base + 10).position, Point<3>(-3., .5, .5));
      EXPECT_EQ(output.nodes.at(base).owner, source);
      EXPECT_EQ(output.nodes.at(base).inward_port_normal, Point<3>(1., 0., 0.));
    }
  if (output.vessels.count(base))
    {
      EXPECT_EQ(output.nodes.at(base).position, Point<3>(-1., .5, .5));
      EXPECT_DOUBLE_EQ(length(output, base), 4.);
    }
}
TEST(VascularDistribution,
     BOTH_RadiusHaloCanSpanSeveralPartitionsAndDropObsoleteGhosts)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  stripes(bg, 16);
  auto local = redistribute(
    bg, MappingQ1<3>(), segment(.2, .3, size() - 1), 0., MPI_COMM_WORLD);
  for (auto &[id, v] : local.vessels)
    if (v.owner == rank())
      v.radii = {{9., 9.}};
  const auto wide = redistribute(bg, MappingQ1<3>(), local, 0., MPI_COMM_WORLD);
  const auto copies =
    Utilities::MPI::sum(static_cast<unsigned int>(wide.vessels.size()),
                        MPI_COMM_WORLD);
  if (size() >= 4)
    EXPECT_GE(copies, 3u);
  EXPECT_EQ(owned_vessels(wide), 1u);
  auto shrinking = wide;
  for (auto &[id, v] : shrinking.vessels)
    {
      EXPECT_EQ(v.radii, (std::array<double, 2>{{9., 9.}}));
      if (v.owner == rank())
        v.radii = {{.01, .01}};
    }
  const auto narrow =
    redistribute(bg, MappingQ1<3>(), shrinking, 0., MPI_COMM_WORLD);
  EXPECT_EQ(Utilities::MPI::sum(
              static_cast<unsigned int>(narrow.vessels.size()), MPI_COMM_WORLD),
            1u);
  EXPECT_EQ(owned_vessels(narrow), 1u);
}
TEST(VascularDistribution,
     BOTH_AuthoritativeNodeMotionRefreshesStaleGhostsAndVesselOwnership)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  stripes(bg);
  auto initial = redistribute(
    bg, MappingQ1<3>(), segment(.2, .3, size() - 1), 9., MPI_COMM_WORLD);
  EXPECT_EQ(initial.nodes.size(), 2u);
  if (initial.nodes.count(base + 1) &&
      initial.nodes.at(base + 1).owner == rank())
    initial.nodes.at(base + 1).position = Point<3>(7.6, .5, .5);
  const auto output =
    redistribute(bg, MappingQ1<3>(), initial, 0., MPI_COMM_WORLD);
  const auto expected_v = owner(bg, Point<3>(3.9, .5, .5), size() - 1),
             expected_b = owner(bg, Point<3>(7.6, .5, .5), size() - 1);
  EXPECT_EQ(owned_vessels(output), 1u);
  ASSERT_EQ(output.vessels.size(), 1u);
  EXPECT_EQ(output.nodes.at(base + 1).position, Point<3>(7.6, .5, .5));
  EXPECT_EQ(output.nodes.at(base + 1).owner, expected_b);
  EXPECT_EQ(output.vessels.at(base).owner, expected_v);
  EXPECT_EQ(output.terminals.at(base + 2).owner, expected_b);
  EXPECT_NEAR(length(output, base), 7.4, 1.e-13);
}
TEST(VascularDistribution, BOTH_InterfaceMidpointsUseTheSmallestContainingRank)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  stripes(bg);
  const auto expected = owner(bg, Point<3>(4., .5, .5), size() - 1);
  const auto output   = redistribute(
    bg, MappingQ1<3>(), segment(3.9, 4.1, size() - 1), 0., MPI_COMM_WORLD);
  EXPECT_EQ(owned_vessels(output), 1u);
  if (output.vessels.count(base))
    EXPECT_EQ(output.vessels.at(base).owner, expected);
}
TEST(VascularDistribution, BOTH_EmptyBackgroundRanksAndEmptyNetworkParticipate)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  const auto empty = redistribute(bg, MappingQ1<3>(), {}, 0., MPI_COMM_WORLD);
  EXPECT_TRUE(empty.nodes.empty());
  EXPECT_TRUE(empty.vessels.empty());
  const auto output = redistribute(
    bg, MappingQ1<3>(), segment(.2, .3, size() - 1), 0., MPI_COMM_WORLD);
  EXPECT_EQ(owned_vessels(output), 1u);
  if (bg.n_locally_owned_active_cells() == 0)
    EXPECT_TRUE(output.vessels.empty());
}
TEST(VascularDistribution,
     BOTH_InvalidOwnershipFailsCollectivelyWithoutPartialResults)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  stripes(bg);
  EXPECT_THROW(redistribute(bg,
                            MappingQ1<3>(),
                            {},
                            rank() == 0 ? -1. : 0.,
                            MPI_COMM_WORLD),
               ExceptionBase);
  auto bad = segment(.2, .3, 0);
  if (rank() == 0)
    bad.vessels.at(base).viscosity = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(redistribute(bg, MappingQ1<3>(), bad, 0., MPI_COMM_WORLD),
               ExceptionBase);
  if (size() > 1)
    {
      NetworkData duplicate;
      duplicate.nodes[base].position = Point<3>(rank() + .5, .5, .5);
      duplicate.nodes[base].owner    = rank();
      EXPECT_THROW(
        redistribute(bg, MappingQ1<3>(), duplicate, 0., MPI_COMM_WORLD),
        ExceptionBase);
      auto orphan = segment(.2, .3, 1);
      if (rank() == 1)
        for (auto &[id, n] : orphan.nodes)
          n.owner = 0;
      EXPECT_THROW(redistribute(bg, MappingQ1<3>(), orphan, 0., MPI_COMM_WORLD),
                   ExceptionBase);
    }
}
TEST(VascularDistribution,
     BOTH_GhostEditsDoNotReplaceOwnedGeometryOrPhysicalData)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  stripes(bg);
  auto input = redistribute(
    bg, MappingQ1<3>(), segment(.2, .3, size() - 1), 9., MPI_COMM_WORLD);
  for (auto &[id, n] : input.nodes)
    if (n.owner != rank())
      n.position += Point<3>(.01, .1, 0.);
  for (auto &[id, v] : input.vessels)
    if (v.owner != rank())
      {
        v.radii                          = {{.8, .8}};
        v.physical_properties["density"] = 5.;
      }
  for (auto &[id, t] : input.terminals)
    if (t.owner != rank())
      t.demand = 1000.;
  const auto output =
    redistribute(bg, MappingQ1<3>(), input, 9., MPI_COMM_WORLD);
  EXPECT_EQ(owned_vessels(output), 1u);
  ASSERT_EQ(output.vessels.size(), 1u);
  EXPECT_EQ(output.nodes.at(base).position, Point<3>(.2, .5, .5));
  EXPECT_EQ(output.nodes.at(base + 1).position, Point<3>(.3, .5, .5));
  EXPECT_EQ(output.vessels.at(base).radii, (std::array<double, 2>{{.01, .02}}));
  EXPECT_DOUBLE_EQ(output.vessels.at(base).physical_properties.at("density"),
                   1050.);
  EXPECT_DOUBLE_EQ(output.terminals.at(base + 2).demand, 7.);
}
TEST(VascularDistribution,
     BOTH_TetrahedralBoundingBoxesDoNotDeterminePointOwnership)
{
  Triangulation<3> serial;
  GridGenerator::reference_cell(serial, ReferenceCells::Tetrahedron);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<3> bg(MPI_COMM_WORLD);
  bg.create_triangulation(description);
  NetworkData input;
  const auto  source = size() - 1;
  if (rank() == source)
    {
      for (unsigned int i = 0; i < 2; ++i)
        {
          const Id     a = base + 10 * i, b = a + 1;
          const double coordinate = i == 0 ? .1 : .7;
          input.nodes[a].position =
            Point<3>(coordinate, coordinate, coordinate);
          input.nodes[a].owner = source;
          input.nodes[a].port  = Port::arterial_inlet;
          input.nodes[b].position =
            Point<3>(coordinate + .1, coordinate, coordinate);
          input.nodes[b].owner = source;
          Vessel v;
          v.nodes          = {{a, b}};
          v.radii          = {{.01, .01}};
          v.viscosity      = .004;
          v.owner          = source;
          input.vessels[a] = v;
        }
    }
  const auto output = redistribute(
    bg, MappingFE<3>(FE_SimplexP<3>(1)), input, 0., MPI_COMM_WORLD);
  EXPECT_EQ(owned_vessels(output), 2u);
  if (rank() == 0)
    {
      ASSERT_EQ(output.vessels.count(base), 1u);
      EXPECT_EQ(output.vessels.at(base).owner, 0u);
      EXPECT_EQ(output.nodes.at(base).owner, 0u);
    }
  if (rank() == source)
    {
      ASSERT_EQ(output.vessels.count(base + 10), 1u);
      EXPECT_EQ(output.vessels.at(base + 10).owner, source);
      EXPECT_EQ(output.nodes.at(base + 10).owner, source);
    }
}
