// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <cmath>
#include <limits>
#include <set>

using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  void
  synthetic(Parameters &p)
  {
    p.seed_position           = Point<3>(0., .5, .5);
    p.seed_direction          = Point<3>(1., 0., 0.);
    p.seed_inward_port_normal = Point<3>(2., 0., 0.);
    p.seed_radius             = .01;
    p.seed_length             = .3;
    p.viscosity               = .004;
    p.clearance               = .002;
  }
} // namespace

TEST(VascularSurfacePorts, ExplicitPlaneOpensOnlyTheDeclaredSeedSection)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  const auto open = generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(open.validation.valid());
  EXPECT_EQ(open.network.nodes.at(0).position, p.seed_position);
  EXPECT_EQ(open.network.nodes.at(0).port, Port::arterial_inlet);
  EXPECT_EQ(open.network.nodes.at(0).inward_port_normal, Point<3>(1., 0., 0.));
  EXPECT_EQ(open.network.nodes.size(), 2u);
  EXPECT_NEAR(open.tissue.inside_vessel_volume / (numbers::PI * .01 * .01 * .3),
              1.,
              1.e-13);
  for (const double scale : {1.e-200, 1.e200})
    {
      p.seed_inward_port_normal = Point<3>(scale, 0., 0.);
      const auto scaled =
        generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
      ASSERT_TRUE(scaled.validation.valid());
      EXPECT_EQ(scaled.network.nodes.at(0).inward_port_normal,
                Point<3>(1., 0., 0.));
    }
  p.seed_inward_port_normal = Point<3>();
  const auto closed =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(closed.validation.valid());
  EXPECT_TRUE(closed.network.vessels.empty());
}

TEST(VascularSurfacePorts,
     DeclarationCannotExcuseSidewallOrOppositeBoundaryEscape)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.seed_position[1] = .005;
  EXPECT_FALSE(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD)
                 .validation.valid());
  p.seed_position[1] = .5;
  p.seed_length      = 1.;
  EXPECT_FALSE(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD)
                 .validation.valid());
  p.seed_length = 1.2;
  EXPECT_FALSE(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD)
                 .validation.valid());
  p.seed_length = .005;
  EXPECT_FALSE(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD)
                 .validation.valid());
}

TEST(VascularSurfacePorts, InwardDirectionAndActualSectionAreRequired)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.seed_inward_port_normal = Point<3>(-1., 0., 0.);
  EXPECT_FALSE(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD)
                 .validation.valid());
  p.seed_inward_port_normal = Point<3>(1., 0., 0.);
  p.seed_direction          = Point<3>(0., 1., 0.);
  EXPECT_FALSE(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD)
                 .validation.valid());
  p.seed_direction = Point<3>(-1., 0., 0.);
  EXPECT_FALSE(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD)
                 .validation.valid());
  p.seed_direction   = Point<3>(1., 0., 0.);
  p.seed_position[0] = .1;
  EXPECT_FALSE(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD)
                 .validation.valid());
  p.seed_position[0]        = 0.;
  p.seed_inward_port_normal = Point<3>(1., .1, 0.);
  EXPECT_FALSE(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD)
                 .validation.valid());
  p.seed_inward_port_normal = Point<3>(1., 0., 0.);
  p.seed_direction          = Point<3>(1., 1., 0.);
  EXPECT_TRUE(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD)
                .validation.valid());
}

TEST(VascularSurfacePorts, CCORechecksInteriorJunctionsAndKeepsTheFixedPort)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.refine_global(2);
  Parameters p;
  synthetic(p);
  p.terminal_count = 12;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_EQ(result.territories.size(), 12u);
  EXPECT_EQ(result.network.nodes.at(0).position, p.seed_position);
  EXPECT_EQ(result.network.nodes.at(0).inward_port_normal,
            Point<3>(1., 0., 0.));
  EXPECT_EQ(result.network.vessels.at(0).radii,
            (std::array<double, 2>{{.01, .01}}));
  for (const auto &[id, e] : result.network.vessels)
    {
      const auto   a      = result.network.nodes.at(e.nodes[0]).position;
      const auto   b      = result.network.nodes.at(e.nodes[1]).position;
      const double radius = e.radii[0] + p.clearance;
      EXPECT_GT(b[0], radius);
      if (e.nodes[0] != 0)
        EXPECT_GT(a[0], radius);
      EXPECT_LT(std::max(a[0], b[0]) + radius, 1.);
      for (unsigned int d = 1; d < 3; ++d)
        {
          EXPECT_GT(std::min(a[d], b[d]), radius);
          EXPECT_LT(std::max(a[d], b[d]) + radius, 1.);
        }
    }
}

TEST(VascularSurfacePorts, ImportedPortGeometryRemainsFrozenOnCompletion)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.terminal_count = 2;
  const auto original =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(original.validation.valid());
  p.seed_direction = Point<3>();
  p.seed_length    = .05;
  const auto completed =
    generate(background, MappingQ1<3>(), original.network, p, MPI_COMM_WORLD);
  ASSERT_TRUE(completed.validation.valid());
  EXPECT_EQ(completed.attachment_roots.size(), 2u);
  for (const auto &[id, node] : original.network.nodes)
    {
      const auto &retained = completed.network.nodes.at(id);
      EXPECT_EQ(retained.position, node.position);
      EXPECT_EQ(retained.port, node.port);
      EXPECT_EQ(retained.inward_port_normal, node.inward_port_normal);
    }
  for (const auto &[id, e] : original.network.vessels)
    {
      EXPECT_EQ(completed.network.vessels.at(id).nodes, e.nodes);
      EXPECT_EQ(completed.network.vessels.at(id).radii, e.radii);
    }
}

TEST(VascularSurfacePorts, ObliqueTetrahedralFaceAndRotatedHexahedralFace)
{
  Triangulation<3> tetrahedron;
  GridGenerator::reference_cell(tetrahedron, ReferenceCells::Tetrahedron);
  Parameters p;
  synthetic(p);
  p.seed_position           = Point<3>(1. / 3., 1. / 3., 1. / 3.);
  p.seed_direction          = Point<3>(-1., -1., -1.);
  p.seed_inward_port_normal = p.seed_direction;
  p.seed_length             = .1;
  const auto tet            = generate(
    tetrahedron, MappingFE<3>(FE_SimplexP<3>(1)), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(tet.validation.valid());
  EXPECT_NEAR(tet.network.nodes.at(0).inward_port_normal.norm(), 1., 1.e-14);
  EXPECT_LT(tet.network.nodes.at(1).position[0], 1. / 3.);
  Triangulation<3> hexahedron;
  GridGenerator::hyper_cube(hexahedron);
  const auto rotate = [](const Point<3> &x) {
    const double c = std::cos(.6), s = std::sin(.6);
    return Point<3>(c * x[0] - s * x[1], s * x[0] + c * x[1], x[2]);
  };
  GridTools::transform(rotate, hexahedron);
  synthetic(p);
  p.seed_position           = rotate(p.seed_position);
  p.seed_direction          = rotate(Point<3>(1., 0., 0.));
  p.seed_inward_port_normal = p.seed_direction;
  const auto hex = generate(hexahedron, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(hex.validation.valid());
  EXPECT_NEAR(hex.network.nodes.at(0).inward_port_normal.distance(
                p.seed_direction),
              0.,
              1.e-14);
  EXPECT_NEAR(length(hex.network, 0), .3, 1.e-13);
}

TEST(VascularSurfacePorts, WarpedBoundaryIsNotFlattenedIntoAPlanarPort)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.begin_active()->vertex(7)[2] = 1.4;
  Parameters p;
  synthetic(p);
  EXPECT_TRUE(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD)
                .validation.valid());
  p.seed_position           = Point<3>(.5, .5, 1.1);
  p.seed_inward_port_normal = Point<3>(.2, .2, -1.);
  p.seed_direction          = p.seed_inward_port_normal;
  p.seed_length             = .1;
  EXPECT_FALSE(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD)
                 .validation.valid());
}

TEST(VascularSurfacePorts, PortPlaneDoesNotHideANonconvexHole)
{
  Triangulation<3> full, background;
  GridGenerator::subdivided_hyper_rectangle(full,
                                            {5, 3, 1},
                                            Point<3>(0., 0., 0.),
                                            Point<3>(5., 3., 1.));
  std::set<Triangulation<3>::active_cell_iterator> remove;
  for (const auto &cell : full.active_cell_iterators())
    if (cell->center()[0] > 1. && cell->center()[0] < 2. &&
        cell->center()[1] > 1.)
      remove.insert(cell);
  GridGenerator::create_triangulation_with_removed_cells(full,
                                                         remove,
                                                         background);
  Parameters p;
  synthetic(p);
  p.seed_position = Point<3>(0., 2.5, .5);
  p.seed_length   = 4.5;
  // Start, midpoint and end are in the background; the segment still crosses
  // the removed column between x=1 and x=2.
  const auto rejected =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(rejected.validation.valid());
  EXPECT_TRUE(rejected.network.vessels.empty());
}

TEST(VascularSurfacePorts, PhysicalPortNormalsRequireRolesAndFiniteData)
{
  NetworkData n;
  n.nodes[0].position           = Point<3>(0., .5, .5);
  n.nodes[0].inward_port_normal = Point<3>(1., 0., 0.);
  EXPECT_FALSE(validate_data(n, 1).valid());
  for (const auto role :
       {Port::arterial_inlet, Port::arterial_outlet, Port::venous_outlet})
    {
      n.nodes[0].port = role;
      EXPECT_TRUE(validate_data(n, 1).valid());
    }
  n.nodes[0].inward_port_normal[0] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(validate_data(n, 1).valid());
  Parameters p;
  synthetic(p);
  p.seed_inward_port_normal[1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(p.validate(), ExceptionBase);
}
