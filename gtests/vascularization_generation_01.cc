// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <set>

using namespace dealii;
using namespace ImmersX::Vascularization;

namespace
{
  void
  synthetic(Parameters &p)
  {
    p.seed_position  = Point<3>(.3, .5, .5);
    p.seed_direction = Point<3>(1., 0., 0.);
    p.seed_radius    = .01;
    p.seed_length    = .4;
    p.viscosity      = .004;
  }
} // namespace

TEST(VascularGeneration, CubeSeedAnalyticalGate)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.refine_global(2);
  const MappingQ1<3> mapping;
  Parameters         p;
  synthetic(p);
  const NetworkData initial;
  const auto result = generate(background, mapping, initial, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  ASSERT_EQ(result.network.vessels.size(), 1u);
  ASSERT_EQ(result.network.terminals.size(), 1u);
  EXPECT_EQ(result.network.nodes.at(0).port, Port::arterial_inlet);
  EXPECT_DOUBLE_EQ(result.network.vessels.at(0).radii[0], p.seed_radius);
  EXPECT_DOUBLE_EQ(result.network.vessels.at(0).radii[1], p.seed_radius);
  EXPECT_NEAR(volume(result.network, 0) / (numbers::PI * .01 * .01 * .4),
              1.,
              1.e-14);
  EXPECT_NEAR(resistance(result.network, 0) /
                (8. * .004 * .4 / (numbers::PI * 1.e-8)),
              1.,
              1.e-14);
  EXPECT_TRUE(initial.nodes.empty());
  EXPECT_TRUE(initial.vessels.empty());
}

TEST(VascularGeneration, RejectsWallEscapeWithInteriorCentreline)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.seed_position[1] = .005;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_TRUE(result.network.vessels.empty());
}

TEST(VascularGeneration, NonconvexDomainRejectsSegmentBetweenInteriorEndpoints)
{
  Triangulation<3> full, background;
  GridGenerator::subdivided_hyper_rectangle(full,
                                            {3, 3, 1},
                                            Point<3>(0., 0., 0.),
                                            Point<3>(3., 3., 1.));
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
  p.seed_position = Point<3>(.5, 2.5, .5);
  p.seed_length   = 2.;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_TRUE(result.network.vessels.empty());
}

TEST(VascularGeneration, TetrahedralBoundaryBoxIsOnlyBroadPhase)
{
  Triangulation<3> background;
  GridGenerator::reference_cell(background, ReferenceCells::Tetrahedron);
  const MappingFE<3> mapping(FE_SimplexP<3>(1));
  Parameters         p;
  synthetic(p);
  p.seed_position   = Point<3>(.15, .15, .15);
  p.seed_length     = .1;
  p.seed_radius     = .02;
  const auto result = generate(background, mapping, {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_NEAR(length(result.network, 0), .1, 1.e-14);
}

TEST(VascularGeneration, DeterministicInteriorDirectionAndInvalidPhysiology)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.seed_position  = Point<3>(.5, .5, .5);
  p.seed_direction = Point<3>();
  p.seed_length    = .1;
  const auto a = generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  const auto b = generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(a.validation.valid());
  ASSERT_TRUE(b.validation.valid());
  EXPECT_EQ(a.network.nodes.at(1).position, b.network.nodes.at(1).position);
  p.seed_radius = 0.;
  EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD),
               ExceptionBase);
}

TEST(VascularGeneration, BilinearWallContainment)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.begin_active()->vertex(7)[2] = 1.4;
  Parameters p;
  synthetic(p);
  p.seed_position = Point<3>(.6, .6, 1.05);
  p.seed_length   = .1;
  p.seed_radius   = .02;
  // The warped upper face is z=1+0.4*x*y. Its bounding box overlaps
  // both proposals, while only the second proposal's wall reaches the face.
  const auto accepted =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_TRUE(accepted.validation.valid());
  p.seed_position[2] = 1.14;
  const auto rejected =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(rejected.validation.valid());
  EXPECT_TRUE(rejected.network.vessels.empty());
}

TEST(VascularGeneration, ExhaustedGeometryBudgetDoesNotAccept)
{
  Triangulation<3> background;
  GridGenerator::reference_cell(background, ReferenceCells::Tetrahedron);
  const MappingFE<3> mapping(FE_SimplexP<3>(1));
  Parameters         p;
  synthetic(p);
  p.seed_position     = Point<3>(.15, .15, .15);
  p.seed_length       = .1;
  p.seed_radius       = .02;
  p.geometry_work     = 1;
  const auto rejected = generate(background, mapping, {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(rejected.validation.valid());
  EXPECT_TRUE(rejected.network.vessels.empty());
  p.geometry_work     = 100000;
  const auto accepted = generate(background, mapping, {}, p, MPI_COMM_WORLD);
  EXPECT_TRUE(accepted.validation.valid());
}
