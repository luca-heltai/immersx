// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <limits>

using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  void
  terminal(NetworkData &n, Point<3> position, System system = System::arterial)
  {
    const Id id          = n.nodes.size();
    n.nodes[id].position = position;
    n.terminals[id]      = {id, invalid_id, system, 0, 1.};
  }

  TissueMetrics
  measure(const Triangulation<3> &background,
          const Mapping<3>       &mapping,
          const NetworkData      &n,
          double                  radius,
          unsigned int            order = 2)
  {
    return measure_tissue(
      background,
      mapping,
      n,
      discretize(background, mapping, n, {}, MPI_COMM_WORLD),
      {radius, order},
      MPI_COMM_WORLD);
  }
} // namespace

TEST(VascularMetrics, ImportedTaperVolumeExcludesExteriorArclength)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  NetworkData n;
  n.nodes[0].position = Point<3>(-.2, .3, .4);
  n.nodes[1].position = Point<3>(.4, .3, .4);
  Vessel e;
  e.nodes               = {{0, 1}};
  e.radii               = {{.01, .02}};
  e.viscosity           = .004;
  n.vessels[0]          = e;
  const auto   metrics  = measure(background, MappingQ1<3>(), n, 0.);
  const double r        = .01 + .01 / 3.;
  const double expected = numbers::PI * .4 * (r * r + r * .02 + .02 * .02) / 3.;
  EXPECT_NEAR(metrics.background_volume, 1., 1.e-13);
  EXPECT_NEAR(metrics.inside_vessel_volume / expected, 1., 1.e-13);
  EXPECT_NEAR(metrics.volume_fraction / expected, 1., 1.e-13);
  EXPECT_FALSE(metrics.coverage_evaluated);
}

TEST(VascularMetrics, NearestTerminalAssignmentDoesNotImplyFullCoverage)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.refine_global(1);
  NetworkData n;
  terminal(n, Point<3>(.25, .25, .25));
  const auto small = measure(background, MappingQ1<3>(), n, .1, 1);
  EXPECT_TRUE(small.coverage_evaluated);
  EXPECT_NEAR(small.arterial_coverage, 1. / 8., 1.e-13);
  EXPECT_DOUBLE_EQ(small.venous_coverage, 0.);
  EXPECT_DOUBLE_EQ(small.dual_coverage, 0.);
  const auto large = measure(background, MappingQ1<3>(), n, 2., 1);
  EXPECT_NEAR(large.arterial_coverage, 1., 1.e-13);
  EXPECT_DOUBLE_EQ(large.volume_fraction, small.volume_fraction);
  const auto off = measure(background, MappingQ1<3>(), n, 0., 1);
  EXPECT_FALSE(off.coverage_evaluated);
  EXPECT_DOUBLE_EQ(off.arterial_coverage, 0.);
}

TEST(VascularMetrics, ArterialVenousAndJointCoverageUseServiceUnions)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.refine_global(1);
  NetworkData n;
  for (const double y : {.25, .75})
    for (const double z : {.25, .75})
      terminal(n, Point<3>(.25, y, z));
  for (const double x : {.25, .75})
    for (const double z : {.25, .75})
      terminal(n, Point<3>(x, .25, z), System::venous);
  const auto result = measure(background, MappingQ1<3>(), n, .01, 1);
  EXPECT_NEAR(result.arterial_coverage, .5, 1.e-13);
  EXPECT_NEAR(result.venous_coverage, .5, 1.e-13);
  EXPECT_NEAR(result.dual_coverage, .25, 1.e-13);
  // Duplicated service centres change no union measure or 1D volume.
  terminal(n, Point<3>(.25, .25, .25));
  const auto duplicate = measure(background, MappingQ1<3>(), n, .01, 1);
  EXPECT_DOUBLE_EQ(duplicate.arterial_coverage, result.arterial_coverage);
  EXPECT_DOUBLE_EQ(duplicate.dual_coverage, result.dual_coverage);
  EXPECT_DOUBLE_EQ(duplicate.inside_vessel_volume, 0.);
}

TEST(VascularMetrics, LocalRefinementUsesPhysicalVolumeInsteadOfPointCounts)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.refine_global(1);
  for (const auto &cell : background.active_cell_iterators())
    if (cell->center()[0] < .5)
      cell->set_refine_flag();
  background.execute_coarsening_and_refinement();
  NetworkData n;
  terminal(n, Point<3>(.25, .5, .5));
  const auto result = measure(background, MappingQ1<3>(), n, .61, 1);
  // All 32 fine-cell midpoints on the left are served. All four coarse-cell
  // midpoints on the right are farther than .61. Each half has volume .5.
  EXPECT_EQ(background.n_active_cells(), 36u);
  EXPECT_NEAR(result.background_volume, 1., 1.e-13);
  EXPECT_NEAR(result.arterial_coverage, .5, 1.e-13);
  EXPECT_GT(32. / 36., result.arterial_coverage + .3);
}

TEST(VascularMetrics, SimplexAndWarpedHexahedronHaveAnalyticalVolumes)
{
  Triangulation<3>         tetrahedron;
  std::vector<CellData<3>> cells(1, CellData<3>(4));
  cells[0].vertices = {0, 1, 2, 3};
  tetrahedron.create_triangulation({Point<3>(0., 0., 0.),
                                    Point<3>(1., 0., 0.),
                                    Point<3>(0., 1., 0.),
                                    Point<3>(0., 0., 1.)},
                                   cells,
                                   SubCellData());
  NetworkData n;
  terminal(n, Point<3>(.25, .25, .25));
  const auto tet = measure(tetrahedron, MappingFE<3>(FE_SimplexP<3>(1)), n, 2.);
  EXPECT_NEAR(tet.background_volume, 1. / 6., 1.e-13);
  EXPECT_NEAR(tet.arterial_coverage, 1., 1.e-13);
  Triangulation<3> hexahedron;
  GridGenerator::hyper_cube(hexahedron);
  hexahedron.begin_active()->vertex(7)[2] = 1.4;
  const auto hex = measure(hexahedron, MappingQ1<3>(), n, 2.);
  // Integral of z_top=1+.4*x*y over the unit square.
  EXPECT_NEAR(hex.background_volume, 1.1, 1.e-13);
  EXPECT_NEAR(hex.arterial_coverage, 1., 1.e-13);
}

TEST(VascularMetrics, PortsAreNotServiceCentres)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  NetworkData n;
  n.nodes[0].position = Point<3>(.5, .5, .5);
  n.nodes[0].port     = Port::arterial_inlet;
  n.nodes[1].position = Point<3>(.6, .5, .5);
  n.nodes[1].port     = Port::venous_outlet;
  const auto result   = measure(background, MappingQ1<3>(), n, 10.);
  EXPECT_TRUE(result.coverage_evaluated);
  EXPECT_DOUBLE_EQ(result.arterial_coverage, 0.);
  EXPECT_DOUBLE_EQ(result.venous_coverage, 0.);
}

TEST(VascularMetrics, GenerationReportsIndependentVolumeAndCoverage)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  p.seed_position           = Point<3>(.2, .5, .5);
  p.seed_direction          = Point<3>(1., 0., 0.);
  p.seed_radius             = .01;
  p.seed_length             = .3;
  p.viscosity               = .004;
  p.coverage.service_radius = 2.;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_NEAR(result.tissue.volume_fraction / (numbers::PI * .01 * .01 * .3),
              1.,
              1.e-13);
  EXPECT_TRUE(result.tissue.coverage_evaluated);
  EXPECT_NEAR(result.tissue.arterial_coverage, 1., 1.e-13);
  EXPECT_DOUBLE_EQ(result.tissue.venous_coverage, 0.);
}

TEST(VascularMetrics, VolumeDoesNotDependOnCoverageQuadratureOrder)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  for (const auto &cell : background.active_cell_iterators())
    for (const auto v : cell->vertex_indices())
      {
        const auto p = cell->vertex(v);
        cell->vertex(v) =
          Point<3>(p[0] + .4 * p[1] * p[2], p[1] + .4 * p[0] * p[2], p[2]);
      }
  NetworkData n;
  terminal(n, Point<3>(.25, .25, .25));
  n.nodes[1].position = Point<3>(.2, .2, .4);
  n.nodes[2].position = Point<3>(.4, .2, .4);
  Vessel e;
  e.nodes             = {{1, 2}};
  e.radii             = {{.01, .01}};
  e.viscosity         = .004;
  n.vessels[0]        = e;
  const auto midpoint = measure(background, MappingQ1<3>(), n, 2., 1);
  const auto gauss    = measure(background, MappingQ1<3>(), n, 2., 2);
  // x=u+.4*v*w,y=v+.4*u*w,z=w has determinant 1-.16*w^2.
  const double exact_volume = 1. - .16 / 3.;
  EXPECT_NEAR(midpoint.background_volume, exact_volume, 1.e-13);
  EXPECT_NEAR(gauss.background_volume, exact_volume, 1.e-13);
  EXPECT_NEAR(midpoint.coverage_quadrature_volume, 1. - .16 / 4., 1.e-13);
  EXPECT_NEAR(gauss.coverage_quadrature_volume, exact_volume, 1.e-13);
  EXPECT_NEAR(midpoint.arterial_coverage, 1., 1.e-13);
  EXPECT_NEAR(gauss.arterial_coverage, 1., 1.e-13);
  EXPECT_DOUBLE_EQ(midpoint.volume_fraction, gauss.volume_fraction);
  EXPECT_NEAR(midpoint.volume_fraction /
                (numbers::PI * .01 * .01 * .2 / exact_volume),
              1.,
              1.e-13);
}

TEST(VascularMetrics, InvalidControlsAndDataAreRejected)
{
  CoverageParameters p;
  p.service_radius = -1.;
  EXPECT_THROW(p.validate(), ExceptionBase);
  p.service_radius = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(p.validate(), ExceptionBase);
  p.service_radius   = .1;
  p.quadrature_order = 0;
  EXPECT_THROW(p.validate(), ExceptionBase);
  p.quadrature_order = 33;
  EXPECT_THROW(p.validate(), ExceptionBase);
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  DiscretizationResult invalid;
  invalid.validation.issues.push_back(
    {"discretization", invalid_id, "Invalid input."});
  EXPECT_THROW(
    measure_tissue(background, MappingQ1<3>(), {}, invalid, {}, MPI_COMM_WORLD),
    ExceptionBase);
}
