// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>

#include <gtest/gtest.h>
#include <immersx/grid/lift_co_refinement.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  struct Configuration
  {
    DomainParameters<3>           solid{"/Lift refinement/solid/"};
    DomainParameters<1, 3>        fluid{"/Lift refinement/fluid/"};
    TensorProductLift<1, 2, 3, 3> lift{"/Lift refinement/geometry/"};
    RefinementParameters<1>       refinement{"/Lift refinement/policy/"};

    Configuration()
    {
      solid.name_of_grid                 = "hyper_cube";
      solid.arguments_for_grid           = "-0.5:0.5:false";
      solid.initial_refinement           = 0;
      solid.triangulation_type           = "distributed";
      fluid.name_of_grid                 = "hyper_cube";
      fluid.arguments_for_grid           = "-0.4:0.4:false";
      fluid.initial_refinement           = 0;
      fluid.triangulation_type           = "fullydistributed";
      lift.section.inclusion_degree      = 1;
      lift.section.selected_coefficients = {3, 7};
      lift.section.refinement_level      = 0;
      lift.section.n_q_points            = 2;
      lift.representative_n_q_points     = 2;
      lift.representative_n_repetitions  = 1;
      refinement.refinement_strategy     = "both";
      refinement.lower_factor            = .5;
      refinement.upper_factor            = 2.;
      refinement.max_refinement_level    = 6;
    }
  };

  void
  prepare_solid(Domain<3> &solid)
  {
    solid.prepare_grid();
    solid.refinement_triangulation().set_mesh_smoothing(
      Triangulation<3>::limit_level_difference_at_vertices);
  }
} // namespace

TEST(LiftCoRefinement, BOTH_RefinesSolidAndFluid)
{
  ParameterAcceptor::clear();
  Configuration p;
  Domain<3>     solid(p.solid);
  Domain<1, 3>  fluid(p.fluid);
  prepare_solid(solid);
  const auto report = co_refine_lifted_domains(
    solid,
    fluid,
    p.lift,
    p.refinement,
    [](const CellId &, const Point<1> &) { return .08; },
    4);
  EXPECT_TRUE(report.satisfied());
  EXPECT_EQ(report.initial_solid_cells, 1u);
  EXPECT_EQ(report.initial_fluid_cells, 1u);
  EXPECT_GT(report.solid_cells, 1u);
  EXPECT_GT(report.fluid_cells, 1u);
  EXPECT_GT(report.solid_passes, 0u);
  EXPECT_GT(report.fluid_passes, 0u);
  for (unsigned int d = 0; d < 2; ++d)
    {
      EXPECT_GT(report.minimum_ratio[d], p.refinement.lower_factor);
      EXPECT_LT(report.maximum_ratio[d], p.refinement.upper_factor);
    }
  TensorProductLiftSupport<1, 2, 3, 3> support(p.lift.parameters());
  const auto                           expected_points =
    report.fluid_cells * support.representative_quadrature().size() *
    support.reference_cross_section().n_quadrature_points();
  // The Cartesian fixture has no points on cell faces: exactly one physical
  // solid cell is reached by each point, not a radius-expanded neighborhood.
  EXPECT_EQ(report.point_cell_pairs, expected_points);
  unsigned int minimum = std::numeric_limits<unsigned int>::max(), maximum = 0;
  for (const auto &cell : solid.triangulation().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        minimum = std::min(minimum, static_cast<unsigned int>(cell->level()));
        maximum = std::max(maximum, static_cast<unsigned int>(cell->level()));
      }
  EXPECT_LT(Utilities::MPI::min(minimum, MPI_COMM_WORLD),
            Utilities::MPI::max(maximum, MPI_COMM_WORLD));
}

TEST(LiftCoRefinement, BOTH_PointSelectionHasNoRadiusPadding)
{
  ParameterAcceptor::clear();
  Configuration p;
  p.fluid.arguments_for_grid               = "-0.1:0.1:false";
  p.refinement.refinement_strategy         = "space";
  p.refinement.lower_factor                = .2;
  p.refinement.upper_factor                = 4.;
  p.refinement.space_pre_refinement_cycles = 2;
  p.refinement.max_refinement_level        = 3;
  Domain<3>    solid(p.solid);
  Domain<1, 3> fluid(p.fluid);
  prepare_solid(solid);
  fluid.prepare_grid(true);
  Tensor<1, 3> translation;
  translation[1] = translation[2] = .1;
  GridTools::shift(translation, fluid.refinement_triangulation());
  const auto report = co_refine_lifted_domains(
    solid,
    fluid,
    p.lift,
    p.refinement,
    [](const CellId &, const Point<1> &) { return .1; },
    4);
  EXPECT_TRUE(report.satisfied());
  // All wall points have y,z strictly between 0 and .25, and x in (-.1,.1).
  // Only two of the 64 level-2 cells contain points; each adds seven cells.
  EXPECT_EQ(report.solid_cells, 64u + 2u * 7u);
  EXPECT_EQ(report.fluid_cells, 1u);
}

TEST(LiftCoRefinement, BOTH_SectionAndAxialControlsAreIndependent)
{
  ParameterAcceptor::clear();
  Configuration p;
  p.solid.arguments_for_grid               = "-1:1:false";
  p.fluid.arguments_for_grid               = "-0.5:0.5:false";
  p.refinement.refinement_strategy         = "embedded";
  p.refinement.space_pre_refinement_cycles = 4;
  Domain<3>    solid(p.solid);
  Domain<1, 3> fluid(p.fluid);
  prepare_solid(solid);
  const auto report = co_refine_lifted_domains(
    solid,
    fluid,
    p.lift,
    p.refinement,
    [](const CellId &, const Point<1> &) { return .8; },
    4);
  EXPECT_TRUE(report.satisfied());
  EXPECT_EQ(report.solid_passes, 0u);
  EXPECT_GT(report.fluid_passes, 0u);
  EXPECT_GT(report.section_passes, 0u);
  EXPECT_GT(report.section_level, 0u);
  EXPECT_EQ(report.solid_cells, 4096u);
}

TEST(LiftCoRefinement, BOTH_ReportsCapsAndQuadratureScale)
{
  ParameterAcceptor::clear();
  Configuration p;
  p.refinement.max_refinement_level = 0;
  Domain<3>    solid(p.solid);
  Domain<1, 3> fluid(p.fluid);
  const auto   report = co_refine_lifted_domains(
    solid,
    fluid,
    p.lift,
    p.refinement,
    [](const CellId &, const Point<1> &) { return .08; },
    0);
  EXPECT_FALSE(report.satisfied());
  EXPECT_EQ(report.solid_cells, 1u);
  EXPECT_EQ(report.fluid_cells, 1u);
  EXPECT_GT(report.lower_violations[0], 0u);
  EXPECT_GT(report.lower_violations[1], 0u);
  EXPECT_EQ(report.outside_points, 0u);
  // One Gauss-2 section point carries half a chord of a unit-square circle.
  const double theta = .08 * std::sqrt(2.) / 2.;
  EXPECT_NEAR(report.minimum_ratio[1], theta / std::sqrt(3.), 1.e-13);
  EXPECT_NEAR(report.maximum_ratio[1], theta / std::sqrt(3.), 1.e-13);
  EXPECT_NEAR(report.minimum_ratio[0], .8 / std::sqrt(3.), 1.e-13);
}

#ifdef DEAL_II_WITH_VTK
TEST(LiftCoRefinement, BOTH_TransfersFieldsAndRefinesFluidLocally)
{
  ParameterAcceptor::clear();
  Configuration p;
#  ifdef DEBUG
  const std::string mode = "debug";
#  else
  const std::string mode = "release";
#  endif
  const auto filename = TestPaths::output_path(
    "lift-co-refinement/fields-" + mode + "-" +
    std::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + ".vtk");
  const double left_area  = numbers::PI * .07 * .07;
  const double right_area = numbers::PI * .11 * .11;
  bool         written    = true;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::error_code error;
      std::filesystem::create_directories(filename.parent_path(), error);
      written = !error;
      std::ofstream file(filename);
      file.precision(17);
      file
        << "# vtk DataFile Version 3.0\nRetained fields\nASCII\n"
           "DATASET UNSTRUCTURED_GRID\nPOINTS 2 double\n"
           "-0.4 0 0\n0.4 0 0\nCELLS 1 3\n2 0 1\nCELL_TYPES 1\n3\n"
           "POINT_DATA 2\nSCALARS tracer double 1\nLOOKUP_TABLE default\n"
           "0.8 3.2\nSCALARS a0 double 1\nLOOKUP_TABLE default\n"
        << left_area << ' ' << right_area
        << "\nCELL_DATA 1\nSCALARS marker double 1\nLOOKUP_TABLE default\n17\n";
      written = written && file.good();
    }
  ASSERT_EQ(Utilities::MPI::min(written ? 1u : 0u, MPI_COMM_WORLD), 1u);
  p.fluid.name_of_grid    = filename.string();
  p.fluid.read_vtk_fields = true;
  Domain<3>    solid(p.solid);
  Domain<1, 3> fluid(p.fluid);
  prepare_solid(solid);
  double     preparation_error = 0.;
  const auto report            = co_refine_lifted_domains(
    solid,
    fluid,
    p.lift,
    p.refinement,
    [&](const CellId &id, const Point<1> &unit) {
      const auto cell =
        fluid.refinement_triangulation().create_cell_iterator(id);
      const double x =
        (1. - unit[0]) * cell->vertex(0)[0] + unit[0] * cell->vertex(1)[0];
      const double area = fluid.prepared_field_value("a0", id, unit);
      const double expected =
        left_area + (x + .4) / .8 * (right_area - left_area);
      preparation_error =
        std::max(preparation_error, std::abs(area - expected));
      return std::sqrt(area / numbers::PI);
    },
    4);
  EXPECT_TRUE(report.satisfied());
  EXPECT_NEAR(Utilities::MPI::max(preparation_error, MPI_COMM_WORLD),
              0.,
              1.e-13);
  const auto                                    a0     = fluid.field("a0");
  const auto                                    tracer = fluid.field("tracer");
  const auto                                    marker = fluid.field("marker");
  const auto                                   &view   = a0.space();
  ImportedFiniteElementFields<1, 3>::VectorType relevant_values;
  relevant_values.reinit(view.locally_owned_dofs(),
                         view.locally_relevant_dofs(),
                         MPI_COMM_WORLD);
  relevant_values = a0.coefficients();
  relevant_values.update_ghost_values();
  const QGauss<1>             quadrature(3);
  FEValues<1, 3>              values(view.mapping(),
                        view.finite_element(),
                        quadrature,
                        update_values | update_quadrature_points);
  std::vector<Vector<double>> samples(
    quadrature.size(), Vector<double>(view.finite_element().n_components()));
  unsigned int minimum = std::numeric_limits<unsigned int>::max(), maximum = 0;
  for (const auto &cell : view.dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        minimum = std::min(minimum, static_cast<unsigned int>(cell->level()));
        maximum = std::max(maximum, static_cast<unsigned int>(cell->level()));
        values.reinit(cell);
        values.get_function_values(relevant_values, samples);
        for (const auto q : values.quadrature_point_indices())
          {
            const double x = values.quadrature_point(q)[0];
            EXPECT_NEAR(samples[q][a0.descriptor().first_fe_component],
                        left_area + (x + .4) / .8 * (right_area - left_area),
                        1.e-13);
            EXPECT_NEAR(samples[q][tracer.descriptor().first_fe_component],
                        2. + 3. * x,
                        1.e-12);
            EXPECT_NEAR(samples[q][marker.descriptor().first_fe_component],
                        17.,
                        1.e-12);
          }
      }
  EXPECT_LT(Utilities::MPI::min(minimum, MPI_COMM_WORLD),
            Utilities::MPI::max(maximum, MPI_COMM_WORLD));
}
#endif
