// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/vessel_elasticity.h>

#include "test_paths.h"
using namespace ImmersX;
using namespace dealii;
namespace
{
  void
  pointwise_tissue_feedback(const bool simplex)
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<3> parameters;
    VesselElasticityProblem<3>     problem(parameters);
    initialize_parameters(TestPaths::parameter_path(
      "gtests/parameters/coupled_elasticity_simplex.prm"));
    if (!simplex)
      {
        parameters.domain_type        = "generate";
        parameters.triangulation_type = "distributed";
        parameters.name_of_grid       = "hyper_cube";
        parameters.arguments_for_grid = "0:1:false";
        parameters.initial_refinement = 2;
      }
    parameters.remove_interior_stress_from_pressure = false;
    problem.run_timestep0();
    ASSERT_EQ(problem.n_vessels(), 1u);
    problem.update_vessel_radius_data({.001, .002}, {2});
    problem.run_timestep();
    EXPECT_EQ(problem.fe->reference_cell().is_simplex(), simplex);
    EXPECT_GT(problem.solution.block(0).l2_norm(), 1.e-6);
    Vector<double> coefficients(problem.solution.block(1));
    Vector<double> pressure(problem.pressure_at_immersed_surface_by_inclusion);
    ASSERT_EQ(coefficients.size(), 4u);
    ASSERT_EQ(pressure.size(), 2u);
    for (unsigned int k = 0; k < pressure.size(); ++k)
      EXPECT_NEAR(pressure[k],
                  (coefficients[2 * k] + coefficients[2 * k + 1]) /
                    (2. * problem.inclusions.get_section_measure(k)),
                  1.e-10);
    EXPECT_GT(std::abs(pressure[0] - pressure[1]), 1.e-6);
    // The 1D driver gathers once before calling this routine on rank zero.
    // No MPI operation may occur inside this interpolation callback.
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      [&] {
        const auto feedback =
          problem.split_pressure_over_inclusions({3}, pressure);
        ASSERT_EQ(feedback.size(), 1u);
        ASSERT_EQ(feedback[0].size(), 3u);
        EXPECT_NEAR(feedback[0][0], pressure[0], 1.e-10);
        EXPECT_NEAR(feedback[0][1], .5 * (pressure[0] + pressure[1]), 1.e-10);
        EXPECT_NEAR(feedback[0][2], pressure[1], 1.e-10);
      }();
    MPI_Barrier(MPI_COMM_WORLD);
    auto displacement    = problem.solution.block(0);
    auto multiplier      = problem.solution.block(1);
    auto vessel_pressure = problem.pressure_at_immersed_surface;
    problem.update_vessel_radius_data({.002, .004}, {2});
    problem.run_timestep();
    displacement *= 2.;
    multiplier *= 2.;
    vessel_pressure *= 2.;
    displacement -= problem.solution.block(0);
    multiplier -= problem.solution.block(1);
    vessel_pressure -= problem.pressure_at_immersed_surface;
    EXPECT_LT(displacement.linfty_norm(), 1.e-9);
    EXPECT_LT(multiplier.linfty_norm(), 1.e-9);
    EXPECT_LT(vessel_pressure.linfty_norm(), 1.e-9);
  }
} // namespace
TEST(ElasticityValidation, BOTH_SimplexPointwiseTissueFeedback)
{
  pointwise_tissue_feedback(true);
}
TEST(ElasticityValidation, BOTH_HypercubePointwiseTissueFeedback)
{
  pointwise_tissue_feedback(false);
}
