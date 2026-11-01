// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <immersx/physics/lifted_elasticity.h>

#include "test_paths.h"
using namespace ImmersX;
using namespace dealii;
namespace
{
  template <int dim>
  void
  quadratic_elasticity(const bool simplex,
                       const bool weak,
                       const bool coupled = false)
  {
    ParameterAcceptor::clear();
    LiftedElasticityParameters<dim> parameters;
    const std::string               variables = dim == 2 ? "x,y,t" : "x,y,z,t";
    const std::string               exact = dim == 2 ? "1+x*x;0" : "1+x*x;0;0";
    const std::string               forcing = dim == 2 ? "-6;0" : "-6;0;0";
    initialize_parameters_from_string(
      "subsection Lifted elasticity\n"
      " set FE degree = 2\n"
      " subsection Functions\n"
      " subsection Right hand side\n set Variable names = " +
      variables + "\n set Function expression = " + forcing +
      "\n end\n"
      " subsection Dirichlet boundary conditions\n set Variable names = " +
      variables + "\n set Function expression = " + exact +
      "\n end\n end\n end\n");
    parameters.default_material_properties.Lame_mu     = 1.;
    parameters.default_material_properties.Lame_lambda = 1.;
    parameters.domain_parameters.triangulation_type    = "fullydistributed";
    parameters.domain_parameters.initial_refinement    = 0;
    parameters.domain_parameters.name_of_grid =
      simplex ? TestPaths::data_filename(dim == 2 ? "tests/simplex_square.msh" :
                                                    "tests/simplex_cube.msh") :
                "hyper_cube";
    parameters.domain_parameters.arguments_for_grid = "0:1:false";
    parameters.execution.solver                     = LinearSolver::direct;
    parameters.coupling                             = "none";
    if (weak)
      {
        parameters.dirichlet_ids.clear();
        parameters.weak_dirichlet_ids = {0};
        parameters.penalty_term       = 100.;
      }
    if constexpr (dim == 3)
      if (coupled)
        {
          parameters.coupling = "multiplier";
          parameters.immersed_domain.name_of_grid =
            TestPaths::data_filename("tests/simplex_interior_line_3d.vtk");
          parameters.geometry.thickness                 = "0.05";
          parameters.co_refinement.max_refinement_level = 0;
          ReferenceCrossSection<1, 3, 3> section(parameters.geometry.section);
          Tensor<1, 3>                   tangent;
          tangent[0]       = 1.;
          const auto modes = section.get_transformed_mode_values(0, tangent);
          for (unsigned int slot = 0; slot < 3; ++slot)
            parameters.modal_data[slot] =
              std::to_string(modes[3 * slot]) + "*(1+x*x)";
        }
    LiftedElasticity<dim> problem(parameters);
    problem.setup();
    problem.solve();
    EXPECT_EQ(
      problem.displacement().finite_element().reference_cell().is_simplex(),
      simplex);
    FunctionParser<dim> function(dim);
    function.initialize(variables, exact, {}, true);
    auto expected = problem.solution();
    VectorTools::interpolate(problem.displacement().mapping(),
                             problem.displacement().dof_handler(),
                             function,
                             expected);
    expected -= problem.solution();
    EXPECT_LT(expected.linfty_norm(), 1.e-9);
    EXPECT_LT(problem.residual_norm(), 1.e-9);
    if (coupled)
      {
        EXPECT_EQ(problem.multipliers().size(), 3u);
        EXPECT_LT(problem.modal_coefficients().l2_norm(), 1.e-8);
      }
  }
} // namespace
TEST(LiftedElasticity, BOTH_SimplexQuadraticStrong)
{
  quadratic_elasticity<2>(true, false);
  quadratic_elasticity<3>(true, false);
}
TEST(LiftedElasticity, BOTH_SimplexQuadraticWeak)
{
  quadratic_elasticity<2>(true, true);
  quadratic_elasticity<3>(true, true);
}
TEST(LiftedElasticity, BOTH_HypercubeQuadraticBoundary)
{
  quadratic_elasticity<2>(false, false);
  quadratic_elasticity<3>(false, true);
}

TEST(LiftedElasticity, BOTH_SimplexQuadraticMultiplier)
{
  quadratic_elasticity<3>(true, false, true);
}
