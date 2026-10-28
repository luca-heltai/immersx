// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_parser.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics.h>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;
namespace
{
  void
  check_native_transient(const std::string &input)
  {
    ParameterAcceptor::clear();
    ElastodynamicsParameters<3> parameters;
    initialize_parameters(TestPaths::parameter_path(input));
    ElastodynamicsSolver<3> problem(parameters);
    problem.run();
    EXPECT_NEAR(problem.current_time(),
                parameters.time_parameters.final_time,
                1.e-14);
    FunctionParser<3> exact(3);
    exact.initialize("x,y,z,t", "(1+t*t*t)*(1+x*x);0;0", {}, true);
    exact.set_time(problem.current_time());
    ElastodynamicsSolver<3>::VectorType expected;
    expected.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    VectorTools::interpolate(problem.mapping(),
                             problem.dof_handler(),
                             exact,
                             expected);
    expected -= problem.displacement();
    EXPECT_LT(expected.linfty_norm(), 1.e-7);
    exact.initialize("x,y,z,t", "3*t*t*(1+x*x);0;0", {}, true);
    exact.set_time(problem.current_time());
    VectorTools::interpolate(problem.mapping(),
                             problem.dof_handler(),
                             exact,
                             expected);
    expected -= problem.velocity();
    EXPECT_LT(expected.linfty_norm(), 1.e-7);
  }
} // namespace

TEST(ElastodynamicsValidation, BOTH_SimplexQuadraticSpaceCubicTime)
{
  check_native_transient("gtests/parameters/elastodynamics_simplex.prm");
}

TEST(ElastodynamicsValidation, BOTH_HypercubeQuadraticSpaceCubicTime)
{
  check_native_transient("gtests/parameters/elastodynamics_hypercube.prm");
}
