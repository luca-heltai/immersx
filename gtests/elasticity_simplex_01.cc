// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_parser.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elasticity.h>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;
namespace
{
  template <int dim>
  void
  check_legacy_elasticity(const bool simplex, const bool weak)
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<dim> parameters;
    ElasticityProblem<dim>           problem(parameters);
    const std::string                variables = dim == 2 ? "x,y,t" : "x,y,z,t";
    const std::string                exact = dim == 2 ? "1+x*x;0" : "1+x*x;0;0";
    const std::string                force = dim == 2 ? "-6;0" : "-6;0;0";
    initialize_parameters_from_string(
      "subsection Immersed Problem\n"
      " set FE degree = 2\n"
      " set Initial refinement = 0\n"
      " set Dirichlet boundary ids = " +
      std::string(weak ? "" : "0") +
      "\n"
      " set Weak Dirichlet boundary ids = " +
      std::string(weak ? "0" : "") +
      "\n"
      " set Output directory = " +
      TestPaths::output_directory("elasticity-simplex-" + std::to_string(dim) +
                                  (simplex ? "p" : "q") +
                                  (weak ? "weak" : "strong")) +
      "\n"
      " subsection Grid generation\n"
      " set Triangulation type = fullydistributed\n"
      " set Domain type = " +
      std::string(simplex ? "file" : "generate") +
      "\n"
      " set Grid generator = " +
      (simplex ?
         TestPaths::data_filename(dim == 2 ? "tests/simplex_square.msh" :
                                             "tests/simplex_cube.msh") :
         "hyper_cube") +
      "\n"
      " set Grid generator arguments = 0:1:false\n"
      " end\n"
      " subsection Immersed inclusions\n"
      " set Inclusions file =\n"
      " set Inclusions =\n"
      " end\n"
      " subsection Material properties\n"
      " subsection default\n"
      " set Lame mu = 1\n"
      " set Lame lambda = 1\n"
      " set Density = 0\n"
      " end\nend\n"
      " subsection Time parameters\n set Initial time = 0\n set Final time = 0\n end\n"
      "end\n"
      "subsection Functions\n"
      " subsection Right hand side\n set Variable names = " +
      variables + "\n set Function expression = " + force +
      "\n end\n"
      " subsection Exact solution\n set Variable names = " +
      variables + "\n set Function expression = " + exact +
      "\n end\n"
      " subsection Dirichlet boundary conditions\n set Variable names = " +
      variables + "\n set Function expression = " + exact +
      "\n end\n"
      "end\n"
      "subsection Solvers\n subsection Displacement\n set Max steps = 1000\n set Tolerance = 1.e-13\n set Reduction = 1.e-13\n end\nend\n");
    problem.run();
    EXPECT_EQ(problem.fe->reference_cell().is_simplex(), simplex);
    FunctionParser<dim> function(dim);
    function.initialize(variables, exact, {}, true);
    auto expected = problem.solution.block(0);
    VectorTools::interpolate(compatibility::linear_mapping<dim>(
                               problem.fe->reference_cell()),
                             problem.dh,
                             function,
                             expected);
    expected -= problem.solution.block(0);
    EXPECT_LT(expected.linfty_norm(), 1.e-9);
  }
} // namespace
TEST(ElasticityValidation, BOTH_SimplexQuadraticStrongBoundary)
{
  check_legacy_elasticity<2>(true, false);
  check_legacy_elasticity<3>(true, false);
}
TEST(ElasticityValidation, BOTH_SimplexQuadraticWeakBoundary)
{
  check_legacy_elasticity<2>(true, true);
  check_legacy_elasticity<3>(true, true);
}
TEST(ElasticityValidation, BOTH_HypercubeQuadraticBoundary)
{
  check_legacy_elasticity<2>(false, false);
  check_legacy_elasticity<3>(false, true);
}
