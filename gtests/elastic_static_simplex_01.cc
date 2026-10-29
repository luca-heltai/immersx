// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_parser.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastic_static.h>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  template <int dim, int spacedim = dim>
  void
  check_static_simplex()
  {
    ParameterAcceptor::clear();
    ElasticStaticParameters<dim, spacedim> parameters;
    const std::string variables        = spacedim == 2 ? "x,y,t" : "x,y,z,t";
    std::string       exact_expression = "1+x*x", force_expression = "-6";
    for (unsigned int component = 1; component < spacedim; ++component)
      {
        exact_expression += "; 0";
        force_expression += "; 0";
      }
    initialize_parameters_from_string(
      "subsection Elastic static\n"
      "  set FE degree = 2\n"
      "  set Dirichlet boundary ids = 0\n"
      "  set Output directory = " +
      TestPaths::output_directory("elastic-static-simplex-" +
                                  std::to_string(dim) +
                                  std::to_string(spacedim)) +
      "\n"
      "  subsection Domain\n"
      "    set Triangulation type = fullydistributed\n"
      "    set Initial refinement = 0\n"
      "    set Grid generator = " +
      TestPaths::data_filename(dim == 2 ? "tests/simplex_square.msh" :
                                          "tests/simplex_cube.msh") +
      "\n"
      "  end\n"
      "  subsection Material properties\n"
      "    subsection default\n"
      "      set Lame mu = 1\n"
      "      set Lame lambda = 1\n"
      "    end\n"
      "  end\n"
      "  subsection Functions\n"
      "    subsection Right hand side\n"
      "      set Variable names = " +
      variables +
      "\n"
      "      set Function expression = " +
      force_expression +
      "\n"
      "    end\n"
      "    subsection Dirichlet boundary conditions\n"
      "      set Variable names = " +
      variables +
      "\n"
      "      set Function expression = " +
      exact_expression +
      "\n"
      "    end\n"
      "  end\n"
      "  subsection Solver\n"
      "    subsection Control\n"
      "      set Max steps = 1000\n"
      "      set Tolerance = 1.e-13\n"
      "      set Reduction = 1.e-13\n"
      "    end\n"
      "  end\n"
      "end\n");
    ElasticStaticProblem<dim, spacedim> problem(parameters);
    problem.run();
    EXPECT_TRUE(problem.triangulation().all_reference_cells_are_simplex());
    FunctionParser<spacedim> exact(spacedim);
    exact.initialize(variables, exact_expression, {}, true);
    typename ElasticStaticProblem<dim, spacedim>::VectorType expected;
    expected.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    VectorTools::interpolate(problem.mapping(),
                             problem.dof_handler(),
                             exact,
                             expected);
    expected -= problem.solution();
    EXPECT_LT(expected.linfty_norm(), 1.e-10);
    EXPECT_THROW(problem.compute_error(parameters.convergence_table),
                 ExceptionBase);
  }
} // namespace

TEST(ElasticStaticProblem, BOTH_SimplexQuadraticSolution)
{
  check_static_simplex<2>();
  check_static_simplex<2, 3>();
  check_static_simplex<3>();
}
