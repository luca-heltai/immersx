// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_parser.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  template <int dim, int spacedim = dim>
  void
  check_quadratic_poisson(const bool simplex)
  {
    ParameterAcceptor::clear();
    PoissonParameters<dim, spacedim> parameters;
    const std::string name = std::to_string(dim) + std::to_string(spacedim) +
                             (simplex ? "simplex" : "hypercube");
    const std::string variables  = spacedim == 2 ? "x,y" : "x,y,z";
    const std::string expression = dim == 2 ? "x*x+y*y" : "x*x+y*y+z*z";
    initialize_parameters_from_string(
      R"(
      subsection Poisson
        set FE degree = 2
        set Dirichlet boundary ids = 0
        set Output directory = )" +
      TestPaths::output_directory("poisson_simplex_01/" + name) + R"(
        subsection Domain
          set Initial refinement = 0
          set Triangulation type = fullydistributed
          set Grid generator = )" +
      (simplex ?
         TestPaths::data_filename(dim == 2 ? "tests/simplex_square.msh" :
                                             "tests/simplex_cube.msh") :
         "subdivided_hyper_cube") +
      R"(
          set Grid generator arguments = )" +
      (simplex ? "" : "2: 0: 1: false") + R"(
        end
        subsection Right hand side
          set Function expression = )" +
      std::to_string(-2 * dim) + R"(
          set Variable names = )" +
      variables + R"(,t
        end
        subsection Dirichlet boundary conditions
          set Function expression = )" +
      expression + R"(
          set Variable names = )" +
      variables + R"(,t
        end
        subsection Solver
          subsection Control
            set Max steps = 1000
            set Reduction = 1.e-13
            set Tolerance = 1.e-13
          end
        end
      end
    )");
    PoissonSolver<dim, spacedim> problem(parameters);
    problem.run();
    EXPECT_EQ(problem.triangulation().all_reference_cells_are_simplex(),
              simplex);
    FunctionParser<spacedim> exact;
    exact.initialize(variables, expression, {});
    typename PoissonSolver<dim, spacedim>::VectorType expected;
    expected.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    VectorTools::interpolate(problem.mapping(),
                             problem.dof_handler(),
                             exact,
                             expected);
    expected -= problem.solution();
    EXPECT_LT(expected.linfty_norm(), 1.e-10);
  }
} // namespace

TEST(Poisson, BOTH_QuadraticSolutionOnSimplex)
{
  check_quadratic_poisson<2>(true);
  check_quadratic_poisson<2, 3>(true);
  check_quadratic_poisson<3>(true);
}

TEST(Poisson, BOTH_QuadraticSolutionOnHypercube)
{
  check_quadratic_poisson<2>(false);
  check_quadratic_poisson<2, 3>(false);
  check_quadratic_poisson<3>(false);
}
