// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the LICENSE.md file at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/physics/poisson.h>

#include <filesystem>
#include <sstream>

using namespace ImmersX;
#include <immersx/io/utils.h>

#include "test_paths.h"


using namespace dealii;


template <int dim, int spacedim = dim>
void
check_poisson_construction()
{
  ParameterAcceptor::clear();
  PoissonParameters<dim, spacedim> parameters;
  PoissonSolver<dim, spacedim>     problem(parameters);

  EXPECT_EQ(problem.n_dofs(), 0u);
  EXPECT_FALSE(problem.solution_is_finite());
}


TEST(Poisson, Construction)
{
  check_poisson_construction<1>();
  check_poisson_construction<1, 2>();
  check_poisson_construction<1, 3>();
  check_poisson_construction<2>();
  check_poisson_construction<2, 3>();
  check_poisson_construction<3>();
}


TEST(Poisson, MPI_OneCycleSolve)
{
  ParameterAcceptor::clear();
  PoissonParameters<2> parameters;
  initialize_parameters_from_string(R"(
    subsection Poisson
      set FE degree                   = 1
      set Dirichlet boundary ids      = 0
      set Output name                 = poisson_01
      subsection Domain
        set Initial refinement          = 1
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Refinement and remeshing
        set Number of refinement cycles = 2
        set Strategy                   = global
      end
      subsection Right hand side
        set Function expression = 1
        set Variable names      = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names      = x,y,t
      end
      subsection Solver
        subsection Control
          set Max steps  = 100
          set Reduction  = 1.e-10
          set Tolerance  = 1.e-12
          set Log result = false
        end
      end
    end
  )");

  parameters.output_directory =
    ImmersX::TestPaths::output_directory("poisson-01");

  PoissonSolver<2> problem(parameters);
  problem.run();

  EXPECT_GT(problem.n_dofs(), 0u);
  EXPECT_TRUE(problem.solution_is_finite());
  EXPECT_GT(problem.solution_l2_norm(), 0.);
}


TEST(Poisson, MPI_EmbeddedOneDimensionalSolve)
{
  ParameterAcceptor::clear();
  PoissonParameters<1, 2> parameters;
  initialize_parameters_from_string(R"(
    subsection Poisson
      set FE degree                   = 1
      set Dirichlet boundary ids      = 0,1
      subsection Domain
        set Initial refinement          = 1
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Refinement and remeshing
        set Number of refinement cycles = 1
        set Strategy                   = global
      end
      subsection Right hand side
        set Function expression = 1
        set Variable names      = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names      = x,y,t
      end
      subsection Solver
        subsection Control
          set Max steps  = 100
          set Reduction  = 1.e-10
          set Tolerance  = 1.e-12
          set Log result = false
        end
      end
    end
  )");

  parameters.output_directory =
    ImmersX::TestPaths::output_directory("poisson-embedded-01");

  PoissonSolver<1, 2> problem(parameters);
  problem.run();

  EXPECT_GT(problem.n_dofs(), 0u);
  EXPECT_TRUE(problem.solution_is_finite());
  EXPECT_GT(problem.solution_l2_norm(), 0.);
}

namespace
{
  template <int dim, int spacedim = dim>
  void
  check_robin_solution(const bool simplex       = false,
                       const bool mixed         = false,
                       const bool neumann_limit = false)
  {
    ParameterAcceptor::clear();
    PoissonParameters<dim, spacedim> parameters;
    const std::string                coordinates[] = {"x", "y", "z"};
    std::string                      variables;
    std::string                      exact = "1";
    std::string                      flux  = "0";
    for (unsigned int d = 0; d < spacedim; ++d)
      variables += coordinates[d] + ",";
    variables += "t";
    for (unsigned int d = 0; d < dim; ++d)
      {
        const auto factor = std::to_string(d + 1);
        exact += "+" + factor + "*" + coordinates[d];
        flux += "+" + factor + "*if(" + coordinates[d] + "<1e-9,-1,if(" +
                coordinates[d] + ">1-1e-9,1,0))";
      }
    const std::string  beta = neumann_limit ? "0" : "2+x";
    std::ostringstream input;
    input << "subsection Poisson\n"
          << " set Dirichlet boundary ids = " << (mixed ? "0" : "") << '\n'
          << " subsection Domain\n"
          << "  set Initial refinement = 0\n"
          << "  set Triangulation type = fullydistributed\n"
          << "  set Grid generator = "
          << (simplex ?
                TestPaths::data_filename(dim == 2 ? "tests/simplex_square.msh" :
                                                    "tests/simplex_cube.msh") :
                "subdivided_hyper_cube")
          << '\n'
          << "  set Grid generator arguments = "
          << (simplex ? "" :
              mixed   ? "2:0:1:true" :
                        "2:0:1:false")
          << '\n'
          << " end\n"
          << " subsection Dirichlet boundary conditions\n"
          << "  set Variable names = " << variables << '\n'
          << "  set Function expression = " << exact << "\n end\n"
          << " subsection Robin coefficient\n"
          << "  set Variable names = " << variables << '\n'
          << "  set Function expression = " << beta << "\n end\n"
          << " subsection Robin boundary conditions\n"
          << "  set Variable names = " << variables << '\n'
          << "  set Function expression = " << flux << "+(" << beta << ")*("
          << exact << ")\n end\n"
          << " subsection Solver\n subsection Control\n"
          << "  set Max steps = 500\n set Reduction = 1e-13\n"
          << "  set Tolerance = 1e-13\n end\n end\n end\n";
    initialize_parameters_from_string(input.str());
    if (mixed)
      for (unsigned int id = 1; id < 2 * dim; ++id)
        parameters.robin_ids.push_back(id);
    else
      parameters.robin_ids = dim == 1 ? std::list<types::boundary_id>{0, 1} :
                                        std::list<types::boundary_id>{0};
    PoissonSolver<dim, spacedim> problem(parameters);
    problem.setup();
    problem.solve();
    FunctionParser<spacedim> exact_function;
    exact_function.initialize(variables, exact, {}, true);
    typename PoissonSolver<dim, spacedim>::VectorType expected;
    expected.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    VectorTools::interpolate(problem.mapping(),
                             problem.dof_handler(),
                             exact_function,
                             expected);
    expected -= problem.solution();
    EXPECT_LT(expected.linfty_norm(), 1e-10);

    if (!mixed)
      {
        // For u=1 the volume action vanishes; the boundary integral of
        // beta=2+x on the unit cube is exactly 5*dim (also for embedded grids).
        typename PoissonSolver<dim, spacedim>::VectorType one, action;
        one.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
        action.reinit(one);
        one = 1.;
        problem.system_matrix().vmult(action, one);
        EXPECT_NEAR(one * action, 5. * dim, 1e-11);
        problem.assemble_system();
        problem.system_matrix().vmult(action, one);
        EXPECT_NEAR(one * action, 5. * dim, 1e-11);
      }
  }
} // namespace

TEST(Poisson, BOTH_RobinHypercubeManufacturedSolution)
{
  check_robin_solution<1>();
  check_robin_solution<1, 2>();
  check_robin_solution<1, 3>();
  check_robin_solution<2>();
  check_robin_solution<2, 3>();
  check_robin_solution<3>();
}

TEST(Poisson, BOTH_RobinSimplexManufacturedSolution)
{
  check_robin_solution<2>(true);
  check_robin_solution<2, 3>(true);
  check_robin_solution<3>(true);
}

TEST(Poisson, BOTH_RobinMixedAndNeumannLimit)
{
  check_robin_solution<2>(false, true);
  check_robin_solution<2>(false, true, true);
}

TEST(Poisson, BOTH_RobinRejectsInvalidDataCollectively)
{
  for (const auto &data :
       {std::pair<std::string, std::string>{"if(x<0.1,-1,1)", "0"},
        {"exp(1000*x)", "0"},
        {"2", "exp(1000*x)"}})
    {
      ParameterAcceptor::clear();
      PoissonParameters<2> parameters;
      initialize_parameters_from_string(R"(
    subsection Poisson
      set Dirichlet boundary ids =
      set Robin boundary ids = 0
      subsection Domain
        set Initial refinement = 2
        set Grid generator = hyper_cube
        set Grid generator arguments = 0:1:false
      end
      subsection Robin coefficient
        set Function expression = )" + data.first +
                                        R"(
      end
      subsection Robin boundary conditions
        set Function expression = )" + data.second +
                                        R"(
      end
    end
  )");
      PoissonSolver<2> problem(parameters);
      EXPECT_THROW(problem.setup(), ExceptionBase);
    }
}

TEST(Poisson, BOTH_RobinRejectsDirichletOverlap)
{
  ParameterAcceptor::clear();
  PoissonParameters<2> parameters;
  parameters.domain_parameters.initial_refinement = 0;
  parameters.robin_ids                            = {0};
  PoissonSolver<2> problem(parameters);
  EXPECT_THROW(problem.setup(), ExceptionBase);
}
