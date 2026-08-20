// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>

#include <vector>

#include "navier_stokes.h"
#include "utils.h"


using namespace dealii;


TEST(NavierStokes, Step80ManufacturedSolutionConvergence)
{
  constexpr unsigned int dim      = 2;
  constexpr unsigned int spacedim = 2;

  ParameterAcceptor::clear();
  NavierStokesParameters<dim, spacedim> parameters;
  initialize_parameters(SOURCE_DIR
                        "/gtests/parameters/navier_stokes_step80_mms_2d.prm");

  const std::vector<unsigned int> levels = {2, 3, 4};

  for (const unsigned int level : levels)
    {
      parameters.initial_refinement = level;

      NavierStokesSolver<dim, spacedim> problem(parameters);
      problem.run();

      FunctionParser<spacedim> exact_solution(3);
      exact_solution.initialize(
        FunctionParser<spacedim>::default_variable_names(),
        parameters.analytical_solution_expression,
        {{"pi", numbers::PI}});
      exact_solution.set_time(problem.current_time());

      parameters.convergence_table.error_from_exact(
        problem.dof_handler(),
        problem.locally_relevant_solution(),
        exact_solution);
    }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout
        << "\nNavier-Stokes step-80 manufactured-solution convergence:\n";
      parameters.convergence_table.output_table(std::cout);
    }
}
