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

#include <deal.II/base/mpi.h>

#include <immersx/core/linear_adapter.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>

#include <iostream>
#include <string>

using namespace ImmersX;
#include <immersx/io/utils.h>

namespace
{
  template <int dim, int spacedim = dim>
  void
  run_poisson(const std::string &parameter_file)
  {
    PoissonParameters<dim, spacedim> parameters;
    LinearSolverParameters           adapter_parameters;
    adapter_parameters.solver         = LinearSolver::iterative;
    adapter_parameters.preconditioner = LinearPreconditioner::block_diagonal;
    adapter_parameters.maximum_iterations =
      parameters.solver_control.max_steps();
    adapter_parameters.tolerance = parameters.solver_control.tolerance();
    initialize_parameters(parameter_file);

    PoissonSolver<dim, spacedim> problem(parameters);
    problem.make_grid();
    problem.setup_fe();

    for (unsigned int cycle = 0; cycle < parameters.n_refinement_cycles;
         ++cycle)
      {
        problem.setup_system();
        if (parameters.output_results_before_solving)
          problem.output_results(cycle);

        problem.assemble_system();

        using FieldVector  = ImmersXLA::MPI::Vector;
        using GlobalVector = ImmersXLA::MPI::BlockVector;
        using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

        Adapter    adapter(adapter_parameters, MPI_COMM_WORLD);
        const auto fields = adapter.add(problem, "poisson");
        auto       state  = adapter.make_state();
        adapter.solve(state);
        problem.set_solution(adapter.field(state, fields.fields().solution));

        problem.output_results(cycle);
        parameters.convergence_table.error_from_exact(problem.dof_handler(),
                                                      problem.solution(),
                                                      parameters.bc);

        if (cycle + 1 < parameters.n_refinement_cycles)
          problem.refine_grid();
      }

    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      parameters.convergence_table.output_table(std::cout);
  }
} // namespace

int
main(int argc, char *argv[])
{
  using namespace dealii;

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      const std::string prm_file   = argc > 1 ? argv[1] : "parameters.prm";
      const auto        dimensions = get_dimension_parameters(prm_file);

      if (dimensions.dimension == 1 && dimensions.space_dimension == 3)
        run_poisson<1, 3>(prm_file);
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 2)
        run_poisson<1, 2>(prm_file);
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 1)
        run_poisson<1>(prm_file);
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 3)
        run_poisson<2, 3>(prm_file);
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        run_poisson<3>(prm_file);
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 2)
        run_poisson<2>(prm_file);
      else
        throw_unsupported_dimension_combination(dimensions);
    }
  catch (const std::exception &exc)
    {
      std::cerr << "Exception on processing: " << exc.what() << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << "Unknown exception!" << std::endl;
      return 1;
    }

  return 0;
}
