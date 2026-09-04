// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>

#include <iostream>
#include <string>

namespace
{
  template <int dim, int spacedim = dim>
  void
  run_poisson(const std::string &parameter_file)
  {
    using namespace ImmersX;

    PoissonParameters<dim, spacedim> parameters;
    LinearSolverParameters           adapter_parameters;
    initialize_parameters(parameter_file);
    adapter_parameters.solver         = LinearSolver::iterative;
    adapter_parameters.preconditioner = LinearPreconditioner::block_diagonal;
    adapter_parameters.maximum_iterations =
      parameters.solver_control.max_steps();
    adapter_parameters.tolerance = parameters.solver_control.tolerance();

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
  using namespace ImmersX;
  using namespace dealii;

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      const std::string parameter_file = argc > 1 ? argv[1] : "parameters.prm";
      const auto        dimensions = get_dimension_parameters(parameter_file);

      if (dimensions.dimension == 1 && dimensions.space_dimension == 1)
        run_poisson<1>(parameter_file);
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 2)
        run_poisson<1, 2>(parameter_file);
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 3)
        run_poisson<1, 3>(parameter_file);
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 2)
        run_poisson<2>(parameter_file);
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 3)
        run_poisson<2, 3>(parameter_file);
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        run_poisson<3>(parameter_file);
      else
        throw_unsupported_dimension_combination(dimensions);
    }
  catch (const std::exception &exception)
    {
      std::cerr << exception.what() << std::endl;
      return 1;
    }

  return 0;
}
