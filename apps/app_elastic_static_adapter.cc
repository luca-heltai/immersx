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
#include <immersx/physics/elastic_static.h>

#include <iostream>
#include <string>

namespace
{
  template <int dim, int spacedim = dim>
  void
  run_static_elasticity(const std::string &parameter_file)
  {
    using namespace ImmersX;
    using Problem = ElasticStaticProblem<dim, spacedim>;

    ElasticStaticParameters<dim, spacedim> parameters;
    LinearSolverParameters                 adapter_parameters;
    initialize_parameters(parameter_file);

    adapter_parameters.solver         = LinearSolver::iterative;
    adapter_parameters.preconditioner = LinearPreconditioner::block_diagonal;
    adapter_parameters.maximum_iterations =
      parameters.solver_control.max_steps();
    adapter_parameters.tolerance = parameters.solver_control.tolerance();

    AssertThrow(
      (dim != 1 && parameters.triangulation_type != "fullydistributed") ||
        parameters.n_refinement_cycles <= 1,
      ExcMessage("parallel::fullydistributed::Triangulation supports only "
                 "one static refinement cycle because its mesh is immutable "
                 "after copy_triangulation()."));

    Problem problem(parameters);
    problem.setup();

    for (unsigned int cycle = 0; cycle < parameters.n_refinement_cycles;
         ++cycle)
      {
        using FieldVector  = typename Problem::VectorType;
        using GlobalVector = ImmersXLA::MPI::BlockVector;
        using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

        Adapter    adapter(adapter_parameters, MPI_COMM_WORLD);
        const auto fields = adapter.add(problem, "elastic-static");
        auto       state  = adapter.make_state();
        adapter.solve(state);
        problem.set_solution(
          adapter.field(state, fields.fields().displacement));

        FieldVector residual;
        residual.reinit(problem.solution());
        problem.stiffness_operator().vmult(residual, problem.solution());
        residual -= problem.forcing();
        problem.constraints().set_zero(residual);
        const double residual_norm = residual.l2_norm();
        AssertThrow(residual_norm < 1.e-9,
                    ExcMessage("Static elasticity residual is too large in "
                               "cycle " +
                               std::to_string(cycle) + "."));

        problem.compute_error(parameters.convergence_table);
        problem.output_results(cycle);

        if (cycle + 1 < parameters.n_refinement_cycles)
          problem.refine_global();
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
        run_static_elasticity<1>(parameter_file);
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 2)
        run_static_elasticity<1, 2>(parameter_file);
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 3)
        run_static_elasticity<1, 3>(parameter_file);
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 2)
        run_static_elasticity<2>(parameter_file);
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 3)
        run_static_elasticity<2, 3>(parameter_file);
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        run_static_elasticity<3>(parameter_file);
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
