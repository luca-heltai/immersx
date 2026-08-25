// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastic_static.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <type_traits>

namespace
{
  template <int dim>
  void
  run_static_elasticity(const std::string &parameter_file)
  {
    using namespace ImmersX;
    using FieldVector  = ImmersXLA::MPI::Vector;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = LinearAdapter<FieldVector, GlobalVector>;
    using Problem      = ElasticStaticProblem<dim>;

    ElasticStaticParameters<dim> parameters;
    initialize_parameters(parameter_file);

    Problem problem(parameters);
    problem.setup();

    Adapter adapter(
      MPI_COMM_WORLD,
      [](const auto &operator_view, const auto &rhs, auto &solution) {
        using Vector = std::decay_t<decltype(solution)>;
        dealii::SolverControl        control(1000,
                                      1.e-12 * std::max(1., rhs.l2_norm()));
        dealii::SolverGMRES<Vector>  solver(control);
        dealii::PreconditionIdentity preconditioner;
        solver.solve(operator_view, solution, rhs, preconditioner);
      });

    const auto fields = adapter.add(problem, "elastic-static");
    auto       state  = adapter.make_state();
    adapter.solve(state);
    problem.set_solution(adapter.field(state, fields.fields().displacement));

    GlobalVector residual;
    adapter.evaluate_residual(state, residual);
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "elastic_static_residual = " << residual.l2_norm() << '\n';

    problem.output_results();
    AssertThrow(residual.l2_norm() < 1.e-9,
                ExcMessage("Static elasticity residual is too large."));
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

      if (dimensions.dimension == 2 && dimensions.space_dimension == 2)
        run_static_elasticity<2>(parameter_file);
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
