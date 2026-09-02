// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/physics/elastodynamics.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using namespace ImmersX;
#include <immersx/io/utils.h>

namespace
{
  template <int dim>
  void
  run_elastodynamics(const std::string &parameter_file)
  {
    ElastodynamicsParameters<dim> parameters;
#ifdef DEAL_II_WITH_SUNDIALS
    using FieldVector  = typename ElastodynamicsSolver<dim>::VectorType;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = IDAAdapter<FieldVector, GlobalVector>;
#endif
    initialize_parameters(parameter_file);

    ElastodynamicsSolver<dim> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_operators();
    problem.set_initial_conditions();

#ifdef DEAL_II_WITH_SUNDIALS
    Adapter    adapter(parameters.time_parameters, MPI_COMM_WORLD);
    const auto fields = adapter.add(problem, "elastodynamics");
    adapter.set_output_step(
      [&problem, &adapter, fields, &parameters](const double        time,
                                                const GlobalVector &state,
                                                const GlobalVector &state_dot,
                                                const unsigned int  step) {
        problem.accept_state(adapter.field(state, fields.fields().displacement),
                             adapter.field(state, fields.fields().velocity),
                             time,
                             step);
        if ((parameters.time_parameters.output_frequency == 0 &&
             (step == 0 || time >= parameters.time_parameters.final_time)) ||
            (parameters.time_parameters.output_frequency > 0 &&
             (step % parameters.time_parameters.output_frequency == 0 ||
              time >= parameters.time_parameters.final_time)))
          problem.output_results();
        (void)state_dot;
      });

    auto state                                         = adapter.make_state();
    auto state_dot                                     = adapter.make_state();
    adapter.field(state, fields.fields().displacement) = problem.displacement();
    adapter.field(state, fields.fields().velocity)     = problem.velocity();
    adapter.field(state_dot, fields.fields().displacement) = problem.velocity();

    FieldVector acceleration;
    problem.initial_acceleration(acceleration);
    adapter.field(state_dot, fields.fields().velocity) = acceleration;

    adapter.solve(state, state_dot);
#else
    problem.run();
#endif
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

      if (dimensions.dimension == 2 && dimensions.space_dimension == 2)
        run_elastodynamics<2>(prm_file);
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        run_elastodynamics<3>(prm_file);
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
