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
#include <immersx/physics/navier_stokes.h>
#include <immersx/physics/navier_stokes_semidiscrete.h>

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
  run_navier_stokes(const std::string &parameter_file)
  {
    NavierStokesParameters<dim> parameters;
    initialize_parameters(parameter_file);

    NavierStokesSolver<dim> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_system();

#ifdef DEAL_II_WITH_SUNDIALS
    using FieldVector  = typename NavierStokesSolver<dim>::VectorType;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = IDAAdapter<FieldVector, GlobalVector>;

    typename Adapter::AdditionalData data;
    data.initial_time                  = parameters.initial_time;
    data.final_time                    = parameters.final_time;
    data.initial_step_size             = std::min(problem.time_step(), 1.e-5);
    data.output_period                 = parameters.output_frequency > 0 ?
                                           parameters.output_frequency * problem.time_step() :
                                           parameters.final_time - parameters.initial_time;
    data.maximum_order                 = 1;
    data.maximum_non_linear_iterations = 50;
    data.absolute_tolerance            = 1.e-4;
    data.relative_tolerance            = 1.e-4;
    data.ic_type                       = Adapter::AdditionalData::use_y_diff;
    data.reset_type                    = Adapter::AdditionalData::none;

    Adapter    adapter(data, MPI_COMM_WORLD);
    const auto fields = adapter.add(problem, "navier-stokes");
    adapter.set_output_step(
      [&problem, &adapter, fields, &parameters](const double        time,
                                                const GlobalVector &state,
                                                const GlobalVector &state_dot,
                                                const unsigned int  step) {
        problem.accept_state(adapter.field(state, fields.fields().velocity),
                             adapter.field(state, fields.fields().pressure),
                             time,
                             step);
        if (parameters.output_frequency > 0)
          problem.output_results();
        (void)state_dot;
      });

    auto state     = adapter.make_state();
    auto state_dot = adapter.make_state();
    adapter.field(state, fields.fields().velocity) =
      problem.solution().block(0);
    adapter.field(state, fields.fields().pressure) =
      problem.solution().block(1);
    adapter.field(state_dot, fields.fields().velocity) = 0.;
    adapter.field(state_dot, fields.fields().pressure) = 0.;

    if (parameters.output_frequency > 0)
      problem.output_results();
    adapter.solve(state, state_dot);

    const unsigned int n_steps =
      parameters.time_step_policy == "number_of_steps" ?
        parameters.number_of_time_steps :
        static_cast<unsigned int>(
          std::ceil((parameters.final_time - parameters.initial_time) /
                    parameters.time_step));
    problem.accept_state(adapter.field(state, fields.fields().velocity),
                         adapter.field(state, fields.fields().pressure),
                         parameters.final_time,
                         n_steps);
    if (parameters.output_frequency == 0)
      problem.output_results();
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
        run_navier_stokes<2>(prm_file);
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        run_navier_stokes<3>(prm_file);
      else
        throw_unsupported_dimension_combination(dimensions);

      MPI_Barrier(MPI_COMM_WORLD);
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
