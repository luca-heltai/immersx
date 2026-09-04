// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <immersx/physics/elastodynamics.h>

#ifdef DEAL_II_WITH_SUNDIALS
#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/physics/elastodynamics_semidiscrete.h>
#endif

#include <iostream>
#include <string>

using namespace ImmersX;
#include <immersx/io/utils.h>

namespace
{
#ifdef DEAL_II_WITH_SUNDIALS
  template <int dim, int spacedim = dim>
  void
  run_elastodynamics_adapter(const std::string &parameter_file)
  {
    ElastodynamicsParameters<dim, spacedim> parameters;
    initialize_parameters(parameter_file);

    ElastodynamicsSolver<dim, spacedim> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_operators();
    problem.set_initial_conditions();

    using FieldVector =
      typename ElastodynamicsSolver<dim, spacedim>::VectorType;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = IDAAdapter<FieldVector, GlobalVector>;

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
  }
#else
  template <int dim, int spacedim = dim>
  void
  run_elastodynamics_adapter(const std::string &)
  {
    AssertThrow(false,
                dealii::ExcMessage(
                  "elastodynamics_adapter requires deal.II with SUNDIALS "
                  "support."));
  }
#endif
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

      if (dimensions.dimension == 1 && dimensions.space_dimension == 1)
        run_elastodynamics_adapter<1, 1>(prm_file);
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 2)
        run_elastodynamics_adapter<1, 2>(prm_file);
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 3)
        run_elastodynamics_adapter<1, 3>(prm_file);
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 2)
        run_elastodynamics_adapter<2, 2>(prm_file);
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 3)
        run_elastodynamics_adapter<2, 3>(prm_file);
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        run_elastodynamics_adapter<3, 3>(prm_file);
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
