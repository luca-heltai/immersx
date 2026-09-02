// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <immersx/algebra/vector_lagrange_multiplier_interaction.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>
#include <immersx/physics/fiber_reinforced_elastodynamics.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using namespace ImmersX;

namespace
{
  template <int dim>
  void
  run_fiber_reinforced_elastodynamics(const std::string &parameter_file)
  {
    FiberReinforcedElastodynamicsParameters<dim> parameters;
    initialize_parameters(parameter_file);

#ifdef DEAL_II_WITH_SUNDIALS
    using Problem        = ElastodynamicsSolver<dim, dim>;
    using VectorType     = typename Problem::VectorType;
    using Representation = VectorFiniteElementRepresentation<dim, dim>;
    using Interaction =
      VectorLagrangeMultiplierInteraction<Representation, Representation>;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = IDAAdapter<VectorType, GlobalVector>;
    Problem matrix_problem(parameters.matrix_parameters);
    Problem fiber_problem(parameters.fiber_parameters);
    matrix_problem.make_grid();
    matrix_problem.setup_fe();
    matrix_problem.setup_system();
    matrix_problem.assemble_operators();
    matrix_problem.set_initial_conditions();
    fiber_problem.make_grid();
    fiber_problem.setup_fe();
    fiber_problem.setup_system();
    fiber_problem.assemble_operators();
    fiber_problem.set_initial_conditions();

    Representation matrix_velocity(matrix_problem.triangulation(),
                                   matrix_problem.dof_handler(),
                                   matrix_problem.locally_owned_dofs(),
                                   matrix_problem.locally_relevant_dofs(),
                                   matrix_problem.velocity_constraints());
    Representation fiber_velocity(fiber_problem.triangulation(),
                                  fiber_problem.dof_handler(),
                                  fiber_problem.locally_owned_dofs(),
                                  fiber_problem.locally_relevant_dofs(),
                                  fiber_problem.velocity_constraints());
    Interaction    interaction(matrix_velocity,
                            fiber_velocity,
                            parameters.coupling_parameters);
    interaction.assemble();

    Adapter    adapter(parameters.time_parameters, MPI_COMM_WORLD);
    const auto matrix   = adapter.add(matrix_problem, "matrix");
    const auto fiber    = adapter.add(fiber_problem, "fiber");
    const auto coupling = adapter.add(interaction,
                                      "fiber-coupling",
                                      matrix.fields().velocity,
                                      fiber.fields().velocity);

    const auto output = [&parameters,
                         &matrix_problem,
                         &fiber_problem,
                         &interaction](const double       time,
                                       const unsigned int step) {
      matrix_problem.output_results();
      fiber_problem.output_results();
      interaction.output_results(parameters.output_directory + "/interaction",
                                 parameters.multiplier_output_name,
                                 step,
                                 time);
    };
    adapter.set_output_step([&adapter,
                             &matrix_problem,
                             &fiber_problem,
                             &interaction,
                             matrix,
                             fiber,
                             coupling,
                             &parameters,
                             output](const double        time,
                                     const GlobalVector &state,
                                     const GlobalVector &state_dot,
                                     const unsigned int  step) {
      matrix_problem.accept_state(
        adapter.field(state, matrix.fields().displacement),
        adapter.field(state, matrix.fields().velocity),
        time,
        step);
      fiber_problem.accept_state(adapter.field(state,
                                               fiber.fields().displacement),
                                 adapter.field(state, fiber.fields().velocity),
                                 time,
                                 step);
      interaction.set_multiplier(
        adapter.field(state, coupling.fields().multiplier));
      if ((parameters.time_parameters.output_frequency == 0 &&
           (step == 0 || time >= parameters.time_parameters.final_time)) ||
          (parameters.time_parameters.output_frequency > 0 &&
           (step % parameters.time_parameters.output_frequency == 0 ||
            time >= parameters.time_parameters.final_time)))
        output(time, step);
      (void)state_dot;
    });

    auto state     = adapter.make_state();
    auto state_dot = adapter.make_state();
    adapter.field(state, matrix.fields().displacement) =
      matrix_problem.displacement();
    adapter.field(state, matrix.fields().velocity) = matrix_problem.velocity();
    adapter.field(state, fiber.fields().displacement) =
      fiber_problem.displacement();
    adapter.field(state, fiber.fields().velocity) = fiber_problem.velocity();
    adapter.field(state, coupling.fields().multiplier) = 0.;
    adapter.field(state_dot, matrix.fields().displacement) =
      matrix_problem.velocity();
    adapter.field(state_dot, fiber.fields().displacement) =
      fiber_problem.velocity();
    adapter.field(state_dot, matrix.fields().velocity)     = 0.;
    adapter.field(state_dot, fiber.fields().velocity)      = 0.;
    adapter.field(state_dot, coupling.fields().multiplier) = 0.;

    interaction.set_multiplier(
      adapter.field(state, coupling.fields().multiplier));
    adapter.solve(state, state_dot);
#else
    FiberReinforcedElastodynamics<dim> driver(parameters);
    driver.run();
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
        run_fiber_reinforced_elastodynamics<2>(prm_file);
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        run_fiber_reinforced_elastodynamics<3>(prm_file);
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
