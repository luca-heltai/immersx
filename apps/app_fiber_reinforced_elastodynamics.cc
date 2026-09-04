// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/core/constraint.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>
#include <immersx/physics/fiber_reinforced_elastodynamics.h>

#include <algorithm>
#include <cmath>
#include <fstream>
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
    using Problem      = ElastodynamicsSolver<dim, dim>;
    using VectorType   = typename Problem::VectorType;
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

    dealii::DoFHandler<dim> multiplier_dh(
      fiber_problem.dof_handler().get_triangulation());
    multiplier_dh.distribute_dofs(fiber_problem.fe());
    const auto multiplier_owned = multiplier_dh.locally_owned_dofs();
    const auto multiplier_relevant =
      dealii::DoFTools::extract_locally_relevant_dofs(multiplier_dh);
    dealii::AffineConstraints<double> multiplier_constraints;
    multiplier_constraints.reinit(multiplier_owned, multiplier_relevant);
    dealii::DoFTools::make_hanging_node_constraints(multiplier_dh,
                                                    multiplier_constraints);
    multiplier_constraints.close();

    const auto matrix_view     = fe_space(matrix_problem.dof_handler(),
                                      matrix_problem.mapping(),
                                      matrix_problem.velocity_constraints());
    const auto fiber_view      = fe_space(fiber_problem.dof_handler(),
                                     fiber_problem.mapping(),
                                     fiber_problem.velocity_constraints());
    const auto multiplier_view = fe_space(multiplier_dh,
                                          fiber_problem.mapping(),
                                          multiplier_constraints,
                                          &multiplier_relevant);

    Adapter    adapter(parameters.time_parameters, MPI_COMM_WORLD);
    const auto matrix = adapter.add(matrix_problem, "matrix");
    const auto fiber  = adapter.add(fiber_problem, "fiber");
    const auto matrix_velocity =
      matrix_view.field(matrix.fields().velocity,
                        "matrix_velocity",
                        dealii::FEValuesExtractors::Vector(0));
    const auto fiber_velocity =
      fiber_view.field(fiber.fields().velocity,
                       "fiber_velocity",
                       dealii::FEValuesExtractors::Vector(0));
    const auto multiplier =
      multiplier_view.field("velocity_multiplier",
                            dealii::FEValuesExtractors::Vector(0));
    const auto constraint =
      make_constraint(weak_term(value(matrix_velocity), multiplier) -
                      weak_term(value(fiber_velocity), multiplier));
    const auto coupling = adapter.add(constraint, "fiber-coupling");

    const auto output_multiplier = [&parameters,
                                    &multiplier_dh](const VectorType  &values,
                                                    const unsigned int step) {
      std::filesystem::create_directories(parameters.output_directory);
      dealii::DataOut<dim> data_out;
      data_out.attach_dof_handler(multiplier_dh);
      data_out.add_data_vector(values, parameters.multiplier_output_name);
      data_out.build_patches();
      const auto rank =
        dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      const auto filename =
        std::filesystem::path(parameters.output_directory) /
        (parameters.multiplier_output_name + "-" + std::to_string(step) + "." +
         std::to_string(rank) + ".vtu");
      std::ofstream output(filename);
      data_out.write_vtu(output);
    };

    const auto output = [&matrix_problem,
                         &fiber_problem](const double       time,
                                         const unsigned int step) {
      matrix_problem.output_results();
      fiber_problem.output_results();
      (void)time;
      (void)step;
    };
    adapter.set_output_step([&adapter,
                             &matrix_problem,
                             &fiber_problem,
                             matrix,
                             fiber,
                             coupling,
                             &parameters,
                             output_multiplier,
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
      if ((parameters.time_parameters.output_frequency == 0 &&
           (step == 0 || time >= parameters.time_parameters.final_time)) ||
          (parameters.time_parameters.output_frequency > 0 &&
           (step % parameters.time_parameters.output_frequency == 0 ||
            time >= parameters.time_parameters.final_time)))
        {
          output(time, step);
          output_multiplier(adapter.field(state, coupling.fields().multiplier),
                            step);
        }
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
