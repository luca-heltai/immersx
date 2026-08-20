// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/physics/fiber_reinforced_elastodynamics.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace ImmersX
{
  using namespace dealii;

  namespace
  {
    std::string
    normalize_subsection(const std::string &subsection)
    {
      if (subsection.empty())
        return "/Fiber Reinforced Elastodynamics/";

      std::string normalized = subsection;
      if (normalized.front() != '/')
        normalized.insert(normalized.begin(), '/');
      if (normalized.back() != '/')
        normalized.push_back('/');
      return normalized;
    }

    void
    ensure_directory(const std::string &directory)
    {
      std::error_code error;
      std::filesystem::create_directories(directory, error);
      AssertThrow(!error,
                  ExcMessage("Could not create output directory '" + directory +
                             "': " + error.message()));
    }

    void
    assert_homogeneous_constraints(const AffineConstraints<double> &constraints)
    {
      for (const auto &line : constraints.get_lines())
        AssertThrow(std::abs(line.inhomogeneity) <= 1.e-14,
                    ExcMessage(
                      "The coupled backward-Euler driver currently supports "
                      "homogeneous velocity constraints only."));
    }

    void
    impose_homogeneous_constraints(ImmersXLA::MPI::SparseMatrix    &matrix,
                                   const AffineConstraints<double> &constraints)
    {
      assert_homogeneous_constraints(constraints);

      // AffineConstraints::condense(SparseMatrix) has no explicit distributed
      // Trilinos instantiation in the supported deal.II package.  Apply the
      // homogeneous elimination directly through the distributed matrix API:
      // zero constrained columns on locally owned rows, then replace owned
      // constrained rows by identity rows.
      const auto owned_rows = matrix.locally_owned_range_indices();
      for (const auto row : owned_rows)
        {
          std::vector<types::global_dof_index> constrained_columns;
          for (auto entry = matrix.begin(row); entry != matrix.end(row);
               ++entry)
            if (constraints.is_constrained(entry->column()))
              constrained_columns.push_back(entry->column());
          for (const auto column : constrained_columns)
            matrix.set(row, column, 0.);
        }
      matrix.compress(VectorOperation::insert);

      for (const auto &line : constraints.get_lines())
        if (matrix.in_local_range(line.index))
          matrix.clear_row(line.index, 1.);
      matrix.compress(VectorOperation::insert);
    }

    void
    impose_homogeneous_constraints(ImmersXLA::MPI::Vector          &vector,
                                   const AffineConstraints<double> &constraints)
    {
      assert_homogeneous_constraints(constraints);
      for (const auto &line : constraints.get_lines())
        if (vector.locally_owned_elements().is_element(line.index))
          vector(line.index) = 0.;
    }
  } // namespace


  template <int dim>
  FiberReinforcedElastodynamicsParameters<
    dim>::FiberReinforcedElastodynamicsParameters(const std::string &subsection)
    : ParameterAcceptor(normalize_subsection(subsection))
    , matrix_parameters(normalize_subsection(subsection) +
                        "Matrix Elastodynamics/")
    , fiber_parameters(normalize_subsection(subsection) +
                       "Fiber Elastodynamics/")
    , coupling_parameters(normalize_subsection(subsection) +
                          "Fiber Coupling/Particle search/")
  {
    add_parameter("Output directory", output_directory);
    add_parameter("Output frequency", output_frequency);

    enter_subsection("Time integration");
    {
      add_parameter("Initial time", initial_time);
      add_parameter("Final time", final_time);
      add_parameter("Time step", time_step, "", prm, Patterns::Double(0));
      add_parameter("Number of time steps", number_of_steps);
    }
    leave_subsection();

    enter_subsection("Coupling solver");
    {
      add_parameter("Maximum steps", schur_max_steps);
      add_parameter("Tolerance", schur_tolerance, "", prm, Patterns::Double(0));
      add_parameter(
        "Block tolerance", block_tolerance, "", prm, Patterns::Double(0));
      add_parameter("Initial compatibility tolerance",
                    initial_compatibility_tolerance,
                    "",
                    prm,
                    Patterns::Double(0));
    }
    leave_subsection();

    parse_parameters_call_back.connect([this]() {
      ensure_directory(output_directory);
      AssertThrow(final_time >= initial_time,
                  ExcMessage("The coupled final time must not precede the "
                             "initial time."));
      if (final_time > initial_time || number_of_steps > 0)
        AssertThrow(time_step > 0.,
                    ExcMessage("The coupled time step must be positive."));
      // The two physical Problems retain their own parameter objects for
      // material, mesh, and function data. Their time controls are synchronized
      // here so the coupled driver remains the sole owner of the time policy.
      matrix_parameters.initial_time     = initial_time;
      matrix_parameters.final_time       = final_time;
      matrix_parameters.time_step        = time_step;
      matrix_parameters.number_of_steps  = number_of_steps;
      matrix_parameters.output_frequency = 0;
      matrix_parameters.output_directory = output_directory + "/matrix";
      matrix_parameters.output_name      = "matrix";

      fiber_parameters.initial_time     = initial_time;
      fiber_parameters.final_time       = final_time;
      fiber_parameters.time_step        = time_step;
      fiber_parameters.number_of_steps  = number_of_steps;
      fiber_parameters.output_frequency = 0;
      fiber_parameters.output_directory = output_directory + "/fiber";
      fiber_parameters.output_name      = "fiber";
    });
  }


  template <int dim>
  FiberReinforcedElastodynamics<dim>::FiberReinforcedElastodynamics(
    const Parameters &parameters)
    : parameters(parameters)
    , matrix_problem_storage(parameters.matrix_parameters)
    , fiber_problem_storage(parameters.fiber_parameters)
    , current_time_storage(parameters.initial_time)
  {}


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::setup()
  {
    AssertThrow(!setup_complete,
                ExcMessage("The fiber-reinforced driver was already set up."));

    matrix_problem_storage.make_grid();
    matrix_problem_storage.setup_fe();
    matrix_problem_storage.setup_system();
    matrix_problem_storage.assemble_operators();

    fiber_problem_storage.make_grid();
    fiber_problem_storage.setup_fe();
    fiber_problem_storage.setup_system();
    fiber_problem_storage.assemble_operators();

    matrix_velocity_representation = std::make_unique<Representation>(
      matrix_problem_storage.triangulation(),
      matrix_problem_storage.dof_handler(),
      matrix_problem_storage.locally_owned_dofs(),
      matrix_problem_storage.locally_relevant_dofs(),
      matrix_problem_storage.velocity_constraints(),
      matrix_problem_storage.mapping(),
      dealii::FEValuesExtractors::Vector(0));
    fiber_velocity_representation = std::make_unique<Representation>(
      fiber_problem_storage.triangulation(),
      fiber_problem_storage.dof_handler(),
      fiber_problem_storage.locally_owned_dofs(),
      fiber_problem_storage.locally_relevant_dofs(),
      fiber_problem_storage.velocity_constraints(),
      fiber_problem_storage.mapping(),
      dealii::FEValuesExtractors::Vector(0));

    interaction_storage =
      std::make_unique<Interaction>(*matrix_velocity_representation,
                                    *fiber_velocity_representation,
                                    parameters.coupling_parameters);
    interaction_storage->assemble();
    multiplier_storage.reinit(
      interaction_storage->multiplier_locally_owned_dofs(), MPI_COMM_WORLD);
    multiplier_storage = 0.;

    setup_complete = true;
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::set_initial_conditions()
  {
    AssertThrow(setup_complete,
                ExcMessage("setup() must precede initial conditions."));

    matrix_problem_storage.set_initial_conditions();
    fiber_problem_storage.set_initial_conditions();
    current_time_storage     = parameters.initial_time;
    time_step_number_storage = 0;

    matrix_only_displacement_storage = matrix_problem_storage.displacement();

    VectorType                            compatibility;
    const std::vector<const VectorType *> states{
      &matrix_problem_storage.displacement(),
      &fiber_problem_storage.displacement()};
    interaction_storage->constraint_equation().residual(states, compatibility);
    residuals_storage.displacement_compatibility = compatibility.l2_norm();
    AssertThrow(
      residuals_storage.displacement_compatibility <=
        parameters.initial_compatibility_tolerance,
      ExcMessage("The configured initial displacement is incompatible with "
                 "the matrix/fiber coupling: norm = " +
                 std::to_string(residuals_storage.displacement_compatibility)));

    initial_conditions_set = true;
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::build_effective_matrices(const double dt)
  {
    matrix_effective_matrix.copy_from(matrix_problem_storage.mass_matrix());
    matrix_effective_matrix *= 1. / dt;
    matrix_effective_matrix.add(1., matrix_problem_storage.damping_matrix());
    matrix_effective_matrix.add(dt, matrix_problem_storage.stiffness_matrix());
    impose_homogeneous_constraints(
      matrix_effective_matrix, matrix_problem_storage.velocity_constraints());

    fiber_effective_matrix.copy_from(fiber_problem_storage.mass_matrix());
    fiber_effective_matrix *= 1. / dt;
    fiber_effective_matrix.add(1., fiber_problem_storage.damping_matrix());
    fiber_effective_matrix.add(dt, fiber_problem_storage.stiffness_matrix());
    impose_homogeneous_constraints(
      fiber_effective_matrix, fiber_problem_storage.velocity_constraints());

    schur_solver = std::make_unique<SchurSolver>(
      matrix_effective_matrix,
      fiber_effective_matrix,
      interaction_storage->coupling_matrix(),
      interaction_storage->pairing_matrix(),
      matrix_problem_storage.locally_owned_dofs(),
      fiber_problem_storage.locally_owned_dofs(),
      interaction_storage->multiplier_locally_owned_dofs(),
      MPI_COMM_WORLD,
      parameters.schur_max_steps,
      parameters.schur_tolerance,
      parameters.block_tolerance);
    effective_matrices_valid = true;
    effective_time_step      = dt;
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::build_effective_rhs(
    const Problem    &problem,
    const VectorType &previous_displacement,
    const VectorType &previous_velocity,
    const double      time,
    const double      dt,
    VectorType       &rhs) const
  {
    problem.body_force_at_time(time, rhs);

    VectorType temporary;
    temporary.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    problem.mass_matrix().vmult(temporary, previous_velocity);
    rhs.add(1. / dt, temporary);

    problem.stiffness_matrix().vmult(temporary, previous_displacement);
    rhs.add(-1., temporary);

    // The tutorial and the verification tests use homogeneous Dirichlet data.
    // Condensation is kept here so hanging-node and homogeneous essential
    // constraints are treated consistently with the effective block matrix.
    impose_homogeneous_constraints(rhs, problem.velocity_constraints());
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::advance_one_timestep()
  {
    AssertThrow(setup_complete && initial_conditions_set,
                ExcMessage("setup() and set_initial_conditions() must "
                           "precede a coupled step."));
    const double remaining = parameters.final_time - current_time_storage;
    AssertThrow(parameters.number_of_steps > 0 || remaining > 0.,
                ExcMessage("The coupled run has no remaining time."));

    double dt = parameters.time_step;
    if (parameters.number_of_steps == 0)
      dt = std::min(dt, remaining);
    AssertThrow(dt > 0., ExcMessage("The coupled time step must be positive."));

    if (!effective_matrices_valid ||
        std::abs(dt - effective_time_step) >
          10. * std::numeric_limits<double>::epsilon())
      build_effective_matrices(dt);

    const auto previous_matrix_displacement =
      matrix_problem_storage.displacement();
    const auto previous_matrix_velocity = matrix_problem_storage.velocity();
    const auto previous_fiber_displacement =
      fiber_problem_storage.displacement();
    const auto   previous_fiber_velocity = fiber_problem_storage.velocity();
    const double next_time               = current_time_storage + dt;

    VectorType matrix_rhs;
    VectorType fiber_rhs;
    build_effective_rhs(matrix_problem_storage,
                        previous_matrix_displacement,
                        previous_matrix_velocity,
                        next_time,
                        dt,
                        matrix_rhs);
    build_effective_rhs(fiber_problem_storage,
                        previous_fiber_displacement,
                        previous_fiber_velocity,
                        next_time,
                        dt,
                        fiber_rhs);

    VectorType next_matrix_velocity;
    VectorType next_fiber_velocity;

    // Keep a matrix-only reference under the same load and time policy.  This
    // is a diagnostic for the additive/excess interpretation, not a second
    // coupled solve and not part of the production state.
    VectorType matrix_only_velocity;
    matrix_only_velocity.reinit(matrix_problem_storage.locally_owned_dofs(),
                                MPI_COMM_WORLD);
    matrix_only_velocity = 0.;
    SolverControl matrix_only_control(parameters.schur_max_steps,
                                      parameters.block_tolerance);
    ImmersXLA::MPI::PreconditionJacobi matrix_only_preconditioner;
    matrix_only_preconditioner.initialize(matrix_effective_matrix);
    SolverGMRES<VectorType> matrix_only_solver(matrix_only_control);
    matrix_only_solver.solve(matrix_effective_matrix,
                             matrix_only_velocity,
                             matrix_rhs,
                             matrix_only_preconditioner);
    matrix_only_velocity *= dt;
    matrix_only_displacement_storage += matrix_only_velocity;

    schur_solver->solve(next_matrix_velocity,
                        next_fiber_velocity,
                        multiplier_storage,
                        matrix_rhs,
                        fiber_rhs);

    VectorType next_matrix_displacement = previous_matrix_displacement;
    VectorType next_fiber_displacement  = previous_fiber_displacement;
    next_matrix_velocity *= dt;
    next_fiber_velocity *= dt;
    next_matrix_displacement += next_matrix_velocity;
    next_fiber_displacement += next_fiber_velocity;
    next_matrix_velocity *= 1. / dt;
    next_fiber_velocity *= 1. / dt;

    const unsigned int next_step = time_step_number_storage + 1;
    matrix_problem_storage.accept_state(next_matrix_displacement,
                                        next_matrix_velocity,
                                        next_time,
                                        next_step);
    fiber_problem_storage.accept_state(next_fiber_displacement,
                                       next_fiber_velocity,
                                       next_time,
                                       next_step);
    current_time_storage     = next_time;
    time_step_number_storage = next_step;
    update_diagnostics(matrix_rhs, fiber_rhs);
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::update_diagnostics(
    const VectorType &matrix_rhs,
    const VectorType &fiber_rhs)
  {
    VectorType matrix_residual;
    VectorType fiber_residual;
    VectorType temporary;
    matrix_residual.reinit(matrix_problem_storage.locally_owned_dofs(),
                           MPI_COMM_WORLD);
    fiber_residual.reinit(fiber_problem_storage.locally_owned_dofs(),
                          MPI_COMM_WORLD);
    temporary.reinit(fiber_problem_storage.locally_owned_dofs(),
                     MPI_COMM_WORLD);

    matrix_effective_matrix.vmult(matrix_residual,
                                  matrix_problem_storage.velocity());
    interaction_storage->coupling_matrix().vmult_add(matrix_residual,
                                                     multiplier_storage);
    matrix_residual -= matrix_rhs;

    fiber_effective_matrix.vmult(fiber_residual,
                                 fiber_problem_storage.velocity());
    interaction_storage->pairing_matrix().Tvmult(temporary, multiplier_storage);
    fiber_residual -= temporary;
    fiber_residual -= fiber_rhs;

    const std::vector<const VectorType *> velocities{
      &matrix_problem_storage.velocity(), &fiber_problem_storage.velocity()};
    VectorType constraint_residual;
    interaction_storage->constraint_equation().residual(velocities,
                                                        constraint_residual);

    const std::vector<const VectorType *> displacements{
      &matrix_problem_storage.displacement(),
      &fiber_problem_storage.displacement()};
    VectorType displacement_residual;
    interaction_storage->constraint_equation().residual(displacements,
                                                        displacement_residual);

    residuals_storage.matrix_velocity     = matrix_residual.l2_norm();
    residuals_storage.fiber_velocity      = fiber_residual.l2_norm();
    residuals_storage.velocity_constraint = constraint_residual.l2_norm();
    residuals_storage.displacement_compatibility =
      displacement_residual.l2_norm();
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::output_results() const
  {
    matrix_problem_storage.output_results();
    fiber_problem_storage.output_results();

    ensure_directory(parameters.output_directory + "/fiber");
    dealii::DataOut<dim, dim> data_out;
    data_out.attach_dof_handler(fiber_problem_storage.dof_handler());

    VectorType relevant_multiplier;
    relevant_multiplier.reinit(
      interaction_storage->multiplier_locally_owned_dofs(),
      interaction_storage->multiplier_locally_relevant_dofs(),
      MPI_COMM_WORLD);
    relevant_multiplier = multiplier_storage;
    relevant_multiplier.update_ghost_values();

    const std::vector<std::string> names(dim, "lambda");
    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(relevant_multiplier,
                             names,
                             DataOut<dim, dim>::type_dof_data,
                             interpretation);
    data_out.build_patches();
    data_out.write_vtu_in_parallel(parameters.output_directory +
                                     "/fiber/lambda_" +
                                     std::to_string(time_step_number_storage) +
                                     ".vtu",
                                   MPI_COMM_WORLD);
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::run()
  {
    setup();
    set_initial_conditions();
    if (parameters.output_frequency > 0)
      output_results();

    unsigned int n_steps = parameters.number_of_steps;
    if (n_steps == 0 && parameters.final_time > parameters.initial_time)
      n_steps = static_cast<unsigned int>(
        std::ceil((parameters.final_time - parameters.initial_time) /
                  parameters.time_step));

    for (unsigned int step = 0; step < n_steps; ++step)
      {
        advance_one_timestep();
        if (parameters.output_frequency > 0 &&
            (time_step_number_storage % parameters.output_frequency == 0 ||
             step + 1 == n_steps))
          output_results();
      }
  }


  template <int dim>
  const typename FiberReinforcedElastodynamics<dim>::Interaction &
  FiberReinforcedElastodynamics<dim>::interaction() const
  {
    AssertThrow(interaction_storage != nullptr, ExcNotInitialized());
    return *interaction_storage;
  }


  template <int dim>
  const typename FiberReinforcedElastodynamics<dim>::VectorType &
  FiberReinforcedElastodynamics<dim>::multiplier() const
  {
    AssertThrow(interaction_storage != nullptr, ExcNotInitialized());
    return multiplier_storage;
  }


  template <int dim>
  double
  FiberReinforcedElastodynamics<dim>::fiber_excess_elastic_energy() const
  {
    VectorType product;
    product.reinit(fiber_problem_storage.locally_owned_dofs(), MPI_COMM_WORLD);
    fiber_problem_storage.stiffness_matrix().vmult(
      product, fiber_problem_storage.displacement());
    return 0.5 * (fiber_problem_storage.displacement() * product);
  }


  template <int dim>
  double
  FiberReinforcedElastodynamics<dim>::matrix_only_displacement_difference()
    const
  {
    VectorType difference = matrix_problem_storage.displacement();
    difference -= matrix_only_displacement_storage;
    return difference.l2_norm();
  }


  template class FiberReinforcedElastodynamicsParameters<2>;
  template class FiberReinforcedElastodynamicsParameters<3>;
  template class FiberReinforcedElastodynamics<2>;
  template class FiberReinforcedElastodynamics<3>;

} // namespace ImmersX
