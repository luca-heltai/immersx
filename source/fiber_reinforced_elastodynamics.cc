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

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/core/constraint.h>
#include <immersx/core/matrix_operator.h>
#include <immersx/core/observable.h>
#include <immersx/core/weak_term.h>
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
    template <int dim>
    using FiberVectorField =
      Field<dim, dim, dealii::FEValuesExtractors::Vector>;

    template <int dim>
    void
    assemble_fiber_constraint_matrices(
      const FESpaceView<dim, dim>                   &matrix_space,
      const FESpaceView<dim, dim>                   &fiber_space,
      const FESpaceView<dim, dim>                   &multiplier_space,
      std::shared_ptr<ImmersXLA::MPI::SparseMatrix> &matrix_to_multiplier,
      std::shared_ptr<ImmersXLA::MPI::SparseMatrix> &fiber_to_multiplier,
      std::shared_ptr<ImmersXLA::MPI::SparseMatrix> &matrix_coupling)
    {
      const auto matrix_velocity =
        matrix_space.field(FieldId(0),
                           "matrix_velocity",
                           dealii::FEValuesExtractors::Vector(0));
      const auto fiber_velocity =
        fiber_space.field(FieldId(0),
                          "fiber_velocity",
                          dealii::FEValuesExtractors::Vector(0));
      const auto multiplier =
        multiplier_space.field(FieldId(0),
                               "velocity_multiplier",
                               dealii::FEValuesExtractors::Vector(0));

      const auto matrix_observable     = value(matrix_velocity);
      const auto fiber_observable      = value(fiber_velocity);
      using MatrixObservable           = decltype(matrix_observable);
      using FiberObservable            = decltype(fiber_observable);
      using MultiplierField            = FiberVectorField<dim>;
      const auto multiplier_observable = value(multiplier);
      using MultiplierObservable       = decltype(multiplier_observable);
      using MatrixType                 = ImmersXLA::MPI::SparseMatrix;
      using VectorType                 = ImmersXLA::MPI::Vector;

      matrix_to_multiplier =
        detail::WeakAssembly<MatrixObservable, MultiplierObservable>::
          template assemble<VectorType, MatrixType>(matrix_observable,
                                                    multiplier_observable)
            .matrix;
      fiber_to_multiplier =
        detail::WeakAssembly<FiberObservable, MultiplierObservable>::
          template assemble<VectorType, MatrixType>(fiber_observable,
                                                    multiplier_observable)
            .matrix;
      const auto matrix_operator =
        ImmersX::matrix_operator<VectorType, MatrixType>(*matrix_to_multiplier);
      matrix_coupling = ImmersX::transpose_operator(matrix_operator).matrix();
    }

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
    , time_parameters(normalize_subsection(subsection) + "Time parameters/")
    , matrix_parameters(normalize_subsection(subsection) +
                          "Matrix Elastodynamics/",
                        &time_parameters)
    , fiber_parameters(normalize_subsection(subsection) +
                         "Fiber Elastodynamics/",
                       &time_parameters)
  {
    add_parameter("Output directory", output_directory);
    add_parameter("Multiplier output name", multiplier_output_name);
    add_parameter("Multiplier FE degree",
                  multiplier_degree,
                  "Zero follows the fiber FE degree.",
                  prm,
                  Patterns::Integer(0));
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
      matrix_parameters.output_directory = output_directory + "/matrix";
      matrix_parameters.output_name      = "matrix";

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
    , current_time_storage(parameters.time_parameters.initial_time)
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

    matrix_space_storage = std::make_unique<FESpaceView<dim, dim>>(
      fe_space(matrix_problem_storage.dof_handler(),
               matrix_problem_storage.mapping(),
               matrix_problem_storage.velocity_constraints(),
               &matrix_problem_storage.locally_relevant_dofs()));
    fiber_space_storage = std::make_unique<FESpaceView<dim, dim>>(
      fe_space(fiber_problem_storage.dof_handler(),
               fiber_problem_storage.mapping(),
               fiber_problem_storage.velocity_constraints(),
               &fiber_problem_storage.locally_relevant_dofs()));

    multiplier_dof_handler_storage = std::make_unique<dealii::DoFHandler<dim>>(
      fiber_problem_storage.triangulation());
    const unsigned int degree = parameters.multiplier_degree == 0 ?
                                  fiber_problem_storage.fe().degree :
                                  parameters.multiplier_degree;
    multiplier_fe_storage =
      std::make_unique<dealii::FESystem<dim>>(dealii::FE_Q<dim>(degree), dim);
    multiplier_dof_handler_storage->distribute_dofs(*multiplier_fe_storage);
    multiplier_constraints_storage =
      std::make_unique<dealii::AffineConstraints<double>>();
    multiplier_constraints_storage->reinit(
      multiplier_dof_handler_storage->locally_owned_dofs(),
      dealii::DoFTools::extract_locally_relevant_dofs(
        *multiplier_dof_handler_storage));
    dealii::DoFTools::make_hanging_node_constraints(
      *multiplier_dof_handler_storage, *multiplier_constraints_storage);
    multiplier_constraints_storage->close();
    multiplier_relevant_storage = std::make_unique<dealii::IndexSet>(
      dealii::DoFTools::extract_locally_relevant_dofs(
        *multiplier_dof_handler_storage));
    multiplier_space_storage = std::make_unique<FESpaceView<dim, dim>>(
      fe_space(*multiplier_dof_handler_storage,
               fiber_problem_storage.mapping(),
               *multiplier_constraints_storage,
               multiplier_relevant_storage.get()));

    multiplier_storage.reinit(
      multiplier_dof_handler_storage->locally_owned_dofs(), MPI_COMM_WORLD);
    multiplier_storage = 0.;

    setup_complete = true;
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::assemble_coupling_matrices()
  {
    AssertThrow(setup_complete,
                ExcMessage("setup() must precede coupling assembly."));
    assemble_fiber_constraint_matrices(*matrix_space_storage,
                                       *fiber_space_storage,
                                       *multiplier_space_storage,
                                       matrix_to_multiplier_storage,
                                       fiber_to_multiplier_storage,
                                       matrix_coupling_storage);
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::set_initial_conditions()
  {
    AssertThrow(setup_complete,
                ExcMessage("setup() must precede initial conditions."));

    if (!matrix_to_multiplier_storage)
      assemble_coupling_matrices();

    matrix_problem_storage.set_initial_conditions();
    fiber_problem_storage.set_initial_conditions();
    current_time_storage     = parameters.time_parameters.initial_time;
    time_step_number_storage = 0;

    matrix_only_displacement_storage = matrix_problem_storage.displacement();

    VectorType compatibility;
    VectorType fiber_compatibility;
    compatibility.reinit(multiplier_storage);
    fiber_compatibility.reinit(multiplier_storage);
    matrix_to_multiplier_storage->vmult(compatibility,
                                        matrix_problem_storage.displacement());
    fiber_to_multiplier_storage->vmult(fiber_compatibility,
                                       fiber_problem_storage.displacement());
    compatibility -= fiber_compatibility;
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
      *matrix_coupling_storage,
      *fiber_to_multiplier_storage,
      matrix_problem_storage.locally_owned_dofs(),
      fiber_problem_storage.locally_owned_dofs(),
      multiplier_dof_handler_storage->locally_owned_dofs(),
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
    const double remaining =
      parameters.time_parameters.final_time - current_time_storage;
    AssertThrow(parameters.time_parameters.number_of_steps > 0 ||
                  remaining > 0.,
                ExcMessage("The coupled run has no remaining time."));

    double dt = parameters.time_parameters.time_step;
    if (parameters.time_parameters.number_of_steps == 0)
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
    matrix_coupling_storage->vmult_add(matrix_residual, multiplier_storage);
    matrix_residual -= matrix_rhs;

    fiber_effective_matrix.vmult(fiber_residual,
                                 fiber_problem_storage.velocity());
    fiber_to_multiplier_storage->Tvmult(temporary, multiplier_storage);
    fiber_residual -= temporary;
    fiber_residual -= fiber_rhs;

    VectorType constraint_residual;
    VectorType fiber_constraint;
    constraint_residual.reinit(multiplier_storage);
    fiber_constraint.reinit(multiplier_storage);
    matrix_to_multiplier_storage->vmult(constraint_residual,
                                        matrix_problem_storage.velocity());
    fiber_to_multiplier_storage->vmult(fiber_constraint,
                                       fiber_problem_storage.velocity());
    constraint_residual -= fiber_constraint;

    VectorType displacement_residual;
    displacement_residual.reinit(multiplier_storage);
    fiber_constraint = 0.;
    matrix_to_multiplier_storage->vmult(displacement_residual,
                                        matrix_problem_storage.displacement());
    fiber_to_multiplier_storage->vmult(fiber_constraint,
                                       fiber_problem_storage.displacement());
    displacement_residual -= fiber_constraint;

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

    const auto output_directory =
      std::filesystem::path(parameters.output_directory) / "interaction";
    std::filesystem::create_directories(output_directory);
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(*multiplier_dof_handler_storage);
    const std::vector<std::string> names(dim, "lagrange_multiplier");
    const std::vector<
      dealii::DataComponentInterpretation::DataComponentInterpretation>
      interpretation(
        dim, dealii::DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(multiplier_storage,
                             names,
                             dealii::DataOut<dim>::type_dof_data,
                             interpretation);
    data_out.build_patches(fiber_problem_storage.mapping());

    const auto filename = parameters.multiplier_output_name + "_" +
                          std::to_string(time_step_number_storage) + ".vtu";
    data_out.write_vtu_in_parallel((output_directory / filename).string(),
                                   MPI_COMM_WORLD);
    multiplier_output_records_storage.emplace_back(current_time_storage,
                                                   filename);
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::ofstream pvd(output_directory /
                          (parameters.multiplier_output_name + ".pvd"));
        dealii::DataOutBase::write_pvd_record(
          pvd, multiplier_output_records_storage);
      }
  }


#ifdef DEAL_II_WITH_SUNDIALS
  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::setup_ida()
  {
    using Adapter = IDAAdapterType;

    ida_storage =
      std::make_unique<Adapter>(parameters.time_parameters, MPI_COMM_WORLD);
    const auto matrix_fields =
      ida_storage->add(matrix_problem_storage, "matrix");
    const auto fiber_fields = ida_storage->add(fiber_problem_storage, "fiber");
    const auto matrix_velocity =
      matrix_space_storage->field(matrix_fields.fields().velocity,
                                  "matrix_velocity",
                                  dealii::FEValuesExtractors::Vector(0));
    const auto fiber_velocity =
      fiber_space_storage->field(fiber_fields.fields().velocity,
                                 "fiber_velocity",
                                 dealii::FEValuesExtractors::Vector(0));
    const auto multiplier =
      multiplier_space_storage->field("velocity_multiplier",
                                      dealii::FEValuesExtractors::Vector(0));
    const auto constraint =
      make_constraint(weak_term(value(matrix_velocity), multiplier) -
                      weak_term(value(fiber_velocity), multiplier));
    const auto coupling_fields = ida_storage->add(constraint, "fiber-coupling");

    matrix_fields_storage   = matrix_fields.fields();
    fiber_fields_storage    = fiber_fields.fields();
    coupling_fields_storage = coupling_fields.fields();

    const auto state = ida_storage->make_state();
    const auto matrix_to_multiplier =
      ida_storage->state_matrix_operator(state,
                                         coupling_fields_storage.multiplier,
                                         matrix_fields_storage.velocity);
    const auto negative_fiber_to_multiplier =
      ida_storage->state_matrix_operator(state,
                                         coupling_fields_storage.multiplier,
                                         fiber_fields_storage.velocity);
    const auto matrix_coupling =
      ida_storage->state_matrix_operator(state,
                                         matrix_fields_storage.velocity,
                                         coupling_fields_storage.multiplier);
    AssertThrow(
      matrix_to_multiplier.has_value() &&
        negative_fiber_to_multiplier.has_value() && matrix_coupling.has_value(),
      dealii::ExcMessage(
        "The fiber Constraint must have matrix-based coupling blocks."));
    matrix_to_multiplier_storage = matrix_to_multiplier->matrix();
    fiber_to_multiplier_storage  = std::make_shared<MatrixType>();
    fiber_to_multiplier_storage->copy_from(
      *negative_fiber_to_multiplier->matrix());
    *fiber_to_multiplier_storage *= -1.;
    matrix_coupling_storage = matrix_coupling->matrix();

    ida_storage->set_output_step([this](const double            time,
                                        const GlobalVectorType &state,
                                        const GlobalVectorType &state_dot,
                                        const unsigned int      step) {
      update_from_ida_state(state, state_dot, time, step);
      if ((parameters.time_parameters.output_frequency == 0 &&
           (step == 0 || time >= parameters.time_parameters.final_time)) ||
          (parameters.time_parameters.output_frequency > 0 &&
           (step % parameters.time_parameters.output_frequency == 0 ||
            time >= parameters.time_parameters.final_time)))
        output_results();
    });
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::update_from_ida_state(
    const GlobalVectorType &state,
    const GlobalVectorType &state_dot,
    const double            time,
    const unsigned int      step)
  {
    current_time_storage     = time;
    time_step_number_storage = step;

    matrix_problem_storage.accept_state(
      ida_storage->field(state, matrix_fields_storage.displacement),
      ida_storage->field(state, matrix_fields_storage.velocity),
      time,
      step);
    fiber_problem_storage.accept_state(
      ida_storage->field(state, fiber_fields_storage.displacement),
      ida_storage->field(state, fiber_fields_storage.velocity),
      time,
      step);
    multiplier_storage =
      ida_storage->field(state, coupling_fields_storage.multiplier);

    auto residual = ida_storage->make_state();
    ida_storage->solver().residual(time, state, state_dot, residual);
    residuals_storage.matrix_velocity =
      ida_storage->field(residual, matrix_fields_storage.velocity).l2_norm();
    residuals_storage.fiber_velocity =
      ida_storage->field(residual, fiber_fields_storage.velocity).l2_norm();

    VectorType velocity_constraint;
    VectorType fiber_constraint;
    velocity_constraint.reinit(multiplier_storage);
    fiber_constraint.reinit(multiplier_storage);
    matrix_to_multiplier_storage->vmult(velocity_constraint,
                                        matrix_problem_storage.velocity());
    fiber_to_multiplier_storage->vmult(fiber_constraint,
                                       fiber_problem_storage.velocity());
    velocity_constraint -= fiber_constraint;
    residuals_storage.velocity_constraint = velocity_constraint.l2_norm();

    VectorType displacement_constraint;
    displacement_constraint.reinit(multiplier_storage);
    matrix_to_multiplier_storage->vmult(displacement_constraint,
                                        matrix_problem_storage.displacement());
    fiber_to_multiplier_storage->vmult(fiber_constraint,
                                       fiber_problem_storage.displacement());
    displacement_constraint -= fiber_constraint;
    residuals_storage.displacement_compatibility =
      displacement_constraint.l2_norm();
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::initialize_ida_derivative(
    GlobalVectorType &state_dot)
  {
    MatrixType matrix_mass;
    matrix_mass.copy_from(matrix_problem_storage.mass_matrix());
    impose_homogeneous_constraints(
      matrix_mass, matrix_problem_storage.velocity_constraints());

    MatrixType fiber_mass;
    fiber_mass.copy_from(fiber_problem_storage.mass_matrix());
    impose_homogeneous_constraints(
      fiber_mass, fiber_problem_storage.velocity_constraints());

    // The initial-derivative solver observes these local mass matrices, so it
    // must not outlive this function.  The persistent Schur solver is reserved
    // for the backward-Euler path, where it observes member matrices.
    SchurSolver initial_solver(
      matrix_mass,
      fiber_mass,
      *matrix_coupling_storage,
      *fiber_to_multiplier_storage,
      matrix_problem_storage.locally_owned_dofs(),
      fiber_problem_storage.locally_owned_dofs(),
      multiplier_dof_handler_storage->locally_owned_dofs(),
      MPI_COMM_WORLD,
      parameters.schur_max_steps,
      parameters.schur_tolerance,
      parameters.block_tolerance);

    VectorType matrix_rhs;
    VectorType fiber_rhs;
    matrix_problem_storage.body_force_at_time(
      parameters.time_parameters.initial_time, matrix_rhs);
    fiber_problem_storage.body_force_at_time(
      parameters.time_parameters.initial_time, fiber_rhs);
    impose_homogeneous_constraints(
      matrix_rhs, matrix_problem_storage.velocity_constraints());
    impose_homogeneous_constraints(
      fiber_rhs, fiber_problem_storage.velocity_constraints());

    VectorType matrix_acceleration;
    VectorType fiber_acceleration;
    VectorType multiplier;
    initial_solver.solve(matrix_acceleration,
                         fiber_acceleration,
                         multiplier,
                         matrix_rhs,
                         fiber_rhs);
    ida_storage->field(state_dot, matrix_fields_storage.velocity) =
      matrix_acceleration;
    ida_storage->field(state_dot, fiber_fields_storage.velocity) =
      fiber_acceleration;
    ida_storage->field(state_dot, coupling_fields_storage.multiplier) = 0.;
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::run_with_ida()
  {
    setup_ida();

    auto state     = ida_storage->make_state();
    auto state_dot = ida_storage->make_state();
    ida_storage->field(state, matrix_fields_storage.displacement) =
      matrix_problem_storage.displacement();
    ida_storage->field(state, matrix_fields_storage.velocity) =
      matrix_problem_storage.velocity();
    ida_storage->field(state, fiber_fields_storage.displacement) =
      fiber_problem_storage.displacement();
    ida_storage->field(state, fiber_fields_storage.velocity) =
      fiber_problem_storage.velocity();
    ida_storage->field(state, coupling_fields_storage.multiplier) = 0.;

    ida_storage->field(state_dot, matrix_fields_storage.displacement) =
      matrix_problem_storage.velocity();
    ida_storage->field(state_dot, matrix_fields_storage.velocity) = 0.;
    ida_storage->field(state_dot, fiber_fields_storage.displacement) =
      fiber_problem_storage.velocity();
    ida_storage->field(state_dot, fiber_fields_storage.velocity)      = 0.;
    ida_storage->field(state_dot, coupling_fields_storage.multiplier) = 0.;
    initialize_ida_derivative(state_dot);

    ida_storage->solve(state, state_dot);
    matrix_only_displacement_storage = matrix_problem_storage.displacement();
  }
#endif


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::run()
  {
    setup();
#ifdef DEAL_II_WITH_SUNDIALS
    run_with_ida();
    return;
#endif

    set_initial_conditions();

    unsigned int n_steps = parameters.time_parameters.number_of_steps;
    if (n_steps == 0 && parameters.time_parameters.final_time >
                          parameters.time_parameters.initial_time)
      n_steps = static_cast<unsigned int>(
        std::ceil((parameters.time_parameters.final_time -
                   parameters.time_parameters.initial_time) /
                  parameters.time_parameters.time_step));

    if (parameters.time_parameters.output_frequency > 0)
      output_results();
    for (unsigned int step = 0; step < n_steps; ++step)
      {
        advance_one_timestep();
        if (parameters.time_parameters.output_frequency > 0 &&
            (time_step_number_storage %
                 parameters.time_parameters.output_frequency ==
               0 ||
             step + 1 == n_steps))
          output_results();
      }
  }


  template <int dim>
  const typename FiberReinforcedElastodynamics<dim>::VectorType &
  FiberReinforcedElastodynamics<dim>::multiplier() const
  {
    AssertThrow(multiplier_dof_handler_storage != nullptr, ExcNotInitialized());
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
