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

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/affine_constraints.h>

#include <immersx/core/constraint.h>
#include <immersx/core/matrix_operator.h>
#include <immersx/core/observable.h>
#include <immersx/core/overlap_integration_support.h>
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
    template <int matrix_dim, int fiber_dim, int spacedim>
    void
    assemble_fiber_constraint_matrices(
      const FiniteElementSpaceView<matrix_dim, spacedim> &matrix_space,
      const FiniteElementSpaceView<fiber_dim, spacedim>  &fiber_space,
      const FiniteElementSpaceView<fiber_dim, spacedim>  &multiplier_space,
      std::shared_ptr<ImmersXLA::MPI::SparseMatrix>      &matrix_to_multiplier,
      std::shared_ptr<ImmersXLA::MPI::SparseMatrix>      &fiber_to_multiplier,
      std::shared_ptr<ImmersXLA::MPI::SparseMatrix>      &matrix_coupling,
      std::shared_ptr<ImmersXLA::MPI::SparseMatrix>      &outside_mass)
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
      const auto multiplier_observable = value(multiplier);
      using MultiplierObservable       = decltype(multiplier_observable);
      using MatrixType                 = ImmersXLA::MPI::SparseMatrix;
      using VectorType                 = ImmersXLA::MPI::Vector;

      GridTools::Cache<spacedim> background(
        matrix_space.dof_handler().get_triangulation(), matrix_space.mapping());
      const auto quadrature = compatibility::gauss_quadrature<fiber_dim>(
        multiplier_space.finite_element().reference_cell(),
        std::max({matrix_space.finite_element().degree,
                  fiber_space.finite_element().degree,
                  multiplier_space.finite_element().degree}) +
          1);
      const auto supports = make_overlap_supports(
        IntegrationSupport<fiber_dim, spacedim>(
          multiplier_space.dof_handler().get_triangulation()),
        multiplier_space.mapping(),
        quadrature,
        background);
      matrix_to_multiplier =
        detail::WeakAssembly<MatrixObservable, MultiplierObservable>::
          template assemble_on_support<VectorType, MatrixType>(
            matrix_observable, multiplier_observable, supports.overlap)
            .matrix;
      fiber_to_multiplier =
        detail::WeakAssembly<FiberObservable, MultiplierObservable>::
          template assemble_on_support<VectorType, MatrixType>(
            fiber_observable, multiplier_observable, supports.overlap)
            .matrix;
      outside_mass =
        detail::WeakAssembly<MultiplierObservable, MultiplierObservable>::
          template assemble_on_support<VectorType, MatrixType>(
            multiplier_observable, multiplier_observable, supports.outside)
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
          if (matrix.row_length(row) != 0)
            for (auto entry = matrix.begin(row), end = matrix.end(row);
                 entry != end;
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
    , time_parameters(normalize_subsection(subsection) + "Time interval/")
    , fixed_step_parameters(normalize_subsection(subsection) + "Fixed step/")
    , ida_parameters(normalize_subsection(subsection) + "IDA/")
    , matrix_parameters(normalize_subsection(subsection) +
                          "Matrix Elastodynamics/",
                        &time_parameters,
                        &fixed_step_parameters,
                        &ida_parameters)
    , fiber_parameters(normalize_subsection(subsection) +
                         "Fiber Elastodynamics/",
                       &time_parameters,
                       &fixed_step_parameters,
                       &ida_parameters)
    , multiplier_parameters(normalize_subsection(subsection) +
                              "Multiplier space/",
                            "auto")
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
  FiberReinforcedElastodynamics<dim>::prepare_matrix_problem()
  {
    AssertThrow(!matrix_setup_complete,
                ExcMessage("The matrix Problem was already prepared."));

    matrix_problem_storage.make_grid();
    matrix_problem_storage.setup_fe();
    matrix_problem_storage.setup_system();
    matrix_problem_storage.assemble_operators();

    matrix_setup_complete = true;
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::prepare_fiber_problem()
  {
    AssertThrow(!fiber_setup_complete,
                ExcMessage("The fiber Problem was already prepared."));

    fiber_problem_storage.make_grid();
    fiber_problem_storage.setup_fe();
    fiber_problem_storage.setup_system();
    fiber_problem_storage.assemble_operators();

    fiber_setup_complete = true;
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::prepare_velocity_continuity()
  {
    AssertThrow(matrix_setup_complete && fiber_setup_complete,
                ExcMessage("Both Problems must be prepared before the "
                           "velocity-continuity interaction."));
    AssertThrow(!coupling_setup_complete,
                ExcMessage("The velocity-continuity interaction was already "
                           "prepared."));

    const unsigned int degree = parameters.multiplier_degree == 0 ?
                                  fiber_problem_storage.fe().degree :
                                  parameters.multiplier_degree;
    multiplier_space_owner    = std::make_unique<FiniteElementSpace<1, dim>>(
      fiber_problem_storage.triangulation(),
      parameters.multiplier_parameters.finite_element == "auto" ?
           std::make_unique<dealii::FESystem<1, dim>>(dealii::FE_Q<1, dim>(degree),
                                                   dim) :
           dealii::FETools::get_fe_by_name<1, dim>(
          parameters.multiplier_parameters.finite_element));
    AssertThrow(multiplier_space_owner->finite_element().n_components() == dim,
                ExcMessage("The multiplier FE must have dim components."));
    auto &constraints = multiplier_space_owner->constraints();
    constraints.reinit(multiplier_space_owner->locally_owned_dofs(),
                       multiplier_space_owner->locally_relevant_dofs());
    dealii::DoFTools::make_hanging_node_constraints(
      multiplier_space_owner->dof_handler(), constraints);
    constraints.close();
    multiplier_space_storage = std::make_unique<FiniteElementSpaceView<1, dim>>(
      multiplier_space_owner->view(fiber_problem_storage.mapping()));

    multiplier_storage.reinit(multiplier_space_owner->locally_owned_dofs(),
                              MPI_COMM_WORLD);
    multiplier_storage = 0.;

    const auto multiplier_output_field =
      multiplier_space_storage->field(multiplier_output_layout_storage,
                                      "lagrange_multiplier",
                                      dealii::FEValuesExtractors::Vector(0));
    multiplier_output_field_id_storage = multiplier_output_field.id();
    multiplier_output_storage = std::make_unique<MultiplierOutputHandler>(
      *multiplier_space_storage,
      std::filesystem::path(parameters.output_directory) / "interaction",
      parameters.multiplier_output_name);
    multiplier_output_storage->add_field(multiplier_output_field);

    coupling_setup_complete = true;
    setup_complete          = true;
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::setup()
  {
    AssertThrow(!setup_complete,
                ExcMessage("The fiber-reinforced driver was already set up."));
    prepare_matrix_problem();
    prepare_fiber_problem();
    prepare_velocity_continuity();
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::assemble_coupling_matrices()
  {
    AssertThrow(setup_complete,
                ExcMessage("setup() must precede coupling assembly."));
    const auto matrix_space =
      fe_space(matrix_problem_storage.dof_handler(),
               matrix_problem_storage.mapping(),
               matrix_problem_storage.velocity_constraints(),
               &matrix_problem_storage.locally_relevant_dofs());
    const auto fiber_space =
      fe_space(fiber_problem_storage.dof_handler(),
               fiber_problem_storage.mapping(),
               fiber_problem_storage.velocity_constraints(),
               &fiber_problem_storage.locally_relevant_dofs());
    assemble_fiber_constraint_matrices(matrix_space,
                                       fiber_space,
                                       *multiplier_space_storage,
                                       matrix_to_multiplier_storage,
                                       fiber_to_multiplier_storage,
                                       matrix_coupling_storage,
                                       outside_mass_storage);
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::set_initial_conditions()
  {
    AssertThrow(coupling_setup_complete,
                ExcMessage("Problem and velocity-continuity preparation must "
                           "precede initial conditions."));

    if (!matrix_to_multiplier_storage)
      assemble_coupling_matrices();

    matrix_problem_storage.set_initial_conditions();
    fiber_problem_storage.set_initial_conditions();
    current_time_storage     = parameters.time_parameters.initial_time;
    time_step_number_storage = 0;

    residuals_storage.displacement_compatibility =
      continuity_defect(matrix_problem_storage.displacement(),
                        fiber_problem_storage.displacement());
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
      multiplier_space_owner->locally_owned_dofs(),
      MPI_COMM_WORLD,
      parameters.schur_max_steps,
      parameters.schur_tolerance,
      parameters.block_tolerance,
      outside_mass_storage.get());
    effective_matrices_valid = true;
    effective_time_step      = dt;
  }


  template <int dim>
  template <int problem_dim, int spacedim>
  void
  FiberReinforcedElastodynamics<dim>::build_effective_rhs(
    const ElastodynamicsSolver<problem_dim, spacedim> &problem,
    const VectorType                                  &previous_displacement,
    const VectorType                                  &previous_velocity,
    const double                                       time,
    const double                                       dt,
    VectorType                                        &rhs) const
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
    AssertThrow(parameters.fixed_step_parameters.number_of_steps > 0 ||
                  remaining > 0.,
                ExcMessage("The coupled run has no remaining time."));

    double dt = parameters.fixed_step_parameters.time_step;
    if (parameters.fixed_step_parameters.number_of_steps == 0)
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

    residuals_storage.matrix_velocity = matrix_residual.l2_norm();
    residuals_storage.fiber_velocity  = fiber_residual.l2_norm();
    VectorType constraint_residual;
    constraint_residual.reinit(multiplier_storage);
    matrix_to_multiplier_storage->vmult(constraint_residual,
                                        matrix_problem_storage.velocity());
    VectorType fiber_trace;
    fiber_trace.reinit(multiplier_storage);
    fiber_to_multiplier_storage->vmult(fiber_trace,
                                       fiber_problem_storage.velocity());
    constraint_residual -= fiber_trace;
    if (outside_mass_storage)
      {
        outside_mass_storage->vmult(fiber_trace, multiplier_storage);
        constraint_residual -= fiber_trace;
      }
    residuals_storage.velocity_constraint = constraint_residual.l2_norm();
    residuals_storage.displacement_compatibility =
      continuity_defect(matrix_problem_storage.displacement(),
                        fiber_problem_storage.displacement());
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::output_results() const
  {
    matrix_problem_storage.output_results();
    fiber_problem_storage.output_results();

    StateView<VectorType> output_state(multiplier_output_layout_storage,
                                       current_time_storage);
    output_state.bind(multiplier_output_field_id_storage, multiplier_storage);
    multiplier_output_storage->write(output_state, current_time_storage);
  }


#ifdef DEAL_II_WITH_SUNDIALS
  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::run_ida_execution()
  {
    AssertThrow(
      coupling_setup_complete,
      ExcMessage("Prepare both Problems and the multiplier space before IDA."));

    if (!initial_conditions_set)
      set_initial_conditions();

    IDAAdapterType adapter(parameters.time_parameters,
                           parameters.ida_parameters,
                           MPI_COMM_WORLD);
    const auto matrix = adapter.add(matrix_problem_storage, "matrix").fields();
    const auto fiber  = adapter.add(fiber_problem_storage, "fiber").fields();
    const auto lambda =
      adapter
        .add(make_continuity_constraint(
               matrix.velocity,
               fiber.velocity,
               multiplier_space_storage->field("lagrange_multiplier",
                                               FEValuesExtractors::Vector(0))),
             "fiber-coupling")
        .fields()
        .multiplier;

    auto       state            = adapter.make_state();
    auto       state_dot        = adapter.make_state();
    const auto constraint_block = [&](const auto &row, const auto &column) {
      const auto block =
        adapter.state_matrix_operator(state, row.id(), column.id());
      AssertThrow(
        block.has_value(),
        ExcMessage("The fiber Constraint needs matrix-based coupling blocks."));
      return block->matrix();
    };
    matrix_to_multiplier_storage = constraint_block(lambda, matrix.velocity);
    matrix_coupling_storage      = constraint_block(matrix.velocity, lambda);
    fiber_to_multiplier_storage  = std::make_shared<MatrixType>();
    fiber_to_multiplier_storage->copy_from(
      *constraint_block(lambda, fiber.velocity));
    *fiber_to_multiplier_storage *= -1.;
    const auto exterior =
      adapter.state_matrix_operator(state, lambda.id(), lambda.id());
    outside_mass_storage =
      exterior ? exterior->matrix() : std::shared_ptr<MatrixType>();
    if (outside_mass_storage)
      {
        auto positive = std::make_shared<MatrixType>();
        positive->copy_from(*outside_mass_storage);
        *positive *= -1.;
        outside_mass_storage = std::move(positive);
      }

    const auto initialize = [&](const auto &problem, const auto &fields) {
      adapter.field(state, fields.displacement)     = problem.displacement();
      adapter.field(state, fields.velocity)         = problem.velocity();
      adapter.field(state_dot, fields.displacement) = problem.velocity();
    };
    initialize(matrix_problem_storage, matrix);
    initialize(fiber_problem_storage, fiber);

    // The initial mass solve observes these matrices only within this scope.
    {
      MatrixType matrix_mass, fiber_mass;
      matrix_mass.copy_from(matrix_problem_storage.mass_matrix());
      fiber_mass.copy_from(fiber_problem_storage.mass_matrix());
      impose_homogeneous_constraints(
        matrix_mass, matrix_problem_storage.velocity_constraints());
      impose_homogeneous_constraints(
        fiber_mass, fiber_problem_storage.velocity_constraints());
      SchurSolver initial_solver(matrix_mass,
                                 fiber_mass,
                                 *matrix_coupling_storage,
                                 *fiber_to_multiplier_storage,
                                 matrix_problem_storage.locally_owned_dofs(),
                                 fiber_problem_storage.locally_owned_dofs(),
                                 multiplier_space_owner->locally_owned_dofs(),
                                 MPI_COMM_WORLD,
                                 parameters.schur_max_steps,
                                 parameters.schur_tolerance,
                                 parameters.block_tolerance);
      VectorType  matrix_rhs, fiber_rhs;
      matrix_problem_storage.body_force_at_time(
        parameters.time_parameters.initial_time, matrix_rhs);
      fiber_problem_storage.body_force_at_time(
        parameters.time_parameters.initial_time, fiber_rhs);
      impose_homogeneous_constraints(
        matrix_rhs, matrix_problem_storage.velocity_constraints());
      impose_homogeneous_constraints(
        fiber_rhs, fiber_problem_storage.velocity_constraints());
      VectorType reaction;
      initial_solver.solve(adapter.field(state_dot, matrix.velocity),
                           adapter.field(state_dot, fiber.velocity),
                           reaction,
                           matrix_rhs,
                           fiber_rhs);
    }

    MultiplierOutputHandler output(
      lambda,
      std::filesystem::path(parameters.output_directory) / "interaction",
      parameters.multiplier_output_name);
    adapter.set_output_step([this, &adapter, &output, matrix, fiber, lambda](
                              const double            time,
                              const GlobalVectorType &state,
                              const GlobalVectorType &state_dot,
                              const unsigned int      step) {
      matrix_problem_storage.accept_state(adapter.field(state,
                                                        matrix.displacement),
                                          adapter.field(state, matrix.velocity),
                                          time,
                                          step);
      fiber_problem_storage.accept_state(adapter.field(state,
                                                       fiber.displacement),
                                         adapter.field(state, fiber.velocity),
                                         time,
                                         step);
      multiplier_storage       = adapter.field(state, lambda);
      current_time_storage     = time;
      time_step_number_storage = step;
      auto residual            = adapter.make_state();
      adapter.solver().residual(time, state, state_dot, residual);
      residuals_storage.matrix_velocity =
        adapter.field(residual, matrix.velocity).l2_norm();
      residuals_storage.fiber_velocity =
        adapter.field(residual, fiber.velocity).l2_norm();
      residuals_storage.velocity_constraint =
        adapter.field(residual, lambda).l2_norm();
      residuals_storage.displacement_compatibility =
        continuity_defect(matrix_problem_storage.displacement(),
                          fiber_problem_storage.displacement());
      matrix_problem_storage.output_results();
      fiber_problem_storage.output_results();
      adapter.write_output(output, state, time);
    });
    adapter.solve(state, state_dot);
  }
#endif


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::run_execution()
  {
#ifdef DEAL_II_WITH_SUNDIALS
    run_ida_execution();
    return;
#endif

    set_initial_conditions();

    unsigned int n_steps = parameters.fixed_step_parameters.number_of_steps;
    if (n_steps == 0 && parameters.time_parameters.final_time >
                          parameters.time_parameters.initial_time)
      n_steps = static_cast<unsigned int>(
        std::ceil((parameters.time_parameters.final_time -
                   parameters.time_parameters.initial_time) /
                  parameters.fixed_step_parameters.time_step));

    if (parameters.time_parameters.output_time_interval > 0)
      output_results();
    double next_output_time = parameters.time_parameters.initial_time +
                              parameters.time_parameters.output_time_interval;
    for (unsigned int step = 0; step < n_steps; ++step)
      {
        advance_one_timestep();
        if (parameters.time_parameters.output_time_interval > 0 &&
            (current_time_storage >= next_output_time || step + 1 == n_steps))
          {
            output_results();
            while (next_output_time <= current_time_storage)
              next_output_time +=
                parameters.time_parameters.output_time_interval;
          }
      }
  }


  template <int dim>
  void
  FiberReinforcedElastodynamics<dim>::run()
  {
    setup();
    run_execution();
  }


  template <int dim>
  const typename FiberReinforcedElastodynamics<dim>::VectorType &
  FiberReinforcedElastodynamics<dim>::multiplier() const
  {
    AssertThrow(multiplier_space_owner != nullptr, ExcNotInitialized());
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
  FiberReinforcedElastodynamics<dim>::continuity_defect(
    const VectorType &matrix_values,
    const VectorType &fiber_values) const
  {
    VectorType matrix_trace, fiber_trace;
    matrix_trace.reinit(multiplier_storage);
    fiber_trace.reinit(multiplier_storage);
    matrix_to_multiplier_storage->vmult(matrix_trace, matrix_values);
    fiber_to_multiplier_storage->vmult(fiber_trace, fiber_values);
    matrix_trace -= fiber_trace;
    return matrix_trace.l2_norm();
  }


  template class FiberReinforcedElastodynamicsParameters<2>;
  template class FiberReinforcedElastodynamicsParameters<3>;
  template class FiberReinforcedElastodynamics<2>;
  template class FiberReinforcedElastodynamics<3>;

} // namespace ImmersX
