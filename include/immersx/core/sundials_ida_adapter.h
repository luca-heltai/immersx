// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_sundials_ida_adapter_h
#define immersx_sundials_ida_adapter_h

#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>

#include <immersx/core/detail/execution_composition.h>
#include <immersx/core/problem_handle.h>
#include <immersx/core/time_parameters.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef DEAL_II_WITH_SUNDIALS
#  include <immersx/compatibility/dealii_9_9/ida.h>
#endif

#ifdef DEAL_II_WITH_SUNDIALS
namespace ImmersX
{
  /** Public transient DAE adapter backed by the private composition engine. */
  template <typename FieldVectorType, typename GlobalVectorType>
  class IDAAdapter
  {
  public:
    using Composition =
      detail::ExecutionComposition<FieldVectorType, GlobalVectorType>;
    using Operator            = dealii::LinearOperator<GlobalVectorType>;
    using LocalOperator       = dealii::LinearOperator<FieldVectorType>;
    using SaddlePointMetadata = typename Composition::SaddlePointMetadata;
    using LinearSolveFunction = std::function<void(const Operator &,
                                                   const GlobalVectorType &,
                                                   GlobalVectorType &,
                                                   double)>;
    using OutputFunction      = std::function<void(const double,
                                              const GlobalVectorType &,
                                              const GlobalVectorType &,
                                              const unsigned int)>;
    using ConsistentInitialConditionsFunction =
      std::function<void(const double, GlobalVectorType &, GlobalVectorType &)>;
    using RestartFunction =
      std::function<bool(const double, GlobalVectorType &, GlobalVectorType &)>;
    using AdditionalData =
      typename Compatibility::SUNDIALS::IDA<GlobalVectorType>::AdditionalData;
    enum class LinearSolver
    {
      iterative,
      umfpack,
      mumps
    };

    IDAAdapter(const TimeIntervalParameters &time_parameters,
               const IDAParameters          &ida_parameters,
               const MPI_Comm                communicator,
               LinearSolveFunction           solve = {})
      : composition_(communicator)
      , solve_(std::move(solve))
      , time_parameters_(time_parameters)
      , ida_parameters_(ida_parameters)
      , pcout(std::cout,
              dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
      , computing_timer(communicator,
                        pcout,
                        dealii::TimerOutput::summary,
                        dealii::TimerOutput::wall_times)
    {}

    /** Set a constant absolute tolerance for a semantic Field. */
    void
    set_absolute_tolerance(const FieldId field, const double tolerance)
    {
      AssertThrow(std::isfinite(tolerance) && tolerance > 0.,
                  dealii::ExcMessage(
                    "Absolute tolerances must be finite and positive."));
      set_local_tolerances(field, [tolerance](FieldVectorType &values) {
        values = tolerance;
      });
    }

    template <typename SemanticField>
    void
    set_absolute_tolerance(const SemanticField &field, const double tolerance)
    {
      set_absolute_tolerance(field.id(), tolerance);
    }

    /** Fill locally owned absolute tolerances for one Field. Called on every
     * IDA initialization/reset, after the execution layout has been rebuilt.
     * Unspecified Fields retain the configured scalar absolute tolerance.
     */
    void
    set_local_tolerances(const FieldId                          field,
                         std::function<void(FieldVectorType &)> fill)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "Configure tolerances before solver initialization."));
      AssertThrow(static_cast<bool>(fill),
                  dealii::ExcMessage("Empty local tolerance callback."));
      AssertThrow(field.value() < composition_.n_fields(),
                  dealii::ExcMessage("Unknown tolerance Field."));
      for (auto &entry : local_tolerances_)
        if (entry.first == field)
          {
            entry.second = std::move(fill);
            return;
          }
      local_tolerances_.emplace_back(field, std::move(fill));
    }

    template <typename SemanticField>
    void
    set_local_tolerances(const SemanticField                   &field,
                         std::function<void(FieldVectorType &)> fill)
    {
      set_local_tolerances(field.id(), std::move(fill));
    }

    /** IDA counters accumulated across adapter-managed restarts. */
    struct Statistics
    {
      long int steps = 0, residual_evaluations = 0, nonlinear_iterations = 0,
               error_test_failures = 0, convergence_failures = 0;
      unsigned long jacobian_setups = 0, linear_solves = 0, restarts = 0,
                    weight_evaluations       = 0;
      int                       order        = 0;
      double                    current_time = 0., last_step = 0.;
      std::pair<double, double> tolerance_scales = {1., 1.};
    };

    Statistics
    statistics() const
    {
      Statistics result = accumulated_statistics_;
      if (ida_ && ida_->get_ida_memory())
        {
          auto     memory = ida_->get_ida_memory();
          long int value;
          check_ida(IDAGetNumSteps(memory, &value));
          result.steps += value;
          check_ida(IDAGetNumResEvals(memory, &value));
          result.residual_evaluations += value;
          check_ida(IDAGetNumNonlinSolvIters(memory, &value));
          result.nonlinear_iterations += value;
          check_ida(IDAGetNumErrTestFails(memory, &value));
          result.error_test_failures += value;
          check_ida(IDAGetNumNonlinSolvConvFails(memory, &value));
          result.convergence_failures += value;
          check_ida(IDAGetCurrentOrder(memory, &result.order));
          check_ida(IDAGetCurrentTime(memory, &result.current_time));
          check_ida(IDAGetLastStep(memory, &result.last_step));
        }
      result.jacobian_setups    = jacobian_setups_;
      result.linear_solves      = linear_solves_;
      result.restarts           = restarts_;
      result.weight_evaluations = weight_evaluations_;
      result.tolerance_scales =
        tolerance_ramp_.scales(additional_data().initial_time,
                               result.current_time);
      return result;
    }

    /** Select the coupled linear solver before initialization.
     * Direct solvers reuse their factorization until IDA requests a new
     * Jacobian setup. UMFPACK requires one rank; MUMPS uses the composition's
     * communicator and distributed matrix.
     */
    void
    set_linear_solver(const LinearSolver solver)
    {
      AssertThrow(
        !connected_,
        dealii::ExcMessage(
          "The linear solver must be selected before solver initialization."));
      if (solver == LinearSolver::umfpack)
        {
          AssertThrow(dealii::Utilities::MPI::n_mpi_processes(
                        composition_.communicator()) == 1,
                      dealii::ExcMessage(
                        "The UMFPACK solver requires one MPI rank."));
#  ifndef DEAL_II_WITH_UMFPACK
          AssertThrow(
            false,
            dealii::ExcMessage(
              "The UMFPACK solver requires deal.II configured with UMFPACK."));
#  endif
        }
      else if (solver == LinearSolver::mumps)
        {
#  ifndef DEAL_II_WITH_MUMPS
          AssertThrow(
            false,
            dealii::ExcMessage(
              "The MUMPS solver requires deal.II configured with MUMPS."));
#  endif
        }
      AssertThrow(
        solver == LinearSolver::iterative || !solve_,
        dealii::ExcMessage(
          "A direct solver cannot be combined with a custom linear solver."));
      linear_solver_ = solver;
    }

    /** Select serial UMFPACK, or restore the default iterative solver. */
    void
    set_serial_direct_solver(const bool enabled = true)
    {
      set_linear_solver(enabled ? LinearSolver::umfpack :
                                  LinearSolver::iterative);
    }

    /** Include one off-diagonal primal step in the Schur approximation. */
    void
    set_schur_primal_coupling(const bool enabled = true)
    {
      AssertThrow(
        !connected_,
        dealii::ExcMessage(
          "Schur controls must be configured before solver initialization."));
      include_schur_primal_coupling_ = enabled;
    }

    /** Configure IDA controls before solver() or solve() connects callbacks. */
    void
    set_additional_data(const AdditionalData &data)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "IDAAdapter controls must be configured before solver "
                    "initialization."));
      additional_data_ = data;
    }

    const AdditionalData &
    additional_data() const
    {
      if (!additional_data_.has_value())
        additional_data_ =
          ida_parameters_.template additional_data<GlobalVectorType>(
            time_parameters_);
      return *additional_data_;
    }

    template <typename Problem, typename... Arguments>
    auto
    add(const Problem     &problem,
        const std::string &prefix = {},
        const Arguments &...arguments)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "IDAAdapter contributors must be added before reinit or "
                    "solve."));
      auto fields = composition_.add(problem, prefix, arguments...);
      pcout << "IDAAdapter: registered contributor";
      if (!prefix.empty())
        pcout << " '" << prefix << "'";
      pcout << "; " << composition_.n_fields()
            << " semantic field(s) available." << std::endl;
      return ProblemHandle<IDAAdapter, decltype(fields)>(*this,
                                                         std::move(fields));
    }

    void
    reinit(GlobalVectorType &vector)
    {
      composition_.reinit(vector);
      pcout << "IDAAdapter: execution vector initialized with "
            << composition_.n_fields() << " semantic field(s)." << std::endl;
    }

    Compatibility::SUNDIALS::IDA<GlobalVectorType> &
    solver()
    {
      finalize();
      return *ida_;
    }

    unsigned int
    solve(GlobalVectorType &state, GlobalVectorType &state_dot)
    {
      current_state_          = &state;
      accumulated_statistics_ = {};
      jacobian_setups_ = linear_solves_ = restarts_ = weight_evaluations_ = 0;
      if (compute_consistent_initial_conditions_ &&
          additional_data().ic_type == AdditionalData::use_y_diff)
        {
          pcout << "IDAAdapter: computing consistent initial conditions "
                   "with the application callback."
                << std::endl;
          compute_consistent_initial_conditions_(additional_data().initial_time,
                                                 state,
                                                 state_dot);
        }
      finalize();
      pcout << "IDAAdapter: starting DAE solve with " << composition_.n_fields()
            << " semantic field(s)." << std::endl;
      dealii::TimerOutput::Scope timer(computing_timer, "Solve DAE");
      ida_->solve_dae(state, state_dot);
      const auto n_steps = static_cast<unsigned int>(statistics().steps);
      pcout << "IDAAdapter: DAE solve finished after " << n_steps
            << " internal step(s)." << std::endl;
      return n_steps;
    }

    /** Register native output for IDA's accepted/interpolated output states. */
    void
    set_output_step(OutputFunction output)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "IDAAdapter output must be configured before solve."));
      output_ = std::move(output);
      pcout << "IDAAdapter: accepted-state output callback configured."
            << std::endl;
    }

    /**
     * Set an application-provided consistent initial-condition callback.
     *
     * The callback is used when the corresponding IDA initial-condition
     * correction type is `use_y_diff`. It may modify both the state and its
     * derivative, including algebraic components of the state.
     */
    void
    set_compute_consistent_initial_conditions(
      ConsistentInitialConditionsFunction callback)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "IDAAdapter initial conditions must be configured "
                    "before solve."));
      compute_consistent_initial_conditions_ = std::move(callback);
    }

    /**
     * Set the application callback that requests an IDA restart.
     *
     * When the callback returns true and `reset_type` is `use_y_diff`, the
     * consistent initial-condition callback, if present, is called before IDA
     * is reset. If no consistent callback is present, IDA retains its native
     * `IDACalcIC` fallback.
     */
    void
    set_solver_should_restart(RestartFunction callback)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "IDAAdapter restart must be configured before solve."));
      solver_should_restart_ = std::move(callback);
    }

    GlobalVectorType
    make_state() const
    {
      return composition_.make_state();
    }

    FieldVectorType &
    field(GlobalVectorType &state, const FieldId id)
    {
      return composition_.field(state, id);
    }

    const FieldVectorType &
    field(const GlobalVectorType &state, const FieldId id) const
    {
      return composition_.field(state, id);
    }

    template <typename SemanticField,
              typename = std::void_t<
                decltype(std::declval<const SemanticField &>().id())>>
    FieldVectorType &
    field(GlobalVectorType &state, const SemanticField &semantic_field)
    {
      return field(state, semantic_field.id());
    }

    template <typename SemanticField,
              typename = std::void_t<
                decltype(std::declval<const SemanticField &>().id())>>
    const FieldVectorType &
    field(const GlobalVectorType &state,
          const SemanticField    &semantic_field) const
    {
      return field(state, semantic_field.id());
    }

    /** Write semantic fields without exposing the adapter's execution layout.
     */
    template <typename Output>
    void
    write_output(Output                 &output,
                 const GlobalVectorType &state,
                 const double            time = 0.) const
    {
      StateView<FieldVectorType> view(composition_.state_layout(), time);
      for (std::size_t i = 0; i < composition_.state_layout().n_fields(); ++i)
        view.bind(FieldId(i), composition_.field(state, FieldId(i)));
      output.write(view, time);
    }

    dealii::IndexSet
    differential_components()
    {
      finalize();
      return composition_.differential_components();
    }

    const Operator &
    current_jacobian() const
    {
      AssertThrow(current_jacobian_.has_value(),
                  dealii::ExcMessage("IDA has not prepared a Jacobian."));
      return *current_jacobian_;
    }

    bool
    has_current_preconditioner() const
    {
      return current_preconditioner_.has_value();
    }

    bool
    can_materialize_matrix(const GlobalVectorType &state,
                           const GlobalVectorType &state_dot,
                           const double            alpha) const
    {
      return composition_.can_materialize_matrix(state, &state_dot, alpha);
    }

    typename Composition::BlockMatrixType
    block_matrix(const GlobalVectorType &state,
                 const GlobalVectorType &state_dot,
                 const double            alpha) const
    {
      return composition_.block_matrix(state, &state_dot, alpha);
    }

    std::optional<typename Composition::Model::MatrixOperator>
    state_matrix_operator(const GlobalVectorType &state,
                          const FieldId           row,
                          const FieldId           column,
                          const double            time = 0.) const
    {
      return composition_.state_matrix_operator(state, row, column, time);
    }

    typename Composition::MatrixType
    monolithic_matrix(const GlobalVectorType &state,
                      const GlobalVectorType &state_dot,
                      const double            alpha) const
    {
      return composition_.monolithic_matrix(
        composition_.block_matrix(state, &state_dot, alpha));
    }

    bool
    has_local_preconditioner(const FieldId field) const
    {
      return composition_.has_local_preconditioner(field);
    }

    LocalOperator
    local_preconditioner(const FieldId           field,
                         const GlobalVectorType &state) const
    {
      const auto result = composition_.local_preconditioner(field, state);
      AssertThrow(result.has_value(),
                  dealii::ExcMessage("No local preconditioner is registered "
                                     "for this Field."));
      return *result;
    }

    const std::vector<SaddlePointMetadata> &
    saddle_points() const
    {
      return composition_.saddle_points();
    }

    LocalOperator
    schur_operator(const FieldId           multiplier,
                   const GlobalVectorType &state) const
    {
      return composition_.schur_operator(multiplier, state);
    }

    Operator
    schur_preconditioner(const FieldId           multiplier,
                         const GlobalVectorType &state) const
    {
      return composition_.schur_preconditioner(multiplier, state);
    }

  private:
    void
    finalize()
    {
      if (connected_)
        return;

      AssertThrow(composition_.n_fields() > 0,
                  dealii::ExcMessage("IDAAdapter has no semantic fields."));
      tolerance_ramp_ = ida_parameters_.tolerance_ramp;
      auto ida_data   = additional_data();
      tolerance_ramp_.validate(ida_data.initial_time,
                               ida_data.final_time,
                               ida_data.output_period);
      if (compute_consistent_initial_conditions_)
        {
          if (ida_data.ic_type == AdditionalData::use_y_diff)
            ida_data.ic_type = AdditionalData::none;
          if (ida_data.reset_type == AdditionalData::use_y_diff)
            ida_data.reset_type = AdditionalData::none;
        }
      ida_ = std::make_unique<Compatibility::SUNDIALS::IDA<GlobalVectorType>>(
        ida_data, composition_.communicator());
      ida_->reinit_vector = [this](GlobalVectorType &vector) {
        composition_.reinit(vector);
      };
      if (!local_tolerances_.empty() || tolerance_ramp_.enabled)
        {
          ida_->custom_setup = [this](void *) { initialize_tolerances(); };
          ida_->compute_error_weights = [this](const GlobalVectorType &state,
                                               GlobalVectorType &weights) {
            compute_error_weights(state, weights);
          };
        }

      ida_->residual = [this](const double            time,
                              const GlobalVectorType &state,
                              const GlobalVectorType &state_dot,
                              GlobalVectorType       &residual) {
        dealii::TimerOutput::Scope timer(computing_timer, "Evaluate residual");
        composition_.evaluate_residual(time, state, &state_dot, residual);
      };
      ida_->setup_jacobian = [this](const double            time,
                                    const GlobalVectorType &state,
                                    const GlobalVectorType &state_dot,
                                    const double            alpha) {
        prepare_jacobian(time, state, state_dot, alpha);
      };
      ida_->solve_with_jacobian = [this](const GlobalVectorType &rhs,
                                         GlobalVectorType       &dst,
                                         const double            tolerance) {
        ++linear_solves_;
        dealii::TimerOutput::Scope timer(computing_timer, "Linear solve");
        AssertThrow(current_jacobian_.has_value(),
                    dealii::ExcMessage("IDA requested a solve without a "
                                       "current Jacobian."));
        if (linear_solver_ == LinearSolver::umfpack)
          {
            dealii::TimerOutput::Scope backsolve_timer(computing_timer,
                                                       "Direct backsolve");
            AssertThrow(
              direct_solver_,
              dealii::ExcMessage(
                "IDA requested a direct solve without a factorization."));
            dealii::Vector<double> source(rhs.size()), destination(rhs.size());
            for (dealii::types::global_dof_index i = 0; i < rhs.size(); ++i)
              source[i] = rhs[i];
            direct_solver_->vmult(destination, source);
            for (dealii::types::global_dof_index i = 0; i < rhs.size(); ++i)
              dst[i] = destination[i];
            dst.compress(dealii::VectorOperation::insert);
          }
        else if (linear_solver_ == LinearSolver::mumps)
          {
#  ifdef DEAL_II_WITH_MUMPS
            dealii::TimerOutput::Scope backsolve_timer(computing_timer,
                                                       "Direct backsolve");
            AssertThrow(
              mumps_solver_,
              dealii::ExcMessage(
                "IDA requested a MUMPS solve without a factorization."));
            auto source = composition_.pack(rhs);
            source.compress(dealii::VectorOperation::insert);
            mumps_solver_->vmult(mumps_solution_, source);
            dealii::LinearAlgebra::ReadWriteVector<double> imported(
              source.locally_owned_elements());
            imported.import_elements(mumps_solution_,
                                     dealii::VectorOperation::insert);
            for (const auto index : source.locally_owned_elements())
              source[index] = imported[index];
            source.compress(dealii::VectorOperation::insert);
            composition_.unpack(source, dst);
            dst.compress(dealii::VectorOperation::insert);
#  else
            AssertThrow(
              false,
              dealii::ExcMessage(
                "The MUMPS solver requires deal.II configured with MUMPS."));
#  endif
          }
        else if (solve_)
          {
            pcout << "IDAAdapter: invoking the configured linear solver."
                  << std::endl;
            solve_(*current_jacobian_, rhs, dst, tolerance);
            pcout << "IDAAdapter: configured linear solver finished."
                  << std::endl;
          }
        else
          {
            dst = 0.;
            // Small physical loads still need a correction. An absolute floor
            // can accept the initial zero iterate and stall the DAE solve.
            dealii::SolverControl control(5000,
                                          std::min(tolerance,
                                                   1.e-8 * rhs.l2_norm()));
            // Avoid division by zero at exact Krylov breakdown in deal.II 9.7.
            typename dealii::SolverGMRES<GlobalVectorType>::AdditionalData data;
            data.orthogonalization_strategy = dealii::LinearAlgebra::
              OrthogonalizationStrategy::modified_gram_schmidt;
            dealii::SolverGMRES<GlobalVectorType> solver(control, data);
            if (current_preconditioner_.has_value())
              if (current_solver_is_flexible_)
                {
                  typename dealii::SolverFGMRES<
                    GlobalVectorType>::AdditionalData flexible_data(100);
                  flexible_data.orthogonalization_strategy =
                    dealii::LinearAlgebra::OrthogonalizationStrategy::
                      modified_gram_schmidt;
                  dealii::SolverFGMRES<GlobalVectorType> flexible_solver(
                    control, flexible_data);
                  flexible_solver.solve(*current_jacobian_,
                                        dst,
                                        rhs,
                                        *current_preconditioner_);
                  // An inexact Schur preconditioner can give a small projected
                  // residual while the original system is still inaccurate.
                  // Solve defect equations within the same iteration budget.
                  const auto   maximum_iterations = control.max_steps();
                  unsigned int iterations         = control.last_step();
                  auto         defect             = composition_.make_state();
                  current_jacobian_->vmult(defect, dst);
                  defect.sadd(-1., 1., rhs);
                  while (defect.l2_norm() > control.tolerance() &&
                         iterations < maximum_iterations)
                    {
                      auto correction = composition_.make_state();
                      correction      = 0.;
                      control.set_max_steps(maximum_iterations - iterations);
                      flexible_solver.solve(*current_jacobian_,
                                            correction,
                                            defect,
                                            *current_preconditioner_);
                      AssertThrow(control.last_step() > 0,
                                  dealii::SolverControl::NoConvergence(
                                    iterations, defect.l2_norm()));
                      iterations += control.last_step();
                      dst += correction;
                      current_jacobian_->vmult(defect, dst);
                      defect.sadd(-1., 1., rhs);
                    }
                  AssertThrow(defect.l2_norm() <= control.tolerance(),
                              dealii::SolverControl::NoConvergence(
                                iterations, defect.l2_norm()));
                  pcout << "IDAAdapter: FGMRES converged in " << iterations
                        << " iteration(s), true residual " << defect.l2_norm()
                        << "." << std::endl;
                }
              else
                {
                  solver.solve(*current_jacobian_,
                               dst,
                               rhs,
                               *current_preconditioner_);
                  pcout << "IDAAdapter: GMRES converged in "
                        << control.last_step() << " iteration(s), residual "
                        << control.last_value() << "." << std::endl;
                }
            else
              {
                solver.solve(*current_jacobian_,
                             dst,
                             rhs,
                             dealii::PreconditionIdentity());
                pcout << "IDAAdapter: GMRES converged in "
                      << control.last_step() << " iteration(s), residual "
                      << control.last_value() << "." << std::endl;
              }
          }
      };
      ida_->differential_components = [this]() {
        return current_state_ != nullptr ?
                 composition_.differential_components(*current_state_) :
                 composition_.differential_components();
      };
      ida_->solver_should_restart = [this](const double      time,
                                           GlobalVectorType &state,
                                           GlobalVectorType &state_dot) {
        current_state_ = &state;
        if (!solver_should_restart_)
          return false;

        const bool restart = solver_should_restart_(time, state, state_dot);
        if (restart)
          {
            accumulated_statistics_ = statistics();
            ++restarts_;
          }
        if (restart && compute_consistent_initial_conditions_ &&
            additional_data().reset_type == AdditionalData::use_y_diff)
          {
            pcout << "IDAAdapter: computing consistent initial conditions "
                     "after restart with the application callback."
                  << std::endl;
            compute_consistent_initial_conditions_(time, state, state_dot);
          }
        return restart;
      };
      // Internal bookkeeping must run even when no application output is set.
      ida_->output_step = [this](const double            time,
                                 const GlobalVectorType &state,
                                 const GlobalVectorType &state_dot,
                                 const unsigned int      step) {
        update_ramp_stop();
        if (ida_parameters_.print_statistics)
          {
            const auto info = statistics();
            pcout << "IDA: output t=" << time
                  << ", internal t=" << info.current_time
                  << ", steps=" << info.steps << ", order=" << info.order
                  << ", h=" << info.last_step
                  << ", Newton=" << info.nonlinear_iterations
                  << ", Jacobians=" << info.jacobian_setups
                  << ", linear solves=" << info.linear_solves
                  << ", error failures=" << info.error_test_failures
                  << ", convergence failures=" << info.convergence_failures
                  << std::endl;
          }
        if (output_)
          output_(time, state, state_dot, step);
      };
      pcout << "IDAAdapter: finalized with " << composition_.n_fields()
            << " semantic field(s)." << std::endl;
      connected_ = true;
    }

    static void
    check_ida(const int status)
    {
      AssertThrow(status >= 0,
                  dealii::ExcMessage("SUNDIALS IDA returned status " +
                                     std::to_string(status)));
    }

    void
    initialize_tolerances()
    {
      AssertThrow(current_state_ != nullptr,
                  dealii::ExcMessage(
                    "Use IDAAdapter::solve() with adapter-owned tolerances."));
      absolute_tolerances_.reinit(*current_state_);
      absolute_tolerances_ = additional_data().absolute_tolerance;
      for (const auto &entry : local_tolerances_)
        entry.second(composition_.field(absolute_tolerances_, entry.first));
      absolute_tolerances_.compress(dealii::VectorOperation::insert);
      bool valid = std::isfinite(additional_data().relative_tolerance) &&
                   additional_data().relative_tolerance >= 0.;
      const auto scales =
        tolerance_ramp_.scales(additional_data().initial_time,
                               additional_data().initial_time);
      valid = valid && std::isfinite(additional_data().relative_tolerance *
                                     scales.first);
      for (const auto index : absolute_tolerances_.locally_owned_elements())
        {
          const double value = absolute_tolerances_[index];
          valid              = valid && std::isfinite(value) && value > 0. &&
                  std::isfinite(value * scales.second);
        }
      AssertThrow(
        dealii::Utilities::MPI::min(valid ? 1 : 0,
                                    composition_.communicator()) == 1,
        dealii::ExcMessage(
          "IDA absolute tolerances must be finite and positive on every rank."));
      ramp_stop_armed_    = false;
      ramp_stop_released_ = false;
    }

    void
    compute_error_weights(const GlobalVectorType &state,
                          GlobalVectorType       &weights)
    {
      double time;
      check_ida(IDAGetCurrentTime(ida_->get_ida_memory(), &time));
      const auto scales =
        tolerance_ramp_.scales(additional_data().initial_time, time);
      bool valid = std::isfinite(time);
      for (const auto index : weights.locally_owned_elements())
        {
          const double value = state[index];
          const double denominator =
            additional_data().relative_tolerance * scales.first *
              std::abs(value) +
            scales.second * absolute_tolerances_[index];
          weights[index] = 1. / denominator;
          valid          = valid && std::isfinite(value) &&
                  std::isfinite(weights[index]) && weights[index] > 0.;
        }
      weights.compress(dealii::VectorOperation::insert);
      AssertThrow(dealii::Utilities::MPI::min(valid ? 1 : 0,
                                              composition_.communicator()) == 1,
                  dealii::ExcMessage(
                    "Non-finite state or invalid IDA error weights."));
      ++weight_evaluations_;
    }

    void
    update_ramp_stop()
    {
      const auto &ramp = tolerance_ramp_;
      if (!ramp.enabled)
        return;
      double time;
      auto   memory = ida_->get_ida_memory();
      check_ida(IDAGetCurrentTime(memory, &time));
      const bool at_cutoff =
        time >=
        ramp.end_time -
          IDAToleranceRamp::time_tolerance(additional_data().initial_time,
                                           ramp.end_time);
      if (!ramp_stop_armed_ && !at_cutoff)
        {
          check_ida(IDASetStopTime(memory, ramp.end_time));
          ramp_stop_armed_ = true;
        }
      if (at_cutoff && !ramp_stop_released_)
        {
          check_ida(IDASetStopTime(memory, additional_data().final_time));
          ramp_stop_released_ = true;
        }
    }

    void
    prepare_jacobian(const double            time,
                     const GlobalVectorType &state,
                     const GlobalVectorType &state_dot,
                     const double            alpha)
    {
      ++jacobian_setups_;
      dealii::TimerOutput::Scope timer(computing_timer, "Setup Jacobian");
      const auto snapshot = composition_.make_snapshot(time, state, state_dot);
      const auto operator_view =
        composition_.jacobian(snapshot->context, alpha);

      Operator stable;
      stable.reinit_range_vector = [operator_view](GlobalVectorType &vector,
                                                   const bool        omit) {
        operator_view.reinit_range_vector(vector, omit);
      };
      stable.reinit_domain_vector = [operator_view](GlobalVectorType &vector,
                                                    const bool        omit) {
        operator_view.reinit_domain_vector(vector, omit);
      };
      stable.vmult = [operator_view, snapshot](GlobalVectorType &destination,
                                               const GlobalVectorType &source) {
        operator_view.vmult(destination, source);
      };
      stable.vmult_add = [operator_view,
                          snapshot](GlobalVectorType       &destination,
                                    const GlobalVectorType &source) {
        operator_view.vmult_add(destination, source);
      };
      stable.Tvmult = [operator_view,
                       snapshot](GlobalVectorType       &destination,
                                 const GlobalVectorType &source) {
        operator_view.Tvmult(destination, source);
      };
      stable.Tvmult_add = [operator_view,
                           snapshot](GlobalVectorType       &destination,
                                     const GlobalVectorType &source) {
        operator_view.Tvmult_add(destination, source);
      };
      current_jacobian_ = std::move(stable);
      current_preconditioner_.reset();
      current_solver_is_flexible_ = false;
      if (linear_solver_ == LinearSolver::umfpack ||
          linear_solver_ == LinearSolver::mumps)
        {
          dealii::TimerOutput::Scope direct_timer(computing_timer,
                                                  "Materialize Jacobian");
          const auto                 matrix = composition_.monolithic_matrix(
            composition_.block_matrix(snapshot->state_storage,
                                      &snapshot->derivative_storage,
                                      alpha,
                                      time));
          if (linear_solver_ == LinearSolver::umfpack)
            {
              dealii::DynamicSparsityPattern sparsity(matrix.m(), matrix.n());
              for (dealii::types::global_dof_index row = 0; row < matrix.m();
                   ++row)
                for (auto entry = matrix.begin(row), end = matrix.end(row);
                     entry != end;
                     ++entry)
                  sparsity.add(row, entry->column());
              dealii::SparsityPattern pattern;
              pattern.copy_from(sparsity);
              dealii::SparseMatrix<double> serial_matrix(pattern);
#  ifndef IMMERSX_USE_PETSC_LA
              serial_matrix.copy_from(matrix);
#  else
              for (dealii::types::global_dof_index row = 0; row < matrix.m();
                   ++row)
                for (auto entry = matrix.begin(row), end = matrix.end(row);
                     entry != end;
                     ++entry)
                  serial_matrix.set(row, entry->column(), entry->value());
#  endif
              direct_timer.stop();
              dealii::TimerOutput::Scope factor_timer(computing_timer,
                                                      "Factorize Jacobian");
              direct_solver_ = std::make_unique<dealii::SparseDirectUMFPACK>();
              direct_solver_->initialize(serial_matrix);
              pcout << "IDAAdapter: UMFPACK factorized " << matrix.m()
                    << " rows, " << serial_matrix.n_nonzero_elements()
                    << " matrix entries." << std::endl;
            }
          else if (linear_solver_ == LinearSolver::mumps)
            {
#  ifdef DEAL_II_WITH_MUMPS
              AssertThrow(
                matrix.m() >= dealii::Utilities::MPI::n_mpi_processes(
                                composition_.communicator()),
                dealii::ExcMessage(
                  "MUMPS requires at least one coupled unknown per MPI rank."));
              // MUMPS wrappers require contiguous, nonempty solution ownership.
              mumps_solution_.reinit(
                dealii::Utilities::MPI::create_evenly_distributed_partitioning(
                  composition_.communicator(), matrix.m()),
                composition_.communicator());
              direct_timer.stop();
              dealii::TimerOutput::Scope factor_timer(computing_timer,
                                                      "Factorize Jacobian");
              mumps_solver_ = std::make_unique<dealii::SparseDirectMUMPS>(
                dealii::SparseDirectMUMPS::AdditionalData(),
                composition_.communicator());
              mumps_solver_->initialize(matrix);
              pcout << "IDAAdapter: MUMPS factorized " << matrix.m()
                    << " rows, " << matrix.n_nonzero_elements()
                    << " matrix entries." << std::endl;
#  else
              AssertThrow(
                false,
                dealii::ExcMessage(
                  "The MUMPS solver requires deal.II configured with MUMPS."));
#  endif
            }
        }
      else if (!solve_)
        {
          const auto keep_snapshot = [snapshot](Operator operator_view) {
            Operator result;
            result.reinit_range_vector = [operator_view,
                                          snapshot](GlobalVectorType &vector,
                                                    const bool        omit) {
              operator_view.reinit_range_vector(vector, omit);
            };
            result.reinit_domain_vector = [operator_view,
                                           snapshot](GlobalVectorType &vector,
                                                     const bool        omit) {
              operator_view.reinit_domain_vector(vector, omit);
            };
            result.vmult = [operator_view,
                            snapshot](GlobalVectorType       &destination,
                                      const GlobalVectorType &source) {
              operator_view.vmult(destination, source);
            };
            result.vmult_add = [operator_view,
                                snapshot](GlobalVectorType       &destination,
                                          const GlobalVectorType &source) {
              operator_view.vmult_add(destination, source);
            };
            result.Tvmult = [operator_view,
                             snapshot](GlobalVectorType       &destination,
                                       const GlobalVectorType &source) {
              operator_view.Tvmult(destination, source);
            };
            result.Tvmult_add = [operator_view,
                                 snapshot](GlobalVectorType       &destination,
                                           const GlobalVectorType &source) {
              operator_view.Tvmult_add(destination, source);
            };
            return result;
          };
          if (!composition_.saddle_points().empty())
            {
              dealii::TimerOutput::Scope schur_timer(
                computing_timer, "Build Schur preconditioner");
              const auto &saddle = composition_.saddle_points().front();
              current_preconditioner_ =
                keep_snapshot(composition_.schur_preconditioner(
                  saddle.multiplier,
                  snapshot->state_storage,
                  &snapshot->derivative_storage,
                  alpha,
                  1000,
                  1.e-9,
                  include_schur_primal_coupling_));
              current_solver_is_flexible_ = true;
            }
          else if (composition_.has_complete_local_preconditioners())
            {
              current_preconditioner_ =
                keep_snapshot(composition_.n_fields() > 1 ?
                                composition_.block_triangular_preconditioner(
                                  snapshot->state_storage,
                                  true,
                                  &snapshot->derivative_storage,
                                  alpha) :
                                composition_.block_diagonal_preconditioner(
                                  snapshot->state_storage,
                                  &snapshot->derivative_storage,
                                  alpha));
              current_solver_is_flexible_ = composition_.n_fields() > 1;
            }
        }
    }

    std::vector<std::pair<FieldId, std::function<void(FieldVectorType &)>>>
                     local_tolerances_;
    GlobalVectorType absolute_tolerances_;
    IDAToleranceRamp tolerance_ramp_;
    Statistics       accumulated_statistics_;
    unsigned long    jacobian_setups_ = 0, linear_solves_ = 0, restarts_ = 0,
                  weight_evaluations_ = 0;
    bool         ramp_stop_armed_ = false, ramp_stop_released_ = false;
    LinearSolver linear_solver_ = LinearSolver::iterative;
#  ifdef DEAL_II_WITH_MUMPS
    std::unique_ptr<dealii::SparseDirectMUMPS> mumps_solver_;
    FieldVectorType                            mumps_solution_;
#  endif
    std::unique_ptr<dealii::SparseDirectUMFPACK> direct_solver_;
    Composition                                  composition_;
    LinearSolveFunction                          solve_;
    const TimeIntervalParameters                &time_parameters_;
    const IDAParameters                         &ida_parameters_;
    mutable std::optional<AdditionalData>        additional_data_;
    std::unique_ptr<Compatibility::SUNDIALS::IDA<GlobalVectorType>> ida_;
    mutable dealii::ConditionalOStream                              pcout;
    mutable dealii::TimerOutput         computing_timer;
    std::optional<Operator>             current_jacobian_;
    std::optional<Operator>             current_preconditioner_;
    OutputFunction                      output_;
    ConsistentInitialConditionsFunction compute_consistent_initial_conditions_;
    RestartFunction                     solver_should_restart_;
    GlobalVectorType                   *current_state_              = nullptr;
    bool                                current_solver_is_flexible_ = false;
    bool                                connected_                  = false;
    bool                                include_schur_primal_coupling_ = false;
  };
} // namespace ImmersX
#endif

#endif // immersx_sundials_ida_adapter_h
