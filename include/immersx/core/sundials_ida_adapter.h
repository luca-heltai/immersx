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
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector_memory.h>

#include <immersx/core/detail/execution_composition.h>
#include <immersx/core/problem_handle.h>
#include <immersx/core/time_parameters.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
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
  namespace detail
  {
    /** Minimize a stalled direct-solve defect using the existing backsolve.
     * The caller must independently check the original Jacobian residual:
     * FGMRES's projected residual is not an acceptance criterion.
     */
    template <typename VectorType>
    unsigned int
    mumps_krylov_refinement(
      const dealii::LinearOperator<VectorType>                    &jacobian,
      const std::function<void(const VectorType &, VectorType &)> &backsolve,
      const VectorType                                            &rhs,
      VectorType                                                  &solution,
      const unsigned int maximum_iterations,
      const double       absolute_target)
    {
      if (maximum_iterations == 0)
        return 0;
      VectorType residual, correction;
      jacobian.reinit_range_vector(residual, false);
      jacobian.reinit_domain_vector(correction, false);
      jacobian.vmult(residual, solution);
      residual.sadd(-1., 1., rhs);
      dealii::LinearOperator<VectorType> inverse;
      bool                               backsolve_failed = false;
      inverse.vmult                                       = [&backsolve,
                       &backsolve_failed](VectorType       &destination,
                                          const VectorType &source) {
        try
          {
            backsolve(source, destination);
          }
        catch (const dealii::SolverControl::NoConvergence &)
          {
            backsolve_failed = true;
            throw;
          }
      };
      unsigned int iterations    = 0;
      double       residual_norm = residual.l2_norm();
      while (iterations < maximum_iterations && std::isfinite(residual_norm) &&
             residual_norm > absolute_target)
        {
          const auto remaining = maximum_iterations - iterations;
          correction           = 0.;
          dealii::SolverControl control(remaining, absolute_target);
          typename dealii::SolverFGMRES<VectorType>::AdditionalData data(
            remaining);
          data.orthogonalization_strategy = dealii::LinearAlgebra::
            OrthogonalizationStrategy::modified_gram_schmidt;
          dealii::SolverFGMRES<VectorType> solver(control, data);
          try
            {
              solver.solve(jacobian, correction, residual, inverse);
            }
          catch (const dealii::SolverControl::NoConvergence &)
            {
              if (backsolve_failed)
                throw;
              // An exhausted budget may still leave a useful correction.
            }
          iterations += control.last_step();
          solution += correction;
          solution.compress(dealii::VectorOperation::insert);
          jacobian.vmult(residual, solution);
          residual.sadd(-1., 1., rhs);
          const double next_norm = residual.l2_norm();
          // Projected convergence can precede convergence of the accumulated
          // solution. Restart on its true defect, without resetting the budget.
          if (!std::isfinite(next_norm) || next_norm <= absolute_target ||
              control.last_step() == 0 || next_norm >= residual_norm)
            break;
          residual_norm = next_norm;
        }
      return iterations;
    }

    /** Solve in native IDA error-weight coordinates and verify both norms.
     * The caller supplies the weighted L2 target. No scaled matrix is stored.
     */
    template <typename VectorType>
    unsigned int
    ida_weighted_krylov(
      const dealii::LinearOperator<VectorType> &jacobian,
      const dealii::LinearOperator<VectorType> *preconditioner,
      const VectorType                         &rhs,
      VectorType                               &solution,
      const VectorType                         &weights,
      const double                              weighted_target,
      const double                              raw_target,
      const unsigned int                        maximum_iterations,
      const bool                                flexible,
      const MPI_Comm                            communicator)
    {
      bool valid = std::isfinite(weighted_target) && weighted_target > 0. &&
                   std::isfinite(raw_target) && raw_target >= 0.;
      for (const auto i : weights.locally_owned_elements())
        valid = valid && std::isfinite(weights[i]) && weights[i] > 0. &&
                std::isfinite(1. / weights[i]);
      AssertThrow(dealii::Utilities::MPI::min(valid ? 1u : 0u, communicator),
                  dealii::ExcMessage(
                    "Invalid native IDA linear weights or targets."));
      VectorType input, defect, scaled_rhs, scaled_correction, weighted_defect;
      jacobian.reinit_domain_vector(input, false);
      jacobian.reinit_range_vector(defect, false);
      scaled_rhs.reinit(defect);
      weighted_defect.reinit(defect);
      scaled_correction.reinit(input);
      const auto scale =
        [&weights](VectorType &dst, const VectorType &src, const bool inverse) {
          for (const auto i : dst.locally_owned_elements())
            dst[i] = inverse ? src[i] / weights[i] : src[i] * weights[i];
          dst.compress(dealii::VectorOperation::insert);
        };
      // Krylov consumes vmult; do not inherit unscaled additive/transpose
      // actions.
      dealii::LinearOperator<VectorType> scaled_operator;
      scaled_operator.reinit_domain_vector = jacobian.reinit_domain_vector;
      scaled_operator.reinit_range_vector  = jacobian.reinit_range_vector;
      scaled_operator.vmult =
        [&input, &scale, &jacobian](VectorType &dst, const VectorType &src) {
          scale(input, src, true);
          jacobian.vmult(dst, input);
          scale(dst, dst, false);
        };
      dealii::LinearOperator<VectorType> scaled_inverse;
      scaled_inverse.reinit_domain_vector = jacobian.reinit_range_vector;
      scaled_inverse.reinit_range_vector  = jacobian.reinit_domain_vector;
      scaled_inverse.vmult =
        [&input, &scale, preconditioner](VectorType       &dst,
                                         const VectorType &src) {
          scale(input, src, true);
          if (preconditioner)
            preconditioner->vmult(dst, input);
          else
            dst = input;
          scale(dst, dst, false);
        };
      solution = 0.;
      defect   = rhs;
      // Native IDA vectors may require an import into this map. Close Insert
      // mode before a matrix-free action accumulates into the same vector.
      defect.compress(dealii::VectorOperation::insert);
      unsigned int iterations = 0;
      for (;;)
        {
          scale(weighted_defect, defect, false);
          const double raw_norm      = defect.l2_norm();
          const double weighted_norm = weighted_defect.l2_norm();
          AssertThrow(std::isfinite(raw_norm) && std::isfinite(weighted_norm),
                      dealii::ExcMessage(
                        "Nonfinite true IDA linear residual."));
          if (raw_norm <= raw_target && weighted_norm <= weighted_target)
            return iterations;
          AssertThrow(iterations < maximum_iterations && weighted_norm > 0.,
                      dealii::SolverControl::NoConvergence(iterations,
                                                           weighted_norm));
          scaled_rhs        = weighted_defect;
          scaled_correction = 0.;
          // A weighted pass can still fail the independent raw gate. Force
          // progress instead of accepting a zero-iteration correction.
          dealii::SolverControl control(maximum_iterations - iterations,
                                        std::min(weighted_target,
                                                 .5 * weighted_norm));
          if (flexible)
            {
              typename dealii::SolverFGMRES<VectorType>::AdditionalData data(
                100);
              data.orthogonalization_strategy = dealii::LinearAlgebra::
                OrthogonalizationStrategy::modified_gram_schmidt;
              dealii::SolverFGMRES<VectorType> solver(control, data);
              solver.solve(scaled_operator,
                           scaled_correction,
                           scaled_rhs,
                           scaled_inverse);
            }
          else
            {
              typename dealii::SolverGMRES<VectorType>::AdditionalData data;
              data.right_preconditioning      = true;
              data.orthogonalization_strategy = dealii::LinearAlgebra::
                OrthogonalizationStrategy::modified_gram_schmidt;
              dealii::SolverGMRES<VectorType> solver(control, data);
              solver.solve(scaled_operator,
                           scaled_correction,
                           scaled_rhs,
                           scaled_inverse);
            }
          AssertThrow(control.last_step() > 0,
                      dealii::SolverControl::NoConvergence(iterations,
                                                           weighted_norm));
          iterations += control.last_step();
          scale(input, scaled_correction, true);
          solution += input;
          solution.compress(dealii::VectorOperation::insert);
          jacobian.vmult(defect, solution);
          defect.sadd(-1., 1., rhs);
        }
    }

    /** Certify an indeterminate native target only through an exact defect.
     * A zero norm can hide underflow; compare every owned entry instead.
     */
    template <typename VectorType>
    unsigned int
    ida_exact_residual_krylov(
      const dealii::LinearOperator<VectorType> &jacobian,
      const dealii::LinearOperator<VectorType> *preconditioner,
      const VectorType                         &rhs,
      VectorType                               &solution,
      const double                              raw_target,
      const unsigned int                        maximum_iterations,
      const bool                                flexible,
      const MPI_Comm                            communicator)
    {
      AssertThrow(dealii::Utilities::MPI::min(
                    std::isfinite(raw_target) && raw_target >= 0. ? 1u : 0u,
                    communicator),
                  dealii::ExcMessage("Invalid raw IDA linear target."));
      VectorType ones, defect, correction;
      jacobian.reinit_domain_vector(ones, false);
      jacobian.reinit_domain_vector(correction, false);
      jacobian.reinit_range_vector(defect, false);
      ones     = 1.;
      solution = 0.;
      defect   = rhs;
      defect.compress(dealii::VectorOperation::insert);
      unsigned int iterations      = 0;
      double       previous_defect = std::numeric_limits<double>::infinity();
      for (;;)
        {
          bool finite = true, exact = true;
          for (const auto i : defect.locally_owned_elements())
            {
              finite = finite && std::isfinite(defect[i]);
              exact  = exact && defect[i] == 0.;
            }
          AssertThrow(dealii::Utilities::MPI::min(finite ? 1u : 0u,
                                                  communicator),
                      dealii::ExcMessage("Nonfinite exact-only IDA defect."));
          if (dealii::Utilities::MPI::min(exact ? 1u : 0u, communicator))
            {
              AssertThrow(defect.l2_norm() <= raw_target,
                          dealii::ExcMessage(
                            "Raw IDA linear residual gate failed."));
              return iterations;
            }
          const double norm = defect.linfty_norm();
          AssertThrow(
            norm > 0. && norm < previous_defect,
            dealii::ExcMessage(
              "Exact-only IDA refinement made no true-defect progress."));
          AssertThrow(iterations < maximum_iterations,
                      dealii::SolverControl::NoConvergence(iterations, norm));
          previous_defect = norm;
          // A raw-sized projected target could accept zero work. Ask for
          // genuine progress, then independently recalculate the full defect.
          // The infinity norm avoids squaring tiny entries before this test.
          const double target =
            std::max(.5 * norm, std::numeric_limits<double>::denorm_min());
          const auto steps =
            ida_weighted_krylov(jacobian,
                                preconditioner,
                                defect,
                                correction,
                                ones,
                                target,
                                target,
                                maximum_iterations - iterations,
                                flexible,
                                communicator);
          AssertThrow(
            steps > 0,
            dealii::ExcMessage(
              "Indeterminate or zero IDA target requires an exactly zero full-J residual; correction made no progress."));
          iterations += steps;
          solution += correction;
          solution.compress(dealii::VectorOperation::insert);
          jacobian.vmult(defect, solution);
          defect.sadd(-1., 1., rhs);
        }
    }
  } // namespace detail

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
      mumps,
      mumps_blr_fgmres
    };

    enum class BlockPreconditioner
    {
      automatic,
      schur,
      block_diagonal,
      block_triangular
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

    /** Include a semantic Field in the configured tolerance ramp. Fields
     * participate by default. Disabling it retains both the relative and
     * absolute tolerances for this Field during initialization and integration.
     */
    void
    set_tolerance_ramp_enabled(const FieldId field, const bool enabled)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "Configure tolerances before solver initialization."));
      AssertThrow(field.value() < composition_.n_fields(),
                  dealii::ExcMessage("Unknown tolerance Field."));
      const auto found =
        std::find(unramped_fields_.begin(), unramped_fields_.end(), field);
      if (enabled && found != unramped_fields_.end())
        unramped_fields_.erase(found);
      else if (!enabled && found == unramped_fields_.end())
        unramped_fields_.push_back(field);
    }

    template <typename SemanticField>
    void
    set_tolerance_ramp_enabled(const SemanticField &field, const bool enabled)
    {
      set_tolerance_ramp_enabled(field.id(), enabled);
    }

    /** IDA counters accumulated across adapter-managed restarts. */
    struct Statistics
    {
      long int steps = 0, residual_evaluations = 0, nonlinear_iterations = 0,
               error_test_failures = 0, convergence_failures = 0;
      unsigned long jacobian_setups = 0, linear_solves = 0, restarts = 0,
                    weight_evaluations = 0, kinematic_factorizations = 0,
                    kinematic_factorization_reuses = 0;
      int                       order              = 0;
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
      result.jacobian_setups                = jacobian_setups_;
      result.linear_solves                  = linear_solves_;
      result.restarts                       = restarts_;
      result.weight_evaluations             = weight_evaluations_;
      result.kinematic_factorizations       = kinematic_factorizations_;
      result.kinematic_factorization_reuses = kinematic_factorization_reuses_;
      result.tolerance_scales =
        tolerance_ramp_.scales(additional_data().initial_time,
                               result.current_time);
      return result;
    }

    /** Select the coupled linear solver before initialization.
     * Direct solvers reuse their factorization until IDA requests a new
     * Jacobian setup. UMFPACK requires one rank; MUMPS uses the composition's
     * communicator and distributed matrix. A MUMPS backsolve may use
     * correction solves with the same factorization to satisfy the retained
     * Jacobian residual check, as set in IDALinearSolverControls.
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
      else if (solver == LinearSolver::mumps_blr_fgmres)
        {
#  if !defined(DEAL_II_WITH_MUMPS) || defined(IMMERSX_USE_PETSC_LA)
          AssertThrow(
            false,
            dealii::ExcMessage(
              "MUMPS BLR FGMRES requires MUMPS and the Trilinos vector backend."));
#  endif
        }
      AssertThrow(
        solver == LinearSolver::iterative || !solve_,
        dealii::ExcMessage(
          "A direct solver cannot be combined with a custom linear solver."));
      linear_solver_ = solver;
    }

    /** Select the iterative block preconditioner before initialization. */
    void
    set_block_preconditioner(const BlockPreconditioner preconditioner)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "The block preconditioner must be selected before "
                    "solver initialization."));
      block_preconditioner_ = preconditioner;
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

    /** Materialize Schur operators up to this global field size per snapshot.
     * Zero keeps matrix-free inner actions. Cached solves retain the original
     * GMRES metric/tolerance and verify the original operator residual.
     */
    void
    set_schur_materialization_limit(const unsigned int maximum_dofs)
    {
      AssertThrow(
        !connected_,
        dealii::ExcMessage(
          "Configure Schur materialization before solver initialization."));
      schur_materialization_limit_ = maximum_dofs;
    }

    /** Configure the MUMPS block-low-rank factorization used by FGMRES.
     * The tolerance is the MUMPS BLR truncation threshold. A memory limit of
     * zero leaves MUMPS unlimited. Available only with MUMPS and Trilinos.
     */
    void
    set_mumps_blr_parameters(const double       tolerance,
                             const unsigned int memory_limit_mb = 0,
                             const bool         output_details  = false)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "Configure MUMPS BLR before solver initialization."));
      AssertThrow(std::isfinite(tolerance) && tolerance > 0. && tolerance < 1.,
                  dealii::ExcMessage(
                    "MUMPS BLR tolerance must be between zero and one."));
      AssertThrow(memory_limit_mb <=
                    static_cast<unsigned int>(std::numeric_limits<int>::max()),
                  dealii::ExcMessage(
                    "MUMPS BLR memory limit exceeds its integer range."));
      mumps_blr_tolerance_      = tolerance;
      mumps_blr_memory_limit_   = memory_limit_mb;
      mumps_blr_output_details_ = output_details;
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

    /** Prepare a fresh steady Jacobian F_y(t,y,y') for an external Newton
     * solve. This uses the configured materialized direct solver and does not
     * initialize or advance IDA. Optionally fill positive function scaling
     * 1 / sum_j |J_ij| c_j from supplied positive state characteristics c.
     * Call with scaling outputs once, then keep them fixed during the solve.
     * Uses the existing second-order condensation when available. The PETSc
     * backend is not supported here.
     */
    void
    setup_steady_jacobian(const double            time,
                          const GlobalVectorType &state,
                          const GlobalVectorType &state_dot,
                          const GlobalVectorType *characteristics  = nullptr,
                          GlobalVectorType       *function_scaling = nullptr)
    {
      AssertThrow(
        linear_solver_ == LinearSolver::mumps ||
          linear_solver_ == LinearSolver::umfpack,
        dealii::ExcMessage(
          "Steady Newton requires a materialized MUMPS or UMFPACK solver."));
      AssertThrow(
        (characteristics == nullptr) == (function_scaling == nullptr),
        dealii::ExcMessage(
          "Steady scaling requires both characteristics and an output vector."));
#  ifdef IMMERSX_USE_PETSC_LA
      AssertThrow(false,
                  dealii::ExcMessage(
                    "Steady Newton requires the Trilinos backend."));
#  endif
      finalize();
      prepare_jacobian(
        time, state, state_dot, 0., true, characteristics, function_scaling);
    }

    /** Compute positive function scaling from the full steady Jacobian without
     * preparing a linear solver: scaling_i = 1 / sum_j |F_y(t,y,ydot)_ij| c_j.
     * The state characteristics c must be finite and positive, and every
     * weighted row sum must be finite and strictly positive.
     *
     * This assembles the original alpha=0 blocks, including kinematic rows,
     * independently of the selected linear solver. It does not factor a matrix,
     * prepare a preconditioner, or replace the current Jacobian. State and rate
     * inputs are unchanged. For a fixed acceptance metric, evaluate once at the
     * reference initial state and retain the resulting scales for candidates.
     * Only the Trilinos backend is supported, as for setup_steady_jacobian().
     */
    void
    steady_function_scaling(const double            time,
                            const GlobalVectorType &state,
                            const GlobalVectorType &state_dot,
                            const GlobalVectorType &characteristics,
                            GlobalVectorType       &scaling)
    {
#  ifdef IMMERSX_USE_PETSC_LA
      AssertThrow(false,
                  dealii::ExcMessage("Steady scaling requires Trilinos."));
#  endif
      finalize();
      const auto snapshot =
        composition_.make_snapshot(time, state, state_dot, true);
      const auto blocks =
        composition_.block_matrix(snapshot->state_storage,
                                  &snapshot->derivative_storage,
                                  0.,
                                  time,
                                  &snapshot->linearization);
      make_steady_function_scaling(blocks, characteristics, scaling);
    }

    unsigned int
    solve(GlobalVectorType &state, GlobalVectorType &state_dot)
    {
      current_state_          = &state;
      accumulated_statistics_ = {};
      jacobian_setups_ = linear_solves_ = restarts_ = weight_evaluations_ = 0;
      initial_condition_jacobian_time_.reset();
      kinematic_factorizations_ = kinematic_factorization_reuses_ = 0;
      diagnostic_residuals_                                       = 0;
      if (compute_consistent_initial_conditions_ &&
          additional_data().ic_type == AdditionalData::use_y_diff)
        {
          pcout << "IDAAdapter: computing consistent initial conditions "
                   "with the application callback."
                << std::endl;
          dealii::TimerOutput::Scope timer(computing_timer,
                                           "Consistent initial conditions");
          compute_consistent_initial_conditions_(additional_data().initial_time,
                                                 state,
                                                 state_dot);
        }
      finalize();
      pcout << "IDAAdapter: starting DAE solve with " << composition_.n_fields()
            << " semantic field(s)." << std::endl;
      dealii::TimerOutput::Scope timer(computing_timer, "Solve DAE");
      try
        {
          ida_->solve_dae(state, state_dot);
        }
      catch (const std::exception &error)
        {
          if (ida_parameters_.print_nonlinear_diagnostics)
            pcout << "IDA nonlinear failure original_exception=" << error.what()
                  << std::endl;
          print_nonlinear_summary("failure");
          throw;
        }
      catch (...)
        {
          print_nonlinear_summary("failure");
          throw;
        }
      print_nonlinear_summary("success");
      const auto n_steps = static_cast<unsigned int>(statistics().steps);
      pcout << "IDAAdapter: DAE solve finished after " << n_steps
            << " internal step(s)." << std::endl;
      return n_steps;
    }

    /** Force IDA to end an internal step at each output time.
     * Configure before initialization. The default uses IDA's dense output,
     * which can violate nonlinear algebraic constraints between internal steps.
     * Stop times preserve the BDF history and do not restart the solver.
     */
    void
    set_stop_at_output_times(const bool enabled = true)
    {
      AssertThrow(
        !connected_,
        dealii::ExcMessage(
          "Output stops must be configured before solver initialization."));
      stop_at_output_times_ = enabled;
    }

    /** Override MUMPS ICNTL(14), the extra workspace percentage.
     * Without an override, the native MUMPS default is retained.
     */
    void
    set_mumps_workspace_relaxation(const unsigned int percentage)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "Configure MUMPS workspace before solver initialization."));
      AssertThrow(percentage <=
                    static_cast<unsigned int>(std::numeric_limits<int>::max()),
                  dealii::ExcMessage(
                    "MUMPS workspace percentage exceeds its integer range."));
      mumps_workspace_relaxation_ = static_cast<int>(percentage);
    }

    /** Set MUMPS ICNTL(23), the working-memory limit in MB per rank.
     * Zero keeps MUMPS's automatic allocation. This is not a process RSS limit.
     */
    void
    set_mumps_memory_limit_mb(const unsigned int megabytes)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "Configure MUMPS memory before solver initialization."));
      AssertThrow(
        megabytes <= static_cast<unsigned int>(std::numeric_limits<int>::max()),
        dealii::ExcMessage("MUMPS memory limit exceeds its integer range."));
      mumps_memory_limit_mb_ = static_cast<int>(megabytes);
    }

    /** Enable native MUMPS analysis/factorization diagnostics on every setup.
     * By default, report the first factorization of each direct solver only.
     */
    void
    set_mumps_output_details(const bool enabled)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "Configure MUMPS output before solver initialization."));
      mumps_output_details_ = enabled;
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
      const auto &controls = ida_parameters_.linear_solver_controls;
      return composition_.schur_preconditioner(
        multiplier,
        state,
        nullptr,
        0.,
        controls.schur_maximum_iterations,
        controls.schur_relative_tolerance,
        false,
        nullptr,
        schur_materialization_limit_);
    }

  private:
    using MatrixType = typename Composition::MatrixType;

    /** One sparse LU factorization through UMFPACK or MUMPS. */
    class DirectFactorization
    {
    public:
      bool
      initialized() const
      {
        return diagonal_ || static_cast<bool>(umfpack_)
#  ifdef DEAL_II_WITH_MUMPS
               || static_cast<bool>(mumps_)
#  endif
          ;
      }

      std::size_t
      n_entries() const
      {
        return n_entries_;
      }

      /** Whether factorize() found a diagonal matrix and inverted it. */
      bool
      diagonal() const
      {
        return diagonal_;
      }

      void
      factorize(const LinearSolver        solver,
                const MatrixType         &matrix,
                const MPI_Comm            communicator,
                const std::optional<int> &workspace_relaxation,
                const int                 memory_limit_mb,
                const bool                output_details)
      {
        umfpack_.reset();
#  ifdef DEAL_II_WITH_MUMPS
        mumps_.reset();
#  endif
        // A diagonal matrix, such as a lumped kinematic mass, is inverted
        // entrywise. MUMPS needs several seconds to analyze a matrix with
        // 10^5 rows and no off-diagonal entries.
        diagonal_ = is_invertible_diagonal(matrix, communicator);
        if (diagonal_)
          {
            inverse_diagonal_.reinit(matrix.locally_owned_range_indices(),
                                     communicator);
            for (const auto row : matrix.locally_owned_range_indices())
              inverse_diagonal_(row) = 1. / matrix.diag_element(row);
            inverse_diagonal_.compress(dealii::VectorOperation::insert);
            n_entries_ = matrix.m();
            return;
          }
        if (solver == LinearSolver::umfpack)
          {
            dealii::DynamicSparsityPattern sparsity(matrix.m(), matrix.n());
            for (dealii::types::global_dof_index row = 0; row < matrix.m();
                 ++row)
              if (matrix.row_length(row) != 0)
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
              if (matrix.row_length(row) != 0)
                for (auto entry = matrix.begin(row), end = matrix.end(row);
                     entry != end;
                     ++entry)
                  serial_matrix.set(row, entry->column(), entry->value());
#  endif
            umfpack_ = std::make_unique<dealii::SparseDirectUMFPACK>();
            umfpack_->initialize(serial_matrix);
            n_entries_ = serial_matrix.n_nonzero_elements();
          }
        else if (solver == LinearSolver::mumps)
          {
#  ifdef DEAL_II_WITH_MUMPS
            AssertThrow(
              matrix.m() >=
                dealii::Utilities::MPI::n_mpi_processes(communicator),
              dealii::ExcMessage(
                "MUMPS requires at least one coupled unknown per MPI rank."));
            // MUMPS wrappers require contiguous, nonempty solution ownership.
            solution_.reinit(
              dealii::Utilities::MPI::create_evenly_distributed_partitioning(
                communicator, matrix.m()),
              communicator);
            mumps_ = std::make_unique<dealii::SparseDirectMUMPS>(
              dealii::SparseDirectMUMPS::AdditionalData(output_details ||
                                                        !reported_memory_),
              communicator);
            if (workspace_relaxation)
              mumps_->get_icntl()[13] = *workspace_relaxation;
            mumps_->get_icntl()[22] = memory_limit_mb;
            mumps_->initialize(matrix);
            if (!output_details)
              {
                // Keep the first analysis/factorization report without
                // repeating native diagnostics for every backsolve.
                auto controls = mumps_->get_icntl();
                controls[0] = controls[1] = controls[2] = -1;
                controls[3]                             = 0;
              }
            reported_memory_ = true;
            n_entries_       = matrix.n_nonzero_elements();
#  else
            (void)communicator;
            (void)workspace_relaxation;
            (void)memory_limit_mb;
            (void)output_details;
            AssertThrow(
              false,
              dealii::ExcMessage(
                "The MUMPS solver requires deal.II configured with MUMPS."));
#  endif
          }
        else
          AssertThrow(false,
                      dealii::ExcMessage(
                        "A direct factorization needs UMFPACK or MUMPS."));
      }

      /** Solve with a distributed right-hand side of the factored size. */
      void
      solve(FieldVectorType &destination, const FieldVectorType &source) const
      {
        destination.reinit(source, true);
        if (diagonal_)
          {
            destination = source;
            destination.scale(inverse_diagonal_);
          }
        else if (umfpack_)
          {
            dealii::Vector<double> serial_source(source.size()),
              serial_destination(source.size());
            for (dealii::types::global_dof_index i = 0; i < source.size(); ++i)
              serial_source[i] = source[i];
            umfpack_->vmult(serial_destination, serial_source);
            for (dealii::types::global_dof_index i = 0; i < source.size(); ++i)
              destination[i] = serial_destination[i];
          }
#  ifdef DEAL_II_WITH_MUMPS
        else if (mumps_)
          {
            mumps_->vmult(solution_, source);
            dealii::LinearAlgebra::ReadWriteVector<double> imported(
              source.locally_owned_elements());
            imported.import_elements(solution_,
                                     dealii::VectorOperation::insert);
            for (const auto index : source.locally_owned_elements())
              destination[index] = imported[index];
          }
#  endif
        else
          AssertThrow(false, dealii::ExcNotInitialized());
        destination.compress(dealii::VectorOperation::insert);
      }

    private:
      /** True on all ranks if every owned row has only a nonzero diagonal. */
      static bool
      is_invertible_diagonal(const MatrixType &matrix,
                             const MPI_Comm    communicator)
      {
        bool diagonal = true;
        for (const auto row : matrix.locally_owned_range_indices())
          {
            bool       has_diagonal = false;
            const auto n_entries    = matrix.row_length(row);
            if (n_entries != 0)
              {
                auto entry = matrix.begin(row);
                for (dealii::types::global_dof_index i = 0; i < n_entries; ++i)
                  {
                    if (entry->column() == row)
                      has_diagonal = entry->value() != 0.;
                    else if (entry->value() != 0.)
                      diagonal = false;
                    // Global ownership can have gaps between semantic fields.
                    // Never construct an accessor past this owned row.
                    if (i + 1 < n_entries)
                      ++entry;
                  }
              }
            if (!has_diagonal || !diagonal)
              {
                diagonal = false;
                break;
              }
          }
        return dealii::Utilities::MPI::min(diagonal ? 1 : 0, communicator) == 1;
      }

      bool                                         diagonal_        = false;
      bool                                         reported_memory_ = false;
      FieldVectorType                              inverse_diagonal_;
      std::unique_ptr<dealii::SparseDirectUMFPACK> umfpack_;
#  ifdef DEAL_II_WITH_MUMPS
      std::unique_ptr<dealii::SparseDirectMUMPS> mumps_;
      mutable FieldVectorType                    solution_;
#  endif
      std::size_t n_entries_ = 0;
    };
    using KinematicCondensation = typename Composition::SecondOrderCondensation;

    // Factors captured by an older preconditioner remain immutable when the
    // next Jacobian has a different mass. The cache retains only the current
    // global sparse kinematic mass, never cell-local assembly matrices.
    void
    prepare_kinematic_solvers(
      const std::shared_ptr<const KinematicCondensation> &condensation,
      const LinearSolver                                  mass_solver)
    {
      kinematic_solvers_.resize(condensation->n_pairs());
      kinematic_masses_.resize(condensation->n_pairs());
      for (std::size_t p = 0; p < condensation->n_pairs(); ++p)
        {
          const auto &mass = condensation->kinematic_mass(p);
          if (kinematic_solvers_[p] &&
              dealii::Utilities::MPI::min(kinematic_masses_[p] == mass ? 1 : 0,
                                          composition_.communicator()) == 1)
            {
              ++kinematic_factorization_reuses_;
              continue;
            }
          const auto matrix =
            mass.make_matrix(condensation->kinematic_partition(p),
                             composition_.communicator());
          auto factor = std::make_shared<DirectFactorization>();
          factor->factorize(mass_solver,
                            *matrix,
                            composition_.communicator(),
                            mumps_workspace_relaxation_,
                            mumps_memory_limit_mb_,
                            mumps_output_details_);
          ++kinematic_factorizations_;
          kinematic_masses_[p]  = mass;
          kinematic_solvers_[p] = std::move(factor);
          if (kinematic_solvers_[p]->diagonal())
            pcout << "IDAAdapter: inverted a diagonal second-order mass with "
                  << matrix->m() << " rows." << std::endl;
          else
            pcout << "IDAAdapter: "
                  << (mass_solver == LinearSolver::umfpack ? "UMFPACK" :
                                                             "MUMPS")
                  << " factorized a second-order mass with " << matrix->m()
                  << " rows, " << kinematic_solvers_[p]->n_entries()
                  << " matrix entries." << std::endl;
        }
    }

    using BlrAction =
      std::function<void(FieldVectorType &, const FieldVectorType &, bool)>;

    BlrAction
    make_mumps_blr_action(const MatrixType &matrix) const
    {
#  if defined(DEAL_II_WITH_MUMPS) && !defined(IMMERSX_USE_PETSC_LA)
      const auto communicator = composition_.communicator();
      AssertThrow(
        matrix.m() >= dealii::Utilities::MPI::n_mpi_processes(communicator),
        dealii::ExcMessage(
          "MUMPS BLR requires at least one coupled unknown per MPI rank."));

      dealii::SparseDirectMUMPS::AdditionalData data;
      data.output_details        = mumps_blr_output_details_;
      data.symmetric             = false;
      data.posdef                = false;
      data.blr_factorization     = true;
      data.blr.lowrank_threshold = mumps_blr_tolerance_;
      data.blr.blr_ucfs          = true;
      auto factor =
        std::make_shared<dealii::SparseDirectMUMPS>(data, communicator);
      auto *control = factor->get_icntl();
      if (mumps_workspace_relaxation_)
        control[13] = *mumps_workspace_relaxation_;
      control[7]  = 8; // Scale before truncating frontal matrices.
      control[9]  = 0; // FGMRES controls the physical solve accuracy.
      control[14] = 0;
      control[22] = static_cast<int>(mumps_blr_memory_limit_);
      control[35] = 1;
      pcout << "IDAAdapter: factorizing the MUMPS BLR FGMRES preconditioner."
            << std::endl;
      factor->initialize(matrix);
      auto balanced = std::make_shared<FieldVectorType>();
      balanced->reinit(
        dealii::Utilities::MPI::create_evenly_distributed_partitioning(
          communicator, matrix.m()),
        communicator);

      return [factor, balanced](FieldVectorType       &dst,
                                const FieldVectorType &src,
                                const bool             transpose) {
        dst.reinit(src, true);
        const double rhs_norm = src.l2_norm();
        AssertThrow(std::isfinite(rhs_norm),
                    dealii::ExcMessage(
                      "Non-finite MUMPS BLR right-hand side."));
        if (rhs_norm == 0.)
          {
            dst = 0.;
            return;
          }
        if (transpose)
          factor->Tvmult(*balanced, src);
        else
          factor->vmult(*balanced, src);
        dealii::LinearAlgebra::ReadWriteVector<double> imported(
          src.locally_owned_elements());
        imported.import_elements(*balanced, dealii::VectorOperation::insert);
        for (const auto index : src.locally_owned_elements())
          dst[index] = imported[index];
        dst.compress(dealii::VectorOperation::insert);
        const double norm = dst.l2_norm();
        AssertThrow(std::isfinite(norm) && norm > 0.,
                    dealii::ExcMessage(
                      "MUMPS BLR returned an invalid preconditioner action."));
      };
#  else
      (void)matrix;
      AssertThrow(
        false,
        dealii::ExcMessage(
          "MUMPS BLR FGMRES requires MUMPS and the Trilinos vector backend."));
      return {};
#  endif
    }

    template <typename BlockMatrixType>
    Operator
    make_mumps_blr_preconditioner(const Operator        &operator_view,
                                  const BlockMatrixType &blocks)
    {
      std::function<void(GlobalVectorType &, const GlobalVectorType &, bool)>
           apply;
      auto condensed = composition_.second_order_condensation(blocks);
      if (condensed)
        {
          // Native MUMPS needs a nonempty balanced solution partition. A
          // small pair can have fewer unknowns than ranks even when the full
          // coupled matrix is large enough; retain the original full fallback.
          const auto ranks = dealii::Utilities::MPI::n_mpi_processes(
            composition_.communicator());
          bool supported = condensed->reduced_matrix().m() >= ranks;
          for (std::size_t p = 0; p < condensed->n_pairs(); ++p)
            supported =
              supported && condensed->kinematic_partition(p).size() >= ranks;
          if (!supported)
            condensed.reset();
        }
      if (condensed)
        {
          prepare_kinematic_solvers(condensed, LinearSolver::mumps);
          const auto reduced_action =
            make_mumps_blr_action(condensed->reduced_matrix());
          // Capture the exact immutable factors, not the mutable cache slots.
          const auto masses = kinematic_solvers_;
          pcout << "IDAAdapter: MUMPS BLR preconditioner has "
                << condensed->reduced_matrix().m()
                << " rows after second-order condensation (" << blocks.m()
                << " coupled), "
                << condensed->reduced_matrix().n_nonzero_elements()
                << " matrix entries." << std::endl;
          apply =
            [condensed, masses, reduced_action](GlobalVectorType       &dst,
                                                const GlobalVectorType &src,
                                                const bool transpose) {
              // FGMRES only uses the forward action. The adjoint would require
              // the transposes of both elimination maps, not just S^-T.
              AssertThrow(!transpose, dealii::ExcNotImplemented());
              std::vector<FieldVectorType> kinematic(condensed->n_pairs());
              FieldVectorType              rhs;
              for (std::size_t p = 0; p < kinematic.size(); ++p)
                {
                  condensed->kinematic_rhs(src, p, rhs);
                  masses[p]->solve(kinematic[p], rhs);
                  condensed->finish_kinematic_solution(src, p, kinematic[p]);
                }
              FieldVectorType reduced_rhs, solution;
              condensed->reduced_rhs(src, kinematic, reduced_rhs);
              reduced_action(solution, reduced_rhs, false);
              condensed->expand(solution, kinematic, dst);
            };
        }
      else
        {
          MatrixType matrix;
          composition_.monolithic_matrix(blocks, matrix);
          const auto inverse = make_mumps_blr_action(matrix);
          apply              = [this, inverse](GlobalVectorType       &dst,
                                  const GlobalVectorType &src,
                                  const bool              transpose) {
            const auto      rhs = composition_.pack(src);
            FieldVectorType solution;
            inverse(solution, rhs, transpose);
            composition_.unpack(solution, dst);
          };
        }

      Operator result;
      result.reinit_range_vector  = operator_view.reinit_range_vector;
      result.reinit_domain_vector = operator_view.reinit_domain_vector;
      result.vmult                = [apply](GlobalVectorType       &dst,
                             const GlobalVectorType &src) {
        apply(dst, src, false);
      };
      result.Tvmult = [apply](GlobalVectorType       &dst,
                              const GlobalVectorType &src) {
        apply(dst, src, true);
      };
      auto memory =
        std::make_shared<dealii::GrowingVectorMemory<GlobalVectorType>>();
      result.vmult_add = [apply, memory](GlobalVectorType       &dst,
                                         const GlobalVectorType &src) {
        typename dealii::VectorMemory<GlobalVectorType>::Pointer temporary(
          *memory);
        temporary->reinit(dst);
        apply(*temporary, src, false);
        dst += *temporary;
      };
      result.Tvmult_add = [apply, memory](GlobalVectorType       &dst,
                                          const GlobalVectorType &src) {
        typename dealii::VectorMemory<GlobalVectorType>::Pointer temporary(
          *memory);
        temporary->reinit(dst);
        apply(*temporary, src, true);
        dst += *temporary;
      };
      return result;
    }

    void
    finalize()
    {
      if (connected_)
        return;

      dealii::TimerOutput::Scope timer(computing_timer, "Finalize adapter");
      AssertThrow(composition_.n_fields() > 0,
                  dealii::ExcMessage("IDAAdapter has no semantic fields."));
      tolerance_ramp_ = ida_parameters_.tolerance_ramp;
      composition_.set_second_order_inner_solver(
        ida_parameters_.linear_solver_controls.second_order_maximum_iterations,
        ida_parameters_.linear_solver_controls.second_order_relative_tolerance);
      auto ida_data = additional_data();
      if (stop_at_output_times_)
        AssertThrow(
          std::isfinite(ida_data.output_period) && ida_data.output_period > 0.,
          dealii::ExcMessage("IDA output period must be finite and positive."));
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
        solved_since_residual_ = false;
        diagnostic_time_       = time;
        if (ida_parameters_.print_nonlinear_diagnostics)
          ++diagnostic_residuals_;
        try
          {
            composition_.evaluate_residual(time, state, &state_dot, residual);
          }
        catch (const dealii::RecoverableUserCallbackError &error)
          {
            if (ida_parameters_.print_nonlinear_diagnostics)
              std::cerr << "IDA nonlinear recoverable rank="
                        << dealii::Utilities::MPI::this_mpi_process(
                             composition_.communicator())
                        << " trial=" << diagnostic_residuals_ << " t=" << time
                        << " reason=" << error.what() << std::endl;
            throw;
          }
        if (ida_parameters_.print_nonlinear_diagnostics)
          print_nonlinear_residual(time, state, residual);
        // Let IDA reject a nonfinite trial instead of using it to construct a
        // Newton correction. All ranks must report the same callback status.
        bool finite = true;
        for (const auto index : residual.locally_owned_elements())
          if (!std::isfinite(residual[index]))
            {
              finite = false;
              break;
            }
        if (dealii::Utilities::MPI::min(finite ? 1 : 0,
                                        composition_.communicator()) == 0)
          {
            if (ida_parameters_.print_nonlinear_diagnostics)
              pcout << "IDA nonlinear recoverable trial="
                    << diagnostic_residuals_ << " t=" << time
                    << " reason=nonfinite residual" << std::endl;
            throw dealii::RecoverableUserCallbackError();
          }
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
        if (ida_parameters_.print_nonlinear_diagnostics)
          pcout << "IDA nonlinear linear begin solve=" << linear_solves_
                << " t=" << diagnostic_time_
                << " requested_tolerance=" << tolerance << std::endl;
        solved_since_residual_ = true;
        dealii::TimerOutput::Scope timer(computing_timer, "Linear solve");
        AssertThrow(current_jacobian_.has_value(),
                    dealii::ExcMessage("IDA requested a solve without a "
                                       "current Jacobian."));
        if (linear_solver_ == LinearSolver::umfpack ||
            linear_solver_ == LinearSolver::mumps)
          {
            dealii::TimerOutput::Scope backsolve_timer(computing_timer,
                                                       "Direct backsolve");
            AssertThrow(
              direct_solver_.initialized(),
              dealii::ExcMessage(
                "IDA requested a direct solve without a factorization."));
            GlobalVectorType original_rhs(rhs);
            original_rhs.compress(dealii::VectorOperation::insert);
            const auto backsolve = [this](const GlobalVectorType &input,
                                          GlobalVectorType       &output) {
              if (condensation_)
                {
                  std::vector<FieldVectorType> kinematic(
                    condensation_->n_pairs());
                  FieldVectorType kinematic_rhs;
                  for (std::size_t p = 0; p < kinematic.size(); ++p)
                    {
                      condensation_->kinematic_rhs(input, p, kinematic_rhs);
                      kinematic_solvers_[p]->solve(kinematic[p], kinematic_rhs);
                      condensation_->finish_kinematic_solution(input,
                                                               p,
                                                               kinematic[p]);
                    }
                  FieldVectorType reduced_rhs, reduced_solution;
                  condensation_->reduced_rhs(input, kinematic, reduced_rhs);
                  direct_solver_.solve(reduced_solution, reduced_rhs);
                  condensation_->expand(reduced_solution, kinematic, output);
                }
              else
                {
                  auto source = composition_.pack(input);
                  source.compress(dealii::VectorOperation::insert);
                  FieldVectorType solution;
                  direct_solver_.solve(solution, source);
                  composition_.unpack(solution, output);
                  output.compress(dealii::VectorOperation::insert);
                }
            };
            backsolve(original_rhs, dst);
            if (linear_solver_ == LinearSolver::mumps)
              {
                dealii::TimerOutput::Scope refinement_timer(
                  computing_timer, "MUMPS iterative refinement");
                // MUMPS disables its internal refinement for distributed
                // right-hand sides. Reuse the factorization for at most three
                // corrections, measuring each defect with the retained Jacobian
                // action.
                GlobalVectorType defect;
                composition_.reinit(defect);
                const auto  &controls = ida_parameters_.linear_solver_controls;
                const double relative_target =
                  controls.mumps_relative_tolerance;
                const unsigned int max_refinements =
                  controls.mumps_maximum_corrections;
                const double        rhs_norm    = original_rhs.l2_norm();
                double              defect_norm = 0.;
                unsigned int        refinement  = 0;
                std::vector<double> defect_history(max_refinements + 1);
                for (;; ++refinement)
                  {
                    current_jacobian_->vmult(defect, dst);
                    defect -= original_rhs;
                    defect_norm                = defect.l2_norm();
                    defect_history[refinement] = defect_norm;
                    if (!std::isfinite(defect_norm) ||
                        defect_norm <= relative_target * rhs_norm ||
                        refinement == max_refinements)
                      break;
                    GlobalVectorType correction;
                    composition_.reinit(correction);
                    backsolve(defect, correction);
                    dst -= correction;
                    dst.compress(dealii::VectorOperation::insert);
                  }
                unsigned int krylov_corrections = 0;
                if (std::isfinite(defect_norm) &&
                    defect_norm > relative_target * rhs_norm &&
                    controls.mumps_maximum_krylov_corrections > 0)
                  {
                    krylov_corrections =
                      detail::mumps_krylov_refinement<GlobalVectorType>(
                        *current_jacobian_,
                        backsolve,
                        original_rhs,
                        dst,
                        controls.mumps_maximum_krylov_corrections,
                        relative_target * rhs_norm);
                    current_jacobian_->vmult(defect, dst);
                    defect -= original_rhs;
                    defect_norm = defect.l2_norm();
                    std::ostringstream recovery_message;
                    recovery_message
                      << "IDAAdapter: MUMPS Krylov recovery used "
                      << krylov_corrections << " iteration(s), true residual "
                      << std::scientific << std::setprecision(17) << defect_norm
                      << ", target " << relative_target * rhs_norm;
                    pcout << recovery_message.str() << std::endl;
                  }
                if (ida_parameters_.print_nonlinear_diagnostics)
                  {
                    print_nonlinear_linear_defect(original_rhs, defect);
                    print_nonlinear_correction(dst);
                  }
                // Some supported deal.II MUMPS wrappers do not report failed
                // factorizations. Do not let a zero correction freeze the DAE.
                if (!(std::isfinite(defect_norm) &&
                      defect_norm <= relative_target * rhs_norm))
                  {
                    std::ostringstream message;
                    message << std::scientific << std::setprecision(17)
                            << "MUMPS backsolve failed the retained-Jacobian "
                               "residual check. RHS L2 norm = "
                            << rhs_norm << ", defect L2 norm = " << defect_norm
                            << ", required defect L2 norm <= "
                            << relative_target * rhs_norm
                            << ", relative target = " << relative_target
                            << ", relative defect = ";
                    if (rhs_norm > 0.)
                      message << defect_norm / rhs_norm;
                    else
                      message << "undefined";
                    message << ", correction solves = " << refinement
                            << ", Krylov correction iterations = "
                            << krylov_corrections << ", defect L2 history = [";
                    for (unsigned int i = 0; i <= refinement; ++i)
                      {
                        if (i != 0)
                          message << ", ";
                        message << defect_history[i];
                      }
                    message
                      << "]. Inspect native diagnostics and the workspace "
                         "relaxation.";
                    AssertThrow(false, dealii::ExcMessage(message.str()));
                  }
              }
          }
        else if (solve_)
          {
            pcout << "IDAAdapter: invoking the configured linear solver."
                  << std::endl;
            solve_(*current_jacobian_, rhs, dst, tolerance);
            pcout << "IDAAdapter: configured linear solver finished."
                  << std::endl;
          }
        else if (ida_ && ida_->get_ida_memory())
          {
            // A restart may resize native vectors without changing field
            // registration. Follow the current IDA callback map, not that
            // original registration-time partition.
            GlobalVectorType weights, ones;
            weights.reinit(rhs);
            ones.reinit(rhs);
            ones = 1.;
            auto weight_view =
              Compatibility::SUNDIALS::internal::make_nvector_view(
                weights
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
                ,
                ida_->get_sun_context()
#  endif
              );
            auto ones_view =
              Compatibility::SUNDIALS::internal::make_nvector_view(
                ones
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
                ,
                ida_->get_sun_context()
#  endif
              );
            const int status =
              IDAGetErrWeights(ida_->get_ida_memory(), weight_view);
            AssertThrow(
              dealii::Utilities::MPI::min(status == IDA_SUCCESS ? 1u : 0u,
                                          composition_.communicator()),
              dealii::ExcMessage(
                "Could not obtain native IDA linear error weights."));
            bool valid = !std::isnan(tolerance) && tolerance >= 0.;
            for (const auto i : weights.locally_owned_elements())
              valid = valid && std::isfinite(weights[i]) && weights[i] > 0.;
            AssertThrow(dealii::Utilities::MPI::min(
                          valid ? 1u : 0u, composition_.communicator()),
                        dealii::ExcMessage(
                          "Invalid native IDA linear weights or tolerance."));
            const double weight_rms = N_VWrmsNorm(weight_view, ones_view);
            AssertThrow(!std::isnan(weight_rms) && weight_rms >= 0.,
                        dealii::ExcMessage("Invalid native IDA weight RMS."));
            bool   exact_only      = weight_rms == 0. || tolerance == 0.;
            double weighted_target = 0.;
            if (!exact_only)
              {
                // Avoid evaluating 0*Inf. Classify overflow before converting
                // the recovered target back to double precision.
                const long double product =
                  static_cast<long double>(tolerance) * weight_rms;
                weighted_target = product > std::numeric_limits<double>::max() ?
                                    std::numeric_limits<double>::infinity() :
                                    tolerance * weight_rms;
                exact_only      = weighted_target == 0.;
              }
            const bool   unbounded = std::isinf(weighted_target);
            const auto  &controls  = ida_parameters_.linear_solver_controls;
            const double raw_target =
              controls.krylov_relative_tolerance * rhs.l2_norm();
            const auto *preconditioner =
              current_preconditioner_ ? &*current_preconditioner_ : nullptr;
            if (exact_only || (unbounded && raw_target == 0.))
              {
                const auto iterations = detail::ida_exact_residual_krylov(
                  *current_jacobian_,
                  preconditioner,
                  rhs,
                  dst,
                  raw_target,
                  controls.krylov_maximum_iterations,
                  current_solver_is_flexible_,
                  composition_.communicator());
                pcout
                  << "IDAAdapter: indeterminate native weighted target; exact full-J residual certified after "
                  << iterations << " iteration(s)." << std::endl;
              }
            else if (unbounded)
              {
                // An infinite native target imposes no finite weighted bound.
                // Keep the independent raw full-J gate, with identity scaling.
                const auto iterations = detail::ida_weighted_krylov(
                  *current_jacobian_,
                  preconditioner,
                  rhs,
                  dst,
                  ones,
                  raw_target,
                  raw_target,
                  controls.krylov_maximum_iterations,
                  current_solver_is_flexible_,
                  composition_.communicator());
                pcout
                  << "IDAAdapter: unbounded native weighted target; raw full-J gate passed after "
                  << iterations << " iteration(s)." << std::endl;
              }
            else
              {
                // Undo IDAS's homogeneous-weight fallback without sqrt(N).
                const auto iterations = detail::ida_weighted_krylov(
                  *current_jacobian_,
                  preconditioner,
                  rhs,
                  dst,
                  weights,
                  weighted_target,
                  raw_target,
                  controls.krylov_maximum_iterations,
                  current_solver_is_flexible_,
                  composition_.communicator());
                pcout << "IDAAdapter: weighted Krylov converged in "
                      << iterations << " iteration(s), native weighted target "
                      << weighted_target << ", raw relative target "
                      << controls.krylov_relative_tolerance << std::endl;
              }
          }
        else
          {
            dst = 0.;
            // Small physical loads still need a correction. An absolute floor
            // can accept the initial zero iterate and stall the DAE solve.
            const auto &controls = ida_parameters_.linear_solver_controls;
            dealii::SolverControl control(
              controls.krylov_maximum_iterations,
              std::min(tolerance,
                       controls.krylov_relative_tolerance * rhs.l2_norm()));
            const auto target = [&control, tolerance]() {
              std::ostringstream text;
              text << ", target " << control.tolerance() << " (IDA "
                   << tolerance << ")";
              return text.str();
            };
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
                        << target() << "." << std::endl;
                }
              else
                {
                  solver.solve(*current_jacobian_,
                               dst,
                               rhs,
                               *current_preconditioner_);
                  pcout << "IDAAdapter: GMRES converged in "
                        << control.last_step() << " iteration(s), residual "
                        << control.last_value() << target() << "." << std::endl;
                }
            else
              {
                solver.solve(*current_jacobian_,
                             dst,
                             rhs,
                             dealii::PreconditionIdentity());
                pcout << "IDAAdapter: GMRES converged in "
                      << control.last_step() << " iteration(s), residual "
                      << control.last_value() << target() << "." << std::endl;
              }
          }
        if (ida_parameters_.print_nonlinear_diagnostics &&
            linear_solver_ != LinearSolver::mumps)
          print_nonlinear_correction(dst);
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

        bool restart = false;
        {
          dealii::TimerOutput::Scope timer(computing_timer, "Restart check");
          restart = solver_should_restart_(time, state, state_dot);
        }
        if (restart)
          {
            accumulated_statistics_ = statistics();
            ++restarts_;
            initial_condition_jacobian_time_.reset();
          }
        if (restart && compute_consistent_initial_conditions_ &&
            additional_data().reset_type == AdditionalData::use_y_diff)
          {
            pcout << "IDAAdapter: computing consistent initial conditions "
                     "after restart with the application callback."
                  << std::endl;
            dealii::TimerOutput::Scope timer(computing_timer,
                                             "Consistent initial conditions");
            compute_consistent_initial_conditions_(time, state, state_dot);
          }
        return restart;
      };
      // Internal bookkeeping must run even when no application output is set.
      ida_->output_step = [this](const double            time,
                                 const GlobalVectorType &state,
                                 const GlobalVectorType &state_dot,
                                 const unsigned int      step) {
        if (stop_at_output_times_)
          update_output_stop(time);
        else
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
          {
            dealii::TimerOutput::Scope timer(computing_timer,
                                             "Output callback");
            output_(time, state, state_dot, step);
          }
      };
      pcout << "IDAAdapter: finalized with " << composition_.n_fields()
            << " semantic field(s)." << std::endl;
      connected_ = true;
    }

    void
    print_nonlinear_residual(const double            time,
                             const GlobalVectorType &state,
                             const GlobalVectorType &residual) const
    {
      const auto communicator = composition_.communicator();
      for (std::size_t i = 0; i < composition_.n_fields(); ++i)
        {
          const FieldId id(i);
          const auto   &values  = composition_.field(state, id);
          const auto   &forces  = composition_.field(residual, id);
          double        squares = 0., maximum = 0.;
          double        lower     = std::numeric_limits<double>::infinity();
          double        upper     = -std::numeric_limits<double>::infinity();
          unsigned long bad_state = 0, bad_residual = 0;
          for (const auto index : values.locally_owned_elements())
            {
              const double value = values[index];
              const double force = forces[index];
              if (std::isfinite(value))
                {
                  lower = std::min(lower, value);
                  upper = std::max(upper, value);
                }
              else
                ++bad_state;
              if (std::isfinite(force))
                {
                  squares += force * force;
                  maximum = std::max(maximum, std::abs(force));
                }
              else
                ++bad_residual;
            }
          squares   = dealii::Utilities::MPI::sum(squares, communicator);
          maximum   = dealii::Utilities::MPI::max(maximum, communicator);
          lower     = dealii::Utilities::MPI::min(lower, communicator);
          upper     = dealii::Utilities::MPI::max(upper, communicator);
          bad_state = dealii::Utilities::MPI::sum(bad_state, communicator);
          bad_residual =
            dealii::Utilities::MPI::sum(bad_residual, communicator);
          std::ostringstream line;
          line << std::scientific << std::setprecision(17)
               << "IDA nonlinear residual trial=" << diagnostic_residuals_
               << " t=" << time << " field=" << i << " name="
               << std::quoted(composition_.state_layout().field(id).name)
               << " F_l2="
               << (bad_residual ? std::numeric_limits<double>::quiet_NaN() :
                                  std::sqrt(squares))
               << " F_linf="
               << (bad_residual ? std::numeric_limits<double>::quiet_NaN() :
                                  maximum)
               << " y_min=" << lower << " y_max=" << upper
               << " nonfinite_y=" << bad_state
               << " nonfinite_F=" << bad_residual;
          pcout << line.str() << std::endl;
        }
    }

    void
    print_nonlinear_linear_defect(const GlobalVectorType &rhs,
                                  const GlobalVectorType &defect) const
    {
      const auto communicator = composition_.communicator();
      for (std::size_t i = 0; i < composition_.n_fields(); ++i)
        {
          const FieldId id(i);
          const auto   &source      = composition_.field(rhs, id);
          const auto   &error       = composition_.field(defect, id);
          double        rhs_squared = 0., defect_squared = 0., maximum = 0.;
          unsigned long bad_rhs = 0, bad_defect = 0;
          for (const auto index : source.locally_owned_elements())
            {
              const double value = source[index], difference = error[index];
              if (std::isfinite(value))
                rhs_squared += value * value;
              else
                ++bad_rhs;
              if (std::isfinite(difference))
                {
                  defect_squared += difference * difference;
                  maximum = std::max(maximum, std::abs(difference));
                }
              else
                ++bad_defect;
            }
          rhs_squared = dealii::Utilities::MPI::sum(rhs_squared, communicator);
          defect_squared =
            dealii::Utilities::MPI::sum(defect_squared, communicator);
          maximum    = dealii::Utilities::MPI::max(maximum, communicator);
          bad_rhs    = dealii::Utilities::MPI::sum(bad_rhs, communicator);
          bad_defect = dealii::Utilities::MPI::sum(bad_defect, communicator);
          const double       invalid = std::numeric_limits<double>::quiet_NaN();
          std::ostringstream line;
          line << std::scientific << std::setprecision(17)
               << "IDA nonlinear linear defect solve=" << linear_solves_
               << " field=" << i << " name="
               << std::quoted(composition_.state_layout().field(id).name)
               << " rhs_l2=" << (bad_rhs ? invalid : std::sqrt(rhs_squared))
               << " defect_l2="
               << (bad_defect ? invalid : std::sqrt(defect_squared))
               << " defect_linf=" << (bad_defect ? invalid : maximum)
               << " nonfinite_rhs=" << bad_rhs
               << " nonfinite_defect=" << bad_defect;
          pcout << line.str() << std::endl;
        }
    }

    void
    print_nonlinear_correction(const GlobalVectorType &correction) const
    {
      if (!ida_ || !ida_->get_ida_memory())
        {
          pcout
            << "IDA nonlinear correction solve=" << linear_solves_
            << " field=all wrms=unavailable reason=IDA weights not initialized"
            << std::endl;
          return;
        }
      GlobalVectorType weights;
      composition_.reinit(weights);
      auto view = Compatibility::SUNDIALS::internal::make_nvector_view(
        weights
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
        ,
        ida_->get_sun_context()
#  endif
      );
      const int status = IDAGetErrWeights(ida_->get_ida_memory(), view);
      if (status < 0)
        {
          pcout << "IDA nonlinear correction solve=" << linear_solves_
                << " field=all wrms=unavailable IDAGetErrWeights_status="
                << status << std::endl;
          return;
        }
      const auto communicator = composition_.communicator();
      double     total        = 0.;
      for (std::size_t i = 0; i < composition_.n_fields(); ++i)
        {
          const FieldId id(i);
          const auto   &values  = composition_.field(correction, id);
          const auto   &errors  = composition_.field(weights, id);
          double        squares = 0.;
          for (const auto index : values.locally_owned_elements())
            {
              const double scaled = values[index] * errors[index];
              squares += scaled * scaled;
            }
          squares = dealii::Utilities::MPI::sum(squares, communicator);
          total += squares;
          std::ostringstream line;
          line << std::scientific << std::setprecision(17)
               << "IDA nonlinear correction solve=" << linear_solves_
               << " t=" << diagnostic_time_ << " field=" << i << " name="
               << std::quoted(composition_.state_layout().field(id).name)
               << " wrms="
               << (values.size() ? std::sqrt(squares / values.size()) : 0.);
          pcout << line.str() << std::endl;
        }
      std::ostringstream line;
      line << std::scientific << std::setprecision(17)
           << "IDA nonlinear correction solve=" << linear_solves_
           << " t=" << diagnostic_time_ << " field=all wrms="
           << (correction.size() ? std::sqrt(total / correction.size()) : 0.);
      pcout << line.str() << std::endl;
    }

    void
    print_nonlinear_summary(const char *outcome) const
    {
      if (!ida_parameters_.print_nonlinear_diagnostics || !ida_ ||
          !ida_->get_ida_memory())
        return;
      // No collectives or throwing checks here: this also runs while handling
      // an exception and must preserve the original solver failure.
      long int backtracks = 0, residuals = 0, iterations = 0, failures = 0;
      void    *memory = ida_->get_ida_memory();
      if (IDAGetNumBacktrackOps(memory, &backtracks) >= 0 &&
          IDAGetNumResEvals(memory, &residuals) >= 0 &&
          IDAGetNumNonlinSolvIters(memory, &iterations) >= 0 &&
          IDAGetNumNonlinSolvConvFails(memory, &failures) >= 0)
        pcout << "IDA nonlinear summary outcome=" << outcome
              << " backtracks=" << backtracks << " residuals=" << residuals
              << " iterations=" << iterations << " failures=" << failures
              << " residual_callbacks=" << diagnostic_residuals_
              << " Jacobian_setups=" << jacobian_setups_
              << " linear_solves=" << linear_solves_ << std::endl;
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
      dealii::TimerOutput::Scope timer(computing_timer,
                                       "Initialize tolerances");
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
      for (std::size_t i = 0; i < composition_.n_fields(); ++i)
        {
          const FieldId field(i);
          const bool    ramp = std::find(unramped_fields_.begin(),
                                      unramped_fields_.end(),
                                      field) == unramped_fields_.end();
          const auto   &differential =
            composition_.state_layout().field(field).differential_components;
          const auto &values = composition_.field(absolute_tolerances_, field);
          for (const auto index : values.locally_owned_elements())
            {
              const bool ramp_index =
                ramp && (!tolerance_ramp_.protect_algebraic_variables ||
                         differential.is_element(index));
              const double value = values[index];
              valid              = valid &&
                      std::isfinite(additional_data().relative_tolerance *
                                    (ramp_index ? scales.first : 1.)) &&
                      std::isfinite(value) && value > 0. &&
                      std::isfinite(value * (ramp_index ? scales.second : 1.));
            }
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
      dealii::TimerOutput::Scope timer(computing_timer, "Error weights");
      double                     time;
      check_ida(IDAGetCurrentTime(ida_->get_ida_memory(), &time));
      const auto scales =
        tolerance_ramp_.scales(additional_data().initial_time, time);
      bool valid = std::isfinite(time);
      for (std::size_t i = 0; i < composition_.n_fields(); ++i)
        {
          const FieldId field(i);
          const bool    ramp   = std::find(unramped_fields_.begin(),
                                      unramped_fields_.end(),
                                      field) == unramped_fields_.end();
          const auto   &values = composition_.field(state, field);
          const auto   &absolute =
            composition_.field(absolute_tolerances_, field);
          auto       &field_weights = composition_.field(weights, field);
          const auto &differential =
            composition_.state_layout().field(field).differential_components;
          for (const auto index : field_weights.locally_owned_elements())
            {
              const bool ramp_index =
                ramp && (!tolerance_ramp_.protect_algebraic_variables ||
                         differential.is_element(index));
              const double value = values[index];
              const double denominator =
                additional_data().relative_tolerance *
                  (ramp_index ? scales.first : 1.) * std::abs(value) +
                (ramp_index ? scales.second : 1.) * absolute[index];
              field_weights[index] = 1. / denominator;
              valid                = valid && std::isfinite(value) &&
                      std::isfinite(field_weights[index]) &&
                      field_weights[index] > 0.;
            }
        }
      weights.compress(dealii::VectorOperation::insert);
      AssertThrow(dealii::Utilities::MPI::min(valid ? 1 : 0,
                                              composition_.communicator()) == 1,
                  dealii::ExcMessage(
                    "Non-finite state or invalid IDA error weights."));
      ++weight_evaluations_;
    }

    void
    update_output_stop(const double output_time)
    {
      const auto  &data = additional_data();
      const double tolerance =
        IDAToleranceRamp::time_tolerance(data.initial_time, data.final_time);
      if (output_time >= data.final_time - tolerance)
        return;
      double next = std::min(output_time + data.output_period, data.final_time);
      // Accumulated nonbinary output periods can approach the final time from
      // below. Avoid creating a final interval shorter than roundoff.
      if (next >= data.final_time - tolerance)
        next = data.final_time;
      if (tolerance_ramp_.enabled &&
          output_time < tolerance_ramp_.end_time - tolerance)
        next = std::min(next, tolerance_ramp_.end_time);
      check_ida(IDASetStopTime(ida_->get_ida_memory(), next));
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
                     const double            alpha,
                     const bool              steady_setup     = false,
                     const GlobalVectorType *characteristics  = nullptr,
                     GlobalVectorType       *function_scaling = nullptr)
    {
      // With `use_y_dot`, IDACalcIC solves F(t0, y, y') = 0 for y with alpha
      // = 0. After the first correction pass it resets the error weights and
      // runs a second pass, which evaluates the residual at the converged
      // state and requests a new Jacobian before its first solve. Keep the
      // first-pass Jacobian for that pass, as IDA's modified Newton iteration
      // does within a pass; the convergence test still uses the residual at
      // the current state. A request that follows a solve signals slow
      // convergence and always rebuilds the Jacobian.
      if (!steady_setup && alpha == 0. && !solved_since_residual_ &&
          current_jacobian_.has_value() &&
          initial_condition_jacobian_time_ == time)
        {
          pcout << "IDAAdapter: reusing the initial-condition Jacobian for "
                   "IDA's second correction pass."
                << std::endl;
          return;
        }
      ++jacobian_setups_;
      diagnostic_time_ = time;
      if (ida_parameters_.print_nonlinear_diagnostics)
        pcout << "IDA nonlinear Jacobian setup=" << jacobian_setups_
              << " t=" << time << " alpha=" << alpha << std::endl;
      initial_condition_jacobian_time_.reset();
      dealii::TimerOutput::Scope timer(computing_timer, "Setup Jacobian");
      const bool                 retain_matrices = !solve_;
      const auto                 snapshot =
        composition_.make_snapshot(time, state, state_dot, retain_matrices);
      const auto operator_view = std::make_shared<const Operator>(
        composition_.jacobian(snapshot->context,
                              alpha,
                              retain_matrices ? &snapshot->linearization :
                                                nullptr));

      Operator stable;
      stable.reinit_range_vector =
        [operator_view, snapshot](GlobalVectorType &vector, const bool omit) {
          operator_view->reinit_range_vector(vector, omit);
        };
      stable.reinit_domain_vector =
        [operator_view, snapshot](GlobalVectorType &vector, const bool omit) {
          operator_view->reinit_domain_vector(vector, omit);
        };
      stable.vmult = [operator_view, snapshot](GlobalVectorType &destination,
                                               const GlobalVectorType &source) {
        operator_view->vmult(destination, source);
      };
      stable.vmult_add = [operator_view,
                          snapshot](GlobalVectorType       &destination,
                                    const GlobalVectorType &source) {
        operator_view->vmult_add(destination, source);
      };
      stable.Tvmult = [operator_view,
                       snapshot](GlobalVectorType       &destination,
                                 const GlobalVectorType &source) {
        operator_view->Tvmult(destination, source);
      };
      stable.Tvmult_add = [operator_view,
                           snapshot](GlobalVectorType       &destination,
                                     const GlobalVectorType &source) {
        operator_view->Tvmult_add(destination, source);
      };
      current_jacobian_ = std::move(stable);
      current_preconditioner_.reset();
      current_solver_is_flexible_ = false;
      if (linear_solver_ == LinearSolver::umfpack ||
          linear_solver_ == LinearSolver::mumps)
        {
          dealii::TimerOutput::Scope direct_timer(computing_timer,
                                                  "Materialize Jacobian");
          const auto                 blocks =
            composition_.block_matrix(snapshot->state_storage,
                                      &snapshot->derivative_storage,
                                      alpha,
                                      time,
                                      &snapshot->linearization);
          condensation_ = composition_.second_order_condensation(blocks);
          if (function_scaling != nullptr)
            make_steady_function_scaling(blocks,
                                         *characteristics,
                                         *function_scaling);
          std::shared_ptr<MatrixType> monolithic;
          if (!condensation_)
            {
              monolithic = std::make_shared<MatrixType>();
              composition_.monolithic_matrix(blocks, *monolithic);
            }
          direct_timer.stop();
          dealii::TimerOutput::Scope factor_timer(computing_timer,
                                                  "Factorize Jacobian");
          const std::string          solver_name =
            linear_solver_ == LinearSolver::umfpack ? "UMFPACK" : "MUMPS";
          if (condensation_)
            {
              prepare_kinematic_solvers(condensation_, linear_solver_);
              const auto &reduced = condensation_->reduced_matrix();
              direct_solver_.factorize(linear_solver_,
                                       reduced,
                                       composition_.communicator(),
                                       mumps_workspace_relaxation_,
                                       mumps_memory_limit_mb_,
                                       mumps_output_details_);
              pcout << "IDAAdapter: " << solver_name << " factorized "
                    << reduced.m() << " rows after second-order condensation ("
                    << blocks.m() << " coupled), " << direct_solver_.n_entries()
                    << " matrix entries." << std::endl;
            }
          else
            {
              direct_solver_.factorize(linear_solver_,
                                       *monolithic,
                                       composition_.communicator(),
                                       mumps_workspace_relaxation_,
                                       mumps_memory_limit_mb_,
                                       mumps_output_details_);
              pcout << "IDAAdapter: " << solver_name << " factorized "
                    << monolithic->m() << " rows, "
                    << direct_solver_.n_entries() << " matrix entries."
                    << std::endl;
            }
        }
      else if (!solve_)
        {
          const auto keep_snapshot = [snapshot](Operator operator_view) {
            const auto retained =
              std::make_shared<const Operator>(std::move(operator_view));
            Operator result;
            result.reinit_range_vector =
              [retained, snapshot](GlobalVectorType &vector, const bool omit) {
                retained->reinit_range_vector(vector, omit);
              };
            result.reinit_domain_vector =
              [retained, snapshot](GlobalVectorType &vector, const bool omit) {
                retained->reinit_domain_vector(vector, omit);
              };
            result.vmult = [retained,
                            snapshot](GlobalVectorType       &destination,
                                      const GlobalVectorType &source) {
              retained->vmult(destination, source);
            };
            result.vmult_add = [retained,
                                snapshot](GlobalVectorType       &destination,
                                          const GlobalVectorType &source) {
              retained->vmult_add(destination, source);
            };
            result.Tvmult = [retained,
                             snapshot](GlobalVectorType       &destination,
                                       const GlobalVectorType &source) {
              retained->Tvmult(destination, source);
            };
            result.Tvmult_add = [retained,
                                 snapshot](GlobalVectorType       &destination,
                                           const GlobalVectorType &source) {
              retained->Tvmult_add(destination, source);
            };
            return result;
          };
          if (linear_solver_ == LinearSolver::mumps_blr_fgmres)
            {
              auto blocks =
                composition_.block_matrix(snapshot->state_storage,
                                          &snapshot->derivative_storage,
                                          alpha,
                                          time,
                                          &snapshot->linearization);
              current_preconditioner_ =
                make_mumps_blr_preconditioner(*current_jacobian_, blocks);
              current_solver_is_flexible_ = true;
            }
          else if (block_preconditioner_ == BlockPreconditioner::block_diagonal)
            {
              current_preconditioner_ =
                keep_snapshot(composition_.block_diagonal_preconditioner(
                  snapshot->state_storage,
                  &snapshot->derivative_storage,
                  alpha,
                  snapshot.get()));
            }
          else if (block_preconditioner_ ==
                   BlockPreconditioner::block_triangular)
            {
              current_preconditioner_ =
                keep_snapshot(composition_.block_triangular_preconditioner(
                  snapshot->state_storage,
                  true,
                  &snapshot->derivative_storage,
                  alpha,
                  snapshot.get()));
              current_solver_is_flexible_ = true;
            }
          else if (!composition_.saddle_points().empty() &&
                   (block_preconditioner_ == BlockPreconditioner::automatic ||
                    block_preconditioner_ == BlockPreconditioner::schur))
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
                  ida_parameters_.linear_solver_controls
                    .schur_maximum_iterations,
                  ida_parameters_.linear_solver_controls
                    .schur_relative_tolerance,
                  include_schur_primal_coupling_,
                  snapshot.get(),
                  schur_materialization_limit_));
              current_solver_is_flexible_ = true;
            }
          else if (composition_.has_isolated_second_order_pair())
            {
              current_preconditioner_ =
                keep_snapshot(composition_.second_order_preconditioner(
                  snapshot->state_storage,
                  snapshot->derivative_storage,
                  alpha,
                  snapshot.get()));
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
                                  alpha,
                                  snapshot.get()) :
                                composition_.block_diagonal_preconditioner(
                                  snapshot->state_storage,
                                  &snapshot->derivative_storage,
                                  alpha,
                                  snapshot.get()));
              current_solver_is_flexible_ = composition_.n_fields() > 1;
            }
        }
      if (alpha == 0.)
        initial_condition_jacobian_time_ = time;
    }

    void
    make_steady_function_scaling(
      const typename Composition::BlockMatrixType &blocks,
      const GlobalVectorType                      &characteristics,
      GlobalVectorType                            &scaling)
    {
#  ifndef IMMERSX_USE_PETSC_LA
      bool valid = true;
      for (const auto index : characteristics.locally_owned_elements())
        {
          const double value = characteristics[index];
          valid              = valid && std::isfinite(value) && value > 0. &&
                  std::isfinite(1. / value);
        }
      AssertThrow(
        dealii::Utilities::MPI::min(valid ? 1u : 0u,
                                    composition_.communicator()) == 1,
        dealii::ExcMessage(
          "Steady state characteristics and their reciprocals must be finite and positive."));
      // Scale the original residual, including kinematic rows, even when its
      // linear solve uses condensation. Blocks have already summed all terms
      // in each entry; taking absolute values here preserves cancellations.
      // Only owned row sums and one column import exist at a time.
      for (unsigned int i = 0; i < blocks.n_block_rows(); ++i)
        {
          auto                    &output = scaling.block(i);
          const auto              &owned  = output.locally_owned_elements();
          std::vector<long double> sums(owned.n_elements(), 0.);
          for (unsigned int j = 0; j < blocks.n_block_cols(); ++j)
            {
              const auto     &matrix = blocks.block(i, j);
              FieldVectorType columns;
              columns.import_nonlocal_data_for_fe(matrix,
                                                  characteristics.block(j));
              const auto &native        = matrix.trilinos_matrix();
              const auto *column_values = columns.trilinos_vector()[0];
              valid                     = valid && native.NumMyRows() ==
                                 static_cast<int>(owned.n_elements());
              // Native local views never advance into holes between fields.
              for (int local_row = 0; local_row < native.NumMyRows();
                   ++local_row)
                {
                  int        n_entries      = 0;
                  double    *values         = nullptr;
                  int       *column_indices = nullptr;
                  const auto row = static_cast<dealii::types::global_dof_index>(
                    native.GRID64(local_row));
                  if (!owned.is_element(row) ||
                      native.ExtractMyRowView(
                        local_row, n_entries, values, column_indices) != 0)
                    {
                      valid = false;
                      continue;
                    }
                  auto &sum = sums[owned.index_within_set(row)];
                  for (int entry = 0; entry < n_entries; ++entry)
                    sum += std::abs(static_cast<long double>(values[entry])) *
                           column_values[column_indices[entry]];
                }
            }
          for (const auto row : owned)
            {
              const double value =
                static_cast<double>(sums[owned.index_within_set(row)]);
              const double inverse =
                std::isfinite(value) && value > 0. ? 1. / value : 0.;
              valid = valid && std::isfinite(value) && value > 0. &&
                      std::isfinite(inverse) && inverse > 0.;
              output[row] = inverse;
            }
        }
      AssertThrow(
        dealii::Utilities::MPI::min(valid ? 1u : 0u,
                                    composition_.communicator()) == 1,
        dealii::ExcMessage(
          "Steady Jacobian row scales must be finite and positive; zero rows are unsupported."));
      scaling.compress(dealii::VectorOperation::insert);
#  else
      (void)blocks;
      (void)characteristics;
      (void)scaling;
      AssertThrow(false,
                  dealii::ExcMessage(
                    "Steady scaling requires the Trilinos backend."));
#  endif
    }

    std::vector<std::pair<FieldId, std::function<void(FieldVectorType &)>>>
                         local_tolerances_;
    std::vector<FieldId> unramped_fields_;
    GlobalVectorType     absolute_tolerances_;
    IDAToleranceRamp     tolerance_ramp_;
    Statistics           accumulated_statistics_;
    unsigned long jacobian_setups_ = 0, linear_solves_ = 0, restarts_ = 0,
                  weight_evaluations_    = 0;
    bool                ramp_stop_armed_ = false, ramp_stop_released_ = false;
    std::optional<int>  mumps_workspace_relaxation_;
    double              mumps_blr_tolerance_      = 1.e-5;
    unsigned int        mumps_blr_memory_limit_   = 0;
    int                 mumps_memory_limit_mb_    = 0;
    bool                mumps_blr_output_details_ = false;
    bool                mumps_output_details_     = false;
    bool                stop_at_output_times_     = false;
    LinearSolver        linear_solver_            = LinearSolver::iterative;
    BlockPreconditioner block_preconditioner_ = BlockPreconditioner::automatic;
    DirectFactorization direct_solver_;
    std::shared_ptr<const typename Composition::SecondOrderCondensation>
                                                            condensation_;
    std::vector<std::shared_ptr<const DirectFactorization>> kinematic_solvers_;
    unsigned long kinematic_factorizations_       = 0,
                  kinematic_factorization_reuses_ = 0;
    std::vector<typename Composition::SecondOrderCondensation::CompressedRows>
                                          kinematic_masses_;
    Composition                           composition_;
    LinearSolveFunction                   solve_;
    const TimeIntervalParameters         &time_parameters_;
    const IDAParameters                  &ida_parameters_;
    mutable std::optional<AdditionalData> additional_data_;
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
    bool                                solved_since_residual_      = false;
    std::optional<double>               initial_condition_jacobian_time_;
    unsigned long                       diagnostic_residuals_          = 0;
    double                              diagnostic_time_               = 0.;
    bool                                connected_                     = false;
    bool                                include_schur_primal_coupling_ = false;
    unsigned int                        schur_materialization_limit_   = 0;
  };
} // namespace ImmersX
#endif

#endif // immersx_sundials_ida_adapter_h
