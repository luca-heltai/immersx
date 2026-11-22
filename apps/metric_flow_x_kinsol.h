// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_apps_metric_flow_x_kinsol_h
#define immersx_apps_metric_flow_x_kinsol_h

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/scope_exit.h>

#include <deal.II/hp/fe_values.h>

#include <immersx/compatibility/dealii_9_9/n_vector.templates.h>
#include <immersx/compatibility/dealii_9_9/utilities.h>
#include <kinsol/kinsol.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
#  include <kinsol/kinsol_ls.h>
#else
#  include <kinsol/kinsol_direct.h>
#endif
#include <optional>
#include <string>
#include <utility>

namespace ImmersX::metric_flow_x_app_detail
{
  inline bool
  kinsol_stage_can_retry(const int status)
  {
    return status == KIN_STEP_LT_STPTOL || status == KIN_MAXITER_REACHED ||
           status == KIN_LINESEARCH_NONCONV ||
           status == KIN_MXNEWT_5X_EXCEEDED || status == KIN_LSOLVE_FAIL
#if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
           || status == KIN_LINESEARCH_BCFAIL
#endif
      ;
  }

  inline bool
  kinsol_mumps_backsolve_failure_can_retry(const int  status,
                                           const bool mumps_failure)
  {
    return mumps_failure && status == KIN_LSOLVE_FAIL;
  }

  // SUNDIALS' default residual-monitor interval is five iterations. Keep the
  // existing one-iteration exact-Newton setting as a compatibility exception;
  // larger setup intervals must be multiples of the residual-monitor interval.
  inline void
  validate_kinsol_maximum_jacobian_setup_interval(const unsigned int interval)
  {
    AssertThrow(interval == 1 || (interval >= 5 && interval % 5 == 0),
                dealii::ExcMessage(
                  "The KINSOL maximum Jacobian setup interval must be 1 or "
                  "a positive multiple of 5."));
  }

  template <class GlobalVector>
  void
  predict_steady_continuation_state(GlobalVector       &prediction,
                                    const GlobalVector &previous_accepted,
                                    const GlobalVector &latest_accepted,
                                    const double        previous_eta,
                                    const double        latest_eta,
                                    const double        target_eta)
  {
    AssertThrow(std::isfinite(previous_eta) && std::isfinite(latest_eta) &&
                  std::isfinite(target_eta) && previous_eta < latest_eta &&
                  latest_eta < target_eta,
                dealii::ExcMessage(
                  "Invalid eta values for steady continuation prediction."));

    auto secant = latest_accepted;
    secant -= previous_accepted;
    prediction = latest_accepted;
    prediction.add((target_eta - latest_eta) / (latest_eta - previous_eta),
                   secant);
  }

  inline double
  next_steady_continuation_increment(const double stage_increment,
                                     const double maximum_increment,
                                     const bool   stage_recovered_after_retry)
  {
    const double growth = stage_recovered_after_retry ? 1. : 1.5;
    return std::min(maximum_increment, stage_increment * growth);
  }

  // Installed deal.II 9.8 N_Vectors lack the constraint operations required by
  // KINSOL. Use the complete compatibility vectors, retaining native KINSOL and
  // the same public adapter callbacks. The backend owns the borrowed context.
  template <class Adapter, class GlobalVector>
  std::pair<unsigned int, int>
  solve_coupled_kinsol(
    Adapter                                  &adapter,
    const double                              time,
    GlobalVector                             &candidate,
    const GlobalVector                       &zero,
    GlobalVector                             &solution_scaling,
    GlobalVector                             &function_scaling,
    GlobalVector                             &constraints,
    const unsigned int                        max_iterations,
    const double                              function_tolerance,
    const double                              step_tolerance,
    const int                                 strategy          = KIN_NONE,
    [[maybe_unused]] const bool print_diagnostics = false,
    const GlobalVector                       *residual_offset   = nullptr,
    const double                              offset_eta        = 1.,
    const MPI_Comm                            communicator = MPI_COMM_WORLD,
    std::function<void(const GlobalVector &)> initial_residual_observer = {},
    const unsigned int maximum_jacobian_setup_interval                  = 1)
  {
    validate_kinsol_maximum_jacobian_setup_interval(
      maximum_jacobian_setup_interval);
    namespace Native     = Compatibility::SUNDIALS;
    auto      &backend   = adapter.solver();
    const auto make_view = [&](auto &vector) {
      return Native::internal::make_nvector_view(vector
#if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
                                                 ,
                                                 backend.get_sun_context()
#endif
      );
    };
    auto y = make_view(candidate), u_scale = make_view(solution_scaling),
         f_scale = make_view(function_scaling), bounds = make_view(constraints);
    struct Callbacks
    {
      Adapter                                  &adapter;
      double                                    time;
      const GlobalVector                       &zero;
      std::exception_ptr                        pending;
      bool                                      initial_factor_available;
      std::optional<typename Adapter::Operator> matrix_view;
      const GlobalVector                       *residual_offset;
      double                                    offset_eta;
      MPI_Comm                                  communicator;
      std::function<void(const GlobalVector &)> initial_residual_observer;
      bool                                      initial_residual_observed;
      int
      agree_status(const int local_status)
      {
        if (residual_offset == nullptr)
          return local_status;
        const int minimum =
          dealii::Utilities::MPI::min(local_status, communicator);
        const int maximum =
          dealii::Utilities::MPI::max(local_status, communicator);
        if (minimum < 0)
          {
            if (local_status >= 0)
              pending = std::make_exception_ptr(std::runtime_error(
                "KINSOL continuation callback failed on another MPI rank."));
            return -1;
          }
        return maximum > 0 ? 1 : 0;
      }
    } callbacks{adapter,
                time,
                zero,
                nullptr,
                true,
                adapter.current_jacobian(),
                residual_offset,
                offset_eta,
                communicator,
                std::move(initial_residual_observer),
                false};
    SUNMatrix       matrix = nullptr;
    SUNLinearSolver linear = nullptr;
#if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
    void *memory = KINCreate(backend.get_sun_context());
#else
    void *memory = KINCreate();
#endif
    dealii::ScopeExit cleanup([&] {
      if (memory)
        KINFree(&memory);
      if (matrix)
        SUNMatFreeEmpty(matrix);
      if (linear)
        SUNLinSolFreeEmpty(linear);
    });
    const auto        check = [](const int status) {
      AssertThrow(status >= 0,
                  dealii::ExcMessage("Native KINSOL returned status " +
                                     std::to_string(status)));
    };
    AssertThrow(memory != nullptr,
                dealii::ExcMessage("Could not allocate KINSOL."));
    check(KINSetUserData(memory, &callbacks));
    check(KINInit(
      memory,
      [](N_Vector state, N_Vector residual, void *user) -> int {
        auto      &c        = *static_cast<Callbacks *>(user);
        const auto function = [&](const auto &src, auto &dst) {
          c.adapter.solver().residual(c.time, src, c.zero, dst);
          if (!c.initial_residual_observed)
            {
              if (c.initial_residual_observer)
                c.initial_residual_observer(dst);
              c.initial_residual_observed = true;
            }
          // Observe the original residual before applying any continuation
          // offset.
          // At eta=1 use the original residual without a zero-vector update.
          if (c.residual_offset != nullptr && c.offset_eta != 1.)
            dst.add(-(1. - c.offset_eta), *c.residual_offset);
        };
        return c.agree_status(
          Native::Utilities::call_and_possibly_capture_exception(
            function,
            c.pending,
            *Native::internal::unwrap_nvector_const<GlobalVector>(state),
            *Native::internal::unwrap_nvector<GlobalVector>(residual)));
      },
      y));
#if !DEAL_II_SUNDIALS_VERSION_GTE(7, 0, 0)
    // SUNDIALS 7 removed KINSetPrintLevel in favor of SUNLogger; the
    // diagnostics flag has no effect there.
    check(KINSetPrintLevel(memory,
                           print_diagnostics &&
                               dealii::Utilities::MPI::this_mpi_process(
                                 communicator) == 0 ?
                             1 :
                             0));
#endif
    check(KINSetNumMaxIters(memory, max_iterations));
    check(KINSetFuncNormTol(memory, function_tolerance));
    check(KINSetScaledStepTol(memory, step_tolerance));
    check(KINSetMaxSetupCalls(
      memory, static_cast<long int>(maximum_jacobian_setup_interval)));
    check(KINSetNoInitSetup(memory, SUNTRUE));
    check(KINSetConstraints(memory, bounds));
#if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
    linear = SUNLinSolNewEmpty(backend.get_sun_context());
    matrix = SUNMatNewEmpty(backend.get_sun_context());
#else
    linear       = SUNLinSolNewEmpty();
    matrix       = SUNMatNewEmpty();
#endif
    AssertThrow(linear != nullptr && matrix != nullptr,
                dealii::ExcMessage(
                  "Could not allocate KINSOL callback handles."));
    linear->content      = &callbacks;
    linear->ops->gettype = [](SUNLinearSolver) {
      return SUNLINEARSOLVER_DIRECT;
    };
    linear->ops->free = [](SUNLinearSolver handle) {
      SUNLinSolFreeEmpty(handle);
      return 0;
    };
    linear->ops->solve = [](SUNLinearSolver handle,
                            SUNMatrix,
                            N_Vector x,
                            N_Vector b,
                            dealii::SUNDIALS::realtype tolerance) -> int {
      auto &c                    = *static_cast<Callbacks *>(handle->content);
      c.initial_factor_available = false;
      const auto solve = [&](const auto &rhs, auto &dst, const double tol) {
        AssertThrow(c.matrix_view.has_value(),
                    dealii::ExcMessage(
                      "KINSOL matrix view is zero/unpopulated."));
        c.adapter.solver().solve_with_jacobian(rhs, dst, tol);
      };
      return c.agree_status(
        Native::Utilities::call_and_possibly_capture_exception(
          solve,
          c.pending,
          *Native::internal::unwrap_nvector_const<GlobalVector>(b),
          *Native::internal::unwrap_nvector<GlobalVector>(x),
          tolerance));
    };
    matrix->content   = &callbacks;
    matrix->ops->zero = [](SUNMatrix handle) -> int {
      static_cast<Callbacks *>(handle->content)->matrix_view.reset();
#if DEAL_II_SUNDIALS_VERSION_GTE(7, 0, 0)
      return SUN_SUCCESS;
#else
      return SUNMAT_SUCCESS;
#endif
    };
    matrix->ops->matvec = [](SUNMatrix handle, N_Vector x, N_Vector y) -> int {
      auto      &c      = *static_cast<Callbacks *>(handle->content);
      const auto action = [&](const auto &src, auto &dst) {
        if (c.matrix_view)
          c.matrix_view->vmult(dst, src);
        else
          dst = 0.;
      };
      return c.agree_status(
        Native::Utilities::call_and_possibly_capture_exception(
          action,
          c.pending,
          *Native::internal::unwrap_nvector_const<GlobalVector>(x),
          *Native::internal::unwrap_nvector<GlobalVector>(y)));
    };
    matrix->ops->getid   = [](SUNMatrix) { return SUNMATRIX_CUSTOM; };
    matrix->ops->destroy = SUNMatFreeEmpty;
    check(KINSetLinearSolver(memory, linear, matrix));
    check(KINSetJacFn(
      memory,
      [](N_Vector state, N_Vector, SUNMatrix, void *user, N_Vector, N_Vector)
        -> int {
        auto &c = *static_cast<Callbacks *>(user);
        // KINSOL may request setup before its first solve despite
        // noInitSetup. J0 was already prepared for scales and is still the
        // correct initial factorization; the configured interval applies to
        // subsequent setup requests.
        if (c.initial_factor_available)
          {
            const auto reuse = [&]() {
              c.matrix_view = c.adapter.current_jacobian();
            };
            return c.agree_status(
              Native::Utilities::call_and_possibly_capture_exception(
                reuse, c.pending));
          }
        const auto setup = [&](const auto &src) {
          c.adapter.setup_steady_jacobian(c.time, src, c.zero);
          c.matrix_view = c.adapter.current_jacobian();
        };
        return c.agree_status(
          Native::Utilities::call_and_possibly_capture_exception(
            setup,
            c.pending,
            *Native::internal::unwrap_nvector_const<GlobalVector>(state)));
      }));
    const int status = KINSol(memory, y, strategy, u_scale, f_scale);
    if (residual_offset == nullptr)
      {
        if (callbacks.pending)
          try
            {
              std::rethrow_exception(std::exchange(callbacks.pending, nullptr));
            }
          catch (const dealii::RecoverableUserCallbackError &)
            {
              if (status != KIN_SUCCESS)
                throw;
            }
        check(status);
      }
    else
      {
        int         pending_kind = 0;
        std::string mumps_backsolve_error;
        if (callbacks.pending)
          try
            {
              std::rethrow_exception(callbacks.pending);
            }
          catch (const ImmersX::detail::MumpsBacksolveResidualError &error)
            {
              pending_kind          = 3;
              mumps_backsolve_error = error.what();
            }
          catch (const dealii::RecoverableUserCallbackError &)
            {
              pending_kind = 1;
            }
          catch (...)
            {
              pending_kind = 2;
            }
        const bool local_mumps_backsolve_failure =
          kinsol_mumps_backsolve_failure_can_retry(status, pending_kind == 3);
        const auto every_rank_mumps_backsolve_failure =
          dealii::Utilities::MPI::min(local_mumps_backsolve_failure ? 1u : 0u,
                                      communicator);
        const bool failed_callback =
          pending_kind == 2 || (pending_kind == 1 && status != KIN_SUCCESS) ||
          (pending_kind == 3 && every_rank_mumps_backsolve_failure == 0);
        const auto any_failed_callback =
          dealii::Utilities::MPI::max(failed_callback ? 1u : 0u, communicator);
        if (any_failed_callback)
          {
            if (callbacks.pending)
              std::rethrow_exception(callbacks.pending);
            AssertThrow(
              false,
              dealii::ExcMessage(
                "KINSOL continuation callback failed on another MPI rank."));
          }
        const int minimum_status =
          dealii::Utilities::MPI::min(status, communicator);
        const int maximum_status =
          dealii::Utilities::MPI::max(status, communicator);
        AssertThrow(
          minimum_status == maximum_status,
          dealii::ExcMessage(
            "KINSOL returned different statuses on different MPI ranks."));
        if (every_rank_mumps_backsolve_failure == 1 &&
            dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
          std::cerr << "KINSOL continuation rejected a trial after "
                    << mumps_backsolve_error << std::endl;
        AssertThrow(status >= 0 || (kinsol_stage_can_retry(status) &&
                                    (status != KIN_LSOLVE_FAIL ||
                                     every_rank_mumps_backsolve_failure == 1)),
                    dealii::ExcMessage("Native KINSOL returned status " +
                                       std::to_string(status)));
      }
    long iterations = 0;
    check(KINGetNumNonlinSolvIters(memory, &iterations));
    return {static_cast<unsigned int>(iterations), status};
  }

  template <class Flow, class FieldVector>
  void
  fill_flow_steady_characteristics(Flow        &flow,
                                   FieldVector &characteristics,
                                   const double velocity,
                                   const double pressure)
  {
    characteristics = pressure;
    auto reference  = flow.make_state();
    flow.interpolate_reference_area(reference);
    for (const auto component :
         {Flow::Component::area, Flow::Component::area_trace})
      for (const auto index : flow.component_dofs(component))
        characteristics[index] = reference[index];
    for (const auto component :
         {Flow::Component::velocity, Flow::Component::velocity_trace})
      for (const auto index : flow.component_dofs(component))
        characteristics[index] = velocity;
  }

  // This gate tests the full original residual, never a condensed RHS or the
  // KINSOL small-step flag. Area minima are diagnostics against the native
  // numerical floor; the acceptance condition remains strict positivity.
  template <class Adapter,
            class GlobalVector,
            class Flow,
            class FieldVector,
            class AreaField>
  void
  check_coupled_steady_state(Adapter            &adapter,
                             const double        time,
                             const GlobalVector &candidate,
                             const GlobalVector &zero,
                             const GlobalVector &function_scaling,
                             const GlobalVector &constraints,
                             const double        tolerance,
                             Flow               &flow,
                             const FieldVector  &flow_state,
                             const AreaField    &area_field)
  {
    using namespace dealii;
    auto residual = adapter.make_state();
    adapter.solver().residual(time, candidate, zero, residual);
    bool   valid      = true;
    double scaled_max = 0.;
    for (const auto index : candidate.locally_owned_elements())
      {
        const double scaled =
          std::abs(residual[index]) * function_scaling[index];
        valid = valid && std::isfinite(candidate[index]) &&
                std::isfinite(scaled) && zero[index] == 0. &&
                (constraints[index] != 2. || candidate[index] > 0.);
        if (std::isfinite(scaled))
          scaled_max = std::max(scaled_max, scaled);
      }
    FieldVector relevant(flow.locally_owned_dofs(),
                         flow.locally_relevant_dofs(),
                         MPI_COMM_WORLD);
    relevant = flow_state;
    relevant.update_ghost_values();
    const hp::MappingCollection<1, 3> mappings(area_field.mapping());
    const hp::QCollection<1>          quadratures(
      QGauss<1>(flow.finite_element().degree + 1));
    hp::FEValues<1, 3> values(mappings,
                              flow.dof_handler().get_fe_collection(),
                              quadratures,
                              update_values);
    double             cell_min  = std::numeric_limits<double>::infinity();
    double             trace_min = cell_min;
    for (const auto index : flow.component_dofs(Flow::Component::area_trace))
      trace_min = std::min(trace_min, static_cast<double>(flow_state[index]));
    for (const auto &cell : flow.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          values.reinit(cell);
          const auto         &fe_values = values.get_present_fe_values();
          std::vector<double> areas(fe_values.n_quadrature_points);
          fe_values[FEValuesExtractors::Scalar(0)].get_function_values(relevant,
                                                                       areas);
          for (const double area : areas)
            {
              valid    = valid && std::isfinite(area) && area > 0.;
              cell_min = std::min(cell_min, area);
            }
        }
    valid      = Utilities::MPI::min(valid ? 1u : 0u, MPI_COMM_WORLD) == 1;
    scaled_max = Utilities::MPI::max(scaled_max, MPI_COMM_WORLD);
    cell_min   = Utilities::MPI::min(cell_min, MPI_COMM_WORLD);
    trace_min  = Utilities::MPI::min(trace_min, MPI_COMM_WORLD);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << std::setprecision(17)
                << "Steady KINSOL post-gate scaled F max: " << scaled_max
                << "; target: " << tolerance
                << "; finite/positive/zero-rate: " << valid
                << "; min cell quadrature A: " << cell_min
                << "; min trace A: " << trace_min << std::endl;
    AssertThrow(
      valid && scaled_max <= tolerance,
      ExcMessage(
        "Steady KINSOL post-gate failed; no accepted state output or checkpoint was written."));
  }
} // namespace ImmersX::metric_flow_x_app_detail
#endif
