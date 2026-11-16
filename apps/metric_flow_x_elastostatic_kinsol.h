// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_apps_metric_flow_x_elastostatic_kinsol_h
#define immersx_apps_metric_flow_x_elastostatic_kinsol_h

#include <deal.II/base/scope_exit.h>

#include <immersx/compatibility/dealii_9_9/n_vector.templates.h>
#include <immersx/compatibility/dealii_9_9/utilities.h>
#include <kinsol/kinsol.h>
#if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
#  include <kinsol/kinsol_ls.h>
#else
#  include <kinsol/kinsol_direct.h>
#endif
#include <string>
#include <utility>

namespace ImmersX::metric_flow_x_app_detail
{
  // Installed deal.II 9.8 N_Vectors lack the constraint operations required by
  // KINSOL. Use the complete compatibility vectors, retaining native KINSOL and
  // the same public adapter callbacks. The backend owns the borrowed context.
  template <class Adapter, class GlobalVector>
  std::pair<unsigned int, int>
  solve_elastostatic_kinsol(Adapter            &adapter,
                            const double        time,
                            GlobalVector       &candidate,
                            const GlobalVector &zero,
                            GlobalVector       &solution_scaling,
                            GlobalVector       &function_scaling,
                            GlobalVector       &constraints,
                            const unsigned int  max_iterations,
                            const double        function_tolerance,
                            const double        step_tolerance)
  {
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
      Adapter            &adapter;
      double              time;
      const GlobalVector &zero;
      std::exception_ptr  pending;
    } callbacks{adapter, time, zero, nullptr};
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
        };
        return Native::Utilities::call_and_possibly_capture_exception(
          function,
          c.pending,
          *Native::internal::unwrap_nvector_const<GlobalVector>(state),
          *Native::internal::unwrap_nvector<GlobalVector>(residual));
      },
      y));
    check(KINSetNumMaxIters(memory, max_iterations));
    check(KINSetFuncNormTol(memory, function_tolerance));
    check(KINSetScaledStepTol(memory, step_tolerance));
    check(KINSetMaxSetupCalls(memory, 1));
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
      return SUNLINEARSOLVER_MATRIX_ITERATIVE;
    };
    linear->ops->free = [](SUNLinearSolver handle) {
      SUNLinSolFreeEmpty(handle);
      return 0;
    };
    linear->ops->solve = [](SUNLinearSolver handle,
                            SUNMatrix,
                            N_Vector x,
                            N_Vector b,
                            realtype tolerance) -> int {
      auto &c = *static_cast<Callbacks *>(handle->content);
      return Native::Utilities::call_and_possibly_capture_exception(
        c.adapter.solver().solve_with_jacobian,
        c.pending,
        *Native::internal::unwrap_nvector_const<GlobalVector>(b),
        *Native::internal::unwrap_nvector<GlobalVector>(x),
        tolerance);
    };
    matrix->ops->getid   = [](SUNMatrix) { return SUNMATRIX_CUSTOM; };
    matrix->ops->destroy = SUNMatFreeEmpty;
    check(KINSetLinearSolver(memory, linear, matrix));
    check(KINSetJacFn(
      memory,
      [](N_Vector state, N_Vector, SUNMatrix, void *user, N_Vector, N_Vector)
        -> int {
        auto      &c     = *static_cast<Callbacks *>(user);
        const auto setup = [&](const auto &src) {
          c.adapter.setup_steady_jacobian(c.time, src, c.zero);
        };
        return Native::Utilities::call_and_possibly_capture_exception(
          setup,
          c.pending,
          *Native::internal::unwrap_nvector_const<GlobalVector>(state));
      }));
    const int status = KINSol(memory, y, KIN_NONE, u_scale, f_scale);
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
    long iterations = 0;
    check(KINGetNumNonlinSolvIters(memory, &iterations));
    return {static_cast<unsigned int>(iterations), status};
  }
} // namespace ImmersX::metric_flow_x_app_detail
#endif
