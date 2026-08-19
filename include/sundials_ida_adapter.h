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

#include <functional>
#include <optional>
#include <utility>

#include "time_residual.h"

#ifdef DEAL_II_WITH_SUNDIALS
#  include <deal.II/sundials/ida.h>
#endif


#ifdef DEAL_II_WITH_SUNDIALS
/**
 * Thin adapter from a term-wise ImmersX residual model to deal.II's IDA
 * wrapper.
 *
 * The adapter owns no physics and no linear algebra.  It only translates the
 * IDA callback contracts into a TimeResidualContext and forwards the cached
 * Jacobian action to a user-supplied linear solve policy.
 */
template <typename VectorType>
class SundialsIDAResidualAdapter
{
public:
  using Model               = TimeResidualModel<VectorType>;
  using Action              = JacobianAction<VectorType>;
  using ReinitFunction      = std::function<void(VectorType &)>;
  using LinearSolveFunction = std::function<
    void(const Action &, const VectorType &, VectorType &, double)>;

  SundialsIDAResidualAdapter(const Model                         &model,
                             const DifferentialAlgebraicMetadata &metadata,
                             ReinitFunction                       reinit_vector,
                             LinearSolveFunction                  solve)
    : model(model)
    , metadata(metadata)
    , reinit_vector(std::move(reinit_vector))
    , solve(std::move(solve))
  {
    AssertThrow(this->reinit_vector,
                dealii::ExcMessage("IDA adapter requires a vector reinit "
                                   "callback."));
    AssertThrow(this->solve,
                dealii::ExcMessage("IDA adapter requires a linear solve "
                                   "callback."));
  }

  /** Attach all required callbacks to an existing deal.II IDA object. */
  void
  connect(dealii::SUNDIALS::IDA<VectorType> &ida)
  {
    ida.reinit_vector = reinit_vector;

    ida.residual = [this](const double      time,
                          const VectorType &state,
                          const VectorType &state_derivative,
                          VectorType       &residual) {
      auto context = make_context(time, state, state_derivative);
      model.residual(context, residual);
    };

    ida.setup_jacobian = [this](const double      time,
                                const VectorType &state,
                                const VectorType &state_derivative,
                                const double      alpha) {
      auto context              = make_context(time, state, state_derivative);
      context.state_weight      = 1.;
      context.derivative_weight = alpha;
      current_action            = model.jacobian_action(context);
    };

    ida.solve_with_jacobian =
      [this](const VectorType &rhs, VectorType &dst, const double tolerance) {
        AssertThrow(current_action.has_value(),
                    dealii::ExcMessage("IDA requested a linear solve before "
                                       "the Jacobian was set up."));
        solve(*current_action, rhs, dst, tolerance);
      };

    ida.differential_components = [this]() {
      return metadata.differential_components();
    };
  }

  /** Select terms for subsequent callback evaluations. */
  void
  set_term_selection(const TermSelection &selection)
  {
    selected_terms = selection;
  }

  /** Inspect the most recently prepared Jacobian action in a test or driver. */
  const Action &
  current_jacobian_action() const
  {
    AssertThrow(current_action.has_value(),
                dealii::ExcMessage("No IDA Jacobian action is available."));
    return *current_action;
  }

private:
  TimeResidualContext<VectorType>
  make_context(const double      time,
               const VectorType &state,
               const VectorType &state_derivative) const
  {
    TimeResidualContext<VectorType> context(time, state, state_derivative);
    context.selected_terms = selected_terms;
    return context;
  }

  const Model                         &model;
  const DifferentialAlgebraicMetadata &metadata;
  ReinitFunction                       reinit_vector;
  LinearSolveFunction                  solve;
  TermSelection                        selected_terms;
  std::optional<Action>                current_action;
};
#endif

#endif // immersx_sundials_ida_adapter_h
