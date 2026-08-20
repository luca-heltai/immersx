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
#include <vector>

#include "native_field_layout.h"
#include "time_residual.h"

#ifdef DEAL_II_WITH_SUNDIALS
#  include <deal.II/sundials/ida.h>
#endif


#ifdef DEAL_II_WITH_SUNDIALS
/**
 * Thin adapter from a semantic residual model to deal.II's IDA wrapper.
 *
 * IDA owns the monolithic vectors. The adapter extracts their registered
 * fields, evaluates the model through StateAccessor and ResidualAccumulator,
 * and scatters the result back. No Problem or Interaction-specific logic is
 * present here. The model, field layout, metadata, and adapter itself must
 * outlive the connected IDA object and any current Jacobian action.
 */
template <typename VectorType>
class SundialsIDAResidualAdapter
{
public:
  using Model          = ImmersX::SemiDiscreteModel<VectorType>;
  using FieldLayout    = ImmersX::detail::MonolithicFieldLayout<VectorType>;
  using Action         = ImmersX::JacobianAction<VectorType>;
  using ReinitFunction = std::function<void(VectorType &)>;
  using LinearSolveFunction = std::function<
    void(const Action &, const VectorType &, VectorType &, double)>;
  using HistoryQuery =
    typename ImmersX::EvaluationContext<VectorType>::HistoryQuery;

  SundialsIDAResidualAdapter(
    const Model                                  &model,
    const FieldLayout                            &field_layout,
    const ImmersX::DifferentialAlgebraicMetadata &metadata,
    ReinitFunction                                reinit_vector,
    LinearSolveFunction                           solve)
    : model_(model)
    , field_layout_(field_layout)
    , metadata_(metadata)
    , reinit_vector_(std::move(reinit_vector))
    , solve_(std::move(solve))
  {
    AssertThrow(reinit_vector_,
                dealii::ExcMessage("IDA adapter requires a vector reinit "
                                   "callback."));
    AssertThrow(solve_,
                dealii::ExcMessage("IDA adapter requires a linear solve "
                                   "callback."));
  }

  /** Attach all required callbacks to an existing deal.II IDA object. */
  void
  connect(dealii::SUNDIALS::IDA<VectorType> &ida)
  {
    ida.reinit_vector = reinit_vector_;

    ida.residual = [this](const double      time,
                          const VectorType &state,
                          const VectorType &state_derivative,
                          VectorType       &residual) {
      evaluate_residual(time, state, state_derivative, residual);
    };

    ida.setup_jacobian = [this](const double      time,
                                const VectorType &state,
                                const VectorType &state_derivative,
                                const double      alpha) {
      prepare_jacobian(time, state, state_derivative, alpha);
    };

    ida.solve_with_jacobian =
      [this](const VectorType &rhs, VectorType &dst, const double tolerance) {
        AssertThrow(current_action_.has_value(),
                    dealii::ExcMessage("IDA requested a linear solve before "
                                       "the Jacobian was set up."));
        solve_(*current_action_, rhs, dst, tolerance);
      };

    ida.differential_components = [this]() {
      return metadata_.differential_components();
    };
  }

  /** Select terms for subsequent callback evaluations. */
  void
  set_term_selection(const ImmersX::TermSelection &selection)
  {
    selected_terms_ = selection;
  }

  /** Supply history-backed values to the shared evaluation context. */
  void
  set_history_query(HistoryQuery query)
  {
    history_query_ = std::move(query);
  }

  /** Inspect the most recently prepared monolithic Jacobian action. */
  const Action &
  current_jacobian_action() const
  {
    AssertThrow(current_action_.has_value(),
                dealii::ExcMessage("No IDA Jacobian action is available."));
    return *current_action_;
  }

private:
  void
  evaluate_residual(const double      time,
                    const VectorType &state,
                    const VectorType &state_derivative,
                    VectorType       &residual) const
  {
    std::vector<VectorType> state_fields;
    std::vector<VectorType> derivative_fields;
    field_layout_.extract(state, state_fields);
    field_layout_.extract(state_derivative, derivative_fields);

    ImmersX::StateView<VectorType> state_view(field_layout_.state_layout(),
                                              time);
    ImmersX::StateView<VectorType> derivative_view(field_layout_.state_layout(),
                                                   time);
    field_layout_.bind(state_view, state_fields);
    field_layout_.bind(derivative_view, derivative_fields);

    const ImmersX::EvaluationContext<VectorType> context(
      time, state_view, &derivative_view, selected_terms_, history_query_);

    std::vector<VectorType> residual_fields;
    residual_fields.assign(field_layout_.state_layout().n_fields(),
                           VectorType());
    ImmersX::ResidualAccumulator<VectorType> accumulator(
      field_layout_.state_layout());
    for (std::size_t field = 0; field < field_layout_.state_layout().n_fields();
         ++field)
      if (field_layout_.has_field(ImmersX::FieldId(field)))
        {
          residual_fields[field].reinit(state_fields[field]);
          residual_fields[field] = 0.;
          accumulator.bind(ImmersX::FieldId(field), residual_fields[field]);
        }

    model_.evaluate(context, accumulator);
    residual = 0.;
    field_layout_.scatter_add(residual_fields, residual);
  }

  void
  prepare_jacobian(const double      time,
                   const VectorType &state,
                   const VectorType &state_derivative,
                   const double      alpha)
  {
    std::vector<VectorType> state_fields;
    std::vector<VectorType> derivative_fields;
    field_layout_.extract(state, state_fields);
    field_layout_.extract(state_derivative, derivative_fields);

    const Model       *model          = &model_;
    const FieldLayout *field_layout   = &field_layout_;
    const auto         selected_terms = selected_terms_;
    const auto         history_query  = history_query_;

    current_action_ = Action([model,
                              field_layout,
                              selected_terms,
                              history_query,
                              time,
                              alpha,
                              state_fields      = std::move(state_fields),
                              derivative_fields = std::move(
                                derivative_fields)](VectorType &destination,
                                                    const VectorType &source) {
      std::vector<VectorType> increment_fields;
      field_layout->extract(source, increment_fields);

      ImmersX::StateView<VectorType> state_view(field_layout->state_layout(),
                                                time);
      ImmersX::StateView<VectorType> derivative_view(
        field_layout->state_layout(), time);
      ImmersX::StateView<VectorType> increment_view(
        field_layout->state_layout(), time);
      field_layout->bind(state_view, state_fields);
      field_layout->bind(derivative_view, derivative_fields);
      field_layout->bind(increment_view, increment_fields);

      const ImmersX::EvaluationContext<VectorType> evaluation(
        time, state_view, &derivative_view, selected_terms, history_query);
      const ImmersX::LinearizationContext<VectorType> linearization(evaluation,
                                                                    1.,
                                                                    alpha);

      std::vector<VectorType> residual_fields;
      residual_fields.assign(field_layout->state_layout().n_fields(),
                             VectorType());
      ImmersX::ResidualAccumulator<VectorType> accumulator(
        field_layout->state_layout());
      for (std::size_t field = 0;
           field < field_layout->state_layout().n_fields();
           ++field)
        if (field_layout->has_field(ImmersX::FieldId(field)))
          {
            residual_fields[field].reinit(state_fields[field]);
            residual_fields[field] = 0.;
            accumulator.bind(ImmersX::FieldId(field), residual_fields[field]);
          }

      model->add_jacobian_action(linearization, increment_view, accumulator);
      destination = 0.;
      field_layout->scatter_add(residual_fields, destination);
    });
  }

  const Model                                  &model_;
  const FieldLayout                            &field_layout_;
  const ImmersX::DifferentialAlgebraicMetadata &metadata_;
  ReinitFunction                                reinit_vector_;
  LinearSolveFunction                           solve_;
  ImmersX::TermSelection                        selected_terms_;
  HistoryQuery                                  history_query_;
  std::optional<Action>                         current_action_;
};
#endif

#endif // immersx_sundials_ida_adapter_h
