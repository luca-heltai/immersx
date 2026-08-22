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

#include <immersx/core/native_field_layout.h>
#include <immersx/core/time_residual.h>

#include <functional>
#include <memory>
#include <optional>
#include <utility>

#ifdef DEAL_II_WITH_SUNDIALS
#  include <deal.II/sundials/ida.h>
#endif


#ifdef DEAL_II_WITH_SUNDIALS
namespace ImmersX
{
  /**
   * Thin adapter from a semantic residual model to deal.II's IDA wrapper.
   *
   * The adapter distinguishes the vector used by one semantic field from the
   * global vector owned by IDA. With the production instantiation
   *
   *   SundialsIDAResidualAdapter<LA::MPI::Vector, LA::MPI::BlockVector>
   *
   * each IDA block is bound directly as one semantic field. No Problem or
   * Interaction-specific logic is present here. The model, field layout, and
   * adapter itself must outlive the connected IDA object and any current
   * Jacobian action. The BlockVector field layout also supplies the IDA
   * differential mask directly from each field's semantic TimeRole.
   */
  template <typename FieldVectorType, typename GlobalVectorType>
  class SundialsIDAResidualAdapter
  {
  public:
    using Model = ImmersX::SemiDiscreteModel<FieldVectorType>;
    using FieldLayout =
      ImmersX::BlockFieldLayout<FieldVectorType, GlobalVectorType>;
    using Action              = ImmersX::JacobianAction<GlobalVectorType>;
    using ReinitFunction      = std::function<void(GlobalVectorType &)>;
    using LinearSolveFunction = std::function<void(const Action &,
                                                   const GlobalVectorType &,
                                                   GlobalVectorType &,
                                                   double)>;
    using HistoryQuery =
      typename ImmersX::EvaluationContext<FieldVectorType>::HistoryQuery;

    SundialsIDAResidualAdapter(const Model        &model,
                               const FieldLayout  &field_layout,
                               ReinitFunction      reinit_vector,
                               LinearSolveFunction solve)
      : model_(model)
      , field_layout_(field_layout)
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
    connect(dealii::SUNDIALS::IDA<GlobalVectorType> &ida)
    {
      ida.reinit_vector = reinit_vector_;

      ida.residual = [this](const double            time,
                            const GlobalVectorType &state,
                            const GlobalVectorType &state_derivative,
                            GlobalVectorType       &residual) {
        evaluate_residual(time, state, state_derivative, residual);
      };

      ida.setup_jacobian = [this](const double            time,
                                  const GlobalVectorType &state,
                                  const GlobalVectorType &state_derivative,
                                  const double            alpha) {
        prepare_jacobian(time, state, state_derivative, alpha);
      };

      ida.solve_with_jacobian = [this](const GlobalVectorType &rhs,
                                       GlobalVectorType       &dst,
                                       const double            tolerance) {
        AssertThrow(current_action_.has_value(),
                    dealii::ExcMessage("IDA requested a linear solve before "
                                       "the Jacobian was set up."));
        solve_(*current_action_, rhs, dst, tolerance);
      };

      ida.differential_components = [this]() {
        return field_layout_.differential_components();
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

    /** Inspect the most recently prepared IDA Jacobian action. */
    const Action &
    current_jacobian_action() const
    {
      AssertThrow(current_action_.has_value(),
                  dealii::ExcMessage("No IDA Jacobian action is available."));
      return *current_action_;
    }

  private:
    void
    evaluate_residual(const double            time,
                      const GlobalVectorType &state,
                      const GlobalVectorType &state_derivative,
                      GlobalVectorType       &residual) const
    {
      ImmersX::StateView<FieldVectorType> state_view(
        field_layout_.state_layout(), time);
      ImmersX::StateView<FieldVectorType> derivative_view(
        field_layout_.state_layout(), time);
      field_layout_.bind_state(state_view, state);
      field_layout_.bind_state(derivative_view, state_derivative);

      const ImmersX::EvaluationContext<FieldVectorType> context(
        time, state_view, &derivative_view, selected_terms_, history_query_);
      residual = 0.;
      ImmersX::ResidualAccumulator<FieldVectorType> accumulator(
        field_layout_.state_layout());
      field_layout_.bind_residual(accumulator, residual);
      model_.evaluate(context, accumulator);
    }

    void
    prepare_jacobian(const double            time,
                     const GlobalVectorType &state,
                     const GlobalVectorType &state_derivative,
                     const double            alpha)
    {
      const Model       *model          = &model_;
      const FieldLayout *field_layout   = &field_layout_;
      const auto         selected_terms = selected_terms_;
      const auto         history_query  = history_query_;

      // IDA may reuse its input vectors after this callback. Keep stable base
      // vectors in the action rather than capturing callback-local
      // StateView/EvaluationContext objects or borrowed IDA storage.
      const auto base_state = std::make_shared<GlobalVectorType>(state);
      const auto base_derivative =
        std::make_shared<GlobalVectorType>(state_derivative);

      current_action_ =
        Action([model,
                field_layout,
                selected_terms,
                history_query,
                time,
                alpha,
                base_state,
                base_derivative](GlobalVectorType       &destination,
                                 const GlobalVectorType &source) {
          ImmersX::StateView<FieldVectorType> state_view(
            field_layout->state_layout(), time);
          ImmersX::StateView<FieldVectorType> derivative_view(
            field_layout->state_layout(), time);
          ImmersX::StateView<FieldVectorType> increment_view(
            field_layout->state_layout(), time);
          field_layout->bind_state(state_view, *base_state);
          field_layout->bind_state(derivative_view, *base_derivative);
          field_layout->bind_state(increment_view, source);

          const ImmersX::EvaluationContext<FieldVectorType> evaluation(
            time, state_view, &derivative_view, selected_terms, history_query);
          const ImmersX::LinearizationContext<FieldVectorType> linearization(
            evaluation, 1., alpha);

          destination = 0.;
          ImmersX::ResidualAccumulator<FieldVectorType> accumulator(
            field_layout->state_layout());
          field_layout->bind_residual(accumulator, destination);
          model->add_jacobian_action(linearization,
                                     increment_view,
                                     accumulator);
        });
    }

    const Model           &model_;
    const FieldLayout     &field_layout_;
    ReinitFunction         reinit_vector_;
    LinearSolveFunction    solve_;
    ImmersX::TermSelection selected_terms_;
    HistoryQuery           history_query_;
    std::optional<Action>  current_action_;
  };
#endif

} // namespace ImmersX

#endif // immersx_sundials_ida_adapter_h
