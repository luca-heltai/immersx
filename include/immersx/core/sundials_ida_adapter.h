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
#include <type_traits>
#include <utility>
#include <vector>

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
   * global vector owned by IDA.  With the production instantiation
   *
   *   SundialsIDAResidualAdapter<LA::MPI::Vector, LA::MPI::BlockVector>
   *
   * each IDA block is bound directly as one semantic field.  The one-type
   * default remains for the small serial prototype tests and uses the legacy
   * scalar-index adapter below; real distributed Problems use the block path.
   * No Problem or Interaction-specific logic is present here. The model, field
   * layout, metadata, and adapter itself must outlive the connected IDA object
   * and any current Jacobian action.
   */
  template <typename FieldVectorType,
            typename GlobalVectorType = FieldVectorType>
  class SundialsIDAResidualAdapter
  {
  public:
    using Model       = ImmersX::SemiDiscreteModel<FieldVectorType>;
    using FieldLayout = std::conditional_t<
      std::is_same_v<FieldVectorType, GlobalVectorType>,
      ImmersX::detail::MonolithicFieldLayout<FieldVectorType>,
      ImmersX::BlockFieldLayout<FieldVectorType, GlobalVectorType>>;
    using Action              = ImmersX::JacobianAction<GlobalVectorType>;
    using ReinitFunction      = std::function<void(GlobalVectorType &)>;
    using LinearSolveFunction = std::function<void(const Action &,
                                                   const GlobalVectorType &,
                                                   GlobalVectorType &,
                                                   double)>;
    using HistoryQuery =
      typename ImmersX::EvaluationContext<FieldVectorType>::HistoryQuery;

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
    evaluate_residual(const double            time,
                      const GlobalVectorType &state,
                      const GlobalVectorType &state_derivative,
                      GlobalVectorType       &residual) const
    {
      if constexpr (std::is_same_v<FieldVectorType, GlobalVectorType>)
        {
          std::vector<FieldVectorType> state_fields;
          std::vector<FieldVectorType> derivative_fields;
          field_layout_.extract(state, state_fields);
          field_layout_.extract(state_derivative, derivative_fields);

          ImmersX::StateView<FieldVectorType> state_view(
            field_layout_.state_layout(), time);
          ImmersX::StateView<FieldVectorType> derivative_view(
            field_layout_.state_layout(), time);
          field_layout_.bind(state_view, state_fields);
          field_layout_.bind(derivative_view, derivative_fields);

          const ImmersX::EvaluationContext<FieldVectorType> context(
            time,
            state_view,
            &derivative_view,
            selected_terms_,
            history_query_);

          std::vector<FieldVectorType> residual_fields;
          residual_fields.assign(field_layout_.state_layout().n_fields(),
                                 FieldVectorType());
          ImmersX::ResidualAccumulator<FieldVectorType> accumulator(
            field_layout_.state_layout());
          for (std::size_t field = 0;
               field < field_layout_.state_layout().n_fields();
               ++field)
            if (field_layout_.has_field(ImmersX::FieldId(field)))
              {
                residual_fields[field].reinit(state_fields[field]);
                residual_fields[field] = 0.;
                accumulator.bind(ImmersX::FieldId(field),
                                 residual_fields[field]);
              }

          model_.evaluate(context, accumulator);
          residual = 0.;
          field_layout_.scatter_add(residual_fields, residual);
        }
      else
        {
          ImmersX::StateView<FieldVectorType> state_view(
            field_layout_.state_layout(), time);
          ImmersX::StateView<FieldVectorType> derivative_view(
            field_layout_.state_layout(), time);
          field_layout_.bind_state(state_view, state);
          field_layout_.bind_state(derivative_view, state_derivative);

          const ImmersX::EvaluationContext<FieldVectorType> context(
            time,
            state_view,
            &derivative_view,
            selected_terms_,
            history_query_);
          residual = 0.;
          ImmersX::ResidualAccumulator<FieldVectorType> accumulator(
            field_layout_.state_layout());
          field_layout_.bind_residual(accumulator, residual);
          model_.evaluate(context, accumulator);
        }
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

      if constexpr (std::is_same_v<FieldVectorType, GlobalVectorType>)
        {
          std::vector<FieldVectorType> state_fields;
          std::vector<FieldVectorType> derivative_fields;
          field_layout_.extract(state, state_fields);
          field_layout_.extract(state_derivative, derivative_fields);

          current_action_ =
            Action([model,
                    field_layout,
                    selected_terms,
                    history_query,
                    time,
                    alpha,
                    state_fields      = std::move(state_fields),
                    derivative_fields = std::move(
                      derivative_fields)](GlobalVectorType       &destination,
                                          const GlobalVectorType &source) {
              std::vector<FieldVectorType> increment_fields;
              field_layout->extract(source, increment_fields);

              ImmersX::StateView<FieldVectorType> state_view(
                field_layout->state_layout(), time);
              ImmersX::StateView<FieldVectorType> derivative_view(
                field_layout->state_layout(), time);
              ImmersX::StateView<FieldVectorType> increment_view(
                field_layout->state_layout(), time);
              field_layout->bind(state_view, state_fields);
              field_layout->bind(derivative_view, derivative_fields);
              field_layout->bind(increment_view, increment_fields);

              const ImmersX::EvaluationContext<FieldVectorType> evaluation(
                time,
                state_view,
                &derivative_view,
                selected_terms,
                history_query);
              const ImmersX::LinearizationContext<FieldVectorType>
                linearization(evaluation, 1., alpha);

              std::vector<FieldVectorType> residual_fields;
              residual_fields.assign(field_layout->state_layout().n_fields(),
                                     FieldVectorType());
              ImmersX::ResidualAccumulator<FieldVectorType> accumulator(
                field_layout->state_layout());
              for (std::size_t field = 0;
                   field < field_layout->state_layout().n_fields();
                   ++field)
                if (field_layout->has_field(ImmersX::FieldId(field)))
                  {
                    residual_fields[field].reinit(state_fields[field]);
                    residual_fields[field] = 0.;
                    accumulator.bind(ImmersX::FieldId(field),
                                     residual_fields[field]);
                  }

              model->add_jacobian_action(linearization,
                                         increment_view,
                                         accumulator);
              destination = 0.;
              field_layout->scatter_add(residual_fields, destination);
            });
        }
      else
        {
          // IDA may reuse its input vectors after this callback.  Keep stable
          // base vectors in the action rather than capturing callback-local
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
                time,
                state_view,
                &derivative_view,
                selected_terms,
                history_query);
              const ImmersX::LinearizationContext<FieldVectorType>
                linearization(evaluation, 1., alpha);

              destination = 0.;
              ImmersX::ResidualAccumulator<FieldVectorType> accumulator(
                field_layout->state_layout());
              field_layout->bind_residual(accumulator, destination);
              model->add_jacobian_action(linearization,
                                         increment_view,
                                         accumulator);
            });
        }
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

} // namespace ImmersX

#endif // immersx_sundials_ida_adapter_h
