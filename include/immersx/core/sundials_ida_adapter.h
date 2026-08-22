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

#include <immersx/core/contributor.h>
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

  /**
   * Public execution adapter for a dynamically composed distributed IDA
   * problem. Contributors are added before the first call to reinit/solve;
   * execution blocks and the differential mask are then derived privately
   * from their FieldDescriptors.
   */
  template <typename FieldVectorType, typename GlobalVectorType>
  class IDAAdapter
  {
  public:
    using Model               = SemiDiscreteModel<FieldVectorType>;
    using Builder             = SemidiscreteBuilder<FieldVectorType>;
    using Operator            = dealii::LinearOperator<GlobalVectorType>;
    using LinearSolveFunction = std::function<void(const Operator &,
                                                   const GlobalVectorType &,
                                                   GlobalVectorType &,
                                                   double)>;
    using AdditionalData =
      typename dealii::SUNDIALS::IDA<GlobalVectorType>::AdditionalData;

    IDAAdapter(const AdditionalData &data,
               const MPI_Comm        communicator,
               LinearSolveFunction   solve)
      : communicator_(communicator)
      , solve_(std::move(solve))
      , ida_(data, communicator)
      , field_layout_(layout_)
    {
      AssertThrow(solve_,
                  dealii::ExcMessage("IDAAdapter requires a linear solve "
                                     "callback."));
    }

    template <typename Contributor>
    auto
    add(const Contributor &contributor, const std::string &prefix)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "IDAAdapter contributors must be added before reinit or "
                    "solve."));
      const auto first_new_field = layout_.n_fields();
      Builder    builder(layout_, model_);
      auto       fields = contributor(builder, prefix);
      for (std::size_t i = first_new_field; i < layout_.n_fields(); ++i)
        field_layout_.add_field(FieldId(i));
      return fields;
    }

    /** Reinitialize a global state vector from the semantic partitions. */
    void
    reinit(GlobalVectorType &vector)
    {
      finalize();
      vector.reinit(field_layout_.block_partitions(), communicator_);
      vector = 0.;
    }

    /** Access the internally owned IDA object for optional configuration. */
    dealii::SUNDIALS::IDA<GlobalVectorType> &
    solver()
    {
      finalize();
      return ida_;
    }

    unsigned int
    solve(GlobalVectorType &state, GlobalVectorType &state_dot)
    {
      finalize();
      return ida_.solve_dae(state, state_dot);
    }

    const Operator &
    current_jacobian() const
    {
      AssertThrow(current_jacobian_.has_value(),
                  dealii::ExcMessage("IDA has not prepared a Jacobian."));
      return *current_jacobian_;
    }

    const StateLayout &
    layout() const
    {
      return layout_;
    }

    unsigned int
    execution_block(const FieldId field) const
    {
      return field_layout_.block(field);
    }

  private:
    void
    finalize()
    {
      if (connected_)
        return;

      AssertThrow(layout_.n_fields() > 0,
                  dealii::ExcMessage("IDAAdapter has no semantic fields."));
      ida_.reinit_vector = [this](GlobalVectorType &vector) {
        vector.reinit(field_layout_.block_partitions(), communicator_);
      };
      ida_.residual = [this](const double            time,
                             const GlobalVectorType &state,
                             const GlobalVectorType &state_dot,
                             GlobalVectorType       &residual) {
        evaluate_residual(time, state, state_dot, residual);
      };
      ida_.setup_jacobian = [this](const double            time,
                                   const GlobalVectorType &state,
                                   const GlobalVectorType &state_dot,
                                   const double            alpha) {
        prepare_jacobian(time, state, state_dot, alpha);
      };
      ida_.solve_with_jacobian = [this](const GlobalVectorType &rhs,
                                        GlobalVectorType       &dst,
                                        const double            tolerance) {
        AssertThrow(current_jacobian_.has_value(),
                    dealii::ExcMessage("IDA requested a solve without a "
                                       "current Jacobian."));
        solve_(*current_jacobian_, rhs, dst, tolerance);
      };
      ida_.differential_components = [this]() {
        return field_layout_.differential_components();
      };
      connected_ = true;
    }

    void
    evaluate_residual(const double            time,
                      const GlobalVectorType &state,
                      const GlobalVectorType &state_dot,
                      GlobalVectorType       &residual) const
    {
      StateView<FieldVectorType> state_view(layout_, time);
      StateView<FieldVectorType> derivative_view(layout_, time);
      field_layout_.bind_state(state_view, state);
      field_layout_.bind_state(derivative_view, state_dot);
      const EvaluationContext<FieldVectorType> context(time,
                                                       state_view,
                                                       &derivative_view);
      residual = 0.;
      for (std::size_t i = 0; i < layout_.n_fields(); ++i)
        model_.evaluate_row(FieldId(i),
                            context,
                            residual.block(field_layout_.block(FieldId(i))));
    }

    void
    prepare_jacobian(const double            time,
                     const GlobalVectorType &state,
                     const GlobalVectorType &state_dot,
                     const double            alpha)
    {
      StateView<FieldVectorType> state_view(layout_, time);
      StateView<FieldVectorType> derivative_view(layout_, time);
      field_layout_.bind_state(state_view, state);
      field_layout_.bind_state(derivative_view, state_dot);
      const EvaluationContext<FieldVectorType> context(time,
                                                       state_view,
                                                       &derivative_view);

      using FieldOperator                       = typename Model::Operator;
      const unsigned int                      n = field_layout_.n_blocks();
      std::vector<std::vector<FieldOperator>> blocks(
        n, std::vector<FieldOperator>(n));
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          {
            const auto row    = field_layout_.field(i);
            const auto column = field_layout_.field(j);
            blocks[i][j]      = model_.state_operator(row, column, context);
            blocks[i][j] +=
              alpha * model_.derivative_operator(row, column, context);
          }

      const auto block_layout = &field_layout_;
      current_jacobian_       = Operator();
      current_jacobian_->reinit_range_vector =
        [block_layout, this](GlobalVectorType &vector, const bool) {
          vector.reinit(block_layout->block_partitions(), communicator_);
        };
      current_jacobian_->reinit_domain_vector =
        [block_layout, this](GlobalVectorType &vector, const bool) {
          vector.reinit(block_layout->block_partitions(), communicator_);
        };
      current_jacobian_->vmult = [blocks](GlobalVectorType       &destination,
                                          const GlobalVectorType &source) {
        destination = 0.;
        for (unsigned int i = 0; i < blocks.size(); ++i)
          for (unsigned int j = 0; j < blocks[i].size(); ++j)
            blocks[i][j].vmult_add(destination.block(i), source.block(j));
      };
      current_jacobian_->vmult_add = [blocks](GlobalVectorType &destination,
                                              const GlobalVectorType &source) {
        for (unsigned int i = 0; i < blocks.size(); ++i)
          for (unsigned int j = 0; j < blocks[i].size(); ++j)
            blocks[i][j].vmult_add(destination.block(i), source.block(j));
      };
      current_jacobian_->Tvmult = [](GlobalVectorType &,
                                     const GlobalVectorType &) {
        AssertThrow(false,
                    dealii::ExcMessage("The IDA Jacobian transpose is not "
                                       "implemented."));
      };
      current_jacobian_->Tvmult_add = current_jacobian_->Tvmult;
    }

    MPI_Comm                                            communicator_;
    LinearSolveFunction                                 solve_;
    dealii::SUNDIALS::IDA<GlobalVectorType>             ida_;
    StateLayout                                         layout_;
    Model                                               model_;
    BlockFieldLayout<FieldVectorType, GlobalVectorType> field_layout_;
    std::optional<Operator>                             current_jacobian_;
    bool                                                connected_ = false;
  };
#endif

} // namespace ImmersX

#endif // immersx_sundials_ida_adapter_h
