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
      -> decltype(std::declval<const Contributor &>()(
        std::declval<Builder &>(),
        std::declval<const std::string &>()))
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

    template <typename Problem, typename... Arguments>
    auto
    add(const Problem     &problem,
        const std::string &prefix,
        const Arguments &...arguments)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "IDAAdapter contributors must be added before reinit or "
                    "solve."));
      const auto first_new_field = layout_.n_fields();
      Builder    builder(layout_, model_, prefix);
      auto       fields = contribute(builder, problem, arguments...);
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

    GlobalVectorType
    make_state()
    {
      GlobalVectorType state;
      reinit(state);
      return state;
    }

    FieldVectorType &
    field(GlobalVectorType &state, const FieldId id)
    {
      finalize();
      return state.block(field_layout_.block(id));
    }

    const FieldVectorType &
    field(const GlobalVectorType &state, const FieldId id) const
    {
      return state.block(field_layout_.block(id));
    }

    const Operator &
    current_jacobian() const
    {
      AssertThrow(current_jacobian_.has_value(),
                  dealii::ExcMessage("IDA has not prepared a Jacobian."));
      return *current_jacobian_;
    }

  private:
    /** Own callback state captured by state-dependent operator factories. */
    struct JacobianSnapshot
    {
      JacobianSnapshot(
        const StateLayout                                         &layout,
        const BlockFieldLayout<FieldVectorType, GlobalVectorType> &field_layout,
        const double                                               time,
        const GlobalVectorType                                    &state,
        const GlobalVectorType &state_derivative)
        : state_storage(state)
        , derivative_storage(state_derivative)
        , state_view(layout, time)
        , derivative_view(layout, time)
        , context(time, state_view, &derivative_view)
      {
        field_layout.bind_state(state_view, state_storage);
        field_layout.bind_state(derivative_view, derivative_storage);
      }

      GlobalVectorType                   state_storage;
      GlobalVectorType                   derivative_storage;
      StateView<FieldVectorType>         state_view;
      StateView<FieldVectorType>         derivative_view;
      EvaluationContext<FieldVectorType> context;
    };

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
      const auto snapshot = std::make_shared<JacobianSnapshot>(
        layout_, field_layout_, time, state, state_dot);

      using FieldOperator                       = typename Model::Operator;
      const unsigned int                      n = field_layout_.n_blocks();
      std::vector<std::vector<FieldOperator>> blocks(
        n, std::vector<FieldOperator>(n));
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          {
            const auto row    = field_layout_.field(i);
            const auto column = field_layout_.field(j);
            blocks[i][j] =
              model_.state_operator(row, column, snapshot->context);
            blocks[i][j] +=
              alpha *
              model_.derivative_operator(row, column, snapshot->context);
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
      current_jacobian_->vmult = [blocks,
                                  snapshot](GlobalVectorType       &destination,
                                            const GlobalVectorType &source) {
        destination = 0.;
        for (unsigned int i = 0; i < blocks.size(); ++i)
          for (unsigned int j = 0; j < blocks[i].size(); ++j)
            blocks[i][j].vmult_add(destination.block(i), source.block(j));
      };
      current_jacobian_->vmult_add =
        [blocks, snapshot](GlobalVectorType       &destination,
                           const GlobalVectorType &source) {
          for (unsigned int i = 0; i < blocks.size(); ++i)
            for (unsigned int j = 0; j < blocks[i].size(); ++j)
              blocks[i][j].vmult_add(destination.block(i), source.block(j));
        };
      current_jacobian_->Tvmult = [blocks,
                                   snapshot](GlobalVectorType &destination,
                                             const GlobalVectorType &source) {
        destination = 0.;
        for (unsigned int i = 0; i < blocks.size(); ++i)
          for (unsigned int j = 0; j < blocks[i].size(); ++j)
            blocks[i][j].Tvmult_add(destination.block(j), source.block(i));
      };
      current_jacobian_->Tvmult_add =
        [blocks, snapshot](GlobalVectorType       &destination,
                           const GlobalVectorType &source) {
          for (unsigned int i = 0; i < blocks.size(); ++i)
            for (unsigned int j = 0; j < blocks[i].size(); ++j)
              blocks[i][j].Tvmult_add(destination.block(j), source.block(i));
        };
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
} // namespace ImmersX
#endif

#endif // immersx_sundials_ida_adapter_h
