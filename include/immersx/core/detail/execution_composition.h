// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_detail_execution_composition_h
#define immersx_detail_execution_composition_h

#include <deal.II/base/index_set.h>

#include <immersx/core/contributor.h>

#include <memory>
#include <optional>
#include <vector>

namespace ImmersX::detail
{
  /** Internal semantic-field to distributed-block mapping. */
  template <typename FieldVectorType, typename GlobalBlockVectorType>
  class BlockFieldLayout
  {
  public:
    explicit BlockFieldLayout(const StateLayout &layout)
      : layout_(layout)
      , blocks_by_field_(layout.n_fields())
    {}

    void
    add_field(const FieldId field)
    {
      AssertThrow(layout_.contains(field),
                  dealii::ExcMessage("Field is not in the state layout."));
      const auto block = static_cast<unsigned int>(fields_by_block_.size());
      if (blocks_by_field_.size() < layout_.n_fields())
        blocks_by_field_.resize(layout_.n_fields());
      AssertThrow(!blocks_by_field_[field.value()].has_value(),
                  dealii::ExcMessage("Field was registered twice."));
      blocks_by_field_[field.value()] = block;
      fields_by_block_.push_back(field);
      block_sizes_.push_back(layout_.field(field).locally_owned.size());
      block_owned_.push_back(layout_.field(field).locally_owned);
    }

    unsigned int
    block(const FieldId field) const
    {
      validate_field(field);
      AssertThrow(blocks_by_field_[field.value()].has_value(),
                  dealii::ExcMessage("Field has no execution block."));
      return *blocks_by_field_[field.value()];
    }

    FieldId
    field(const unsigned int block_number) const
    {
      AssertIndexRange(block_number, fields_by_block_.size());
      return fields_by_block_[block_number];
    }

    unsigned int
    n_blocks() const
    {
      return static_cast<unsigned int>(fields_by_block_.size());
    }

    std::vector<dealii::IndexSet>
    block_partitions() const
    {
      validate_complete();
      return block_owned_;
    }

    void
    bind_state(StateView<FieldVectorType>  &view,
               const GlobalBlockVectorType &global) const
    {
      validate_complete();
      AssertThrow(global.n_blocks() == fields_by_block_.size(),
                  dealii::ExcMessage(
                    "Global block vector does not match its field layout."));
      for (unsigned int block_number = 0;
           block_number < fields_by_block_.size();
           ++block_number)
        view.bind(fields_by_block_[block_number], global.block(block_number));
    }

    dealii::IndexSet
    differential_components() const
    {
      validate_complete();
      std::size_t total_size = 0;
      for (const auto size : block_sizes_)
        total_size += size;

      dealii::IndexSet result(total_size);
      std::size_t      offset = 0;
      for (unsigned int block_number = 0;
           block_number < fields_by_block_.size();
           ++block_number)
        {
          if (layout_.field(fields_by_block_[block_number]).time_role ==
              TimeRole::differential)
            for (const auto index : block_owned_[block_number])
              result.add_index(offset + index);
          offset += block_sizes_[block_number];
        }
      result.compress();
      return result;
    }

  private:
    void
    validate_field(const FieldId field) const
    {
      AssertThrow(layout_.contains(field),
                  dealii::ExcMessage("Field is not in the state layout."));
    }

    void
    validate_complete() const
    {
      AssertThrow(fields_by_block_.size() == layout_.n_fields(),
                  dealii::ExcMessage(
                    "Every semantic field must have an execution block."));
    }

    const StateLayout                       &layout_;
    std::vector<std::optional<unsigned int>> blocks_by_field_;
    std::vector<FieldId>                     fields_by_block_;
    std::vector<std::size_t>                 block_sizes_;
    std::vector<dealii::IndexSet>            block_owned_;
  };

  /**
   * Private shared execution composition used by steady and IDA adapters.
   * Applications only interact with the adapters and semantic FieldIds.
   */
  template <typename FieldVectorType, typename GlobalVectorType>
  class ExecutionComposition
  {
  public:
    using Model    = SemiDiscreteModel<FieldVectorType>;
    using Builder  = SemidiscreteBuilder<FieldVectorType>;
    using Operator = dealii::LinearOperator<GlobalVectorType>;

    explicit ExecutionComposition(const MPI_Comm communicator)
      : communicator_(communicator)
      , field_layout_(layout_)
    {}

    template <typename Problem, typename... Arguments>
    auto
    add(const Problem     &problem,
        const std::string &prefix,
        const Arguments &...arguments)
    {
      AssertThrow(!finalized_,
                  dealii::ExcMessage(
                    "Contributors must be added before execution starts."));
      const auto first_new_field = layout_.n_fields();
      Builder    builder(layout_, model_, prefix);
      auto       fields = contribute(builder, problem, arguments...);
      for (std::size_t i = first_new_field; i < layout_.n_fields(); ++i)
        field_layout_.add_field(FieldId(i));
      return fields;
    }

    GlobalVectorType
    make_state() const
    {
      GlobalVectorType state;
      reinit(state);
      return state;
    }

    void
    reinit(GlobalVectorType &state) const
    {
      finalize();
      state.reinit(field_layout_.block_partitions(), communicator_);
      state = 0.;
    }

    FieldVectorType &
    field(GlobalVectorType &state, const FieldId id) const
    {
      validate_state(state);
      return state.block(field_layout_.block(id));
    }

    const FieldVectorType &
    field(const GlobalVectorType &state, const FieldId id) const
    {
      validate_state(state);
      return state.block(field_layout_.block(id));
    }

    void
    evaluate_residual(const double            time,
                      const GlobalVectorType &state,
                      const GlobalVectorType *state_dot,
                      GlobalVectorType       &residual,
                      const TermSelection    &terms = {}) const
    {
      finalize();
      validate_state(state);
      if (state_dot != nullptr)
        validate_state(*state_dot);

      StateView<FieldVectorType> state_view(layout_, time);
      field_layout_.bind_state(state_view, state);
      StateView<FieldVectorType>                derivative_view(layout_, time);
      const EvaluationContext<FieldVectorType> *context = nullptr;
      std::optional<EvaluationContext<FieldVectorType>> derivative_context;
      if (state_dot != nullptr)
        {
          field_layout_.bind_state(derivative_view, *state_dot);
          derivative_context.emplace(time, state_view, &derivative_view, terms);
          context = &*derivative_context;
        }
      else
        {
          derivative_context.emplace(time, state_view, nullptr, terms);
          context = &*derivative_context;
        }

      residual.reinit(field_layout_.block_partitions(), communicator_);
      residual = 0.;
      for (std::size_t i = 0; i < layout_.n_fields(); ++i)
        model_.evaluate_row(FieldId(i),
                            *context,
                            residual.block(field_layout_.block(FieldId(i))));
    }

    Operator
    jacobian(const double            time,
             const GlobalVectorType &state,
             const GlobalVectorType *state_dot,
             const double            alpha,
             const TermSelection    &terms = {}) const
    {
      finalize();
      validate_state(state);
      if (state_dot != nullptr)
        validate_state(*state_dot);

      StateView<FieldVectorType> state_view(layout_, time);
      field_layout_.bind_state(state_view, state);
      StateView<FieldVectorType> derivative_view(layout_, time);
      std::optional<EvaluationContext<FieldVectorType>> context;
      if (state_dot != nullptr)
        {
          field_layout_.bind_state(derivative_view, *state_dot);
          context.emplace(time, state_view, &derivative_view, terms);
        }
      else
        context.emplace(time, state_view, nullptr, terms);
      return make_global_operator(*context, alpha);
    }

    struct Snapshot
    {
      Snapshot(const ExecutionComposition &composition,
               const double                time,
               const GlobalVectorType     &state,
               const GlobalVectorType     &state_dot)
        : state_storage(state)
        , derivative_storage(state_dot)
        , state_view(composition.layout_, time)
        , derivative_view(composition.layout_, time)
        , context(time, state_view, &derivative_view)
      {
        composition.field_layout_.bind_state(state_view, state_storage);
        composition.field_layout_.bind_state(derivative_view,
                                             derivative_storage);
      }

      GlobalVectorType                   state_storage;
      GlobalVectorType                   derivative_storage;
      StateView<FieldVectorType>         state_view;
      StateView<FieldVectorType>         derivative_view;
      EvaluationContext<FieldVectorType> context;
    };

    std::shared_ptr<Snapshot>
    make_snapshot(const double            time,
                  const GlobalVectorType &state,
                  const GlobalVectorType &state_dot) const
    {
      validate_state(state);
      validate_state(state_dot);
      return std::make_shared<Snapshot>(*this, time, state, state_dot);
    }

    Operator
    jacobian(const EvaluationContext<FieldVectorType> &context,
             const double                              alpha) const
    {
      finalize();
      return make_global_operator(context, alpha);
    }

    dealii::IndexSet
    differential_components() const
    {
      finalize();
      return field_layout_.differential_components();
    }

    unsigned int
    n_fields() const
    {
      return layout_.n_fields();
    }

    const Model &
    model() const
    {
      return model_;
    }

    const StateLayout &
    state_layout() const
    {
      return layout_;
    }

    MPI_Comm
    communicator() const
    {
      return communicator_;
    }

  private:
    void
    finalize() const
    {
      finalized_ = true;
      AssertThrow(layout_.n_fields() > 0,
                  dealii::ExcMessage("Execution has no semantic fields."));
    }

    void
    validate_state(const GlobalVectorType &state) const
    {
      finalize();
      AssertThrow(state.n_blocks() == field_layout_.n_blocks(),
                  dealii::ExcMessage(
                    "Global state has the wrong number of semantic blocks."));
    }

    Operator
    make_global_operator(const EvaluationContext<FieldVectorType> &context,
                         const double                              alpha) const
    {
      const unsigned int n = field_layout_.n_blocks();
      std::vector<std::vector<typename Model::Operator>> blocks(
        n, std::vector<typename Model::Operator>(n));
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          blocks[i][j] = zero_operator(context.state(field_layout_.field(i)),
                                       context.state(field_layout_.field(j)));
      for (const auto &[row, column] : model_.active_operator_blocks())
        {
          const auto i = field_layout_.block(row);
          const auto j = field_layout_.block(column);
          blocks[i][j] = model_.state_operator(row, column, context);
          if (alpha != 0.)
            blocks[i][j] +=
              alpha * model_.derivative_operator(row, column, context);
        }

      Operator result;
      result.reinit_range_vector = [this](GlobalVectorType &vector, bool) {
        vector.reinit(field_layout_.block_partitions(), communicator_);
      };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult = [blocks](GlobalVectorType       &destination,
                              const GlobalVectorType &source) {
        destination = 0.;
        for (unsigned int i = 0; i < blocks.size(); ++i)
          for (unsigned int j = 0; j < blocks[i].size(); ++j)
            blocks[i][j].vmult_add(destination.block(i), source.block(j));
      };
      result.vmult_add = [blocks](GlobalVectorType       &destination,
                                  const GlobalVectorType &source) {
        for (unsigned int i = 0; i < blocks.size(); ++i)
          for (unsigned int j = 0; j < blocks[i].size(); ++j)
            blocks[i][j].vmult_add(destination.block(i), source.block(j));
      };
      result.Tvmult = [blocks](GlobalVectorType       &destination,
                               const GlobalVectorType &source) {
        destination = 0.;
        for (unsigned int i = 0; i < blocks.size(); ++i)
          for (unsigned int j = 0; j < blocks[i].size(); ++j)
            blocks[i][j].Tvmult_add(destination.block(j), source.block(i));
      };
      result.Tvmult_add = [blocks](GlobalVectorType       &destination,
                                   const GlobalVectorType &source) {
        for (unsigned int i = 0; i < blocks.size(); ++i)
          for (unsigned int j = 0; j < blocks[i].size(); ++j)
            blocks[i][j].Tvmult_add(destination.block(j), source.block(i));
      };
      return result;
    }

    static typename Model::Operator
    zero_operator(const FieldVectorType &range, const FieldVectorType &domain)
    {
      typename Model::Operator result;
      result.reinit_range_vector = [range](FieldVectorType &vector,
                                           const bool       omit) {
        vector.reinit(range, omit);
      };
      result.reinit_domain_vector = [domain](FieldVectorType &vector,
                                             const bool       omit) {
        vector.reinit(domain, omit);
      };
      result.vmult = [](FieldVectorType &vector, const FieldVectorType &) {
        vector = 0.;
      };
      result.vmult_add = [](FieldVectorType &, const FieldVectorType &) {};
      result.Tvmult    = [](FieldVectorType &vector, const FieldVectorType &) {
        vector = 0.;
      };
      result.Tvmult_add = [](FieldVectorType &, const FieldVectorType &) {};
      return result;
    }

    MPI_Comm                                                    communicator_;
    mutable StateLayout                                         layout_;
    mutable Model                                               model_;
    mutable BlockFieldLayout<FieldVectorType, GlobalVectorType> field_layout_;
    mutable bool finalized_ = false;
  };
} // namespace ImmersX::detail

#endif // immersx_detail_execution_composition_h
