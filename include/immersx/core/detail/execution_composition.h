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

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <immersx/algebra/linear_algebra.h>
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

    std::vector<std::size_t>
    block_offsets() const
    {
      validate_complete();
      std::vector<std::size_t> result(fields_by_block_.size() + 1, 0);
      for (unsigned int block_number = 0;
           block_number < fields_by_block_.size();
           ++block_number)
        result[block_number + 1] =
          result[block_number] + block_sizes_[block_number];
      return result;
    }

    dealii::IndexSet
    monolithic_partition() const
    {
      const auto       offsets = block_offsets();
      dealii::IndexSet result(offsets.back());
      for (unsigned int block_number = 0;
           block_number < fields_by_block_.size();
           ++block_number)
        for (const auto index : block_owned_[block_number])
          result.add_index(offsets[block_number] + index);
      result.compress();
      return result;
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
          for (const auto index : layout_.field(fields_by_block_[block_number])
                                    .differential_components)
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
    using Model           = SemiDiscreteModel<FieldVectorType>;
    using Builder         = SemidiscreteBuilder<FieldVectorType>;
    using Operator        = dealii::LinearOperator<GlobalVectorType>;
    using LocalOperator   = typename Model::Operator;
    using MatrixType      = typename Model::MatrixOperator::Matrix;
    using BlockMatrixType = ImmersXLA::MPI::BlockSparseMatrix;

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

    bool
    can_materialize_matrix(const GlobalVectorType &state,
                           const GlobalVectorType *state_dot = nullptr,
                           const double            alpha     = 0.) const
    {
      finalize();
      validate_state(state);
      if (state_dot != nullptr)
        validate_state(*state_dot);

      StateView<FieldVectorType> state_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      StateView<FieldVectorType> derivative_view(layout_, 0.);
      std::optional<EvaluationContext<FieldVectorType>> context;
      if (state_dot != nullptr)
        {
          field_layout_.bind_state(derivative_view, *state_dot);
          context.emplace(0., state_view, &derivative_view);
        }
      else
        context.emplace(0., state_view, nullptr);
      return can_materialize_matrix(*context, alpha);
    }

    /** Materialize the current linearization as a true block sparse matrix. */
    BlockMatrixType
    block_matrix(const GlobalVectorType &state,
                 const GlobalVectorType *state_dot = nullptr,
                 const double            alpha     = 0.) const
    {
      finalize();
      validate_state(state);
      if (state_dot != nullptr)
        validate_state(*state_dot);

      StateView<FieldVectorType> state_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      StateView<FieldVectorType> derivative_view(layout_, 0.);
      std::optional<EvaluationContext<FieldVectorType>> context;
      if (state_dot != nullptr)
        {
          field_layout_.bind_state(derivative_view, *state_dot);
          context.emplace(0., state_view, &derivative_view);
        }
      else
        context.emplace(0., state_view, nullptr);
      return make_block_matrix(*context, alpha);
    }

    /** Convert a block matrix to its concatenated monolithic sparse matrix. */
    MatrixType
    monolithic_matrix(const BlockMatrixType &block_matrix) const
    {
      finalize();
      AssertThrow(block_matrix.n_block_rows() == field_layout_.n_blocks() &&
                    block_matrix.n_block_cols() == field_layout_.n_blocks(),
                  dealii::ExcMessage(
                    "Block matrix does not match the execution layout."));

      const auto offsets   = field_layout_.block_offsets();
      const auto partition = field_layout_.monolithic_partition();
      dealii::DynamicSparsityPattern sparsity(offsets.back(), offsets.back());
      const auto block_partitions = field_layout_.block_partitions();
      for (unsigned int i = 0; i < field_layout_.n_blocks(); ++i)
        for (unsigned int j = 0; j < field_layout_.n_blocks(); ++j)
          {
            const auto &block = block_matrix.block(i, j);
            for (const auto row : block_partitions[i])
              for (auto entry = block.begin(row); entry != block.end(row);
                   ++entry)
                sparsity.add(offsets[i] + row, offsets[j] + entry->column());
          }

      MatrixType result;
      result.reinit(partition, partition, sparsity, communicator_, true);
      for (unsigned int i = 0; i < field_layout_.n_blocks(); ++i)
        for (unsigned int j = 0; j < field_layout_.n_blocks(); ++j)
          {
            const auto &block = block_matrix.block(i, j);
            for (const auto row : block_partitions[i])
              for (auto entry = block.begin(row); entry != block.end(row);
                   ++entry)
                result.set(offsets[i] + row,
                           offsets[j] + entry->column(),
                           entry->value());
          }
      result.compress(dealii::VectorOperation::insert);
      return result;
    }

    MatrixType
    monolithic_matrix(const GlobalVectorType &state,
                      const GlobalVectorType *state_dot = nullptr,
                      const double            alpha     = 0.) const
    {
      return monolithic_matrix(block_matrix(state, state_dot, alpha));
    }

    bool
    has_local_preconditioner(const FieldId field) const
    {
      finalize();
      return model_.has_preconditioner(field);
    }

    std::optional<LocalOperator>
    local_preconditioner(const FieldId           field,
                         const GlobalVectorType &state) const
    {
      finalize();
      validate_state(state);
      if (!model_.has_preconditioner(field))
        return std::nullopt;

      StateView<FieldVectorType> state_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      EvaluationContext<FieldVectorType> context(0., state_view, nullptr);
      auto matrix = materialized_block(field, field, context, 0.);
      AssertThrow(matrix != nullptr,
                  dealii::ExcMessage(
                    "A local preconditioner requires a materialized diagonal "
                    "block."));
      return model_.preconditioner(field,
                                   *matrix,
                                   state.block(field_layout_.block(field)));
    }

    /** Assemble registered local inverses into a global block diagonal map. */
    Operator
    block_diagonal_preconditioner(const GlobalVectorType &state) const
    {
      finalize();
      validate_state(state);

      std::vector<LocalOperator> diagonal;
      diagonal.reserve(field_layout_.n_blocks());
      for (unsigned int block = 0; block < field_layout_.n_blocks(); ++block)
        {
          const auto field = field_layout_.field(block);
          const auto local = local_preconditioner(field, state);
          AssertThrow(local.has_value(),
                      dealii::ExcMessage(
                        "Block diagonal preconditioning requires a local "
                        "preconditioner for every semantic field."));
          diagonal.push_back(*local);
        }

      Operator result;
      result.reinit_range_vector = [state](GlobalVectorType &vector,
                                           const bool        omit) {
        vector.reinit(state, omit);
      };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult                = [diagonal](GlobalVectorType       &dst,
                                const GlobalVectorType &src) {
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].vmult(dst.block(block), src.block(block));
      };
      result.vmult_add = [diagonal](GlobalVectorType       &dst,
                                    const GlobalVectorType &src) {
        GlobalVectorType contribution;
        contribution.reinit(dst);
        contribution = 0.;
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].vmult(contribution.block(block), src.block(block));
        dst += contribution;
      };
      result.Tvmult = [diagonal](GlobalVectorType       &dst,
                                 const GlobalVectorType &src) {
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].Tvmult(dst.block(block), src.block(block));
      };
      result.Tvmult_add = [diagonal](GlobalVectorType       &dst,
                                     const GlobalVectorType &src) {
        GlobalVectorType contribution;
        contribution.reinit(dst);
        contribution = 0.;
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].Tvmult(contribution.block(block), src.block(block));
        dst += contribution;
      };
      return result;
    }

    /** Apply a lower or upper approximate block substitution. */
    Operator
    block_triangular_preconditioner(const GlobalVectorType &state,
                                    const bool              lower = true) const
    {
      finalize();
      validate_state(state);

      const unsigned int         n = field_layout_.n_blocks();
      std::vector<LocalOperator> diagonal;
      diagonal.reserve(n);
      for (unsigned int block = 0; block < n; ++block)
        {
          const auto local =
            local_preconditioner(field_layout_.field(block), state);
          AssertThrow(local.has_value(),
                      dealii::ExcMessage(
                        "Block triangular preconditioning requires a local "
                        "preconditioner for every semantic field."));
          diagonal.push_back(*local);
        }

      StateView<FieldVectorType> state_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      EvaluationContext<FieldVectorType>      context(0., state_view, nullptr);
      std::vector<std::vector<LocalOperator>> off_diagonal(
        n, std::vector<LocalOperator>(n));
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          off_diagonal[i][j] = model_.state_operator(field_layout_.field(i),
                                                     field_layout_.field(j),
                                                     context);

      auto apply = [diagonal,
                    off_diagonal,
                    lower](GlobalVectorType &dst, const GlobalVectorType &src) {
        GlobalVectorType rhs;
        rhs.reinit(src);
        rhs                  = src;
        const unsigned int n = diagonal.size();
        if (lower)
          for (unsigned int i = 0; i < n; ++i)
            {
              auto value = rhs.block(i);
              value *= -1.;
              for (unsigned int j = 0; j < i; ++j)
                off_diagonal[i][j].vmult_add(value, dst.block(j));
              value *= -1.;
              diagonal[i].vmult(dst.block(i), value);
            }
        else
          for (int i = static_cast<int>(n) - 1; i >= 0; --i)
            {
              auto value = rhs.block(i);
              value *= -1.;
              for (unsigned int j = i + 1; j < n; ++j)
                off_diagonal[i][j].vmult_add(value, dst.block(j));
              value *= -1.;
              diagonal[i].vmult(dst.block(i), value);
            }
      };

      Operator result;
      result.reinit_range_vector = [state](GlobalVectorType &vector,
                                           const bool        omit) {
        vector.reinit(state, omit);
      };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult                = apply;
      result.vmult_add            = [apply](GlobalVectorType       &dst,
                                 const GlobalVectorType &src) {
        GlobalVectorType contribution;
        contribution.reinit(dst);
        contribution = 0.;
        apply(contribution, src);
        dst += contribution;
      };
      result.Tvmult     = apply;
      result.Tvmult_add = result.vmult_add;
      return result;
    }

    /** Pack execution blocks into the concatenated field ordering. */
    FieldVectorType
    pack(const GlobalVectorType &global) const
    {
      validate_state(global);
      const auto      offsets = field_layout_.block_offsets();
      FieldVectorType result;
      result.reinit(field_layout_.monolithic_partition(), communicator_);
      result                = 0.;
      const auto partitions = field_layout_.block_partitions();
      for (unsigned int block = 0; block < partitions.size(); ++block)
        for (const auto index : partitions[block])
          result(offsets[block] + index) = global.block(block)(index);
      return result;
    }

    /** Unpack a concatenated vector into execution blocks. */
    void
    unpack(const FieldVectorType &flat, GlobalVectorType &global) const
    {
      finalize();
      const auto offsets = field_layout_.block_offsets();
      AssertThrow(flat.size() == offsets.back(),
                  dealii::ExcMessage(
                    "Flat vector does not match the execution layout."));
      reinit(global);
      const auto partitions = field_layout_.block_partitions();
      for (unsigned int block = 0; block < partitions.size(); ++block)
        for (const auto index : partitions[block])
          global.block(block)(index) = flat(offsets[block] + index);
    }

  private:
    bool
    can_materialize_matrix(const EvaluationContext<FieldVectorType> &context,
                           const double alpha) const
    {
      for (unsigned int i = 0; i < field_layout_.n_blocks(); ++i)
        for (unsigned int j = 0; j < field_layout_.n_blocks(); ++j)
          {
            const auto row       = field_layout_.field(i);
            const auto column    = field_layout_.field(j);
            const bool has_state = model_.has_state_operator(row, column);
            const bool has_derivative =
              model_.has_derivative_operator(row, column);
            if (has_state &&
                !model_.state_matrix_operator(row, column, context).has_value())
              return false;
            if (alpha != 0. && has_derivative &&
                !model_.derivative_matrix_operator(row, column, context)
                   .has_value())
              return false;
          }
      return true;
    }

    std::unique_ptr<MatrixType>
    materialized_block(const FieldId                             row,
                       const FieldId                             column,
                       const EvaluationContext<FieldVectorType> &context,
                       const double                              alpha) const
    {
      const bool has_state      = model_.has_state_operator(row, column);
      const bool has_derivative = model_.has_derivative_operator(row, column);
      if (!has_state && (alpha == 0. || !has_derivative))
        return {};

      std::optional<typename Model::MatrixOperator> state_matrix;
      std::optional<typename Model::MatrixOperator> derivative_matrix;
      if (has_state)
        state_matrix = model_.state_matrix_operator(row, column, context);
      if (alpha != 0. && has_derivative)
        derivative_matrix =
          model_.derivative_matrix_operator(row, column, context);
      AssertThrow(!has_state || state_matrix.has_value(),
                  dealii::ExcMessage(
                    "A state operator is not completely matrix-based."));
      AssertThrow(alpha == 0. || !has_derivative ||
                    derivative_matrix.has_value(),
                  dealii::ExcMessage(
                    "A derivative operator is not completely matrix-based."));

      std::unique_ptr<MatrixType> result;
      if (state_matrix.has_value())
        result = state_matrix->matrix();
      if (derivative_matrix.has_value())
        {
          auto derivative = derivative_matrix->matrix();
          if (result)
            result->add(alpha, *derivative);
          else
            {
              *derivative *= alpha;
              result = std::move(derivative);
            }
        }
      return result;
    }

    BlockMatrixType
    make_block_matrix(const EvaluationContext<FieldVectorType> &context,
                      const double                              alpha) const
    {
      AssertThrow(can_materialize_matrix(context, alpha),
                  dealii::ExcMessage(
                    "The current execution linearization is not completely "
                    "matrix-based."));

      const auto         partitions = field_layout_.block_partitions();
      const unsigned int n          = field_layout_.n_blocks();
      BlockMatrixType    result;
      result.reinit(n, n);
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          {
            auto  matrix = materialized_block(field_layout_.field(i),
                                             field_layout_.field(j),
                                             context,
                                             alpha);
            auto &block  = result.block(i, j);
            if (matrix)
              {
                block.reinit(*matrix);
                block.copy_from(*matrix);
              }
            else
              {
                dealii::DynamicSparsityPattern empty(partitions[i].size(),
                                                     partitions[j].size());
                block.reinit(
                  partitions[i], partitions[j], empty, communicator_, false);
              }
          }
      result.collect_sizes();
      return result;
    }

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
