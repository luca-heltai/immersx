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
#include <deal.II/base/mpi.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/vector_memory.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/contributor.h>
#include <immersx/core/second_order_preconditioner.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <type_traits>
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

    template <typename StateAccessorType>
    std::vector<dealii::IndexSet>
    block_partitions(const StateAccessorType &state, const double time) const
    {
      validate_complete();
      std::vector<dealii::IndexSet> result;
      result.reserve(fields_by_block_.size());
      for (const auto field : fields_by_block_)
        result.push_back(state.field(field, time).locally_owned_elements());
      return result;
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

    dealii::IndexSet
    differential_components(const GlobalBlockVectorType &state) const
    {
      validate_complete();
      std::size_t total_size = 0;
      for (unsigned int block_number = 0;
           block_number < fields_by_block_.size();
           ++block_number)
        total_size += state.block(block_number).locally_owned_elements().size();

      dealii::IndexSet result(total_size);
      std::size_t      offset = 0;
      for (unsigned int block_number = 0;
           block_number < fields_by_block_.size();
           ++block_number)
        {
          const auto field = fields_by_block_[block_number];
          const auto owned = state.block(block_number).locally_owned_elements();
          const auto &descriptor = layout_.field(field);

          // The descriptor stores differential components in the field's
          // execution numbering.  Intersect with the current state ownership
          // explicitly: on a distributed vector the descriptor may contain a
          // stale or rank-local mask, and comparing IndexSet sizes is not a
          // valid way to distinguish those cases.
          for (const auto index : descriptor.differential_components)
            if (owned.is_element(index))
              result.add_index(offset + index);
          offset += owned.size();
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
    using Model               = SemiDiscreteModel<FieldVectorType>;
    using Builder             = SemidiscreteBuilder<FieldVectorType>;
    using Operator            = dealii::LinearOperator<GlobalVectorType>;
    using LocalOperator       = typename Model::Operator;
    using SaddlePointMetadata = typename Model::SaddlePointMetadata;
    using MatrixType          = typename Model::MatrixOperator::Matrix;
    using BlockMatrixType     = ImmersXLA::MPI::BlockSparseMatrix;
    using SecondOrderPreconditioner =
      SecondOrderBlockPreconditioner<FieldVectorType, MatrixType>;

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

      model_.update_context(*context);

      // The execution layout fixes the number and ordering of semantic
      // fields, but an adaptive Problem may replace the vector behind a field
      // during a solver restart. Keep the residual storage conforming to the
      // candidate state rather than to the partitions captured before mesh
      // adaptation.
      residual.reinit(state);
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
      model_.update_context(*context);
      return make_global_operator(*context, alpha);
    }

    struct Snapshot
    {
      Snapshot(const ExecutionComposition &composition,
               const double                time,
               const GlobalVectorType     &state,
               const GlobalVectorType     *state_dot,
               const bool                  retain_matrices)
        : state_storage(state)
        , derivative_storage(state_dot ? *state_dot : composition.make_state())
        , state_view(composition.layout_, time)
        , derivative_view(composition.layout_, time)
        , context(time, state_view, state_dot ? &derivative_view : nullptr)
      {
        composition.field_layout_.bind_state(state_view, state_storage);
        if (state_dot)
          composition.field_layout_.bind_state(derivative_view,
                                               derivative_storage);
        if (retain_matrices)
          {
            composition.model_.update_context(context);
            linearization = composition.model_.retained_linearization(context);
          }
      }

      Model                              linearization;
      GlobalVectorType                   state_storage;
      GlobalVectorType                   derivative_storage;
      StateView<FieldVectorType>         state_view;
      StateView<FieldVectorType>         derivative_view;
      EvaluationContext<FieldVectorType> context;
    };

    std::shared_ptr<Snapshot>
    make_snapshot(const double            time,
                  const GlobalVectorType &state,
                  const GlobalVectorType &state_dot,
                  const bool              retain_matrices = false) const
    {
      validate_state(state);
      validate_state(state_dot);
      return std::make_shared<Snapshot>(
        *this, time, state, &state_dot, retain_matrices);
    }

    std::shared_ptr<Snapshot>
    make_snapshot(const double            time,
                  const GlobalVectorType &state,
                  const bool              retain_matrices) const
    {
      validate_state(state);
      return std::make_shared<Snapshot>(
        *this, time, state, nullptr, retain_matrices);
    }

    struct StateSnapshot
    {
      StateSnapshot(const ExecutionComposition &composition,
                    const double                time,
                    const GlobalVectorType     &state)
        : state_storage(state)
        , state_view(composition.layout_, time)
        , context(time, state_view)
      {
        composition.field_layout_.bind_state(state_view, state_storage);
      }

      GlobalVectorType                   state_storage;
      StateView<FieldVectorType>         state_view;
      EvaluationContext<FieldVectorType> context;
    };

    std::shared_ptr<StateSnapshot>
    make_snapshot(const double time, const GlobalVectorType &state) const
    {
      validate_state(state);
      return std::make_shared<StateSnapshot>(*this, time, state);
    }

    Operator
    jacobian(const EvaluationContext<FieldVectorType> &context,
             const double                              alpha,
             const Model *prepared_model = nullptr) const
    {
      finalize();
      if (prepared_model == nullptr)
        model_.update_context(context);
      return make_global_operator(context, alpha, prepared_model);
    }

    dealii::IndexSet
    differential_components() const
    {
      finalize();
      return field_layout_.differential_components();
    }

    dealii::IndexSet
    differential_components(const GlobalVectorType &state) const
    {
      return field_layout_.differential_components(state);
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

    std::optional<typename Model::MatrixOperator>
    state_matrix_operator(const GlobalVectorType &state,
                          const FieldId           row,
                          const FieldId           column,
                          const double            time = 0.) const
    {
      finalize();
      validate_state(state);
      StateView<FieldVectorType> state_view(layout_, time);
      field_layout_.bind_state(state_view, state);
      const EvaluationContext<FieldVectorType> context(time, state_view);
      return model_.state_matrix_operator(row, column, context);
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
                 const GlobalVectorType *state_dot      = nullptr,
                 const double            alpha          = 0.,
                 const double            time           = 0.,
                 const Model            *prepared_model = nullptr) const
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
          context.emplace(time, state_view, &derivative_view);
        }
      else
        context.emplace(time, state_view, nullptr);
      return make_block_matrix(*context, alpha, prepared_model);
    }

    /** Convert a block matrix to its concatenated monolithic sparse matrix. */
    void
    monolithic_matrix(const BlockMatrixType &block_matrix,
                      MatrixType            &result) const
    {
      finalize();
      AssertThrow(block_matrix.n_block_rows() == field_layout_.n_blocks() &&
                    block_matrix.n_block_cols() == field_layout_.n_blocks(),
                  dealii::ExcMessage(
                    "Block matrix does not match the execution layout."));

      const auto offsets          = field_layout_.block_offsets();
      const auto partition        = field_layout_.monolithic_partition();
      const auto block_partitions = field_layout_.block_partitions();
      dealii::DynamicSparsityPattern sparsity(offsets.back(),
                                              offsets.back(),
                                              partition);
      for (unsigned int i = 0; i < field_layout_.n_blocks(); ++i)
        for (unsigned int j = 0; j < field_layout_.n_blocks(); ++j)
          {
            const auto &block = block_matrix.block(i, j);
            for (const auto row : block_partitions[i])
              if (block.row_length(row) != 0)
                for (auto entry = block.begin(row), end = block.end(row);
                     entry != end;
                     ++entry)
                  sparsity.add(offsets[i] + row, offsets[j] + entry->column());
          }

      result.reinit(partition, partition, sparsity, communicator_, false);
      for (unsigned int i = 0; i < field_layout_.n_blocks(); ++i)
        for (unsigned int j = 0; j < field_layout_.n_blocks(); ++j)
          {
            const auto &block = block_matrix.block(i, j);
            for (const auto row : block_partitions[i])
              if (block.row_length(row) != 0)
                for (auto entry = block.begin(row), end = block.end(row);
                     entry != end;
                     ++entry)
                  result.set(offsets[i] + row,
                             offsets[j] + entry->column(),
                             entry->value());
          }
      result.compress(dealii::VectorOperation::insert);
    }

    MatrixType
    monolithic_matrix(const BlockMatrixType &block_matrix) const
    {
      MatrixType result;
      monolithic_matrix(block_matrix, result);
      return result;
    }

    MatrixType
    monolithic_matrix(const GlobalVectorType &state,
                      const GlobalVectorType *state_dot = nullptr,
                      const double            alpha     = 0.) const
    {
      return monolithic_matrix(block_matrix(state, state_dot, alpha));
    }

    void
    monolithic_matrix(const GlobalVectorType &state,
                      MatrixType             &result,
                      const GlobalVectorType *state_dot = nullptr,
                      const double            alpha     = 0.) const
    {
      monolithic_matrix(block_matrix(state, state_dot, alpha), result);
    }

    bool
    has_local_preconditioner(const FieldId field) const
    {
      finalize();
      return model_.has_preconditioner(field);
    }

    bool
    has_complete_local_preconditioners() const
    {
      finalize();
      for (unsigned int block = 0; block < field_layout_.n_blocks(); ++block)
        if (!model_.has_preconditioner(field_layout_.field(block)))
          return false;
      return true;
    }

    /** Register a problem-local approximate inverse factory. */
    template <typename Factory>
    void
    add_preconditioner(const FieldId field, Factory factory)
    {
      AssertThrow(!finalized_,
                  dealii::ExcMessage(
                    "Preconditioners must be registered before execution "
                    "starts."));
      model_.add_preconditioner(
        field, typename Model::PreconditionerFactory(std::move(factory)));
    }

    bool
    has_isolated_second_order_pair() const
    {
      return layout_.n_fields() == 2 && model_.second_order_pairs().size() == 1;
    }

    /** Inexact full LDU for an isolated mass-based second-order system. */
    Operator
    second_order_preconditioner(const GlobalVectorType &state,
                                const GlobalVectorType &state_dot,
                                const double            alpha,
                                const Snapshot *snapshot = nullptr) const
    {
      const auto *prepared_model =
        snapshot ? &snapshot->linearization : nullptr;
      const auto &model = prepared_model ? *prepared_model : model_;
      finalize();
      validate_state(state);
      validate_state(state_dot);
      AssertThrow(has_isolated_second_order_pair(),
                  dealii::ExcMessage("The full second-order factorization "
                                     "requires an isolated field pair."));
      AssertThrow(alpha > 0.,
                  dealii::ExcMessage("Second-order Schur reduction requires "
                                     "a positive derivative weight."));
      StateView<FieldVectorType> state_view(layout_, 0.);
      StateView<FieldVectorType> derivative_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      field_layout_.bind_state(derivative_view, state_dot);
      EvaluationContext<FieldVectorType> fallback_context(0.,
                                                          state_view,
                                                          &derivative_view);
      const auto &context = snapshot ? snapshot->context : fallback_context;
      const auto &pair    = model.second_order_pairs().front();
      const auto  inverse =
        second_order_inverse(pair, context, alpha, prepared_model);
      const auto displacement = field_layout_.block(pair.displacement);
      const auto velocity     = field_layout_.block(pair.velocity);
      const auto partitions   = field_layout_.block_partitions();
      const auto communicator = communicator_;
      Operator   result;
      result.reinit_range_vector =
        [partitions_storage =
           std::make_shared<const std::decay_t<decltype(partitions)>>(
             partitions),
         communicator](GlobalVectorType &vector, const bool) {
          const auto &partitions = *partitions_storage;
          vector.reinit(partitions, communicator);
        };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult =
        [inverse, displacement, velocity](GlobalVectorType       &destination,
                                          const GlobalVectorType &source) {
          inverse->vmult(destination.block(displacement),
                         destination.block(velocity),
                         source.block(displacement),
                         source.block(velocity));
        };
      auto memory =
        std::make_shared<dealii::GrowingVectorMemory<GlobalVectorType>>();
      result.vmult_add = [apply = result.vmult,
                          memory](GlobalVectorType       &destination,
                                  const GlobalVectorType &source) {
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *memory);
        contribution->reinit(destination);
        apply(*contribution, source);
        destination += *contribution;
      };
      return result;
    }

    const std::vector<SaddlePointMetadata> &
    saddle_points() const
    {
      finalize();
      return model_.saddle_points();
    }

    /** Build S ~= sum_i B_i P_i^-1 D_i - J_lambda,lambda.
     * At zero derivative weight, retain each registered second-order pair
     * in full; other primal cross-couplings remain in the outer operator.
     */
    LocalOperator
    schur_operator(const FieldId           multiplier,
                   const GlobalVectorType &state,
                   const GlobalVectorType *state_dot               = nullptr,
                   const double            alpha                   = 0.,
                   const bool              include_primal_coupling = false,
                   const Snapshot         *snapshot = nullptr) const
    {
      const auto *prepared_model =
        snapshot ? &snapshot->linearization : nullptr;
      const auto &model = prepared_model ? *prepared_model : model_;
      finalize();
      validate_state(state);
      const auto metadata =
        std::find_if(model.saddle_points().begin(),
                     model.saddle_points().end(),
                     [multiplier](const auto &entry) {
                       return entry.multiplier == multiplier;
                     });
      AssertThrow(metadata != model.saddle_points().end(),
                  dealii::ExcMessage(
                    "No saddle-point metadata exists for this multiplier."));
      AssertThrow(!metadata->participants.empty(),
                  dealii::ExcMessage(
                    "A Schur relation must contain at least one participant."));

      StateView<FieldVectorType> state_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      StateView<FieldVectorType> derivative_view(layout_, 0.);
      std::optional<EvaluationContext<FieldVectorType>> derivative_context;
      if (state_dot != nullptr)
        {
          validate_state(*state_dot);
          field_layout_.bind_state(derivative_view, *state_dot);
          derivative_context.emplace(0., state_view, &derivative_view);
        }
      else
        derivative_context.emplace(0., state_view, nullptr);
      const auto &context = snapshot ? snapshot->context : *derivative_context;

      if (alpha == 0. && has_static_second_order_pair(*metadata, model))
        return static_schur_operators(*metadata, state, context, prepared_model)
          .schur;

      std::vector<LocalOperator> terms;
      std::vector<LocalOperator> inverses, to, from;
      terms.reserve(metadata->participants.size());
      for (const auto participant : metadata->participants)
        {
          // Use the same fixed actions in Schur construction, elimination
          // and recovery to preserve the multiplier row.
          const auto inverse = local_preconditioner(
            participant, state, context, alpha, true, prepared_model);
          AssertThrow(inverse.has_value(),
                      dealii::ExcMessage(
                        "Schur construction requires a local inverse for "
                        "every participant field."));
          const auto to_multiplier =
            model.state_operator(multiplier, participant, context);
          const auto from_multiplier =
            model.state_operator(participant, multiplier, context);
          terms.push_back(to_multiplier * *inverse * from_multiplier);
          if (include_primal_coupling)
            {
              inverses.push_back(*inverse);
              to.push_back(to_multiplier);
              from.push_back(from_multiplier);
            }
        }

      // One Neumann correction, D^-1 - D^-1 E D^-1, retains paths through
      // two primal fields. In first-order mechanics the force enters v,
      // while displacement compatibility observes u.
      if (include_primal_coupling)
        for (unsigned int i = 0; i < inverses.size(); ++i)
          for (unsigned int j = 0; j < inverses.size(); ++j)
            if (i != j &&
                model.has_state_operator(multiplier,
                                         metadata->participants[i]) &&
                model.has_state_operator(metadata->participants[j],
                                         multiplier) &&
                (model.has_state_operator(metadata->participants[i],
                                          metadata->participants[j]) ||
                 model.has_derivative_operator(metadata->participants[i],
                                               metadata->participants[j])))
              terms.push_back(-1. * to[i] * inverses[i] *
                              linearized_operator(metadata->participants[i],
                                                  metadata->participants[j],
                                                  context,
                                                  alpha,
                                                  prepared_model) *
                              inverses[j] * from[j]);

      if (model.has_state_operator(multiplier, multiplier) ||
          (alpha != 0. &&
           model.has_derivative_operator(multiplier, multiplier)))
        terms.push_back(
          -1. * linearized_operator(
                  multiplier, multiplier, context, alpha, prepared_model));

      LocalOperator result = terms.front();
      for (std::size_t i = 1; i < terms.size(); ++i)
        result += terms[i];
      return result;
    }

    /** Build a global block-factorization preconditioner using a Schur solve.
     * The inner solve uses tolerance as a relative residual reduction.
     * Registered second-order pairs use the same fixed AMG actions in Schur
     * construction, elimination and recovery.
     */
    Operator
    schur_preconditioner(const FieldId           multiplier,
                         const GlobalVectorType &state,
                         const GlobalVectorType *state_dot          = nullptr,
                         const double            alpha              = 0.,
                         const unsigned int      maximum_iterations = 1000,
                         const double            tolerance          = 1.e-9,
                         const bool         include_primal_coupling = false,
                         const Snapshot    *snapshot                = nullptr,
                         const unsigned int materialization_limit   = 0) const
    {
      const auto *prepared_model =
        snapshot ? &snapshot->linearization : nullptr;
      const auto &model = prepared_model ? *prepared_model : model_;
      finalize();
      validate_state(state);
      const auto metadata =
        std::find_if(model.saddle_points().begin(),
                     model.saddle_points().end(),
                     [multiplier](const auto &entry) {
                       return entry.multiplier == multiplier;
                     });
      AssertThrow(metadata != model.saddle_points().end(),
                  dealii::ExcMessage(
                    "No saddle-point metadata exists for this multiplier."));
      AssertThrow(!metadata->participants.empty(),
                  dealii::ExcMessage(
                    "A Schur relation must contain at least one participant."));

      StateView<FieldVectorType> state_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      StateView<FieldVectorType> derivative_view(layout_, 0.);
      std::optional<EvaluationContext<FieldVectorType>> derivative_context;
      if (state_dot != nullptr)
        {
          validate_state(*state_dot);
          field_layout_.bind_state(derivative_view, *state_dot);
          derivative_context.emplace(0., state_view, &derivative_view);
        }
      else
        derivative_context.emplace(0., state_view, nullptr);
      const auto &context = snapshot ? snapshot->context : *derivative_context;
      if (alpha == 0. && has_static_second_order_pair(*metadata, model))
        return static_schur_preconditioner(*metadata,
                                           state,
                                           context,
                                           maximum_iterations,
                                           tolerance,
                                           prepared_model,
                                           materialization_limit);
      struct Participant
      {
        unsigned int  block;
        LocalOperator to_multiplier;
        LocalOperator from_multiplier;
        LocalOperator inverse;
      };
      std::vector<Participant> participants;
      participants.reserve(metadata->participants.size());
      for (const auto field : metadata->participants)
        {
          const auto inverse = local_preconditioner(
            field, state, context, alpha, true, prepared_model);
          AssertThrow(inverse.has_value(),
                      dealii::ExcMessage(
                        "Schur preconditioning requires local inverses for "
                        "all participant fields."));
          participants.push_back(
            {field_layout_.block(field),
             linearized_operator(
               multiplier, field, context, alpha, prepared_model),
             linearized_operator(
               field, multiplier, context, alpha, prepared_model),
             *inverse});
        }

      struct NonParticipant
      {
        unsigned int  block;
        LocalOperator inverse;
      };
      std::vector<NonParticipant> non_participants;
      for (unsigned int block = 0; block < field_layout_.n_blocks(); ++block)
        {
          const auto field = field_layout_.field(block);
          if (field == multiplier ||
              std::any_of(metadata->participants.begin(),
                          metadata->participants.end(),
                          [field](const auto participant) {
                            return participant == field;
                          }))
            continue;

          const auto inverse = local_preconditioner(
            field, state, context, alpha, false, prepared_model);
          AssertThrow(inverse.has_value(),
                      dealii::ExcMessage(
                        "Schur preconditioning requires local inverses for "
                        "all non-participant fields."));
          non_participants.push_back({block, *inverse});
        }

      const auto schur = schur_operator(
        multiplier, state, state_dot, alpha, include_primal_coupling, snapshot);
      struct PrimalCorrection
      {
        unsigned int  row, column;
        LocalOperator inverse;
      };
      std::vector<PrimalCorrection> primal_corrections;
      if (include_primal_coupling)
        for (unsigned int i = 0; i < participants.size(); ++i)
          for (unsigned int j = 0; j < participants.size(); ++j)
            if (i != j &&
                (model.has_state_operator(metadata->participants[i],
                                          metadata->participants[j]) ||
                 model.has_derivative_operator(metadata->participants[i],
                                               metadata->participants[j])))
              primal_corrections.push_back(
                {i,
                 j,
                 -1. * participants[i].inverse *
                   linearized_operator(metadata->participants[i],
                                       metadata->participants[j],
                                       context,
                                       alpha,
                                       prepared_model) *
                   participants[j].inverse});
      const auto multiplier_block          = field_layout_.block(multiplier);
      const auto multiplier_preconditioner = this->multiplier_preconditioner(
        multiplier, state, context, prepared_model);
      auto solve_multiplier = make_schur_solver(schur,
                                                multiplier_preconditioner,
                                                maximum_iterations,
                                                tolerance,
                                                materialization_limit,
                                                false);

      auto apply = [participants,
                    primal_corrections,
                    non_participants,
                    multiplier_block,
                    solve_multiplier](GlobalVectorType       &dst,
                                      const GlobalVectorType &src,
                                      const bool              transpose) {
        dst = 0.;

        for (const auto &non_participant : non_participants)
          if (transpose)
            non_participant.inverse.Tvmult(dst.block(non_participant.block),
                                           src.block(non_participant.block));
          else
            non_participant.inverse.vmult(dst.block(non_participant.block),
                                          src.block(non_participant.block));

        FieldVectorType schur_rhs;
        FieldVectorType multiplier_solution;
        if (transpose)
          {
            const auto &first = participants.front();
            first.from_multiplier.reinit_domain_vector(schur_rhs, false);
            schur_rhs = src.block(multiplier_block);
            schur_rhs *= -1.;

            for (const auto &participant : participants)
              schur_rhs +=
                dealii::transpose_operator(participant.from_multiplier) *
                dealii::transpose_operator(participant.inverse) *
                src.block(participant.block);
          }
        else
          {
            const auto &first = participants.front();
            first.to_multiplier.reinit_range_vector(schur_rhs, false);
            schur_rhs = src.block(multiplier_block);
            schur_rhs *= -1.;

            for (const auto &participant : participants)
              schur_rhs += participant.to_multiplier * participant.inverse *
                           src.block(participant.block);
          }

        // Use the same corrected primal inverse in elimination and recovery
        // as in the Schur approximation.
        for (const auto &correction : primal_corrections)
          if (transpose)
            schur_rhs += dealii::transpose_operator(
                           participants[correction.column].from_multiplier) *
                         dealii::transpose_operator(correction.inverse) *
                         src.block(participants[correction.row].block);
          else
            schur_rhs += participants[correction.row].to_multiplier *
                         correction.inverse *
                         src.block(participants[correction.column].block);

        participants.front().to_multiplier.reinit_range_vector(
          multiplier_solution, false);
        multiplier_solution = 0.;
        solve_multiplier(multiplier_solution, schur_rhs, transpose);
        dst.block(multiplier_block) = multiplier_solution;

        for (const auto &participant : participants)
          if (transpose)
            dst.block(participant.block) =
              dealii::transpose_operator(participant.inverse) *
              (src.block(participant.block) -
               dealii::transpose_operator(participant.to_multiplier) *
                 multiplier_solution);
          else
            dst.block(participant.block) =
              participant.inverse *
              (src.block(participant.block) -
               participant.from_multiplier * multiplier_solution);

        for (const auto &correction : primal_corrections)
          if (transpose)
            dst.block(participants[correction.column].block) +=
              dealii::transpose_operator(correction.inverse) *
              (src.block(participants[correction.row].block) -
               dealii::transpose_operator(
                 participants[correction.row].to_multiplier) *
                 multiplier_solution);
          else
            dst.block(participants[correction.row].block) +=
              correction.inverse *
              (src.block(participants[correction.column].block) -
               participants[correction.column].from_multiplier *
                 multiplier_solution);
      };

      auto global_vector_memory =
        std::make_shared<dealii::GrowingVectorMemory<GlobalVectorType>>();
      const auto owned_apply =
        std::make_shared<const decltype(apply)>(std::move(apply));
      Operator   result;
      const auto partitions   = field_layout_.block_partitions();
      const auto communicator = communicator_;
      result.reinit_range_vector =
        [partitions_storage =
           std::make_shared<const std::decay_t<decltype(partitions)>>(
             partitions),
         communicator](GlobalVectorType &vector, const bool) {
          const auto &partitions = *partitions_storage;
          vector.reinit(partitions, communicator);
        };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult                = [owned_apply](GlobalVectorType       &dst,
                                   const GlobalVectorType &src) {
        (*owned_apply)(dst, src, false);
      };
      result.vmult_add = [owned_apply,
                          global_vector_memory](GlobalVectorType       &dst,
                                                const GlobalVectorType &src) {
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *global_vector_memory);
        contribution->reinit(dst);
        (*owned_apply)(*contribution, src, false);
        dst += *contribution;
      };
      result.Tvmult = [owned_apply](GlobalVectorType       &dst,
                                    const GlobalVectorType &src) {
        (*owned_apply)(dst, src, true);
      };
      result.Tvmult_add = [owned_apply,
                           global_vector_memory](GlobalVectorType       &dst,
                                                 const GlobalVectorType &src) {
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *global_vector_memory);
        contribution->reinit(dst);
        (*owned_apply)(*contribution, src, true);
        dst += *contribution;
      };
      return result;
    }

    /** Add gamma D W^-1 times the constraint row to each primal row. */
    Operator
    augmented_lagrangian_operator(const GlobalVectorType &state,
                                  const double            gamma     = 1.e1,
                                  const GlobalVectorType *state_dot = nullptr,
                                  const double            alpha     = 0.) const
    {
      finalize();
      validate_state(state);
      if (state_dot != nullptr)
        validate_state(*state_dot);
      AssertThrow(gamma > 0.,
                  dealii::ExcMessage(
                    "The augmented-Lagrangian parameter must be positive."));

      StateView<FieldVectorType> state_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      StateView<FieldVectorType> derivative_view(layout_, 0.);
      std::optional<EvaluationContext<FieldVectorType>> context_storage;
      if (state_dot != nullptr)
        {
          field_layout_.bind_state(derivative_view, *state_dot);
          context_storage.emplace(0., state_view, &derivative_view);
        }
      else
        context_storage.emplace(0., state_view, nullptr);
      const auto &context = *context_storage;

      struct Augmentation
      {
        std::vector<unsigned int>               participants;
        std::vector<unsigned int>               columns;
        std::vector<std::vector<LocalOperator>> terms;
      };
      std::vector<Augmentation> augmentations;
      augmentations.reserve(model_.saddle_points().size());
      for (const auto &metadata : model_.saddle_points())
        {
          const auto metric =
            model_.multiplier_metric(metadata.multiplier, context);
          LocalOperator inverse_metric_operator;
          if (metric.has_value())
            {
              const auto metric_matrix = metric->matrix();
              const auto multiplier_partition =
                layout_.field(metadata.multiplier).locally_owned;
              auto inverse_metric = std::make_shared<FieldVectorType>();
              inverse_metric->reinit(multiplier_partition, communicator_);
              for (const auto index : multiplier_partition)
                {
                  const auto value         = metric_matrix->diag_element(index);
                  (*inverse_metric)(index) = inverse_lumped_metric_value(value);
                }
              inverse_metric->compress(dealii::VectorOperation::insert);

              inverse_metric_operator.reinit_range_vector =
                [multiplier_partition_storage = std::make_shared<
                   const std::decay_t<decltype(multiplier_partition)>>(
                   multiplier_partition),
                 communicator = communicator_](FieldVectorType &vector,
                                               const bool       omit) {
                  const auto &multiplier_partition =
                    *multiplier_partition_storage;
                  vector.reinit(multiplier_partition, communicator, omit);
                };
              inverse_metric_operator.reinit_domain_vector =
                inverse_metric_operator.reinit_range_vector;
              inverse_metric_operator.vmult =
                [inverse_metric,
                 multiplier_partition_storage = std::make_shared<
                   const std::decay_t<decltype(multiplier_partition)>>(
                   multiplier_partition)](FieldVectorType       &dst,
                                          const FieldVectorType &src) {
                  const auto &multiplier_partition =
                    *multiplier_partition_storage;
                  dst = src;
                  for (const auto index : multiplier_partition)
                    dst(index) = (*inverse_metric)(index)*src(index);
                };
              inverse_metric_operator.vmult_add =
                [inverse_metric,
                 multiplier_partition_storage = std::make_shared<
                   const std::decay_t<decltype(multiplier_partition)>>(
                   multiplier_partition)](FieldVectorType       &dst,
                                          const FieldVectorType &src) {
                  const auto &multiplier_partition =
                    *multiplier_partition_storage;
                  for (const auto index : multiplier_partition)
                    dst(index) += (*inverse_metric)(index)*src(index);
                };
              inverse_metric_operator.Tvmult = inverse_metric_operator.vmult;
              inverse_metric_operator.Tvmult_add =
                inverse_metric_operator.vmult_add;
            }
          else
            {
              const auto inverse =
                local_preconditioner(metadata.multiplier, state);
              AssertThrow(inverse.has_value(),
                          dealii::ExcMessage(
                            "Augmented-Lagrangian composition requires a "
                            "multiplier metric or local preconditioner."));
              inverse_metric_operator = *inverse;
            }

          Augmentation augmentation;
          for (const auto participant : metadata.participants)
            {
              augmentation.participants.push_back(
                field_layout_.block(participant));
            }

          const auto participant_count = augmentation.participants.size();
          std::vector<LocalOperator> to_multiplier;
          std::vector<LocalOperator> from_multiplier;
          to_multiplier.reserve(participant_count);
          from_multiplier.reserve(participant_count);
          for (const auto participant : metadata.participants)
            {
              to_multiplier.push_back(linearized_operator(
                metadata.multiplier, participant, context, alpha));
              from_multiplier.push_back(linearized_operator(
                participant, metadata.multiplier, context, alpha));
            }

          augmentation.columns = augmentation.participants;
          if (model_.has_state_operator(metadata.multiplier,
                                        metadata.multiplier) ||
              (alpha != 0. &&
               model_.has_derivative_operator(metadata.multiplier,
                                              metadata.multiplier)))
            {
              augmentation.columns.push_back(
                field_layout_.block(metadata.multiplier));
              to_multiplier.push_back(linearized_operator(
                metadata.multiplier, metadata.multiplier, context, alpha));
            }
          augmentation.terms.resize(participant_count,
                                    std::vector<LocalOperator>(
                                      to_multiplier.size()));
          for (std::size_t i = 0; i < participant_count; ++i)
            for (std::size_t j = 0; j < to_multiplier.size(); ++j)
              augmentation.terms[i][j] =
                gamma * (from_multiplier[i] * inverse_metric_operator *
                         to_multiplier[j]);
          augmentations.push_back(std::move(augmentation));
        }

      const auto partitions   = field_layout_.block_partitions();
      const auto communicator = communicator_;
      Operator   result       = make_global_operator(context, alpha);
      for (auto &augmentation : augmentations)
        {
          const auto augmentation_storage =
            std::make_shared<const Augmentation>(std::move(augmentation));
          Operator contribution;
          contribution.reinit_range_vector =
            [partitions_storage =
               std::make_shared<const std::decay_t<decltype(partitions)>>(
                 partitions),
             communicator](GlobalVectorType &vector, const bool) {
              const auto &partitions = *partitions_storage;
              vector.reinit(partitions, communicator);
            };
          contribution.reinit_domain_vector = contribution.reinit_range_vector;
          contribution.vmult =
            [augmentation_storage](GlobalVectorType       &destination,
                                   const GlobalVectorType &source) {
              const auto &augmentation = *augmentation_storage;
              destination              = 0.;
              for (std::size_t i = 0; i < augmentation.participants.size(); ++i)
                for (std::size_t j = 0; j < augmentation.columns.size(); ++j)
                  augmentation.terms[i][j].vmult_add(
                    destination.block(augmentation.participants[i]),
                    source.block(augmentation.columns[j]));
            };
          contribution.vmult_add =
            [augmentation_storage](GlobalVectorType       &destination,
                                   const GlobalVectorType &source) {
              const auto &augmentation = *augmentation_storage;
              for (std::size_t i = 0; i < augmentation.participants.size(); ++i)
                for (std::size_t j = 0; j < augmentation.columns.size(); ++j)
                  augmentation.terms[i][j].vmult_add(
                    destination.block(augmentation.participants[i]),
                    source.block(augmentation.columns[j]));
            };
          contribution.Tvmult =
            [augmentation_storage](GlobalVectorType       &destination,
                                   const GlobalVectorType &source) {
              const auto &augmentation = *augmentation_storage;
              destination              = 0.;
              for (std::size_t i = 0; i < augmentation.participants.size(); ++i)
                for (std::size_t j = 0; j < augmentation.columns.size(); ++j)
                  dealii::transpose_operator(augmentation.terms[i][j])
                    .vmult_add(destination.block(augmentation.columns[j]),
                               source.block(augmentation.participants[i]));
            };
          contribution.Tvmult_add =
            [augmentation_storage](GlobalVectorType       &destination,
                                   const GlobalVectorType &source) {
              const auto &augmentation = *augmentation_storage;
              for (std::size_t i = 0; i < augmentation.participants.size(); ++i)
                for (std::size_t j = 0; j < augmentation.columns.size(); ++j)
                  dealii::transpose_operator(augmentation.terms[i][j])
                    .vmult_add(destination.block(augmentation.columns[j]),
                               source.block(augmentation.participants[i]));
            };
          result += contribution;
        }

      return result;
    }

    /**
     * Materialize the augmented operator, including its multiplier column.
     *
     * The multiplier metric is intentionally reduced to its diagonal here:
     * this is the matrix-based counterpart of the lumped metric used by the
     * first AL implementation.  Each participant pair is assembled, so
     * augmentation cross terms are retained when a relation has multiple
     * primal fields.
     */
    BlockMatrixType
    augmented_lagrangian_matrix(const GlobalVectorType &state,
                                const double            gamma     = 1.e1,
                                const GlobalVectorType *state_dot = nullptr,
                                const double            alpha     = 0.) const
    {
      finalize();
      validate_state(state);
      if (state_dot != nullptr)
        validate_state(*state_dot);
      AssertThrow(gamma > 0.,
                  dealii::ExcMessage(
                    "The augmented-Lagrangian parameter must be positive."));

      StateView<FieldVectorType> state_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      StateView<FieldVectorType> derivative_view(layout_, 0.);
      std::optional<EvaluationContext<FieldVectorType>> context_storage;
      if (state_dot != nullptr)
        {
          field_layout_.bind_state(derivative_view, *state_dot);
          context_storage.emplace(0., state_view, &derivative_view);
        }
      else
        context_storage.emplace(0., state_view, nullptr);
      const auto &context    = *context_storage;
      auto        result     = make_block_matrix(context, alpha);
      const auto  partitions = field_layout_.block_partitions();

      for (const auto &metadata : model_.saddle_points())
        {
          const auto metric =
            model_.multiplier_metric(metadata.multiplier, context);
          AssertThrow(metric.has_value(),
                      dealii::ExcMessage(
                        "Matrix-based augmented-Lagrangian composition "
                        "requires a materialized multiplier metric."));

          const auto multiplier_block =
            field_layout_.block(metadata.multiplier);
          const auto metric_matrix = metric->matrix();
          const auto multiplier_partition =
            layout_.field(metadata.multiplier).locally_owned;
          FieldVectorType inverse_metric;
          inverse_metric.reinit(state.block(multiplier_block));
          for (const auto index : multiplier_partition)
            {
              const auto value      = metric_matrix->diag_element(index);
              inverse_metric(index) = inverse_lumped_metric_value(value);
            }
          inverse_metric.compress(dealii::VectorOperation::insert);

          auto columns = metadata.participants;
          if (model_.has_state_operator(metadata.multiplier,
                                        metadata.multiplier) ||
              (alpha != 0. &&
               model_.has_derivative_operator(metadata.multiplier,
                                              metadata.multiplier)))
            columns.push_back(metadata.multiplier);
          for (std::size_t i = 0; i < metadata.participants.size(); ++i)
            for (std::size_t j = 0; j < columns.size(); ++j)
              {
                const auto row_field = metadata.participants[i];
                const auto col_field = columns[j];
                const auto from      = materialized_block(row_field,
                                                     metadata.multiplier,
                                                     context,
                                                     alpha);
                const auto to        = materialized_block(metadata.multiplier,
                                                   col_field,
                                                   context,
                                                   alpha);
                AssertThrow(from && to,
                            dealii::ExcMessage(
                              "Matrix-based augmented-Lagrangian composition "
                              "requires materialized coupling blocks."));
                AssertThrow(from->n() == to->m(),
                            dealii::ExcDimensionMismatch(from->n(), to->m()));
                MatrixType product;
                const auto column_partition =
                  layout_.field(col_field).locally_owned;
                const auto from_normalized =
                  repartition_matrix(*from,
                                     layout_.field(row_field).locally_owned,
                                     multiplier_partition);
                const auto to_normalized =
                  repartition_matrix(*to,
                                     multiplier_partition,
                                     column_partition);
                from_normalized.mmult(product, to_normalized, inverse_metric);
                const auto row_block = field_layout_.block(row_field);
                const auto col_block = field_layout_.block(col_field);
                const auto combined =
                  add_matrices(result.block(row_block, col_block),
                               product,
                               gamma,
                               partitions[row_block],
                               partitions[col_block]);
                result.block(row_block, col_block).reinit(combined);
                result.block(row_block, col_block).copy_from(combined);
              }
        }

      result.collect_sizes();
      return result;
    }

    /** Apply an augmented-Lagrangian block factorization. */
    Operator
    augmented_lagrangian_preconditioner(
      const GlobalVectorType &state,
      const double            gamma     = 1.e1,
      const GlobalVectorType *state_dot = nullptr,
      const double            alpha     = 0.) const
    {
      finalize();
      validate_state(state);
      if (state_dot != nullptr)
        validate_state(*state_dot);
      AssertThrow(gamma > 0.,
                  dealii::ExcMessage(
                    "The augmented-Lagrangian parameter must be positive."));
      AssertThrow(model_.saddle_points().size() == 1u,
                  dealii::ExcMessage(
                    "Augmented-Lagrangian preconditioning currently requires "
                    "exactly one saddle-point relation."));

      const auto &metadata         = model_.saddle_points().front();
      const auto  multiplier_block = field_layout_.block(metadata.multiplier);
      StateView<FieldVectorType> state_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      StateView<FieldVectorType> derivative_view(layout_, 0.);
      std::optional<EvaluationContext<FieldVectorType>> context_storage;
      if (state_dot != nullptr)
        {
          field_layout_.bind_state(derivative_view, *state_dot);
          context_storage.emplace(0., state_view, &derivative_view);
        }
      else
        context_storage.emplace(0., state_view, nullptr);
      const auto               &context = *context_storage;
      std::vector<unsigned int> participant_blocks;
      participant_blocks.reserve(metadata.participants.size());
      for (const auto field : metadata.participants)
        participant_blocks.push_back(field_layout_.block(field));

      const auto               partitions = field_layout_.block_partitions();
      std::vector<std::size_t> participant_offsets(participant_blocks.size() +
                                                     1,
                                                   0);
      for (std::size_t i = 0; i < participant_blocks.size(); ++i)
        participant_offsets[i + 1] =
          participant_offsets[i] + partitions[participant_blocks[i]].size();

      const auto primal_partition = [&]() {
        dealii::IndexSet result(participant_offsets.back());
        for (std::size_t i = 0; i < participant_blocks.size(); ++i)
          for (const auto index : partitions[participant_blocks[i]])
            result.add_index(participant_offsets[i] + index);
        result.compress();
        return result;
      }();

      const auto augmented_matrix =
        augmented_lagrangian_matrix(state, gamma, state_dot, alpha);
      dealii::DynamicSparsityPattern primal_sparsity(participant_offsets.back(),
                                                     participant_offsets.back(),
                                                     primal_partition);
      for (std::size_t i = 0; i < participant_blocks.size(); ++i)
        for (std::size_t j = 0; j < participant_blocks.size(); ++j)
          {
            const auto &block = augmented_matrix.block(participant_blocks[i],
                                                       participant_blocks[j]);
            for (const auto row : partitions[participant_blocks[i]])
              if (block.row_length(row) != 0)
                for (auto entry = block.begin(row), end = block.end(row);
                     entry != end;
                     ++entry)
                  primal_sparsity.add(participant_offsets[i] + row,
                                      participant_offsets[j] + entry->column());
          }

      auto primal_matrix = std::make_shared<MatrixType>();
      primal_matrix->reinit(primal_partition,
                            primal_partition,
                            primal_sparsity,
                            communicator_,
                            false);
      for (std::size_t i = 0; i < participant_blocks.size(); ++i)
        for (std::size_t j = 0; j < participant_blocks.size(); ++j)
          {
            const auto &block = augmented_matrix.block(participant_blocks[i],
                                                       participant_blocks[j]);
            for (const auto row : partitions[participant_blocks[i]])
              if (block.row_length(row) != 0)
                for (auto entry = block.begin(row), end = block.end(row);
                     entry != end;
                     ++entry)
                  primal_matrix->set(participant_offsets[i] + row,
                                     participant_offsets[j] + entry->column(),
                                     entry->value());
          }
      primal_matrix->compress(dealii::VectorOperation::insert);

      const auto primal_reinit =
        [primal_partition_storage =
           std::make_shared<const std::decay_t<decltype(primal_partition)>>(
             primal_partition),
         communicator = communicator_](FieldVectorType &vector,
                                       const bool       omit) {
          const auto &primal_partition = *primal_partition_storage;
          vector.reinit(primal_partition, communicator, omit);
        };
      auto primal_inverse =
        make_amg_preconditioner<FieldVectorType, MatrixType>(*primal_matrix,
                                                             primal_reinit);
      // PreconditionAMG observes its matrix.  Capture the concrete matrix in
      // every action so the inverse cannot outlive the assembled superblock.
      const auto primal_vmult      = primal_inverse.vmult;
      const auto primal_vmult_add  = primal_inverse.vmult_add;
      const auto primal_Tvmult     = primal_inverse.Tvmult;
      const auto primal_Tvmult_add = primal_inverse.Tvmult_add;
      primal_inverse.vmult         = [primal_matrix,
                              primal_vmult](FieldVectorType       &dst,
                                            const FieldVectorType &src) {
        primal_vmult(dst, src);
      };
      primal_inverse.vmult_add =
        [primal_matrix, primal_vmult_add](FieldVectorType       &dst,
                                          const FieldVectorType &src) {
          primal_vmult_add(dst, src);
        };
      primal_inverse.Tvmult = [primal_matrix,
                               primal_Tvmult](FieldVectorType       &dst,
                                              const FieldVectorType &src) {
        primal_Tvmult(dst, src);
      };
      primal_inverse.Tvmult_add =
        [primal_matrix, primal_Tvmult_add](FieldVectorType       &dst,
                                           const FieldVectorType &src) {
          primal_Tvmult_add(dst, src);
        };

      const auto multiplier_inverse =
        multiplier_preconditioner(metadata.multiplier, state, context);
      std::vector<LocalOperator> from_multiplier;
      std::vector<LocalOperator> to_multiplier;
      from_multiplier.reserve(participant_blocks.size());
      to_multiplier.reserve(participant_blocks.size());
      for (const auto field : metadata.participants)
        {
          from_multiplier.push_back(
            linearized_operator(field, metadata.multiplier, context, alpha));
          to_multiplier.push_back(
            linearized_operator(metadata.multiplier, field, context, alpha));
        }

      struct NonParticipant
      {
        unsigned int  block;
        LocalOperator inverse;
      };
      std::vector<NonParticipant> non_participants;
      for (unsigned int block = 0; block < field_layout_.n_blocks(); ++block)
        if (block != multiplier_block &&
            std::find(participant_blocks.begin(),
                      participant_blocks.end(),
                      block) == participant_blocks.end())
          {
            const auto inverse = local_preconditioner(
              field_layout_.field(block), state, context, alpha);
            AssertThrow(inverse.has_value(),
                        dealii::ExcMessage(
                          "Augmented-Lagrangian preconditioning requires a "
                          "local preconditioner for every nonparticipant "
                          "field."));
            non_participants.push_back({block, *inverse});
          }

      auto field_vector_memory =
        std::make_shared<dealii::GrowingVectorMemory<FieldVectorType>>();
      auto vector_memory =
        std::make_shared<dealii::GrowingVectorMemory<GlobalVectorType>>();
      auto apply =
        [participant_blocks,
         participant_offsets,
         primal_partition_storage =
           std::make_shared<const std::decay_t<decltype(primal_partition)>>(
             primal_partition),
         partitions_storage =
           std::make_shared<const std::decay_t<decltype(partitions)>>(
             partitions),
         multiplier_block,
         multiplier_inverse,
         primal_inverse,
         from_multiplier,
         to_multiplier,
         non_participants,
         gamma,
         communicator = communicator_,
         field_vector_memory](GlobalVectorType       &dst,
                              const GlobalVectorType &src,
                              const bool              transpose) {
          const auto &primal_partition = *primal_partition_storage;
          const auto &partitions       = *partitions_storage;
          dst                          = 0.;
          for (const auto &non_participant : non_participants)
            if (transpose)
              non_participant.inverse.Tvmult(dst.block(non_participant.block),
                                             src.block(non_participant.block));
            else
              non_participant.inverse.vmult(dst.block(non_participant.block),
                                            src.block(non_participant.block));

          typename dealii::VectorMemory<FieldVectorType>::Pointer primal_rhs(
            *field_vector_memory);
          typename dealii::VectorMemory<FieldVectorType>::Pointer
            primal_solution(*field_vector_memory);
          primal_rhs->reinit(primal_partition, communicator);
          primal_solution->reinit(primal_partition, communicator);
          *primal_rhs = 0.;
          if (!transpose)
            {
              typename dealii::VectorMemory<FieldVectorType>::Pointer lambda(
                *field_vector_memory);
              multiplier_inverse.reinit_range_vector(*lambda, false);
              multiplier_inverse.vmult(*lambda, src.block(multiplier_block));
              *lambda *= -gamma;
              for (std::size_t i = 0; i < participant_blocks.size(); ++i)
                for (const auto index : partitions[participant_blocks[i]])
                  (*primal_rhs)(participant_offsets[i] + index) =
                    src.block(participant_blocks[i])(index);
              primal_rhs->compress(dealii::VectorOperation::insert);
              for (std::size_t i = 0; i < participant_blocks.size(); ++i)
                {
                  typename dealii::VectorMemory<FieldVectorType>::Pointer value(
                    *field_vector_memory);
                  from_multiplier[i].reinit_range_vector(*value, false);
                  from_multiplier[i].vmult(*value, *lambda);
                  for (const auto index : partitions[participant_blocks[i]])
                    (*primal_rhs)(participant_offsets[i] + index) +=
                      (*value)(index);
                }
              primal_rhs->compress(dealii::VectorOperation::add);
              primal_inverse.vmult(*primal_solution, *primal_rhs);
              for (std::size_t i = 0; i < participant_blocks.size(); ++i)
                for (const auto index : partitions[participant_blocks[i]])
                  dst.block(participant_blocks[i])(index) =
                    (*primal_solution)(participant_offsets[i] + index);
              dst.block(multiplier_block) = *lambda;
            }
          else
            {
              for (std::size_t i = 0; i < participant_blocks.size(); ++i)
                for (const auto index : partitions[participant_blocks[i]])
                  (*primal_rhs)(participant_offsets[i] + index) =
                    src.block(participant_blocks[i])(index);
              primal_rhs->compress(dealii::VectorOperation::insert);
              primal_inverse.Tvmult(*primal_solution, *primal_rhs);
              typename dealii::VectorMemory<FieldVectorType>::Pointer
                multiplier_rhs(*field_vector_memory);
              multiplier_inverse.reinit_domain_vector(*multiplier_rhs, false);
              *multiplier_rhs = src.block(multiplier_block);
              multiplier_rhs->compress(dealii::VectorOperation::insert);
              for (std::size_t i = 0; i < participant_blocks.size(); ++i)
                {
                  typename dealii::VectorMemory<FieldVectorType>::Pointer value(
                    *field_vector_memory);
                  typename dealii::VectorMemory<FieldVectorType>::Pointer
                    participant_solution(*field_vector_memory);
                  to_multiplier[i].reinit_domain_vector(*participant_solution,
                                                        false);
                  for (const auto index : partitions[participant_blocks[i]])
                    (*participant_solution)(index) =
                      (*primal_solution)(participant_offsets[i] + index);
                  to_multiplier[i].reinit_range_vector(*value, false);
                  to_multiplier[i].vmult(*value, *participant_solution);
                  for (const auto index : partitions[participant_blocks[i]])
                    dst.block(participant_blocks[i])(index) =
                      (*primal_solution)(participant_offsets[i] + index);
                  multiplier_rhs->add(1., *value);
                }
              multiplier_rhs->compress(dealii::VectorOperation::add);
              multiplier_inverse.Tvmult(dst.block(multiplier_block),
                                        *multiplier_rhs);
              dst.block(multiplier_block) *= -gamma;
            }
        };

      const auto communicator = communicator_;
      const auto apply_storage =
        std::make_shared<const decltype(apply)>(std::move(apply));
      Operator result;
      result.reinit_range_vector =
        [partitions_storage =
           std::make_shared<const std::decay_t<decltype(partitions)>>(
             partitions),
         communicator](GlobalVectorType &vector, const bool) {
          const auto &partitions = *partitions_storage;
          vector.reinit(partitions, communicator);
        };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult                = [apply_storage](GlobalVectorType       &dst,
                                     const GlobalVectorType &src) {
        const auto &apply = *apply_storage;
        apply(dst, src, false);
      };
      result.vmult_add = [apply_storage,
                          vector_memory](GlobalVectorType       &dst,
                                         const GlobalVectorType &src) {
        const auto &apply = *apply_storage;
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *vector_memory);
        contribution->reinit(dst);
        apply(*contribution, src, false);
        dst += *contribution;
      };
      result.Tvmult = [apply_storage](GlobalVectorType       &dst,
                                      const GlobalVectorType &src) {
        const auto &apply = *apply_storage;
        apply(dst, src, true);
      };
      result.Tvmult_add = [apply_storage,
                           vector_memory](GlobalVectorType       &dst,
                                          const GlobalVectorType &src) {
        const auto &apply = *apply_storage;
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *vector_memory);
        contribution->reinit(dst);
        apply(*contribution, src, true);
        dst += *contribution;
      };
      return result;
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
      return local_preconditioner(field, state, context, 0.);
    }

    std::optional<LocalOperator>
    local_preconditioner(const FieldId                             field,
                         const GlobalVectorType                   &state,
                         const EvaluationContext<FieldVectorType> &context,
                         const double                              alpha,
                         const bool   fixed_linear_action = false,
                         const Model *prepared_model      = nullptr) const
    {
      const auto &model = prepared_model ? *prepared_model : model_;
      finalize();
      validate_state(state);
      if (alpha > 0.)
        for (const auto &pair : model.second_order_pairs())
          if (field == pair.displacement || field == pair.velocity)
            {
              const auto inverse =
                second_order_inverse(pair, context, alpha, prepared_model);
              const bool    displacement = field == pair.displacement;
              LocalOperator result;
              const auto    owned = layout_.field(field).locally_owned;
              result.reinit_range_vector =
                [owned_storage =
                   std::make_shared<const std::decay_t<decltype(owned)>>(owned),
                 communicator = communicator_](FieldVectorType &vector,
                                               const bool) {
                  const auto &owned = *owned_storage;
                  vector.reinit(owned, communicator);
                };
              result.reinit_domain_vector = result.reinit_range_vector;
              result.vmult = [inverse, displacement, fixed_linear_action](
                               FieldVectorType       &destination,
                               const FieldVectorType &source) {
                if (fixed_linear_action && displacement)
                  inverse->apply_mass_amg(destination, source);
                else if (fixed_linear_action)
                  inverse->apply_schur_amg(destination, source);
                else if (displacement)
                  inverse->mass_inverse(destination, source);
                else
                  inverse->schur_inverse(destination, source);
              };
              // Each inner system and its AMG cycle are symmetric. A flexible
              // outer solver is still required for residual-stopped CG.
              result.Tvmult = result.vmult;
              auto memory   = std::make_shared<
                  dealii::GrowingVectorMemory<FieldVectorType>>();
              result.vmult_add = [apply = result.vmult,
                                  memory](FieldVectorType       &destination,
                                          const FieldVectorType &source) {
                typename dealii::VectorMemory<FieldVectorType>::Pointer
                  contribution(*memory);
                contribution->reinit(destination);
                apply(*contribution, source);
                destination += *contribution;
              };
              result.Tvmult_add = result.vmult_add;
              return result;
            }
      if (!model.has_preconditioner(field))
        return std::nullopt;

      std::shared_ptr<MatrixType> matrix;
      if (model.has_state_operator(field, field) ||
          (alpha != 0. && model.has_derivative_operator(field, field)))
        {
          matrix =
            materialized_block(field, field, context, alpha, prepared_model);
        }
      if (matrix)
        {
          const auto owned        = layout_.field(field).locally_owned;
          const auto communicator = communicator_;
          const auto reinit_vector =
            [owned_storage =
               std::make_shared<const std::decay_t<decltype(owned)>>(owned),
             communicator](FieldVectorType &vector, const bool) {
              const auto &owned = *owned_storage;
              vector.reinit(owned, communicator);
            };
          const auto local =
            model.preconditioner(field, *matrix, reinit_vector);
          if (local.has_value())
            {
              auto       result     = *local;
              const auto vmult      = result.vmult;
              const auto vmult_add  = result.vmult_add;
              const auto Tvmult     = result.Tvmult;
              const auto Tvmult_add = result.Tvmult_add;
              result.vmult = [matrix, vmult](FieldVectorType       &dst,
                                             const FieldVectorType &src) {
                vmult(dst, src);
              };
              result.vmult_add = [matrix,
                                  vmult_add](FieldVectorType       &dst,
                                             const FieldVectorType &src) {
                vmult_add(dst, src);
              };
              result.Tvmult = [matrix, Tvmult](FieldVectorType       &dst,
                                               const FieldVectorType &src) {
                Tvmult(dst, src);
              };
              result.Tvmult_add = [matrix,
                                   Tvmult_add](FieldVectorType       &dst,
                                               const FieldVectorType &src) {
                Tvmult_add(dst, src);
              };
              return result;
            }
        }

      // A structurally null diagonal is valid for a mixed saddle system.
      // Give a registered metric factory an empty, correctly partitioned
      // matrix; do not manufacture an inverse for an absent block.
      const auto partition = layout_.field(field).locally_owned;
      dealii::DynamicSparsityPattern empty(partition.size(),
                                           partition.size(),
                                           partition);
      auto empty_matrix = std::make_shared<MatrixType>();
      empty_matrix->reinit(partition, partition, empty, communicator_, false);
      const auto local = model.preconditioner(
        field,
        *empty_matrix,
        [partition_storage =
           std::make_shared<const std::decay_t<decltype(partition)>>(partition),
         communicator = communicator_](FieldVectorType &vector, const bool) {
          const auto &partition = *partition_storage;
          vector.reinit(partition, communicator);
        });
      if (!local.has_value())
        return std::nullopt;
      auto       result     = *local;
      const auto vmult      = result.vmult;
      const auto vmult_add  = result.vmult_add;
      const auto Tvmult     = result.Tvmult;
      const auto Tvmult_add = result.Tvmult_add;
      result.vmult          = [empty_matrix, vmult](FieldVectorType       &dst,
                                           const FieldVectorType &src) {
        vmult(dst, src);
      };
      result.vmult_add = [empty_matrix, vmult_add](FieldVectorType       &dst,
                                                   const FieldVectorType &src) {
        vmult_add(dst, src);
      };
      result.Tvmult = [empty_matrix, Tvmult](FieldVectorType       &dst,
                                             const FieldVectorType &src) {
        Tvmult(dst, src);
      };
      result.Tvmult_add = [empty_matrix,
                           Tvmult_add](FieldVectorType       &dst,
                                       const FieldVectorType &src) {
        Tvmult_add(dst, src);
      };
      return result;
    }

    /** Build the metric preconditioner used by a multiplier Schur solve. */
    LocalOperator
    multiplier_preconditioner(const FieldId           multiplier,
                              const GlobalVectorType &state,
                              const EvaluationContext<FieldVectorType> &context,
                              const Model *prepared_model = nullptr) const
    {
      const auto &model     = prepared_model ? *prepared_model : model_;
      const auto  partition = layout_.field(multiplier).locally_owned;
      const auto  metric    = model.multiplier_metric(multiplier, context);
      if (metric.has_value())
        {
          const auto metric_matrix = metric->matrix();
          if (model.has_preconditioner(multiplier))
            {
              const auto reinitializer =
                [partition_storage =
                   std::make_shared<const std::decay_t<decltype(partition)>>(
                     partition),
                 communicator = communicator_](FieldVectorType &vector,
                                               const bool       omit) {
                  const auto &partition = *partition_storage;
                  vector.reinit(partition, communicator, omit);
                };
              const auto local =
                model.preconditioner(multiplier, *metric_matrix, reinitializer);
              if (local.has_value())
                return *local;
            }

          auto inverse_metric = std::make_shared<FieldVectorType>();
          inverse_metric->reinit(partition, communicator_);
          for (const auto index : partition)
            (*inverse_metric)(index) =
              inverse_lumped_metric_value(metric_matrix->diag_element(index));
          inverse_metric->compress(dealii::VectorOperation::insert);

          LocalOperator result;
          result.reinit_range_vector =
            [partition_storage =
               std::make_shared<const std::decay_t<decltype(partition)>>(
                 partition),
             communicator = communicator_](FieldVectorType &vector,
                                           const bool       omit) {
              const auto &partition = *partition_storage;
              vector.reinit(partition, communicator, omit);
            };
          result.reinit_domain_vector = result.reinit_range_vector;
          result.vmult = [inverse_metric](FieldVectorType       &dst,
                                          const FieldVectorType &src) {
            dst = src;
            for (const auto index : dst.locally_owned_elements())
              dst(index) *= (*inverse_metric)(index);
            dst.compress(dealii::VectorOperation::insert);
          };
          result.vmult_add = [result](FieldVectorType       &dst,
                                      const FieldVectorType &src) mutable {
            FieldVectorType contribution;
            result.reinit_range_vector(contribution, false);
            result.vmult(contribution, src);
            dst += contribution;
          };
          result.Tvmult     = result.vmult;
          result.Tvmult_add = result.vmult_add;
          return result;
        }

      if (const auto local = local_preconditioner(
            multiplier, state, context, 0., false, prepared_model);
          local.has_value())
        return *local;

      return dealii::identity_operator<FieldVectorType>(
        [partition_storage =
           std::make_shared<const std::decay_t<decltype(partition)>>(partition),
         communicator = communicator_](FieldVectorType &vector,
                                       const bool       omit) {
          const auto &partition = *partition_storage;
          vector.reinit(partition, communicator, omit);
        });
    }

    /** Assemble registered local inverses into a global block diagonal map. */
    Operator
    block_diagonal_preconditioner(const GlobalVectorType &state,
                                  const GlobalVectorType *state_dot = nullptr,
                                  const double            alpha     = 0.,
                                  const Snapshot *snapshot = nullptr) const
    {
      const auto *prepared_model =
        snapshot ? &snapshot->linearization : nullptr;
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

      const auto &selected_context = snapshot ? snapshot->context : *context;
      std::vector<LocalOperator> diagonal;
      diagonal.reserve(field_layout_.n_blocks());
      for (unsigned int block = 0; block < field_layout_.n_blocks(); ++block)
        {
          const auto field = field_layout_.field(block);
          const auto local = local_preconditioner(
            field, state, selected_context, alpha, false, prepared_model);
          AssertThrow(local.has_value(),
                      dealii::ExcMessage(
                        "Block diagonal preconditioning requires a local "
                        "preconditioner for every semantic field."));
          diagonal.push_back(*local);
        }

      const auto partitions   = field_layout_.block_partitions();
      const auto communicator = communicator_;
      const auto diagonal_storage =
        std::make_shared<const decltype(diagonal)>(std::move(diagonal));
      Operator   result;
      const auto partitions_storage =
        std::make_shared<const decltype(partitions)>(partitions);
      result.reinit_range_vector = [partitions_storage,
                                    communicator](GlobalVectorType &vector,
                                                  const bool) {
        const auto &partitions = *partitions_storage;
        vector.reinit(partitions, communicator);
      };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult = [diagonal_storage](GlobalVectorType       &destination,
                                        const GlobalVectorType &source) {
        const auto &diagonal = *diagonal_storage;
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].vmult(destination.block(block), source.block(block));
      };
      result.vmult_add = [diagonal_storage](GlobalVectorType       &destination,
                                            const GlobalVectorType &source) {
        const auto &diagonal = *diagonal_storage;
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].vmult_add(destination.block(block),
                                    source.block(block));
      };
      result.Tvmult = [diagonal_storage](GlobalVectorType       &destination,
                                         const GlobalVectorType &source) {
        const auto &diagonal = *diagonal_storage;
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].Tvmult(destination.block(block), source.block(block));
      };
      result.Tvmult_add = [diagonal_storage](GlobalVectorType &destination,
                                             const GlobalVectorType &source) {
        const auto &diagonal = *diagonal_storage;
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].Tvmult_add(destination.block(block),
                                     source.block(block));
      };
      return result;
    }

    /** Apply a lower or upper approximate block substitution. */
    Operator
    block_triangular_preconditioner(const GlobalVectorType &state,
                                    const bool              lower     = true,
                                    const GlobalVectorType *state_dot = nullptr,
                                    const double            alpha     = 0.,
                                    const Snapshot *snapshot = nullptr) const
    {
      const auto *prepared_model =
        snapshot ? &snapshot->linearization : nullptr;
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

      const unsigned int n         = field_layout_.n_blocks();
      const auto &selected_context = snapshot ? snapshot->context : *context;
      std::vector<LocalOperator> diagonal;
      diagonal.reserve(n);
      for (unsigned int block = 0; block < n; ++block)
        {
          const auto local = local_preconditioner(field_layout_.field(block),
                                                  state,
                                                  selected_context,
                                                  alpha,
                                                  false,
                                                  prepared_model);
          AssertThrow(local.has_value(),
                      dealii::ExcMessage(
                        "Block triangular preconditioning requires a local "
                        "preconditioner for every semantic field."));
          diagonal.push_back(*local);
        }

      std::vector<std::vector<LocalOperator>> off_diagonal(
        n, std::vector<LocalOperator>(n));
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          if (i != j && (lower ? j < i : j > i))
            off_diagonal[i][j] = linearized_operator(field_layout_.field(i),
                                                     field_layout_.field(j),
                                                     selected_context,
                                                     alpha,
                                                     prepared_model);

      auto apply = [diagonal, off_diagonal, lower](GlobalVectorType       &dst,
                                                   const GlobalVectorType &src,
                                                   const bool transpose) {
        const unsigned int n           = diagonal.size();
        const bool         forward     = lower != transpose;
        const auto         apply_block = [&](const unsigned int i) {
          dealii::PackagedOperation<FieldVectorType> rhs(src.block(i));
          if (!transpose)
            {
              if (lower)
                for (unsigned int j = 0; j < i; ++j)
                  rhs -= off_diagonal[i][j] * dst.block(j);
              else
                for (unsigned int j = i + 1; j < n; ++j)
                  rhs -= off_diagonal[i][j] * dst.block(j);
            }
          else if (lower)
            for (unsigned int j = i + 1; j < n; ++j)
              rhs -=
                dealii::transpose_operator(off_diagonal[j][i]) * dst.block(j);
          else
            for (unsigned int j = 0; j < i; ++j)
              rhs -=
                dealii::transpose_operator(off_diagonal[j][i]) * dst.block(j);

          const auto inverse =
            transpose ? dealii::transpose_operator(diagonal[i]) : diagonal[i];
          dst.block(i) = inverse * rhs;
        };

        if (forward)
          for (unsigned int i = 0; i < n; ++i)
            apply_block(i);
        else
          for (unsigned int i = n; i-- > 0;)
            apply_block(i);
      };

      const auto partitions   = field_layout_.block_partitions();
      const auto communicator = communicator_;
      auto       vector_memory =
        std::make_shared<dealii::GrowingVectorMemory<GlobalVectorType>>();
      const auto apply_storage =
        std::make_shared<const decltype(apply)>(std::move(apply));
      Operator result;
      result.reinit_range_vector =
        [partitions_storage =
           std::make_shared<const std::decay_t<decltype(partitions)>>(
             partitions),
         communicator](GlobalVectorType &vector, const bool) {
          const auto &partitions = *partitions_storage;
          vector.reinit(partitions, communicator);
        };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult = [apply_storage](GlobalVectorType       &destination,
                                     const GlobalVectorType &source) {
        const auto &apply = *apply_storage;
        apply(destination, source, false);
      };
      result.vmult_add = [apply_storage,
                          vector_memory](GlobalVectorType       &destination,
                                         const GlobalVectorType &source) {
        const auto &apply = *apply_storage;
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *vector_memory);
        contribution->reinit(destination);
        apply(*contribution, source, false);
        destination += *contribution;
      };
      result.Tvmult = [apply_storage](GlobalVectorType       &destination,
                                      const GlobalVectorType &source) {
        const auto &apply = *apply_storage;
        apply(destination, source, true);
      };
      result.Tvmult_add = [apply_storage,
                           vector_memory](GlobalVectorType       &destination,
                                          const GlobalVectorType &source) {
        const auto &apply = *apply_storage;
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *vector_memory);
        contribution->reinit(destination);
        apply(*contribution, source, true);
        destination += *contribution;
      };
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
    /** Retain one bounded Schur matrix per snapshot and verify its solves
     * against the original action in the original left-preconditioned norm.
     */
    std::function<void(FieldVectorType &, const FieldVectorType &, bool)>
    make_schur_solver(const LocalOperator &schur,
                      const LocalOperator &metric,
                      const unsigned int   maximum_iterations,
                      const double         tolerance,
                      const unsigned int   materialization_limit,
                      const bool           retain_full_basis) const
    {
      AssertThrow(
        dealii::Utilities::MPI::min(materialization_limit, communicator_) ==
          dealii::Utilities::MPI::max(materialization_limit, communicator_),
        dealii::ExcMessage(
          "Schur materialization limits differ across ranks."));
      FieldVectorType prototype;
      schur.reinit_domain_vector(prototype, false);
      AssertThrow(
        dealii::Utilities::MPI::min(prototype.size(), communicator_) ==
          dealii::Utilities::MPI::max(prototype.size(), communicator_),
        dealii::ExcMessage("Schur dimensions differ across ranks."));
      const bool materialized = materialization_limit > 0 &&
                                prototype.size() > 0 &&
                                prototype.size() <= materialization_limit;
      const auto solver_schur =
        materialized ?
          detail::materialize_replicated_operator(schur,
                                                  communicator_,
                                                  materialization_limit) :
          schur;
      return [schur,
              solver_schur,
              metric,
              maximum_iterations,
              tolerance,
              materialized,
              retain_full_basis](FieldVectorType       &solution,
                                 const FieldVectorType &rhs,
                                 const bool             transpose) {
        schur.reinit_domain_vector(solution, false);
        solution = 0.;
        dealii::ReductionControl control(maximum_iterations, 0., tolerance);
        // A bounded static multiplier system can retain its complete basis.
        // Include the workspace slots used by older supported constructors.
        const auto basis_size =
          std::min<dealii::types::global_dof_index>(rhs.size(),
                                                    maximum_iterations);
        const unsigned int workspace =
          materialized && retain_full_basis ?
            static_cast<unsigned int>(basis_size) + 2 :
            30;
        typename dealii::SolverGMRES<FieldVectorType>::AdditionalData data(
          workspace);
        data.orthogonalization_strategy = dealii::LinearAlgebra::
          OrthogonalizationStrategy::modified_gram_schmidt;
        const auto solve = [&](FieldVectorType       &value,
                               const FieldVectorType &right_hand_side,
                               dealii::SolverControl &solver_control) {
          dealii::SolverGMRES<FieldVectorType> solver(solver_control, data);
          if (transpose)
            solver.solve(dealii::transpose_operator(solver_schur),
                         value,
                         right_hand_side,
                         dealii::transpose_operator(metric));
          else
            solver.solve(solver_schur, value, right_hand_side, metric);
        };
        solve(solution, rhs, control);
        unsigned int iterations = control.last_step();
        if (materialized)
          {
            FieldVectorType residual, weighted_residual, weighted_rhs;
            schur.reinit_range_vector(residual, false);
            metric.reinit_range_vector(weighted_residual, false);
            metric.reinit_range_vector(weighted_rhs, false);
            if (transpose)
              metric.Tvmult(weighted_rhs, rhs);
            else
              metric.vmult(weighted_rhs, rhs);
            const double reference = weighted_rhs.l2_norm();
            AssertThrow(std::isfinite(reference),
                        dealii::ExcMessage(
                          "The Schur right-hand side is not finite."));
            const double target = tolerance * reference;
            // Sampling and applying an operator can round differently. Refine
            // against the original action without relaxing its residual target.
            for (unsigned int refinement = 0;; ++refinement)
              {
                if (transpose)
                  schur.Tvmult(residual, solution);
                else
                  schur.vmult(residual, solution);
                residual -= rhs;
                if (transpose)
                  metric.Tvmult(weighted_residual, residual);
                else
                  metric.vmult(weighted_residual, residual);
                const double defect = weighted_residual.l2_norm();
                if (std::isfinite(defect) && defect <= target)
                  break;
                AssertThrow(std::isfinite(defect) && refinement < 3 &&
                              iterations < maximum_iterations,
                            dealii::SolverControl::NoConvergence(iterations,
                                                                 defect));
                FieldVectorType correction;
                schur.reinit_domain_vector(correction, false);
                correction = 0.;
                residual *= -1.;
                dealii::ReductionControl correction_control(
                  maximum_iterations - iterations, target, tolerance);
                solve(correction, residual, correction_control);
                const auto used = correction_control.last_step();
                AssertThrow(used > 0,
                            dealii::SolverControl::NoConvergence(iterations,
                                                                 defect));
                iterations += used;
                solution += correction;
              }
          }
      };
    }

    static bool
    has_static_second_order_pair(const SaddlePointMetadata &metadata,
                                 const Model               &model)
    {
      for (const auto &pair : model.second_order_pairs())
        if (std::find(metadata.participants.begin(),
                      metadata.participants.end(),
                      pair.displacement) != metadata.participants.end() &&
            std::find(metadata.participants.begin(),
                      metadata.participants.end(),
                      pair.velocity) != metadata.participants.end())
          return true;
      return false;
    }

    using CouplingColumn =
      dealii::LinearOperator<GlobalVectorType, FieldVectorType>;
    using CouplingRow =
      dealii::LinearOperator<FieldVectorType, GlobalVectorType>;

    struct StaticSchurOperators
    {
      Operator       inverse;
      CouplingColumn from_multiplier;
      CouplingRow    to_multiplier;
      LocalOperator  schur;
    };

    StaticSchurOperators
    static_schur_operators(const SaddlePointMetadata                &metadata,
                           const GlobalVectorType                   &state,
                           const EvaluationContext<FieldVectorType> &context,
                           const Model *prepared_model) const
    {
      // Retain each static mechanical pair in full. Other primal fields use
      // their local inverses; cross-couplings between distinct groups remain
      // in the outer Krylov operator, as with the block approximation above.
      const auto &model = prepared_model ? *prepared_model : model_;
      struct Pair
      {
        unsigned int                               displacement, velocity;
        std::shared_ptr<SecondOrderPreconditioner> inverse;
      };
      std::vector<Pair>         pairs;
      std::vector<unsigned int> paired_blocks;
      for (const auto &pair : model.second_order_pairs())
        {
          const bool displacement =
            std::find(metadata.participants.begin(),
                      metadata.participants.end(),
                      pair.displacement) != metadata.participants.end();
          const bool velocity =
            std::find(metadata.participants.begin(),
                      metadata.participants.end(),
                      pair.velocity) != metadata.participants.end();
          if (!displacement && !velocity)
            continue;
          AssertThrow(
            displacement && velocity,
            dealii::ExcMessage(
              "A static Schur inverse requires both second-order fields."));
          const auto u = field_layout_.block(pair.displacement);
          const auto v = field_layout_.block(pair.velocity);
          pairs.push_back(
            {u, v, second_order_inverse(pair, context, 0., prepared_model)});
          paired_blocks.insert(paired_blocks.end(), {u, v});
        }
      struct Participant
      {
        unsigned int  block;
        LocalOperator to_multiplier, from_multiplier;
      };
      struct Diagonal
      {
        unsigned int  block;
        LocalOperator inverse;
      };
      std::vector<Participant> participants;
      std::vector<Diagonal>    diagonal;
      for (const auto field : metadata.participants)
        {
          const auto block = field_layout_.block(field);
          participants.push_back(
            {block,
             model.state_operator(metadata.multiplier, field, context),
             model.state_operator(field, metadata.multiplier, context)});
          if (std::find(paired_blocks.begin(), paired_blocks.end(), block) ==
              paired_blocks.end())
            {
              const auto inverse = local_preconditioner(
                field, state, context, 0., true, prepared_model);
              AssertThrow(
                inverse.has_value(),
                dealii::ExcMessage(
                  "Static Schur construction requires a primal inverse."));
              diagonal.push_back({block, *inverse});
            }
        }

      const auto partitions   = field_layout_.block_partitions();
      const auto communicator = communicator_;
      const auto reinit_global =
        [partitions_storage =
           std::make_shared<const std::decay_t<decltype(partitions)>>(
             partitions),
         communicator](GlobalVectorType &v, const bool) {
          const auto &partitions = *partitions_storage;
          v.reinit(partitions, communicator);
        };
      const auto multiplier_partition =
        layout_.field(metadata.multiplier).locally_owned;
      const auto reinit_multiplier =
        [multiplier_partition_storage =
           std::make_shared<const std::decay_t<decltype(multiplier_partition)>>(
             multiplier_partition),
         communicator](FieldVectorType &v, const bool) {
          const auto &multiplier_partition = *multiplier_partition_storage;
          v.reinit(multiplier_partition, communicator);
        };
      auto primal_inverse = [pairs, diagonal, reinit_global]() {
        const auto apply = [pairs, diagonal](GlobalVectorType       &dst,
                                             const GlobalVectorType &src,
                                             const bool transpose) {
          dst = 0.;
          for (const auto &entry : diagonal)
            if (transpose)
              entry.inverse.Tvmult(dst.block(entry.block),
                                   src.block(entry.block));
            else
              entry.inverse.vmult(dst.block(entry.block),
                                  src.block(entry.block));
          for (const auto &pair : pairs)
            if (transpose)
              pair.inverse->static_Tvmult(dst.block(pair.displacement),
                                          dst.block(pair.velocity),
                                          src.block(pair.displacement),
                                          src.block(pair.velocity),
                                          true);
            else
              pair.inverse->static_vmult(dst.block(pair.displacement),
                                         dst.block(pair.velocity),
                                         src.block(pair.displacement),
                                         src.block(pair.velocity),
                                         true);
        };
        const auto apply_storage =
          std::make_shared<const decltype(apply)>(apply);
        Operator result;
        result.reinit_range_vector  = reinit_global;
        result.reinit_domain_vector = reinit_global;
        result.vmult = [apply_storage](GlobalVectorType       &dst,
                                       const GlobalVectorType &src) {
          const auto &apply = *apply_storage;
          apply(dst, src, false);
        };
        result.Tvmult = [apply_storage](GlobalVectorType       &dst,
                                        const GlobalVectorType &src) {
          const auto &apply = *apply_storage;
          apply(dst, src, true);
        };
        auto memory =
          std::make_shared<dealii::GrowingVectorMemory<GlobalVectorType>>();
        result.vmult_add = [apply_storage,
                            memory](GlobalVectorType       &dst,
                                    const GlobalVectorType &src) {
          const auto &apply = *apply_storage;
          typename dealii::VectorMemory<GlobalVectorType>::Pointer tmp(*memory);
          tmp->reinit(dst);
          apply(*tmp, src, false);
          dst += *tmp;
        };
        result.Tvmult_add = [apply_storage,
                             memory](GlobalVectorType       &dst,
                                     const GlobalVectorType &src) {
          const auto &apply = *apply_storage;
          typename dealii::VectorMemory<GlobalVectorType>::Pointer tmp(*memory);
          tmp->reinit(dst);
          apply(*tmp, src, true);
          dst += *tmp;
        };
        return result;
      };
      const auto participants_storage =
        std::make_shared<const decltype(participants)>(std::move(participants));
      StaticSchurOperators result;
      result.inverse                              = primal_inverse();
      result.from_multiplier.reinit_range_vector  = reinit_global;
      result.from_multiplier.reinit_domain_vector = reinit_multiplier;
      result.from_multiplier.vmult =
        [participants_storage](GlobalVectorType      &dst,
                               const FieldVectorType &src) {
          const auto &participants = *participants_storage;
          dst                      = 0.;
          for (const auto &p : participants)
            p.from_multiplier.vmult(dst.block(p.block), src);
        };
      result.from_multiplier.vmult_add =
        [participants_storage](GlobalVectorType      &dst,
                               const FieldVectorType &src) {
          const auto &participants = *participants_storage;
          for (const auto &p : participants)
            p.from_multiplier.vmult_add(dst.block(p.block), src);
        };
      result.from_multiplier.Tvmult =
        [participants_storage, reinit_multiplier](FieldVectorType        &dst,
                                                  const GlobalVectorType &src) {
          const auto &participants = *participants_storage;
          reinit_multiplier(dst, false);
          for (const auto &p : participants)
            p.from_multiplier.Tvmult_add(dst, src.block(p.block));
        };
      result.from_multiplier.Tvmult_add =
        [participants_storage](FieldVectorType        &dst,
                               const GlobalVectorType &src) {
          const auto &participants = *participants_storage;
          for (const auto &p : participants)
            p.from_multiplier.Tvmult_add(dst, src.block(p.block));
        };
      result.to_multiplier.reinit_range_vector  = reinit_multiplier;
      result.to_multiplier.reinit_domain_vector = reinit_global;
      result.to_multiplier.vmult =
        [participants_storage, reinit_multiplier](FieldVectorType        &dst,
                                                  const GlobalVectorType &src) {
          const auto &participants = *participants_storage;
          reinit_multiplier(dst, false);
          for (const auto &p : participants)
            p.to_multiplier.vmult_add(dst, src.block(p.block));
        };
      result.to_multiplier.vmult_add =
        [participants_storage](FieldVectorType        &dst,
                               const GlobalVectorType &src) {
          const auto &participants = *participants_storage;
          for (const auto &p : participants)
            p.to_multiplier.vmult_add(dst, src.block(p.block));
        };
      result.to_multiplier.Tvmult =
        [participants_storage](GlobalVectorType      &dst,
                               const FieldVectorType &src) {
          const auto &participants = *participants_storage;
          dst                      = 0.;
          for (const auto &p : participants)
            p.to_multiplier.Tvmult(dst.block(p.block), src);
        };
      result.to_multiplier.Tvmult_add =
        [participants_storage](GlobalVectorType      &dst,
                               const FieldVectorType &src) {
          const auto &participants = *participants_storage;
          for (const auto &p : participants)
            p.to_multiplier.Tvmult_add(dst.block(p.block), src);
        };
      result.schur =
        result.to_multiplier * result.inverse * result.from_multiplier;
      if (model.has_state_operator(metadata.multiplier, metadata.multiplier))
        result.schur -= model.state_operator(metadata.multiplier,
                                             metadata.multiplier,
                                             context);
      return result;
    }

    Operator
    static_schur_preconditioner(
      const SaddlePointMetadata                &metadata,
      const GlobalVectorType                   &state,
      const EvaluationContext<FieldVectorType> &context,
      const unsigned int                        maximum_iterations,
      const double                              tolerance,
      const Model                              *prepared_model,
      const unsigned int                        materialization_limit) const
    {
      const auto operators =
        static_schur_operators(metadata, state, context, prepared_model);
      const auto metric = multiplier_preconditioner(metadata.multiplier,
                                                    state,
                                                    context,
                                                    prepared_model);
      const auto solve_multiplier = make_schur_solver(operators.schur,
                                                      metric,
                                                      maximum_iterations,
                                                      tolerance,
                                                      materialization_limit,
                                                      true);
      const auto multiplier_block = field_layout_.block(metadata.multiplier);
      struct NonParticipant
      {
        unsigned int  block;
        LocalOperator inverse;
      };
      std::vector<NonParticipant> non_participants;
      for (unsigned int block = 0; block < field_layout_.n_blocks(); ++block)
        {
          const auto field = field_layout_.field(block);
          if (field == metadata.multiplier ||
              std::find(metadata.participants.begin(),
                        metadata.participants.end(),
                        field) != metadata.participants.end())
            continue;
          const auto inverse = local_preconditioner(
            field, state, context, 0., false, prepared_model);
          AssertThrow(
            inverse.has_value(),
            dealii::ExcMessage(
              "Static Schur preconditioning requires non-participant inverses."));
          non_participants.push_back({block, *inverse});
        }
      const auto apply = [operators,
                          multiplier_block,
                          non_participants,
                          solve_multiplier](GlobalVectorType       &dst,
                                            const GlobalVectorType &src,
                                            const bool              transpose) {
        GlobalVectorType primal;
        operators.inverse.reinit_range_vector(primal, false);
        if (transpose)
          operators.inverse.Tvmult(primal, src);
        else
          operators.inverse.vmult(primal, src);
        FieldVectorType rhs, lambda;
        operators.to_multiplier.reinit_range_vector(rhs, false);
        lambda.reinit(rhs);
        if (transpose)
          operators.from_multiplier.Tvmult(rhs, primal);
        else
          operators.to_multiplier.vmult(rhs, primal);
        rhs -= src.block(multiplier_block);
        solve_multiplier(lambda, rhs, transpose);
        if (transpose)
          operators.to_multiplier.Tvmult(primal, lambda);
        else
          operators.from_multiplier.vmult(primal, lambda);
        primal.sadd(-1., 1., src);
        if (transpose)
          operators.inverse.Tvmult(dst, primal);
        else
          operators.inverse.vmult(dst, primal);
        dst.block(multiplier_block) = lambda;
        for (const auto &entry : non_participants)
          if (transpose)
            entry.inverse.Tvmult(dst.block(entry.block),
                                 src.block(entry.block));
          else
            entry.inverse.vmult(dst.block(entry.block), src.block(entry.block));
      };
      const auto apply_storage =
        std::make_shared<const decltype(apply)>(std::move(apply));
      Operator result;
      result.reinit_range_vector  = operators.inverse.reinit_range_vector;
      result.reinit_domain_vector = operators.inverse.reinit_domain_vector;
      result.vmult                = [apply_storage](GlobalVectorType       &dst,
                                     const GlobalVectorType &src) {
        const auto &apply = *apply_storage;
        apply(dst, src, false);
      };
      result.Tvmult = [apply_storage](GlobalVectorType       &dst,
                                      const GlobalVectorType &src) {
        const auto &apply = *apply_storage;
        apply(dst, src, true);
      };
      auto memory =
        std::make_shared<dealii::GrowingVectorMemory<GlobalVectorType>>();
      result.vmult_add = [apply_storage, memory](GlobalVectorType       &dst,
                                                 const GlobalVectorType &src) {
        const auto &apply = *apply_storage;
        typename dealii::VectorMemory<GlobalVectorType>::Pointer tmp(*memory);
        tmp->reinit(dst);
        apply(*tmp, src, false);
        dst += *tmp;
      };
      result.Tvmult_add = [apply_storage, memory](GlobalVectorType       &dst,
                                                  const GlobalVectorType &src) {
        const auto &apply = *apply_storage;
        typename dealii::VectorMemory<GlobalVectorType>::Pointer tmp(*memory);
        tmp->reinit(dst);
        apply(*tmp, src, true);
        dst += *tmp;
      };
      return result;
    }

    std::shared_ptr<SecondOrderPreconditioner>
    second_order_inverse(const SecondOrderPair                    &pair,
                         const EvaluationContext<FieldVectorType> &context,
                         const double                              alpha,
                         const Model *prepared_model = nullptr) const
    {
      const auto &model = prepared_model ? *prepared_model : model_;
      AssertThrow(
        alpha >= 0.,
        dealii::ExcMessage(
          "Second-order Schur reduction requires a nonnegative derivative weight."));
      const auto owned =
        context.state(pair.displacement).locally_owned_elements();
      const auto same_partition = [&owned](const dealii::IndexSet &partition) {
        return owned.size() == partition.size() && owned == partition;
      };
      const bool partition_matches =
        same_partition(layout_.field(pair.displacement).locally_owned) &&
        same_partition(layout_.field(pair.velocity).locally_owned) &&
        same_partition(context.state(pair.velocity).locally_owned_elements());
      AssertThrow(dealii::Utilities::MPI::min(static_cast<unsigned int>(
                                                partition_matches),
                                              communicator_) == 1,
                  dealii::ExcMessage(
                    "A changed second-order field partition requires a new "
                    "execution composition."));
      const auto revision = pair.revision();
      auto      &cached   = second_order_preconditioners_[pair.displacement];
      if (!cached.inverse || cached.revision != revision)
        {
          const auto surrogate =
            model.second_order_preconditioner_operators(pair.displacement);
          const auto mass =
            surrogate ?
              std::optional<typename Model::MatrixOperator>((*surrogate)[0]) :
              model.derivative_matrix_operator(pair.displacement,
                                               pair.displacement,
                                               context);
          const auto stiffness =
            surrogate ?
              std::optional<typename Model::MatrixOperator>((*surrogate)[1]) :
              model.state_matrix_operator(pair.velocity,
                                          pair.displacement,
                                          context);
          const auto damping =
            surrogate ?
              std::optional<typename Model::MatrixOperator>((*surrogate)[2]) :
              model.state_matrix_operator(pair.velocity,
                                          pair.velocity,
                                          context);
          AssertThrow(mass && stiffness && damping,
                      dealii::ExcMessage("Second-order factorization requires "
                                         "matrix-backed M, K and D blocks."));
          const auto mass_matrix      = mass->matrix();
          const auto stiffness_matrix = stiffness->matrix();
          const auto damping_matrix   = damping->matrix();
          const auto matrix_matches =
            [&same_partition](const MatrixType &matrix) {
              return same_partition(matrix.locally_owned_range_indices()) &&
                     same_partition(matrix.locally_owned_domain_indices());
            };
          const bool spatial_partitions_match =
            matrix_matches(*mass_matrix) && matrix_matches(*stiffness_matrix) &&
            matrix_matches(*damping_matrix);
          AssertThrow(dealii::Utilities::MPI::min(static_cast<unsigned int>(
                                                    spatial_partitions_match),
                                                  communicator_) == 1,
                      dealii::ExcMessage(
                        "Changed second-order spatial operator partitions "
                        "require a new execution composition and state."));
          const auto data    = pair.preconditioner_data();
          auto       inverse = std::make_shared<SecondOrderPreconditioner>();
          inverse->initialize(
            *mass_matrix,
            *stiffness_matrix,
            *damping_matrix,
            data.constrained,
            [owned_storage =
               std::make_shared<const std::decay_t<decltype(owned)>>(owned),
             communicator = communicator_](FieldVectorType &vector,
                                           const bool) {
              const auto &owned = *owned_storage;
              vector.reinit(owned, communicator);
            },
            data.amg_data);
          cached.inverse  = std::move(inverse);
          cached.revision = revision;
        }
      if (alpha == 0.)
        cached.inverse->prepare_static_inverse();
      else
        cached.inverse->set_coefficients(alpha, 1., 1., 1.);
      return cached.inverse;
    }

    LocalOperator
    linearized_operator(const FieldId                             row,
                        const FieldId                             column,
                        const EvaluationContext<FieldVectorType> &context,
                        const double                              alpha,
                        const Model *prepared_model = nullptr) const
    {
      const auto   &model          = prepared_model ? *prepared_model : model_;
      const bool    has_state      = model.has_state_operator(row, column);
      const bool    has_derivative = model.has_derivative_operator(row, column);
      LocalOperator result;
      if (has_state)
        result = model.state_operator(row, column, context);
      if (alpha != 0. && has_derivative)
        {
          const auto derivative =
            model.derivative_operator(row, column, context);
          result = has_state ? result + alpha * derivative : alpha * derivative;
        }
      if (!has_state && (alpha == 0. || !has_derivative))
        result = model.state_operator(row, column, context);
      return result;
    }

    /** Return a safe inverse for the lumped multiplier metric. */
    static double
    inverse_lumped_metric_value(const double value)
    {
      AssertThrow(std::isfinite(value),
                  dealii::ExcMessage(
                    "The multiplier metric has an invalid diagonal entry."));
      // Constrained multiplier rows can have a zero physical mass diagonal.
      // Use the unit algebraic scale on those rows so the AL factor remains
      // invertible while retaining the physical metric wherever available.
      return std::abs(value) > 1.e-14 ? 1. / value : 1.;
    }

    MatrixType
    repartition_matrix(const MatrixType       &source,
                       const dealii::IndexSet &row_partition,
                       const dealii::IndexSet &column_partition) const
    {
      dealii::DynamicSparsityPattern sparsity(row_partition.size(),
                                              column_partition.size(),
                                              row_partition);
      for (const auto row : row_partition)
        if (source.in_local_range(row) && source.row_length(row) != 0)
          for (auto entry = source.begin(row), end = source.end(row);
               entry != end;
               ++entry)
            sparsity.add(row, entry->column());

      MatrixType result;
      result.reinit(
        row_partition, column_partition, sparsity, communicator_, false);
      for (const auto row : row_partition)
        if (source.in_local_range(row) && source.row_length(row) != 0)
          for (auto entry = source.begin(row), end = source.end(row);
               entry != end;
               ++entry)
            result.set(row, entry->column(), entry->value());
      result.compress(dealii::VectorOperation::insert);
      return result;
    }

    MatrixType
    add_matrices(const MatrixType       &base,
                 const MatrixType       &increment,
                 const double            factor,
                 const dealii::IndexSet &row_partition,
                 const dealii::IndexSet &column_partition) const
    {
      using size_type = typename MatrixType::size_type;
      dealii::DynamicSparsityPattern sparsity(row_partition.size(),
                                              column_partition.size(),
                                              row_partition);
      for (const auto row : row_partition)
        {
          if (base.row_length(row) != 0)
            for (auto entry = base.begin(row), end = base.end(row);
                 entry != end;
                 ++entry)
              sparsity.add(row, entry->column());
          if (increment.row_length(row) != 0)
            for (auto entry = increment.begin(row), end = increment.end(row);
                 entry != end;
                 ++entry)
              sparsity.add(row, entry->column());
        }

      MatrixType result;
      result.reinit(
        row_partition, column_partition, sparsity, communicator_, false);
      for (const auto row : row_partition)
        {
          std::map<size_type, double> values;
          if (base.row_length(row) != 0)
            for (auto entry = base.begin(row), end = base.end(row);
                 entry != end;
                 ++entry)
              values[entry->column()] += entry->value();
          if (increment.row_length(row) != 0)
            for (auto entry = increment.begin(row), end = increment.end(row);
                 entry != end;
                 ++entry)
              values[entry->column()] += factor * entry->value();
          for (const auto &[column, value] : values)
            result.set(row, column, value);
        }
      result.compress(dealii::VectorOperation::insert);
      return result;
    }

    bool
    can_materialize_matrix(const EvaluationContext<FieldVectorType> &context,
                           const double                              alpha,
                           const Model *prepared_model = nullptr) const
    {
      const auto &model = prepared_model ? *prepared_model : model_;
      for (unsigned int i = 0; i < field_layout_.n_blocks(); ++i)
        for (unsigned int j = 0; j < field_layout_.n_blocks(); ++j)
          {
            const auto row       = field_layout_.field(i);
            const auto column    = field_layout_.field(j);
            const bool has_state = model.has_state_operator(row, column);
            const bool has_derivative =
              model.has_derivative_operator(row, column);
            if (has_state &&
                !model.state_matrix_operator(row, column, context).has_value())
              return false;
            if (alpha != 0. && has_derivative &&
                !model.derivative_matrix_operator(row, column, context)
                   .has_value())
              return false;
          }
      return true;
    }

    std::shared_ptr<MatrixType>
    materialized_block(const FieldId                             row,
                       const FieldId                             column,
                       const EvaluationContext<FieldVectorType> &context,
                       const double                              alpha,
                       const Model *prepared_model = nullptr) const
    {
      const auto &model          = prepared_model ? *prepared_model : model_;
      const bool  has_state      = model.has_state_operator(row, column);
      const bool  has_derivative = model.has_derivative_operator(row, column);
      if (!has_state && (alpha == 0. || !has_derivative))
        return {};

      std::optional<typename Model::MatrixOperator> state_matrix;
      std::optional<typename Model::MatrixOperator> derivative_matrix;
      if (has_state)
        state_matrix = model.state_matrix_operator(row, column, context);
      if (alpha != 0. && has_derivative)
        derivative_matrix =
          model.derivative_matrix_operator(row, column, context);
      AssertThrow(!has_state || state_matrix.has_value(),
                  dealii::ExcMessage(
                    "A state operator is not completely matrix-based."));
      AssertThrow(alpha == 0. || !has_derivative ||
                    derivative_matrix.has_value(),
                  dealii::ExcMessage(
                    "A derivative operator is not completely matrix-based."));

      std::vector<typename Model::MatrixOperator> matrices;
      if (state_matrix.has_value())
        matrices.push_back(*state_matrix);
      if (derivative_matrix.has_value())
        matrices.push_back(alpha * *derivative_matrix);
      if (matrices.empty())
        return {};
      if (matrices.size() == 1)
        return matrices.front().matrix();
      return detail::sum_matrices(matrices);
    }

    BlockMatrixType
    make_block_matrix(const EvaluationContext<FieldVectorType> &context,
                      const double                              alpha,
                      const Model *prepared_model = nullptr) const
    {
      AssertThrow(can_materialize_matrix(context, alpha, prepared_model),
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
                                             alpha,
                                             prepared_model);
            auto &block  = result.block(i, j);
            if (matrix)
              {
                AssertThrow(matrix->m() == partitions[i].size() &&
                              matrix->n() == partitions[j].size(),
                            dealii::ExcMessage(
                              "Materialized block dimensions do not match "
                              "the semantic execution layout."));
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
                         const double                              alpha,
                         const Model *prepared_model = nullptr) const
    {
      const auto        &model = prepared_model ? *prepared_model : model_;
      const unsigned int n     = field_layout_.n_blocks();
      std::vector<std::vector<typename Model::Operator>> blocks(
        n, std::vector<typename Model::Operator>(n));
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          blocks[i][j] = zero_operator(context.state(field_layout_.field(i)),
                                       context.state(field_layout_.field(j)));
      for (const auto &[row, column] : model.active_operator_blocks())
        {
          const auto i         = field_layout_.block(row);
          const auto j         = field_layout_.block(column);
          const bool has_state = model.has_state_operator(row, column);
          const bool has_derivative =
            model.has_derivative_operator(row, column);
          if (has_state)
            blocks[i][j] = model.state_operator(row, column, context);
          if (alpha != 0. && has_derivative)
            {
              const auto derivative =
                alpha * model.derivative_operator(row, column, context);
              blocks[i][j] = has_state ? blocks[i][j] + derivative : derivative;
            }
        }

      const auto blocks_storage =
        std::make_shared<const decltype(blocks)>(std::move(blocks));
      Operator   result;
      const auto partitions =
        field_layout_.block_partitions(context.state(), context.time());
      const auto partitions_storage =
        std::make_shared<const decltype(partitions)>(partitions);
      result.reinit_range_vector =
        [this, partitions_storage](GlobalVectorType &vector, bool) {
          const auto &partitions = *partitions_storage;
          vector.reinit(partitions, communicator_);
        };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult = [blocks_storage](GlobalVectorType       &destination,
                                      const GlobalVectorType &source) {
        const auto &blocks = *blocks_storage;
        destination        = 0.;
        for (unsigned int i = 0; i < blocks.size(); ++i)
          for (unsigned int j = 0; j < blocks[i].size(); ++j)
            blocks[i][j].vmult_add(destination.block(i), source.block(j));
      };
      result.vmult_add = [blocks_storage](GlobalVectorType       &destination,
                                          const GlobalVectorType &source) {
        const auto &blocks = *blocks_storage;
        for (unsigned int i = 0; i < blocks.size(); ++i)
          for (unsigned int j = 0; j < blocks[i].size(); ++j)
            blocks[i][j].vmult_add(destination.block(i), source.block(j));
      };
      result.Tvmult = [blocks_storage](GlobalVectorType       &destination,
                                       const GlobalVectorType &source) {
        const auto &blocks = *blocks_storage;
        destination        = 0.;
        for (unsigned int i = 0; i < blocks.size(); ++i)
          for (unsigned int j = 0; j < blocks[i].size(); ++j)
            blocks[i][j].Tvmult_add(destination.block(j), source.block(i));
      };
      result.Tvmult_add = [blocks_storage](GlobalVectorType       &destination,
                                           const GlobalVectorType &source) {
        const auto &blocks = *blocks_storage;
        for (unsigned int i = 0; i < blocks.size(); ++i)
          for (unsigned int j = 0; j < blocks[i].size(); ++j)
            blocks[i][j].Tvmult_add(destination.block(j), source.block(i));
      };
      return result;
    }

    static typename Model::Operator
    zero_operator(const FieldVectorType &range, const FieldVectorType &domain)
    {
      const auto range_storage = std::make_shared<FieldVectorType>();
      range_storage->reinit(range, true);
      const auto domain_storage = std::make_shared<FieldVectorType>();
      domain_storage->reinit(domain, true);
      typename Model::Operator result;
      result.reinit_range_vector = [range_storage](FieldVectorType &vector,
                                                   const bool       omit) {
        vector.reinit(*range_storage, omit);
      };
      result.reinit_domain_vector = [domain_storage](FieldVectorType &vector,
                                                     const bool       omit) {
        vector.reinit(*domain_storage, omit);
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
    struct CachedSecondOrderPreconditioner
    {
      unsigned long                              revision = 0;
      std::shared_ptr<SecondOrderPreconditioner> inverse;
    };
    mutable std::map<FieldId, CachedSecondOrderPreconditioner>
      second_order_preconditioners_;
  };
} // namespace ImmersX::detail

#endif // immersx_detail_execution_composition_h
