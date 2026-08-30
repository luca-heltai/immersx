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
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/vector_memory.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/contributor.h>

#include <algorithm>
#include <cmath>
#include <map>
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
    using Model               = SemiDiscreteModel<FieldVectorType>;
    using Builder             = SemidiscreteBuilder<FieldVectorType>;
    using Operator            = dealii::LinearOperator<GlobalVectorType>;
    using LocalOperator       = typename Model::Operator;
    using MatrixType          = typename Model::MatrixOperator::Matrix;
    using SaddlePointMetadata = typename Model::SaddlePointMetadata;
    using BlockMatrixType     = ImmersXLA::MPI::BlockSparseMatrix;

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
    void
    monolithic_matrix(const BlockMatrixType &block_matrix,
                      MatrixType            &result) const
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

    const std::vector<SaddlePointMetadata> &
    saddle_points() const
    {
      finalize();
      return model_.saddle_points();
    }

    /** Build S ~= sum_i B_i P_i^-1 B_i^T on a semantic multiplier field. */
    LocalOperator
    schur_operator(const FieldId           multiplier,
                   const GlobalVectorType &state,
                   const GlobalVectorType *state_dot = nullptr,
                   const double            alpha     = 0.) const
    {
      finalize();
      validate_state(state);
      const auto metadata =
        std::find_if(model_.saddle_points().begin(),
                     model_.saddle_points().end(),
                     [multiplier](const auto &entry) {
                       return entry.multiplier == multiplier;
                     });
      AssertThrow(metadata != model_.saddle_points().end(),
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
      const auto &context = *derivative_context;

      struct SchurTerm
      {
        LocalOperator coupling;
        LocalOperator transpose_coupling;
        LocalOperator inverse;
      };
      std::vector<SchurTerm> terms;
      terms.reserve(metadata->participants.size());
      for (const auto participant : metadata->participants)
        {
          const auto inverse =
            local_preconditioner(participant, state, context, alpha);
          AssertThrow(inverse.has_value(),
                      dealii::ExcMessage(
                        "Schur construction requires a local inverse for "
                        "every participant field."));
          terms.push_back(
            {model_.state_operator(multiplier, participant, context),
             model_.state_operator(participant, multiplier, context),
             *inverse});
        }

      const auto multiplier_owned = layout_.field(multiplier).locally_owned;
      const auto communicator     = communicator_;
      auto       vector_memory =
        std::make_shared<dealii::GrowingVectorMemory<FieldVectorType>>();
      LocalOperator result;
      result.reinit_range_vector =
        [multiplier_owned, communicator](FieldVectorType &vector, const bool) {
          vector.reinit(multiplier_owned, communicator);
        };
      result.reinit_domain_vector = result.reinit_range_vector;
      auto apply = [terms, vector_memory](FieldVectorType       &dst,
                                          const FieldVectorType &src,
                                          const bool             transpose) {
        dst = 0.;
        for (const auto &term : terms)
          {
            typename dealii::VectorMemory<FieldVectorType>::Pointer
              participant_rhs(*vector_memory);
            typename dealii::VectorMemory<FieldVectorType>::Pointer
              participant_solution(*vector_memory);
            if (!transpose)
              {
                term.transpose_coupling.reinit_range_vector(*participant_rhs,
                                                            false);
                term.transpose_coupling.vmult(*participant_rhs, src);
                term.inverse.reinit_range_vector(*participant_solution, false);
                term.inverse.vmult(*participant_solution, *participant_rhs);
                term.coupling.vmult_add(dst, *participant_solution);
              }
            else
              {
                term.coupling.reinit_domain_vector(*participant_rhs, false);
                term.coupling.Tvmult(*participant_rhs, src);
                term.inverse.reinit_domain_vector(*participant_solution, false);
                term.inverse.Tvmult(*participant_solution, *participant_rhs);
                term.transpose_coupling.Tvmult_add(dst, *participant_solution);
              }
          }
      };
      result.vmult = [apply](FieldVectorType &dst, const FieldVectorType &src) {
        apply(dst, src, false);
      };
      result.vmult_add = [apply, vector_memory](FieldVectorType       &dst,
                                                const FieldVectorType &src) {
        typename dealii::VectorMemory<FieldVectorType>::Pointer contribution(
          *vector_memory);
        contribution->reinit(dst);
        apply(*contribution, src, false);
        dst += *contribution;
      };
      result.Tvmult = [apply](FieldVectorType       &dst,
                              const FieldVectorType &src) {
        apply(dst, src, true);
      };
      result.Tvmult_add = [apply, vector_memory](FieldVectorType       &dst,
                                                 const FieldVectorType &src) {
        typename dealii::VectorMemory<FieldVectorType>::Pointer contribution(
          *vector_memory);
        contribution->reinit(dst);
        apply(*contribution, src, true);
        dst += *contribution;
      };
      return result;
    }

    /** Build a global block-factorization preconditioner using a Schur solve.
     */
    Operator
    schur_preconditioner(const FieldId           multiplier,
                         const GlobalVectorType &state,
                         const GlobalVectorType *state_dot          = nullptr,
                         const double            alpha              = 0.,
                         const unsigned int      maximum_iterations = 1000,
                         const double            tolerance = 1.e-10) const
    {
      finalize();
      validate_state(state);
      const auto metadata =
        std::find_if(model_.saddle_points().begin(),
                     model_.saddle_points().end(),
                     [multiplier](const auto &entry) {
                       return entry.multiplier == multiplier;
                     });
      AssertThrow(metadata != model_.saddle_points().end(),
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
      const auto &context = *derivative_context;
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
          const auto inverse =
            local_preconditioner(field, state, context, alpha);
          AssertThrow(inverse.has_value(),
                      dealii::ExcMessage(
                        "Schur preconditioning requires local inverses for "
                        "all participant fields."));
          participants.push_back(
            {field_layout_.block(field),
             model_.state_operator(multiplier, field, context),
             model_.state_operator(field, multiplier, context),
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

          const auto inverse =
            local_preconditioner(field, state, context, alpha);
          AssertThrow(inverse.has_value(),
                      dealii::ExcMessage(
                        "Schur preconditioning requires local inverses for "
                        "all non-participant fields."));
          non_participants.push_back({block, *inverse});
        }

      const auto schur = schur_operator(multiplier, state, state_dot, alpha);
      const auto multiplier_block = field_layout_.block(multiplier);
      const auto multiplier_preconditioner =
        this->multiplier_preconditioner(multiplier, state, context);
      auto field_vector_memory =
        std::make_shared<dealii::GrowingVectorMemory<FieldVectorType>>();

      auto transpose_operator = [](const LocalOperator &operator_to_transpose) {
        LocalOperator result = operator_to_transpose;
        result.vmult         = operator_to_transpose.Tvmult;
        result.vmult_add     = operator_to_transpose.Tvmult_add;
        result.Tvmult        = operator_to_transpose.vmult;
        result.Tvmult_add    = operator_to_transpose.vmult_add;
        std::swap(result.reinit_range_vector, result.reinit_domain_vector);
        return result;
      };
      const auto schur_transpose = transpose_operator(schur);
      const auto multiplier_preconditioner_transpose =
        transpose_operator(multiplier_preconditioner);

      auto solve_multiplier = [schur,
                               schur_transpose,
                               multiplier_preconditioner,
                               multiplier_preconditioner_transpose,
                               maximum_iterations,
                               tolerance](FieldVectorType       &solution,
                                          const FieldVectorType &rhs,
                                          const bool             transpose) {
        dealii::SolverControl control(maximum_iterations, tolerance);
        dealii::SolverGMRES<FieldVectorType> solver(control);
        if (transpose)
          solver.solve(schur_transpose,
                       solution,
                       rhs,
                       multiplier_preconditioner_transpose);
        else
          solver.solve(schur, solution, rhs, multiplier_preconditioner);
      };

      auto apply = [participants,
                    non_participants,
                    multiplier_block,
                    field_vector_memory,
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

        typename dealii::VectorMemory<FieldVectorType>::Pointer schur_rhs(
          *field_vector_memory);
        typename dealii::VectorMemory<FieldVectorType>::Pointer
          multiplier_solution(*field_vector_memory);
        if (transpose)
          {
            participants.front().from_multiplier.reinit_domain_vector(
              *schur_rhs, false);
            *schur_rhs = src.block(multiplier_block);
            *schur_rhs *= -1.;

            for (const auto &participant : participants)
              {
                typename dealii::VectorMemory<FieldVectorType>::Pointer
                  participant_solution(*field_vector_memory);
                participant.inverse.reinit_domain_vector(*participant_solution,
                                                         false);
                participant.inverse.Tvmult(*participant_solution,
                                           src.block(participant.block));
                participant.from_multiplier.Tvmult_add(*schur_rhs,
                                                       *participant_solution);
              }
          }
        else
          {
            participants.front().to_multiplier.reinit_range_vector(*schur_rhs,
                                                                   false);
            *schur_rhs = src.block(multiplier_block);
            *schur_rhs *= -1.;

            for (const auto &participant : participants)
              {
                typename dealii::VectorMemory<FieldVectorType>::Pointer
                  participant_solution(*field_vector_memory);
                participant.inverse.reinit_range_vector(*participant_solution,
                                                        false);
                participant.inverse.vmult(*participant_solution,
                                          src.block(participant.block));
                participant.to_multiplier.vmult_add(*schur_rhs,
                                                    *participant_solution);
                dst.block(participant.block) = *participant_solution;
              }
          }

        participants.front().to_multiplier.reinit_range_vector(
          *multiplier_solution, false);
        *multiplier_solution = 0.;
        solve_multiplier(*multiplier_solution, *schur_rhs, transpose);
        dst.block(multiplier_block) = *multiplier_solution;

        for (const auto &participant : participants)
          {
            typename dealii::VectorMemory<FieldVectorType>::Pointer
              participant_rhs(*field_vector_memory);
            typename dealii::VectorMemory<FieldVectorType>::Pointer
              participant_solution(*field_vector_memory);
            if (transpose)
              {
                participant.to_multiplier.reinit_domain_vector(*participant_rhs,
                                                               false);
                participant.to_multiplier.Tvmult(*participant_rhs,
                                                 *multiplier_solution);
                *participant_rhs *= -1.;
                *participant_rhs += src.block(participant.block);
                participant.inverse.reinit_domain_vector(*participant_solution,
                                                         false);
                participant.inverse.Tvmult(*participant_solution,
                                           *participant_rhs);
              }
            else
              {
                participant.from_multiplier.reinit_range_vector(
                  *participant_rhs, false);
                participant.from_multiplier.vmult(*participant_rhs,
                                                  *multiplier_solution);
                participant.inverse.reinit_range_vector(*participant_solution,
                                                        false);
                participant.inverse.vmult(*participant_solution,
                                          *participant_rhs);
                *participant_solution *= -1.;
                dst.block(participant.block) += *participant_solution;
                continue;
              }
            dst.block(participant.block) = *participant_solution;
          }
      };

      auto global_vector_memory =
        std::make_shared<dealii::GrowingVectorMemory<GlobalVectorType>>();
      Operator   result;
      const auto partitions   = field_layout_.block_partitions();
      const auto communicator = communicator_;
      result.reinit_range_vector =
        [partitions, communicator](GlobalVectorType &vector, const bool) {
          vector.reinit(partitions, communicator);
        };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult                = [apply](GlobalVectorType       &dst,
                             const GlobalVectorType &src) {
        apply(dst, src, false);
      };
      result.vmult_add = [apply,
                          global_vector_memory](GlobalVectorType       &dst,
                                                const GlobalVectorType &src) {
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *global_vector_memory);
        contribution->reinit(dst);
        apply(*contribution, src, false);
        dst += *contribution;
      };
      result.Tvmult = [apply](GlobalVectorType       &dst,
                              const GlobalVectorType &src) {
        apply(dst, src, true);
      };
      result.Tvmult_add = [apply,
                           global_vector_memory](GlobalVectorType       &dst,
                                                 const GlobalVectorType &src) {
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *global_vector_memory);
        contribution->reinit(dst);
        apply(*contribution, src, true);
        dst += *contribution;
      };
      return result;
    }

    /** Apply A_gamma = A + gamma B^T W^-1 B on the coupled state. */
    Operator
    augmented_lagrangian_operator(const GlobalVectorType &state,
                                  const double            gamma = 1.e1) const
    {
      finalize();
      validate_state(state);
      AssertThrow(gamma > 0.,
                  dealii::ExcMessage(
                    "The augmented-Lagrangian parameter must be positive."));

      StateView<FieldVectorType> state_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      EvaluationContext<FieldVectorType> context(0., state_view, nullptr);
      const unsigned int                 n = field_layout_.n_blocks();

      std::vector<std::vector<LocalOperator>> base(
        n, std::vector<LocalOperator>(n));
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          base[i][j] = model_.state_operator(field_layout_.field(i),
                                             field_layout_.field(j),
                                             context);

      struct Augmentation
      {
        unsigned int               multiplier;
        std::vector<unsigned int>  participants;
        std::vector<LocalOperator> to_multiplier;
        std::vector<LocalOperator> from_multiplier;
        LocalOperator              inverse_metric;
      };
      std::vector<Augmentation> augmentations;
      augmentations.reserve(model_.saddle_points().size());
      for (const auto &metadata : model_.saddle_points())
        {
          const auto multiplier = field_layout_.block(metadata.multiplier);
          const auto metric =
            model_.multiplier_metric(metadata.multiplier, context);
          LocalOperator inverse_metric_operator;
          if (metric.has_value())
            {
              const auto metric_matrix = metric->matrix();
              const auto multiplier_partition =
                layout_.field(metadata.multiplier).locally_owned;
              auto inverse_metric = std::make_shared<FieldVectorType>();
              inverse_metric->reinit(state.block(multiplier));
              for (const auto index : multiplier_partition)
                {
                  const auto value         = metric_matrix->diag_element(index);
                  (*inverse_metric)(index) = inverse_lumped_metric_value(value);
                }
              inverse_metric->compress(dealii::VectorOperation::insert);

              inverse_metric_operator.reinit_range_vector =
                [prototype = state.block(multiplier)](FieldVectorType &vector,
                                                      const bool       omit) {
                  vector.reinit(prototype, omit);
                };
              inverse_metric_operator.reinit_domain_vector =
                inverse_metric_operator.reinit_range_vector;
              inverse_metric_operator.vmult =
                [inverse_metric,
                 multiplier_partition](FieldVectorType       &dst,
                                       const FieldVectorType &src) {
                  dst = src;
                  for (const auto index : multiplier_partition)
                    dst(index) = (*inverse_metric)(index)*src(index);
                };
              inverse_metric_operator.vmult_add =
                [inverse_metric,
                 multiplier_partition](FieldVectorType       &dst,
                                       const FieldVectorType &src) {
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

          Augmentation augmentation{
            multiplier, {}, {}, {}, inverse_metric_operator};
          for (const auto participant : metadata.participants)
            {
              augmentation.participants.push_back(
                field_layout_.block(participant));
              augmentation.to_multiplier.push_back(model_.state_operator(
                metadata.multiplier, participant, context));
              augmentation.from_multiplier.push_back(model_.state_operator(
                participant, metadata.multiplier, context));
            }
          augmentations.push_back(std::move(augmentation));
        }

      auto field_vector_memory =
        std::make_shared<dealii::GrowingVectorMemory<FieldVectorType>>();
      auto apply = [base,
                    augmentations,
                    gamma,
                    field_vector_memory](GlobalVectorType       &dst,
                                         const GlobalVectorType &src,
                                         const bool              transpose) {
        const unsigned int n = base.size();
        for (unsigned int i = 0; i < n; ++i)
          {
            dst.block(i) = 0.;
            for (unsigned int j = 0; j < n; ++j)
              if (transpose)
                base[j][i].Tvmult_add(dst.block(i), src.block(j));
              else
                base[i][j].vmult_add(dst.block(i), src.block(j));
          }

        for (const auto &augmentation : augmentations)
          {
            typename dealii::VectorMemory<FieldVectorType>::Pointer
              constraint_value(*field_vector_memory);
            typename dealii::VectorMemory<FieldVectorType>::Pointer
              weighted_constraint(*field_vector_memory);
            if (transpose)
              {
                augmentation.from_multiplier[0].reinit_domain_vector(
                  *constraint_value, false);
                *constraint_value = 0.;
                for (unsigned int i = 0; i < augmentation.participants.size();
                     ++i)
                  augmentation.from_multiplier[i].Tvmult_add(
                    *constraint_value, src.block(augmentation.participants[i]));
                augmentation.inverse_metric.reinit_domain_vector(
                  *weighted_constraint, false);
                augmentation.inverse_metric.Tvmult(*weighted_constraint,
                                                   *constraint_value);
                weighted_constraint->operator*=(gamma);
                for (unsigned int i = 0; i < augmentation.participants.size();
                     ++i)
                  augmentation.to_multiplier[i].Tvmult_add(
                    dst.block(augmentation.participants[i]),
                    *weighted_constraint);
              }
            else
              {
                augmentation.to_multiplier[0].reinit_range_vector(
                  *constraint_value, false);
                *constraint_value = 0.;
                for (unsigned int i = 0; i < augmentation.participants.size();
                     ++i)
                  augmentation.to_multiplier[i].vmult_add(
                    *constraint_value, src.block(augmentation.participants[i]));
                augmentation.inverse_metric.reinit_range_vector(
                  *weighted_constraint, false);
                augmentation.inverse_metric.vmult(*weighted_constraint,
                                                  *constraint_value);
                weighted_constraint->operator*=(gamma);
                for (unsigned int i = 0; i < augmentation.participants.size();
                     ++i)
                  augmentation.from_multiplier[i].vmult_add(
                    dst.block(augmentation.participants[i]),
                    *weighted_constraint);
              }
          }
      };

      auto vector_memory =
        std::make_shared<dealii::GrowingVectorMemory<GlobalVectorType>>();
      const auto partitions   = field_layout_.block_partitions();
      const auto communicator = communicator_;
      Operator   result;
      result.reinit_range_vector =
        [partitions, communicator](GlobalVectorType &vector, const bool) {
          vector.reinit(partitions, communicator);
        };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult                = [apply](GlobalVectorType       &dst,
                             const GlobalVectorType &src) {
        apply(dst, src, false);
      };
      result.vmult_add = [apply, vector_memory](GlobalVectorType       &dst,
                                                const GlobalVectorType &src) {
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *vector_memory);
        contribution->reinit(dst);
        apply(*contribution, src, false);
        dst += *contribution;
      };
      result.Tvmult = [apply](GlobalVectorType       &dst,
                              const GlobalVectorType &src) {
        apply(dst, src, true);
      };
      result.Tvmult_add = [apply, vector_memory](GlobalVectorType       &dst,
                                                 const GlobalVectorType &src) {
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *vector_memory);
        contribution->reinit(dst);
        apply(*contribution, src, true);
        dst += *contribution;
      };
      return result;
    }

    /**
     * Materialize the matrix A + gamma B^T diag(W^{-1}) B.
     *
     * The multiplier metric is intentionally reduced to its diagonal here:
     * this is the matrix-based counterpart of the lumped metric used by the
     * first AL implementation.  Each participant pair is assembled, so
     * augmentation cross terms are retained when a relation has multiple
     * primal fields.
     */
    BlockMatrixType
    augmented_lagrangian_matrix(const GlobalVectorType &state,
                                const double            gamma = 1.e1) const
    {
      finalize();
      validate_state(state);
      AssertThrow(gamma > 0.,
                  dealii::ExcMessage(
                    "The augmented-Lagrangian parameter must be positive."));

      StateView<FieldVectorType> state_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      EvaluationContext<FieldVectorType> context(0., state_view, nullptr);
      auto       result     = make_block_matrix(context, 0.);
      const auto partitions = field_layout_.block_partitions();

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

          for (std::size_t i = 0; i < metadata.participants.size(); ++i)
            for (std::size_t j = 0; j < metadata.participants.size(); ++j)
              {
                const auto row_field = metadata.participants[i];
                const auto col_field = metadata.participants[j];
                const auto from      = materialized_block(row_field,
                                                     metadata.multiplier,
                                                     context,
                                                     0.);
                const auto to        = materialized_block(metadata.multiplier,
                                                   col_field,
                                                   context,
                                                   0.);
                AssertThrow(from && to,
                            dealii::ExcMessage(
                              "Matrix-based augmented-Lagrangian composition "
                              "requires materialized coupling blocks."));
                AssertThrow(from->n() == to->m(),
                            dealii::ExcDimensionMismatch(from->n(), to->m()));
                MatrixType product;
                const auto column_partition =
                  layout_.field(col_field).locally_owned;
                MatrixType from_normalized;
                MatrixType to_normalized;
                repartition_matrix(*from,
                                   layout_.field(row_field).locally_owned,
                                   multiplier_partition,
                                   from_normalized);
                repartition_matrix(*to,
                                   multiplier_partition,
                                   column_partition,
                                   to_normalized);
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
    augmented_lagrangian_preconditioner(const GlobalVectorType &state,
                                        const double gamma = 1.e1) const
    {
      finalize();
      validate_state(state);
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
      EvaluationContext<FieldVectorType> context(0., state_view, nullptr);
      auto                               field_vector_memory =
        std::make_shared<dealii::GrowingVectorMemory<FieldVectorType>>();
      std::vector<LocalOperator> diagonal(field_layout_.n_blocks());
      for (unsigned int block = 0; block < field_layout_.n_blocks(); ++block)
        {
          if (block == multiplier_block)
            {
              diagonal[block] =
                multiplier_preconditioner(metadata.multiplier, state, context);
              const auto multiplier_scale    = gamma;
              const auto multiplier_operator = diagonal[block];
              diagonal[block].vmult =
                [multiplier_operator,
                 multiplier_scale](FieldVectorType       &dst,
                                   const FieldVectorType &src) {
                  multiplier_operator.vmult(dst, src);
                  dst *= -multiplier_scale;
                };
              diagonal[block].vmult_add =
                [multiplier_operator, multiplier_scale, field_vector_memory](
                  FieldVectorType &dst, const FieldVectorType &src) {
                  typename dealii::VectorMemory<FieldVectorType>::Pointer
                    contribution(*field_vector_memory);
                  multiplier_operator.reinit_range_vector(*contribution, false);
                  multiplier_operator.vmult(*contribution, src);
                  *contribution *= -multiplier_scale;
                  dst += *contribution;
                };
              diagonal[block].Tvmult =
                [multiplier_operator,
                 multiplier_scale](FieldVectorType       &dst,
                                   const FieldVectorType &src) {
                  multiplier_operator.Tvmult(dst, src);
                  dst *= -multiplier_scale;
                };
              diagonal[block].Tvmult_add =
                [multiplier_operator, multiplier_scale, field_vector_memory](
                  FieldVectorType &dst, const FieldVectorType &src) {
                  typename dealii::VectorMemory<FieldVectorType>::Pointer
                    contribution(*field_vector_memory);
                  multiplier_operator.reinit_range_vector(*contribution, false);
                  multiplier_operator.Tvmult(*contribution, src);
                  *contribution *= -multiplier_scale;
                  dst += *contribution;
                };
            }
          else
            {
              const auto inverse = local_preconditioner(
                field_layout_.field(block), state, context, 0.);
              AssertThrow(inverse.has_value(),
                          dealii::ExcMessage(
                            "Augmented-Lagrangian preconditioning requires "
                            "a local preconditioner for every primal field."));
              diagonal[block] = *inverse;
            }
        }

      auto vector_memory =
        std::make_shared<dealii::GrowingVectorMemory<GlobalVectorType>>();
      const auto partitions   = field_layout_.block_partitions();
      const auto communicator = communicator_;
      Operator   result;
      result.reinit_range_vector =
        [partitions, communicator](GlobalVectorType &vector, const bool) {
          vector.reinit(partitions, communicator);
        };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult                = [diagonal](GlobalVectorType       &dst,
                                const GlobalVectorType &src) {
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].vmult(dst.block(block), src.block(block));
      };
      result.vmult_add = [diagonal,
                          vector_memory](GlobalVectorType       &dst,
                                         const GlobalVectorType &src) {
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *vector_memory);
        contribution->reinit(dst);
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].vmult(contribution->block(block), src.block(block));
        dst += *contribution;
      };
      result.Tvmult = [diagonal](GlobalVectorType       &dst,
                                 const GlobalVectorType &src) {
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].Tvmult(dst.block(block), src.block(block));
      };
      result.Tvmult_add = [diagonal,
                           vector_memory](GlobalVectorType       &dst,
                                          const GlobalVectorType &src) {
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *vector_memory);
        contribution->reinit(dst);
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].Tvmult(contribution->block(block), src.block(block));
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
                         const double                              alpha) const
    {
      finalize();
      validate_state(state);
      if (!model_.has_preconditioner(field))
        return std::nullopt;

      std::shared_ptr<MatrixType> matrix;
      if (model_.has_state_operator(field, field) ||
          (alpha != 0. && model_.has_derivative_operator(field, field)))
        {
          matrix = std::make_shared<MatrixType>();
          materialized_block(field, field, context, alpha, *matrix);
        }
      if (matrix)
        {
          const auto owned        = layout_.field(field).locally_owned;
          const auto communicator = communicator_;
          const auto reinit_vector =
            [owned, communicator](FieldVectorType &vector, const bool) {
              vector.reinit(owned, communicator);
            };
          const auto local =
            model_.preconditioner(field, *matrix, reinit_vector);
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
      const auto local =
        model_.preconditioner(field,
                              *empty_matrix,
                              [partition, communicator = communicator_](
                                FieldVectorType &vector, const bool) {
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
    multiplier_preconditioner(
      const FieldId                             multiplier,
      const GlobalVectorType                   &state,
      const EvaluationContext<FieldVectorType> &context) const
    {
      const auto prototype = state.block(field_layout_.block(multiplier));
      const auto metric    = model_.multiplier_metric(multiplier, context);
      if (metric.has_value())
        {
          const auto metric_matrix = metric->matrix();
          if (model_.has_preconditioner(multiplier))
            {
              const auto owned        = layout_.field(multiplier).locally_owned;
              const auto communicator = communicator_;
              const auto reinit_vector =
                [owned, communicator](FieldVectorType &vector, const bool) {
                  vector.reinit(owned, communicator);
                };
              const auto local = model_.preconditioner(multiplier,
                                                       *metric_matrix,
                                                       reinit_vector);
              if (local.has_value())
                {
                  auto       result     = *local;
                  const auto vmult      = result.vmult;
                  const auto vmult_add  = result.vmult_add;
                  const auto Tvmult     = result.Tvmult;
                  const auto Tvmult_add = result.Tvmult_add;
                  result.vmult          = [metric_matrix,
                                  vmult](FieldVectorType       &dst,
                                         const FieldVectorType &src) {
                    vmult(dst, src);
                  };
                  result.vmult_add = [metric_matrix,
                                      vmult_add](FieldVectorType       &dst,
                                                 const FieldVectorType &src) {
                    vmult_add(dst, src);
                  };
                  result.Tvmult = [metric_matrix,
                                   Tvmult](FieldVectorType       &dst,
                                           const FieldVectorType &src) {
                    Tvmult(dst, src);
                  };
                  result.Tvmult_add = [metric_matrix,
                                       Tvmult_add](FieldVectorType       &dst,
                                                   const FieldVectorType &src) {
                    Tvmult_add(dst, src);
                  };
                  return result;
                }
            }

          auto inverse_metric = std::make_shared<FieldVectorType>();
          inverse_metric->reinit(prototype);
          for (const auto index : prototype.locally_owned_elements())
            (*inverse_metric)(index) =
              inverse_lumped_metric_value(metric_matrix->diag_element(index));
          inverse_metric->compress(dealii::VectorOperation::insert);

          LocalOperator result;
          result.reinit_range_vector = [prototype](FieldVectorType &vector,
                                                   const bool       omit) {
            vector.reinit(prototype, omit);
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

      if (const auto local = local_preconditioner(multiplier, state);
          local.has_value())
        return *local;

      return dealii::identity_operator<FieldVectorType>(
        [prototype](FieldVectorType &vector, const bool omit) {
          vector.reinit(prototype, omit);
        });
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

      Operator   result;
      const auto partitions   = field_layout_.block_partitions();
      const auto communicator = communicator_;
      result.reinit_range_vector =
        [partitions, communicator](GlobalVectorType &vector, const bool) {
          vector.reinit(partitions, communicator);
        };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult                = [diagonal](GlobalVectorType       &dst,
                                const GlobalVectorType &src) {
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].vmult(dst.block(block), src.block(block));
      };
      result.vmult_add = [diagonal](GlobalVectorType       &dst,
                                    const GlobalVectorType &src) {
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].vmult_add(dst.block(block), src.block(block));
      };
      result.Tvmult = [diagonal](GlobalVectorType       &dst,
                                 const GlobalVectorType &src) {
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].Tvmult(dst.block(block), src.block(block));
      };
      result.Tvmult_add = [diagonal](GlobalVectorType       &dst,
                                     const GlobalVectorType &src) {
        for (unsigned int block = 0; block < diagonal.size(); ++block)
          diagonal[block].Tvmult_add(dst.block(block), src.block(block));
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

      auto substitution =
        [diagonal, off_diagonal, lower](GlobalVectorType       &dst,
                                        const GlobalVectorType &src,
                                        const bool              transpose) {
          const unsigned int n       = diagonal.size();
          const bool         forward = lower != transpose;
          if (forward)
            for (unsigned int i = 0; i < n; ++i)
              {
                dst.block(i) = src.block(i);
                dst.block(i) *= -1.;
                for (unsigned int j = 0; j < i; ++j)
                  if (transpose)
                    off_diagonal[j][i].Tvmult_add(dst.block(i), dst.block(j));
                  else
                    off_diagonal[i][j].vmult_add(dst.block(i), dst.block(j));
                dst.block(i) *= -1.;
                if (transpose)
                  diagonal[i].Tvmult(dst.block(i), dst.block(i));
                else
                  diagonal[i].vmult(dst.block(i), dst.block(i));
              }
          else
            for (int i = static_cast<int>(n) - 1; i >= 0; --i)
              {
                dst.block(i) = src.block(i);
                dst.block(i) *= -1.;
                for (unsigned int j = i + 1; j < n; ++j)
                  if (transpose)
                    off_diagonal[j][i].Tvmult_add(dst.block(i), dst.block(j));
                  else
                    off_diagonal[i][j].vmult_add(dst.block(i), dst.block(j));
                dst.block(i) *= -1.;
                if (transpose)
                  diagonal[i].Tvmult(dst.block(i), dst.block(i));
                else
                  diagonal[i].vmult(dst.block(i), dst.block(i));
              }
        };
      auto vector_memory =
        std::make_shared<dealii::GrowingVectorMemory<GlobalVectorType>>();

      Operator   result;
      const auto partitions   = field_layout_.block_partitions();
      const auto communicator = communicator_;
      result.reinit_range_vector =
        [partitions, communicator](GlobalVectorType &vector, const bool) {
          vector.reinit(partitions, communicator);
        };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult                = [substitution](GlobalVectorType       &dst,
                                    const GlobalVectorType &src) {
        substitution(dst, src, false);
      };
      result.vmult_add = [substitution,
                          vector_memory](GlobalVectorType       &dst,
                                         const GlobalVectorType &src) {
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *vector_memory);
        contribution->reinit(dst);
        substitution(*contribution, src, false);
        dst += *contribution;
      };
      result.Tvmult = [substitution](GlobalVectorType       &dst,
                                     const GlobalVectorType &src) {
        substitution(dst, src, true);
      };
      result.Tvmult_add = [substitution,
                           vector_memory](GlobalVectorType       &dst,
                                          const GlobalVectorType &src) {
        typename dealii::VectorMemory<GlobalVectorType>::Pointer contribution(
          *vector_memory);
        contribution->reinit(dst);
        substitution(*contribution, src, true);
        dst += *contribution;
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
          for (auto entry = base.begin(row); entry != base.end(row); ++entry)
            sparsity.add(row, entry->column());
          for (auto entry = increment.begin(row); entry != increment.end(row);
               ++entry)
            sparsity.add(row, entry->column());
        }

      MatrixType result;
      result.reinit(
        row_partition, column_partition, sparsity, communicator_, false);
      for (const auto row : row_partition)
        {
          std::map<size_type, double> values;
          for (auto entry = base.begin(row); entry != base.end(row); ++entry)
            values[entry->column()] += entry->value();
          for (auto entry = increment.begin(row); entry != increment.end(row);
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

    std::shared_ptr<MatrixType>
    materialized_block(const FieldId                             row,
                       const FieldId                             column,
                       const EvaluationContext<FieldVectorType> &context,
                       const double                              alpha) const
    {
      const bool has_state      = model_.has_state_operator(row, column);
      const bool has_derivative = model_.has_derivative_operator(row, column);
      if (!has_state && (alpha == 0. || !has_derivative))
        return {};

      auto result = std::make_shared<MatrixType>();
      materialized_block(row, column, context, alpha, *result);
      return result;
    }

    void
    repartition_matrix(const MatrixType       &source,
                       const dealii::IndexSet &row_partition,
                       const dealii::IndexSet &column_partition,
                       MatrixType             &destination) const
    {
      dealii::DynamicSparsityPattern sparsity(row_partition.size(),
                                              column_partition.size(),
                                              row_partition);
      for (const auto row : row_partition)
        for (auto entry = source.begin(row); entry != source.end(row); ++entry)
          sparsity.add(row, entry->column());

      destination.reinit(
        row_partition, column_partition, sparsity, communicator_, false);
      for (const auto row : row_partition)
        for (auto entry = source.begin(row); entry != source.end(row); ++entry)
          destination.set(row, entry->column(), entry->value());
      destination.compress(dealii::VectorOperation::insert);
    }

    void
    materialized_block(const FieldId                             row,
                       const FieldId                             column,
                       const EvaluationContext<FieldVectorType> &context,
                       const double                              alpha,
                       MatrixType &destination) const
    {
      const bool has_state      = model_.has_state_operator(row, column);
      const bool has_derivative = model_.has_derivative_operator(row, column);
      if (!has_state && (alpha == 0. || !has_derivative))
        return;

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

      const auto row_partition    = layout_.field(row).locally_owned;
      const auto column_partition = layout_.field(column).locally_owned;
      if (state_matrix.has_value())
        {
          const auto source = state_matrix->matrix();
          repartition_matrix(*source,
                             row_partition,
                             column_partition,
                             destination);
        }
      if (derivative_matrix.has_value())
        {
          const auto derivative = derivative_matrix->matrix();
          MatrixType derivative_repartitioned;
          repartition_matrix(*derivative,
                             row_partition,
                             column_partition,
                             derivative_repartitioned);
          if (state_matrix.has_value())
            destination.add(alpha, derivative_repartitioned);
          else
            {
              derivative_repartitioned *= alpha;
              destination.reinit(derivative_repartitioned);
              destination.copy_from(derivative_repartitioned);
            }
        }
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
            auto &block = result.block(i, j);
            if (model_.has_state_operator(field_layout_.field(i),
                                          field_layout_.field(j)) ||
                (alpha != 0. &&
                 model_.has_derivative_operator(field_layout_.field(i),
                                                field_layout_.field(j))))
              materialized_block(field_layout_.field(i),
                                 field_layout_.field(j),
                                 context,
                                 alpha,
                                 block);
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
