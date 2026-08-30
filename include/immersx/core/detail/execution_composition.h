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

#ifdef DEAL_II_WITH_TRILINOS
#  include <Amesos2.hpp>
#  include <Epetra_CrsMatrix.h>
#  include <Epetra_Export.h>
#  include <Epetra_Map.h>
#  include <Epetra_MultiVector.h>
#endif

#include <immersx/algebra/linear_algebra.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/contributor.h>

#include <algorithm>
#include <cmath>
#include <limits>
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
        LocalOperator   coupling;
        LocalOperator   transpose_coupling;
        LocalOperator   inverse;
        FieldVectorType participant_prototype;
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
             *inverse,
             state.block(field_layout_.block(participant))});
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
        unsigned int    block;
        LocalOperator   to_multiplier;
        LocalOperator   from_multiplier;
        LocalOperator   inverse;
        FieldVectorType prototype;
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
             *inverse,
             state.block(field_layout_.block(field))});
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
      auto apply = [participants,
                    non_participants,
                    schur,
                    multiplier_block,
                    multiplier_preconditioner](GlobalVectorType       &dst,
                                               const GlobalVectorType &src) {
        GlobalVectorType rhs;
        rhs.reinit(src);
        rhs = src;
        dst = 0.;

        for (const auto &non_participant : non_participants)
          non_participant.inverse.vmult(dst.block(non_participant.block),
                                        rhs.block(non_participant.block));

        FieldVectorType schur_rhs;
        schur_rhs.reinit(rhs.block(multiplier_block));
        schur_rhs = rhs.block(multiplier_block);
        schur_rhs *= -1.;
        for (const auto &participant : participants)
          {
            participant.inverse.vmult(dst.block(participant.block),
                                      rhs.block(participant.block));
            participant.to_multiplier.vmult_add(schur_rhs,
                                                dst.block(participant.block));
          }

        FieldVectorType multiplier_solution;
        multiplier_solution.reinit(schur_rhs);
        multiplier_solution = 0.;
        dealii::SolverControl                control(1000, 1.e-10);
        dealii::SolverGMRES<FieldVectorType> solver(control);
        solver.solve(schur,
                     multiplier_solution,
                     schur_rhs,
                     multiplier_preconditioner);
        dst.block(multiplier_block) = multiplier_solution;

        for (const auto &participant : participants)
          {
            FieldVectorType correction_rhs;
            correction_rhs.reinit(participant.prototype);
            participant.from_multiplier.vmult(correction_rhs,
                                              multiplier_solution);
            FieldVectorType correction;
            correction.reinit(correction_rhs);
            participant.inverse.vmult(correction, correction_rhs);
            dst.block(participant.block) -= correction;
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

      auto apply = [base, augmentations, gamma](GlobalVectorType       &dst,
                                                const GlobalVectorType &src) {
        const unsigned int n = base.size();
        for (unsigned int i = 0; i < n; ++i)
          {
            dst.block(i) = 0.;
            for (unsigned int j = 0; j < n; ++j)
              base[i][j].vmult_add(dst.block(i), src.block(j));
          }

        for (const auto &augmentation : augmentations)
          {
            FieldVectorType constraint_value;
            constraint_value.reinit(src.block(augmentation.multiplier));
            constraint_value = 0.;
            for (unsigned int i = 0; i < augmentation.participants.size(); ++i)
              augmentation.to_multiplier[i].vmult_add(
                constraint_value, src.block(augmentation.participants[i]));

            FieldVectorType weighted_constraint;
            weighted_constraint.reinit(constraint_value);
            augmentation.inverse_metric.vmult(weighted_constraint,
                                              constraint_value);
            weighted_constraint *= gamma;
            for (unsigned int i = 0; i < augmentation.participants.size(); ++i)
              augmentation.from_multiplier[i].vmult_add(
                dst.block(augmentation.participants[i]), weighted_constraint);
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
                AssertThrow(from->trilinos_matrix().DomainMap().SameAs(
                              to->trilinos_matrix().RangeMap()),
                            dealii::ExcMessage(
                              "Augmented-Lagrangian coupling maps do not "
                              "share the multiplier partition."));
                AssertThrow(to->local_range() == inverse_metric.local_range(),
                            dealii::ExcMessage(
                              "Augmented-Lagrangian metric and coupling "
                              "partitions do not match."));

                MatrixType product;
                from->mmult(product, *to, inverse_metric);
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
      const auto augmented = augmented_lagrangian_matrix(state, gamma);

      using size_type = typename MatrixType::size_type;
      std::vector<size_type>        offsets;
      std::vector<dealii::IndexSet> partitions;
      offsets.reserve(metadata.participants.size());
      partitions.reserve(metadata.participants.size());
      size_type participant_size = 0;
      for (const auto field : metadata.participants)
        {
          offsets.push_back(participant_size);
          partitions.push_back(layout_.field(field).locally_owned);
          participant_size += partitions.back().size();
        }

      dealii::IndexSet participant_partition(participant_size);
      for (unsigned int i = 0; i < partitions.size(); ++i)
        for (const auto index : partitions[i])
          participant_partition.add_index(offsets[i] + index);
      participant_partition.compress();

      dealii::DynamicSparsityPattern sparsity(participant_size,
                                              participant_size,
                                              participant_partition);
      for (unsigned int i = 0; i < metadata.participants.size(); ++i)
        for (unsigned int j = 0; j < metadata.participants.size(); ++j)
          {
            const auto row_block =
              field_layout_.block(metadata.participants[i]);
            const auto col_block =
              field_layout_.block(metadata.participants[j]);
            const auto &block = augmented.block(row_block, col_block);
            for (const auto row : partitions[i])
              for (auto entry = block.begin(row); entry != block.end(row);
                   ++entry)
                sparsity.add(offsets[i] + row, offsets[j] + entry->column());
          }

      MatrixType participant_matrix;
      participant_matrix.reinit(participant_partition,
                                participant_partition,
                                sparsity,
                                communicator_,
                                false);
      for (unsigned int i = 0; i < metadata.participants.size(); ++i)
        for (unsigned int j = 0; j < metadata.participants.size(); ++j)
          {
            const auto row_block =
              field_layout_.block(metadata.participants[i]);
            const auto col_block =
              field_layout_.block(metadata.participants[j]);
            const auto &block = augmented.block(row_block, col_block);
            for (const auto row : partitions[i])
              for (auto entry = block.begin(row); entry != block.end(row);
                   ++entry)
                participant_matrix.set(offsets[i] + row,
                                       offsets[j] + entry->column(),
                                       entry->value());
          }
      participant_matrix.compress(dealii::VectorOperation::insert);

      FieldVectorType participant_prototype;
      participant_prototype.reinit(participant_partition, communicator_);
      const auto participant_inverse =
        make_mumps_inverse(participant_matrix, participant_prototype);

      const auto metric =
        model_.multiplier_metric(metadata.multiplier, context);
      AssertThrow(metric.has_value(),
                  dealii::ExcMessage(
                    "Augmented-Lagrangian preconditioning requires a "
                    "materialized multiplier metric."));
      const auto metric_matrix = metric->matrix();
      const auto multiplier_partition =
        layout_.field(metadata.multiplier).locally_owned;
      auto inverse_metric = std::make_shared<FieldVectorType>();
      inverse_metric->reinit(state.block(multiplier_block));
      for (const auto index : multiplier_partition)
        {
          const auto value         = metric_matrix->diag_element(index);
          (*inverse_metric)(index) = inverse_lumped_metric_value(value);
        }
      inverse_metric->compress(dealii::VectorOperation::insert);

      struct Participant
      {
        unsigned int  block;
        LocalOperator from_multiplier;
      };
      std::vector<Participant> participants;
      participants.reserve(metadata.participants.size());
      for (const auto field : metadata.participants)
        participants.push_back(
          {field_layout_.block(field),
           model_.state_operator(field, metadata.multiplier, context)});

      std::vector<LocalOperator> diagonal(field_layout_.n_blocks());
      std::vector<bool> is_participant(field_layout_.n_blocks(), false);
      for (const auto participant : participants)
        is_participant[participant.block] = true;
      for (unsigned int block = 0; block < field_layout_.n_blocks(); ++block)
        if (block != multiplier_block && !is_participant[block])
          {
            const auto inverse =
              local_preconditioner(field_layout_.field(block), state);
            AssertThrow(inverse.has_value(),
                        dealii::ExcMessage(
                          "Augmented-Lagrangian preconditioning requires "
                          "local inverses for non-participant fields."));
            diagonal[block] = *inverse;
          }

      const auto multiplier_prototype = state.block(multiplier_block);
      Operator   result;
      result.reinit_range_vector = [state](GlobalVectorType &vector,
                                           const bool        omit) {
        vector.reinit(state, omit);
      };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult                = [diagonal,
                      participants,
                      offsets,
                      partitions,
                      participant_inverse,
                      participant_prototype,
                      inverse_metric,
                      multiplier_partition,
                      multiplier_block,
                      multiplier_prototype,
                      gamma](GlobalVectorType       &dst,
                             const GlobalVectorType &src) {
        dst = 0.;
        FieldVectorType multiplier;
        multiplier.reinit(multiplier_prototype);
        multiplier = src.block(multiplier_block);
        for (const auto index : multiplier_partition)
          multiplier(index) *= -gamma * (*inverse_metric)(index);
        multiplier.compress(dealii::VectorOperation::insert);
        dst.block(multiplier_block) = multiplier;

        for (unsigned int block = 0; block < diagonal.size(); ++block)
          if (block != multiplier_block &&
              std::none_of(participants.begin(),
                           participants.end(),
                           [block](const auto &participant) {
                             return participant.block == block;
                           }))
            diagonal[block].vmult(dst.block(block), src.block(block));

        FieldVectorType rhs;
        rhs.reinit(participant_prototype);
        rhs = 0.;
        for (unsigned int i = 0; i < participants.size(); ++i)
          {
            FieldVectorType block_rhs;
            block_rhs.reinit(src.block(participants[i].block));
            block_rhs = src.block(participants[i].block);
            participants[i].from_multiplier.vmult_add(block_rhs, multiplier);
            for (const auto index : partitions[i])
              rhs(offsets[i] + index) = block_rhs(index);
          }
        rhs.compress(dealii::VectorOperation::insert);

        FieldVectorType solution;
        solution.reinit(participant_prototype);
        participant_inverse.vmult(solution, rhs);
        for (unsigned int i = 0; i < participants.size(); ++i)
          {
            for (const auto index : partitions[i])
              dst.block(participants[i].block)(index) =
                solution(offsets[i] + index);
            dst.block(participants[i].block)
              .compress(dealii::VectorOperation::insert);
          }
      };
      result.vmult_add = [apply = result.vmult](GlobalVectorType       &dst,
                                                const GlobalVectorType &src) {
        GlobalVectorType contribution;
        contribution.reinit(dst);
        apply(contribution, src);
        dst += contribution;
      };
      result.Tvmult     = result.vmult;
      result.Tvmult_add = result.vmult_add;
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

      auto matrix = materialized_block(field, field, context, alpha);
      if (matrix)
        {
          make_preconditioner_matrix_solve_ready(*matrix, field);
          return model_.preconditioner(field,
                                       *matrix,
                                       state.block(field_layout_.block(field)));
        }

      // A structurally null diagonal is valid for a mixed saddle system.
      // Give a registered metric factory an empty, correctly partitioned
      // matrix; do not manufacture an inverse for an absent block.
      const auto partition = layout_.field(field).locally_owned;
      dealii::DynamicSparsityPattern empty(partition.size(),
                                           partition.size(),
                                           partition);
      MatrixType                     empty_matrix;
      empty_matrix.reinit(partition, partition, empty, communicator_, false);
      return model_.preconditioner(field,
                                   empty_matrix,
                                   state.block(field_layout_.block(field)));
    }

    /** Make constrained rows invertible in a problem-local preconditioner. */
    void
    make_preconditioner_matrix_solve_ready(MatrixType   &matrix,
                                           const FieldId field) const
    {
      const auto &descriptor = layout_.field(field);
      if (descriptor.differential_components.n_elements() ==
          descriptor.locally_owned.n_elements())
        return;

      for (const auto row : matrix.locally_owned_range_indices())
        {
          std::vector<dealii::types::global_dof_index> constrained_columns;
          for (auto entry = matrix.begin(row); entry != matrix.end(row);
               ++entry)
            if (!descriptor.differential_components.is_element(entry->column()))
              constrained_columns.push_back(entry->column());
          for (const auto column : constrained_columns)
            matrix.set(row, column, 0.);
        }
      matrix.compress(dealii::VectorOperation::insert);

      for (const auto index : descriptor.locally_owned)
        if (!descriptor.differential_components.is_element(index) &&
            matrix.locally_owned_range_indices().is_element(index))
          matrix.clear_row(index, 1.);
      matrix.compress(dealii::VectorOperation::insert);
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
              const auto local =
                model_.preconditioner(multiplier, *metric_matrix, prototype);
              if (local.has_value())
                return *local;
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

    LocalOperator
    make_mumps_inverse(const MatrixType      &matrix,
                       const FieldVectorType &prototype) const
    {
#ifdef DEAL_II_WITH_TRILINOS
      AssertThrow(matrix.m() <= static_cast<typename MatrixType::size_type>(
                                  std::numeric_limits<int>::max()),
                  dealii::ExcMessage(
                    "Amesos2 MUMPS requires a 32-bit global matrix "
                    "index range."));

      const auto      &epetra_matrix = matrix.trilinos_matrix();
      const auto      &row_map       = epetra_matrix.RowMap();
      const auto      &column_map    = epetra_matrix.ColMap();
      std::vector<int> owned_indices;
      owned_indices.reserve(row_map.NumMyElements());
      for (int local_row = 0; local_row < row_map.NumMyElements(); ++local_row)
        owned_indices.push_back(static_cast<int>(row_map.GID64(local_row)));

      const Epetra_Map int_map(static_cast<int>(matrix.m()),
                               static_cast<int>(owned_indices.size()),
                               owned_indices.data(),
                               0,
                               epetra_matrix.Comm());
      auto int_matrix = std::make_shared<Epetra_CrsMatrix>(Copy, int_map, 0);
      for (int local_row = 0; local_row < row_map.NumMyElements(); ++local_row)
        {
          int     n_entries = 0;
          double *values    = nullptr;
          int    *columns   = nullptr;
          AssertThrow(epetra_matrix.ExtractMyRowView(
                        local_row, n_entries, values, columns) == 0,
                      dealii::ExcMessage(
                        "Unable to inspect the Epetra matrix row."));
          std::vector<int>    global_columns(n_entries);
          std::vector<double> global_values(values, values + n_entries);
          for (int entry = 0; entry < n_entries; ++entry)
            global_columns[entry] =
              static_cast<int>(column_map.GID64(columns[entry]));
          AssertThrow(int_matrix->InsertGlobalValues(owned_indices[local_row],
                                                     n_entries,
                                                     global_values.data(),
                                                     global_columns.data()) ==
                        0,
                      dealii::ExcMessage(
                        "Unable to convert the matrix to 32-bit Epetra global "
                        "indices."));
        }
      AssertThrow(int_matrix->FillComplete(int_map, int_map) == 0,
                  dealii::ExcMessage(
                    "Unable to complete the Epetra matrix for Amesos2 "
                    "MUMPS."));

      const Epetra_Map solver_map(static_cast<int>(matrix.m()),
                                  0,
                                  epetra_matrix.Comm());
      Epetra_Export    exporter(int_map, solver_map);
      auto             solver_matrix =
        std::make_shared<Epetra_CrsMatrix>(Copy, solver_map, 0);
      AssertThrow(solver_matrix->Export(*int_matrix, exporter, Insert) == 0,
                  dealii::ExcMessage(
                    "Unable to redistribute the Epetra matrix for Amesos2 "
                    "MUMPS."));
      AssertThrow(solver_matrix->FillComplete(solver_map, solver_map) == 0,
                  dealii::ExcMessage(
                    "Unable to complete the contiguous Epetra matrix for "
                    "Amesos2 MUMPS."));

      auto rhs      = Teuchos::rcp(new Epetra_MultiVector(solver_map, 1));
      auto solution = Teuchos::rcp(new Epetra_MultiVector(solver_map, 1));
      auto solver   = Amesos2::create<Epetra_CrsMatrix, Epetra_MultiVector>(
        "MUMPS", Teuchos::rcp(solver_matrix.get(), false), solution, rhs);
      solver->symbolicFactorization();
      solver->numericFactorization();

      const auto    communicator = matrix.get_mpi_communicator();
      const int     global_size  = static_cast<int>(matrix.m());
      LocalOperator result;
      result.reinit_range_vector = [prototype](FieldVectorType &vector,
                                               const bool       omit) {
        vector.reinit(prototype, omit);
      };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult                = [rhs,
                      solution,
                      solver,
                      solver_map,
                      communicator,
                      global_size](FieldVectorType       &dst,
                                   const FieldVectorType &src) {
        std::vector<double> global_rhs(global_size, 0.);
        const auto         &src_epetra = src.trilinos_vector();
        const auto         &src_map    = src_epetra.Map();
        for (int local = 0; local < src_map.NumMyElements(); ++local)
          global_rhs[static_cast<std::size_t>(src_map.GID64(local))] =
            src_epetra[0][local];
        AssertThrow(MPI_Allreduce(MPI_IN_PLACE,
                                  global_rhs.data(),
                                  global_size,
                                  MPI_DOUBLE,
                                  MPI_SUM,
                                  communicator) == MPI_SUCCESS,
                    dealii::ExcMessage(
                      "Unable to collect the MUMPS preconditioner right-hand "
                                     "side."));
        for (int local = 0; local < solver_map.NumMyElements(); ++local)
          (*rhs)[0][local] = global_rhs[solver_map.GID(local)];
        solver->solve(solution.get(), rhs.get());

        std::vector<double> global_solution(global_size, 0.);
        for (int local = 0; local < solver_map.NumMyElements(); ++local)
          global_solution[solver_map.GID(local)] = (*solution)[0][local];
        AssertThrow(MPI_Allreduce(MPI_IN_PLACE,
                                  global_solution.data(),
                                  global_size,
                                  MPI_DOUBLE,
                                  MPI_SUM,
                                  communicator) == MPI_SUCCESS,
                    dealii::ExcMessage(
                      "Unable to collect the MUMPS preconditioner solution."));
        const auto &dst_epetra = dst.trilinos_vector();
        const auto &dst_map    = dst_epetra.Map();
        for (int local = 0; local < dst_map.NumMyElements(); ++local)
          dst[dst_map.GID64(local)] =
            global_solution[static_cast<std::size_t>(dst_map.GID64(local))];
        dst.compress(dealii::VectorOperation::insert);
      };
      result.vmult_add = [apply = result.vmult](FieldVectorType       &dst,
                                                const FieldVectorType &src) {
        FieldVectorType contribution;
        contribution.reinit(dst);
        apply(contribution, src);
        dst += contribution;
      };
      result.Tvmult     = result.vmult;
      result.Tvmult_add = result.vmult_add;
      return result;
#else
      (void)matrix;
      (void)prototype;
      AssertThrow(false,
                  dealii::ExcMessage(
                    "The MUMPS preconditioner requires Trilinos."));
      return {};
#endif
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

      if (state_matrix.has_value())
        state_matrix->materialize_into_matrix(destination);
      if (derivative_matrix.has_value())
        {
          auto derivative = derivative_matrix->matrix();
          if (state_matrix.has_value())
            destination.add(alpha, *derivative);
          else
            {
              *derivative *= alpha;
              detail::copy_matrix(destination, *derivative);
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
