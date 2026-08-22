// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_linear_adapter_h
#define immersx_linear_adapter_h

#include <immersx/core/contributor.h>
#include <immersx/core/native_field_layout.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ImmersX
{
  /**
   * Execution adapter for affine steady semantic systems.
   *
   * Contributors register F(y)=0.  The adapter evaluates F(0), constructs
   * dF/dy, and asks the caller's solve policy to solve
   * dF/dy y = -F(0).  The residual must be affine and its state Jacobian must
   * not depend on y.  Time-derivative terms are rejected because this adapter
   * represents a steady problem.
   */
  template <typename FieldVectorType, typename GlobalVectorType>
  class LinearAdapter
  {
  public:
    using Model         = SemiDiscreteModel<FieldVectorType>;
    using Builder       = SemidiscreteBuilder<FieldVectorType>;
    using Operator      = dealii::LinearOperator<GlobalVectorType>;
    using SolveFunction = std::function<
      void(const Operator &, const GlobalVectorType &, GlobalVectorType &)>;

    LinearAdapter(const MPI_Comm communicator, SolveFunction solve)
      : communicator_(communicator)
      , solve_(std::move(solve))
      , field_layout_(layout_)
    {
      AssertThrow(solve_,
                  dealii::ExcMessage(
                    "LinearAdapter requires a linear solve callback."));
    }

    template <typename Contributor>
    auto
    add(const Contributor &contributor, const std::string &prefix)
      -> decltype(std::declval<const Contributor &>()(
        std::declval<Builder &>(),
        std::declval<const std::string &>()))
    {
      AssertThrow(!finalized_,
                  dealii::ExcMessage(
                    "LinearAdapter contributors must be added before solve."));
      const auto first_new_field = layout_.n_fields();
      Builder    builder(layout_, model_, prefix);
      auto       fields = contributor(builder, prefix);
      register_new_fields(first_new_field);
      return fields;
    }

    template <typename Problem, typename... Arguments>
    auto
    add(const Problem     &problem,
        const std::string &prefix,
        const Arguments &...arguments)
    {
      AssertThrow(!finalized_,
                  dealii::ExcMessage(
                    "LinearAdapter contributors must be added before solve."));
      const auto first_new_field = layout_.n_fields();
      Builder    builder(layout_, model_, prefix);
      auto       fields = contribute(builder, problem, arguments...);
      register_new_fields(first_new_field);
      return fields;
    }

    /** Construct a zero global semantic state. */
    GlobalVectorType
    make_state()
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
      finalize();
      return state.block(field_layout_.block(id));
    }

    const FieldVectorType &
    field(const GlobalVectorType &state, const FieldId id) const
    {
      finalize();
      return state.block(field_layout_.block(id));
    }

    void
    evaluate_residual(const GlobalVectorType &state,
                      GlobalVectorType       &residual) const
    {
      finalize();
      StateView<FieldVectorType> state_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      const EvaluationContext<FieldVectorType> context(0., state_view);
      residual.reinit(field_layout_.block_partitions(), communicator_);
      residual = 0.;
      for (std::size_t i = 0; i < layout_.n_fields(); ++i)
        model_.evaluate_row(FieldId(i),
                            context,
                            residual.block(field_layout_.block(FieldId(i))));
    }

    Operator
    jacobian(const GlobalVectorType &state) const
    {
      finalize();
      StateView<FieldVectorType> state_view(layout_, 0.);
      field_layout_.bind_state(state_view, state);
      const EvaluationContext<FieldVectorType> context(0., state_view);

      const unsigned int n = field_layout_.n_blocks();
      std::vector<std::vector<typename Model::Operator>> blocks(
        n, std::vector<typename Model::Operator>(n));
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          blocks[i][j] = model_.state_operator(field_layout_.field(i),
                                               field_layout_.field(j),
                                               context);

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

    void
    solve(GlobalVectorType &state) const
    {
      finalize();
      AssertThrow(!model_.has_derivative_terms(),
                  dealii::ExcMessage(
                    "LinearAdapter cannot solve a model with derivative "
                    "terms."));
      state.reinit(field_layout_.block_partitions(), communicator_);
      state = 0.;
      GlobalVectorType residual;
      evaluate_residual(state, residual);
      residual *= -1.;
      solve_(jacobian(state), residual, state);
    }

  private:
    void
    register_new_fields(const std::size_t first_new_field)
    {
      for (std::size_t i = first_new_field; i < layout_.n_fields(); ++i)
        field_layout_.add_field(FieldId(i));
    }

    void
    finalize() const
    {
      finalized_ = true;
      AssertThrow(layout_.n_fields() > 0,
                  dealii::ExcMessage("LinearAdapter has no semantic fields."));
    }

    MPI_Comm                                                    communicator_;
    SolveFunction                                               solve_;
    mutable StateLayout                                         layout_;
    mutable Model                                               model_;
    mutable BlockFieldLayout<FieldVectorType, GlobalVectorType> field_layout_;
    mutable bool finalized_ = false;
  };

  template <typename FieldVectorType, typename GlobalVectorType>
  using DistributedLinearAdapter =
    LinearAdapter<FieldVectorType, GlobalVectorType>;

} // namespace ImmersX

#endif // immersx_linear_adapter_h
