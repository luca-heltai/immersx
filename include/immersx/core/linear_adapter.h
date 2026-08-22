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

#include <immersx/core/detail/execution_composition.h>

#include <functional>

namespace ImmersX
{
  /**
   * Execution adapter for affine steady semantic systems.
   *
   * `solve()` evaluates the affine residual at zero and overwrites its state
   * with the result of the supplied direct linear solve. The incoming state is
   * therefore storage, not a nonlinear initial guess.
   */
  template <typename FieldVectorType, typename GlobalVectorType>
  class LinearAdapter
  {
  public:
    using Composition =
      detail::ExecutionComposition<FieldVectorType, GlobalVectorType>;
    using Operator      = typename Composition::Operator;
    using SolveFunction = std::function<
      void(const Operator &, const GlobalVectorType &, GlobalVectorType &)>;

    LinearAdapter(const MPI_Comm communicator, SolveFunction solve)
      : composition_(communicator)
      , solve_(std::move(solve))
    {
      AssertThrow(solve_,
                  dealii::ExcMessage(
                    "LinearAdapter requires a linear solve callback."));
    }

    template <typename Problem, typename... Arguments>
    auto
    add(const Problem     &problem,
        const std::string &prefix,
        const Arguments &...arguments)
    {
      return composition_.add(problem, prefix, arguments...);
    }

    GlobalVectorType
    make_state() const
    {
      return composition_.make_state();
    }

    void
    reinit(GlobalVectorType &state) const
    {
      composition_.reinit(state);
    }

    FieldVectorType &
    field(GlobalVectorType &state, const FieldId id) const
    {
      return composition_.field(state, id);
    }

    const FieldVectorType &
    field(const GlobalVectorType &state, const FieldId id) const
    {
      return composition_.field(state, id);
    }

    void
    evaluate_residual(const GlobalVectorType &state,
                      GlobalVectorType       &residual) const
    {
      composition_.evaluate_residual(0., state, nullptr, residual);
    }

    Operator
    jacobian(const GlobalVectorType &state) const
    {
      return composition_.jacobian(0., state, nullptr, 0.);
    }

    void
    solve(GlobalVectorType &state) const
    {
      const auto &model = composition_.model();
      AssertThrow(!model.has_derivative_terms(),
                  dealii::ExcMessage(
                    "LinearAdapter cannot solve a model with derivative "
                    "terms."));
      auto residual = composition_.make_state();
      state         = composition_.make_state();
      composition_.evaluate_residual(0., state, nullptr, residual);
      residual *= -1.;
      solve_(composition_.jacobian(0., state, nullptr, 0.), residual, state);
    }

  private:
    detail::ExecutionComposition<FieldVectorType, GlobalVectorType>
                  composition_;
    SolveFunction solve_;
  };
} // namespace ImmersX

#endif // immersx_linear_adapter_h
