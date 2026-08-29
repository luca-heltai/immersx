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

#include <deal.II/lac/solver_control.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/detail/execution_composition.h>
#include <immersx/core/problem_handle.h>
#include <immersx/core/representation.h>

#include <functional>
#include <string>
#include <utility>

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
    using Composition =
      detail::ExecutionComposition<FieldVectorType, GlobalVectorType>;

  public:
    using RepresentationType = Representation<FieldVectorType>;
    using ComponentRepresentationType =
      ComponentRepresentation<FieldVectorType>;
    using Operator        = dealii::LinearOperator<GlobalVectorType>;
    using LocalOperator   = dealii::LinearOperator<FieldVectorType>;
    using MatrixType      = typename Composition::MatrixType;
    using BlockMatrixType = typename Composition::BlockMatrixType;
    using SolveFunction   = std::function<
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
        const std::string &prefix = {},
        const Arguments &...arguments)
    {
      auto fields = composition_.add(problem, prefix, arguments...);
      return ProblemHandle<LinearAdapter, decltype(fields)>(*this,
                                                            std::move(fields));
    }

    template <typename Quantity, typename Target, typename Coupling>
    auto
    couple(const Quantity &quantity,
           const Target   &target,
           const Coupling &coupling)
    {
      auto interaction = detail::invoke_coupling(quantity, target, coupling, 0);
      return add(std::move(interaction),
                 "coupling" + std::to_string(coupling_count_++));
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

    RepresentationType
    observe(const FieldId id) const
    {
      AssertThrow(composition_.state_layout().contains(id),
                  dealii::ExcMessage("Cannot observe an unknown Field."));
      return RepresentationType(id);
    }

    FieldComponentView
    component(const FieldId id, const dealii::IndexSet &components) const
    {
      AssertThrow(composition_.state_layout().contains(id),
                  dealii::ExcMessage("Cannot select an unknown Field."));
      AssertThrow(components.size() ==
                    composition_.state_layout().field(id).locally_owned.size(),
                  dealii::ExcMessage(
                    "Component view must have the Field's global size."));
      return FieldComponentView(id, components);
    }

    ComponentRepresentationType
    observe(const FieldComponentView &view) const
    {
      AssertThrow(composition_.state_layout().contains(view.source()),
                  dealii::ExcMessage("Cannot observe an unknown Field."));
      return ComponentRepresentationType(view);
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

    bool
    can_materialize_matrix(const GlobalVectorType &state) const
    {
      return composition_.can_materialize_matrix(state);
    }

    BlockMatrixType
    block_matrix(const GlobalVectorType &state) const
    {
      return composition_.block_matrix(state);
    }

    MatrixType
    monolithic_matrix(const GlobalVectorType &state) const
    {
      return composition_.monolithic_matrix(state);
    }

    bool
    has_local_preconditioner(const FieldId field) const
    {
      return composition_.has_local_preconditioner(field);
    }

    const std::vector<SaddlePointMetadata> &
    saddle_points() const
    {
      return composition_.saddle_points();
    }

    LocalOperator
    schur_operator(const FieldId           multiplier,
                   const GlobalVectorType &state) const
    {
      return composition_.schur_operator(multiplier, state);
    }

    Operator
    augmented_lagrangian_operator(const GlobalVectorType &state,
                                  const double            gamma = 1.e1) const
    {
      return composition_.augmented_lagrangian_operator(state, gamma);
    }

    std::optional<LocalOperator>
    local_preconditioner(const FieldId           field,
                         const GlobalVectorType &state) const
    {
      return composition_.local_preconditioner(field, state);
    }

    Operator
    block_diagonal_preconditioner(const GlobalVectorType &state) const
    {
      return composition_.block_diagonal_preconditioner(state);
    }

    Operator
    block_triangular_preconditioner(const GlobalVectorType &state,
                                    const bool              lower = true) const
    {
      return composition_.block_triangular_preconditioner(state, lower);
    }

    FieldVectorType
    pack(const GlobalVectorType &state) const
    {
      return composition_.pack(state);
    }

    void
    unpack(const FieldVectorType &flat, GlobalVectorType &state) const
    {
      composition_.unpack(flat, state);
    }

    /** Solve using a freshly materialized matrix and the configured direct
     * backend. */
    void
    solve_direct(GlobalVectorType &state) const
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

      auto                    matrix   = composition_.monolithic_matrix(state);
      auto                    rhs      = composition_.pack(residual);
      auto                    result   = composition_.make_state();
      auto                    solution = composition_.pack(result);
      dealii::SolverControl   control(0, 1.e-12);
      ImmersXLA::SolverDirect solver(control);
      solver.initialize(matrix);
      solver.solve(solution, rhs);
      composition_.unpack(solution, state);
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
    Composition   composition_;
    SolveFunction solve_;
    std::size_t   coupling_count_ = 0;
  };
} // namespace ImmersX

#endif // immersx_linear_adapter_h
