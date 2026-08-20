// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_time_residual_h
#define immersx_time_residual_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/linear_operator.h>

#include <immersx/core/field.h>
#include <immersx/core/residual.h>
#include <immersx/core/state.h>

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ImmersX
{
  /** Names used by the small residual examples and tests. */
  namespace time_residual_terms
  {
    inline constexpr const char *mass      = "mass";
    inline constexpr const char *diffusion = "diffusion";
    inline constexpr const char *nonlinear = "nonlinear";
  } // namespace time_residual_terms


  /**
   * A matrix-free linearization action.
   *
   * The action is the universal interface exposed to a driver.  Native deal.II
   * matrices and block matrices can be wrapped through `from_matrix()`; no
   * conversion from a generic action back to a sparse matrix is provided.
   * The callback is copied into the action and invoked synchronously by
   * `vmult()`. It must not capture a temporary state, evaluation context, or
   * another object whose lifetime ends before the action is used.
   */
  template <typename VectorType>
  class JacobianAction
  {
  public:
    using NativeOperator = dealii::LinearOperator<VectorType, VectorType>;
    using ApplyFunction = std::function<void(VectorType &, const VectorType &)>;

    JacobianAction() = default;

    explicit JacobianAction(ApplyFunction apply)
      : apply_function(std::move(apply))
    {}

    static JacobianAction
    from_linear_operator(const NativeOperator &op)
    {
      return JacobianAction([op](VectorType       &dst,
                                 const VectorType &src) { op.vmult(dst, src); },
                            op);
    }

    /**
     * Wrap a deal.II LinearOperator with a backend-specific payload.
     *
     * The payload is intentionally erased at the action boundary.  This lets
     * Trilinos and PETSc operators use their specialized deal.II construction
     * helpers while keeping the contributor-facing capability equal to vmult.
     */
    template <typename Payload>
    static JacobianAction
    from_linear_operator(
      const dealii::LinearOperator<VectorType, VectorType, Payload> &op)
    {
      return JacobianAction(
        [op](VectorType &dst, const VectorType &src) { op.vmult(dst, src); });
    }

    /**
     * Wrap a deal.II Matrix, SparseMatrix, or BlockSparseMatrix.
     *
     * The matrix is not owned; it must outlive the returned action and every
     * `LinearOperator` view derived from it.
     */
    template <typename MatrixType>
    static JacobianAction
    from_matrix(const MatrixType &matrix)
    {
      return from_linear_operator(
        dealii::linear_operator<VectorType, VectorType>(matrix));
    }

    bool
    empty() const
    {
      return !apply_function;
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      AssertThrow(!empty(), dealii::ExcMessage("Empty Jacobian action."));
      apply_function(dst, src);
    }

    void
    vmult_add(VectorType &dst, const VectorType &src) const
    {
      VectorType contribution;
      contribution.reinit(dst);
      vmult(contribution, src);
      dst += contribution;
    }

    JacobianAction
    scaled(const double factor) const
    {
      AssertThrow(!empty(),
                  dealii::ExcMessage("Cannot scale an empty "
                                     "Jacobian action."));

      if (native_operator)
        return from_linear_operator(factor * *native_operator);

      const JacobianAction base = *this;
      return JacobianAction(
        [base, factor](VectorType &dst, const VectorType &src) {
          base.vmult(dst, src);
          dst *= factor;
        });
    }

    /** Return whether this action retains a native deal.II operator hook. */
    bool
    has_native_operator() const
    {
      return native_operator.has_value();
    }

    /**
     * Return a LinearOperator view of the action.
     *
     * The prototype vector supplies the vector layout for a matrix-free action.
     * This is useful with deal.II Krylov solvers and does not assemble a
     * matrix.
     */
    NativeOperator
    as_linear_operator(const VectorType &prototype) const
    {
      const JacobianAction base = *this;
      NativeOperator       op;
      op.reinit_range_vector = [prototype](VectorType &vector, const bool) {
        vector.reinit(prototype);
      };
      op.reinit_domain_vector = [prototype](VectorType &vector, const bool) {
        vector.reinit(prototype);
      };
      op.vmult = [base](VectorType &dst, const VectorType &src) {
        base.vmult(dst, src);
      };
      op.vmult_add = [base](VectorType &dst, const VectorType &src) {
        base.vmult_add(dst, src);
      };
      return op;
    }

  private:
    JacobianAction(ApplyFunction apply, const NativeOperator &op)
      : apply_function(std::move(apply))
      , native_operator(op)
    {}

    ApplyFunction                 apply_function;
    std::optional<NativeOperator> native_operator;
  };


  /** Additive accumulation of contributor Jacobian actions. */
  template <typename VectorType>
  class JacobianAccumulator
  {
  public:
    using Action = JacobianAction<VectorType>;

    void
    add(Action action)
    {
      if (!action.empty())
        actions.emplace_back(std::move(action));
    }

    std::size_t
    size() const
    {
      return actions.size();
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      dst = 0.;
      if (actions.empty())
        return;

      VectorType contribution;
      contribution.reinit(dst);
      for (const auto &action : actions)
        {
          contribution = 0.;
          action.vmult(contribution, src);
          dst += contribution;
        }
    }

    Action
    action() const
    {
      if (actions.empty())
        return Action();

      const auto captured_actions =
        std::make_shared<std::vector<Action>>(actions);
      return Action([captured_actions](VectorType &dst, const VectorType &src) {
        dst = 0.;
        VectorType contribution;
        contribution.reinit(dst);
        for (const auto &action : *captured_actions)
          {
            contribution = 0.;
            action.vmult(contribution, src);
            dst += contribution;
          }
      });
    }

  private:
    std::vector<Action> actions;
  };


  /**
   * Minimal metadata bridge for IDA-like solvers.
   *
   * The semantic time role comes from StateLayout. The IndexSet registration is
   * only the local adapter information needed to project semantic fields onto
   * the monolithic vector expected by IDA. The StateLayout is held by reference
   * and must outlive this metadata object.
   */
  class DifferentialAlgebraicMetadata
  {
  public:
    DifferentialAlgebraicMetadata(const StateLayout &layout,
                                  const std::size_t  global_size)
      : layout_(layout)
      , global_size_(global_size)
    {}

    void
    add_field(const FieldId field, const dealii::IndexSet &indices)
    {
      AssertThrow(layout_.contains(field),
                  dealii::ExcMessage("DAE field is not in the state layout."));
      AssertThrow(indices.size() == global_size_,
                  dealii::ExcMessage("DAE field has the wrong global "
                                     "IndexSet size."));
      AssertThrow(fields_.find(field) == fields_.end(),
                  dealii::ExcMessage("DAE field was registered twice."));
      fields_.emplace(field, indices);
    }

    void
    add_field(const FieldId     field,
              const std::size_t begin,
              const std::size_t end)
    {
      dealii::IndexSet indices(global_size_);
      indices.add_range(begin, end);
      add_field(field, indices);
    }

    bool
    has_field(const FieldId field) const
    {
      return fields_.find(field) != fields_.end();
    }

    TimeRole
    time_role(const FieldId field) const
    {
      AssertThrow(layout_.contains(field),
                  dealii::ExcMessage("Unknown DAE field."));
      return layout_.field(field).time_role;
    }

    dealii::IndexSet
    differential_components() const
    {
      return components_of(TimeRole::differential);
    }

    dealii::IndexSet
    algebraic_components() const
    {
      return components_of(TimeRole::algebraic);
    }

  private:
    dealii::IndexSet
    components_of(const TimeRole wanted) const
    {
      dealii::IndexSet result(global_size_);
      for (const auto &entry : fields_)
        if (layout_.field(entry.first).time_role == wanted)
          result.add_indices(entry.second);
      result.compress();
      return result;
    }

    const StateLayout                  &layout_;
    std::size_t                         global_size_;
    std::map<FieldId, dealii::IndexSet> fields_;
  };
  /**
   * A term-wise semi-discrete residual model for F(t,y,ydot)=0.
   *
   * Each callback adds to its destination. The driver chooses terms in the
   * EvaluationContext, so steady, partitioned, ARKODE, and IDA paths can call
   * the same residual contributors with different policies. It does not retain
   * an EvaluationContext or any state storage between calls.
   */
  template <typename VectorType>
  class SemiDiscreteModel
  {
  public:
    using Context       = EvaluationContext<VectorType>;
    using Linearization = LinearizationContext<VectorType>;
    using ResidualFunction =
      std::function<void(const Context &, ResidualAccumulator<VectorType> &)>;
    using JacobianFunction =
      std::function<void(const Linearization &,
                         const StateAccessor<VectorType> &,
                         ResidualAccumulator<VectorType> &)>;

    void
    add_term(std::string      term,
             ResidualFunction residual,
             JacobianFunction jacobian = JacobianFunction())
    {
      terms.push_back(
        {std::move(term), std::move(residual), std::move(jacobian)});
    }

    /** Add all selected term contributions to an externally bound residual. */
    void
    evaluate(const Context                   &context,
             ResidualAccumulator<VectorType> &residual) const
    {
      for (const auto &term : terms)
        if (context.terms().includes(term.name, TermTreatment::all) &&
            term.residual)
          term.residual(context, residual);
    }

    /** Apply the selected semantic Jacobian terms to an increment. */
    void
    add_jacobian_action(const Linearization             &linearization,
                        const StateAccessor<VectorType> &increment,
                        ResidualAccumulator<VectorType> &destination) const
    {
      for (const auto &term : terms)
        if (linearization.evaluation().terms().includes(term.name,
                                                        TermTreatment::all) &&
            term.jacobian)
          term.jacobian(linearization, increment, destination);
    }

  private:
    struct Term
    {
      std::string      name;
      ResidualFunction residual;
      JacobianFunction jacobian;
    };

    std::vector<Term> terms;
  };
} // namespace ImmersX

#endif // immersx_time_residual_h
