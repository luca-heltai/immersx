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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>


/** A small, extensible identifier for a residual term. */
using TermId = unsigned int;


/** Identifiers used by the prototype tests and examples. */
namespace time_residual_terms
{
  constexpr TermId mass      = 0;
  constexpr TermId diffusion = 1;
  constexpr TermId nonlinear = 2;
} // namespace time_residual_terms


/** Identify a subsystem or state block in a history query. */
using SubsystemId = std::size_t;


/**
 * A compact selector passed from an integrator to residual contributors.
 *
 * Term ownership remains with the contributor.  The selector is only a
 * driver policy: the same term can be selected differently by two different
 * time integrators.
 */
class TermSelection
{
public:
  static constexpr unsigned int max_terms = 64;

  TermSelection()
    : mask(~std::uint64_t(0))
  {}

  static TermSelection
  all()
  {
    return TermSelection(~std::uint64_t(0));
  }

  static TermSelection
  none()
  {
    return TermSelection(0);
  }

  static TermSelection
  only(const TermId term)
  {
    TermSelection selection = none();
    selection.include(term);
    return selection;
  }

  void
  include(const TermId term)
  {
    mask |= bit(term);
  }

  void
  exclude(const TermId term)
  {
    mask &= ~bit(term);
  }

  bool
  contains(const TermId term) const
  {
    return (mask & bit(term)) != 0;
  }

private:
  explicit TermSelection(const std::uint64_t mask)
    : mask(mask)
  {}

  static std::uint64_t
  bit(const TermId term)
  {
    if (term >= max_terms)
      throw std::out_of_range("TermSelection supports at most 64 terms");
    return std::uint64_t(1) << term;
  }

  std::uint64_t mask;
};


/**
 * Context shared by residual and Jacobian contributors.
 *
 * For IDA, `state_weight` is one and `derivative_weight` is IDA's `alpha`.
 * For other methods the driver may choose the coefficients appropriate for
 * the stage equation.  The optional history query is deliberately type-erased
 * so that the context does not depend on a concrete StateLayout or block
 * vector implementation.
 */
template <typename VectorType>
struct TimeResidualContext
{
  TimeResidualContext(const double      time,
                      const VectorType &state,
                      const VectorType &state_derivative)
    : time(time)
    , state(state)
    , state_derivative(state_derivative)
  {}

  double            time;
  const VectorType &state;
  const VectorType &state_derivative;
  double            state_weight      = 1.;
  double            derivative_weight = 0.;
  TermSelection     selected_terms    = TermSelection::all();
  std::function<VectorType(SubsystemId, double)> history_query;

  /** Query another subsystem at an accepted or interpolated time. */
  VectorType
  historical_state(const SubsystemId subsystem, const double query_time) const
  {
    AssertThrow(history_query,
                dealii::ExcMessage("No state history query is attached to this "
                                   "residual context."));
    return history_query(subsystem, query_time);
  }
};


/**
 * A matrix-free linearization action.
 *
 * The action is the universal interface exposed to a driver.  Native deal.II
 * matrices and block matrices can be wrapped through `from_matrix()`; no
 * conversion from a generic action back to a sparse matrix is provided.
 */
template <typename VectorType>
class JacobianAction
{
public:
  using NativeOperator = dealii::LinearOperator<VectorType, VectorType>;
  using ApplyFunction  = std::function<void(VectorType &, const VectorType &)>;

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

  /** Wrap a deal.II Matrix, SparseMatrix, or BlockSparseMatrix. */
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
    if (native_operator)
      return *native_operator;

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


/** Differential/algebraic classification for one state vector layout. */
enum class StateVariableType
{
  differential,
  algebraic
};


/**
 * Minimal metadata bridge for IDA-like solvers.
 *
 * The keys are deliberately opaque block identifiers.  Worker 1's eventual
 * StateLayout can supply the same ids and IndexSets without changing this
 * adapter-facing interface.
 */
class DifferentialAlgebraicMetadata
{
public:
  using BlockId = std::size_t;

  explicit DifferentialAlgebraicMetadata(const std::size_t global_size)
    : global_size(global_size)
  {}

  void
  add_block(const BlockId           block,
            const dealii::IndexSet &indices,
            const StateVariableType type)
  {
    AssertThrow(indices.size() == global_size,
                dealii::ExcMessage("DAE block has the wrong global "
                                   "IndexSet size."));
    AssertThrow(blocks.find(block) == blocks.end(),
                dealii::ExcMessage("DAE block id was registered "
                                   "twice."));
    blocks.emplace(block, Block{indices, type});
  }

  void
  add_block(const BlockId           block,
            const std::size_t       begin,
            const std::size_t       end,
            const StateVariableType type)
  {
    dealii::IndexSet indices(global_size);
    indices.add_range(begin, end);
    add_block(block, indices, type);
  }

  bool
  has_block(const BlockId block) const
  {
    return blocks.find(block) != blocks.end();
  }

  StateVariableType
  type(const BlockId block) const
  {
    const auto it = blocks.find(block);
    AssertThrow(it != blocks.end(),
                dealii::ExcMessage("Unknown DAE block id."));
    return it->second.type;
  }

  dealii::IndexSet
  differential_components() const
  {
    return components_of(StateVariableType::differential);
  }

  dealii::IndexSet
  algebraic_components() const
  {
    return components_of(StateVariableType::algebraic);
  }

private:
  struct Block
  {
    dealii::IndexSet  indices;
    StateVariableType type;
  };

  dealii::IndexSet
  components_of(const StateVariableType wanted) const
  {
    dealii::IndexSet result(global_size);
    for (const auto &entry : blocks)
      if (entry.second.type == wanted)
        result.add_indices(entry.second.indices);
    result.compress();
    return result;
  }

  std::size_t              global_size;
  std::map<BlockId, Block> blocks;
};


/**
 * A term-wise semi-discrete residual model.
 *
 * Each callback adds to its destination.  The driver chooses `selected_terms`
 * in the context, so monolithic, partitioned, ARKODE, and IDA paths can call
 * the same residual contributors with different policies.
 */
template <typename VectorType>
class TimeResidualModel
{
public:
  using Context          = TimeResidualContext<VectorType>;
  using ResidualFunction = std::function<void(const Context &, VectorType &)>;
  using JacobianFunction =
    std::function<JacobianAction<VectorType>(const Context &)>;

  void
  add_term(const TermId     term,
           ResidualFunction residual,
           JacobianFunction jacobian = JacobianFunction())
  {
    terms.push_back({term, std::move(residual), std::move(jacobian)});
  }

  void
  residual(const Context &context, VectorType &dst) const
  {
    dst = 0.;
    for (const auto &term : terms)
      if (context.selected_terms.contains(term.id) && term.residual)
        term.residual(context, dst);
  }

  JacobianAction<VectorType>
  jacobian_action(const Context &context) const
  {
    JacobianAccumulator<VectorType> accumulator;
    for (const auto &term : terms)
      if (context.selected_terms.contains(term.id) && term.jacobian)
        accumulator.add(term.jacobian(context));
    return accumulator.action();
  }

private:
  struct Term
  {
    TermId           id;
    ResidualFunction residual;
    JacobianFunction jacobian;
  };

  std::vector<Term> terms;
};

#endif // immersx_time_residual_h
