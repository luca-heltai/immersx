#ifndef rdlm_constraint_system_h
#define rdlm_constraint_system_h

#include <type_traits>
#include <utility>

/**
 * Algebraic representation of a constrained primal/multiplier system.
 *
 * This value object intentionally contains no mesh, finite-element, or
 * solver-specific state.
 */
template <typename PrimalOperator,
          typename ConstraintOperator,
          typename AdjointConstraintOperator,
          typename MultiplierMetricOperator>
class ConstraintSystem
{
public:
  ConstraintSystem(PrimalOperator            A,
                   ConstraintOperator        B,
                   AdjointConstraintOperator Bt,
                   MultiplierMetricOperator  M)
    : A(std::move(A))
    , B(std::move(B))
    , Bt(std::move(Bt))
    , M(std::move(M))
  {}

  const PrimalOperator &
  primal_operator() const
  {
    return A;
  }

  const ConstraintOperator &
  constraint_operator() const
  {
    return B;
  }

  const AdjointConstraintOperator &
  adjoint_constraint_operator() const
  {
    return Bt;
  }

  const MultiplierMetricOperator &
  multiplier_metric() const
  {
    return M;
  }

private:
  PrimalOperator            A;
  ConstraintOperator        B;
  AdjointConstraintOperator Bt;
  MultiplierMetricOperator  M;
};

template <typename A, typename B, typename Bt, typename M>
auto
make_constraint_system(A &&A_op, B &&B_op, Bt &&Bt_op, M &&M_op)
{
  using System = ConstraintSystem<std::decay_t<A>,
                                  std::decay_t<B>,
                                  std::decay_t<Bt>,
                                  std::decay_t<M>>;
  return System(std::forward<A>(A_op),
                std::forward<B>(B_op),
                std::forward<Bt>(Bt_op),
                std::forward<M>(M_op));
}

#endif
