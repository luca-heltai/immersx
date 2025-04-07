#ifndef augmented_lagrangian_prec_h
#define augmented_lagrangian_prec_h

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

template <typename VectorType,
          typename BlockVectorType = TrilinosWrappers::MPI::BlockVector>
class BlockPreconditionerAugmentedLagrangian
{
public:
  BlockPreconditionerAugmentedLagrangian(
    const LinearOperator<VectorType> Aug_inv_,
    const LinearOperator<VectorType> C_,
    const LinearOperator<VectorType> Ct_,
    const LinearOperator<VectorType> invW_,
    const double                     gamma_ = 1e1)
  {
    Aug_inv = Aug_inv_;
    C       = C_;
    Ct      = Ct_;
    invW    = invW_;
    gamma   = gamma_;
  }

  void
  vmult(BlockVectorType &v, const BlockVectorType &u) const
  {
    v.block(0) = 0.;
    v.block(1) = 0.;

    v.block(1) = -gamma * invW * u.block(1);
    v.block(0) = Aug_inv * (u.block(0) - Ct * v.block(1));
  }

  LinearOperator<VectorType> K;
  LinearOperator<VectorType> Aug_inv;
  LinearOperator<VectorType> C;
  LinearOperator<VectorType> invW;
  LinearOperator<VectorType> Ct;
  double                     gamma;
};

#endif