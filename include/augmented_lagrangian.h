#ifndef rdlm_augmented_lagrangian_h
#define rdlm_augmented_lagrangian_h

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

/**
 * Two-by-two block preconditioner for augmented Lagrangian systems.
 */
template <typename VectorType,
          typename BlockVectorType = TrilinosWrappers::MPI::BlockVector>
class BlockPreconditionerAugmentedLagrangian
{
public:
  /**
   * Build the block preconditioner from linear-operator building blocks.
   */
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

  /**
   * Apply the block preconditioner to a two-block vector.
   */
  void
  vmult(BlockVectorType &v, const BlockVectorType &u) const
  {
    v.block(0) = 0.;
    v.block(1) = 0.;

    v.block(1) = -gamma * invW * u.block(1);
    v.block(0) = Aug_inv * (u.block(0) - Ct * v.block(1));
  }

  /**
   * Unused placeholder linear operator kept for backward compatibility.
   */
  LinearOperator<VectorType> K;
  /**
   * Approximate inverse of the augmented displacement block.
   */
  LinearOperator<VectorType> Aug_inv;
  /**
   * Coupling operator from displacement to multipliers.
   */
  LinearOperator<VectorType> C;
  /**
   * Inverse scaling/mass operator on multiplier space.
   */
  LinearOperator<VectorType> invW;
  /**
   * Transpose coupling operator from multipliers to displacement.
   */
  LinearOperator<VectorType> Ct;
  /**
   * Augmentation/scaling parameter for the multiplier block.
   */
  double gamma;
};

#endif
