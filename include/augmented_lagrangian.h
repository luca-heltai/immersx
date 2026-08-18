#ifndef rdlm_augmented_lagrangian_h
#define rdlm_augmented_lagrangian_h

#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#ifdef DEAL_II_WITH_TRILINOS
#  include <deal.II/lac/trilinos_precondition.h>
#  include <deal.II/lac/trilinos_sparse_matrix.h>

#  include <EpetraExt_MatrixMatrix.h>
#  include <Epetra_Comm.h>
#  include <Epetra_CrsMatrix.h>
#  include <Epetra_Map.h>
#  include <Epetra_RowMatrixTransposer.h>
#endif

#include <type_traits>
#include <utility>
#include <memory>
#include <string>
#include <vector>

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
    const LinearOperator<VectorType>,
    const LinearOperator<VectorType> Ct_,
    const LinearOperator<VectorType> invW_,
    const double                     gamma_ = 1e1)
    : Aug_inv(Aug_inv_)
    , Ct(Ct_)
    , invW(invW_)
    , gamma(gamma_)
  {}

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

  /** Approximate inverse of the augmented primal block. */
  LinearOperator<VectorType> Aug_inv;
  /** Adjoint coupling operator from the multiplier space to the primal space.
   */
  LinearOperator<VectorType> Ct;
  /** Inverse multiplier metric used by the augmentation. */
  LinearOperator<VectorType> invW;
  /** Augmentation/scaling parameter for the multiplier block. */
  double gamma;
};

/**
 * The operator and inverse prepared by an augmented-block builder.
 */
template <typename OperatorType, typename InverseType>
struct PreparedAugmentedBlock
{
  OperatorType operator_;
  InverseType  inverse;
};

template <typename OperatorType, typename InverseType>
PreparedAugmentedBlock<std::decay_t<OperatorType>, std::decay_t<InverseType>>
make_prepared_augmented_block(OperatorType &&operator_, InverseType &&inverse)
{
  return {std::forward<OperatorType>(operator_),
          std::forward<InverseType>(inverse)};
}

/**
 * Reusable algebraic augmented-Lagrangian block solver.
 *
 * Problem-specific code supplies the four operators, the already-scaled
 * inverse multiplier metric invW, and a builder for the augmented primal
 * operator and its inverse.
 * The builder receives the canonical augmented operator and may replace both
 * values, for example with an explicitly assembled augmented matrix.
 */
template <typename VectorType,
          typename BlockVectorType = TrilinosWrappers::MPI::BlockVector>
class AugmentedLagrangianSolver
{
public:
  struct AdditionalData
  {
    double gamma = 10.0;
  };

  AugmentedLagrangianSolver(SolverControl        &outer_control,
                            const AdditionalData &data = AdditionalData())
    : outer_control(outer_control)
    , data(data)
  {}

  template <typename ConstraintSystemType,
            typename InverseMultiplierMetric,
            typename AugmentedBlockBuilder>
  void
  solve(const ConstraintSystemType    &system,
        const InverseMultiplierMetric &invW,
        AugmentedBlockBuilder        &&build_augmented_block,
        BlockVectorType               &solution,
        const BlockVectorType         &rhs) const
  {
    const auto canonical_Aug =
      system.primal_operator() + data.gamma *
                                   system.adjoint_constraint_operator() * invW *
                                   system.constraint_operator();

    const auto  prepared = build_augmented_block(canonical_Aug);
    const auto &Aug      = prepared.operator_;
    const auto &Aug_inv  = prepared.inverse;

    const auto Zero = system.multiplier_metric() * 0.0;
    const auto AA   = block_operator<2, 2, BlockVectorType>(
      {{{{Aug, system.adjoint_constraint_operator()}},
          {{system.constraint_operator(), Zero}}}});

    BlockVectorType augmented_solution;
    BlockVectorType augmented_rhs;
    AA.reinit_domain_vector(augmented_solution, false);
    AA.reinit_range_vector(augmented_rhs, false);

    VectorType transformed_constraint_rhs;
    transformed_constraint_rhs.reinit(rhs.block(0));
    transformed_constraint_rhs =
      data.gamma * system.adjoint_constraint_operator() * invW * rhs.block(1);
    augmented_rhs.block(0) = rhs.block(0);
    augmented_rhs.block(0).add(1., transformed_constraint_rhs);
    augmented_rhs.block(1) = rhs.block(1);

    BlockPreconditionerAugmentedLagrangian<VectorType, BlockVectorType>
      preconditioner{Aug_inv,
                     system.constraint_operator(),
                     system.adjoint_constraint_operator(),
                     invW,
                     data.gamma};

    SolverFGMRES<BlockVectorType> outer_solver(outer_control);
    outer_solver.solve(AA, augmented_solution, augmented_rhs, preconditioner);
    solution = augmented_solution;
  }

private:
  SolverControl &outer_control;
  AdditionalData data;
};

namespace UtilitiesAL
{
  template <typename MatrixType = SparseMatrix<double>,
            typename VectorType = Vector<typename MatrixType::value_type>,
            typename PreconditionerType = TrilinosWrappers::PreconditionAMG>
  void
  create_augmented_block(const MatrixType &A,
                         const MatrixType &Ct,
                         const VectorType &scaling_vector,
                         const double      gamma,
                         MatrixType       &augmented_matrix)
  {
#ifdef DEAL_II_WITH_TRILINOS
    if constexpr (std::is_same_v<TrilinosWrappers::SparseMatrix, MatrixType>)
      {
        Assert((std::is_same_v<TrilinosWrappers::MPI::Vector, VectorType>),
               ExcMessage("You must use Trilinos vectors, as you are using "
                          "Trilinos matrices."));
        Epetra_CrsMatrix A_trilinos  = A.trilinos_matrix();
        Epetra_CrsMatrix Ct_trilinos = Ct.trilinos_matrix();
        auto             multi_vector = scaling_vector.trilinos_vector();
        Assert((A_trilinos.NumGlobalRows() !=
                Ct_trilinos.DomainMap().NumGlobalElements()),
               ExcMessage("Number of columns in C must match dimension of A"));
        Assert((multi_vector.NumVectors() == 1),
               ExcMessage("The MultiVector must have exactly one column."));
        const Epetra_Map &map =
          static_cast<const Epetra_Map &>(multi_vector.Map());
        Epetra_CrsMatrix diag_matrix(Copy, map, 1);
        for (int i = 0; i < multi_vector.Map().NumMyElements(); ++i)
          {
            int    global_row = multi_vector.Map().GID(i);
            double val        = multi_vector[0][i];
            diag_matrix.InsertGlobalValues(global_row, 1, &val, &global_row);
          }
        diag_matrix.FillComplete();
        Epetra_CrsMatrix *W =
          new Epetra_CrsMatrix(Copy, Ct_trilinos.RowMap(), 0);
        EpetraExt::MatrixMatrix::Multiply(
          Ct_trilinos, false, diag_matrix, false, *W);
        Epetra_CrsMatrix *CtT_W =
          new Epetra_CrsMatrix(Copy, W->RangeMap(), 0);
        EpetraExt::MatrixMatrix::Multiply(
          *W, false, Ct_trilinos, true, *CtT_W);
        Epetra_CrsMatrix *result =
          new Epetra_CrsMatrix(Copy,
                               A_trilinos.RowMap(),
                               A_trilinos.MaxNumEntries());
        EpetraExt::MatrixMatrix::Add(
          A_trilinos, false, 1.0, *CtT_W, false, gamma, result);
        result->FillComplete();
        augmented_matrix.reinit(*result, true);
        delete W;
        delete CtT_W;
        delete result;
      }
    else
      AssertThrow(false, ExcNotImplemented("Matrix type not supported!"));
#else
    AssertThrow(false,
                ExcMessage("This function requires deal.II to be configured "
                           "with Trilinos."));
    (void)A;
    (void)Ct;
    (void)scaling_vector;
    (void)gamma;
    (void)augmented_matrix;
#endif
  }

  template <int spacedim, typename VectorType>
  void
  set_null_space(Teuchos::ParameterList              &parameter_list,
                 std::unique_ptr<Epetra_MultiVector> &ptr_distributed_modes,
                 const Epetra_RowMatrix              &matrix,
                 const std::vector<VectorType>       &modes)
  {
    static_assert((spacedim == 2 || spacedim == 3),
                  "This function only works in 2D and 3D.");
#ifdef DEAL_II_WITH_TRILINOS
    using size_type = TrilinosWrappers::PreconditionAMG::size_type;
    const Epetra_Map   &domain_map      = matrix.OperatorDomainMap();
    constexpr size_type modes_dimension = spacedim == 3 ? 6 : 3;
    ptr_distributed_modes.reset(
      new Epetra_MultiVector(domain_map, modes_dimension));
    Assert(ptr_distributed_modes, ExcNotInitialized());
    Epetra_MultiVector &distributed_modes = *ptr_distributed_modes;
    const size_type global_size =
#  ifdef DEAL_II_WITH_64BIT_INDICES
      static_cast<size_type>(domain_map.NumGlobalElements64());
#  else
      static_cast<size_type>(domain_map.NumGlobalElements());
#  endif
    const size_type my_size = static_cast<size_type>(domain_map.NumMyElements());
    Assert(global_size == static_cast<size_type>(
                            TrilinosWrappers::global_length(distributed_modes)),
           ExcDimensionMismatch(
             global_size, TrilinosWrappers::global_length(distributed_modes)));
    for (size_type d = 0; d < modes_dimension; ++d)
      {
        const size_type mode_size = static_cast<size_type>(modes[d].size());
        Assert(mode_size == my_size || mode_size == global_size,
               ExcMessage("Rigid body mode vector has the wrong size. "
                          "Expected either locally-owned size (" +
                          std::to_string(my_size) + ") or global size (" +
                          std::to_string(global_size) + "), but got " +
                          std::to_string(mode_size) + "."));
        for (size_type row = 0; row < my_size; ++row)
          {
            if (mode_size == my_size)
              distributed_modes[d][row] = static_cast<double>(modes[d][row]);
            else
              {
                const TrilinosWrappers::types::int_type gid =
#  ifdef DEAL_II_WITH_64BIT_INDICES
                  domain_map.GID64(static_cast<int>(row));
#  else
                  domain_map.GID(static_cast<int>(row));
#  endif
                Assert(gid >= 0 && static_cast<size_type>(gid) < global_size,
                       ExcMessage("Invalid global index " +
                                  std::to_string(gid) +
                                  " for mode vector of size " +
                                  std::to_string(global_size) + "."));
                distributed_modes[d][row] =
                  static_cast<double>(modes[d][static_cast<size_type>(gid)]);
              }
          }
      }
    parameter_list.set("null space: type", "pre-computed");
    parameter_list.set("null space: dimension", distributed_modes.NumVectors());
    parameter_list.set("null space: vectors", distributed_modes.Values());
#else
    AssertThrow(false,
                ExcMessage("This function requires deal.II to be configured "
                           "with Trilinos."));
#endif
  }
} // namespace UtilitiesAL

#endif
