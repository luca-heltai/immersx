
#include <deal.II/base/exceptions.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
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


#ifndef augmented_lagrangian_prec_h
#  define augmented_lagrangian_prec_h

using namespace dealii;

namespace UtilitiesAL
{

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
      static_assert(
        std::is_same<VectorType, TrilinosWrappers::MPI::Vector>::value == true);
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

  private:
    LinearOperator<VectorType> K;
    LinearOperator<VectorType> Aug_inv;
    LinearOperator<VectorType> C;
    LinearOperator<VectorType> invW;
    LinearOperator<VectorType> Ct;
    double                     gamma;
  };

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
#  ifdef DEAL_II_WITH_TRILINOS


    if constexpr (std::is_same_v<TrilinosWrappers::SparseMatrix, MatrixType>)
      {
        Assert((std::is_same_v<TrilinosWrappers::MPI::Vector, VectorType>),
               ExcMessage("You must use Trilinos vectors, as you are using "
                          "Trilinos matrices."));


        Epetra_CrsMatrix A_trilinos   = A.trilinos_matrix();
        Epetra_CrsMatrix Ct_trilinos  = Ct.trilinos_matrix();
        auto             multi_vector = scaling_vector.trilinos_vector();


        Assert((A_trilinos.NumGlobalRows() !=
                Ct_trilinos.DomainMap().NumGlobalElements()),
               ExcMessage("Number of columns in C must match dimension of A"));


        // Ensure the MultiVector has only one column.
        Assert((multi_vector.NumVectors() == 1),
               ExcMessage("The MultiVector must have exactly one column."));


        // Create diagonal matrix from first vector of v
        // Explicitly cast the map to Epetra_Map
        const Epetra_Map &map =
          static_cast<const Epetra_Map &>(multi_vector.Map());


        // Create a diagonal matrix with 1 nonzero entry per row
        Epetra_CrsMatrix diag_matrix(Copy, map, 1);
        for (int i = 0; i < multi_vector.Map().NumMyElements(); ++i)
          {
            int    global_row = multi_vector.Map().GID(i);
            double val        = multi_vector[0][i]; // Access first vector
            diag_matrix.InsertGlobalValues(global_row, 1, &val, &global_row);
          }
        diag_matrix.FillComplete();


        Epetra_CrsMatrix *W =
          new Epetra_CrsMatrix(Copy, Ct_trilinos.RowMap(), 0);
        EpetraExt::MatrixMatrix::Multiply(
          Ct_trilinos, false, diag_matrix, false, *W);


        // Compute Ct^T * W, which is equivalent to (C^T * diag(V)) * C
        Epetra_CrsMatrix *CtT_W = new Epetra_CrsMatrix(Copy, W->RangeMap(), 0);
        EpetraExt::MatrixMatrix::Multiply(
          *W, false /* transpose */, Ct_trilinos, true, *CtT_W);


        // Add A to the result, scaling with gamma
        Epetra_CrsMatrix *result =
          new Epetra_CrsMatrix(Copy,
                               A_trilinos.RowMap(),
                               A_trilinos.MaxNumEntries());
        EpetraExt::MatrixMatrix::Add(
          A_trilinos, false, 1.0, *CtT_W, false, gamma, result);
        result->FillComplete();


        // Initialize the final Trilinos matrix
        augmented_matrix.reinit(*result, true /*copy_values*/);


        // Delete unnecessary objects.
        delete W;
        delete CtT_W;
        delete result;
      }
    else
      {
        // PETSc not supported so far.
        AssertThrow(false, ExcNotImplemented("Matrix type not supported!"));
      }
#  else
    AssertThrow(
      false,
      ExcMessage(
        "This function requires deal.II to be configured with Trilinos."));


    (void)velocity_dh;
    (void)C;
    (void)Ct;
    (void)scaling_vector;
    (void)velocity_constraints;
    (void)gamma;
    (void)augmented_matrix;
#  endif
  }
} // namespace UtilitiesAL

#endif