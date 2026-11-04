#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_tools.h>

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
#  include <Epetra_Vector.h>
#endif


#include <memory>
#include <type_traits>

#ifndef augmented_lagrangian_prec_h
#  define augmented_lagrangian_prec_h

namespace ImmersX
{
  using namespace dealii;

  namespace UtilitiesAL
  {

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
        static_assert(
          std::is_same<VectorType, TrilinosWrappers::MPI::Vector>::value ==
          true);
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

    private:
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

    /** Assemble A + gamma Ct diag(scaling_vector) Ct^T with native maps. */
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
          static_assert(
            std::is_same_v<TrilinosWrappers::MPI::Vector, VectorType>);
          AssertDimension(A.m(), A.n());
          AssertDimension(A.m(), Ct.m());
          AssertDimension(Ct.n(), scaling_vector.size());
          const auto         &native_A    = A.trilinos_matrix();
          Epetra_CrsMatrix    weighted_Ct = Ct.trilinos_matrix();
          const Epetra_Vector weights(View,
                                      scaling_vector.trilinos_vector(),
                                      0);
          const int           scaling_error = weighted_Ct.RightScale(weights);
          AssertThrow(scaling_error == 0,
                      TrilinosWrappers::SparseMatrix::ExcTrilinosError(
                        scaling_error));
          Epetra_CrsMatrix product(Copy, native_A.RowMap(), 0);
          const int        product_error = EpetraExt::MatrixMatrix::Multiply(
            weighted_Ct, false, Ct.trilinos_matrix(), true, product);
          AssertThrow(product_error == 0,
                      TrilinosWrappers::SparseMatrix::ExcTrilinosError(
                        product_error));
          Epetra_CrsMatrix *sum       = nullptr;
          const int         sum_error = EpetraExt::MatrixMatrix::Add(
            native_A, false, 1., product, false, gamma, sum);
          const std::unique_ptr<Epetra_CrsMatrix> result(sum);
          AssertThrow(sum_error == 0,
                      TrilinosWrappers::SparseMatrix::ExcTrilinosError(
                        sum_error));
          AssertThrow(result != nullptr, ExcInternalError());
          if (!result->Filled())
            {
              const int fill_error =
                result->FillComplete(native_A.DomainMap(), native_A.RangeMap());
              AssertThrow(fill_error == 0,
                          TrilinosWrappers::SparseMatrix::ExcTrilinosError(
                            fill_error));
            }
          augmented_matrix.reinit(*result, true);
        }
      else
        AssertThrow(false, ExcNotImplemented("Matrix type not supported!"));
#  else
      AssertThrow(false, ExcMessage("This function requires Trilinos."));
      (void)A;
      (void)Ct;
      (void)scaling_vector;
      (void)gamma;
      (void)augmented_matrix;
#  endif
    }


    /**
     * @brief Set the null space of the matrix using a pre-computed set of
     * vectors. Needed for the AMG preconditioner to work properly with linear
     * elasticity. In particular, parameter_list will be filled by this
     * function.
     */
    template <int spacedim, typename VectorType>
    void
    set_null_space(Teuchos::ParameterList              &parameter_list,
                   std::unique_ptr<Epetra_MultiVector> &ptr_distributed_modes,
                   const Epetra_RowMatrix              &matrix,
                   const std::vector<VectorType>       &modes)
    {
      static_assert((spacedim == 2 || spacedim == 3),
                    "This function only works for 2D and 3D.");

#  ifdef DEAL_II_WITH_TRILINOS


      using size_type = TrilinosWrappers::PreconditionAMG::size_type;
      const Epetra_Map   &domain_map      = matrix.OperatorDomainMap();
      constexpr size_type modes_dimension = spacedim == 3 ? 6 : 3;

      ptr_distributed_modes.reset(
        new Epetra_MultiVector(domain_map, modes_dimension));
      Assert(ptr_distributed_modes, ExcNotInitialized());
      Epetra_MultiVector &distributed_modes = *ptr_distributed_modes;

      const size_type global_size =
#    ifdef DEAL_II_WITH_64BIT_INDICES
        static_cast<size_type>(domain_map.NumGlobalElements64());
#    else
        static_cast<size_type>(domain_map.NumGlobalElements());
#    endif
      const size_type my_size =
        static_cast<size_type>(domain_map.NumMyElements());

      Assert(global_size ==
               static_cast<size_type>(
                 TrilinosWrappers::global_length(distributed_modes)),
             ExcDimensionMismatch(global_size,
                                  TrilinosWrappers::global_length(
                                    distributed_modes)));

      // Reshape null space as a contiguous vector of doubles so that
      // Trilinos can read from it.
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
                {
                  distributed_modes[d][row] =
                    static_cast<double>(modes[d][row]);
                }
              else
                {
                  const TrilinosWrappers::types::int_type gid =
#    ifdef DEAL_II_WITH_64BIT_INDICES
                    domain_map.GID64(static_cast<int>(row));
#    else
                    domain_map.GID(static_cast<int>(row));
#    endif
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
      parameter_list.set("null space: dimension",
                         distributed_modes.NumVectors());
      parameter_list.set("null space: vectors", distributed_modes.Values());


#  else
      AssertThrow(
        false,
        ExcMessage(
          "This function requires deal.II to be configured with Trilinos."));

#  endif
    }

  } // namespace UtilitiesAL

} // namespace ImmersX

#endif
