// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_local_preconditioner_h
#define immersx_local_preconditioner_h

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/precondition.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/matrix_operator.h>

#include <memory>

namespace ImmersX
{
  /**
   * Build a lifetime-safe LinearOperator around a deal.II preconditioner.
   *
   * The matrix and preconditioner are kept alive by the returned operator,
   * which is important for deal.II backends whose preconditioners observe the
   * matrix passed to initialize().
   */
  template <typename VectorType,
            typename MatrixType,
            typename PreconditionerType>
  dealii::LinearOperator<VectorType>
  make_local_preconditioner(
    const MatrixType                                  &matrix,
    const VectorType                                  &prototype,
    const typename PreconditionerType::AdditionalData &additional_data = {})
  {
    auto matrix_storage_unique = detail::clone_matrix(matrix);
    std::shared_ptr<MatrixType> matrix_storage =
      std::move(matrix_storage_unique);
    auto preconditioner = std::make_shared<PreconditionerType>();
    preconditioner->initialize(*matrix_storage, additional_data);

    dealii::LinearOperator<VectorType> result;
    result.reinit_range_vector = [prototype](VectorType &vector,
                                             const bool  omit) {
      vector.reinit(prototype, omit);
    };
    result.reinit_domain_vector = result.reinit_range_vector;
    result.vmult = [matrix_storage, preconditioner](VectorType       &dst,
                                                    const VectorType &src) {
      (void)matrix_storage;
      preconditioner->vmult(dst, src);
    };
    result.vmult_add = [matrix_storage, preconditioner](VectorType       &dst,
                                                        const VectorType &src) {
      (void)matrix_storage;
      VectorType contribution;
      contribution.reinit(dst);
      preconditioner->vmult(contribution, src);
      dst += contribution;
    };
    result.Tvmult = [matrix_storage, preconditioner](VectorType       &dst,
                                                     const VectorType &src) {
      (void)matrix_storage;
      preconditioner->vmult(dst, src);
    };
    result.Tvmult_add =
      [matrix_storage, preconditioner](VectorType &dst, const VectorType &src) {
        (void)matrix_storage;
        VectorType contribution;
        contribution.reinit(dst);
        preconditioner->vmult(contribution, src);
        dst += contribution;
      };
    return result;
  }

  /** Build a local AMG approximate inverse using the active linear backend. */
  template <typename VectorType, typename MatrixType>
  dealii::LinearOperator<VectorType>
  make_amg_preconditioner(
    const MatrixType &matrix,
    const VectorType &prototype,
    const typename ImmersXLA::MPI::PreconditionAMG::AdditionalData
      &additional_data = {})
  {
    return make_local_preconditioner<VectorType,
                                     MatrixType,
                                     ImmersXLA::MPI::PreconditionAMG>(
      matrix, prototype, additional_data);
  }
} // namespace ImmersX

#endif // immersx_local_preconditioner_h
