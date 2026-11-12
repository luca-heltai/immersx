// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#ifndef immersx_mumps_preconditioner_h
#define immersx_mumps_preconditioner_h

#include <deal.II/base/mpi.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector_memory.h>

#include <functional>
#include <memory>

namespace ImmersX
{
#ifdef DEAL_II_WITH_MUMPS
  /** Factor one distributed field block, retaining the factor for all actions.
   * Redistribute the solution because native mixed fields need not have
   * contiguous ownership, whereas the deal.II MUMPS backsolve requires it.
   */
  template <typename VectorType, typename MatrixType>
  dealii::LinearOperator<VectorType>
  make_mumps_preconditioner(
    const MatrixType                              &matrix,
    const std::function<void(VectorType &, bool)> &reinit_vector,
    const MPI_Comm                                 communicator)
  {
    AssertThrow(
      matrix.m() == matrix.n() &&
        matrix.m() >= dealii::Utilities::MPI::n_mpi_processes(communicator),
      dealii::ExcMessage(
        "MUMPS field block must be square with at least one row per rank."));
    auto factor = std::make_shared<dealii::SparseDirectMUMPS>(
      dealii::SparseDirectMUMPS::AdditionalData(), communicator);
    factor->initialize(matrix);
    auto solution = std::make_shared<VectorType>();
    solution->reinit(
      dealii::Utilities::MPI::create_evenly_distributed_partitioning(
        communicator, matrix.m()),
      communicator);
    auto memory = std::make_shared<dealii::GrowingVectorMemory<VectorType>>();
    const auto apply = [factor, solution, reinit_vector](VectorType       &dst,
                                                         const VectorType &src,
                                                         const bool transpose) {
      VectorType source(src);
      source.compress(dealii::VectorOperation::insert);
      if (transpose)
        factor->Tvmult(*solution, source);
      else
        factor->vmult(*solution, source);
      reinit_vector(dst, false);
      dealii::LinearAlgebra::ReadWriteVector<double> imported(
        dst.locally_owned_elements());
      imported.import_elements(*solution, dealii::VectorOperation::insert);
      for (const auto index : dst.locally_owned_elements())
        dst[index] = imported[index];
      dst.compress(dealii::VectorOperation::insert);
    };
    dealii::LinearOperator<VectorType> result;
    result.reinit_range_vector  = reinit_vector;
    result.reinit_domain_vector = reinit_vector;
    result.vmult                = [apply](auto &dst, const auto &src) {
      apply(dst, src, false);
    };
    result.Tvmult = [apply](auto &dst, const auto &src) {
      apply(dst, src, true);
    };
    result.vmult_add = [apply, memory](auto &dst, const auto &src) {
      typename dealii::VectorMemory<VectorType>::Pointer temporary(*memory);
      temporary->reinit(dst);
      apply(*temporary, src, false);
      dst += *temporary;
    };
    result.Tvmult_add = [apply, memory](auto &dst, const auto &src) {
      typename dealii::VectorMemory<VectorType>::Pointer temporary(*memory);
      temporary->reinit(dst);
      apply(*temporary, src, true);
      dst += *temporary;
    };
    return result;
  }
#endif
} // namespace ImmersX
#endif
