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
#include <limits>
#include <memory>

namespace ImmersX
{
#ifdef DEAL_II_WITH_MUMPS
  /** Factor one distributed field block, retaining the factor for all actions.
   * Redistribute the solution because native mixed fields need not have
   * contiguous ownership, whereas the deal.II MUMPS backsolve requires it.
   * workspace_relaxation overrides ICNTL(14), the extra workspace percentage;
   * -1 retains the native MUMPS default. memory_limit_mb sets ICNTL(23),
   * the working-memory limit in MB per rank; zero uses automatic allocation.
   * Each new factor reports its native memory estimates and allocation.
   * output_details retains native diagnostics for subsequent solves as well.
   * All ranks must pass the same controls.
   */
  template <typename VectorType, typename MatrixType>
  dealii::LinearOperator<VectorType>
  make_mumps_preconditioner(
    const MatrixType                              &matrix,
    const std::function<void(VectorType &, bool)> &reinit_vector,
    const MPI_Comm                                 communicator,
    const int                                      workspace_relaxation = -1,
    const unsigned int                             memory_limit_mb      = 0,
    const bool                                     output_details       = false)
  {
    const int minimum_workspace =
      dealii::Utilities::MPI::min(workspace_relaxation, communicator);
    const int maximum_workspace =
      dealii::Utilities::MPI::max(workspace_relaxation, communicator);
    AssertThrow(
      minimum_workspace >= -1 && minimum_workspace == maximum_workspace,
      dealii::ExcMessage(
        "MUMPS workspace relaxation must be at least -1 and equal on every rank."));
    const auto minimum_memory =
      dealii::Utilities::MPI::min(memory_limit_mb, communicator);
    const auto maximum_memory =
      dealii::Utilities::MPI::max(memory_limit_mb, communicator);
    AssertThrow(
      minimum_memory == maximum_memory &&
        maximum_memory <=
          static_cast<unsigned int>(std::numeric_limits<int>::max()),
      dealii::ExcMessage(
        "MUMPS memory limit must fit its integer range and be equal on every rank."));
    const int minimum_output =
      dealii::Utilities::MPI::min(output_details ? 1 : 0, communicator);
    const int maximum_output =
      dealii::Utilities::MPI::max(output_details ? 1 : 0, communicator);
    AssertThrow(minimum_output == maximum_output,
                dealii::ExcMessage(
                  "MUMPS output details must be equal on every rank."));
    AssertThrow(
      matrix.m() == matrix.n() &&
        matrix.m() >= dealii::Utilities::MPI::n_mpi_processes(communicator),
      dealii::ExcMessage(
        "MUMPS field block must be square with at least one row per rank."));
    auto factor = std::make_shared<dealii::SparseDirectMUMPS>(
      dealii::SparseDirectMUMPS::AdditionalData(true), communicator);
    if (workspace_relaxation >= 0)
      factor->get_icntl()[13] = workspace_relaxation;
    factor->get_icntl()[22] = static_cast<int>(memory_limit_mb);
    factor->initialize(matrix);
    if (!output_details)
      {
        auto controls = factor->get_icntl();
        controls[0] = controls[1] = controls[2] = -1;
        controls[3]                             = 0;
      }
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
