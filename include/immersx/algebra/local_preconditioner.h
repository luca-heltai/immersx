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
#include <deal.II/lac/vector_memory.h>

#include <immersx/algebra/linear_algebra.h>

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace ImmersX
{
  namespace detail
  {
    template <typename PreconditionerType, typename VectorType, typename = void>
    struct has_transpose_apply : std::false_type
    {};

    template <typename PreconditionerType, typename VectorType>
    struct has_transpose_apply<
      PreconditionerType,
      VectorType,
      std::void_t<decltype(std::declval<const PreconditionerType &>().Tvmult(
        std::declval<VectorType &>(),
        std::declval<const VectorType &>()))>> : std::true_type
    {};

    template <typename PreconditionerType, typename VectorType, typename = void>
    struct has_distributed_preconditioner_maps : std::false_type
    {};

    template <typename PreconditionerType, typename VectorType>
    struct has_distributed_preconditioner_maps<
      PreconditionerType,
      VectorType,
      std::void_t<decltype(std::declval<const PreconditionerType &>()
                             .locally_owned_range_indices()),
                  decltype(std::declval<const PreconditionerType &>()
                             .locally_owned_domain_indices()),
                  decltype(std::declval<VectorType &>().reinit(
                    std::declval<dealii::IndexSet>(),
                    std::declval<MPI_Comm>(),
                    false))>> : std::true_type
    {};

    template <typename MatrixType, typename = void>
    struct has_matrix_communicator : std::false_type
    {};

    template <typename MatrixType>
    struct has_matrix_communicator<
      MatrixType,
      std::void_t<
        decltype(std::declval<const MatrixType &>().get_mpi_communicator())>>
      : std::true_type
    {};

    /** @cond INTERNAL */
    template <typename VectorType, typename PreconditionerType>
    std::function<void(VectorType &, bool)>
    make_range_reinitializer(
      const PreconditionerType                      &preconditioner,
      const MPI_Comm                                 communicator,
      const std::function<void(VectorType &, bool)> &fallback)
    {
      if constexpr (has_distributed_preconditioner_maps<PreconditionerType,
                                                        VectorType>::value)
        {
          const auto owned = preconditioner.locally_owned_range_indices();
          return [owned, communicator](VectorType &vector,
                                       const bool  omit_zeroing_entries) {
            vector.reinit(owned, communicator, omit_zeroing_entries);
          };
        }
      else
        return fallback;
    }

    template <typename VectorType, typename PreconditionerType>
    std::function<void(VectorType &, bool)>
    make_domain_reinitializer(
      const PreconditionerType                      &preconditioner,
      const MPI_Comm                                 communicator,
      const std::function<void(VectorType &, bool)> &fallback)
    {
      if constexpr (has_distributed_preconditioner_maps<PreconditionerType,
                                                        VectorType>::value)
        {
          const auto owned = preconditioner.locally_owned_domain_indices();
          return [owned, communicator](VectorType &vector,
                                       const bool  omit_zeroing_entries) {
            vector.reinit(owned, communicator, omit_zeroing_entries);
          };
        }
      else
        return fallback;
    }
    /** @endcond */
  } // namespace detail

  /**
   * Build a lifetime-safe LinearOperator around a deal.II preconditioner.
   *
   * The preconditioner observes the matrix passed to initialize(). The caller
   * must keep that concrete matrix alive for as long as this operator is used.
   */
  template <typename VectorType,
            typename MatrixType,
            typename PreconditionerType>
  dealii::LinearOperator<VectorType>
  make_local_preconditioner(
    const MatrixType                                  &matrix,
    const std::function<void(VectorType &, bool)>     &reinit_vector,
    const typename PreconditionerType::AdditionalData &additional_data = {})
  {
    auto preconditioner = std::make_shared<PreconditionerType>();
    preconditioner->initialize(matrix, additional_data);
    auto vector_memory =
      std::make_shared<dealii::GrowingVectorMemory<VectorType>>();

    dealii::LinearOperator<VectorType> result;
    // Backend matrix/preconditioner maps, not semantic IndexSets, define the
    // vectors accepted by the backend apply operation.
    if constexpr (detail::has_distributed_preconditioner_maps<
                    PreconditionerType,
                    VectorType>::value &&
                  detail::has_matrix_communicator<MatrixType>::value)
      {
        const auto communicator = matrix.get_mpi_communicator();
        result.reinit_range_vector =
          detail::make_range_reinitializer(*preconditioner,
                                           communicator,
                                           reinit_vector);
        result.reinit_domain_vector =
          detail::make_domain_reinitializer(*preconditioner,
                                            communicator,
                                            reinit_vector);
      }
    else
      {
        result.reinit_range_vector  = reinit_vector;
        result.reinit_domain_vector = result.reinit_range_vector;
      }
    result.vmult = [preconditioner](VectorType &dst, const VectorType &src) {
      preconditioner->vmult(dst, src);
    };
    result.vmult_add = [preconditioner, vector_memory](VectorType       &dst,
                                                       const VectorType &src) {
      typename dealii::VectorMemory<VectorType>::Pointer contribution(
        *vector_memory);
      contribution->reinit(dst);
      preconditioner->vmult(*contribution, src);
      dst += *contribution;
    };
    result.Tvmult = [preconditioner](VectorType &dst, const VectorType &src) {
      if constexpr (detail::has_transpose_apply<PreconditionerType,
                                                VectorType>::value)
        preconditioner->Tvmult(dst, src);
      else
        AssertThrow(false,
                    dealii::ExcMessage(
                      "The local preconditioner has no transpose action."));
    };
    result.Tvmult_add = [preconditioner, vector_memory](VectorType       &dst,
                                                        const VectorType &src) {
      if constexpr (detail::has_transpose_apply<PreconditionerType,
                                                VectorType>::value)
        {
          typename dealii::VectorMemory<VectorType>::Pointer contribution(
            *vector_memory);
          contribution->reinit(dst);
          preconditioner->Tvmult(*contribution, src);
          dst += *contribution;
        }
      else
        AssertThrow(false,
                    dealii::ExcMessage(
                      "The local preconditioner has no transpose action."));
    };
    return result;
  }

  /** Build a local AMG approximate inverse using the active linear backend. */
  template <typename VectorType, typename MatrixType>
  dealii::LinearOperator<VectorType>
  make_amg_preconditioner(
    const MatrixType                              &matrix,
    const std::function<void(VectorType &, bool)> &reinit_vector,
    const typename ImmersXLA::MPI::PreconditionAMG::AdditionalData
      &additional_data = {})
  {
    return make_local_preconditioner<VectorType,
                                     MatrixType,
                                     ImmersXLA::MPI::PreconditionAMG>(
      matrix, reinit_vector, additional_data);
  }
} // namespace ImmersX

#endif // immersx_local_preconditioner_h
