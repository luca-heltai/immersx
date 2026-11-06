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

#include <deal.II/base/mpi.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.h>

#include <immersx/algebra/linear_algebra.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

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


    /** Adapt a replicated serial action to uniquely owned MPI vectors. */
    template <typename VectorType, typename Action>
    dealii::LinearOperator<VectorType>
    replicated_vector_operator(
      const dealii::types::global_dof_index          size,
      const MPI_Comm                                 communicator,
      const std::function<void(VectorType &, bool)> &reinit_vector,
      Action                                         action)
    {
      VectorType prototype;
      reinit_vector(prototype, false);
      const auto owned = prototype.locally_owned_elements();
      const auto apply = [size, communicator, reinit_vector, owned, action](
                           VectorType       &destination,
                           const VectorType &source,
                           const bool        transpose) {
        const bool matching =
          source.size() == size && source.locally_owned_elements() == owned;
        AssertThrow(dealii::Utilities::MPI::min(
                      static_cast<unsigned int>(matching), communicator) == 1,
                    dealii::ExcMessage(
                      "A replicated action received a changed vector layout."));
        std::vector<double> local(size, 0.), global(size, 0.);
        for (const auto i : source.locally_owned_elements())
          local[i] = source(i);
        dealii::Utilities::MPI::sum(local, communicator, global);
        AssertThrow(
          std::all_of(global.begin(),
                      global.end(),
                      [](const double x) { return std::isfinite(x); }),
          dealii::ExcMessage("A replicated action received nonfinite values."));
        dealii::Vector<double> input(size), output(size);
        for (unsigned int i = 0; i < size; ++i)
          input(i) = global[i];
        action(output, input, transpose);
        const bool finite =
          std::all_of(output.begin(), output.end(), [](const double x) {
            return std::isfinite(x);
          });
        AssertThrow(
          dealii::Utilities::MPI::min(static_cast<unsigned int>(finite),
                                      communicator) == 1,
          dealii::ExcMessage("A replicated action returned nonfinite values."));
        // Imported native vectors can retain an Insert state. Overwrite a fresh
        // owned vector before assigning the replicated result.
        reinit_vector(destination, false);
        for (const auto i : destination.locally_owned_elements())
          destination(i) = output(i);
        destination.compress(dealii::VectorOperation::insert);
      };
      auto memory = std::make_shared<dealii::GrowingVectorMemory<VectorType>>();
      dealii::LinearOperator<VectorType> result;
      result.reinit_range_vector  = reinit_vector;
      result.reinit_domain_vector = reinit_vector;
      result.vmult = [apply](VectorType &dst, const VectorType &src) {
        apply(dst, src, false);
      };
      result.Tvmult = [apply](VectorType &dst, const VectorType &src) {
        apply(dst, src, true);
      };
      result.vmult_add = [apply, memory](VectorType       &dst,
                                         const VectorType &src) {
        typename dealii::VectorMemory<VectorType>::Pointer tmp(*memory);
        apply(*tmp, src, false);
        dst += *tmp;
      };
      result.Tvmult_add = [apply, memory](VectorType       &dst,
                                          const VectorType &src) {
        typename dealii::VectorMemory<VectorType>::Pointer tmp(*memory);
        apply(*tmp, src, true);
        dst += *tmp;
      };
      return result;
    }

    inline void
    check_replicated_matrix_size(const dealii::types::global_dof_index size,
                                 const unsigned int maximum_size,
                                 const MPI_Comm     communicator)
    {
      const bool valid =
        size > 0 && size <= maximum_size &&
        size <=
          static_cast<unsigned int>(std::numeric_limits<int>::max()) / size;
      AssertThrow(
        dealii::Utilities::MPI::min(static_cast<unsigned int>(valid),
                                    communicator) == 1,
        dealii::ExcMessage(
          "The replicated matrix exceeds its nonzero size limit or MPI count range."));
      AssertThrow(
        dealii::Utilities::MPI::min(maximum_size, communicator) ==
            dealii::Utilities::MPI::max(maximum_size, communicator) &&
          dealii::Utilities::MPI::min(size, communicator) ==
            dealii::Utilities::MPI::max(size, communicator),
        dealii::ExcMessage(
          "Replicated matrix dimensions or limits differ across ranks."));
    }

    /** Sample one fixed square operator; all ranks retain the small matrix. */
    template <typename VectorType>
    dealii::LinearOperator<VectorType>
    materialize_replicated_operator(
      const dealii::LinearOperator<VectorType> &op,
      const MPI_Comm                            communicator,
      const unsigned int                        maximum_size)
    {
      VectorType basis, image;
      op.reinit_domain_vector(basis, false);
      op.reinit_range_vector(image, false);
      const auto size = basis.size();
      check_replicated_matrix_size(size, maximum_size, communicator);
      const bool same_layout =
        size == image.size() &&
        basis.locally_owned_elements() == image.locally_owned_elements();
      AssertThrow(
        dealii::Utilities::MPI::min(static_cast<unsigned int>(same_layout),
                                    communicator) == 1,
        dealii::ExcMessage(
          "Materialization requires matching square vector layouts."));
      std::vector<unsigned int> local_owners(size, 0), owners(size, 0);
      for (const auto i : basis.locally_owned_elements())
        local_owners[i] = 1;
      dealii::Utilities::MPI::sum(local_owners, communicator, owners);
      AssertThrow(std::all_of(owners.begin(),
                              owners.end(),
                              [](const unsigned int n) { return n == 1; }),
                  dealii::ExcMessage(
                    "Materialization requires unique vector ownership."));
      auto matrix = std::make_shared<dealii::FullMatrix<double>>(size, size);
      std::vector<double> local(size, 0.), global(size, 0.);
      for (unsigned int column = 0; column < size; ++column)
        {
          basis = 0.;
          if (basis.locally_owned_elements().is_element(column))
            basis(column) = 1.;
          basis.compress(dealii::VectorOperation::insert);
          op.vmult(image, basis);
          std::fill(local.begin(), local.end(), 0.);
          for (const auto i : image.locally_owned_elements())
            local[i] = image(i);
          dealii::Utilities::MPI::sum(local, communicator, global);
          for (unsigned int row = 0; row < size; ++row)
            {
              AssertThrow(std::isfinite(global[row]),
                          dealii::ExcMessage(
                            "A materialized column is not finite."));
              (*matrix)(row, column) = global[row];
            }
        }
      return replicated_vector_operator<VectorType>(
        size,
        communicator,
        op.reinit_domain_vector,
        [matrix](dealii::Vector<double>       &dst,
                 const dealii::Vector<double> &src,
                 const bool                    transpose) {
          if (transpose)
            matrix->Tvmult(dst, src);
          else
            matrix->vmult(dst, src);
        });
    }

  } // namespace detail

  /**
   * Build a LinearOperator around a deal.II preconditioner.
   *
   * The returned operator has the same public domain and range as the field
   * vector reinitializer. Backend maps and auxiliary vector spaces remain
   * private to the concrete preconditioner.
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
    result.reinit_range_vector  = reinit_vector;
    result.reinit_domain_vector = result.reinit_range_vector;
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


  /** Factor a bounded square MPI field block with replicated LAPACK LU.
   *
   * Every rank, including ranks without owned rows, participates in setup and
   * each application. The factorization owns its matrix and supports Tvmult.
   * Storage is O(n^2) per rank; maximum_size is an explicit resource bound.
   */
  template <typename VectorType, typename MatrixType>
  dealii::LinearOperator<VectorType>
  make_replicated_direct_preconditioner(
    const MatrixType                              &matrix,
    const std::function<void(VectorType &, bool)> &reinit_vector,
    const unsigned int                             maximum_size)
  {
    const auto size         = matrix.m();
    const auto communicator = matrix.get_mpi_communicator();
    detail::check_replicated_matrix_size(size, maximum_size, communicator);
    AssertThrow(
      dealii::Utilities::MPI::min(static_cast<unsigned int>(size == matrix.n()),
                                  communicator) == 1,
      dealii::ExcMessage("A replicated inverse requires a square matrix."));
    VectorType prototype;
    reinit_vector(prototype, false);
    const bool layout_matches = prototype.size() == size &&
                                prototype.locally_owned_elements() ==
                                  matrix.locally_owned_range_indices() &&
                                prototype.locally_owned_elements() ==
                                  matrix.locally_owned_domain_indices();
    AssertThrow(
      dealii::Utilities::MPI::min(static_cast<unsigned int>(layout_matches),
                                  communicator) == 1,
      dealii::ExcMessage("The inverse matrix and field layouts differ."));
    std::vector<double>       local(size * size, 0.), global(size * size, 0.);
    std::vector<unsigned int> local_owners(size, 0), owners(size, 0);
    for (const auto row : matrix.locally_owned_range_indices())
      {
        local_owners[row] = 1;
        for (auto entry = matrix.begin(row); entry != matrix.end(row); ++entry)
          local[row * size + entry->column()] += entry->value();
      }
    dealii::Utilities::MPI::sum(local_owners, communicator, owners);
    dealii::Utilities::MPI::sum(local, communicator, global);
    AssertThrow(
      std::all_of(owners.begin(),
                  owners.end(),
                  [](const unsigned int n) { return n == 1; }),
      dealii::ExcMessage(
        "A replicated inverse requires unique matrix row ownership."));
    auto factor =
      std::make_shared<dealii::LAPACKFullMatrix<double>>(size, size);
    for (unsigned int row = 0; row < size; ++row)
      for (unsigned int column = 0; column < size; ++column)
        {
          const double value = global[row * size + column];
          AssertThrow(std::isfinite(value),
                      dealii::ExcMessage(
                        "The replicated inverse matrix is not finite."));
          (*factor)(row, column) = value;
        }
    factor->compute_lu_factorization();
    return detail::replicated_vector_operator<VectorType>(
      size,
      communicator,
      reinit_vector,
      [factor](dealii::Vector<double>       &dst,
               const dealii::Vector<double> &src,
               const bool                    transpose) {
        dst = src;
        factor->solve(dst, transpose);
      });
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
