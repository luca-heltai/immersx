// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_matrix_operator_h
#define immersx_matrix_operator_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/linear_operator.h>

#include <functional>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace ImmersX
{
  /** Erase a backend-specific LinearOperator payload at an execution boundary.
   */
  template <typename Range, typename Domain, typename Payload>
  dealii::LinearOperator<Range, Domain>
  payload_free(const dealii::LinearOperator<Range, Domain, Payload> &op)
  {
    dealii::LinearOperator<Range, Domain> result;
    result.vmult = [op](Range &dst, const Domain &src) { op.vmult(dst, src); };
    result.vmult_add = [op](Range &dst, const Domain &src) {
      op.vmult_add(dst, src);
    };
    result.Tvmult = [op](Domain &dst, const Range &src) {
      op.Tvmult(dst, src);
    };
    result.Tvmult_add = [op](Domain &dst, const Range &src) {
      op.Tvmult_add(dst, src);
    };
    result.reinit_range_vector = [op](Range &dst, const bool omit) {
      op.reinit_range_vector(dst, omit);
    };
    result.reinit_domain_vector = [op](Domain &dst, const bool omit) {
      op.reinit_domain_vector(dst, omit);
    };
    return result;
  }

  /**
   * A linear operator together with optional matrix provenance.
   *
   * The operator view is deliberately payload-free so it can be combined with
   * operators from other backends.  The materializer is the only supported way
   * to recover matrix data: callers must never try to infer a matrix from a
   * generic LinearOperator.
   */
  template <typename VectorType, typename MatrixType>
  struct MaterializedOperator
  {
    using Operator    = dealii::LinearOperator<VectorType, VectorType>;
    using Matrix      = MatrixType;
    using Materialize = std::function<std::unique_ptr<MatrixType>()>;

    Operator    view;
    Materialize materialize;

    bool
    is_materializable() const
    {
      return static_cast<bool>(materialize);
    }

    std::unique_ptr<MatrixType>
    matrix() const
    {
      AssertThrow(is_materializable(),
                  dealii::ExcMessage(
                    "This operator has no matrix provenance."));
      return materialize();
    }
  };

  namespace detail
  {
    /** Clone both copyable deal.II matrices and non-copyable Trilinos ones. */
    template <typename MatrixType>
    std::unique_ptr<MatrixType>
    clone_matrix(const MatrixType &matrix)
    {
      if constexpr (std::is_copy_constructible<MatrixType>::value)
        return std::make_unique<MatrixType>(matrix);
      else
        {
          auto result = std::make_unique<MatrixType>();
          result->reinit(matrix);
          result->copy_from(matrix);
          return result;
        }
    }

    template <typename MatrixType, typename = void>
    struct has_in_place_transpose : std::false_type
    {};

    template <typename MatrixType>
    struct has_in_place_transpose<
      MatrixType,
      std::void_t<decltype(std::declval<MatrixType &>().transpose())>>
      : std::true_type
    {};

    template <typename MatrixType, typename = void>
    struct has_distributed_partitions : std::false_type
    {};

    template <typename MatrixType>
    struct has_distributed_partitions<
      MatrixType,
      std::void_t<
        decltype(std::declval<const MatrixType &>()
                   .locally_owned_domain_indices()),
        decltype(std::declval<const MatrixType &>()
                   .locally_owned_range_indices()),
        decltype(std::declval<const MatrixType &>().get_mpi_communicator())>>
      : std::true_type
    {};

    template <typename MatrixType>
    std::unique_ptr<MatrixType>
    transpose_matrix(std::unique_ptr<MatrixType> matrix)
    {
      if constexpr (has_distributed_partitions<MatrixType>::value)
        {
          using size_type = typename MatrixType::size_type;
          using Entry     = std::pair<std::pair<size_type, size_type>, double>;

          // Trilinos' transpose() toggles the apply flag but preserves the
          // matrix shape.  Materialization instead needs a true matrix with
          // exchanged row and column spaces.  Route each source entry to the
          // rank owning its new row before constructing the target sparsity
          // pattern, so rectangular distributed transposes retain remote
          // entries as well as local ones.
          const auto row_partition = matrix->locally_owned_domain_indices();
          const auto col_partition = matrix->locally_owned_range_indices();
          const auto communicator  = matrix->get_mpi_communicator();

          dealii::IndexSet columns(matrix->n());
          for (const auto row : col_partition)
            for (auto entry = matrix->begin(row); entry != matrix->end(row);
                 ++entry)
              columns.add_index(entry->column());
          columns.compress();

          const auto owners =
            dealii::Utilities::MPI::compute_index_owner(row_partition,
                                                        columns,
                                                        communicator);
          std::map<size_type, unsigned int> owner_by_column;
          auto                              column = columns.begin();
          for (unsigned int i = 0; i < owners.size(); ++i, ++column)
            owner_by_column.emplace(*column, owners[i]);

          std::map<unsigned int, std::vector<Entry>> to_send;
          for (const auto row : col_partition)
            for (auto entry = matrix->begin(row); entry != matrix->end(row);
                 ++entry)
              to_send[owner_by_column.at(entry->column())].push_back(
                {{entry->column(), row}, entry->value()});

          const auto received =
            dealii::Utilities::MPI::some_to_some(communicator, to_send);
          dealii::DynamicSparsityPattern sparsity(matrix->n(),
                                                  matrix->m(),
                                                  row_partition);
          std::vector<Entry>             local_entries;
          for (const auto &[rank, entries] : received)
            {
              (void)rank;
              for (const auto &entry : entries)
                {
                  sparsity.add(entry.first.first, entry.first.second);
                  local_entries.push_back(entry);
                }
            }

          auto result = std::make_unique<MatrixType>();
          result->reinit(
            row_partition, col_partition, sparsity, communicator, false);
          for (const auto &entry : local_entries)
            result->set(entry.first.first, entry.first.second, entry.second);
          result->compress(dealii::VectorOperation::insert);
          return result;
        }
      else
        {
          auto result = std::make_unique<MatrixType>(matrix->n(), matrix->m());
          for (typename MatrixType::size_type i = 0; i < matrix->m(); ++i)
            for (typename MatrixType::size_type j = 0; j < matrix->n(); ++j)
              (*result)(j, i) = (*matrix)(i, j);
          return result;
        }
    }
  } // namespace detail

  /** Build a payload-free operator while retaining the source matrix. */
  template <typename VectorType, typename MatrixType>
  MaterializedOperator<VectorType, MatrixType>
  matrix_operator(const MatrixType &matrix)
  {
    MaterializedOperator<VectorType, MatrixType> result;
    result.view =
      payload_free(dealii::linear_operator<VectorType, VectorType>(matrix));
    result.materialize = [&matrix]() { return detail::clone_matrix(matrix); };
    return result;
  }

  /** Scale a matrix-backed operator without losing its matrix provenance. */
  template <typename VectorType, typename MatrixType>
  MaterializedOperator<VectorType, MatrixType>
  operator*(const double                                        factor,
            const MaterializedOperator<VectorType, MatrixType> &source)
  {
    auto result = source;
    result.view = factor * source.view;
    if (source.is_materializable())
      result.materialize = [factor, source]() {
        auto matrix = source.matrix();
        *matrix *= factor;
        return matrix;
      };
    return result;
  }

  template <typename VectorType, typename MatrixType>
  MaterializedOperator<VectorType, MatrixType>
  operator*(const MaterializedOperator<VectorType, MatrixType> &source,
            const double                                        factor)
  {
    return factor * source;
  }

  /** Transpose a matrix-backed operator without losing its provenance. */
  template <typename VectorType, typename MatrixType>
  MaterializedOperator<VectorType, MatrixType>
  transpose_operator(const MaterializedOperator<VectorType, MatrixType> &source)
  {
    auto result = source;
    result.view = dealii::transpose_operator(source.view);
    if (source.is_materializable())
      result.materialize = [source]() {
        return detail::transpose_matrix(source.matrix());
      };
    return result;
  }
} // namespace ImmersX

#endif // immersx_matrix_operator_h
