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
#include <deal.II/base/observer_pointer.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparsity_pattern.h>

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
    using Matrix          = MatrixType;
    using Operator        = dealii::LinearOperator<VectorType, VectorType>;
    using MatrixObserver  = dealii::ObserverPointer<const MatrixType>;
    using Materialize     = std::function<std::shared_ptr<MatrixType>()>;
    using MaterializeInto = std::function<void(MatrixType &)>;

    Operator        view;
    Materialize     materialize;
    MaterializeInto materialize_into;
    MatrixObserver  observed_matrix;
    bool            observes_matrix = false;
    bool            direct_matrix   = false;

    bool
    is_materializable() const
    {
      return static_cast<bool>(materialize);
    }

    std::shared_ptr<MatrixType>
    matrix() const
    {
      AssertThrow(is_materializable(),
                  dealii::ExcMessage(
                    "This operator has no matrix provenance."));
      return materialize();
    }

    /** Materialize directly into the caller's concrete destination. */
    void
    materialize_into_matrix(MatrixType &destination) const
    {
      AssertThrow(static_cast<bool>(materialize_into),
                  dealii::ExcMessage(
                    "This operator cannot materialize into a destination."));
      materialize_into(destination);
    }

    /** Return the observed source for a leaf matrix-backed operator. */
    const MatrixType *
    source_matrix() const
    {
      return direct_matrix ? observed_matrix.get() : nullptr;
    }
  };

  namespace detail
  {
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

    template <typename MatrixType, typename = void>
    struct has_sparse_reinit : std::false_type
    {};

    template <typename MatrixType>
    struct has_sparse_reinit<
      MatrixType,
      std::void_t<decltype(std::declval<MatrixType &>().begin(
                    std::declval<typename MatrixType::size_type>())),
                  decltype(std::declval<MatrixType &>().reinit(
                    std::declval<const dealii::SparsityPattern &>()))>>
      : std::true_type
    {};

    template <typename MatrixType, typename = void>
    struct has_reinit_and_copy_from : std::false_type
    {};

    template <typename MatrixType>
    struct has_reinit_and_copy_from<
      MatrixType,
      std::void_t<decltype(std::declval<MatrixType &>().reinit(
                    std::declval<const MatrixType &>())),
                  decltype(std::declval<MatrixType &>().copy_from(
                    std::declval<const MatrixType &>()))>> : std::true_type
    {};

    template <typename MatrixType, typename = void>
    struct has_reinit_dimensions : std::false_type
    {};

    template <typename MatrixType>
    struct has_reinit_dimensions<
      MatrixType,
      std::void_t<decltype(std::declval<MatrixType &>().reinit(
        std::declval<typename MatrixType::size_type>(),
        std::declval<typename MatrixType::size_type>()))>> : std::true_type
    {};

    template <typename MatrixType>
    std::shared_ptr<dealii::SparsityPattern>
    make_sparsity(const MatrixType &source)
    {
      dealii::DynamicSparsityPattern dynamic_sparsity(source.m(), source.n());
      for (typename MatrixType::size_type i = 0; i < source.m(); ++i)
        for (auto entry = source.begin(i); entry != source.end(i); ++entry)
          dynamic_sparsity.add(i, entry->column());

      auto sparsity = std::make_shared<dealii::SparsityPattern>();
      sparsity->copy_from(dynamic_sparsity);
      return sparsity;
    }

    template <typename MatrixType>
    void
    set_matrix_values(MatrixType &destination, const MatrixType &source)
    {
      for (typename MatrixType::size_type i = 0; i < source.m(); ++i)
        for (auto entry = source.begin(i); entry != source.end(i); ++entry)
          destination.set(i, entry->column(), entry->value());
    }

    template <typename MatrixType>
    std::shared_ptr<MatrixType>
    make_matrix_with_sparsity(
      const std::shared_ptr<dealii::SparsityPattern> &sparsity)
    {
      auto result = std::shared_ptr<MatrixType>(
        new MatrixType(), [sparsity](MatrixType *pointer) { delete pointer; });
      result->reinit(*sparsity);
      return result;
    }

    template <typename MatrixType>
    void
    copy_matrix(MatrixType &destination, const MatrixType &source)
    {
      if constexpr (has_sparse_reinit<MatrixType>::value)
        {
          destination.reinit(source);
          destination.copy_from(source);
        }
      else if constexpr (has_reinit_dimensions<MatrixType>::value)
        {
          destination.reinit(source.m(), source.n());
          destination.copy_from(source);
        }
      else if constexpr (has_reinit_and_copy_from<MatrixType>::value)
        {
          destination.reinit(source);
          destination.copy_from(source);
        }
      else
        {
          static_assert(has_reinit_dimensions<MatrixType>::value,
                        "MatrixType has no supported direct copy operation.");
        }
    }

    /** Clone a matrix using its backend-native copy operation. */
    template <typename MatrixType>
    std::shared_ptr<MatrixType>
    clone_matrix(const MatrixType &matrix)
    {
      if constexpr (has_sparse_reinit<MatrixType>::value)
        {
          auto sparsity = make_sparsity(matrix);
          auto result   = make_matrix_with_sparsity<MatrixType>(sparsity);
          set_matrix_values(*result, matrix);
          return result;
        }
      else
        {
          auto result = std::make_shared<MatrixType>();
          copy_matrix(*result, matrix);
          return result;
        }
    }

    template <typename MatrixType>
    std::shared_ptr<MatrixType>
    transpose_matrix(std::shared_ptr<MatrixType> matrix)
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
      else if constexpr (has_sparse_reinit<MatrixType>::value)
        {
          dealii::DynamicSparsityPattern dynamic_sparsity(matrix->n(),
                                                          matrix->m());
          for (typename MatrixType::size_type i = 0; i < matrix->m(); ++i)
            for (auto entry = matrix->begin(i); entry != matrix->end(i);
                 ++entry)
              dynamic_sparsity.add(entry->column(), i);

          auto sparsity = std::make_shared<dealii::SparsityPattern>();
          sparsity->copy_from(dynamic_sparsity);
          auto result = make_matrix_with_sparsity<MatrixType>(sparsity);
          for (typename MatrixType::size_type i = 0; i < matrix->m(); ++i)
            for (auto entry = matrix->begin(i); entry != matrix->end(i);
                 ++entry)
              result->set(entry->column(), i, entry->value());
          return result;
        }
      else
        {
          auto result = std::make_shared<MatrixType>(matrix->n(), matrix->m());
          auto result = std::make_shared<MatrixType>(matrix->n(), matrix->m());
          for (typename MatrixType::size_type i = 0; i < matrix->m(); ++i)
            for (typename MatrixType::size_type j = 0; j < matrix->n(); ++j)
              (*result)(j, i) = (*matrix)(i, j);
          return result;
        }
    }

    template <typename MatrixOperatorRange>
    auto
    sum_matrices(const MatrixOperatorRange &operators)
    {
      using MatrixOperator = typename MatrixOperatorRange::value_type;
      using MatrixType     = typename MatrixOperator::Matrix;

      auto first = operators.front().matrix();
      if constexpr (has_sparse_reinit<MatrixType>::value)
        {
          dealii::DynamicSparsityPattern dynamic_sparsity(first->m(),
                                                          first->n());
          for (const auto &operator_description : operators)
            {
              const auto matrix = operator_description.matrix();
              for (typename MatrixType::size_type i = 0; i < matrix->m(); ++i)
                for (auto entry = matrix->begin(i); entry != matrix->end(i);
                     ++entry)
                  dynamic_sparsity.add(i, entry->column());
            }

          auto sparsity = std::make_shared<dealii::SparsityPattern>();
          sparsity->copy_from(dynamic_sparsity);
          auto result = make_matrix_with_sparsity<MatrixType>(sparsity);
          for (const auto &operator_description : operators)
            {
              const auto matrix = operator_description.matrix();
              for (typename MatrixType::size_type i = 0; i < matrix->m(); ++i)
                for (auto entry = matrix->begin(i); entry != matrix->end(i);
                     ++entry)
                  result->add(i, entry->column(), entry->value());
            }
          return result;
        }
      else
        {
          for (std::size_t i = 1; i < operators.size(); ++i)
            first->add(1., *operators[i].matrix());
          return first;
        }
    }
  } // namespace detail

  /** Build a payload-free operator while retaining the source matrix. */
  template <typename VectorType, typename MatrixType>
  MaterializedOperator<VectorType, MatrixType>
  matrix_operator(const MatrixType &matrix)
  {
    static_assert(
      std::is_base_of<dealii::EnableObserverPointer, MatrixType>::value,
      "matrix_operator requires an EnableObserverPointer matrix; pass an "
      "explicit lifetime-owning materializer for non-observable matrices.");

    MaterializedOperator<VectorType, MatrixType> result;
    result.view =
      payload_free(dealii::linear_operator<VectorType, VectorType>(matrix));
    result.observed_matrix = &matrix;
    result.observes_matrix = true;
    result.direct_matrix   = true;
    result.materialize     = [source = result.observed_matrix]() {
      return detail::clone_matrix(*source);
    };
    result.materialize_into = [source =
                                 result.observed_matrix](MatrixType &dst) {
      detail::copy_matrix(dst, *source);
    };
    return result;
  }

  /** Scale a matrix-backed operator without losing its matrix provenance. */
  template <typename VectorType, typename MatrixType>
  MaterializedOperator<VectorType, MatrixType>
  operator*(const double                                        factor,
            const MaterializedOperator<VectorType, MatrixType> &source)
  {
    auto result          = source;
    result.direct_matrix = false;
    result.view          = factor * source.view;
    if (source.is_materializable())
      result.materialize = [factor, source]() {
        auto matrix = source.matrix();
        *matrix *= factor;
        return matrix;
      };
    if (source.materialize_into)
      result.materialize_into = [factor, source](MatrixType &destination) {
        source.materialize_into_matrix(destination);
        destination *= factor;
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
    auto result          = source;
    result.direct_matrix = false;
    result.view          = dealii::transpose_operator(source.view);
    if (source.is_materializable())
      result.materialize = [source]() {
        return detail::transpose_matrix(source.matrix());
      };
    if (source.materialize_into)
      result.materialize_into = [source](MatrixType &destination) {
        auto transposed = detail::transpose_matrix(source.matrix());
        destination.reinit(*transposed);
        destination.copy_from(*transposed);
      };
    return result;
  }
} // namespace ImmersX

#endif // immersx_matrix_operator_h
