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

#include <deal.II/lac/linear_operator.h>

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

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

    template <typename MatrixType>
    std::unique_ptr<MatrixType>
    transpose_matrix(std::unique_ptr<MatrixType> matrix)
    {
      if constexpr (has_in_place_transpose<MatrixType>::value)
        {
          matrix->transpose();
          return matrix;
        }
      else
        {
          auto result = std::make_unique<MatrixType>(matrix->n(), matrix->m());
          for (typename MatrixType::size_type i = 0; i < matrix->n(); ++i)
            for (typename MatrixType::size_type j = 0; j < matrix->m(); ++j)
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
