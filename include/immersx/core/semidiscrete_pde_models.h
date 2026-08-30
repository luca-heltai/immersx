// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_semidiscrete_pde_models_h
#define immersx_semidiscrete_pde_models_h

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/affine_constraints.h>

#include <immersx/core/matrix_operator.h>
#include <immersx/core/time_residual.h>

#include <type_traits>
#include <utility>

namespace ImmersX
{
  /** Small matrix and constraint helpers shared by physics contributors. */
  namespace semidiscrete_detail
  {
    template <typename MatrixType, typename = void>
    struct has_local_row_partition : std::false_type
    {};

    template <typename MatrixType>
    struct has_local_row_partition<
      MatrixType,
      std::void_t<decltype(std::declval<const MatrixType &>()
                             .locally_owned_range_indices())>> : std::true_type
    {};

    template <typename MatrixType>
    void
    zero_constrained_rows(MatrixType                              &matrix,
                          const dealii::AffineConstraints<double> &constraints,
                          const dealii::types::global_dof_index    offset = 0)
    {
      if constexpr (has_local_row_partition<MatrixType>::value)
        {
          for (const auto row : matrix.locally_owned_range_indices())
            if (constraints.is_constrained(offset + row))
              for (auto entry = matrix.begin(row); entry != matrix.end(row);
                   ++entry)
                matrix.set(row, entry->column(), 0.);
        }
      else
        {
          for (typename MatrixType::size_type row = 0; row < matrix.m(); ++row)
            if (constraints.is_constrained(offset + row))
              for (auto entry = matrix.begin(row); entry != matrix.end(row);
                   ++entry)
                matrix.set(row, entry->column(), 0.);
        }
      matrix.compress(dealii::VectorOperation::insert);
    }

    template <typename VectorType>
    dealii::PackagedOperation<VectorType>
    constrained_operation(dealii::PackagedOperation<VectorType>    operation,
                          const dealii::AffineConstraints<double> &constraints)
    {
      dealii::PackagedOperation<VectorType> result;
      result.reinit_vector = operation.reinit_vector;
      result.apply = [operation, &constraints](VectorType &destination) {
        operation.apply(destination);
        for (const auto index : destination.locally_owned_elements())
          if (constraints.is_constrained(index))
            destination(index) = 0.;
      };
      result.apply_add = [operation, &constraints](VectorType &destination) {
        VectorType contribution;
        operation.reinit_vector(contribution, false);
        operation.apply(contribution);
        for (const auto index : contribution.locally_owned_elements())
          if (constraints.is_constrained(index))
            contribution(index) = 0.;
        destination += contribution;
      };
      return result;
    }

    template <typename VectorType>
    dealii::LinearOperator<VectorType, VectorType>
    constrained_operator(
      const dealii::LinearOperator<VectorType, VectorType> &operator_view,
      const dealii::AffineConstraints<double>              &constraints)
    {
      auto result  = operator_view;
      result.vmult = [operator_view, &constraints](VectorType &destination,
                                                   const VectorType &source) {
        operator_view.vmult(destination, source);
        for (const auto index : destination.locally_owned_elements())
          if (constraints.is_constrained(index))
            destination(index) = 0.;
      };
      result.vmult_add = [operator_view,
                          &constraints](VectorType       &destination,
                                        const VectorType &source) {
        VectorType contribution;
        operator_view.reinit_range_vector(contribution, false);
        operator_view.vmult(contribution, source);
        for (const auto index : contribution.locally_owned_elements())
          if (constraints.is_constrained(index))
            contribution(index) = 0.;
        destination += contribution;
      };
      result.Tvmult = [operator_view, &constraints](VectorType &destination,
                                                    const VectorType &source) {
        VectorType projected;
        operator_view.reinit_range_vector(projected, false);
        projected = source;
        for (const auto index : projected.locally_owned_elements())
          if (constraints.is_constrained(index))
            projected(index) = 0.;
        operator_view.Tvmult(destination, projected);
      };
      result.Tvmult_add = [operator_view,
                           &constraints](VectorType       &destination,
                                         const VectorType &source) {
        VectorType projected;
        operator_view.reinit_range_vector(projected, false);
        projected = source;
        for (const auto index : projected.locally_owned_elements())
          if (constraints.is_constrained(index))
            projected(index) = 0.;
        operator_view.Tvmult_add(destination, projected);
      };
      return result;
    }

    template <typename VectorType, typename MatrixType>
    MaterializedOperator<VectorType, MatrixType>
    constrained_matrix_operator(
      const MaterializedOperator<VectorType, MatrixType> &source,
      const dealii::AffineConstraints<double>            &constraints)
    {
      auto result        = source;
      result.view        = constrained_operator(source.view, constraints);
      result.materialize = [source, &constraints]() {
        auto matrix = source.matrix();
        zero_constrained_rows(*matrix, constraints);
        return matrix;
      };
      result.materialize_into = [source,
                                 &constraints](MatrixType &destination) {
        source.materialize_into_matrix(destination);
        zero_constrained_rows(destination, constraints);
      };
      result.direct_matrix = false;
      return result;
    }

    template <typename VectorType>
    dealii::PackagedOperation<VectorType>
    mixed_constrained_operation(
      dealii::PackagedOperation<VectorType>    operation,
      const dealii::AffineConstraints<double> &constraints,
      const dealii::types::global_dof_index    offset)
    {
      dealii::PackagedOperation<VectorType> result;
      result.reinit_vector = operation.reinit_vector;
      result.apply =
        [operation, &constraints, offset](VectorType &destination) {
          operation.apply(destination);
          for (const auto index : destination.locally_owned_elements())
            if (constraints.is_constrained(offset + index))
              destination(index) = 0.;
        };
      result.apply_add =
        [operation, &constraints, offset](VectorType &destination) {
          VectorType contribution;
          operation.reinit_vector(contribution, false);
          operation.apply(contribution);
          for (const auto index : contribution.locally_owned_elements())
            if (constraints.is_constrained(offset + index))
              contribution(index) = 0.;
          destination += contribution;
        };
      return result;
    }

    template <typename VectorType>
    dealii::LinearOperator<VectorType, VectorType>
    mixed_constrained_operator(
      const dealii::LinearOperator<VectorType, VectorType> &operator_view,
      const dealii::AffineConstraints<double>              &constraints,
      const dealii::types::global_dof_index                 offset)
    {
      auto result = operator_view;
      result.vmult =
        [operator_view, &constraints, offset](VectorType       &destination,
                                              const VectorType &source) {
          operator_view.vmult(destination, source);
          for (const auto index : destination.locally_owned_elements())
            if (constraints.is_constrained(offset + index))
              destination(index) = 0.;
        };
      result.vmult_add =
        [operator_view, &constraints, offset](VectorType       &destination,
                                              const VectorType &source) {
          VectorType contribution;
          operator_view.reinit_range_vector(contribution, false);
          operator_view.vmult(contribution, source);
          for (const auto index : contribution.locally_owned_elements())
            if (constraints.is_constrained(offset + index))
              contribution(index) = 0.;
          destination += contribution;
        };
      result.Tvmult =
        [operator_view, &constraints, offset](VectorType       &destination,
                                              const VectorType &source) {
          VectorType projected;
          operator_view.reinit_range_vector(projected, false);
          projected = source;
          for (const auto index : projected.locally_owned_elements())
            if (constraints.is_constrained(offset + index))
              projected(index) = 0.;
          operator_view.Tvmult(destination, projected);
        };
      result.Tvmult_add =
        [operator_view, &constraints, offset](VectorType       &destination,
                                              const VectorType &source) {
          VectorType projected;
          operator_view.reinit_range_vector(projected, false);
          projected = source;
          for (const auto index : projected.locally_owned_elements())
            if (constraints.is_constrained(offset + index))
              projected(index) = 0.;
          operator_view.Tvmult_add(destination, projected);
        };
      return result;
    }

    template <typename VectorType, typename MatrixType>
    MaterializedOperator<VectorType, MatrixType>
    mixed_constrained_matrix_operator(
      const MaterializedOperator<VectorType, MatrixType> &source,
      const dealii::AffineConstraints<double>            &constraints,
      const dealii::types::global_dof_index               offset)
    {
      auto result = source;
      result.view =
        mixed_constrained_operator(source.view, constraints, offset);
      result.materialize = [source, &constraints, offset]() {
        auto matrix = source.matrix();
        zero_constrained_rows(*matrix, constraints, offset);
        return matrix;
      };
      result.materialize_into =
        [source, &constraints, offset](MatrixType &destination) {
          source.materialize_into_matrix(destination);
          zero_constrained_rows(destination, constraints, offset);
        };
      result.direct_matrix = false;
      return result;
    }

    template <typename VectorType, typename MatrixType>
    void
    add_matrix_product(const MatrixType &matrix,
                       const VectorType &source,
                       VectorType       &destination,
                       const double      factor = 1.)
    {
      VectorType product;
      product.reinit(destination);
      matrix.vmult(product, source);
      if (factor == 1.)
        destination += product;
      else
        {
          product *= factor;
          destination += product;
        }
    }

    template <typename VectorType>
    void
    zero_constrained(const dealii::AffineConstraints<double> &constraints,
                     VectorType                              &vector)
    {
      for (const auto index : vector.locally_owned_elements())
        if (constraints.is_constrained(index))
          vector(index) = 0.;
    }

    template <typename VectorType>
    void
    zero_mixed_constrained(const dealii::AffineConstraints<double> &constraints,
                           const dealii::types::global_dof_index block_offset,
                           VectorType                           &vector)
    {
      for (const auto index : vector.locally_owned_elements())
        if (constraints.is_constrained(block_offset + index))
          vector(index) = 0.;
    }
  } // namespace semidiscrete_detail
} // namespace ImmersX

#endif // immersx_semidiscrete_pde_models_h
