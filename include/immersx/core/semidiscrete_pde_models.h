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

namespace ImmersX
{
  /** Small matrix and constraint helpers shared by physics contributors. */
  namespace semidiscrete_detail
  {
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
    dealii::PackagedOperation<VectorType>
    constrained_residual(dealii::PackagedOperation<VectorType>    operation,
                         const VectorType                        &state,
                         const dealii::AffineConstraints<double> &constraints)
    {
      dealii::PackagedOperation<VectorType> result;
      result.reinit_vector = operation.reinit_vector;
      result.apply =
        [operation, &state, &constraints](VectorType &destination) {
          operation.apply(destination);
          for (const auto index : destination.locally_owned_elements())
            if (constraints.is_constrained(index))
              destination(index) =
                state(index) - constraints.get_inhomogeneity(index);
        };
      result.apply_add =
        [operation, &state, &constraints](VectorType &destination) {
          VectorType contribution;
          operation.reinit_vector(contribution, false);
          operation.apply(contribution);
          for (const auto index : contribution.locally_owned_elements())
            if (constraints.is_constrained(index))
              contribution(index) =
                state(index) - constraints.get_inhomogeneity(index);
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
      const auto retained =
        std::make_shared<const dealii::LinearOperator<VectorType, VectorType>>(
          operator_view);
      auto result  = operator_view;
      result.vmult = [retained, &constraints](VectorType       &destination,
                                              const VectorType &source) {
        retained->vmult(destination, source);
        for (const auto index : destination.locally_owned_elements())
          if (constraints.is_constrained(index))
            destination(index) = 0.;
      };
      result.vmult_add = [retained, &constraints](VectorType       &destination,
                                                  const VectorType &source) {
        VectorType contribution;
        retained->reinit_range_vector(contribution, false);
        retained->vmult(contribution, source);
        for (const auto index : contribution.locally_owned_elements())
          if (constraints.is_constrained(index))
            contribution(index) = 0.;
        destination += contribution;
      };
      result.Tvmult = [retained, &constraints](VectorType       &destination,
                                               const VectorType &source) {
        VectorType projected;
        retained->reinit_range_vector(projected, false);
        projected = source;
        for (const auto index : projected.locally_owned_elements())
          if (constraints.is_constrained(index))
            projected(index) = 0.;
        retained->Tvmult(destination, projected);
      };
      result.Tvmult_add = [retained, &constraints](VectorType &destination,
                                                   const VectorType &source) {
        VectorType projected;
        retained->reinit_range_vector(projected, false);
        projected = source;
        for (const auto index : projected.locally_owned_elements())
          if (constraints.is_constrained(index))
            projected(index) = 0.;
        retained->Tvmult_add(destination, projected);
      };
      return result;
    }

    template <typename VectorType>
    dealii::LinearOperator<VectorType, VectorType>
    constrained_operator_with_identity(
      const dealii::LinearOperator<VectorType, VectorType> &operator_view,
      const dealii::AffineConstraints<double>              &constraints)
    {
      const auto retained =
        std::make_shared<const dealii::LinearOperator<VectorType, VectorType>>(
          operator_view);
      auto result  = operator_view;
      result.vmult = [retained, &constraints](VectorType       &destination,
                                              const VectorType &source) {
        retained->vmult(destination, source);
        for (const auto index : destination.locally_owned_elements())
          if (constraints.is_constrained(index))
            destination(index) = source(index);
      };
      result.vmult_add = [retained, &constraints](VectorType       &destination,
                                                  const VectorType &source) {
        VectorType contribution;
        retained->reinit_range_vector(contribution, false);
        retained->vmult(contribution, source);
        for (const auto index : contribution.locally_owned_elements())
          if (constraints.is_constrained(index))
            contribution(index) = source(index);
        destination += contribution;
      };
      result.Tvmult     = result.vmult;
      result.Tvmult_add = result.vmult_add;
      return result;
    }

    template <typename VectorType>
    dealii::LinearOperator<VectorType, VectorType>
    constrained_identity_operator(
      const dealii::LinearOperator<VectorType, VectorType> &operator_view,
      const dealii::AffineConstraints<double>              &constraints)
    {
      auto result  = operator_view;
      result.vmult = [&constraints](VectorType       &destination,
                                    const VectorType &source) {
        destination = 0.;
        for (const auto index : destination.locally_owned_elements())
          if (constraints.is_constrained(index))
            destination(index) = source(index);
      };
      result.vmult_add = [&constraints](VectorType       &destination,
                                        const VectorType &source) {
        for (const auto index : destination.locally_owned_elements())
          if (constraints.is_constrained(index))
            destination(index) += source(index);
      };
      result.Tvmult     = result.vmult;
      result.Tvmult_add = result.vmult_add;
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
        for (const auto row : matrix->locally_owned_range_indices())
          if (constraints.is_constrained(row) && matrix->row_length(row) != 0)
            for (auto entry = matrix->begin(row), end = matrix->end(row);
                 entry != end;
                 ++entry)
              matrix->set(row, entry->column(), 0.);
        matrix->compress(dealii::VectorOperation::insert);
        return matrix;
      };
      return result;
    }

    template <typename VectorType, typename MatrixType>
    MaterializedOperator<VectorType, MatrixType>
    constrained_matrix_operator_with_identity(
      const MaterializedOperator<VectorType, MatrixType> &source,
      const dealii::AffineConstraints<double>            &constraints)
    {
      auto result = source;
      result.view =
        constrained_operator_with_identity(source.view, constraints);
      const auto impose_identity = [&constraints](MatrixType &matrix) {
        for (const auto &line : constraints.get_lines())
          if (matrix.in_local_range(line.index))
            {
              matrix.clear_row(line.index, 0.);
              matrix.set(line.index, line.index, 1.);
            }
        matrix.compress(dealii::VectorOperation::insert);
      };
      result.materialize = [source, impose_identity]() {
        auto matrix = source.matrix();
        impose_identity(*matrix);
        return matrix;
      };
      result.materialize_into = [source, impose_identity](MatrixType &matrix) {
        source.materialize_into_matrix(matrix);
        impose_identity(matrix);
      };
      return result;
    }

    template <typename VectorType, typename MatrixType>
    MaterializedOperator<VectorType, MatrixType>
    constrained_matrix_identity_operator(
      const MaterializedOperator<VectorType, MatrixType> &source,
      const dealii::AffineConstraints<double>            &constraints)
    {
      auto result = source;
      result.view = constrained_identity_operator(source.view, constraints);
      const auto impose_identity = [&constraints](MatrixType &matrix) {
        for (const auto row : matrix.locally_owned_range_indices())
          {
            matrix.clear_row(row, 0.);
            if (constraints.is_constrained(row))
              matrix.set(row, row, 1.);
          }
        matrix.compress(dealii::VectorOperation::insert);
      };
      result.materialize = [source, impose_identity]() {
        auto matrix = source.matrix();
        impose_identity(*matrix);
        return matrix;
      };
      result.materialize_into = [source, impose_identity](MatrixType &matrix) {
        source.materialize_into_matrix(matrix);
        impose_identity(matrix);
      };
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
      const auto retained =
        std::make_shared<const dealii::LinearOperator<VectorType, VectorType>>(
          operator_view);
      auto result = operator_view;
      result.vmult =
        [retained, &constraints, offset](VectorType       &destination,
                                         const VectorType &source) {
          retained->vmult(destination, source);
          for (const auto index : destination.locally_owned_elements())
            if (constraints.is_constrained(offset + index))
              destination(index) = 0.;
        };
      result.vmult_add =
        [retained, &constraints, offset](VectorType       &destination,
                                         const VectorType &source) {
          VectorType contribution;
          retained->reinit_range_vector(contribution, false);
          retained->vmult(contribution, source);
          for (const auto index : contribution.locally_owned_elements())
            if (constraints.is_constrained(offset + index))
              contribution(index) = 0.;
          destination += contribution;
        };
      result.Tvmult =
        [retained, &constraints, offset](VectorType       &destination,
                                         const VectorType &source) {
          VectorType projected;
          retained->reinit_range_vector(projected, false);
          projected = source;
          for (const auto index : projected.locally_owned_elements())
            if (constraints.is_constrained(offset + index))
              projected(index) = 0.;
          retained->Tvmult(destination, projected);
        };
      result.Tvmult_add =
        [retained, &constraints, offset](VectorType       &destination,
                                         const VectorType &source) {
          VectorType projected;
          retained->reinit_range_vector(projected, false);
          projected = source;
          for (const auto index : projected.locally_owned_elements())
            if (constraints.is_constrained(offset + index))
              projected(index) = 0.;
          retained->Tvmult_add(destination, projected);
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
        for (const auto row : matrix->locally_owned_range_indices())
          if (constraints.is_constrained(offset + row) &&
              matrix->row_length(row) != 0)
            for (auto entry = matrix->begin(row), end = matrix->end(row);
                 entry != end;
                 ++entry)
              matrix->set(row, entry->column(), 0.);
        matrix->compress(dealii::VectorOperation::insert);
        return matrix;
      };
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
