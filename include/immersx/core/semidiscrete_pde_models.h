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
