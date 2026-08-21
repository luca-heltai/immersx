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
