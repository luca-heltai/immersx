// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

#include "reference_cross_section.h" // Add include for ReferenceCrossSection

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "inclusions.h"

using namespace dealii;

TEST(ReferenceCrossSection, CheckBasisOrthogonality) // NOLINT
{
  const int          dim          = 2;
  const int          spacedim     = 2;
  const int          n_components = 1;
  const unsigned int degree       = 3;

  // Set up parameters for the reference inclusion
  ReferenceCrossSectionParameters<dim, spacedim, n_components> par;
  par.inclusion_degree = degree;
  par.inclusion_type   = "hyper_ball"; // Or "hyper_cube"
  par.refinement_level = 2;            // Use a reasonable refinement level
  // Leaving par.selected_coefficients empty means all basis functions are
  // selected

  // Create the reference inclusion object
  ReferenceCrossSection<dim, spacedim, n_components> ref_inclusion(par);

  // Get the computed basis functions and the mass matrix
  const auto &basis       = ref_inclusion.get_basis_functions();
  const auto &mass_matrix = ref_inclusion.get_mass_matrix();

  const unsigned int n_basis_functions = basis.size();

  // Verify the number of basis functions matches the polynomial space size
  PolynomialsP<spacedim> polynomials(degree);
  ASSERT_EQ(n_basis_functions, polynomials.n() * n_components);

  // Check for M-orthonormality: basis[i]^T * M * basis[j] == delta_ij
  Vector<double> tmp(mass_matrix.m()); // Temporary vector for M * basis[j]

  for (unsigned int i = 0; i < n_basis_functions; ++i)
    {
      ASSERT_GT(basis[i].l2_norm(), 1e-15)
        << "Basis function " << i << " is zero.";
      for (unsigned int j = 0; j < n_basis_functions; ++j)
        {
          mass_matrix.vmult(tmp, basis[j]); // tmp = M * basis_j
          const double dot_product =
            basis[i] * tmp; // dot = basis_i^T * M * basis_j

          if (i == j)
            {
              // Diagonal elements should be close to 1 for orthonormal basis
              ASSERT_NEAR(dot_product, 1.0, 1e-10)
                << "Basis functions " << i << " and " << j
                << " are not M-orthonormal (diagonal check).";
            }
          else
            {
              // Off-diagonal elements should be close to 0 for orthogonal basis
              ASSERT_NEAR(dot_product, 0.0, 1e-10)
                << "Basis functions " << i << " and " << j
                << " are not M-orthogonal (off-diagonal check).";
            }
        }
    }
}