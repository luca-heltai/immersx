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

#include "reference_inclusion.h" // Add include for ReferenceInclusion

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "inclusions.h"

using namespace dealii;

TEST(ReferenceInclusion2, CheckPoints) // NOLINT
{
  // cx, cy, r
  Inclusions<2> ref;
  ref.set_n_q_points(4);
  ref.set_n_coefficients(1);
  ref.inclusions.push_back({{0, 0, 1.0}});
  ref.initialize();
  const auto &p = ref.get_current_support_points(0);

  ASSERT_NEAR(p[0].distance(Point<2>(1, 0)), 0, 1e-10);
  ASSERT_NEAR(p[1].distance(Point<2>(0, 1)), 0, 1e-10);
  ASSERT_NEAR(p[2].distance(Point<2>(-1, 0)), 0, 1e-10);
  ASSERT_NEAR(p[3].distance(Point<2>(0, -1)), 0, 1e-10);
}

TEST(Inclusion3, CheckPoints) // NOLINT
{
  // cx, cy, cz, dx, dy, dz, r
  std::vector<double> inc({{0, 0, 0, 0, 0, 1.0, 1.0, 0}});
  Inclusions<3>       ref;
  ref.set_n_q_points(4);
  ref.set_n_coefficients(1);
  ref.inclusions.push_back(inc);
  ref.initialize();
  const auto &p = ref.get_current_support_points(0);

  ASSERT_NEAR(p[0].distance(Point<3>(1, 0, 0)), 0, 1e-10);
  ASSERT_NEAR(p[1].distance(Point<3>(0, 1, 0)), 0, 1e-10);
  ASSERT_NEAR(p[2].distance(Point<3>(-1, 0, 0)), 0, 1e-10);
  ASSERT_NEAR(p[3].distance(Point<3>(0, -1, 0)), 0, 1e-10);
}

TEST(Inclusion3, CheckPointsRotated) // NOLINT
{
  // cx, cy, cz, dx, dy, dz, r
  std::vector<double> inc({{0, 0, 0, 1.0, 0, 0, 1.0, 0}});
  Inclusions<3>       ref;
  ref.set_n_q_points(4);
  ref.set_n_coefficients(1);
  ref.inclusions.push_back(inc);
  ref.initialize();
  const auto &p = ref.get_current_support_points(0);

  ASSERT_NEAR(p[0].norm(), 1, 1e-10);
  ASSERT_NEAR(p[1].norm(), 1, 1e-10);
  ASSERT_NEAR(p[2].norm(), 1, 1e-10);
  ASSERT_NEAR(p[3].norm(), 1, 1e-10);

  ASSERT_NEAR(p[0].distance(Point<3>(0, 0, -1)), 0, 1e-10);
  ASSERT_NEAR(p[1].distance(Point<3>(0, 1, 0)), 0, 1e-10);
  ASSERT_NEAR(p[2].distance(Point<3>(0, 0, 1)), 0, 1e-10);
  ASSERT_NEAR(p[3].distance(Point<3>(0, -1, 0)), 0, 1e-10);
}


TEST(Inclusion3, CheckNegativeZDirection) // NOLINT
{
  // cx, cy, cz, dx, dy, dz, r
  std::vector<double> inc({{0, 0, 0, 0, 0, -1, 1.0, 0}});
  Inclusions<3>       ref;
  ref.set_n_q_points(4);
  ref.set_n_coefficients(1);
  ref.inclusions.push_back(inc);
  ref.initialize();
  const auto r = ref.get_rotation(0);

  const Tensor<1, 3> zdir({0, 0, 1});
  const Tensor<1, 3> mzdir({0, 0, -1});

  const auto a = r * zdir;

  ASSERT_NEAR((a - mzdir).norm(), 0, 1e-10);
}

TEST(Inclusion3, CheckAlmostNegativeZDirection) // NOLINT
{
  // cx, cy, cz, dx, dy, dz, r
  std::vector<double> inc({{0, 0, 0, 0, 0.5, -1, 1.0, 0}});
  Inclusions<3>       ref;
  ref.set_n_q_points(4);
  ref.set_n_coefficients(1);
  ref.inclusions.push_back(inc);
  ref.initialize();
  const auto r = ref.get_rotation(0);

  const Tensor<1, 3> zdir({0, 0, 1});
  Tensor<1, 3>       mzdir({0, 0.5, -1});
  mzdir /= mzdir.norm();

  const auto a = r * zdir;

  ASSERT_NEAR((a - mzdir).norm(), 0, 1e-10);
}

TEST(ReferenceInclusion, CheckBasisOrthogonality) // NOLINT
{
  const int          dim          = 2;
  const int          spacedim     = 2;
  const int          n_components = 1;
  const unsigned int degree       = 3;

  // Set up parameters for the reference inclusion
  ReferenceInclusionParameters<dim, spacedim, n_components> par;
  par.inclusion_degree = degree;
  par.inclusion_type   = "hyper_ball"; // Or "hyper_cube"
  par.refinement_level = 2;            // Use a reasonable refinement level
  // Leaving par.selected_coefficients empty means all basis functions are
  // selected

  // Create the reference inclusion object
  ReferenceInclusion<dim, spacedim, n_components> ref_inclusion(par);

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