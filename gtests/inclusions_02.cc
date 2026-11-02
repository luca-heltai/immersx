// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>
#include <immersx/coupling/inclusions.h>

#include <vector>

using namespace ImmersX;

using namespace dealii;

TEST(InclusionsBasis2, CheckIntegrals) // NOLINT
{
  // cx, cy, r
  Inclusions<2> ref;
  ref.set_n_q_points(32);
  ref.set_n_coefficients(3);
  ref.inclusions.push_back({{0, 0, 1.0}});
  ref.initialize();
  double integral[4] = {0, 0, 0, 0};
  for (unsigned int q = 0; q < ref.n_particles(); ++q)
    {
      const auto &fe_values = ref.get_fe_values(q);
      const auto &ds        = ref.get_JxW(q);
      integral[0] += ds;
      integral[1] += fe_values[0] * ds;
      integral[2] += fe_values[1] * ds;
      integral[3] += fe_values[2] * ds;
    }

  ASSERT_NEAR(integral[0], 2 * numbers::PI, 1e-10);
  ASSERT_NEAR(integral[1], 2 * numbers::PI, 1e-10);
  ASSERT_NEAR(integral[2], 0, 1e-10);
  ASSERT_NEAR(integral[3], 0, 1e-10);
}

TEST(InclusionsBasis2, CheckScaling) // NOLINT
{
  // cx, cy, r
  Inclusions<2> ref;
  double        radius = 9;
  ref.set_n_q_points(32);
  ref.set_n_coefficients(13);
  ref.inclusions.push_back({{0, 0, radius}});
  ref.initialize();
  std::vector<double> integral(ref.get_n_coefficients() + 1, 0.0);
  for (unsigned int q = 0; q < ref.n_particles(); ++q)
    {
      const auto &fe_values = ref.get_fe_values(q);
      const auto &ds        = ref.get_JxW(q);
      integral[0] += ds;
      for (unsigned int i = 0; i < ref.get_n_coefficients(); ++i)
        integral[i + 1] += fe_values[i] * fe_values[i] * ds;
    }
  // Take the square root of the integral of the basis functions
  // and check that they are equal to sqrt(|D|) where |D| is the length of the
  // inclusion
  auto D = 2 * numbers::PI * radius;
  EXPECT_NEAR(integral[0], D, 1e-10); // integral of 1
  EXPECT_NEAR(integral[1], D, 1e-10); // integral of \phi_0
  auto expected = D / 2;
  for (unsigned int i = 2; i < ref.get_n_coefficients() + 1; ++i)
    {
      EXPECT_NEAR(integral[i], expected, 1e-10);
    }
}
