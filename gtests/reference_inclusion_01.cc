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

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "inclusions.h"
#include "reference_cross_section.h" // Add include for ReferenceCrossSection

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