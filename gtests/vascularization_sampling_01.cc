// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>

#include <limits>

#include "../source/vascularization/growth_sampling.h"
using namespace dealii;
using namespace ImmersX::Vascularization::internal;

TEST(VascularGeneration, CellVolumeSamplingDistinguishesEqualDiameterCells)
{
  Triangulation<3> a, b;
  GridGenerator::hyper_rectangle(a, Point<3>(), Point<3>(1., 4., 8.));
  GridGenerator::hyper_rectangle(b, Point<3>(), Point<3>(4., 4., 7.));
  const auto ca = a.begin_active(), cb = b.begin_active();
  EXPECT_DOUBLE_EQ(ca->diameter(), 9.);
  EXPECT_DOUBLE_EQ(cb->diameter(), 9.);
  const double va = ca->measure(), vb = cb->measure();
  EXPECT_DOUBLE_EQ(va, 32.);
  EXPECT_DOUBLE_EQ(vb, 112.);
  EXPECT_NEAR(mesh_sampling_factor(vb, va, 3.), 2. / 7., 1.e-15);
  // Equal physical mass per cell: changing only the aspect ratio must not
  // replace the volume rule with a diameter-based approximation.
  EXPECT_NEAR(va * mesh_sampling_factor(va, va, 3.), 32., 1.e-13);
  EXPECT_NEAR(vb * mesh_sampling_factor(vb, va, 3.), 32., 1.e-13);
  EXPECT_DOUBLE_EQ(mesh_sampling_factor(va, va, 0.), 1.);
  EXPECT_DOUBLE_EQ(mesh_sampling_factor(vb, va, 0.), 1.);
  EXPECT_NEAR(mesh_sampling_factor(va, vb, -3.), 2. / 7., 1.e-15);
  EXPECT_DOUBLE_EQ(mesh_sampling_factor(vb, vb, -3.), 1.);
}

TEST(VascularGeneration,
     CellVolumeSamplingKeepsItsRelativeDensityUnderNormalization)
{
  EXPECT_NEAR(mesh_sampling_factor(112., 32., 3.) /
                mesh_sampling_factor(32., 32., 3.),
              2. / 7.,
              1.e-15);
  EXPECT_NEAR(mesh_sampling_factor(112., 8., 3.) /
                mesh_sampling_factor(32., 8., 3.),
              2. / 7.,
              1.e-15);
  // The physical volume ratio overflows, while the normalized probability is
  // representable. Do not form that ratio before taking a negative power.
  EXPECT_NEAR(mesh_sampling_factor(1.e-200, 1.e200, -1.5) / 1.e-200,
              1.,
              1.e-13);
  EXPECT_THROW(mesh_sampling_factor(0., 1., 3.), ExceptionBase);
  EXPECT_THROW(mesh_sampling_factor(1., 2., 3.), ExceptionBase);
  EXPECT_THROW(mesh_sampling_factor(2., 1., -3.), ExceptionBase);
  EXPECT_THROW(mesh_sampling_factor(1.,
                                    1.,
                                    std::numeric_limits<double>::infinity()),
               ExceptionBase);
}
