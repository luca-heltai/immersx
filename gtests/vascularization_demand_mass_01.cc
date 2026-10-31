// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "../source/vascularization/demand_mass.h"
using namespace ImmersX::Vascularization::internal;
TEST(VascularTissueDemand, ExactMassRetainsTinyContributionsAfterReassignment)
{
  const auto tiny  = demand_mass_units(1.e-300);
  const auto large = demand_mass_units(1.e300);
  EXPECT_DOUBLE_EQ(physical_demand_mass(tiny + large), 1.e300);
  EXPECT_DOUBLE_EQ(physical_demand_mass(tiny + large - large), 1.e-300);
  EXPECT_DOUBLE_EQ(physical_demand_mass(tiny + large - tiny), 1.e300);
  EXPECT_DOUBLE_EQ(physical_demand_mass(tiny - tiny), 0.);
  for (const double mass : {0.,
                            std::numeric_limits<double>::denorm_min(),
                            std::numeric_limits<double>::min(),
                            1.e-300,
                            .1,
                            1.,
                            1.e300,
                            std::numeric_limits<double>::max()})
    EXPECT_DOUBLE_EQ(physical_demand_mass(demand_mass_units(mass)), mass);
}
TEST(VascularTissueDemand, ExactMassRoundsTiesToEvenWithoutPartitionDependence)
{
  const double step = std::ldexp(1., -52), half = step / 2.;
  const auto   one     = demand_mass_units(1.);
  const auto   epsilon = demand_mass_units(step);
  const auto   tie     = demand_mass_units(half);
  const auto   subnormal =
    demand_mass_units(std::numeric_limits<double>::denorm_min());
  EXPECT_DOUBLE_EQ(physical_demand_mass(one + tie), 1.);
  EXPECT_DOUBLE_EQ(physical_demand_mass(one + tie + subnormal), 1. + step);
  EXPECT_DOUBLE_EQ(physical_demand_mass(one + epsilon + tie), 1. + 2. * step);
  EXPECT_DOUBLE_EQ(physical_demand_mass(one + epsilon + tie - subnormal),
                   1. + step);
  EXPECT_DOUBLE_EQ(physical_demand_mass((one + tie) + (epsilon - tie)),
                   1. + step);
  EXPECT_TRUE(std::isinf(physical_demand_mass(
    demand_mass_units(std::numeric_limits<double>::max()) * 2)));
}
TEST(VascularTissueDemand, ExactMassRejectsInvalidInputs)
{
  for (double value : {-1.,
                       std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(demand_mass_units(value), dealii::ExceptionBase);
  EXPECT_THROW(physical_demand_mass(boost::multiprecision::cpp_int(-1)),
               dealii::ExceptionBase);
}
