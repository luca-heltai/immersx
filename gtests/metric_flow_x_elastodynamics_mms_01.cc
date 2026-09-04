// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/convergence_table.h>

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "metric_flow_x_elastodynamics_mms.h"

namespace
{
  using ImmersX::OneVesselMMS::Parameters;

  TEST(OneVesselMMS, StaticRadialProfileAndTractionJump)
  {
    Parameters   par;
    const double d = 0.001;

    EXPECT_DOUBLE_EQ(ImmersX::OneVesselMMS::radial_profile(par, 0.), 0.);
    EXPECT_NEAR(ImmersX::OneVesselMMS::radial_profile(par, par.reference_r),
                1.,
                1.e-14);
    EXPECT_NEAR(ImmersX::OneVesselMMS::radial_profile(par, par.outer_r),
                0.,
                1.e-14);

    const double inner_sigma_rr =
      2. * (par.shear_modulus + par.lame_lambda) * d / par.reference_r;
    const double outer_phi_prime =
      ImmersX::OneVesselMMS::outer_radial_profile_derivative_at_wall(par);
    const double outer_divergence = outer_phi_prime + 1. / par.reference_r;
    const double outer_sigma_rr = 2. * par.shear_modulus * d * outer_phi_prime +
                                  par.lame_lambda * d * outer_divergence;
    EXPECT_NEAR(inner_sigma_rr - outer_sigma_rr,
                ImmersX::OneVesselMMS::traction_jump_coefficient(par) * d,
                1.e-12);
  }

  TEST(OneVesselMMS, AreaToRadiusMapAndInitialState)
  {
    Parameters par;
    for (const double s : {0., 0.25 * par.length, 0.5 * par.length})
      {
        EXPECT_NEAR(ImmersX::OneVesselMMS::exact_area(par, s, 0.),
                    ImmersX::OneVesselMMS::reference_area(par),
                    1.e-15);
        EXPECT_NEAR(ImmersX::OneVesselMMS::exact_radius_increment(par, s, 0.),
                    0.,
                    1.e-15);
        EXPECT_NEAR(ImmersX::OneVesselMMS::exact_multiplier(par, s, 0.),
                    0.,
                    1.e-15);
      }

    const double area       = 1.1 * ImmersX::OneVesselMMS::reference_area(par);
    const double derivative = 1. / (2. * std::sqrt(dealii::numbers::PI * area));
    const double h          = 1.e-6 * area;
    const double centered   = (std::sqrt((area + h) / dealii::numbers::PI) -
                             std::sqrt((area - h) / dealii::numbers::PI)) /
                            (2. * h);
    EXPECT_NEAR(centered, derivative, 1.e-8);
  }

  TEST(OneVesselMMS, ExactFlowIsMassConserving)
  {
    Parameters   par;
    const double time = 0.37;
    for (unsigned int i = 1; i < 10; ++i)
      {
        const double s = par.length * i / 10.;
        EXPECT_NEAR(ImmersX::OneVesselMMS::mass_conservation_defect(par,
                                                                    s,
                                                                    time),
                    0.,
                    2.e-10);
      }
    EXPECT_NEAR(ImmersX::OneVesselMMS::exact_velocity(par, 0., time),
                0.,
                1.e-14);
    EXPECT_NEAR(ImmersX::OneVesselMMS::exact_velocity(par, par.length, time),
                0.,
                1.e-14);
  }

  TEST(OneVesselMMS, PressureWorkHasUnitNormalization)
  {
    for (const double radius : {0.01, 0.012, 0.05})
      EXPECT_NEAR(ImmersX::OneVesselMMS::pressure_work_normalization(radius),
                  1.,
                  2.e-14);
  }

  TEST(OneVesselMMS, ContinuousSolidSourceIsZeroForStaticConstantRadius)
  {
    Parameters             par;
    const dealii::Point<3> point(0.5 * par.length, 0.02, 0.01);
    const auto force = ImmersX::OneVesselMMS::solid_body_force(par, point, 0.);
    EXPECT_NEAR(force.norm(), 0., 1.e-14);
  }

  TEST(OneVesselMMS, SpatialSourcesUseNativePressureSign)
  {
    Parameters   par;
    const double s = 0.23 * par.length;
    const double h = 1.e-6 * par.length;
    const double pressure_s =
      (ImmersX::OneVesselMMS::spatial_physical_pressure(par, s - 2. * h) -
       8. * ImmersX::OneVesselMMS::spatial_physical_pressure(par, s - h) +
       8. * ImmersX::OneVesselMMS::spatial_physical_pressure(par, s + h) -
       ImmersX::OneVesselMMS::spatial_physical_pressure(par, s + 2. * h)) /
      (12. * h);
    EXPECT_NEAR(pressure_s,
                par.fluid_density *
                  ImmersX::OneVesselMMS::spatial_flow_momentum_source(par, s),
                5.e-5);

    const dealii::Point<3> point(s, 0.02, 0.01);
    EXPECT_GT(
      ImmersX::OneVesselMMS::spatial_solid_body_force(par, point).norm(), 0.);
    const dealii::Point<3> axis_point(s, 0., 0.);
    const auto             axis_force =
      ImmersX::OneVesselMMS::spatial_solid_body_force(par, axis_point);
    EXPECT_NEAR(axis_force[0],
                -(par.shear_modulus + par.lame_lambda) *
                  ImmersX::OneVesselMMS::spatial_radius_increment_s(par, s) *
                  2. / par.reference_r,
                1.e-14);
  }

  TEST(OneVesselMMS, TransientSourcesAndWallTraceAreFinite)
  {
    Parameters             par;
    const double           time = 0.37;
    const double           s    = 0.31 * par.length;
    const dealii::Point<3> wall_point(s, par.reference_r, 0.);
    const auto             displacement =
      ImmersX::OneVesselMMS::radial_displacement(par, wall_point, time);
    const auto velocity =
      ImmersX::OneVesselMMS::radial_velocity(par, wall_point, time);
    const auto force =
      ImmersX::OneVesselMMS::solid_body_force(par, wall_point, time);

    EXPECT_TRUE(std::isfinite(displacement.norm()));
    EXPECT_TRUE(std::isfinite(velocity.norm()));
    EXPECT_TRUE(std::isfinite(force.norm()));
    EXPECT_TRUE(
      std::isfinite(ImmersX::OneVesselMMS::flow_momentum_source(par, s, time)));
    EXPECT_NEAR(displacement[1],
                ImmersX::OneVesselMMS::exact_radius_increment(par, s, time),
                1.e-14);
  }

  TEST(OneVesselMMS, FourLevelAreaRadiusConvergenceTable)
  {
    Parameters   par;
    const double s    = 0.37 * par.length;
    const double time = 0.41;
    const double exact =
      ImmersX::OneVesselMMS::exact_radius_increment_s(par, s, time);

    dealii::ConvergenceTable table;
    std::vector<double>      errors;
    for (unsigned int level = 0; level < 4; ++level)
      {
        const double h = par.length / (4. * std::pow(2., level));
        const double numerical =
          (ImmersX::OneVesselMMS::exact_radius_increment(par, s + h, time) -
           ImmersX::OneVesselMMS::exact_radius_increment(par, s - h, time)) /
          (2. * h);
        const double error = std::abs(numerical - exact);
        errors.push_back(error);
        table.add_value("level", level);
        table.add_value("h", h);
        table.add_value("1/h", 1. / h);
        table.add_value("L2", error);
      }
    table.evaluate_convergence_rates(
      "L2", "1/h", dealii::ConvergenceTable::reduction_rate_log2);
    table.write_text(std::cout);

    for (unsigned int level = 1; level < errors.size(); ++level)
      EXPECT_GT(std::log(errors[level - 1] / errors[level]) / std::log(2.),
                1.8);
  }
} // namespace
