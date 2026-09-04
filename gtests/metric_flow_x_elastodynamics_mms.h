// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef IMMERSX_METRIC_FLOW_X_ELASTODYNAMICS_MMS_H
#define IMMERSX_METRIC_FLOW_X_ELASTODYNAMICS_MMS_H

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <cmath>
#include <functional>
#include <vector>

namespace ImmersX::OneVesselMMS
{
  /** Parameters for the analytical one-vessel verification solution.
   *
   * The area amplitude is deliberately an area amplitude, not a radius
   * amplitude.  This keeps the analytical solution on the same nonlinear
   * Area-to-radius map as the production wall representation.
   */
  struct Parameters
  {
    double length = 0.24137;
    // sqrt(3.0605e-4 / pi), the reference radius represented by the
    // single-vessel MetricFlowX metadata.  This is intentionally not r_d.
    double reference_r    = 0.009870093245078749;
    double outer_r        = 0.05;
    double area_amplitude = 1.e-8;
    double omega          = 2.0;
    double shear_modulus  = 1.0;
    double lame_lambda    = 0.0;
    double solid_density  = 1.0;
    double fluid_density  = 1060.0;
    double tube_a_d       = 3.0605e-4;
    double tube_E         = 4.e5;
    double tube_h_wall    = 1.2e-3;
    double tube_p0        = 0.0;
    double tube_p_d       = 0.0;
  };

  inline double
  wave_number(const Parameters &par)
  {
    return 2.0 * dealii::numbers::PI / par.length;
  }

  inline double
  reference_area(const Parameters &par)
  {
    return dealii::numbers::PI * par.reference_r * par.reference_r;
  }

  inline double
  modulation(const Parameters &par, const double time)
  {
    return 1.0 - std::cos(par.omega * time);
  }

  inline double
  modulation_t(const Parameters &par, const double time)
  {
    return par.omega * std::sin(par.omega * time);
  }

  inline double
  modulation_tt(const Parameters &par, const double time)
  {
    return par.omega * par.omega * std::cos(par.omega * time);
  }

  inline double
  exact_area(const Parameters &par, const double s, const double time)
  {
    return reference_area(par) + par.area_amplitude * modulation(par, time) *
                                   std::sin(wave_number(par) * s);
  }

  inline double
  spatial_area(const Parameters &par, const double s)
  {
    return reference_area(par) +
           par.area_amplitude * std::sin(wave_number(par) * s);
  }

  inline double
  spatial_area_s(const Parameters &par, const double s)
  {
    return par.area_amplitude * wave_number(par) *
           std::cos(wave_number(par) * s);
  }

  inline double
  spatial_area_ss(const Parameters &par, const double s)
  {
    const double k = wave_number(par);
    return -par.area_amplitude * k * k * std::sin(k * s);
  }

  inline double
  spatial_radius_increment(const Parameters &par, const double s)
  {
    return std::sqrt(spatial_area(par, s) / dealii::numbers::PI) -
           par.reference_r;
  }

  inline double
  spatial_radius_increment_s(const Parameters &par, const double s)
  {
    return spatial_area_s(par, s) /
           (2.0 * std::sqrt(dealii::numbers::PI * spatial_area(par, s)));
  }

  inline double
  spatial_radius_increment_ss(const Parameters &par, const double s)
  {
    const double area   = spatial_area(par, s);
    const double area_s = spatial_area_s(par, s);
    return spatial_area_ss(par, s) /
             (2.0 * std::sqrt(dealii::numbers::PI * area)) -
           area_s * area_s /
             (4.0 * std::sqrt(dealii::numbers::PI) * std::pow(area, 1.5));
  }

  inline double
  exact_area_t(const Parameters &par, const double s, const double time)
  {
    return par.area_amplitude * modulation_t(par, time) *
           std::sin(wave_number(par) * s);
  }

  inline double
  exact_area_tt(const Parameters &par, const double s, const double time)
  {
    return par.area_amplitude * modulation_tt(par, time) *
           std::sin(wave_number(par) * s);
  }

  inline double
  exact_area_s(const Parameters &par, const double s, const double time)
  {
    return par.area_amplitude * modulation(par, time) * wave_number(par) *
           std::cos(wave_number(par) * s);
  }

  inline double
  exact_area_ss(const Parameters &par, const double s, const double time)
  {
    const double k = wave_number(par);
    return -par.area_amplitude * modulation(par, time) * k * k *
           std::sin(k * s);
  }

  inline double
  exact_radius_increment(const Parameters &par,
                         const double      s,
                         const double      time)
  {
    return std::sqrt(exact_area(par, s, time) / dealii::numbers::PI) -
           par.reference_r;
  }

  inline double
  exact_radius_increment_t(const Parameters &par,
                           const double      s,
                           const double      time)
  {
    const double area = exact_area(par, s, time);
    return exact_area_t(par, s, time) /
           (2.0 * std::sqrt(dealii::numbers::PI * area));
  }

  inline double
  exact_radius_increment_tt(const Parameters &par,
                            const double      s,
                            const double      time)
  {
    const double area   = exact_area(par, s, time);
    const double area_t = exact_area_t(par, s, time);
    return exact_area_tt(par, s, time) /
             (2.0 * std::sqrt(dealii::numbers::PI * area)) -
           area_t * area_t /
             (4.0 * std::sqrt(dealii::numbers::PI) * std::pow(area, 1.5));
  }

  inline double
  exact_radius_increment_st(const Parameters &par,
                            const double      s,
                            const double      time)
  {
    const double area    = exact_area(par, s, time);
    const double area_s  = exact_area_s(par, s, time);
    const double area_t  = exact_area_t(par, s, time);
    const double area_st = par.area_amplitude * modulation_t(par, time) *
                           wave_number(par) * std::cos(wave_number(par) * s);
    return area_st / (2.0 * std::sqrt(dealii::numbers::PI * area)) -
           area_s * area_t /
             (4.0 * std::sqrt(dealii::numbers::PI) * std::pow(area, 1.5));
  }

  inline double
  exact_radius_increment_s(const Parameters &par,
                           const double      s,
                           const double      time)
  {
    return exact_area_s(par, s, time) /
           (2.0 * std::sqrt(dealii::numbers::PI * exact_area(par, s, time)));
  }

  inline double
  exact_radius_increment_ss(const Parameters &par,
                            const double      s,
                            const double      time)
  {
    const double area   = exact_area(par, s, time);
    const double area_s = exact_area_s(par, s, time);
    return exact_area_ss(par, s, time) /
             (2.0 * std::sqrt(dealii::numbers::PI * area)) -
           area_s * area_s /
             (4.0 * std::sqrt(dealii::numbers::PI) * std::pow(area, 1.5));
  }

  inline double
  radial_profile(const Parameters &par, const double radius)
  {
    if (radius <= par.reference_r)
      return radius / par.reference_r;

    return par.reference_r /
           (par.reference_r * par.reference_r - par.outer_r * par.outer_r) *
           (radius - par.outer_r * par.outer_r / radius);
  }

  inline double
  radial_profile_derivative(const Parameters &par, const double radius)
  {
    if (radius <= par.reference_r)
      return 1.0 / par.reference_r;

    return par.reference_r /
           (par.reference_r * par.reference_r - par.outer_r * par.outer_r) *
           (1.0 + par.outer_r * par.outer_r / (radius * radius));
  }

  inline double
  outer_radial_profile_derivative_at_wall(const Parameters &par)
  {
    return par.reference_r /
           (par.reference_r * par.reference_r - par.outer_r * par.outer_r) *
           (1.0 +
            par.outer_r * par.outer_r / (par.reference_r * par.reference_r));
  }

  inline dealii::Tensor<1, 3>
  radial_displacement(const Parameters       &par,
                      const dealii::Point<3> &point,
                      const double            time)
  {
    const double radius = std::sqrt(point[1] * point[1] + point[2] * point[2]);
    dealii::Tensor<1, 3> result;
    result = 0.;
    if (radius == 0.)
      return result;

    dealii::Tensor<1, 3> direction;
    direction[0] = 0.;
    direction[1] = point[1] / radius;
    direction[2] = point[2] / radius;
    result       = direction * (exact_radius_increment(par, point[0], time) *
                          radial_profile(par, radius));
    return result;
  }

  inline dealii::Tensor<1, 3>
  radial_velocity(const Parameters       &par,
                  const dealii::Point<3> &point,
                  const double            time)
  {
    const double radius = std::sqrt(point[1] * point[1] + point[2] * point[2]);
    dealii::Tensor<1, 3> result;
    result = 0.;
    if (radius == 0.)
      return result;

    dealii::Tensor<1, 3> direction;
    direction[0] = 0.;
    direction[1] = point[1] / radius;
    direction[2] = point[2] / radius;
    result       = direction * (exact_radius_increment_t(par, point[0], time) *
                          radial_profile(par, radius));
    return result;
  }

  inline dealii::Tensor<1, 3>
  spatial_radial_displacement(const Parameters       &par,
                              const dealii::Point<3> &point)
  {
    const double radius = std::sqrt(point[1] * point[1] + point[2] * point[2]);
    dealii::Tensor<1, 3> result;
    result = 0.;
    if (radius == 0.)
      return result;

    dealii::Tensor<1, 3> direction;
    direction[0] = 0.;
    direction[1] = point[1] / radius;
    direction[2] = point[2] / radius;
    return direction * (spatial_radius_increment(par, point[0]) *
                        radial_profile(par, radius));
  }

  /** Return the exact solid displacement or velocity as a deal.II Function.
   *
   * The gradient is supplied explicitly so that the verification driver can
   * use VectorTools::integrate_difference for both L2 and H1 errors.  The
   * limiting value at the axis is handled analytically rather than through a
   * division by the cylindrical radius.
   */
  inline dealii::FunctionFromFunctionObjects<3>
  solid_exact_function(const Parameters &par,
                       const double      time,
                       const bool        spatial,
                       const bool        velocity = false)
  {
    const auto amplitude = [par, time, spatial, velocity](const double s) {
      if (spatial)
        return spatial_radius_increment(par, s);
      return velocity ? exact_radius_increment_t(par, s, time) :
                        exact_radius_increment(par, s, time);
    };
    const auto amplitude_s = [par, time, spatial, velocity](const double s) {
      if (spatial)
        return spatial_radius_increment_s(par, s);
      return velocity ? exact_radius_increment_st(par, s, time) :
                        exact_radius_increment_s(par, s, time);
    };

    std::vector<std::function<double(const dealii::Point<3> &)>> values(3);
    std::vector<std::function<dealii::Tensor<1, 3>(const dealii::Point<3> &)>>
      gradients(3);
    values[0] = [](const dealii::Point<3> &) { return 0.; };
    values[1] = [par, amplitude](const dealii::Point<3> &point) {
      const double radius =
        std::sqrt(point[1] * point[1] + point[2] * point[2]);
      return radius == 0. ? 0. :
                            amplitude(point[0] + par.length / 2.) *
                              radial_profile(par, radius) * point[1] / radius;
    };
    values[2] = [par, amplitude](const dealii::Point<3> &point) {
      const double radius =
        std::sqrt(point[1] * point[1] + point[2] * point[2]);
      return radius == 0. ? 0. :
                            amplitude(point[0] + par.length / 2.) *
                              radial_profile(par, radius) * point[2] / radius;
    };

    gradients[0] = [](const dealii::Point<3> &) {
      return dealii::Tensor<1, 3>();
    };
    gradients[1] =
      [par, amplitude, amplitude_s](const dealii::Point<3> &point) {
        const double         y      = point[1];
        const double         z      = point[2];
        const double         radius = std::sqrt(y * y + z * z);
        dealii::Tensor<1, 3> gradient;
        gradient       = 0.;
        const double s = point[0] + par.length / 2.;
        const double d = amplitude(s);
        if (radius == 0.)
          {
            gradient[1] = d / par.reference_r;
            return gradient;
          }
        const double phi  = radial_profile(par, radius);
        const double dphi = radial_profile_derivative(par, radius);
        gradient[0]       = amplitude_s(s) * phi * y / radius;
        gradient[1] =
          d * (dphi * y * y / (radius * radius) +
               phi * (1. / radius - y * y / (radius * radius * radius)));
        gradient[2] = d * (dphi * y * z / (radius * radius) -
                           phi * y * z / (radius * radius * radius));
        return gradient;
      };
    gradients[2] =
      [par, amplitude, amplitude_s](const dealii::Point<3> &point) {
        const double         y      = point[1];
        const double         z      = point[2];
        const double         radius = std::sqrt(y * y + z * z);
        dealii::Tensor<1, 3> gradient;
        gradient       = 0.;
        const double s = point[0] + par.length / 2.;
        const double d = amplitude(s);
        if (radius == 0.)
          {
            gradient[2] = d / par.reference_r;
            return gradient;
          }
        const double phi  = radial_profile(par, radius);
        const double dphi = radial_profile_derivative(par, radius);
        gradient[0]       = amplitude_s(s) * phi * z / radius;
        gradient[1]       = d * (dphi * y * z / (radius * radius) -
                           phi * y * z / (radius * radius * radius));
        gradient[2] =
          d * (dphi * z * z / (radius * radius) +
               phi * (1. / radius - z * z / (radius * radius * radius)));
        return gradient;
      };
    return dealii::FunctionFromFunctionObjects<3>(values, gradients);
  }

  /** Continuous body force for the radial manufactured displacement.
   *
   * The two radial profiles are harmonic in the cross-section, so only the
   * axial variation and inertia remain.  This expression is deliberately
   * continuous-PDE data; it is not assembled from an FE residual.
   */
  inline dealii::Tensor<1, 3>
  solid_body_force(const Parameters       &par,
                   const dealii::Point<3> &point,
                   const double            time)
  {
    const double radius = std::sqrt(point[1] * point[1] + point[2] * point[2]);
    dealii::Tensor<1, 3> result;
    result = 0.;
    if (radius == 0.)
      {
        result[0] = -(par.shear_modulus + par.lame_lambda) *
                    exact_radius_increment_s(par, point[0], time) * 2. /
                    par.reference_r;
        return result;
      }

    dealii::Tensor<1, 3> direction;
    direction[0]      = 0.;
    direction[1]      = point[1] / radius;
    direction[2]      = point[2] / radius;
    const double phi  = radial_profile(par, radius);
    const double d_s  = exact_radius_increment_s(par, point[0], time);
    const double d_ss = exact_radius_increment_ss(par, point[0], time);
    const double d_tt = exact_radius_increment_tt(par, point[0], time);
    const double divergence_profile =
      radial_profile_derivative(par, radius) + phi / radius;

    result = direction *
             (par.solid_density * d_tt * phi - par.shear_modulus * d_ss * phi);
    result[0] -=
      (par.shear_modulus + par.lame_lambda) * d_s * divergence_profile;
    return result;
  }

  inline dealii::Tensor<1, 3>
  spatial_solid_body_force(const Parameters &par, const dealii::Point<3> &point)
  {
    const double radius = std::sqrt(point[1] * point[1] + point[2] * point[2]);
    dealii::Tensor<1, 3> result;
    result = 0.;
    if (radius == 0.)
      {
        result[0] = -(par.shear_modulus + par.lame_lambda) *
                    spatial_radius_increment_s(par, point[0]) * 2. /
                    par.reference_r;
        return result;
      }

    dealii::Tensor<1, 3> direction;
    direction[0]      = 0.;
    direction[1]      = point[1] / radius;
    direction[2]      = point[2] / radius;
    const double phi  = radial_profile(par, radius);
    const double d_s  = spatial_radius_increment_s(par, point[0]);
    const double d_ss = spatial_radius_increment_ss(par, point[0]);
    const double divergence_profile =
      radial_profile_derivative(par, radius) + phi / radius;
    result = direction * (-par.shear_modulus * d_ss * phi);
    result[0] -=
      (par.shear_modulus + par.lame_lambda) * d_s * divergence_profile;
    return result;
  }

  /** Coefficient of the radial traction jump for the production stress.
   *
   * The implementation uses sigma = 2 mu epsilon + lambda tr(epsilon) I.
   * Evaluating sigma_rr on the two sides of r=R0 gives
   *
   *   K = 2 (2 mu + lambda) R1^2 / (R0 (R1^2-R0^2)).
   */
  inline double
  traction_jump_coefficient(const Parameters &par)
  {
    return 2.0 * (2.0 * par.shear_modulus + par.lame_lambda) * par.outer_r *
           par.outer_r /
           (par.reference_r *
            (par.outer_r * par.outer_r - par.reference_r * par.reference_r));
  }

  /** The sign required by the production residual F_solid + B lambda = 0. */
  inline double
  exact_multiplier(const Parameters &par, const double s, const double time)
  {
    return -traction_jump_coefficient(par) *
           exact_radius_increment(par, s, time);
  }

  inline double
  spatial_multiplier(const Parameters &par, const double s)
  {
    return -traction_jump_coefficient(par) * spatial_radius_increment(par, s);
  }

  inline double
  exact_velocity(const Parameters &par, const double s, const double time)
  {
    const double k = wave_number(par);
    return -par.area_amplitude * modulation_t(par, time) / k *
           (1.0 - std::cos(k * s)) / exact_area(par, s, time);
  }

  inline double
  tube_pressure(const Parameters &par, const double area)
  {
    const double beta =
      4.0 * std::sqrt(dealii::numbers::PI) * par.tube_E * par.tube_h_wall / 3.0;
    return par.tube_p0 +
           beta / par.tube_a_d * (std::sqrt(area) - std::sqrt(par.tube_a_d)) +
           par.tube_p_d;
  }

  inline double
  exact_physical_pressure(const Parameters &par,
                          const double      s,
                          const double      time)
  {
    return tube_pressure(par, exact_area(par, s, time)) -
           exact_multiplier(par, s, time);
  }

  inline double
  spatial_physical_pressure(const Parameters &par, const double s)
  {
    return tube_pressure(par, spatial_area(par, s)) -
           spatial_multiplier(par, s);
  }

  inline double
  spatial_flow_momentum_source(const Parameters &par, const double s)
  {
    const double area = spatial_area(par, s);
    const double beta =
      4.0 * std::sqrt(dealii::numbers::PI) * par.tube_E * par.tube_h_wall / 3.0;
    const double pressure_s =
      beta / (2.0 * par.tube_a_d * std::sqrt(area)) * spatial_area_s(par, s) +
      traction_jump_coefficient(par) * spatial_radius_increment_s(par, s);
    return pressure_s / par.fluid_density;
  }

  inline double
  flow_momentum_source(const Parameters &par, const double s, const double time)
  {
    const double k          = wave_number(par);
    const double q_t        = modulation_t(par, time);
    const double q_tt       = modulation_tt(par, time);
    const double area       = exact_area(par, s, time);
    const double area_t     = exact_area_t(par, s, time);
    const double area_s     = exact_area_s(par, s, time);
    const double h          = 1.0 - std::cos(k * s);
    const double h_s        = k * std::sin(k * s);
    const double c          = par.area_amplitude * q_t / k;
    const double c_t        = par.area_amplitude * q_tt / k;
    const double velocity   = -c * h / area;
    const double velocity_t = -c_t * h / area + c * h * area_t / (area * area);
    const double velocity_s = -c * (h_s / area - h * area_s / (area * area));
    const double beta =
      4.0 * std::sqrt(dealii::numbers::PI) * par.tube_E * par.tube_h_wall / 3.0;
    const double tube_pressure_s =
      beta / (2.0 * par.tube_a_d * std::sqrt(area)) * area_s;
    const double external_pressure_s =
      traction_jump_coefficient(par) * exact_radius_increment_s(par, s, time);
    return velocity_t + velocity * velocity_s +
           (tube_pressure_s + external_pressure_s) / par.fluid_density;
  }

  inline double
  mass_conservation_defect(const Parameters &par,
                           const double      s,
                           const double      time)
  {
    const double h = 1.e-7 * par.length;
    const double flux_plus =
      exact_area(par, s + h, time) * exact_velocity(par, s + h, time);
    const double flux_minus =
      exact_area(par, s - h, time) * exact_velocity(par, s - h, time);
    return exact_area_t(par, s, time) + (flux_plus - flux_minus) / (2.0 * h);
  }

  inline double
  pressure_work_normalization(const double radius)
  {
    return 2.0 * dealii::numbers::PI * radius *
           (1.0 / (2.0 * std::sqrt(dealii::numbers::PI *
                                   (dealii::numbers::PI * radius * radius))));
  }
} // namespace ImmersX::OneVesselMMS

#endif // IMMERSX_METRIC_FLOW_X_ELASTODYNAMICS_MMS_H
