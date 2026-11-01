// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on deal.II.
//
// ---------------------------------------------------------------------
#ifndef immersx_ida_tolerance_ramp_h
#define immersx_ida_tolerance_ramp_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace ImmersX
{
  /** Geometric relaxation of IDA error tolerances during startup.
   * The hold duration is relative to the simulation's initial time. The end
   * time is absolute and must lie on the integrator's output grid.
   */
  struct IDAToleranceRamp
  {
    bool   enabled                 = false;
    double initial_relative_factor = 100.;
    double initial_absolute_factor = 100.;
    double hold_duration           = 0.;
    double end_time                = 0.;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Tolerance ramp");
      prm.add_parameter("Enabled", enabled);
      prm.add_parameter("Initial relative factor", initial_relative_factor);
      prm.add_parameter("Initial absolute factor", initial_absolute_factor);
      prm.add_parameter("Hold duration", hold_duration);
      prm.add_parameter("End time", end_time);
      prm.leave_subsection();
    }

    static double
    time_tolerance(const double initial, const double final)
    {
      return 32. * std::numeric_limits<double>::epsilon() *
             std::max(
               {std::abs(initial), std::abs(final), std::abs(final - initial)});
    }

    void
    validate(const double initial,
             const double final,
             const double output_period) const
    {
      if (!enabled)
        return;
      AssertThrow(std::isfinite(initial) && std::isfinite(final) &&
                    final > initial && std::isfinite(output_period) &&
                    output_period > 0.,
                  dealii::ExcMessage(
                    "Invalid IDA interval for tolerance ramp."));
      AssertThrow(std::isfinite(initial_relative_factor) &&
                    initial_relative_factor >= 1. &&
                    std::isfinite(initial_absolute_factor) &&
                    initial_absolute_factor >= 1. &&
                    std::isfinite(hold_duration) && hold_duration >= 0. &&
                    std::isfinite(initial + hold_duration) &&
                    std::isfinite(end_time) &&
                    initial + hold_duration < end_time && end_time <= final,
                  dealii::ExcMessage(
                    "Invalid IDA tolerance ramp factors or times."));
      const double periods = (end_time - initial) / output_period;
      AssertThrow(
        std::isfinite(periods) && std::round(periods) >= 1. &&
          std::abs(end_time -
                   (initial + std::round(periods) * output_period)) <=
            time_tolerance(initial, end_time),
        dealii::ExcMessage(
          "Tolerance ramp end time must lie on the IDA output grid."));
    }

    /** Relative and absolute multipliers, respectively. */
    std::pair<double, double>
    scales(const double initial, const double time) const
    {
      if (!enabled || time >= end_time - time_tolerance(initial, end_time))
        return {1., 1.};
      const double hold_until = initial + hold_duration;
      if (time <= hold_until)
        return {initial_relative_factor, initial_absolute_factor};
      const double fraction =
        std::clamp((end_time - time) / (end_time - hold_until), 0., 1.);
      return {std::exp(fraction * std::log(initial_relative_factor)),
              std::exp(fraction * std::log(initial_absolute_factor))};
    }
  };
} // namespace ImmersX
#endif
