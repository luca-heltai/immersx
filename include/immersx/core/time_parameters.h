// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_time_parameters_h
#define immersx_time_parameters_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/patterns.h>

#ifdef DEAL_II_WITH_SUNDIALS
#  include <deal.II/sundials/ida.h>
#endif

#include <string>

namespace ImmersX
{
  /** Canonical ownership of application and time-integrator parameters. */
  class TimeParameters : public dealii::ParameterAcceptor
  {
  public:
    explicit TimeParameters(const std::string &subsection = "/Time parameters/")
      : dealii::ParameterAcceptor(subsection)
    {
      parse_parameters_call_back.connect([this]() {
        AssertThrow(final_time >= initial_time,
                    dealii::ExcMessage(
                      "Final time must not precede initial time."));
        if (final_time > initial_time || number_of_steps > 0)
          AssertThrow(time_step > 0.,
                      dealii::ExcMessage(
                        "Time step must be positive for a transient run."));
      });
    }

    double         initial_time     = 0.;
    double         final_time       = 0.1;
    mutable double time_step        = 1.e-2;
    unsigned int   number_of_steps  = 0;
    unsigned int   output_frequency = 1;
    std::string    time_step_policy = "number_of_steps";
    bool           refine_time_step = false;
    double         newmark_beta     = 0.25;
    double         newmark_gamma    = 0.5;

#ifdef DEAL_II_WITH_SUNDIALS
    // IDA-specific execution controls remain canonical here.  They are not
    // exposed through a second ParameterAcceptor or an IDA-specific section.
    double       initial_step_size                 = 1.e-5;
    double       minimum_step_size                 = 0.;
    unsigned int maximum_order                     = 1;
    unsigned int maximum_non_linear_iterations     = 50;
    double       absolute_tolerance                = 1.e-4;
    double       relative_tolerance                = 1.e-4;
    bool         ignore_algebraic_terms_for_errors = true;
    std::string  correction_type_at_initial_time   = "none";
    std::string  correction_type_after_restart     = "none";
    unsigned int maximum_non_linear_iterations_ic  = 5;
    double       ls_norm_factor                    = 1.;
#endif

    /** Build the deal.II IDA execution configuration from parsed values. */
#ifdef DEAL_II_WITH_SUNDIALS
    template <typename GlobalVectorType>
    typename dealii::SUNDIALS::IDA<GlobalVectorType>::AdditionalData
    ida_parameters() const
    {
      using AdditionalData =
        typename dealii::SUNDIALS::IDA<GlobalVectorType>::AdditionalData;

      AdditionalData data;
      const auto     correction_type = [](const std::string &value) {
        if (value == "use_y_diff")
          return AdditionalData::use_y_diff;
        if (value == "use_y_dot")
          return AdditionalData::use_y_dot;
        return AdditionalData::none;
      };

      data.initial_time = initial_time;
      data.final_time   = final_time;
      data.output_period =
        output_frequency > 0 ? output_frequency * time_step : time_step;
      data.initial_step_size             = initial_step_size;
      data.minimum_step_size             = minimum_step_size;
      data.maximum_order                 = maximum_order;
      data.maximum_non_linear_iterations = maximum_non_linear_iterations;
      data.absolute_tolerance            = absolute_tolerance;
      data.relative_tolerance            = relative_tolerance;
      data.ignore_algebraic_terms_for_errors =
        ignore_algebraic_terms_for_errors;
      data.ic_type    = correction_type(correction_type_at_initial_time);
      data.reset_type = correction_type(correction_type_after_restart);
      data.maximum_non_linear_iterations_ic = maximum_non_linear_iterations_ic;
      data.ls_norm_factor                   = ls_norm_factor;
      return data;
    }
#endif

    void
    declare_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter("Initial time", initial_time);
      prm.add_parameter("Final time", final_time);
      prm.add_parameter("Time step",
                        time_step,
                        "",
                        dealii::Patterns::Double(0));
      prm.add_parameter("Number of time steps", number_of_steps);
      prm.add_parameter("Output frequency", output_frequency);
      prm.add_parameter("Policy",
                        time_step_policy,
                        "Use number_of_steps to divide the interval, or fixed "
                        "to use the configured time step",
                        dealii::Patterns::Selection("number_of_steps|fixed"));
      prm.add_parameter("Refine time step", refine_time_step);
      prm.add_parameter("Newmark beta", newmark_beta);
      prm.add_parameter("Newmark gamma", newmark_gamma);

#ifdef DEAL_II_WITH_SUNDIALS
      prm.enter_subsection("Running parameters");
      prm.add_parameter("Initial step size", initial_step_size);
      prm.add_parameter("Minimum step size", minimum_step_size);
      prm.add_parameter("Maximum order of BDF", maximum_order);
      prm.add_parameter("Maximum number of nonlinear iterations",
                        maximum_non_linear_iterations);
      prm.leave_subsection();

      prm.enter_subsection("Error control");
      prm.add_parameter("Absolute error tolerance", absolute_tolerance);
      prm.add_parameter("Relative error tolerance", relative_tolerance);
      prm.add_parameter("Ignore algebraic terms for error computations",
                        ignore_algebraic_terms_for_errors);
      prm.leave_subsection();

      prm.enter_subsection("Initial condition correction parameters");
      prm.add_parameter("Correction type at initial time",
                        correction_type_at_initial_time,
                        "",
                        dealii::Patterns::Selection(
                          "none|use_y_diff|use_y_dot"));
      prm.add_parameter("Correction type after restart",
                        correction_type_after_restart,
                        "",
                        dealii::Patterns::Selection(
                          "none|use_y_diff|use_y_dot"));
      prm.add_parameter("Maximum number of nonlinear iterations",
                        maximum_non_linear_iterations_ic);
      prm.add_parameter(
        "Factor to use when converting from the integrator tolerance to the linear solver tolerance",
        ls_norm_factor);
      prm.leave_subsection();
#endif
    }
  };
} // namespace ImmersX

#endif
