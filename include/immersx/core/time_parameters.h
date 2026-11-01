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
#  include <immersx/compatibility/dealii_9_9/ida.h>
#endif

#include <immersx/core/ida_tolerance_ramp.h>

#include <string>

namespace ImmersX
{
  /** Solver-independent simulation interval and output schedule. */
  class TimeIntervalParameters : public dealii::ParameterAcceptor
  {
  public:
    /** Set expose_parameters to false when IDAParameters registers this
     * interval in its own native IDA section.
     */
    explicit TimeIntervalParameters(
      const std::string &subsection        = "/Time interval/",
      const bool         expose_parameters = true)
      : dealii::ParameterAcceptor(expose_parameters ? subsection : "/")
      , expose_parameters_(expose_parameters)
    {
      parse_parameters_call_back.connect([this]() {
        AssertThrow(final_time >= initial_time,
                    dealii::ExcMessage(
                      "Final time must not precede initial time."));
        if (final_time > initial_time)
          AssertThrow(output_time_interval > 0.,
                      dealii::ExcMessage(
                        "Output time interval must be positive for a transient "
                        "run."));
      });
    }

    double initial_time         = 0.;
    double final_time           = 0.1;
    double output_time_interval = 1.e-2;

    void
    declare_parameters(dealii::ParameterHandler &prm) override
    {
      if (!expose_parameters_)
        return;
      prm.add_parameter("Initial time", initial_time);
      prm.add_parameter("Final time", final_time);
      prm.add_parameter("Output time interval",
                        output_time_interval,
                        "Physical time between accepted output states",
                        dealii::Patterns::Double(0));
    }

  private:
    const bool expose_parameters_;
  };

  /** Fixed-step integration policy for applications that do not use IDA. */
  class FixedStepParameters : public dealii::ParameterAcceptor
  {
  public:
    /** Set expose_parameters to false for an IDA-only execution path. */
    explicit FixedStepParameters(const std::string &subsection = "/Fixed step/",
                                 const bool         expose_parameters = true)
      : dealii::ParameterAcceptor(expose_parameters ? subsection : "/")
      , expose_parameters_(expose_parameters)
    {}

    double       time_step          = 1.e-2;
    unsigned int number_of_steps    = 0;
    std::string  time_step_policy   = "number_of_steps";
    std::string  time_step_strategy = "backward_euler";
    bool         refine_time_step   = false;

    void
    declare_parameters(dealii::ParameterHandler &prm) override
    {
      if (!expose_parameters_)
        return;
      prm.add_parameter("Time step",
                        time_step,
                        "Fixed timestep used by the integrator",
                        dealii::Patterns::Double(0));
      prm.add_parameter("Number of time steps", number_of_steps);
      prm.add_parameter("Policy",
                        time_step_policy,
                        "Use number_of_steps to divide the interval, or fixed "
                        "to use the configured time step",
                        dealii::Patterns::Selection("number_of_steps|fixed"));
      prm.add_parameter(
        "Strategy",
        time_step_strategy,
        "Time integration strategy used by the standalone solver",
        dealii::Patterns::Selection("backward_euler|trapezoidal"));
      prm.add_parameter("Refine time step", refine_time_step);
    }

  private:
    const bool expose_parameters_;
  };

  /** IDA-specific adaptive integration and nonlinear-solver controls. */
  class IDAParameters : public dealii::ParameterAcceptor
#ifdef DEAL_II_WITH_SUNDIALS
    ,
                        public Compatibility::SUNDIALS::IDA<
                          dealii::Vector<double>>::AdditionalData
#endif
  {
  public:
#ifdef DEAL_II_WITH_SUNDIALS
    using Controls =
      Compatibility::SUNDIALS::IDA<dealii::Vector<double>>::AdditionalData;
#endif
    /** Pass an unexposed interval to register native IDA time entries here.
     * Keep the interval alive while this object or its bindings are used.
     */
    explicit IDAParameters(const std::string      &subsection      = "/IDA/",
                           TimeIntervalParameters *shared_interval = nullptr)
      : dealii::ParameterAcceptor(subsection)
      , shared_interval_(shared_interval)
    {
#ifdef DEAL_II_WITH_SUNDIALS
      // Preserve the adapter defaults used by existing parameter files.
      initial_step_size             = 1.e-5;
      minimum_step_size             = 0.;
      maximum_order                 = 1;
      maximum_non_linear_iterations = 50;
      absolute_tolerance = relative_tolerance = 1.e-4;
      ls_norm_factor                          = 1.;
#endif
    }

    IDAToleranceRamp tolerance_ramp;
    bool             print_statistics = false;

    std::string correction_type_at_initial_time = "none";
    std::string correction_type_after_restart   = "none";

#ifdef DEAL_II_WITH_SUNDIALS
    template <typename GlobalVectorType>
    typename Compatibility::SUNDIALS::IDA<GlobalVectorType>::AdditionalData
    additional_data(const TimeIntervalParameters &interval) const
    {
      using Data =
        typename Compatibility::SUNDIALS::IDA<GlobalVectorType>::AdditionalData;
      Data data;
      data.initial_step_size            = initial_step_size;
      data.minimum_step_size            = minimum_step_size;
      data.maximum_step_size            = maximum_step_size;
      data.maximum_number_of_steps      = maximum_number_of_steps;
      data.maximum_error_test_failures  = maximum_error_test_failures;
      data.maximum_convergence_failures = maximum_convergence_failures;
      data.nonlinear_convergence_coefficient =
        nonlinear_convergence_coefficient;
      data.maximum_order                 = maximum_order;
      data.maximum_non_linear_iterations = maximum_non_linear_iterations;
      data.absolute_tolerance            = absolute_tolerance;
      data.relative_tolerance            = relative_tolerance;
      data.ignore_algebraic_terms_for_errors =
        ignore_algebraic_terms_for_errors;
      data.maximum_non_linear_iterations_ic = maximum_non_linear_iterations_ic;
      data.ls_norm_factor                   = ls_norm_factor;
      data.maximum_number_of_steps_ic       = maximum_number_of_steps_ic;
      data.maximum_number_of_jacobians_ic   = maximum_number_of_jacobians_ic;
      data.maximum_number_of_backtracks_ic  = maximum_number_of_backtracks_ic;
      data.line_search_enabled_ic           = line_search_enabled_ic;
      data.step_tolerance_ic                = step_tolerance_ic;
      data.nonlinear_convergence_coefficient_ic =
        nonlinear_convergence_coefficient_ic;
      data.linear_tolerance_factor = linear_tolerance_factor;
      data.increment_factor        = increment_factor;
      data.linear_solution_scaling = linear_solution_scaling;
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 2, 0)
      data.delta_cj_lsetup  = delta_cj_lsetup;
      data.eta_min          = eta_min;
      data.eta_max          = eta_max;
      data.eta_low          = eta_low;
      data.eta_min_err_fail = eta_min_err_fail;
      data.eta_conv_fail    = eta_conv_fail;
      data.eta_min_fixed    = eta_min_fixed;
      data.eta_max_fixed    = eta_max_fixed;
#  endif
#  if DEAL_II_SUNDIALS_VERSION_GTE(7, 6, 0)
      data.maximum_constraint_failures = maximum_constraint_failures;
#  endif
      const auto correction = [](const std::string &value) {
        if (value == "use_y_diff")
          return Data::use_y_diff;
        else if (value == "use_y_dot")
          return Data::use_y_dot;
        AssertThrow(value == "none",
                    dealii::ExcMessage("Invalid IDA correction type: " +
                                       value));
        return Data::none;
      };
      data.initial_time  = interval.initial_time;
      data.final_time    = interval.final_time;
      data.output_period = interval.output_time_interval;
      data.ic_type       = correction(correction_type_at_initial_time);
      data.reset_type    = correction(correction_type_after_restart);
      return data;
    }
#endif

    void
    declare_parameters(dealii::ParameterHandler &prm) override
    {
#ifdef DEAL_II_WITH_SUNDIALS
      if (shared_interval_ != nullptr)
        {
          prm.add_parameter("Initial time", shared_interval_->initial_time);
          prm.add_parameter("Final time", shared_interval_->final_time);
          prm.add_parameter("Time interval between each output",
                            shared_interval_->output_time_interval);
          // A native solver may already have bound these same entries.
          for (const auto &entry : {"Initial time",
                                    "Final time",
                                    "Time interval between each output"})
            prm.set(entry, prm.get(entry));
        }
      Controls::add_parameters(prm, false, false, shared_interval_ != nullptr);
      tolerance_ramp.add_parameters(prm);
      prm.add_parameter("Print current statistics", print_statistics);
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
      if (shared_interval_ != nullptr)
        for (const auto &entry : {"Correction type at initial time",
                                  "Correction type after restart"})
          prm.set(entry, prm.get(entry));
      prm.leave_subsection();
#else
      (void)prm;
#endif
    }

  private:
    TimeIntervalParameters *shared_interval_;
  };

  /** Legacy configuration retained only by the production elasticity path. */
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
    typename Compatibility::SUNDIALS::IDA<GlobalVectorType>::AdditionalData
    ida_parameters() const
    {
      using AdditionalData =
        typename Compatibility::SUNDIALS::IDA<GlobalVectorType>::AdditionalData;

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
