// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on deal.II.
// ---------------------------------------------------------------------
#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <string>

#ifdef DEAL_II_WITH_SUNDIALS
namespace
{
  using namespace ImmersX;
  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = IDAAdapter<FieldVector, GlobalVector>;

  struct OutputResult
  {
    double        maximum_constraint = 0., maximum_time_gap = 0.;
    int           maximum_order = 0;
    unsigned int  outputs       = 0;
    unsigned long restarts      = 0;
  };

  OutputResult
  integrate_output_stops(const bool stop, const double period, const bool ramp)
  {
    TimeIntervalParameters time;
    IDAParameters          parameters;
    time.initial_time                                 = 7.2;
    time.final_time                                   = 8.;
    time.output_time_interval                         = period;
    parameters.initial_step_size                      = .003;
    parameters.maximum_step_size                      = .07;
    parameters.maximum_order                          = 2;
    parameters.absolute_tolerance                     = 1.e-5;
    parameters.relative_tolerance                     = 1.e-4;
    parameters.ignore_algebraic_terms_for_errors      = false;
    parameters.tolerance_ramp.enabled                 = ramp;
    parameters.tolerance_ramp.initial_absolute_factor = 10.;
    parameters.tolerance_ramp.initial_relative_factor = 10.;
    parameters.tolerance_ramp.end_time                = 7.6;
    Adapter adapter(time, parameters, MPI_COMM_WORLD);
    adapter.set_stop_at_output_times(stop);
    const double initial_time = time.initial_time;
    const auto   fields       = adapter.add([&](auto &builder) {
      using Model = typename std::decay_t<decltype(builder)>::Model;
      dealii::IndexSet owned(1);
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        owned.add_index(0);
      owned.compress();
      const auto y = builder.differential_field("y", owned);
      const auto z = builder.algebraic_field("z", owned);
      typename Model::OperatorFactory identity = [y](const auto &context) {
        const auto               reference = context.state(y);
        typename Model::Operator op;
        op.reinit_range_vector = [reference](FieldVector &v, bool) {
          v.reinit(reference);
        };
        op.reinit_domain_vector = op.reinit_range_vector;
        op.vmult = [](FieldVector &dst, const FieldVector &src) { dst = src; };
        op.vmult_add = [](FieldVector &dst, const FieldVector &src) {
          dst += src;
        };
        op.Tvmult     = op.vmult;
        op.Tvmult_add = op.vmult_add;
        return op;
      };
      const auto operation = [](const FieldVector &value) {
        typename Model::Operation op;
        op.reinit_vector = [value](FieldVector &v, bool) { v.reinit(value); };
        op.apply         = [value](FieldVector &v) { v = value; };
        op.apply_add     = [value](FieldVector &v) { v += value; };
        return op;
      };
      builder.term(y, "nonzero forcing")
        .residual([=](const auto &context) {
          FieldVector residual(context.derivative(y));
          residual.add(-(1. + context.time() - initial_time));
          return operation(residual);
        })
        .derivative(y, identity);
      builder.term(z, "nonlinear algebraic constraint")
        .residual([=](const auto &context) {
          FieldVector residual(context.state(z));
          const auto &value = context.state(y);
          for (const auto i : residual.locally_owned_elements())
            residual[i] -= std::exp(value[i]);
          residual.compress(dealii::VectorOperation::add);
          return operation(residual);
        })
        .state(z, identity)
        .state(y, typename Model::OperatorFactory([=](const auto &context) {
                 auto       op    = identity(context);
                 const auto value = context.state(y);
                 op.vmult = [value](FieldVector &dst, const FieldVector &src) {
                   for (const auto i : dst.locally_owned_elements())
                     dst[i] = -std::exp(value[i]) * src[i];
                   dst.compress(dealii::VectorOperation::insert);
                 };
                 op.vmult_add = [value](FieldVector       &dst,
                                        const FieldVector &src) {
                   for (const auto i : dst.locally_owned_elements())
                     dst[i] -= std::exp(value[i]) * src[i];
                   dst.compress(dealii::VectorOperation::add);
                 };
                 op.Tvmult     = op.vmult;
                 op.Tvmult_add = op.vmult_add;
                 return op;
               }));
      return std::make_pair(y, z);
    });
    auto state = adapter.make_state(), derivative = adapter.make_state();
    adapter.field(state, fields.fields().first)  = 0.;
    adapter.field(state, fields.fields().second) = 1.;
    derivative                                   = 1.;
    OutputResult result;
    long int     previous_steps = 0;
    adapter.set_output_step(
      [&](double t, const auto &values, const auto &, unsigned int) {
        const auto info = adapter.statistics();
        EXPECT_GE(info.steps, previous_steps);
        previous_steps       = info.steps;
        result.maximum_order = std::max(result.maximum_order, info.order);
        result.maximum_time_gap =
          std::max(result.maximum_time_gap, std::abs(t - info.current_time));
        const auto &y      = adapter.field(values, fields.fields().first);
        const auto &z      = adapter.field(values, fields.fields().second);
        double      defect = 0.;
        for (const auto i : y.locally_owned_elements())
          defect = std::max(defect, std::abs(z[i] - std::exp(y[i])));
        result.maximum_constraint =
          std::max(result.maximum_constraint,
                   dealii::Utilities::MPI::max(defect, MPI_COMM_WORLD));
        if (ramp && t < 7.6)
          {
            EXPECT_LE(info.current_time, 7.6 + 1.e-13);
          }
        ++result.outputs;
      });
    // Keep Newton error below the interpolation defect, using the same
    // convergence coefficient and LTE tolerances in both output modes.
    const auto setup              = adapter.solver().custom_setup;
    adapter.solver().custom_setup = [setup](void *memory) {
      if (setup)
        setup(memory);
      AssertThrow(IDASetNonlinConvCoef(memory, 1.e-4) == 0,
                  dealii::ExcMessage("Cannot configure Newton convergence."));
    };
    adapter.solve(state, derivative);
    auto         error    = adapter.field(state, fields.fields().first);
    const double duration = time.final_time - time.initial_time;
    error.add(-duration - .5 * duration * duration);
    EXPECT_LT(error.l2_norm(), 5.e-4);
    result.restarts = adapter.statistics().restarts;
    EXPECT_NEAR(adapter.statistics().current_time, time.final_time, 1.e-13);
    EXPECT_EQ(result.restarts, 0u);
    EXPECT_EQ(result.maximum_order, 2);
    EXPECT_THROW(adapter.set_stop_at_output_times(false),
                 dealii::ExceptionBase);
    return result;
  }
  auto
  add_logarithmic_field(Adapter                             &adapter,
                        const std::function<double(double)> &forcing,
                        unsigned int                        &invalid_trials)
  {
    return adapter.add([&](auto &builder) {
      using Model = typename std::decay_t<decltype(builder)>::Model;
      dealii::IndexSet owned(1);
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        owned.add_index(0);
      owned.compress();
      const auto field = builder.algebraic_field("positive value", owned);
      builder.term(field, "logarithmic constraint")
        .residual([field, forcing, &invalid_trials](const auto &context) {
          const auto &value = context.state(field);
          FieldVector residual(value);
          for (const auto i : residual.locally_owned_elements())
            if (value[i] <= 0.)
              {
                ++invalid_trials;
                residual[i] = std::numeric_limits<double>::quiet_NaN();
              }
            else
              residual[i] = std::log(value[i]) + forcing(context.time());
          residual.compress(dealii::VectorOperation::insert);
          typename Model::Operation result;
          result.reinit_vector = [value](FieldVector &v, bool) {
            v.reinit(value);
          };
          result.apply     = [residual](FieldVector &v) { v = residual; };
          result.apply_add = [residual](FieldVector &v) { v += residual; };
          return result;
        })
        .state(field,
               typename Model::OperatorFactory([field](const auto &context) {
                 const auto               value = context.state(field);
                 typename Model::Operator op;
                 op.reinit_range_vector = [value](FieldVector &v, bool) {
                   v.reinit(value);
                 };
                 op.reinit_domain_vector = op.reinit_range_vector;
                 op.vmult = [value](FieldVector &dst, const FieldVector &src) {
                   dst = src;
                   for (const auto i : dst.locally_owned_elements())
                     dst[i] /= value[i];
                   dst.compress(dealii::VectorOperation::insert);
                 };
                 op.vmult_add = [value](FieldVector       &dst,
                                        const FieldVector &src) {
                   FieldVector contribution(src);
                   for (const auto i : contribution.locally_owned_elements())
                     contribution[i] /= value[i];
                   contribution.compress(dealii::VectorOperation::insert);
                   dst += contribution;
                 };
                 op.Tvmult     = op.vmult;
                 op.Tvmult_add = op.vmult_add;
                 return op;
               }));
      return field;
    });
  }

} // namespace

TEST(DistributedIDA, BOTH_OutputStopsPreserveNonlinearAlgebraicConstraint)
{
  const auto number = [](double value) {
    std::ostringstream text;
    text.precision(17);
    text << value;
    return text.str();
  };
  for (const double period : {.1, .01})
    {
      SCOPED_TRACE(period);
      const auto dense   = integrate_output_stops(false, period, false);
      const auto stopped = integrate_output_stops(true, period, false);
      RecordProperty("dense_constraint_" + std::to_string(period),
                     number(dense.maximum_constraint));
      RecordProperty("stopped_constraint_" + std::to_string(period),
                     number(stopped.maximum_constraint));
      RecordProperty("stopped_outputs_" + std::to_string(period),
                     stopped.outputs);
      EXPECT_GT(dense.maximum_time_gap, 1.e-5);
      EXPECT_GT(dense.maximum_constraint, 1.e-8);
      EXPECT_LT(stopped.maximum_time_gap, 1.e-13);
      EXPECT_LT(stopped.maximum_constraint, 1.e-6);
      EXPECT_LT(stopped.maximum_constraint, dense.maximum_constraint * .1);
      EXPECT_GE(stopped.outputs,
                static_cast<unsigned int>(std::lround(.8 / period)) + 1u);
    }
}

TEST(DistributedIDA, BOTH_OutputStopsRespectToleranceRamp)
{
  const auto stopped = integrate_output_stops(true, .01, true);
  EXPECT_LT(stopped.maximum_time_gap, 1.e-13);
  EXPECT_LT(stopped.maximum_constraint, 1.e-5);
}

TEST(DistributedIDA, BOTH_NonfiniteInitialResidualFailsCollectively)
{
  TimeIntervalParameters time;
  IDAParameters          parameters;
  time.initial_time                          = 0.;
  time.final_time                            = .02;
  parameters.correction_type_at_initial_time = "use_y_dot";
  unsigned int invalid_trials                = 0;
  Adapter      adapter(time, parameters, MPI_COMM_WORLD);
  const auto   positive = add_logarithmic_field(
    adapter, [](double t) { return -std::log(2. + t); }, invalid_trials);
  auto state = adapter.make_state(), derivative = adapter.make_state();
  adapter.field(state, positive.fields()) = -1.;
  derivative                              = 0.;
  EXPECT_THROW(adapter.solve(state, derivative),
               dealii::RecoverableUserCallbackError);
  EXPECT_GT(dealii::Utilities::MPI::sum(invalid_trials, MPI_COMM_WORLD), 0u);
}

TEST(DistributedIDA, BOTH_NonfiniteTrialResidualIsRecoverable)
{
  TimeIntervalParameters time;
  IDAParameters          parameters;
  time.initial_time                            = 0.;
  time.final_time                              = .02;
  time.output_time_interval                    = .02;
  parameters.initial_step_size                 = .02;
  parameters.maximum_step_size                 = .02;
  parameters.maximum_order                     = 5;
  parameters.maximum_number_of_steps           = 10000;
  parameters.absolute_tolerance                = 1.e-10;
  parameters.relative_tolerance                = 1.e-9;
  parameters.ignore_algebraic_terms_for_errors = false;
  parameters.correction_type_at_initial_time   = "none";
  unsigned int invalid_trials                  = 0;
  Adapter      adapter(time, parameters, MPI_COMM_WORLD);
  const auto   positive = add_logarithmic_field(
    adapter, [](double t) { return 100. * t; }, invalid_trials);
  auto state = adapter.make_state(), derivative = adapter.make_state();
  adapter.field(state, positive.fields()) = 1.;
  derivative                              = -100.;
  // exp(-100 t) is consistent initially, but the requested first predictor
  // crosses zero. IDA must reject it and reduce its time step.
  adapter.solve(state, derivative);
  const auto rejected =
    dealii::Utilities::MPI::sum(invalid_trials, MPI_COMM_WORLD);
  const double value = adapter.field(state, positive.fields()).l2_norm();
  EXPECT_GT(rejected, 0u);
  EXPECT_NEAR(value, std::exp(-2.), 1.e-7);
  RecordProperty("invalid_trials", std::to_string(rejected));
  std::ostringstream error;
  error.precision(17);
  error << std::abs(value - std::exp(-2.));
  RecordProperty("final_error", error.str());
}
TEST(DistributedIDA, BOTH_PositiveConstraintProtectsInitialCorrection)
{
  TimeIntervalParameters time;
  IDAParameters          parameters;
  time.initial_time                          = 0.;
  time.final_time                            = .02;
  time.output_time_interval                  = .01;
  parameters.initial_step_size               = .001;
  parameters.maximum_step_size               = .01;
  parameters.maximum_order                   = 5;
  parameters.absolute_tolerance              = 1.e-10;
  parameters.relative_tolerance              = 1.e-9;
  parameters.correction_type_at_initial_time = "use_y_dot";
  unsigned int invalid_trials                = 0;
  Adapter      adapter(time, parameters, MPI_COMM_WORLD);
  const auto   positive = add_logarithmic_field(
    adapter, [](double t) { return -std::log(2. + t); }, invalid_trials);
  auto state = adapter.make_state(), derivative = adapter.make_state();
  adapter.field(state, positive.fields()) = 10.;
  derivative                              = 0.;
  auto constraints                        = adapter.make_state();
  constraints                             = 2.;
  adapter.solver().get_constraint_vector  = [&constraints]() -> GlobalVector  &{
    return constraints;
  };
  adapter.solve(state, derivative);
  EXPECT_EQ(dealii::Utilities::MPI::sum(invalid_trials, MPI_COMM_WORLD), 0u);
  EXPECT_NEAR(adapter.field(state, positive.fields()).l2_norm(), 2.02, 1.e-7);
}

TEST(DistributedIDA, BOTH_InitialCorrectionReusesFirstPassJacobian)
{
  TimeIntervalParameters time;
  IDAParameters          parameters;
  time.initial_time                          = 0.;
  time.final_time                            = .02;
  time.output_time_interval                  = .01;
  parameters.initial_step_size               = .001;
  parameters.maximum_step_size               = .01;
  parameters.maximum_order                   = 5;
  parameters.absolute_tolerance              = 1.e-10;
  parameters.relative_tolerance              = 1.e-9;
  parameters.correction_type_at_initial_time = "use_y_dot";
  unsigned int invalid_trials                = 0;
  Adapter      adapter(time, parameters, MPI_COMM_WORLD);
  const auto   positive = add_logarithmic_field(
    adapter, [](double t) { return -std::log(2. + t); }, invalid_trials);
  double initial_value = 0.;
  adapter.set_output_step([&](const double        t,
                              const GlobalVector &state,
                              const GlobalVector &,
                              const unsigned int) {
    if (t == time.initial_time)
      initial_value = adapter.field(state, positive.fields()).l2_norm();
  });
  auto state = adapter.make_state(), derivative = adapter.make_state();
  // The frozen Jacobian 1/3 needs several corrections to reach log(y) =
  // log(2), and every Newton step from y = 3 stays positive.
  adapter.field(state, positive.fields()) = 3.;
  derivative                              = 0.;
  testing::internal::CaptureStdout();
  adapter.solve(state, derivative);
  const std::string output = testing::internal::GetCapturedStdout();
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      const std::string reuse =
        "reusing the initial-condition Jacobian for IDA's second correction "
        "pass";
      // Rank 1 continues into collective norms, so do not return early.
      const auto first = output.find(reuse);
      EXPECT_NE(first, std::string::npos) << output;
      if (first != std::string::npos)
        EXPECT_EQ(output.find(reuse, first + 1), std::string::npos) << output;
    }
  EXPECT_NEAR(initial_value, 2., 1.e-8);
  EXPECT_NEAR(adapter.field(state, positive.fields()).l2_norm(), 2.02, 1.e-7);
}
#endif
