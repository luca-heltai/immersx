// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on deal.II.
// ---------------------------------------------------------------------
#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

#ifdef DEAL_II_WITH_SUNDIALS
#  include <deal.II/sundials/ida.h>
namespace
{
  using namespace ImmersX;
  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = IDAAdapter<FieldVector, GlobalVector>;

  // All DoFs belong to rank zero, so MPI runs also cover empty execution ranks.
  template <typename AdapterType>
  auto
  add_forced_field(AdapterType                 &adapter,
                   const bool                   differential,
                   const double                 scale,
                   const std::function<void()> &probe = {})
  {
    return adapter.add(
      [=](auto &builder) {
        using Model = typename std::decay_t<decltype(builder)>::Model;
        dealii::IndexSet owned(1);
        if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          owned.add_index(0);
        owned.compress();
        const auto                      field = differential ?
                                                  builder.differential_field("value", owned) :
                                                  builder.algebraic_field("value", owned);
        typename Model::OperatorFactory identity =
          [field](const auto &context) {
            const auto              &reference = context.state(field);
            typename Model::Operator op;
            op.reinit_range_vector = [reference](FieldVector &v, bool) {
              v.reinit(reference);
            };
            op.reinit_domain_vector = op.reinit_range_vector;
            op.vmult = [](FieldVector &dst, const FieldVector &src) {
              dst = src;
            };
            op.vmult_add = [](FieldVector &dst, const FieldVector &src) {
              dst += src;
            };
            op.Tvmult     = op.vmult;
            op.Tvmult_add = op.vmult_add;
            return op;
          };
        auto term = builder.term(field, "forced");
        term
          .residual([=](const auto &context) {
            if (probe)
              probe();
            const auto  &y = context.state(field);
            const double t = context.time();
            FieldVector  value;
            value.reinit(y);
            value = y;
            if (differential)
              value += context.derivative(field);
            value.add(-scale *
                      (2. + std::sin(t) + (differential ? std::cos(t) : 0.)));
            typename Model::Operation result;
            result.reinit_vector = [y](FieldVector &v, bool) { v.reinit(y); };
            result.apply         = [value](FieldVector &v) { v = value; };
            result.apply_add     = [value](FieldVector &v) { v += value; };
            return result;
          })
          .state(field, identity);
        if (differential)
          term.derivative(field, identity);
        return field;
      },
      differential ? "differential" : "algebraic");
  }

  struct RunResult
  {
    double              differential, algebraic;
    Adapter::Statistics statistics;
    unsigned int        tolerance_builds;
    std::string         diagnostic_output;
    std::vector<double> expected_correction_norms;
  };

  RunResult
  integrate(const bool   local,
            const double factor,
            const bool   output,
            const bool   restart       = false,
            const int    participation = 0,
            const bool   check_weights = false,
            const bool   diagnostics   = false)
  {
    std::ostringstream diagnostic_log;
    struct RestoreStream
    {
      std::streambuf *original;
      ~RestoreStream()
      {
        std::cout.rdbuf(original);
      }
    } restore{std::cout.rdbuf(diagnostic_log.rdbuf())};
    std::vector<double>    expected_correction_norms;
    TimeIntervalParameters time;
    IDAParameters          parameters;
    parameters.print_nonlinear_diagnostics       = diagnostics;
    time.initial_time                            = .25;
    time.final_time                              = .35;
    time.output_time_interval                    = .01;
    parameters.maximum_order                     = 3;
    parameters.initial_step_size                 = 1.e-3;
    parameters.maximum_step_size                 = .005;
    parameters.absolute_tolerance                = 1.e-10;
    parameters.relative_tolerance                = 1.e-9;
    parameters.ignore_algebraic_terms_for_errors = false;
    parameters.correction_type_at_initial_time   = "use_y_diff";
    parameters.tolerance_ramp.enabled            = factor > 0.;
    parameters.tolerance_ramp.initial_absolute_factor =
      factor > 0. ? factor : 1.;
    parameters.tolerance_ramp.initial_relative_factor =
      factor > 0. ? factor : 1.;
    parameters.tolerance_ramp.hold_duration = .01;
    parameters.tolerance_ramp.end_time      = .30;
    Adapter      adapter(time, parameters, MPI_COMM_WORLD);
    const auto   small  = add_forced_field(adapter, true, 1.e-3);
    const auto   large  = add_forced_field(adapter, false, 1.e3);
    unsigned int builds = 0;
    if (local)
      {
        adapter.set_absolute_tolerance(small.fields(), 1.e-12);
        adapter.set_local_tolerances(large.fields(), [&](FieldVector &v) {
          ++builds;
          v = 1.e-7;
        });
      }
    if (participation != 0)
      {
        adapter.set_tolerance_ramp_enabled(small.fields(), false);
        if (participation > 0)
          adapter.set_tolerance_ramp_enabled(small.fields(), true);
      }
    if (check_weights)
      adapter.set_stop_at_output_times(true);
    unsigned int initial_weight_checks = 0, reset_weight_checks = 0,
                 output_weight_checks = 0;
    bool initial_output_seen = false, reset_pending = false;
    std::function<void(const GlobalVector &, GlobalVector &)> native_weights;
    const auto check_field_weights =
      [&](const auto &state, const auto &weights, const double multiplier) {
        for (const auto index :
             adapter.field(state, small.fields()).locally_owned_elements())
          {
            const double denominator =
              1.e-12 +
              1.e-9 * std::abs(adapter.field(state, small.fields())[index]);
            EXPECT_NEAR(adapter.field(weights, small.fields())[index] *
                          denominator * (participation < 0 ? 1. : multiplier),
                        1.,
                        1.e-13);
          }
        for (const auto index :
             adapter.field(state, large.fields()).locally_owned_elements())
          {
            const double denominator =
              1.e-7 +
              1.e-9 * std::abs(adapter.field(state, large.fields())[index]);
            EXPECT_NEAR(adapter.field(weights, large.fields())[index] *
                          denominator * multiplier,
                        1.,
                        1.e-13);
          }
      };
    auto y = adapter.make_state(), yp = adapter.make_state();
    adapter.field(y, small.fields()) =
      1.e-3 * (2. + std::sin(time.initial_time));
    adapter.field(y, large.fields()) = 0.;
    yp                               = 0.;
    if (restart)
      adapter.set_solver_should_restart([&](double t, auto &, auto &) {
        const bool due       = check_weights ? t >= .28 - 1.e-14 : t > .26;
        const bool requested = due && adapter.statistics().restarts == 0;
        if (requested && check_weights)
          reset_pending = true;
        return requested;
      });
    unsigned int outputs        = 0;
    long int     previous_steps = 0;
    if (output)
      adapter.set_output_step([&](double t,
                                  const auto &,
                                  const auto &,
                                  unsigned int step) {
        ++outputs;
        initial_output_seen = true;
        if (check_weights)
          {
            // Fixed probe values make the expected denominators independent
            // of the numerical trajectory and the ramp implementation.
            for (const auto &point : {std::pair<double, double>{.25, 100.},
                                      {.26, 100.},
                                      {.28, 10.},
                                      {.30, 1.},
                                      {.35, 1.}})
              if (std::abs(t - point.first) < 1.e-13)
                {
                  auto probe                           = adapter.make_state();
                  auto weights                         = adapter.make_state();
                  adapter.field(probe, small.fields()) = 3.e-3;
                  adapter.field(probe, large.fields()) = 3.e3;
                  native_weights(probe, weights);
                  for (const auto index : adapter.field(weights, small.fields())
                                            .locally_owned_elements())
                    EXPECT_NEAR(adapter.field(weights, small.fields())[index] *
                                  4.e-12 *
                                  (participation < 0 ? 1. : point.second),
                                1.,
                                1.e-13);
                  for (const auto index : adapter.field(weights, large.fields())
                                            .locally_owned_elements())
                    EXPECT_NEAR(adapter.field(weights, large.fields())[index] *
                                  3.1e-6 * point.second,
                                1.,
                                1.e-13);
                  ++output_weight_checks;
                }
          }
        const auto stats = adapter.statistics();
        EXPECT_GE(stats.steps, previous_steps);
        previous_steps = stats.steps;
        if (step == 0)
          {
            EXPECT_DOUBLE_EQ(t, time.initial_time);
          }
        if (parameters.tolerance_ramp.enabled && t < .30)
          {
            EXPECT_LE(stats.current_time, .30 + 1.e-14);
          }
      });
    if (check_weights)
      {
        auto &solver                 = adapter.solver();
        native_weights               = solver.compute_error_weights;
        solver.compute_error_weights = [&](const auto &state, auto &weights) {
          native_weights(state, weights);
          if (!initial_output_seen)
            {
              check_field_weights(state, weights, 100.);
              ++initial_weight_checks;
            }
          if (reset_pending)
            {
              double current_time;
              EXPECT_EQ(IDAGetCurrentTime(solver.get_ida_memory(),
                                          &current_time),
                        0);
              EXPECT_NEAR(current_time, .28, 1.e-13);
              check_field_weights(state, weights, 10.);
              ++reset_weight_checks;
              reset_pending = false;
            }
        };
      }
    if (diagnostics)
      {
        auto      &solver          = adapter.solver();
        const auto solve           = solver.solve_with_jacobian;
        solver.solve_with_jacobian = [&, solve](const auto  &rhs,
                                                auto        &dst,
                                                const double tolerance) {
          solve(rhs, dst, tolerance);
          auto weights = adapter.make_state();
          auto weight_view =
            Compatibility::SUNDIALS::internal::make_nvector_view(
              weights
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
              ,
              solver.get_sun_context()
#  endif
            );
          auto correction_view =
            Compatibility::SUNDIALS::internal::make_nvector_view(
              dst
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
              ,
              solver.get_sun_context()
#  endif
            );
          EXPECT_EQ(IDAGetErrWeights(solver.get_ida_memory(), weight_view), 0);
          expected_correction_norms.push_back(
            N_VWrmsNorm(correction_view, weight_view));
        };
      }
    const auto steps = adapter.solve(y, yp);
    if (check_weights)
      {
        EXPECT_GT(initial_weight_checks, 0u);
        EXPECT_EQ(reset_weight_checks, restart ? 1u : 0u);
        EXPECT_EQ(output_weight_checks, 5u);
      }
    const auto stats = adapter.statistics();
    EXPECT_EQ(steps, static_cast<unsigned int>(stats.steps));
    EXPECT_EQ(stats.restarts, restart ? 1u : 0u);
    EXPECT_EQ(outputs, output ? 11u : 0u);
    EXPECT_GT(stats.jacobian_setups, 0u);
    EXPECT_GT(stats.residual_evaluations, 0);
    if (local || parameters.tolerance_ramp.enabled)
      {
        EXPECT_GT(stats.weight_evaluations, 0u);
      }
    EXPECT_DOUBLE_EQ(stats.tolerance_scales.first, 1.);
    EXPECT_DOUBLE_EQ(stats.tolerance_scales.second, 1.);
    if (local || parameters.tolerance_ramp.enabled)
      {
        auto weights = adapter.make_state();
        adapter.solver().compute_error_weights(y, weights);
        for (const auto index :
             adapter.field(y, small.fields()).locally_owned_elements())
          EXPECT_NEAR(adapter.field(weights, small.fields())[index] *
                        ((local ? 1.e-12 : 1.e-10) +
                         1.e-9 *
                           std::abs(adapter.field(y, small.fields())[index])),
                      1.,
                      1.e-14);
        for (const auto index :
             adapter.field(y, large.fields()).locally_owned_elements())
          EXPECT_NEAR(adapter.field(weights, large.fields())[index] *
                        ((local ? 1.e-7 : 1.e-10) +
                         1.e-9 *
                           std::abs(adapter.field(y, large.fields())[index])),
                      1.,
                      1.e-14);
      }
    auto small_error = adapter.field(y, small.fields());
    small_error.add(-1.e-3 * (2. + std::sin(time.final_time)));
    EXPECT_LT(small_error.l2_norm(), 1.e-9);
    auto large_error = adapter.field(y, large.fields());
    large_error.add(-1.e3 * (2. + std::sin(time.final_time)));
    EXPECT_LT(large_error.l2_norm(), 1.e-5);
    return {adapter.field(y, small.fields()).l2_norm(),
            adapter.field(y, large.fields()).l2_norm(),
            stats,
            builds,
            diagnostic_log.str(),
            expected_correction_norms};
  }
} // namespace

TEST(DistributedIDA, BOTH_RepeatedNativeResetSynchronizesStateAndOutput)
{
  using NativeIDA = Compatibility::SUNDIALS::IDA<FieldVector>;
  NativeIDA::AdditionalData data;
  data.initial_time                      = .25;
  data.final_time                        = .27;
  data.output_period                     = .01;
  data.initial_step_size                 = .001;
  data.absolute_tolerance                = 1.e-12;
  data.relative_tolerance                = 1.e-12;
  data.ic_type                           = NativeIDA::AdditionalData::use_y_dot;
  data.ignore_algebraic_terms_for_errors = false;
  NativeIDA        ida(data, MPI_COMM_WORLD);
  dealii::IndexSet owned(3);
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    owned.add_range(0, 3);
  owned.compress();
  FieldVector state(owned, MPI_COMM_WORLD), derivative(owned, MPI_COMM_WORLD);
  state               = 0.;
  derivative          = 0.;
  double       target = 0.;
  unsigned int setups = 0, outputs = 0;
  bool         fail_setup = true;
  ida.reinit_vector = [&](FieldVector &v) { v.reinit(owned, MPI_COMM_WORLD); };
  ida.residual      = [&](const double,
                     const FieldVector &y,
                     const FieldVector &,
                     FieldVector &residual) {
    for (const auto i : owned)
      residual[i] = y[i] - (target + i);
    residual.compress(dealii::VectorOperation::insert);
  };
  ida.setup_jacobian = [](const double,
                          const FieldVector &,
                          const FieldVector &,
                          const double alpha) { EXPECT_GE(alpha, 0.); };
  ida.solve_with_jacobian =
    [](const FieldVector &rhs, FieldVector &dst, const double) { dst = rhs; };
  ida.custom_setup = [&](void *memory) {
    ++setups;
    EXPECT_EQ(memory, ida.get_ida_memory());
    if (fail_setup)
      throw std::runtime_error(
        "Expected setup failure after linear algebra allocation");
  };
  const auto check_state = [&](const FieldVector &y, const FieldVector &ydot) {
    for (const auto i : owned)
      {
        EXPECT_NEAR(y[i], target + i, 1.e-11);
        EXPECT_DOUBLE_EQ(ydot[i], 0.);
      }
  };
  ida.output_step = [&](const double       time,
                        const FieldVector &y,
                        const FieldVector &ydot,
                        const unsigned int step) {
    EXPECT_EQ(step, outputs);
    EXPECT_NEAR(time, data.initial_time + outputs * data.output_period, 1.e-14);
    ++outputs;
    check_state(y, ydot);
  };
  EXPECT_THROW(
    ida.reset(data.initial_time, data.initial_step_size, state, derivative),
    std::runtime_error);
  fail_setup = false;
  EXPECT_EQ(outputs, 0u);
  for (const double value : {2., 4., 1.})
    {
      target = value;
      ida.reset(data.initial_time, data.initial_step_size, state, derivative);
      check_state(state, derivative);
      double   time  = 0.;
      long int steps = -1;
      EXPECT_EQ(IDAGetCurrentTime(ida.get_ida_memory(), &time), IDA_SUCCESS);
      EXPECT_EQ(IDAGetNumSteps(ida.get_ida_memory(), &steps), IDA_SUCCESS);
      EXPECT_DOUBLE_EQ(time, data.initial_time);
      EXPECT_EQ(steps, 0);
      EXPECT_EQ(outputs, 0u);
    }
  EXPECT_EQ(setups, 4u);
  // Integration resets once more, then emits the corrected initial state
  // and the two requested physical output times.
  EXPECT_GT(ida.solve_dae(state, derivative), 0u);
  EXPECT_EQ(setups, 5u);
  EXPECT_EQ(outputs, 3u);
  check_state(state, derivative);
}

TEST(DistributedIDA, BOTH_LocalTolerancesAndHistoryPreservingRamp)
{
  const auto baseline   = integrate(false, 0., true);
  const auto factor_one = integrate(false, 1., true);
  EXPECT_NEAR(baseline.differential, factor_one.differential, 1.e-10);
  EXPECT_NEAR(baseline.algebraic, factor_one.algebraic, 1.e-6);
  const auto local = integrate(true, 1., true);
  EXPECT_EQ(local.tolerance_builds, 1u);
  const auto ramp = integrate(true, 100., false);
  EXPECT_EQ(ramp.tolerance_builds, 1u);
  const auto restarted = integrate(true, 100., true, true);
  EXPECT_EQ(restarted.tolerance_builds, 2u);
}

TEST(DistributedIDA, BOTH_FieldToleranceRampWeightsAndReset)
{
  const auto defaults = integrate(true, 100., true, false, 0, true);
  const auto enabled  = integrate(true, 100., true, false, 1, true);
  EXPECT_DOUBLE_EQ(defaults.differential, enabled.differential);
  EXPECT_DOUBLE_EQ(defaults.algebraic, enabled.algebraic);
  EXPECT_EQ(defaults.statistics.steps, enabled.statistics.steps);
  EXPECT_EQ(defaults.statistics.jacobian_setups,
            enabled.statistics.jacobian_setups);
  const auto fixed = integrate(true, 100., true, true, -1, true);
  EXPECT_EQ(fixed.tolerance_builds, 2u);
}

TEST(DistributedIDA,
     BOTH_NonlinearDiagnosticsPreserveTrajectoryAndNativeWeights)
{
  EXPECT_FALSE(IDAParameters{}.print_nonlinear_diagnostics);
  for (const bool local : {false, true})
    {
      const auto plain = integrate(local, local ? 100. : 0., false);
      const auto logged =
        integrate(local, local ? 100. : 0., false, false, 0, false, true);
      EXPECT_DOUBLE_EQ(plain.differential, logged.differential);
      EXPECT_DOUBLE_EQ(plain.algebraic, logged.algebraic);
      EXPECT_EQ(plain.statistics.steps, logged.statistics.steps);
      EXPECT_EQ(plain.statistics.residual_evaluations,
                logged.statistics.residual_evaluations);
      EXPECT_EQ(plain.statistics.jacobian_setups,
                logged.statistics.jacobian_setups);
      EXPECT_EQ(plain.statistics.linear_solves,
                logged.statistics.linear_solves);
      EXPECT_EQ(plain.statistics.weight_evaluations,
                logged.statistics.weight_evaluations);
      EXPECT_EQ(plain.diagnostic_output.find("IDA nonlinear"),
                std::string::npos);
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          const std::regex    expression("field=all wrms=([^ \n]+)");
          std::vector<double> measured;
          for (auto it = std::sregex_iterator(logged.diagnostic_output.begin(),
                                              logged.diagnostic_output.end(),
                                              expression);
               it != std::sregex_iterator();
               ++it)
            measured.push_back(std::stod((*it)[1]));
          ASSERT_EQ(measured.size(), logged.expected_correction_norms.size());
          ASSERT_FALSE(measured.empty());
          for (unsigned int i = 0; i < measured.size(); ++i)
            EXPECT_NEAR(measured[i],
                        logged.expected_correction_norms[i],
                        1.e-13 *
                          std::max(1., logged.expected_correction_norms[i]));
          EXPECT_NE(logged.diagnostic_output.find(
                      "field=0 name=\"differential.value\" F_l2="),
                    std::string::npos);
          EXPECT_NE(logged.diagnostic_output.find(
                      "field=1 name=\"algebraic.value\" F_l2="),
                    std::string::npos);
          EXPECT_NE(logged.diagnostic_output.find(
                      "outcome=success backtracks="),
                    std::string::npos);
        }
    }
}

TEST(DistributedIDA,
     BOTH_NonlinearDiagnosticsReportRecoverableAndNonfiniteTrials)
{
  TimeIntervalParameters time;
  IDAParameters          parameters;
  parameters.print_nonlinear_diagnostics     = true;
  parameters.correction_type_at_initial_time = "use_y_diff";
  Adapter adapter(time, parameters, MPI_COMM_WORLD);
  bool    fail = false;
  add_forced_field(adapter, true, 1., [&]() {
    if (fail)
      throw dealii::RecoverableUserCallbackError("diagnostic trial marker");
  });
  auto y = adapter.make_state(), yp = adapter.make_state(),
       residual = adapter.make_state();
  y             = 0.;
  // Inject an invalid entry directly: Debug scalar assignment rejects NaN
  // before the residual callback can exercise the recoverable-error path.
  for (const auto index : y.locally_owned_elements())
    y[index] = std::numeric_limits<double>::quiet_NaN();
  y.compress(dealii::VectorOperation::insert);
  yp = 0.;
  std::ostringstream output, errors;
  struct RestoreStreams
  {
    std::streambuf *out, *err;
    ~RestoreStreams()
    {
      std::cout.rdbuf(out);
      std::cerr.rdbuf(err);
    }
  } restore{std::cout.rdbuf(output.rdbuf()), std::cerr.rdbuf(errors.rdbuf())};
  EXPECT_THROW(adapter.solver().residual(0., y, yp, residual),
               dealii::RecoverableUserCallbackError);
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      EXPECT_NE(output.str().find("reason=nonfinite residual"),
                std::string::npos);
      EXPECT_NE(output.str().find("nonfinite_y=1 nonfinite_F=1"),
                std::string::npos);
    }
  fail = true;
  y    = 1.;
  EXPECT_THROW(adapter.solve(y, yp), dealii::RecoverableUserCallbackError);
  EXPECT_NE(errors.str().find("diagnostic trial marker"), std::string::npos);
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      EXPECT_NE(output.str().find("outcome=failure backtracks="),
                std::string::npos);
    }
}

TEST(DistributedIDA, BOTH_MumpsBacksolveFailureRetriesLinearSolve)
{
  TimeIntervalParameters time;
  IDAParameters          parameters;
  time.final_time                            = .01;
  time.output_time_interval                  = .01;
  parameters.initial_step_size               = .01;
  parameters.maximum_step_size               = .01;
  parameters.absolute_tolerance              = 1.e-12;
  parameters.relative_tolerance              = 1.e-10;
  parameters.correction_type_at_initial_time = "none";
  unsigned int solve_calls                   = 0;
  Adapter      adapter(time,
                  parameters,
                  MPI_COMM_WORLD,
                  [&](const Adapter::Operator &op,
                      const GlobalVector      &rhs,
                      GlobalVector            &answer,
                      const double) {
                    ++solve_calls;
                    if (solve_calls == 1)
                      throw detail::MumpsBacksolveResidualError(
                        "injected residual failure");
                    GlobalVector ones(rhs), diagonal(rhs);
                    ones = 1.;
                    op.vmult(diagonal, ones);
                    for (unsigned int b = 0; b < answer.n_blocks(); ++b)
                      for (const auto index :
                           answer.block(b).locally_owned_elements())
                        answer.block(b)[index] =
                          rhs.block(b)[index] / diagonal.block(b)[index];
                    answer.compress(dealii::VectorOperation::insert);
                  });
  const auto   field                = add_forced_field(adapter, true, 1.);
  auto         y                    = adapter.make_state();
  auto         yp                   = adapter.make_state();
  adapter.field(y, field.fields())  = 2.;
  adapter.field(yp, field.fields()) = 1.;

  EXPECT_NO_THROW(adapter.solve(y, yp));
  EXPECT_GT(solve_calls, 1u);
  for (const auto index :
       adapter.field(y, field.fields()).locally_owned_elements())
    EXPECT_NEAR(adapter.field(y, field.fields())[index],
                2. + std::sin(time.final_time),
                1.e-8);
}

TEST(DistributedIDA, BOTH_InvalidLocalTolerancesAreCollective)
{
  TimeIntervalParameters time;
  IDAParameters          parameters;
  Adapter                adapter(time, parameters, MPI_COMM_WORLD);
  const auto             field = add_forced_field(adapter, true, 1.);
  EXPECT_THROW(adapter.set_tolerance_ramp_enabled(FieldId(1), false),
               dealii::ExceptionBase);
  adapter.set_local_tolerances(field.fields(), [](FieldVector &v) { v = 0.; });
  auto y = adapter.make_state(), yp = adapter.make_state();
  y  = 1.;
  yp = -1.;
  EXPECT_THROW(adapter.solve(y, yp), dealii::ExceptionBase);
  EXPECT_THROW(adapter.set_absolute_tolerance(field.fields(), 0.),
               dealii::ExceptionBase);
  EXPECT_THROW(adapter.set_tolerance_ramp_enabled(field.fields(), false),
               dealii::ExceptionBase);
}

TEST(DistributedIDA, BOTH_NVectorMinimumWithEmptyRanks)
{
  dealii::IndexSet owned(1);
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    owned.add_index(0);
  owned.compress();
  FieldVector values(owned, MPI_COMM_WORLD);
  values    = 2.;
  using IDA = Compatibility::SUNDIALS::IDA<FieldVector>;
  IDA  solver(typename IDA::AdditionalData{}, MPI_COMM_WORLD);
  auto view = Compatibility::SUNDIALS::internal::make_nvector_view(
    values
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
    ,
    solver.get_sun_context()
#  endif
  );
  EXPECT_DOUBLE_EQ(N_VMin(view), 2.);
}

TEST(DistributedIDA, BOTH_WeightCallbackFailureDuringInitialization)
{
  TimeIntervalParameters time;
  IDAParameters          parameters;
  parameters.correction_type_at_initial_time = "use_y_diff";
  Adapter    adapter(time, parameters, MPI_COMM_WORLD);
  const auto field = add_forced_field(adapter, true, 1.);
  adapter.set_absolute_tolerance(field.fields(), 1.e-8);
  auto y = adapter.make_state(), yp = adapter.make_state();
  y                                      = 1.;
  yp                                     = 0.;
  adapter.solver().compute_error_weights = [](const auto &, auto &) {
    throw std::runtime_error("weight callback failure");
  };
  EXPECT_THROW(adapter.solve(y, yp), std::runtime_error);
}

TEST(DistributedIDA, SharedNativeParameterSection)
{
  using namespace dealii;
  ParameterAcceptor::clear();
  auto &prm = ParameterAcceptor::prm;
  SUNDIALS::IDA<dealii::Vector<double>>::AdditionalData native;
  prm.enter_subsection("MetricFlowSystem<1, 3>");
  prm.enter_subsection("IDA parameters");
  native.add_parameters(prm);
  prm.leave_subsection();
  prm.leave_subsection();
  TimeIntervalParameters time("/", false);
  FixedStepParameters    step("/", false);
  IDAParameters          ida("/MetricFlowSystem<1, 3>/IDA parameters/", &time);
  ParameterAcceptor::declare_all_parameters(prm);
  EXPECT_DOUBLE_EQ(native.final_time, time.final_time);
  EXPECT_DOUBLE_EQ(native.output_period, time.output_time_interval);
  EXPECT_EQ(native.maximum_order, ida.maximum_order);
  EXPECT_DOUBLE_EQ(native.absolute_tolerance, ida.absolute_tolerance);
  EXPECT_EQ(native.ic_type, decltype(native)::none);
  prm.parse_input_from_string(R"(
    subsection MetricFlowSystem<1, 3>
      subsection IDA parameters
        set Initial time = 2.5
        set Final time = 2.75
        set Time interval between each output = 0.125
        subsection Running parameters
          set Initial step size = 0.002
          set Maximum order of BDF = 2
        end
        subsection Error control
          set Absolute error tolerance = 1e-7
          set Relative error tolerance = 2e-6
        end
        subsection Initial condition correction parameters
          set Correction type at initial time = use_y_diff
        end
      end
    end
  )");
  ParameterAcceptor::parse_all_parameters(prm);
  const auto data = ida.additional_data<GlobalVector>(time);
  EXPECT_DOUBLE_EQ(time.initial_time, 2.5);
  EXPECT_DOUBLE_EQ(data.initial_time, native.initial_time);
  EXPECT_DOUBLE_EQ(data.final_time, native.final_time);
  EXPECT_DOUBLE_EQ(data.final_time, 2.75);
  EXPECT_DOUBLE_EQ(data.output_period, native.output_period);
  EXPECT_DOUBLE_EQ(data.output_period, 0.125);
  EXPECT_DOUBLE_EQ(data.initial_step_size, native.initial_step_size);
  EXPECT_DOUBLE_EQ(data.initial_step_size, 0.002);
  EXPECT_EQ(data.maximum_order, 2u);
  EXPECT_EQ(data.maximum_order, native.maximum_order);
  EXPECT_DOUBLE_EQ(data.absolute_tolerance, 1.e-7);
  EXPECT_DOUBLE_EQ(data.absolute_tolerance, native.absolute_tolerance);
  EXPECT_DOUBLE_EQ(data.relative_tolerance, 2.e-6);
  EXPECT_EQ(data.ic_type, decltype(data)::use_y_diff);
  EXPECT_EQ(native.ic_type, decltype(native)::use_y_diff);
  std::ostringstream output;
  prm.print_parameters(output, ParameterHandler::PRM | ParameterHandler::Short);
  const auto text = output.str();
  EXPECT_EQ(text.find("subsection Time interval"), std::string::npos);
  EXPECT_EQ(text.find("subsection Fixed step"), std::string::npos);
  EXPECT_EQ(text.find("subsection IDA\n"), std::string::npos);
  const auto section = text.find("subsection IDA parameters");
  ASSERT_NE(section, std::string::npos);
  EXPECT_EQ(text.find("subsection IDA parameters", section + 1),
            std::string::npos);
  ParameterAcceptor::clear();
}

TEST(DistributedIDA, LinearSolverControlsFromParameterFile)
{
  using namespace dealii;
  ParameterAcceptor::clear();
  auto         &prm = ParameterAcceptor::prm;
  IDAParameters ida("/IDA/");
  ParameterAcceptor::declare_all_parameters(prm);
  const auto &c = ida.linear_solver_controls;
  EXPECT_EQ(c.krylov_maximum_iterations, 5000u);
  EXPECT_DOUBLE_EQ(c.krylov_relative_tolerance, 1.e-8);
  EXPECT_DOUBLE_EQ(c.schur_relative_tolerance, 1.e-9);
  EXPECT_DOUBLE_EQ(c.second_order_relative_tolerance, 1.e-2);
  EXPECT_DOUBLE_EQ(c.mumps_relative_tolerance, 1.e-8);
  prm.parse_input_from_string(R"(
    subsection IDA
      subsection Linear solver
        set Linear tolerance factor = 0.1
        subsection Krylov
          set Maximum iterations = 200
          set Relative tolerance = 1e-4
        end
        subsection Schur complement
          set Maximum iterations = 50
          set Relative tolerance = 1e-3
        end
        subsection Second-order block
          set Maximum iterations = 20
          set Relative tolerance = 0.1
        end
        subsection MUMPS
          set Relative tolerance = 1e-10
          set Maximum corrections = 5
        end
      end
    end
  )");
  ParameterAcceptor::parse_all_parameters(prm);
  EXPECT_DOUBLE_EQ(ida.linear_tolerance_factor, 0.1);
  EXPECT_EQ(c.krylov_maximum_iterations, 200u);
  EXPECT_DOUBLE_EQ(c.krylov_relative_tolerance, 1.e-4);
  EXPECT_EQ(c.schur_maximum_iterations, 50u);
  EXPECT_DOUBLE_EQ(c.schur_relative_tolerance, 1.e-3);
  EXPECT_EQ(c.second_order_maximum_iterations, 20u);
  EXPECT_DOUBLE_EQ(c.second_order_relative_tolerance, 0.1);
  EXPECT_DOUBLE_EQ(c.mumps_relative_tolerance, 1.e-10);
  EXPECT_EQ(c.mumps_maximum_corrections, 5u);

  prm.parse_input_from_string(R"(
    subsection IDA
      subsection Linear solver
        subsection Krylov
          set Relative tolerance = 0
        end
      end
    end
  )");
  EXPECT_THROW(ParameterAcceptor::parse_all_parameters(prm),
               dealii::ExceptionBase);
  ParameterAcceptor::clear();
}

TEST(DistributedIDA, RampReportsEndTimeBeyondShortenedRun)
{
  IDAToleranceRamp ramp;
  ramp.enabled  = true;
  ramp.end_time = 7.24;
  try
    {
      ramp.validate(7.2, 7.202, .002);
      FAIL() << "The ramp must not end after the shortened simulation.";
    }
  catch (const dealii::ExceptionBase &error)
    {
      std::string message = error.what();
      message.erase(std::remove_if(message.begin(),
                                   message.end(),
                                   [](const unsigned char c) {
                                     return std::isspace(c);
                                   }),
                    message.end());
      EXPECT_NE(message.find("Toleranceramp/Endtime=7.24"), std::string::npos);
      EXPECT_NE(message.find("Initialtime(7.2"), std::string::npos);
      EXPECT_NE(message.find("Holdduration(0)"), std::string::npos);
      EXPECT_NE(message.find("<Endtime<=Finaltime(7.202"), std::string::npos);
    }
}

TEST(DistributedIDA, RampParameterValidation)
{
  IDAToleranceRamp ramp;
  ramp.enabled  = true;
  ramp.end_time = .30;
  EXPECT_NO_THROW(ramp.validate(.25, .35, .01));
  ramp.end_time = .305;
  EXPECT_THROW(ramp.validate(.25, .35, .01), dealii::ExceptionBase);
  ramp.end_time                = .30;
  ramp.initial_relative_factor = std::numeric_limits<double>::infinity();
  EXPECT_THROW(ramp.validate(.25, .35, .01), dealii::ExceptionBase);
}
namespace
{
  auto
  add_mixed_forced_field(Adapter &adapter, const bool split_ownership)
  {
    return adapter.add(
      [=](auto &builder) {
        using Model = typename std::decay_t<decltype(builder)>::Model;
        dealii::IndexSet owned(4), differential(4);
        const auto       rank =
          dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        const auto size =
          dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
        for (unsigned int i = 0; i < 4; ++i)
          if (rank == (split_ownership && size > 1 ? i % 2 : 0))
            {
              owned.add_index(i);
              if (i == 0 || i == 2)
                differential.add_index(i);
            }
        owned.compress();
        differential.compress();
        const auto field    = builder.field("mixed", owned, {}, differential);
        const auto diagonal = [field](const bool mass) {
          return [field, mass](const auto &context) {
            const auto              &prototype = context.state(field);
            typename Model::Operator op;
            op.reinit_range_vector = [prototype](FieldVector &v, bool) {
              v.reinit(prototype);
            };
            op.reinit_domain_vector = op.reinit_range_vector;
            op.vmult = [mass](FieldVector &dst, const FieldVector &src) {
              for (const auto i : dst.locally_owned_elements())
                dst[i] = (!mass || i == 0 || i == 2) ? src[i] : 0.;
              dst.compress(dealii::VectorOperation::insert);
            };
            op.vmult_add = [mass](FieldVector &dst, const FieldVector &src) {
              for (const auto i : dst.locally_owned_elements())
                dst[i] = dst[i] + ((!mass || i == 0 || i == 2) ? src[i] : 0.);
              dst.compress(dealii::VectorOperation::insert);
            };
            op.Tvmult     = op.vmult;
            op.Tvmult_add = op.vmult_add;
            return op;
          };
        };
        builder.term(field, "forced")
          .residual([field](const auto &context) {
            const auto &y  = context.state(field);
            const auto &yp = context.derivative(field);
            FieldVector value;
            value.reinit(y);
            for (const auto i : y.locally_owned_elements())
              value[i] =
                y[i] - (2. + i + std::sin(context.time())) +
                ((i == 0 || i == 2) ? yp[i] - std::cos(context.time()) : 0.);
            value.compress(dealii::VectorOperation::insert);
            typename Model::Operation op;
            op.reinit_vector = [y](FieldVector &v, bool) { v.reinit(y); };
            op.apply         = [value](FieldVector &v) { v = value; };
            op.apply_add     = [value](FieldVector &v) { v += value; };
            return op;
          })
          .state(field, typename Model::OperatorFactory(diagonal(false)))
          .derivative(field, typename Model::OperatorFactory(diagonal(true)));
        return field;
      },
      "mixed");
  }

  void
  check_algebraic_ramp_weights(const bool protect,
                               const bool mixed_opt_out,
                               const bool split_ownership)
  {
    TimeIntervalParameters time;
    IDAParameters          parameters;
    time.final_time                                       = .04;
    time.output_time_interval                             = .01;
    parameters.initial_step_size                          = .001;
    parameters.maximum_step_size                          = .005;
    parameters.relative_tolerance                         = 1.e-6;
    parameters.absolute_tolerance                         = 1.e-7;
    parameters.correction_type_at_initial_time            = "none";
    parameters.correction_type_after_restart              = "none";
    parameters.tolerance_ramp.enabled                     = true;
    parameters.tolerance_ramp.protect_algebraic_variables = protect;
    parameters.tolerance_ramp.initial_relative_factor     = 100.;
    parameters.tolerance_ramp.initial_absolute_factor     = 10000.;
    parameters.tolerance_ramp.hold_duration               = .01;
    parameters.tolerance_ramp.end_time                    = .03;
    const auto solve_diagonal = [](const Adapter::Operator &op,
                                   const GlobalVector      &rhs,
                                   GlobalVector            &dst,
                                   double) {
      GlobalVector ones(rhs), diagonal(rhs);
      ones = 1.;
      op.vmult(diagonal, ones);
      for (unsigned int b = 0; b < dst.n_blocks(); ++b)
        for (const auto i : dst.block(b).locally_owned_elements())
          dst.block(b)[i] = rhs.block(b)[i] / diagonal.block(b)[i];
      dst.compress(dealii::VectorOperation::insert);
    };
    Adapter adapter(time, parameters, MPI_COMM_WORLD, solve_diagonal);
    // Mixed field follows a different-size field: flat global offsets are not
    // the numbering of the contributor's differential mask.
    const auto algebraic    = add_forced_field(adapter, false, 1.e3);
    const auto mixed        = add_mixed_forced_field(adapter, split_ownership);
    const auto differential = add_forced_field(adapter, true, 1.e-3);
    adapter.set_absolute_tolerance(algebraic.fields(), 1.e-4);
    adapter.set_absolute_tolerance(differential.fields(), 1.e-12);
    adapter.set_local_tolerances(mixed.fields(), [](FieldVector &v) {
      for (const auto i : v.locally_owned_elements())
        v[i] = (i + 1) * 1.e-7;
    });
    adapter.set_tolerance_ramp_enabled(mixed.fields(), !mixed_opt_out);
    adapter.set_tolerance_ramp_enabled(differential.fields(), false);
    adapter.set_stop_at_output_times(true);
    auto y = adapter.make_state(), yp = adapter.make_state();
    y  = 0.;
    yp = 0.;
    for (const auto i :
         adapter.field(y, algebraic.fields()).locally_owned_elements())
      adapter.field(y, algebraic.fields())[i] = 2000.;
    for (const auto i :
         adapter.field(y, mixed.fields()).locally_owned_elements())
      {
        adapter.field(y, mixed.fields())[i]  = 2. + i;
        adapter.field(yp, mixed.fields())[i] = (i == 0 || i == 2) ? 1. : 0.;
      }
    for (const auto i :
         adapter.field(y, differential.fields()).locally_owned_elements())
      {
        adapter.field(y, differential.fields())[i]  = .002;
        adapter.field(yp, differential.fields())[i] = .001;
      }
    y.compress(dealii::VectorOperation::insert);
    yp.compress(dealii::VectorOperation::insert);
    std::array<bool, 3> phases{{false, false, false}};
    const auto          check = [&](const auto  &state,
                           const auto  &weights,
                           const double t) {
      double rel = 100., abs = 10000.;
      if (t >= .03 - 1.e-14)
        {
          rel = abs = 1.;
          phases[2] = true;
        }
      else if (t > .01)
        {
          const double f = (.03 - t) / .02;
          rel            = std::pow(100., f);
          abs            = std::pow(10000., f);
          phases[1]      = true;
        }
      else
        phases[0] = true;
      const auto entry = [&](const auto   field,
                             const auto   i,
                             const double atol,
                             const bool   ramp) {
        const double denominator =
          1.e-6 * (ramp ? rel : 1.) * std::abs(adapter.field(state, field)[i]) +
          atol * (ramp ? abs : 1.);
        EXPECT_NEAR(adapter.field(weights, field)[i] * denominator, 1., 2.e-12);
      };
      for (const auto i :
           adapter.field(state, algebraic.fields()).locally_owned_elements())
        entry(algebraic.fields(), i, 1.e-4, !protect);
      for (const auto i :
           adapter.field(state, mixed.fields()).locally_owned_elements())
        entry(mixed.fields(),
              i,
              (i + 1) * 1.e-7,
              !mixed_opt_out && (!protect || i == 0 || i == 2));
      for (const auto i :
           adapter.field(state, differential.fields()).locally_owned_elements())
        entry(differential.fields(), i, 1.e-12, false);
    };
    auto         last_weights  = adapter.make_state();
    bool         valid_weights = false;
    unsigned int native_checks = 0, restarts = 0;
    std::function<void(const GlobalVector &, GlobalVector &)> native_weights;
    adapter.set_solver_should_restart([&](const double t, auto &, auto &) {
      if (restarts == 0 && t >= .02 - 1.e-14)
        {
          ++restarts;
          valid_weights = false;
          return true;
        }
      return false;
    });
    adapter.set_output_step(
      [&](const double, const auto &state, const auto &, unsigned int) {
        auto &solver = adapter.solver();
        // IDA may retain weights from the start of the step, rather than from
        // this output's state. Compare its native vector to the last callback.
        if (valid_weights)
          {
            auto actual = adapter.make_state();
            auto view   = Compatibility::SUNDIALS::internal::make_nvector_view(
              actual
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
              ,
              solver.get_sun_context()
#  endif
            );
            EXPECT_EQ(IDAGetErrWeights(solver.get_ida_memory(), view), 0);
            for (unsigned int b = 0; b < actual.n_blocks(); ++b)
              for (const auto i : actual.block(b).locally_owned_elements())
                EXPECT_DOUBLE_EQ(actual.block(b)[i], last_weights.block(b)[i]);
            ++native_checks;
          }
        // A nonzero arbitrary probe separates both terms of the weight formula.
        GlobalVector probe(state), weights(state);
        probe = 3.25;
        native_weights(probe, weights);
        double t;
        EXPECT_EQ(IDAGetCurrentTime(solver.get_ida_memory(), &t), 0);
        check(probe, weights, t);
      });
    auto &solver                 = adapter.solver();
    native_weights               = solver.compute_error_weights;
    solver.compute_error_weights = [&](const auto &state, auto &weights) {
      native_weights(state, weights);
      double t;
      EXPECT_EQ(IDAGetCurrentTime(solver.get_ida_memory(), &t), 0);
      check(state, weights, t);
      last_weights  = weights;
      valid_weights = true;
    };
    adapter.solve(y, yp);
    EXPECT_GT(native_checks, 0u);
    EXPECT_EQ(restarts, 1u);
    for (const auto reached : phases)
      EXPECT_TRUE(reached);
  }

  double
  nonlinear_algebraic_warmup(const bool protect)
  {
    TimeIntervalParameters time;
    IDAParameters          parameters;
    time.final_time                                       = .1;
    time.output_time_interval                             = .1;
    parameters.initial_step_size                          = .1;
    parameters.maximum_step_size                          = .1;
    parameters.maximum_order                              = 1;
    parameters.nonlinear_convergence_coefficient          = .33;
    parameters.relative_tolerance                         = 1.e-8;
    parameters.absolute_tolerance                         = 1.e-9;
    parameters.ignore_algebraic_terms_for_errors          = true;
    parameters.correction_type_at_initial_time            = "none";
    parameters.tolerance_ramp.enabled                     = true;
    parameters.tolerance_ramp.protect_algebraic_variables = protect;
    parameters.tolerance_ramp.initial_relative_factor     = 1.e8;
    parameters.tolerance_ramp.initial_absolute_factor     = 1.e12;
    parameters.tolerance_ramp.hold_duration               = .05;
    parameters.tolerance_ramp.end_time                    = .1;
    const auto solve = [](const Adapter::Operator &op,
                          const GlobalVector      &rhs,
                          GlobalVector            &dst,
                          double) {
      GlobalVector ones(rhs), values(rhs);
      ones = 1.;
      op.vmult(values, ones);
      for (unsigned int b = 0; b < dst.n_blocks(); ++b)
        if (dst.block(b).locally_owned_elements().is_element(0))
          {
            const double dx = rhs.block(b)[0] / values.block(b)[0];
            dst.block(b)[0] = dx;
            dst.block(b)[1] = rhs.block(b)[1] + (1. - values.block(b)[1]) * dx;
          }
      dst.compress(dealii::VectorOperation::insert);
    };
    Adapter    adapter(time, parameters, MPI_COMM_WORLD, solve);
    const auto mixed = adapter.add(
      [](auto &builder) {
        using Model = typename std::decay_t<decltype(builder)>::Model;
        dealii::IndexSet owned(2), differential(2);
        if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            owned.add_range(0, 2);
            differential.add_index(0);
          }
        owned.compress();
        differential.compress();
        const auto field   = builder.field("state", owned, {}, differential);
        const auto factory = [field](const bool mass) {
          return [field, mass](const auto &context) {
            const auto  &y = context.state(field);
            const double x =
              y.locally_owned_elements().is_element(0) ? y[0] : 0.;
            typename Model::Operator op;
            op.reinit_range_vector = [y](FieldVector &v, bool) { v.reinit(y); };
            op.reinit_domain_vector = op.reinit_range_vector;
            const auto action       = [mass, x](FieldVector       &dst,
                                          const FieldVector &src,
                                          bool               transpose,
                                          bool               add) {
              if (dst.locally_owned_elements().is_element(0))
                {
                  const double a =
                    mass ? src[0] :
                                 (transpose ? src[0] - 2. * x * src[1] : src[0]);
                  const double b =
                    mass ? 0. : (transpose ? src[1] : src[1] - 2. * x * src[0]);
                  dst[0] = (add ? dst[0] : 0.) + a;
                  dst[1] = (add ? dst[1] : 0.) + b;
                }
              dst.compress(dealii::VectorOperation::insert);
            };
            op.vmult = [action](auto &dst, const auto &src) {
              action(dst, src, false, false);
            };
            op.vmult_add = [action](auto &dst, const auto &src) {
              action(dst, src, false, true);
            };
            op.Tvmult = [action](auto &dst, const auto &src) {
              action(dst, src, true, false);
            };
            op.Tvmult_add = [action](auto &dst, const auto &src) {
              action(dst, src, true, true);
            };
            return op;
          };
        };
        builder.term(field, "nonlinear-constraint")
          .residual([field](const auto &context) {
            const auto &y  = context.state(field);
            const auto &yp = context.derivative(field);
            FieldVector r;
            r.reinit(y);
            if (y.locally_owned_elements().is_element(0))
              {
                r[0] = yp[0] + y[0];
                r[1] = y[1] - y[0] * y[0];
              }
            r.compress(dealii::VectorOperation::insert);
            typename Model::Operation op;
            op.reinit_vector = [y](FieldVector &v, bool) { v.reinit(y); };
            op.apply         = [r](FieldVector &v) { v = r; };
            op.apply_add     = [r](FieldVector &v) { v += r; };
            return op;
          })
          .state(field, typename Model::OperatorFactory(factory(false)))
          .derivative(field, typename Model::OperatorFactory(factory(true)));
        return field;
      },
      "nonlinear");
    auto y = adapter.make_state(), yp = adapter.make_state();
    y  = 1.;
    yp = 0.;
    if (adapter.field(yp, mixed.fields())
          .locally_owned_elements()
          .is_element(0))
      adapter.field(yp, mixed.fields())[0] = -1.;
    yp.compress(dealii::VectorOperation::insert);
    double       defect           = 0.;
    unsigned int positive_outputs = 0;
    adapter.set_stop_at_output_times(true);
    adapter.set_output_step(
      [&](const double t, const auto &state, const auto &, unsigned int) {
        if (t > 0.)
          {
            ++positive_outputs;
            const auto &v = adapter.field(state, mixed.fields());
            if (v.locally_owned_elements().is_element(0))
              {
                EXPECT_NEAR(v[0], 1. / 1.1, 1.e-12);
                defect = std::abs(v[1] - v[0] * v[0]);
              }
          }
      });
    adapter.solve(y, yp);
    EXPECT_EQ(positive_outputs, 1u);
    return dealii::Utilities::MPI::max(defect, MPI_COMM_WORLD);
  }
} // namespace

TEST(DistributedIDA, BOTH_AlgebraicRampProtectionUsesOwnedMixedMasks)
{
  for (const bool protect : {false, true})
    for (const bool mixed_opt_out : {false, true})
      for (const bool split_ownership : {false, true})
        check_algebraic_ramp_weights(protect, mixed_opt_out, split_ownership);
}

TEST(DistributedIDA, BOTH_AlgebraicRampProtectionEnforcesNonlinearConstraint)
{
  const double unprotected     = nonlinear_algebraic_warmup(false);
  const double protected_value = nonlinear_algebraic_warmup(true);
  // The first BE predictor is x_p=1-h=.9. One Newton correction
  // reaches x=10/11 but leaves z below x*x by (10/11-.9)^2=1/12100.
  EXPECT_NEAR(unprotected, 1. / 12100., 1.e-12);
  EXPECT_LT(protected_value, 1.e-10);
}

TEST(DistributedIDA, BOTH_AlgebraicRampProtectionValidatesOnlyAppliedScales)
{
  for (const bool differential : {false, true})
    {
      TimeIntervalParameters time;
      IDAParameters          parameters;
      time.final_time                                       = .01;
      time.output_time_interval                             = .01;
      parameters.initial_step_size                          = .001;
      parameters.maximum_step_size                          = .005;
      parameters.absolute_tolerance                         = 1.e307;
      parameters.ignore_algebraic_terms_for_errors          = false;
      parameters.correction_type_at_initial_time            = "none";
      parameters.tolerance_ramp.enabled                     = true;
      parameters.tolerance_ramp.protect_algebraic_variables = true;
      parameters.tolerance_ramp.initial_absolute_factor     = 1000.;
      parameters.tolerance_ramp.end_time                    = .01;
      Adapter adapter(time, parameters, MPI_COMM_WORLD);
      add_forced_field(adapter, differential, 1.);
      auto y = adapter.make_state(), yp = adapter.make_state();
      y  = 2.;
      yp = 0.;
      if (differential)
        {
          EXPECT_THROW(adapter.solve(y, yp), dealii::ExceptionBase);
        }
      else
        {
          auto        &solver         = adapter.solver();
          const auto   original_solve = solver.solve_with_jacobian;
          unsigned int exact_checks   = 0;
          solver.solve_with_jacobian =
            [&](const auto &rhs, auto &answer, const double tolerance) {
              // RMS(EWT) underflows for the valid protected atol=1e307.
              EXPECT_TRUE(std::isinf(tolerance));
              original_solve(rhs, answer, tolerance);
              auto defect = adapter.make_state();
              adapter.current_jacobian().vmult(defect, answer);
              defect -= rhs;
              for (const auto i : defect.locally_owned_elements())
                EXPECT_EQ(defect[i], 0.);
              ++exact_checks;
            };
          EXPECT_NO_THROW(adapter.solve(y, yp));
          EXPECT_GT(exact_checks, 0u);
        }
    }
}


namespace
{
  void
  check_underflowed_native_residual(const bool unbounded)
  {
    TimeIntervalParameters time;
    IDAParameters          parameters;
    time.final_time                            = .001;
    time.output_time_interval                  = .001;
    parameters.initial_step_size               = .001;
    parameters.maximum_step_size               = .001;
    parameters.absolute_tolerance              = unbounded ? 1. : 1e307;
    parameters.relative_tolerance              = 0.;
    parameters.correction_type_at_initial_time = "none";
    Adapter adapter(time, parameters, MPI_COMM_WORLD);
    if (unbounded)
      {
        auto controls = adapter.additional_data();
        controls.linear_tolerance_factor =
          std::numeric_limits<double>::infinity();
        adapter.set_additional_data(controls);
      }
    add_forced_field(adapter, false, 1e-307);
    auto y = adapter.make_state(), yp = adapter.make_state();
    y                           = 0.;
    yp                          = 0.;
    auto        &solver         = adapter.solver();
    const auto   original_solve = solver.solve_with_jacobian;
    unsigned int checks         = 0;
    unsigned int rejections     = 0;
    solver.solve_with_jacobian =
      [&](const auto &rhs, auto &answer, const double tolerance) {
        EXPECT_TRUE(std::isinf(tolerance));
        EXPECT_GT(rhs.linfty_norm(), 0.);
        EXPECT_EQ(rhs.l2_norm(), 0.);
        if (unbounded)
          {
            auto weights = adapter.make_state();
            auto view    = Compatibility::SUNDIALS::internal::make_nvector_view(
              weights
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
              ,
              solver.get_sun_context()
#  endif
            );
            EXPECT_EQ(IDAGetErrWeights(solver.get_ida_memory(), view),
                      IDA_SUCCESS);
            EXPECT_DOUBLE_EQ(weights.l2_norm(), 1.);
          }
        // The native zero initial correction passes a norm-only raw gate, but
        // its actual full-J residual has a nonzero owned entry.
        auto defect = adapter.make_state();
        adapter.current_jacobian().vmult(defect, answer);
        defect -= rhs;
        EXPECT_EQ(defect.l2_norm(), 0.);
        EXPECT_GT(defect.linfty_norm(), 0.);
        ++checks;
        try
          {
            original_solve(rhs, answer, tolerance);
          }
        catch (const dealii::ExceptionBase &)
          {
            ++rejections;
            throw;
          }
        ADD_FAILURE()
          << "A nonzero entry was accepted against a zero raw target.";
        AssertThrow(false,
                    dealii::ExcMessage("Unexpected native linear acceptance."));
      };
    try
      {
        adapter.solve(y, yp);
        FAIL() << "A nonzero residual entry cannot satisfy an exact raw gate.";
      }
    catch (const dealii::ExceptionBase &error)
      {
        ::testing::Test::RecordProperty("native_rejection", error.what());
      }
    EXPECT_GT(checks, 0u);
    EXPECT_EQ(rejections, checks);
  }
} // namespace
TEST(DistributedIDA, BOTH_IndeterminateNativeTargetRejectsUnderflowedResidual)
{
  check_underflowed_native_residual(false);
}
TEST(DistributedIDA, BOTH_InfiniteTargetWithZeroRawBoundRequiresExactResidual)
{
  check_underflowed_native_residual(true);
}
namespace
{
  void
  check_extreme_native_target(const bool unbounded)
  {
    TimeIntervalParameters time;
    IDAParameters          parameters;
    time.final_time                            = .001;
    time.output_time_interval                  = .001;
    parameters.initial_step_size               = .001;
    parameters.maximum_step_size               = .001;
    parameters.absolute_tolerance              = 1.;
    parameters.relative_tolerance              = 0.;
    parameters.correction_type_at_initial_time = "none";
    // A valid loose raw gate must not accept the initial zero correction when
    // native IDA requests a zero linear residual.
    parameters.linear_solver_controls.krylov_relative_tolerance =
      unbounded ? 1.e-8 : 1.;
    Adapter adapter(time, parameters, MPI_COMM_WORLD);
    auto    controls = adapter.additional_data();
    // IDASetEpsLin accepts both positive factors. Their native scalar is zero
    // or infinity while RMS(EWT) remains finite and equal to one.
    controls.linear_tolerance_factor =
      unbounded ? std::numeric_limits<double>::infinity() :
                  std::numeric_limits<double>::denorm_min();
    adapter.set_additional_data(controls);
    add_forced_field(adapter, false, 1.);
    auto y = adapter.make_state(), yp = adapter.make_state();
    y = yp                      = 0.;
    auto        &solver         = adapter.solver();
    const auto   original_solve = solver.solve_with_jacobian;
    unsigned int checks         = 0;
    solver.solve_with_jacobian =
      [&](const auto &rhs, auto &answer, const double tolerance) {
        if (unbounded)
          EXPECT_TRUE(std::isinf(tolerance));
        else
          EXPECT_EQ(tolerance, 0.);
        auto weights = adapter.make_state();
        auto view    = Compatibility::SUNDIALS::internal::make_nvector_view(
          weights
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
          ,
          solver.get_sun_context()
#  endif
        );
        EXPECT_EQ(IDAGetErrWeights(solver.get_ida_memory(), view), IDA_SUCCESS);
        EXPECT_DOUBLE_EQ(weights.l2_norm(), 1.);
        ++checks;
        original_solve(rhs, answer, tolerance);
        auto defect = adapter.make_state();
        adapter.current_jacobian().vmult(defect, answer);
        defect -= rhs;
        if (unbounded)
          {
            EXPECT_LE(
              defect.l2_norm(),
              parameters.linear_solver_controls.krylov_relative_tolerance *
                rhs.l2_norm());
            if (rhs.linfty_norm() > 0.)
              {
                EXPECT_GT(answer.linfty_norm(), 0.);
              }
          }
        else
          for (const auto i : defect.locally_owned_elements())
            EXPECT_EQ(defect[i], 0.);
      };
    EXPECT_NO_THROW(adapter.solve(y, yp));
    EXPECT_GT(checks, 0u);
    for (const auto i : y.locally_owned_elements())
      EXPECT_DOUBLE_EQ(y[i], 2. + std::sin(time.final_time));
  }
} // namespace
TEST(DistributedIDA, BOTH_ZeroNativeTargetRequiresExactCorrection)
{
  check_extreme_native_target(false);
}
TEST(DistributedIDA, BOTH_InfiniteNativeTargetPreservesRawGate)
{
  check_extreme_native_target(true);
}
#endif
