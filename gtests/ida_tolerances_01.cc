// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on deal.II.
// ---------------------------------------------------------------------
#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <cmath>
#include <limits>
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
  add_forced_field(AdapterType &adapter,
                   const bool   differential,
                   const double scale)
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
  };

  RunResult
  integrate(const bool   local,
            const double factor,
            const bool   output,
            const bool   restart       = false,
            const int    participation = 0,
            const bool   check_weights = false)
  {
    TimeIntervalParameters time;
    IDAParameters          parameters;
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
            builds};
  }
} // namespace

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
#endif
