// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <gtest/gtest.h>

#include "native_field_layout.h"
#include "state_history.h"
#include "sundials_ida_adapter.h"


#ifdef DEAL_II_WITH_SUNDIALS
#  include <deal.II/lac/block_vector.h>
#  include <deal.II/lac/full_matrix.h>
#  include <deal.II/lac/precondition.h>
#  include <deal.II/lac/solver_control.h>
#  include <deal.II/lac/solver_gmres.h>
#  include <deal.II/lac/vector.h>

#  include <algorithm>
#  include <cmath>
#endif


#ifdef DEAL_II_WITH_SUNDIALS
namespace
{
  using VectorType = dealii::Vector<double>;

  struct NativeStokesOperator
  {
    dealii::FullMatrix<double> uu{1, 1};
    dealii::FullMatrix<double> up{1, 1};
    dealii::FullMatrix<double> pu{1, 1};
    dealii::FullMatrix<double> pp{1, 1};

    void
    vmult(dealii::BlockVector<double>       &dst,
          const dealii::BlockVector<double> &src) const
    {
      dst.reinit(2);
      dst.block(0).reinit(1);
      dst.block(1).reinit(1);
      dst.collect_sizes();
      uu.vmult(dst.block(0), src.block(0));
      up.vmult_add(dst.block(0), src.block(1));
      pu.vmult(dst.block(1), src.block(0));
      pp.vmult_add(dst.block(1), src.block(1));
    }
  };

  struct MixedModel
  {
    ImmersX::FieldId           velocity;
    ImmersX::FieldId           pressure;
    ImmersX::NativeFieldLayout native_layout;
    NativeStokesOperator       native_operator;

    MixedModel(const ImmersX::StateLayout &layout,
               const ImmersX::FieldId      velocity,
               const ImmersX::FieldId      pressure)
      : velocity(velocity)
      , pressure(pressure)
      , native_layout(layout)
    {
      native_layout.add_block(velocity);
      native_layout.add_block(pressure);
      native_operator.uu(0, 0) = 2.;
      native_operator.up(0, 0) = 3.;
      native_operator.pu(0, 0) = 5.;
      native_operator.pp(0, 0) = 1.;
    }

    void
    add_residual(const ImmersX::EvaluationContext<VectorType> &context,
                 ImmersX::ResidualAccumulator<VectorType>     &residual) const
    {
      dealii::BlockVector<double> state;
      dealii::BlockVector<double> contribution;
      native_layout.gather(context.state(), context.time(), state);
      native_operator.vmult(contribution, state);
      native_layout.scatter_add(contribution, residual);
    }

    void
    add_jacobian(const ImmersX::LinearizationContext<VectorType> &linearization,
                 const ImmersX::StateAccessor<VectorType>        &increment,
                 ImmersX::ResidualAccumulator<VectorType> &residual) const
    {
      dealii::BlockVector<double> native_increment;
      dealii::BlockVector<double> contribution;
      native_layout.gather(increment,
                           linearization.evaluation().time(),
                           native_increment);
      native_operator.vmult(contribution, native_increment);
      native_layout.scatter_add(contribution, residual);
    }
  };

  ImmersX::FieldDescriptor
  descriptor(const std::string &name, const ImmersX::TimeRole role)
  {
    ImmersX::FieldDescriptor result;
    result.name      = name;
    result.time_role = role;
    return result;
  }

  void
  bind_all(ImmersX::StateView<VectorType>      &view,
           const std::vector<VectorType>       &values,
           const std::vector<ImmersX::FieldId> &fields)
  {
    for (const auto field : fields)
      view.bind(field, values[field.value()]);
  }
} // namespace


TEST(MixedDAE, SemanticNativeBlockAndIDASeams) // NOLINT
{
  ImmersX::StateLayout layout;
  const auto           velocity =
    layout.add_field(descriptor("velocity", ImmersX::TimeRole::differential));
  const auto pressure =
    layout.add_field(descriptor("pressure", ImmersX::TimeRole::algebraic));
  const auto network =
    layout.add_field(descriptor("network", ImmersX::TimeRole::differential));
  const auto lambda =
    layout.add_field(descriptor("lambda", ImmersX::TimeRole::algebraic));
  const std::vector<ImmersX::FieldId> fields = {velocity,
                                                pressure,
                                                network,
                                                lambda};

  ImmersX::detail::MonolithicFieldLayout<VectorType> monolithic_layout(layout,
                                                                       4);
  monolithic_layout.add_field(velocity, 0, 1);
  monolithic_layout.add_field(pressure, 1, 2);
  monolithic_layout.add_field(network, 2, 3);
  monolithic_layout.add_field(lambda, 3, 4);

  MixedModel problem(layout, velocity, pressure);

  VectorType velocity_state(1);
  VectorType pressure_state(1);
  VectorType network_state(1);
  VectorType lambda_state(1);
  VectorType velocity_dot(1);
  VectorType pressure_dot(1);
  VectorType network_dot(1);
  VectorType lambda_dot(1);
  velocity_state[0] = 1.;
  pressure_state[0] = -5.;
  network_state[0]  = 1.;
  lambda_state[0]   = 7.5;
  velocity_dot[0]   = 5.5;
  pressure_dot[0]   = 0.;
  network_dot[0]    = 5.5;
  lambda_dot[0]     = 0.;

  std::vector<VectorType> state_values      = {velocity_state,
                                               pressure_state,
                                               network_state,
                                               lambda_state};
  std::vector<VectorType> derivative_values = {velocity_dot,
                                               pressure_dot,
                                               network_dot,
                                               lambda_dot};

  ImmersX::StateView<VectorType> state(layout, 0.);
  ImmersX::StateView<VectorType> state_derivative(layout, 0.);
  bind_all(state, state_values, fields);
  bind_all(state_derivative, derivative_values, fields);

  ImmersX::SemiDiscreteModel<VectorType> model;
  model.add_term(
    ImmersX::time_residual_terms::mass,
    [velocity, network](const auto &context, auto &residual) {
      residual.field(velocity)[0] +=
        context.state_derivative()->field(velocity, context.time())[0];
      residual.field(network)[0] +=
        context.state_derivative()->field(network, context.time())[0];
    },
    [velocity, network](const auto &linearization,
                        const auto &increment,
                        auto       &residual) {
      const double factor = linearization.derivative_weight();
      const double time   = linearization.evaluation().time();
      residual.field(velocity)[0] +=
        factor * increment.field(velocity, time)[0];
      residual.field(network)[0] += factor * increment.field(network, time)[0];
    });

  model.add_term(
    "native_problem",
    [&problem](const auto &context, auto &residual) {
      problem.add_residual(context, residual);
    },
    [&problem](const auto &linearization,
               const auto &increment,
               auto       &residual) {
      problem.add_jacobian(linearization, increment, residual);
    });

  model.add_term(
    "network_operator",
    [network, lambda](const auto &context, auto &residual) {
      const auto  &state = context.state();
      const double time  = context.time();
      residual.field(network)[0] +=
        2. * state.field(network, time)[0] - state.field(lambda, time)[0];
    },
    [network,
     lambda](const auto &linearization, const auto &increment, auto &residual) {
      const double time   = linearization.evaluation().time();
      const double factor = linearization.state_weight();
      residual.field(network)[0] +=
        factor * (2. * increment.field(network, time)[0] -
                  increment.field(lambda, time)[0]);
    });

  // This interaction owns lambda and couples the velocity constraint, but
  // never contributes to the pressure row.
  model.add_term(
    "velocity_interaction",
    [velocity, network, lambda](const auto &context, auto &residual) {
      const auto  &state = context.state();
      const double time  = context.time();
      residual.field(velocity)[0] += state.field(lambda, time)[0];
      residual.field(lambda)[0] +=
        state.field(velocity, time)[0] - state.field(network, time)[0];
    },
    [velocity, network, lambda](const auto &linearization,
                                const auto &increment,
                                auto       &residual) {
      const double time   = linearization.evaluation().time();
      const double factor = linearization.state_weight();
      residual.field(velocity)[0] += factor * increment.field(lambda, time)[0];
      residual.field(lambda)[0] +=
        factor * (increment.field(velocity, time)[0] -
                  increment.field(network, time)[0]);
    });

  const ImmersX::EvaluationContext<VectorType> context(0.,
                                                       state,
                                                       &state_derivative);
  std::vector<VectorType>                  residual_values(4, VectorType(1));
  ImmersX::ResidualAccumulator<VectorType> residual(layout);
  for (const auto field : fields)
    {
      residual_values[field.value()] = VectorType(1);
      residual_values[field.value()] = 0.;
      residual.bind(field, residual_values[field.value()]);
    }
  model.evaluate(context, residual);
  for (const auto field : fields)
    EXPECT_NEAR(residual_values[field.value()][0], 0., 1.e-14);

  // The same model evaluates an interpolated history state without changing
  // any contributor or introducing a monolithic state into the model.
  ImmersX::StateHistory<VectorType> velocity_history;
  ImmersX::StateHistory<VectorType> pressure_history;
  ImmersX::StateHistory<VectorType> network_history;
  ImmersX::StateHistory<VectorType> lambda_history;
  VectorType                        velocity_end(1);
  VectorType                        pressure_end(1);
  VectorType                        network_end(1);
  VectorType                        lambda_end(1);
  velocity_end[0] = 2.;
  pressure_end[0] = -10.;
  network_end[0]  = 2.;
  lambda_end[0]   = 15.;
  VectorType velocity_dot_end(1);
  VectorType network_dot_end(1);
  velocity_dot_end[0] = 11.;
  network_dot_end[0]  = 11.;
  velocity_history.accept(0., velocity_state);
  velocity_history.accept(1., velocity_end);
  pressure_history.accept(0., pressure_state);
  pressure_history.accept(1., pressure_end);
  network_history.accept(0., network_state);
  network_history.accept(1., network_end);
  lambda_history.accept(0., lambda_state);
  lambda_history.accept(1., lambda_end);
  std::vector<VectorType> history_values      = {velocity_history.at(0.5),
                                                 pressure_history.at(0.5),
                                                 network_history.at(0.5),
                                                 lambda_history.at(0.5)};
  std::vector<VectorType> history_derivatives = {VectorType(1),
                                                 VectorType(1),
                                                 VectorType(1),
                                                 VectorType(1)};
  history_derivatives[velocity.value()][0] =
    0.5 * (velocity_dot[0] + velocity_dot_end[0]);
  history_derivatives[network.value()][0] =
    0.5 * (network_dot[0] + network_dot_end[0]);
  ImmersX::StateView<VectorType> history_state(layout, 0.5);
  ImmersX::StateView<VectorType> history_state_derivative(layout, 0.5);
  bind_all(history_state, history_values, fields);
  bind_all(history_state_derivative, history_derivatives, fields);
  const ImmersX::EvaluationContext<VectorType> history_context(
    0.5, history_state, &history_state_derivative);
  for (auto &values : residual_values)
    values = 0.;
  model.evaluate(history_context, residual);
  for (const auto field : fields)
    EXPECT_NEAR(residual_values[field.value()][0], 0., 1.e-14);

  ImmersX::DifferentialAlgebraicMetadata metadata(layout, 4);
  metadata.add_field(velocity, 0, 1);
  metadata.add_field(pressure, 1, 2);
  metadata.add_field(network, 2, 3);
  metadata.add_field(lambda, 3, 4);

  unsigned int                           linear_solves = 0;
  SundialsIDAResidualAdapter<VectorType> adapter(
    model,
    monolithic_layout,
    metadata,
    [](VectorType &vector) { vector.reinit(4); },
    [&linear_solves](
      const auto &action, const auto &rhs, auto &dst, const double tolerance) {
      ++linear_solves;
      dealii::SolverControl           control(100, std::max(1.e-12, tolerance));
      dealii::SolverGMRES<VectorType> solver(control);
      dst = 0.;
      solver.solve(action.as_linear_operator(rhs),
                   dst,
                   rhs,
                   dealii::PreconditionIdentity());
    });

  dealii::SUNDIALS::IDA<VectorType>::AdditionalData data;
  data.initial_time                  = 0.;
  data.final_time                    = 0.02;
  data.initial_step_size             = 0.005;
  data.output_period                 = 0.02;
  data.absolute_tolerance            = 1.e-10;
  data.relative_tolerance            = 1.e-10;
  data.maximum_non_linear_iterations = 20;
  data.ic_type    = dealii::SUNDIALS::IDA<VectorType>::AdditionalData::none;
  data.reset_type = dealii::SUNDIALS::IDA<VectorType>::AdditionalData::none;

  dealii::SUNDIALS::IDA<VectorType> ida(data);
  adapter.connect(ida);

  VectorType monolithic_state(4);
  monolithic_state[0] = velocity_state[0];
  monolithic_state[1] = pressure_state[0];
  monolithic_state[2] = network_state[0];
  monolithic_state[3] = lambda_state[0];
  VectorType monolithic_derivative(4);
  monolithic_derivative[0] = velocity_dot[0];
  monolithic_derivative[1] = pressure_dot[0];
  monolithic_derivative[2] = network_dot[0];
  monolithic_derivative[3] = lambda_dot[0];

  ida.setup_jacobian(0., monolithic_state, monolithic_derivative, 2.);
  VectorType increment(4);
  increment[0] = 1.;
  increment[1] = 2.;
  increment[2] = 3.;
  increment[3] = 4.;
  VectorType jacobian_increment(4);
  adapter.current_jacobian_action()
    .as_linear_operator(monolithic_state)
    .vmult(jacobian_increment, increment);
  EXPECT_DOUBLE_EQ(jacobian_increment[0], 14.);
  EXPECT_DOUBLE_EQ(jacobian_increment[1], 7.);
  EXPECT_DOUBLE_EQ(jacobian_increment[2], 8.);
  EXPECT_DOUBLE_EQ(jacobian_increment[3], -2.);

  const unsigned int steps =
    ida.solve_dae(monolithic_state, monolithic_derivative);
  EXPECT_GT(steps, 0u);
  EXPECT_GT(linear_solves, 0u);
  EXPECT_NEAR(monolithic_state[0], monolithic_state[2], 1.e-8);
  EXPECT_NEAR(monolithic_state[1], -5. * monolithic_state[0], 1.e-8);
  EXPECT_NEAR(monolithic_state[3], 7.5 * monolithic_state[0], 1.e-8);
}

#else

TEST(MixedDAE, SemanticNativeBlockAndIDASeams) // NOLINT
{
  GTEST_SKIP() << "deal.II was configured without SUNDIALS support.";
}

#endif
