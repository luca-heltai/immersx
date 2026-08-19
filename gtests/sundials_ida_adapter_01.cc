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

#include "sundials_ida_adapter.h"


#ifdef DEAL_II_WITH_SUNDIALS
#  include <deal.II/lac/full_matrix.h>
#  include <deal.II/lac/precondition.h>
#  include <deal.II/lac/solver_control.h>
#  include <deal.II/lac/solver_gmres.h>
#  include <deal.II/lac/vector.h>

#  include <algorithm>
#  include <cmath>


TEST(SundialsIDA, SolvesSyntheticDifferentialAlgebraicResidual) // NOLINT
{
  using VectorType = dealii::Vector<double>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor u_descriptor;
  u_descriptor.name          = "u";
  u_descriptor.time_role     = ImmersX::TimeRole::differential;
  const auto               u = layout.add_field(u_descriptor);
  ImmersX::FieldDescriptor w_descriptor;
  w_descriptor.name          = "w";
  w_descriptor.time_role     = ImmersX::TimeRole::differential;
  const auto               w = layout.add_field(w_descriptor);
  ImmersX::FieldDescriptor lambda_descriptor;
  lambda_descriptor.name      = "lambda";
  lambda_descriptor.time_role = ImmersX::TimeRole::algebraic;
  const auto lambda           = layout.add_field(lambda_descriptor);

  ImmersX::MonolithicFieldLayout<VectorType> field_layout(layout, 3);
  field_layout.add_field(u, 0, 1);
  field_layout.add_field(w, 1, 2);
  field_layout.add_field(lambda, 2, 3);

  ImmersX::TimeResidualModel<VectorType> model;
  model.add_term(
    time_residual_terms::mass,
    [u, w](const auto &context, auto &residual) {
      residual.field(u)[0] +=
        context.state_derivative()->field(u, context.time())[0];
      residual.field(w)[0] +=
        context.state_derivative()->field(w, context.time())[0];
    },
    [u, w](const auto &linearization, const auto &increment, auto &residual) {
      residual.field(u)[0] +=
        linearization.derivative_weight() *
        increment.field(u, linearization.evaluation().time())[0];
      residual.field(w)[0] +=
        linearization.derivative_weight() *
        increment.field(w, linearization.evaluation().time())[0];
    });
  model.add_term(
    time_residual_terms::diffusion,
    [u, w, lambda](const auto &context, auto &residual) {
      const auto  &state        = context.state();
      const double u_value      = state.field(u, context.time())[0];
      const double w_value      = state.field(w, context.time())[0];
      const double lambda_value = state.field(lambda, context.time())[0];
      residual.field(u)[0] += u_value + lambda_value;
      residual.field(w)[0] += 2. * w_value - lambda_value;
      residual.field(lambda)[0] += u_value - w_value;
    },
    [u, w, lambda](const auto &linearization,
                   const auto &increment,
                   auto       &residual) {
      const double time    = linearization.evaluation().time();
      const double factor  = linearization.state_weight();
      const double du      = increment.field(u, time)[0];
      const double dw      = increment.field(w, time)[0];
      const double dlambda = increment.field(lambda, time)[0];
      residual.field(u)[0] += factor * (du + dlambda);
      residual.field(w)[0] += factor * (2. * dw - dlambda);
      residual.field(lambda)[0] += factor * (du - dw);
    });

  ImmersX::DifferentialAlgebraicMetadata metadata(layout, 3);
  metadata.add_field(u, 0, 1);
  metadata.add_field(w, 1, 2);
  metadata.add_field(lambda, 2, 3);

  unsigned int                           linear_solves = 0;
  SundialsIDAResidualAdapter<VectorType> adapter(
    model,
    field_layout,
    metadata,
    [](VectorType &vector) { vector.reinit(3); },
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
  data.final_time                    = 0.1;
  data.initial_step_size             = 0.01;
  data.output_period                 = 0.05;
  data.absolute_tolerance            = 1.e-10;
  data.relative_tolerance            = 1.e-10;
  data.maximum_non_linear_iterations = 20;
  data.ic_type    = dealii::SUNDIALS::IDA<VectorType>::AdditionalData::none;
  data.reset_type = dealii::SUNDIALS::IDA<VectorType>::AdditionalData::none;

  dealii::SUNDIALS::IDA<VectorType> ida(data);
  adapter.connect(ida);

  VectorType state(3);
  state[0] = 1.;
  state[1] = 1.;
  state[2] = 0.5;
  VectorType state_derivative(3);
  state_derivative[0] = -1.5;
  state_derivative[1] = -1.5;
  state_derivative[2] = 0.;

  const unsigned int steps = ida.solve_dae(state, state_derivative);
  const double       exact = std::exp(-1.5 * data.final_time);

  EXPECT_GT(steps, 0u);
  EXPECT_GT(linear_solves, 0u);
  EXPECT_NEAR(state[0], exact, 1.e-7);
  EXPECT_NEAR(state[1], exact, 1.e-7);
  EXPECT_NEAR(state[2], 0.5 * exact, 1.e-7);
  EXPECT_NEAR(state_derivative[0], -1.5 * exact, 1.e-7);
  EXPECT_NEAR(state_derivative[1], -1.5 * exact, 1.e-7);
  EXPECT_NEAR(state_derivative[2], -0.75 * exact, 2.e-6);
}

#else

TEST(SundialsIDA, AdapterIsConditionallyUnavailable) // NOLINT
{
  GTEST_SKIP() << "deal.II was configured without SUNDIALS support.";
}

#endif
