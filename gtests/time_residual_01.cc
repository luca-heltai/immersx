// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/index_set.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/core/time_residual.h>

using namespace ImmersX;


using dealii::FullMatrix;
using dealii::IndexSet;
using dealii::Vector;


TEST(TimeResidual, MatrixActionsAccumulate) // NOLINT
{
  using VectorType = Vector<double>;

  FullMatrix<double> mass(2, 2);
  mass(0, 0) = 2.;
  mass(1, 1) = 3.;

  FullMatrix<double> stiffness(2, 2);
  stiffness(0, 0) = 5.;
  stiffness(0, 1) = 1.;
  stiffness(1, 0) = -2.;
  stiffness(1, 1) = 4.;

  const auto mass_action =
    ImmersX::JacobianAction<VectorType>::from_matrix(mass).scaled(3.);
  const auto stiffness_action =
    ImmersX::JacobianAction<VectorType>::from_matrix(stiffness).scaled(2.);

  EXPECT_TRUE(mass_action.has_native_operator());
  EXPECT_TRUE(stiffness_action.has_native_operator());

  ImmersX::JacobianAccumulator<VectorType> accumulator;
  accumulator.add(mass_action);
  accumulator.add(stiffness_action);

  VectorType src(2);
  src[0] = 1.;
  src[1] = -1.;
  VectorType dst(2);
  accumulator.vmult(dst, src);

  EXPECT_DOUBLE_EQ(dst[0], 14.);
  EXPECT_DOUBLE_EQ(dst[1], -21.);

  VectorType operator_dst(2);
  const auto action = accumulator.action();
  action.as_linear_operator(src).vmult(operator_dst, src);
  EXPECT_DOUBLE_EQ(operator_dst[0], dst[0]);
  EXPECT_DOUBLE_EQ(operator_dst[1], dst[1]);
}


TEST(TimeResidual, TermSelectionControlsResidualAndJacobian) // NOLINT
{
  using VectorType = Vector<double>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor u_descriptor;
  u_descriptor.name          = "u";
  const auto               u = layout.add_field(u_descriptor);
  ImmersX::FieldDescriptor p_descriptor;
  p_descriptor.name = "p";
  const auto p      = layout.add_field(p_descriptor);

  VectorType u_state(1);
  u_state[0] = 2.;
  VectorType p_state(1);
  p_state[0] = 3.;
  VectorType u_state_dot(1);
  u_state_dot[0] = 7.;
  VectorType p_state_dot(1);
  p_state_dot[0] = 11.;

  ImmersX::StateView<VectorType> state_view(layout, 0.);
  state_view.bind(u, u_state).bind(p, p_state);
  ImmersX::StateView<VectorType> derivative_view(layout, 0.);
  derivative_view.bind(u, u_state_dot).bind(p, p_state_dot);
  const ImmersX::EvaluationContext<VectorType> context(0.,
                                                       state_view,
                                                       &derivative_view);

  ImmersX::SemiDiscreteModel<VectorType> model;
  model.add_term(
    ImmersX::time_residual_terms::mass,
    [u, p](const auto &ctx, auto &residual) {
      residual.field(u)[0] += ctx.state_derivative()->field(u, ctx.time())[0];
      residual.field(p)[0] += ctx.state_derivative()->field(p, ctx.time())[0];
    },
    [u, p](const auto &linearization, const auto &increment, auto &residual) {
      const double factor = linearization.derivative_weight();
      residual.field(u)[0] +=
        factor * increment.field(u, linearization.evaluation().time())[0];
      residual.field(p)[0] +=
        2. * factor * increment.field(p, linearization.evaluation().time())[0];
    });

  model.add_term(
    ImmersX::time_residual_terms::nonlinear,
    [u, p](const auto &ctx, auto &residual) {
      const auto &values = ctx.state();
      residual.field(u)[0] +=
        values.field(u, ctx.time())[0] * values.field(u, ctx.time())[0];
      residual.field(p)[0] +=
        values.field(p, ctx.time())[0] * values.field(p, ctx.time())[0];
    },
    [u, p](const auto &linearization, const auto &increment, auto &residual) {
      const auto &values = linearization.evaluation().state();
      residual.field(u)[0] +=
        2. * values.field(u, linearization.evaluation().time())[0] *
        increment.field(u, linearization.evaluation().time())[0];
      residual.field(p)[0] +=
        2. * values.field(p, linearization.evaluation().time())[0] *
        increment.field(p, linearization.evaluation().time())[0];
    });

  VectorType                               u_residual(1);
  VectorType                               p_residual(1);
  ImmersX::ResidualAccumulator<VectorType> residual(layout);
  residual.bind(u, u_residual).bind(p, p_residual);
  u_residual = 0.;
  p_residual = 0.;
  model.evaluate(context, residual);
  EXPECT_DOUBLE_EQ(u_residual[0], 11.);
  EXPECT_DOUBLE_EQ(p_residual[0], 20.);

  VectorType increment_u(1);
  VectorType increment_p(1);
  increment_u[0] = 1.;
  increment_p[0] = 1.;
  ImmersX::StateView<VectorType> increment(layout, 0.);
  increment.bind(u, increment_u).bind(p, increment_p);
  u_residual = 0.;
  p_residual = 0.;
  ImmersX::LinearizationContext<VectorType> linearization(context, 1., 4.);
  model.add_jacobian_action(linearization, increment, residual);
  EXPECT_DOUBLE_EQ(u_residual[0], 8.);
  EXPECT_DOUBLE_EQ(p_residual[0], 14.);

  const ImmersX::EvaluationContext<VectorType> mass_context(
    0.,
    state_view,
    &derivative_view,
    ImmersX::TermSelection::only(ImmersX::time_residual_terms::mass));
  u_residual = 0.;
  p_residual = 0.;
  model.evaluate(mass_context, residual);
  EXPECT_DOUBLE_EQ(u_residual[0], 7.);
  EXPECT_DOUBLE_EQ(p_residual[0], 11.);

  u_residual = 0.;
  p_residual = 0.;
  ImmersX::LinearizationContext<VectorType> mass_linearization(mass_context,
                                                               1.,
                                                               4.);
  model.add_jacobian_action(mass_linearization, increment, residual);
  EXPECT_DOUBLE_EQ(u_residual[0], 4.);
  EXPECT_DOUBLE_EQ(p_residual[0], 8.);
}
