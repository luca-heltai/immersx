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

#include "time_residual.h"


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
    JacobianAction<VectorType>::from_matrix(mass).scaled(3.);
  const auto stiffness_action =
    JacobianAction<VectorType>::from_matrix(stiffness).scaled(2.);

  EXPECT_TRUE(mass_action.has_native_operator());
  EXPECT_TRUE(stiffness_action.has_native_operator());

  JacobianAccumulator<VectorType> accumulator;
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

  VectorType state(2);
  state[0] = 2.;
  state[1] = 3.;
  VectorType state_dot(2);
  state_dot[0] = 7.;
  state_dot[1] = 11.;

  TimeResidualContext<VectorType> context(0., state, state_dot);
  context.derivative_weight = 4.;

  TimeResidualModel<VectorType> model;
  model.add_term(
    time_residual_terms::mass,
    [](const auto &ctx, auto &residual) {
      residual[0] += ctx.state_derivative[0];
      residual[1] += ctx.state_derivative[1];
    },
    [](const auto &ctx) {
      const double factor = ctx.derivative_weight;
      return JacobianAction<VectorType>(
        [factor](VectorType &dst, const VectorType &src) {
          dst[0] = factor * src[0];
          dst[1] = 2. * factor * src[1];
        });
    });

  model.add_term(
    time_residual_terms::nonlinear,
    [](const auto &ctx, auto &residual) {
      residual[0] += ctx.state[0] * ctx.state[0];
      residual[1] += ctx.state[1] * ctx.state[1];
    },
    [](const auto &) {
      return JacobianAction<VectorType>(
        [](VectorType &dst, const VectorType &src) {
          dst[0] = 2. * src[0];
          dst[1] = 6. * src[1];
        });
    });

  VectorType residual(2);
  model.residual(context, residual);
  EXPECT_DOUBLE_EQ(residual[0], 11.);
  EXPECT_DOUBLE_EQ(residual[1], 20.);

  VectorType increment(2);
  increment[0] = 1.;
  increment[1] = 1.;
  VectorType jacobian_increment(2);
  model.jacobian_action(context).vmult(jacobian_increment, increment);
  EXPECT_DOUBLE_EQ(jacobian_increment[0], 6.);
  EXPECT_DOUBLE_EQ(jacobian_increment[1], 14.);

  context.selected_terms = TermSelection::only(time_residual_terms::mass);
  model.residual(context, residual);
  EXPECT_DOUBLE_EQ(residual[0], 7.);
  EXPECT_DOUBLE_EQ(residual[1], 11.);

  model.jacobian_action(context).vmult(jacobian_increment, increment);
  EXPECT_DOUBLE_EQ(jacobian_increment[0], 4.);
  EXPECT_DOUBLE_EQ(jacobian_increment[1], 8.);
}


TEST(TimeResidual, DifferentialAlgebraicMetadataProducesIdaMask) // NOLINT
{
  DifferentialAlgebraicMetadata metadata(5);
  metadata.add_block(0, 0, 2, StateVariableType::differential);
  metadata.add_block(1, 2, 5, StateVariableType::algebraic);

  EXPECT_EQ(metadata.type(0), StateVariableType::differential);
  EXPECT_EQ(metadata.type(1), StateVariableType::algebraic);

  const IndexSet differential = metadata.differential_components();
  const IndexSet algebraic    = metadata.algebraic_components();
  EXPECT_EQ(differential.size(), 5u);
  EXPECT_EQ(algebraic.size(), 5u);
  EXPECT_EQ(differential.n_elements(), 2u);
  EXPECT_EQ(algebraic.n_elements(), 3u);
  EXPECT_TRUE(differential.is_element(0));
  EXPECT_TRUE(differential.is_element(1));
  EXPECT_FALSE(differential.is_element(2));
  EXPECT_TRUE(algebraic.is_element(4));
}
