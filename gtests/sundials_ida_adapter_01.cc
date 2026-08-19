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

  // F(t,y,ydot) = M ydot + A y, with u and w differential and lambda
  // algebraic.  The two differential rows are coupled to the multiplier and
  // the last row is the algebraic constraint u-w=0.
  dealii::FullMatrix<double> mass(3, 3);
  mass(0, 0) = 1.;
  mass(1, 1) = 1.;

  dealii::FullMatrix<double> algebraic_operator(3, 3);
  algebraic_operator(0, 0) = 1.;
  algebraic_operator(0, 2) = 1.;
  algebraic_operator(1, 1) = 2.;
  algebraic_operator(1, 2) = -1.;
  algebraic_operator(2, 0) = 1.;
  algebraic_operator(2, 1) = -1.;

  TimeResidualModel<VectorType> model;
  model.add_term(
    time_residual_terms::mass,
    [](const auto &context, auto &residual) {
      residual[0] += context.state_derivative[0];
      residual[1] += context.state_derivative[1];
    },
    [&mass](const auto &context) {
      return JacobianAction<VectorType>::from_matrix(mass).scaled(
        context.derivative_weight);
    });
  model.add_term(
    time_residual_terms::diffusion,
    [&algebraic_operator](const auto &context, auto &residual) {
      algebraic_operator.vmult_add(residual, context.state);
    },
    [&algebraic_operator](const auto &context) {
      return JacobianAction<VectorType>::from_matrix(algebraic_operator)
        .scaled(context.state_weight);
    });

  DifferentialAlgebraicMetadata metadata(3);
  metadata.add_block(0, 0, 2, StateVariableType::differential);
  metadata.add_block(1, 2, 3, StateVariableType::algebraic);

  unsigned int                           linear_solves = 0;
  SundialsIDAResidualAdapter<VectorType> adapter(
    model,
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
