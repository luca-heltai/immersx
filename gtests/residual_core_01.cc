// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception (the "License"); either version 3.0 of the
// License, or (at your option) any later version. The full text of the
// license can be found in the LICENSE.md file at the top level of the
// ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/index_set.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>

#include <string>

#include "poisson_residual.h"
#include "utils.h"

using dealii::FullMatrix;
using dealii::ParameterAcceptor;
using dealii::Vector;

namespace
{
  struct TwoFieldProblem
  {
    ImmersX::FieldId   u;
    ImmersX::FieldId   p;
    FullMatrix<double> native_matrix;
    Vector<double>     native_rhs;

    void
    add_residual(const ImmersX::EvaluationContext<Vector<double>> &context,
                 ImmersX::ResidualAccumulator<Vector<double>> &residual) const
    {
      Vector<double> native_state(2);
      native_state[0] = context.state().field(u, context.time())[0];
      native_state[1] = context.state().field(p, context.time())[0];

      Vector<double> native_residual(2);
      native_matrix.vmult(native_residual, native_state);
      native_residual -= native_rhs;

      residual.field(u)[0] += native_residual[0];
      residual.field(p)[0] += native_residual[1];
    }
  };

  ImmersX::FieldDescriptor
  descriptor(const std::string &name, const ImmersX::TimeRole time_role)
  {
    ImmersX::FieldDescriptor result;
    result.name      = name;
    result.time_role = time_role;
    return result;
  }
} // namespace


TEST(ResidualCore, StateLayoutAndEvaluationContext)
{
  ImmersX::StateLayout layout;
  const auto           u =
    layout.add_field(descriptor("u", ImmersX::TimeRole::differential));
  const auto p =
    layout.add_field(descriptor("p", ImmersX::TimeRole::algebraic));
  const auto lambda =
    layout.add_field(descriptor("lambda", ImmersX::TimeRole::algebraic));

  EXPECT_EQ(layout.n_fields(), 3u);
  EXPECT_EQ(layout.field_id("u"), u);
  EXPECT_EQ(layout.field(p).time_role, ImmersX::TimeRole::algebraic);
  EXPECT_EQ(layout.field(lambda).name, "lambda");

  Vector<double> u_values(1);
  Vector<double> p_values(1);
  Vector<double> lambda_values(1);
  u_values[0]      = 1.25;
  p_values[0]      = -2.5;
  lambda_values[0] = 4.;

  ImmersX::StateView<Vector<double>> state(layout, 3.5);
  state.bind(u, u_values).bind(p, p_values).bind(lambda, lambda_values);

  ImmersX::TermSelection terms;
  terms.set("mass", ImmersX::TermTreatment::implicit_term);
  ImmersX::EvaluationContext<Vector<double>> context(3.5,
                                                     state,
                                                     nullptr,
                                                     terms);

  EXPECT_DOUBLE_EQ(context.time(), 3.5);
  EXPECT_DOUBLE_EQ(context.state().field(u, context.time())[0], 1.25);
  EXPECT_DOUBLE_EQ(context.state().field(p, context.time())[0], -2.5);
  EXPECT_FALSE(context.has_state_derivative());
  EXPECT_EQ(context.terms().treatment("mass"),
            ImmersX::TermTreatment::implicit_term);
  EXPECT_TRUE(
    context.terms().includes("mass", ImmersX::TermTreatment::implicit_term));
  EXPECT_EQ(context.terms().treatment("transport"),
            ImmersX::TermTreatment::all);
}


TEST(ResidualCore, OneProblemContributesToTwoCrossCoupledRows)
{
  ImmersX::StateLayout layout;
  TwoFieldProblem      problem;
  problem.u =
    layout.add_field(descriptor("u", ImmersX::TimeRole::differential));
  problem.p = layout.add_field(descriptor("p", ImmersX::TimeRole::algebraic));

  problem.native_matrix.reinit(2, 2);
  problem.native_matrix(0, 0) = 2.;
  problem.native_matrix(0, 1) = 3.;
  problem.native_matrix(1, 0) = 5.;
  problem.native_matrix(1, 1) = 7.;
  problem.native_rhs.reinit(2);
  problem.native_rhs[0] = 1.;
  problem.native_rhs[1] = 4.;

  Vector<double> u_values(1);
  Vector<double> p_values(1);
  u_values[0] = 1.;
  p_values[0] = 2.;

  ImmersX::StateView<Vector<double>> state(layout, 0.);
  state.bind(problem.u, u_values).bind(problem.p, p_values);
  ImmersX::EvaluationContext<Vector<double>> context(0., state);

  Vector<double> u_residual(1);
  Vector<double> p_residual(1);
  u_residual = 0.;
  p_residual = 0.;
  ImmersX::ResidualAccumulator<Vector<double>> residual(layout);
  residual.bind(problem.u, u_residual).bind(problem.p, p_residual);

  problem.add_residual(context, residual);

  EXPECT_DOUBLE_EQ(u_residual[0], 7.);
  EXPECT_DOUBLE_EQ(p_residual[0], 15.);
}


TEST(ResidualCore, PoissonAdapterMatchesAssembledOperator)
{
  ParameterAcceptor::clear();
  PoissonParameters<2> parameters;
  parameters.initial_refinement  = 1;
  parameters.n_refinement_cycles = 1;
  parameters.refinement_strategy = "global";
  parameters.name_of_grid        = "hyper_cube";
  parameters.arguments_for_grid  = "-1: 1: false";

  PoissonSolver<2> problem(parameters);
  initialize_parameters_from_string(R"(
    subsection Poisson
      subsection Right hand side
        set Function expression = 0
        set Variable names      = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names      = x,y,t
      end
    end
  )");
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_system();

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor field =
    descriptor("u", ImmersX::TimeRole::algebraic);
  field.locally_owned    = problem.locally_owned_dofs();
  field.locally_relevant = problem.locally_relevant_dofs();
  const auto u           = layout.add_field(field);

  using VectorType = PoissonSolver<2>::VectorType;
  VectorType state_values;
  state_values.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  state_values = 1.;

  ImmersX::StateView<VectorType> state(layout, 1.5);
  state.bind(u, state_values);
  ImmersX::EvaluationContext<VectorType> context(1.5, state);

  VectorType residual_values;
  residual_values.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  residual_values = 0.;
  ImmersX::ResidualAccumulator<VectorType> residual(layout);
  residual.bind(u, residual_values);

  ImmersX::PoissonResidualContributor<2> contributor(problem, u);
  contributor.add_residual(context, residual);

  VectorType expected;
  expected.reinit(residual_values);
  problem.system_matrix().vmult(expected, state_values);
  expected -= problem.system_rhs();
  residual_values.add(-1., expected);

  EXPECT_LT(residual_values.l2_norm(), 1.e-12);
}
