// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_lib.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <gtest/gtest.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/physics/elastic_static.h>

#include <cmath>

TEST(ElasticStaticProblem, LinearAdapterSolve)
{
  using Problem      = ImmersX::ElasticStaticProblem<3, 3>;
  using FieldVector  = typename Problem::VectorType;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter      = ImmersX::LinearAdapter<FieldVector, GlobalVector>;

  Problem problem(MPI_COMM_WORLD, 2);
  problem.setup();
  problem.set_forcing(
    dealii::Functions::ConstantFunction<3>(std::vector<double>{0., 0., 1.}));

  Adapter    adapter(MPI_COMM_WORLD,
                  [](const auto &operator_view,
                     const auto &rhs,
                     auto       &solution) {
                    dealii::SolverControl          control(1000, 1.e-10);
                    dealii::SolverCG<GlobalVector> solver(control);
                    dealii::PreconditionIdentity   preconditioner;
                    solver.solve(operator_view, solution, rhs, preconditioner);
                  });
  const auto fields = adapter.add(problem, "elasticity");

  auto state = adapter.make_state();
  adapter.solve(state);
  problem.set_solution(adapter.field(state, fields.fields().displacement));

  EXPECT_TRUE(std::isfinite(problem.solution().l2_norm()));
  EXPECT_GT(problem.solution().l2_norm(), 0.);

  GlobalVector residual;
  adapter.evaluate_residual(state, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-8);
}
