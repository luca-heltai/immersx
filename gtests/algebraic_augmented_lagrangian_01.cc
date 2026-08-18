// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <gtest/gtest.h>

#include "augmented_lagrangian.h"
#include "constraint_system.h"

using namespace dealii;

namespace
{
  struct AlgebraicOperators
  {
    SparsityPattern      sparsity_A;
    SparsityPattern      sparsity_B;
    SparsityPattern      sparsity_M;
    SparseMatrix<double> A;
    SparseMatrix<double> B;
    SparseMatrix<double> M;

    AlgebraicOperators()
      : sparsity_A(2, 2, 2)
      , sparsity_B(1, 2, 2)
      , sparsity_M(1, 1, 1)
    {
      sparsity_A.add(0, 0);
      sparsity_A.add(1, 1);
      sparsity_A.compress();
      A.reinit(sparsity_A);
      A.set(0, 0, 2.0);
      A.set(1, 1, 3.0);

      sparsity_B.add(0, 0);
      sparsity_B.add(0, 1);
      sparsity_B.compress();
      B.reinit(sparsity_B);
      B.set(0, 0, 1.0);
      B.set(0, 1, 1.0);

      sparsity_M.add(0, 0);
      sparsity_M.compress();
      M.reinit(sparsity_M);
      M.set(0, 0, 2.0);
    }
  };
} // namespace

TEST(ConstraintSystem, StoresAndAppliesOperators) // NOLINT
{
  AlgebraicOperators matrices;
  const auto         A      = linear_operator(matrices.A);
  const auto         B      = linear_operator(matrices.B);
  const auto         Bt     = transpose_operator(B);
  const auto         M      = linear_operator(matrices.M);
  const auto         system = make_constraint_system(A, B, Bt, M);

  Vector<double> primal;
  Vector<double> primal_range;
  Vector<double> constraint;
  Vector<double> multiplier;
  Vector<double> multiplier_domain;
  Vector<double> adjoint;
  Vector<double> metric;
  system.primal_operator().reinit_domain_vector(primal, false);
  system.primal_operator().reinit_range_vector(primal_range, false);
  system.constraint_operator().reinit_domain_vector(primal, false);
  system.constraint_operator().reinit_range_vector(constraint, false);
  system.multiplier_metric().reinit_domain_vector(multiplier, false);
  system.multiplier_metric().reinit_range_vector(metric, false);
  system.adjoint_constraint_operator().reinit_domain_vector(multiplier_domain,
                                                            false);
  system.adjoint_constraint_operator().reinit_range_vector(adjoint, false);

  primal[0]     = 1.0;
  primal[1]     = 2.0;
  multiplier[0] = 2.0;

  constraint = system.constraint_operator() * primal;
  adjoint    = system.adjoint_constraint_operator() * multiplier;
  metric     = system.multiplier_metric() * multiplier;

  EXPECT_EQ(primal.size(), 2U);
  EXPECT_EQ(constraint.size(), 1U);
  EXPECT_EQ(adjoint.size(), 2U);
  EXPECT_EQ(metric.size(), 1U);
  Vector<double> primal_result = system.primal_operator() * primal;
  EXPECT_DOUBLE_EQ(primal_result[1], 6.0);
  EXPECT_DOUBLE_EQ(constraint[0], 3.0);
  EXPECT_DOUBLE_EQ(adjoint[0], 2.0);
  EXPECT_DOUBLE_EQ(adjoint[1], 2.0);
  EXPECT_DOUBLE_EQ(metric[0], 4.0);
}

TEST(AugmentedLagrangianSolver, SolvesStandaloneKKT) // NOLINT
{
  AlgebraicOperators matrices;
  const auto         A      = linear_operator(matrices.A);
  const auto         B      = linear_operator(matrices.B);
  const auto         Bt     = transpose_operator(B);
  const auto         M      = linear_operator(matrices.M);
  const auto         system = make_constraint_system(A, B, Bt, M);

  Vector<double> f(2);
  f[0] = 1.0;
  f[1] = 2.0;

  BlockVector<double> rhs(2, 1);
  rhs.block(0)    = f;
  rhs.block(1)[0] = 1.0;

  SolverControl            outer_control(100, 1e-12);
  SolverControl            inner_control(100, 1e-12);
  SolverCG<Vector<double>> inner_solver(inner_control);
  const auto               invW = linear_operator(matrices.M);

  const auto build_augmented_block =
    [&inner_solver](const auto &canonical_Aug) {
      Vector<double> probe(2);
      probe[0]                             = 1.0;
      probe[1]                             = 0.0;
      const Vector<double> augmented_probe = canonical_Aug * probe;
      EXPECT_DOUBLE_EQ(augmented_probe[0], 22.0);
      EXPECT_DOUBLE_EQ(augmented_probe[1], 20.0);
      return make_prepared_augmented_block(canonical_Aug,
                                           inverse_operator(canonical_Aug,
                                                            inner_solver));
    };

  AugmentedLagrangianSolver<Vector<double>, BlockVector<double>> solver(
    outer_control, {10.0});
  BlockVector<double> solution;

  solver.solve(system, invW, build_augmented_block, solution, rhs);

  EXPECT_NEAR(solution.block(0)[0], 0.4, 1e-10);
  EXPECT_NEAR(solution.block(0)[1], 0.6, 1e-10);
  EXPECT_NEAR(solution.block(1)[0], 0.2, 1e-10);

  Vector<double> primal_residual = A * solution.block(0);
  primal_residual.add(1.0, Bt * solution.block(1));
  primal_residual.add(-1.0, f);
  Vector<double> constraint_residual = B * solution.block(0);
  constraint_residual[0] -= rhs.block(1)[0];

  EXPECT_LT(primal_residual.l2_norm(), 1e-10);
  EXPECT_LT(constraint_residual.l2_norm(), 1e-10);
}
