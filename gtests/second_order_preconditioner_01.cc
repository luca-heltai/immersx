// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/solver_gmres.h>

#include <gtest/gtest.h>
#include <immersx/core/detail/execution_composition.h>
#include <immersx/core/second_order_preconditioner.h>

#include <array>
#include <vector>

using namespace dealii;
using namespace ImmersX;

namespace ImmersX
{
  struct SecondOrderRevisionProblem
  {
    ImmersXLA::MPI::SparseMatrix                    mass, stiffness, damping;
    ImmersXLA::MPI::PreconditionAMG::AdditionalData amg_data;
    dealii::IndexSet                                owned;
    unsigned long                                   revision = 0;
    mutable unsigned int preconditioner_data_calls           = 0;
  };

  template <typename Builder>
  auto
  contribute(Builder &builder, const SecondOrderRevisionProblem &problem)
  {
    const auto displacement = builder.differential_field("u", problem.owned);
    const auto velocity     = builder.differential_field("v", problem.owned);
    const auto mass         = builder.matrix_operator(problem.mass);
    builder.term(displacement, "kinematic")
      .derivative(displacement, mass)
      .state(velocity, -1. * mass);
    builder.term(velocity, "dynamic")
      .derivative(velocity, mass)
      .state(displacement, builder.matrix_operator(problem.stiffness))
      .state(velocity, builder.matrix_operator(problem.damping));
    builder.second_order_pair(
      displacement,
      velocity,
      [&problem]() { return problem.revision; },
      [&problem]() {
        ++problem.preconditioner_data_calls;
        SecondOrderPreconditionerData data;
        data.constrained = dealii::IndexSet(problem.owned.size());
        data.amg_data    = problem.amg_data;
        return data;
      });
    return std::array<FieldId, 2>{{displacement, velocity}};
  }
} // namespace ImmersX

namespace
{
  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Matrix       = ImmersXLA::MPI::SparseMatrix;
  using Dense        = std::array<std::array<double, 3>, 3>;

  void
  initialize_matrix(Matrix &matrix, const IndexSet &owned, const Dense &values)
  {
    DynamicSparsityPattern sparsity(3, 3, owned);
    for (const auto row : owned)
      for (unsigned int column = 0; column < 3; ++column)
        sparsity.add(row, column);
    sparsity.compress();
    matrix.reinit(owned, owned, sparsity, MPI_COMM_WORLD);
    for (const auto row : owned)
      for (unsigned int column = 0; column < 3; ++column)
        matrix.set(row, column, values[row][column]);
    matrix.compress(VectorOperation::insert);
  }
} // namespace


TEST(Elastodynamics, BOTH_PhysicalSchurFactorizationAndFlexibleSolve)
{
  // The noncommuting positive definite matrices exercise the complete block
  // factorization, rather than a scalar or simultaneously diagonal case.
  const Dense mass_values{{{{4., 1., 0.}}, {{1., 3., 0.5}}, {{0., 0.5, 2.}}}};
  const Dense stiffness_values{
    {{{6., -2., 0.}}, {{-2., 5., -1.}}, {{0., -1., 3.}}}};
  const Dense damping_values{
    {{{0.5, 0.1, 0.}}, {{0.1, 0.4, 0.05}}, {{0., 0.05, 0.3}}}};
  const auto owned =
    Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, 3);
  const std::vector<IndexSet> partitioning(2, owned);
  Matrix                      mass, stiffness, damping;
  initialize_matrix(mass, owned, mass_values);
  initialize_matrix(stiffness, owned, stiffness_values);
  initialize_matrix(damping, owned, damping_values);

  IndexSet constrained(3);
  constrained.add_index(2);
  const auto reinit = [owned](FieldVector &vector, const bool omit) {
    vector.reinit(owned, MPI_COMM_WORLD, omit);
  };
  SecondOrderBlockPreconditioner<FieldVector> factorization;
  factorization.initialize(mass, stiffness, damping, constrained, reinit);

  const std::array<std::array<double, 4>, 4> coefficients{{
    {{2., 1., 1., 1.}},
    {{40., 0.5, 0.5, 0.5}},
    {{0.125, 1., 1., 1.}},
    {{2., 1., 1., 1.}},
  }};
  for (const auto &coefficient : coefficients)
    {
      const auto [a, b, c, d] = coefficient;
      SCOPED_TRACE(a);
      factorization.set_coefficients(a, b, c, d);

      std::array<Matrix, 4> blocks;
      for (unsigned int row_block = 0; row_block < 2; ++row_block)
        for (unsigned int column_block = 0; column_block < 2; ++column_block)
          {
            Dense values{};
            for (unsigned int i = 0; i < 3; ++i)
              for (unsigned int j = 0; j < 3; ++j)
                {
                  if (i == 2 || j == 2)
                    values[i][j] =
                      row_block == column_block && i == j ? 1. : 0.;
                  else if (row_block == 0 && column_block == 0)
                    values[i][j] = a * mass_values[i][j];
                  else if (row_block == 0 && column_block == 1)
                    values[i][j] = -b * mass_values[i][j];
                  else if (row_block == 1 && column_block == 0)
                    values[i][j] = c * stiffness_values[i][j];
                  else if (row_block == 1 && column_block == 1)
                    values[i][j] =
                      a * mass_values[i][j] + d * damping_values[i][j];
                }
            initialize_matrix(blocks[2 * row_block + column_block],
                              owned,
                              values);
          }

      LinearOperator<GlobalVector> matrix;
      matrix.vmult = [&blocks](GlobalVector       &destination,
                               const GlobalVector &source) {
        blocks[0].vmult(destination.block(0), source.block(0));
        blocks[1].vmult_add(destination.block(0), source.block(1));
        blocks[2].vmult(destination.block(1), source.block(0));
        blocks[3].vmult_add(destination.block(1), source.block(1));
      };
      LinearOperator<GlobalVector> preconditioner;
      preconditioner.vmult = [&factorization](GlobalVector       &destination,
                                              const GlobalVector &source) {
        factorization.vmult(destination.block(0),
                            destination.block(1),
                            source.block(0),
                            source.block(1));
      };

      GlobalVector expected(partitioning, MPI_COMM_WORLD);
      GlobalVector rhs(partitioning, MPI_COMM_WORLD);
      GlobalVector solution(partitioning, MPI_COMM_WORLD);
      GlobalVector residual(partitioning, MPI_COMM_WORLD);
      for (const auto index : owned)
        {
          expected.block(0)(index) = 1. + index;
          expected.block(1)(index) = 0.5 - 2. * index;
        }
      matrix.vmult(rhs, expected);

      // This includes nonzero constrained rows: the preconditioner must
      // preserve their identity action even as a and the Schur operator change.
      preconditioner.vmult(solution, rhs);
      matrix.vmult(residual, solution);
      residual -= rhs;
      EXPECT_LE(residual.l2_norm(), 0.1 * rhs.l2_norm());

      solution = 0.;
      SolverControl              control(100, 1.e-12 * rhs.l2_norm());
      SolverFGMRES<GlobalVector> solver(control);
      solver.solve(matrix, solution, rhs, preconditioner);
      matrix.vmult(residual, solution);
      residual -= rhs;
      EXPECT_LE(residual.l2_norm(), 2.e-12 * rhs.l2_norm());
      solution -= expected;
      EXPECT_LE(solution.l2_norm(), 1.e-9);
    }
}


TEST(Elastodynamics, BOTH_PhysicalSchurRefreshesOnSpatialRevision)
{
  SecondOrderRevisionProblem problem;
  problem.owned =
    Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, 3);
  const auto set_spatial_matrices =
    [&](const double mass, const double stiffness, const double damping) {
      Dense M{}, K{}, D{};
      for (unsigned int i = 0; i < 3; ++i)
        {
          M[i][i] = mass;
          K[i][i] = stiffness;
          D[i][i] = damping;
        }
      initialize_matrix(problem.mass, problem.owned, M);
      initialize_matrix(problem.stiffness, problem.owned, K);
      initialize_matrix(problem.damping, problem.owned, D);
      ++problem.revision;
    };
  set_spatial_matrices(2., 3., 0.5);
  ImmersX::detail::ExecutionComposition<FieldVector, GlobalVector> composition(
    MPI_COMM_WORLD);
  composition.add(problem, "solid");
  auto state     = composition.make_state();
  auto state_dot = composition.make_state();
  auto rhs       = composition.make_state();
  auto action    = composition.make_state();
  for (const auto index : problem.owned)
    {
      rhs.block(0)(index) = 1. + index;
      rhs.block(1)(index) = 2. - index;
    }

  for (unsigned int phase = 0; phase < 4; ++phase)
    {
      const double alpha = phase == 1 ? 40. : 2.;
      if (phase == 2)
        // Keep the same partition and sparsity: a geometry/material rebuild
        // cannot be detected just by checking the number of unknowns.
        set_spatial_matrices(5., 11., 1.5);
      const auto inverse =
        composition.second_order_preconditioner(state, state_dot, alpha);
      inverse.vmult(action, rhs);
      EXPECT_EQ(problem.preconditioner_data_calls, phase < 2 ? 1u : 2u);

      const double mass      = phase < 2 ? 2. : 5.;
      const double stiffness = phase < 2 ? 3. : 11.;
      const double damping   = phase < 2 ? 0.5 : 1.5;
      // Direct inverse of each independent physical 2x2 system. The scalar
      // spatial matrices make both inner CG solves exact in one iteration.
      const double A           = alpha * mass;
      const double B           = -mass;
      const double C           = stiffness;
      const double D           = alpha * mass + damping;
      const double determinant = A * D - B * C;
      for (const auto index : problem.owned)
        {
          EXPECT_NEAR(action.block(0)(index),
                      (D * rhs.block(0)(index) - B * rhs.block(1)(index)) /
                        determinant,
                      1.e-11);
          EXPECT_NEAR(action.block(1)(index),
                      (A * rhs.block(1)(index) - C * rhs.block(0)(index)) /
                        determinant,
                      1.e-11);
        }
    }

  // A new mesh may change physical matrices before the caller replaces the
  // execution state. Reject that stale state collectively before AMG setup.
  const auto new_owned =
    Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, 4);
  DynamicSparsityPattern new_sparsity(4, 4, new_owned);
  for (const auto index : new_owned)
    new_sparsity.add(index, index);
  new_sparsity.compress();
  for (auto *matrix : {&problem.mass, &problem.stiffness, &problem.damping})
    {
      matrix->reinit(new_owned, new_owned, new_sparsity, MPI_COMM_WORLD);
      for (const auto index : new_owned)
        matrix->set(index, index, 2.);
      matrix->compress(VectorOperation::insert);
    }
  ++problem.revision;
  EXPECT_THROW(composition.second_order_preconditioner(state, state_dot, 2.),
               ExceptionBase);
}


TEST(Elastodynamics, BOTH_CoupledSchurActionIsLinear)
{
  SecondOrderRevisionProblem problem;
  problem.owned =
    Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, 3);
  const Dense mass{{{{4., 1., 0.}}, {{1., 3., 0.5}}, {{0., 0.5, 2.}}}};
  const Dense stiffness{{{{6., -2., 0.}}, {{-2., 5., -1.}}, {{0., -1., 3.}}}};
  const Dense damping{
    {{{0.5, 0.1, 0.}}, {{0.1, 0.4, 0.05}}, {{0., 0.05, 0.3}}}};
  const Dense identity{{{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};
  initialize_matrix(problem.mass, problem.owned, mass);
  initialize_matrix(problem.stiffness, problem.owned, stiffness);
  initialize_matrix(problem.damping, problem.owned, damping);
  Matrix coupling;
  initialize_matrix(coupling, problem.owned, identity);
  // A deliberately inexact coarse solve distinguishes a fixed AMG action
  // from a residual-stopped CG inverse inside the Schur matrix action.
  problem.amg_data.coarse_type     = "Jacobi";
  problem.amg_data.smoother_sweeps = 1;
  ImmersX::detail::ExecutionComposition<FieldVector, GlobalVector> composition(
    MPI_COMM_WORLD);
  const auto fields = composition.add(
    [&problem, &coupling](auto &builder) {
      const auto fields     = contribute(builder, problem);
      const auto multiplier = builder.algebraic_field("lambda", problem.owned);
      const auto trace      = builder.matrix_operator(coupling);
      builder.term(fields[1], "force").state(multiplier, trace);
      builder.term(multiplier, "compatibility").state(fields[0], trace);
      builder.saddle_point(multiplier, {fields[0], fields[1]}, trace);
      return std::array<FieldId, 3>{{fields[0], fields[1], multiplier}};
    },
    "solid");
  auto       state     = composition.make_state();
  auto       state_dot = composition.make_state();
  const auto schur =
    composition.schur_operator(fields[2], state, &state_dot, 2., true);
  FieldVector x, y, sum, action_x, action_y, action_sum;
  for (auto *vector : {&x, &y, &sum, &action_x, &action_y, &action_sum})
    vector->reinit(problem.owned, MPI_COMM_WORLD);
  for (const auto index : problem.owned)
    {
      x(index) = 1. + 0.7 * index;
      y(index) = index == 1 ? -0.8 : 0.3 + index;
    }
  sum = x;
  sum += y;
  schur.vmult(action_x, x);
  schur.vmult(action_y, y);
  schur.vmult(action_sum, sum);
  action_sum -= action_x;
  action_sum -= action_y;
  EXPECT_LE(action_sum.l2_norm(),
            1.e-12 * (action_x.l2_norm() + action_y.l2_norm()));
  schur.Tvmult(action_sum, x);
  EXPECT_NEAR(x * action_y,
              action_sum * y,
              1.e-12 * (x.l2_norm() * action_y.l2_norm() +
                        action_sum.l2_norm() * y.l2_norm()));
}
