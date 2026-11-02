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
#include <immersx/core/second_order_preconditioner.h>

#include <array>
#include <vector>

using namespace dealii;
using namespace ImmersX;

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
