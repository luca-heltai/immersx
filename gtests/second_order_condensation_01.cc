// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/detail/second_order_condensation.h>

#include <functional>
#include <vector>

using namespace dealii;

namespace
{
  using FieldVector  = ImmersX::ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Matrix       = ImmersX::ImmersXLA::MPI::SparseMatrix;
  using BlockMatrix  = ImmersX::ImmersXLA::MPI::BlockSparseMatrix;
  using Condensation =
    ImmersX::detail::SecondOrderCondensation<Matrix, FieldVector>;

  constexpr unsigned int n           = 7; // displacement and velocity size
  constexpr unsigned int m           = 2; // multiplier size
  constexpr unsigned int constrained = 4; // one constrained displacement DoF

  // Condensed (constrained row and column removed) symmetric operators.
  double
  mass(const unsigned int i, const unsigned int j)
  {
    if (i == constrained || j == constrained)
      return 0.;
    return i == j ? 4. + 0.1 * i : (i + 1 == j || j + 1 == i ? 1. : 0.);
  }

  double
  stiffness(const unsigned int i, const unsigned int j)
  {
    if (i == constrained || j == constrained)
      return 0.;
    return i == j ? 6. : (i + 1 == j || j + 1 == i ? -2. : 0.);
  }

  double
  damping(const unsigned int i, const unsigned int j)
  {
    return 0.05 * mass(i, j);
  }

  // The multiplier rows see the displacement, including the constrained DoF.
  double
  coupling(const unsigned int row, const unsigned int column)
  {
    return row == 0 ? (column < 3 ? 1. + column : 0.) :
                      (column >= 3 ? 0.5 * column : 0.);
  }

  void
  fill_block(BlockMatrix                 &jacobian,
             const unsigned int           row_block,
             const unsigned int           column_block,
             const std::vector<IndexSet> &partitions,
             const std::function<double(unsigned int, unsigned int)> &value)
  {
    const auto            &rows    = partitions[row_block];
    const auto            &columns = partitions[column_block];
    DynamicSparsityPattern sparsity(rows.size(), columns.size(), rows);
    for (const auto row : rows)
      for (unsigned int column = 0; column < columns.size(); ++column)
        if (value(row, column) != 0.)
          sparsity.add(row, column);
    auto &block = jacobian.block(row_block, column_block);
    block.reinit(rows, columns, sparsity, MPI_COMM_WORLD);
    for (const auto row : rows)
      for (unsigned int column = 0; column < columns.size(); ++column)
        if (value(row, column) != 0.)
          block.set(row, column, value(row, column));
    block.compress(VectorOperation::insert);
  }

  // IDA's J = dF/dy + alpha dF/dydot for M u' - M v = 0 and
  // M v' + D v + K u + B^T lambda = f, B u - eps lambda = g.
  void
  assemble(BlockMatrix                 &jacobian,
           const std::vector<IndexSet> &partitions,
           const double                 alpha,
           const double                 kinematic_perturbation = 0.)
  {
    jacobian.reinit(3, 3);
    const auto zero = [](unsigned int, unsigned int) { return 0.; };
    const auto free = [](const unsigned int i) { return i != constrained; };
    fill_block(jacobian, 0, 0, partitions, [&](auto i, auto j) {
      return free(i) ? alpha * mass(i, j) : (i == j ? 2.5 : 0.);
    });
    fill_block(jacobian, 0, 1, partitions, [&](auto i, auto j) {
      return free(i) ? -mass(i, j) *
                         (i == 0 && j == 0 ? 1. + kinematic_perturbation : 1.) :
                       0.;
    });
    fill_block(jacobian, 0, 2, partitions, zero);
    fill_block(jacobian, 1, 0, partitions, [&](auto i, auto j) {
      return free(i) ? stiffness(i, j) : 0.;
    });
    fill_block(jacobian, 1, 1, partitions, [&](auto i, auto j) {
      return free(i) ? alpha * mass(i, j) + damping(i, j) : (i == j ? 1. : 0.);
    });
    fill_block(jacobian, 1, 2, partitions, [&](auto i, auto j) {
      return free(i) ? coupling(j, i) : 0.;
    });
    fill_block(jacobian, 2, 0, partitions, coupling);
    fill_block(jacobian, 2, 1, partitions, zero);
    fill_block(jacobian, 2, 2, partitions, [](auto i, auto j) {
      return i == j ? -1.e-2 : 0.;
    });
    jacobian.collect_sizes();
  }

  void
  solve(const Matrix &matrix, FieldVector &solution, const FieldVector &rhs)
  {
    SolverControl            control(500, 1.e-14 * rhs.l2_norm());
    SolverGMRES<FieldVector> solver(control);
    solution.reinit(rhs);
    solver.solve(matrix, solution, rhs, PreconditionIdentity());
  }

  std::vector<IndexSet>
  partitions()
  {
    return {
      Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, n),
      Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, n),
      Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,
                                                             m)};
  }
} // namespace


TEST(SecondOrderCondensation, BOTH_ReducedSolveMatchesCoupledJacobian)
{
  const auto                          layout = partitions();
  std::shared_ptr<const Condensation> previous;
  for (const double alpha : {3., 0.})
    {
      SCOPED_TRACE(alpha);
      BlockMatrix jacobian;
      assemble(jacobian, layout, alpha);
      const auto condensation =
        Condensation::build(jacobian, {{0, 1}}, layout, MPI_COMM_WORLD);
      ASSERT_TRUE(condensation);
      ASSERT_EQ(condensation->n_pairs(), 1u);
      EXPECT_EQ(condensation->reduced_matrix().m(), n + m);
      // The kinematic mass does not depend on the derivative weight.
      if (previous)
        EXPECT_TRUE(previous->kinematic_mass(0) ==
                    condensation->kinematic_mass(0));
      previous = condensation;

      GlobalVector rhs(layout, MPI_COMM_WORLD);
      for (unsigned int block = 0; block < 3; ++block)
        for (const auto index : layout[block])
          rhs.block(block)(index) = 1. + 0.3 * index - 0.7 * block;
      rhs.compress(VectorOperation::insert);

      const auto mass_matrix = condensation->kinematic_mass(0).make_matrix(
        condensation->kinematic_partition(0), MPI_COMM_WORLD);
      FieldVector              kinematic_rhs;
      std::vector<FieldVector> kinematic(1);
      condensation->kinematic_rhs(rhs, 0, kinematic_rhs);
      solve(*mass_matrix, kinematic[0], kinematic_rhs);
      FieldVector reduced_rhs, reduced_solution;
      condensation->reduced_rhs(rhs, kinematic, reduced_rhs);
      solve(condensation->reduced_matrix(), reduced_solution, reduced_rhs);
      GlobalVector solution(layout, MPI_COMM_WORLD);
      condensation->expand(reduced_solution, kinematic, solution);

      GlobalVector residual(layout, MPI_COMM_WORLD);
      jacobian.vmult(residual, solution);
      residual -= rhs;
      EXPECT_LT(residual.l2_norm(), 1.e-10 * rhs.l2_norm());
    }
}

TEST(SecondOrderCondensation, BOTH_RejectsNonKinematicRows)
{
  const auto  layout = partitions();
  BlockMatrix jacobian;
  assemble(jacobian, layout, 3., 1.e-3);
  EXPECT_FALSE(Condensation::build(jacobian, {{0, 1}}, layout, MPI_COMM_WORLD));
  EXPECT_FALSE(Condensation::build(jacobian, {}, layout, MPI_COMM_WORLD));
}
