// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#ifdef DEAL_II_WITH_SUNDIALS
namespace
{
  using BlockMatrix = ImmersX::ImmersXLA::MPI::BlockSparseMatrix;

  constexpr std::array<unsigned int, 2> block_sizes = {{4, 3}};
  constexpr unsigned int                n_total     = 7;

  // A nonsymmetric 7x7 matrix with rows of different lengths and entries of
  // different sizes, split into blocks of 4 and 3 rows and columns.
  double
  entry(const unsigned int i, const unsigned int j)
  {
    if (i == j)
      return 10. + i;
    if (j == i + 1)
      return -3.;
    if (i == 0 && j >= 4)
      return 2. + j;
    if (i == 5 && j < 3)
      return -1.5;
    return 0.;
  }

  unsigned int
  offset(const unsigned int block)
  {
    return block == 0 ? 0 : block_sizes[0];
  }

  BlockMatrix
  assemble(const MPI_Comm comm, const double scale = 1.)
  {
    const unsigned int rank  = dealii::Utilities::MPI::this_mpi_process(comm);
    const unsigned int ranks = dealii::Utilities::MPI::n_mpi_processes(comm);
    std::vector<dealii::IndexSet> owned;
    for (const auto size : block_sizes)
      owned.push_back(dealii::Utilities::create_evenly_distributed_partitioning(
        rank, ranks, size));
    BlockMatrix matrix;
    matrix.reinit(2, 2);
    for (unsigned int bi = 0; bi < 2; ++bi)
      for (unsigned int bj = 0; bj < 2; ++bj)
        {
          dealii::DynamicSparsityPattern pattern(block_sizes[bi],
                                                 block_sizes[bj],
                                                 owned[bi]);
          for (const auto i : owned[bi])
            for (unsigned int j = 0; j < block_sizes[bj]; ++j)
              if (entry(offset(bi) + i, offset(bj) + j) != 0.)
                pattern.add(i, j);
          auto &block = matrix.block(bi, bj);
          block.reinit(owned[bi], owned[bj], pattern, comm);
          for (const auto i : owned[bi])
            for (unsigned int j = 0; j < block_sizes[bj]; ++j)
              if (const double value = entry(offset(bi) + i, offset(bj) + j);
                  value != 0.)
                block.set(i, j, scale * value);
          block.compress(dealii::VectorOperation::insert);
        }
    matrix.collect_sizes();
    return matrix;
  }
} // namespace

// The bound must cover ||J||_2 and count the longest row across blocks, on
// any number of ranks.
TEST(DistributedIDA, BOTH_MumpsDefectBoundCoversTheJacobian)
{
  const auto comm   = MPI_COMM_WORLD;
  const auto matrix = assemble(comm);
  const auto bound  = ImmersX::detail::mumps_defect_bound(matrix, comm);

  // Expected values from the dense entries: block-summed row and column
  // norms, and the longest row.
  std::array<std::array<double, 2>, 2> block_linfty{}, block_l1{};
  unsigned int                         longest_row = 0;
  for (unsigned int bi = 0; bi < 2; ++bi)
    for (unsigned int bj = 0; bj < 2; ++bj)
      {
        for (unsigned int i = 0; i < block_sizes[bi]; ++i)
          {
            double sum = 0.;
            for (unsigned int j = 0; j < block_sizes[bj]; ++j)
              sum += std::abs(entry(offset(bi) + i, offset(bj) + j));
            block_linfty[bi][bj] = std::max(block_linfty[bi][bj], sum);
          }
        for (unsigned int j = 0; j < block_sizes[bj]; ++j)
          {
            double sum = 0.;
            for (unsigned int i = 0; i < block_sizes[bi]; ++i)
              sum += std::abs(entry(offset(bi) + i, offset(bj) + j));
            block_l1[bi][bj] = std::max(block_l1[bi][bj], sum);
          }
      }
  for (unsigned int i = 0; i < n_total; ++i)
    {
      unsigned int length = 0;
      for (unsigned int j = 0; j < n_total; ++j)
        length += entry(i, j) != 0. ? 1 : 0;
      longest_row = std::max(longest_row, length);
    }
  const double linfty = std::max(block_linfty[0][0] + block_linfty[0][1],
                                 block_linfty[1][0] + block_linfty[1][1]);
  const double l1 =
    std::max(block_l1[0][0] + block_l1[1][0], block_l1[0][1] + block_l1[1][1]);
  EXPECT_NEAR(bound.jacobian_norm, std::sqrt(l1 * linfty), 1.e-12 * linfty);
  EXPECT_EQ(bound.row_entries, longest_row);

  // ||J v||_2 / ||v||_2 never exceeds the bound.
  for (unsigned int k = 0; k < 4; ++k)
    {
      std::vector<double> v(n_total), w(n_total, 0.);
      for (unsigned int j = 0; j < n_total; ++j)
        v[j] = std::cos(1. + 0.7 * k * j) + (j == k ? 2. : 0.);
      double vv = 0., ww = 0.;
      for (unsigned int i = 0; i < n_total; ++i)
        {
          for (unsigned int j = 0; j < n_total; ++j)
            w[i] += entry(i, j) * v[j];
          ww += w[i] * w[i];
          vv += v[i] * v[i];
        }
      EXPECT_LE(std::sqrt(ww / vv), bound.jacobian_norm);
    }
}

TEST(DistributedIDA, BOTH_MumpsDefectBoundHandlesExtremeMatrixScales)
{
  const auto comm = MPI_COMM_WORLD;
  const auto unscaled =
    ImmersX::detail::mumps_defect_bound(assemble(comm), comm);
  ASSERT_GT(unscaled.jacobian_norm, 0.);
  for (const double scale : {1.e155, 1.e-170})
    {
      const auto scaled =
        ImmersX::detail::mumps_defect_bound(assemble(comm, scale), comm);
      EXPECT_NEAR(scaled.jacobian_norm / scale,
                  unscaled.jacobian_norm,
                  1.e-12 * unscaled.jacobian_norm);
    }
}

// The relative target stays in force for well-scaled solves and for a zero
// correction; it is raised only to the rounding level of J x - b.
TEST(DistributedIDA, BOTH_MumpsDefectTargetFloorsRoundingLevel)
{
  using ImmersX::detail::mumps_defect_target;
  constexpr double                  relative = 1.e-8;
  ImmersX::detail::MumpsDefectBound bound;

  // Without a bound the target is the relative one.
  EXPECT_DOUBLE_EQ(mumps_defect_target(relative, 1., bound, 1.), relative);

  bound.jacobian_norm = 1.e4;
  bound.row_entries   = 80;
  const double gamma  = 81. * std::numeric_limits<double>::epsilon() /
                       (1. - 81. * std::numeric_limits<double>::epsilon());

  // Well scaled: ||J|| ||x|| comparable to ||b||.
  EXPECT_DOUBLE_EQ(mumps_defect_target(relative, 1.e2, bound, 1.e-2),
                   relative * 1.e2);

  // Tiny right-hand side against a large solution term, as at the first step
  // from rest: the relative target lies below the rounding level, and a
  // rounding-level defect is accepted.
  const double rhs      = 1.97e-10;
  const double solution = 1.e-6;
  const double target   = mumps_defect_target(relative, rhs, bound, solution);
  EXPECT_DOUBLE_EQ(target, gamma * (bound.jacobian_norm * solution + rhs));
  EXPECT_GT(target, relative * rhs);
  EXPECT_LE(5.6e-18, target);

  // A zero correction leaves ||J x - b|| = ||b||, which stays rejected.
  EXPECT_DOUBLE_EQ(mumps_defect_target(relative, rhs, bound, 0.),
                   relative * rhs);
  EXPECT_GT(rhs, mumps_defect_target(relative, rhs, bound, 0.));
}

TEST(DistributedIDA, BOTH_MumpsDefectTargetHandlesExtremeNorms)
{
  using ImmersX::detail::mumps_defect_target;
  constexpr double                  relative = 1.e-8;
  constexpr double                  rhs      = 1.e150;
  ImmersX::detail::MumpsDefectBound bound{1.e155, 1};
  const double unit  = 2. * std::numeric_limits<double>::epsilon();
  const double gamma = unit / (1. - unit);

  // The unscaled product would overflow, but the rounding target is finite.
  const double expected = (gamma * 1.e300) * 1.e10;
  EXPECT_NEAR(mumps_defect_target(relative, rhs, bound, 1.e155),
              expected,
              1.e-14 * expected);

  // An unrepresentable estimate must leave the relative target in force.
  EXPECT_DOUBLE_EQ(mumps_defect_target(relative, rhs, bound, 1.e200),
                   relative * rhs);
  bound.jacobian_norm = std::numeric_limits<double>::infinity();
  EXPECT_DOUBLE_EQ(mumps_defect_target(relative, rhs, bound, 1.),
                   relative * rhs);

  // Scaling the tiny Jacobian norm first would underflow to zero.
  bound.jacobian_norm        = 1.e-310;
  const double tiny_expected = gamma * 1.e-160;
  EXPECT_NEAR(mumps_defect_target(relative, 1.e-200, bound, 1.e150),
              tiny_expected,
              1.e-13 * tiny_expected);
}
#endif
