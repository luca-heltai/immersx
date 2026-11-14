// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0

#include <deal.II/base/config.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/algebra/mumps_preconditioner.h>

#include <functional>
#include <limits>

using namespace dealii;
using namespace ImmersX;
using FieldVector = ImmersXLA::MPI::Vector;

TEST(MUMPSPreconditioner, BOTH_NoncontiguousInverseMemoryControls)
{
#ifndef DEAL_II_WITH_MUMPS
  GTEST_SKIP() << "MUMPS block inverse requires deal.II with MUMPS";
#else
  constexpr unsigned int size = 8;
  const auto ranks            = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ASSERT_LE(ranks, size);
  IndexSet owned(size), relevant(size);
  relevant.add_range(0, size);
  for (unsigned int i = rank; i < size; i += ranks)
    owned.add_index(i);
  owned.compress();
  DynamicSparsityPattern pattern(relevant);
  for (const auto i : owned)
    {
      pattern.add(i, i);
      if (i > 0)
        pattern.add(i, i - 1);
      if (i + 1 < size)
        pattern.add(i, i + 1);
    }
  SparsityTools::distribute_sparsity_pattern(pattern,
                                             owned,
                                             MPI_COMM_WORLD,
                                             relevant);
  ImmersXLA::MPI::SparseMatrix matrix;
  matrix.reinit(owned, owned, pattern, MPI_COMM_WORLD);
  for (const auto i : owned)
    {
      matrix.set(i, i, 4.);
      if (i > 0)
        matrix.set(i, i - 1, -0.5);
      if (i + 1 < size)
        matrix.set(i, i + 1, 1.);
    }
  matrix.compress(VectorOperation::insert);
  const std::function<void(FieldVector &, bool)> prototype =
    [owned](auto &v, const bool) { v.reinit(owned, MPI_COMM_WORLD); };
  FieldVector exact, rhs, result, error;
  for (auto *v : {&exact, &rhs, &result, &error})
    v->reinit(owned, MPI_COMM_WORLD);
  for (const auto i : owned)
    exact[i] = 1. + 0.1 * i;
  exact.compress(VectorOperation::insert);
  matrix.vmult(rhs, exact);
  const auto check_inverse = [&](const auto &inverse) {
    matrix.vmult(rhs, exact);
    inverse.vmult(result, rhs);
    error = result;
    error -= exact;
    EXPECT_LT(error.l2_norm(), 1.e-12 * exact.l2_norm());
    inverse.vmult_add(result, rhs);
    error = result;
    error.add(-2., exact);
    EXPECT_LT(error.l2_norm(), 1.e-12 * exact.l2_norm());
    matrix.Tvmult(rhs, exact);
    inverse.Tvmult(result, rhs);
    error = result;
    error -= exact;
    EXPECT_LT(error.l2_norm(), 1.e-12 * exact.l2_norm());
    inverse.Tvmult_add(result, rhs);
    error = result;
    error.add(-2., exact);
    EXPECT_LT(error.l2_norm(), 1.e-12 * exact.l2_norm());
  };
  check_inverse(
    make_mumps_preconditioner<FieldVector>(matrix, prototype, MPI_COMM_WORLD));
  for (const int workspace : {-1, 0, 200})
    check_inverse(make_mumps_preconditioner<FieldVector>(
      matrix, prototype, MPI_COMM_WORLD, workspace));
  check_inverse(make_mumps_preconditioner<FieldVector>(
    matrix, prototype, MPI_COMM_WORLD, -1, 128, false));
  check_inverse(make_mumps_preconditioner<FieldVector>(
    matrix, prototype, MPI_COMM_WORLD, -1, 128, true));
  EXPECT_THROW(make_mumps_preconditioner<FieldVector>(
                 matrix,
                 prototype,
                 MPI_COMM_WORLD,
                 -1,
                 std::numeric_limits<unsigned int>::max()),
               ExceptionBase);
  EXPECT_THROW(make_mumps_preconditioner<FieldVector>(
                 matrix, prototype, MPI_COMM_WORLD, -2),
               ExceptionBase);
  if (ranks > 1)
    {
      EXPECT_THROW(
        make_mumps_preconditioner<FieldVector>(
          matrix, prototype, MPI_COMM_WORLD, -1, rank == 0 ? 0u : 128u),
        ExceptionBase);
      EXPECT_THROW(make_mumps_preconditioner<FieldVector>(
                     matrix, prototype, MPI_COMM_WORLD, -1, 0, rank == 0),
                   ExceptionBase);
      EXPECT_THROW(make_mumps_preconditioner<FieldVector>(
                     matrix, prototype, MPI_COMM_WORLD, rank == 0 ? 0 : 200),
                   ExceptionBase);
    }
#endif
}
