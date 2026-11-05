// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <gtest/gtest.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/linear_adapter.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace
{
  void
  check_nearly_exact_amg(const ImmersX::LinearPreconditioner choice)
  {
    using namespace dealii;
    using namespace ImmersX;
    using Matrix  = ImmersXLA::MPI::SparseMatrix;
    using Vector  = ImmersXLA::MPI::Vector;
    using Adapter = LinearAdapter<Vector, ImmersXLA::MPI::BlockVector>;
    ParameterAcceptor::clear();
    const auto rank  = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    IndexSet   owned(6);
    owned.add_range(6 * rank / ranks, 6 * (rank + 1) / ranks);
    owned.compress();
    // Three SPD two-point blocks. The coarse AMG solve is almost exact, so
    // Arnoldi must handle a nearly dependent next basis vector.
    const std::array<std::array<double, 6>, 6> entries = {
      {{{5., 0., 0., -3., 0., 0.}},
       {{0., 3., 0., 0., -1., 0.}},
       {{0., 0., 3., 0., 0., -1.}},
       {{-3., 0., 0., 7., 0., 0.}},
       {{0., -1., 0., 0., 5., 0.}},
       {{0., 0., -1., 0., 0., 5.}}}};
    const std::array<double, 6> loads = {{2., 4., 6., 4., 8., 12.}};
    DynamicSparsityPattern      sparsity(6, 6, owned);
    for (const auto i : owned)
      for (unsigned int j = 0; j < 6; ++j)
        if (entries[i][j] != 0.)
          sparsity.add(i, j);
    Matrix matrix;
    matrix.reinit(owned, owned, sparsity, MPI_COMM_WORLD);
    Vector rhs;
    rhs.reinit(owned, MPI_COMM_WORLD);
    for (const auto i : owned)
      {
        rhs[i] = loads[i];
        for (unsigned int j = 0; j < 6; ++j)
          if (entries[i][j] != 0.)
            matrix.set(i, j, entries[i][j]);
      }
    matrix.compress(VectorOperation::insert);
    rhs.compress(VectorOperation::insert);
    LinearAdapterParameters parameters;
    parameters.preconditioner = choice;
    Adapter    adapter(parameters, MPI_COMM_WORLD);
    const auto field =
      adapter
        .add([&](auto &builder) {
          const auto u         = builder.algebraic_field("solution", owned);
          const auto stiffness = builder.matrix_operator(matrix);
          builder.term(u, "spd")
            .residual([u, stiffness, &rhs](const auto &context) {
              return stiffness.view * context.state(u) - rhs;
            })
            .state(u, stiffness);
          builder.preconditioner(
            u, [](const auto &linearized, const auto &reinit) {
              return make_amg_preconditioner(linearized, reinit);
            });
          return u;
        })
        .fields();
    auto state = adapter.make_state();
    adapter.solve(state);
    const auto &solution = adapter.field(state, field);
    Vector      defect;
    defect.reinit(rhs);
    matrix.vmult(defect, solution);
    defect -= rhs;
    EXPECT_LE(defect.l2_norm(), parameters.tolerance);
    double error = 0.;
    for (const auto i : owned)
      error = std::max(error, std::abs(solution[i] - (1. + i % 3)));
    EXPECT_LT(Utilities::MPI::max(error, MPI_COMM_WORLD), 1.e-12);
  }
} // namespace

TEST(LinearAdapter, BOTH_NearlyExactAMGWithGMRES)
{
  check_nearly_exact_amg(ImmersX::LinearPreconditioner::block_diagonal);
}

TEST(LinearAdapter, BOTH_NearlyExactAMGWithFGMRES)
{
  check_nearly_exact_amg(ImmersX::LinearPreconditioner::block_triangular);
}
