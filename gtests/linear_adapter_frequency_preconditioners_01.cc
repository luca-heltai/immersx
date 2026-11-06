// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <gtest/gtest.h>
#include <immersx/core/linear_adapter.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace
{
  void
  check_permutation(const ImmersX::LinearPreconditioner choice,
                    const bool                          root_owned)
  {
    using namespace ImmersX;
    using namespace dealii;
    using Vector  = ImmersXLA::MPI::Vector;
    using Matrix  = ImmersXLA::MPI::SparseMatrix;
    using Adapter = LinearAdapter<Vector, ImmersXLA::MPI::BlockVector>;
    ParameterAcceptor::clear();
    const auto comm  = MPI_COMM_WORLD;
    const auto rank  = Utilities::MPI::this_mpi_process(comm);
    const auto ranks = Utilities::MPI::n_mpi_processes(comm);
    IndexSet   owned(3);
    if (root_owned)
      {
        if (rank == 0)
          owned.add_range(0, 3);
      }
    else
      owned.add_range(3 * rank / ranks, 3 * (rank + 1) / ranks);
    owned.compress();
    std::array<std::array<Matrix, 2>, 2> matrices;
    std::array<Vector, 2>                loads;
    // H cycles six coordinates. H H^T=I, while H^2 has a degree-three
    // minimal polynomial for this load. Two outer steps distinguish the
    // transpose action from an accidental forward action.
    const std::array<double, 6> exact{{1., 2., 3., 4., 5., 6.}};
    for (unsigned int i = 0; i < 2; ++i)
      {
        loads[i].reinit(owned, comm);
        for (unsigned int j = 0; j < 2; ++j)
          {
            DynamicSparsityPattern graph(3, 3, owned);
            for (const auto row : owned)
              {
                const unsigned int column = (3 * i + row + 1) % 6;
                if (column / 3 == j)
                  graph.add(row, column % 3);
              }
            matrices[i][j].reinit(owned, owned, graph, comm);
            for (const auto row : owned)
              {
                const unsigned int column = (3 * i + row + 1) % 6;
                if (column / 3 == j)
                  matrices[i][j].set(row, column % 3, 1.);
              }
            matrices[i][j].compress(VectorOperation::insert);
          }
        for (const auto row : owned)
          loads[i][row] = exact[(3 * i + row + 1) % 6];
        loads[i].compress(VectorOperation::insert);
      }
    LinearAdapterParameters options;
    options.solver             = LinearSolver::iterative;
    options.preconditioner     = choice;
    options.maximum_iterations = 2;
    options.tolerance          = 1.e-11;
    Adapter    adapter(options, comm);
    const auto fields =
      adapter
        .add([&](auto &builder) {
          const std::array<FieldId, 2> ids{
            {builder.algebraic_field("first", owned),
             builder.algebraic_field("second", owned)}};
          for (unsigned int i = 0; i < 2; ++i)
            {
              const auto a = builder.matrix_operator(matrices[i][0]);
              const auto b = builder.matrix_operator(matrices[i][1]);
              builder.term(ids[i], "permutation")
                .residual([&, ids, i, a, b](const auto &context) {
                  return a.view * context.state(ids[0]) +
                         b.view * context.state(ids[1]) - loads[i];
                })
                .state(ids[0], a)
                .state(ids[1], b);
              builder.preconditioner(ids[i],
                                     [](const auto &, const auto &reinit) {
                                       return identity_operator<Vector>(reinit);
                                     });
            }
          return ids;
        })
        .fields();
    auto state = adapter.make_state();
    adapter.solve(state);
    auto defect = adapter.make_state();
    adapter.evaluate_residual(state, defect);
    EXPECT_LE(defect.l2_norm(), options.tolerance);
    double error = 0.;
    for (unsigned int i = 0; i < 2; ++i)
      for (const auto row : owned)
        error = std::max(error,
                         std::abs(adapter.field(state, fields[i])[row] -
                                  exact[3 * i + row]));
    EXPECT_LT(Utilities::MPI::max(error, comm), 1.e-10);
  }
} // namespace

TEST(LinearAdapter, BOTH_NormalPreconditionerUsesTranspose)
{
  check_permutation(ImmersX::LinearPreconditioner::normal, false);
  check_permutation(ImmersX::LinearPreconditioner::normal, true);
}

#if defined(DEAL_II_WITH_MUMPS) && !defined(IMMERSX_USE_PETSC_LA)
TEST(LinearAdapter, BOTH_MumpsBlrHandlesFieldPartitions)
{
  check_permutation(ImmersX::LinearPreconditioner::mumps_blr, false);
  check_permutation(ImmersX::LinearPreconditioner::mumps_blr, true);
}
#endif
