// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <gtest/gtest.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/linear_adapter.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

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

  class ScopedCoutCapture
  {
  public:
    explicit ScopedCoutCapture(std::ostringstream &output)
      : original_buffer_(std::cout.rdbuf(output.rdbuf()))
    {}

    ~ScopedCoutCapture()
    {
      std::cout.rdbuf(original_buffer_);
    }

  private:
    std::streambuf *original_buffer_;
  };

  void
  check_fgmres_failure_reports_true_residual()
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
    parameters.solver             = LinearSolver::iterative;
    parameters.preconditioner     = LinearPreconditioner::none;
    parameters.maximum_iterations = 1;
    parameters.fgmres_basis_size  = 1;
    parameters.tolerance          = 1.e-30;
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
          return u;
        })
        .fields();

    auto               state = adapter.make_state();
    std::ostringstream output;
    {
      ScopedCoutCapture capture(output);
      EXPECT_THROW(adapter.solve(state), SolverControl::NoConvergence);
    }
    EXPECT_GT(state.l2_norm(), 0.);

    auto residual = adapter.make_state();
    adapter.evaluate_residual(state, residual);
    const auto   diagnostics    = output.str();
    const double expected_total = residual.l2_norm();
    const double expected_field = adapter.field(residual, field).l2_norm();
    if (rank == 0)
      {
        EXPECT_NE(diagnostics.find("NoConvergence at its iteration 1"),
                  std::string::npos);
        EXPECT_NE(diagnostics.find("projected residual="), std::string::npos);

        const auto read_reported_norm = [&](const std::string &marker) {
          const auto marker_position = diagnostics.find(marker);
          EXPECT_NE(marker_position, std::string::npos);
          if (marker_position == std::string::npos)
            return 0.;
          const auto value_position = marker_position + marker.size();
          const auto value_end =
            diagnostics.find_first_not_of("+-0123456789.eE", value_position);
          return std::stod(
            diagnostics.substr(value_position, value_end - value_position));
        };
        const double reported_total =
          read_reported_norm("true residual ||b-Ax||=");
        EXPECT_NEAR(reported_total,
                    expected_total,
                    1.e-5 * std::max(1., expected_total));
        const double reported_field =
          read_reported_norm("field 'solution' ||b-Ax||=");
        EXPECT_NEAR(reported_field,
                    expected_field,
                    1.e-5 * std::max(1., expected_field));
      }
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

TEST(LinearAdapter, BOTH_FGMRESFailureReportsTrueResidual)
{
  check_fgmres_failure_reports_true_residual();
}
