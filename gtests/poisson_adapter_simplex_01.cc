// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_parser.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson_residual.h>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  void
  check_poisson_adapter(const std::string &parameter_file)
  {
    ParameterAcceptor::clear();
    PoissonParameters<3> parameters;
    initialize_parameters(TestPaths::parameter_path(parameter_file));
    PoissonSolver<3> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_system();

    using VectorType   = PoissonSolver<3>::VectorType;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    LinearSolverParameters options;
    options.solver             = LinearSolver::iterative;
    options.preconditioner     = LinearPreconditioner::block_diagonal;
    options.maximum_iterations = 1000;
    options.tolerance          = 1.e-13;
    LinearAdapter<VectorType, GlobalVector> adapter(options, MPI_COMM_WORLD);
    const auto fields = adapter.add(problem, "poisson");
    auto       state  = adapter.make_state();
    adapter.solve(state);
    problem.set_solution(adapter.field(state, fields.fields().solution));

    FunctionParser<3> exact;
    exact.initialize("x,y,z", "1+x*x+y*y+z*z", {});
    VectorType expected;
    expected.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    VectorTools::interpolate(problem.mapping(),
                             problem.dof_handler(),
                             exact,
                             expected);
    expected -= problem.solution();
    EXPECT_LT(expected.linfty_norm(), 1.e-10);
  }
} // namespace

TEST(LinearAdapter, BOTH_SimplexPoisson)
{
  check_poisson_adapter("gtests/parameters/poisson_adapter_simplex.prm");
}

TEST(LinearAdapter, BOTH_HypercubePoisson)
{
  check_poisson_adapter("gtests/parameters/poisson_adapter_hypercube.prm");
}
