// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_parser.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>

#include <optional>
#include <string>

#include "test_paths.h"

#ifdef DEAL_II_WITH_SUNDIALS
using namespace dealii;
using namespace ImmersX;

namespace
{
  using Adapter = IDAAdapter<ElastodynamicsSolver<3>::VectorType,
                             ImmersXLA::MPI::BlockVector>;

  // A direct solver must factor the condensed system: the polynomial solution
  // is exact, so the same tolerances check the reduced solve.
  void
  check_transient(const std::string                          &input,
                  const std::optional<Adapter::LinearSolver> &solver = {})
  {
    ParameterAcceptor::clear();
    ElastodynamicsParameters<3> parameters;
    initialize_parameters(TestPaths::parameter_path(input));
    ElastodynamicsSolver<3> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_operators();
    problem.set_initial_conditions();
    using VectorType   = ElastodynamicsSolver<3>::VectorType;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    IDAAdapter<VectorType, GlobalVector> adapter(parameters.time_parameters,
                                                 parameters.ida_parameters,
                                                 MPI_COMM_WORLD);
    const auto                           fields = adapter.add(problem, "solid");
    auto                                 state  = adapter.make_state();
    auto                                 state_dot = adapter.make_state();
    initialize_elastodynamics_adapter_state(
      adapter, fields, problem, state, state_dot);
    if (solver)
      adapter.set_linear_solver(*solver);
    testing::internal::CaptureStdout();
    adapter.solve(state, state_dot);
    const auto output = testing::internal::GetCapturedStdout();
    if (solver && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        EXPECT_NE(output.find("after second-order condensation"),
                  std::string::npos);
      }
    EXPECT_GT(adapter.statistics().steps, 0);
    FunctionParser<3> exact(3);
    exact.initialize("x,y,z,t", "(1+t*t*t)*(1+x*x);0;0", {}, true);
    exact.set_time(parameters.time_parameters.final_time);
    VectorType expected;
    expected.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    VectorTools::interpolate(problem.mapping(),
                             problem.dof_handler(),
                             exact,
                             expected);
    expected -= adapter.field(state, fields.fields().displacement);
    EXPECT_LT(expected.linfty_norm(), 1.e-7);
    exact.initialize("x,y,z,t", "3*t*t*(1+x*x);0;0", {}, true);
    exact.set_time(parameters.time_parameters.final_time);
    VectorTools::interpolate(problem.mapping(),
                             problem.dof_handler(),
                             exact,
                             expected);
    expected -= adapter.field(state, fields.fields().velocity);
    EXPECT_LT(expected.linfty_norm(), 1.e-7);
  }
} // namespace

TEST(ElastodynamicsAdapterMMS, BOTH_SimplexPolynomialInTime)
{
  check_transient("gtests/parameters/elastodynamics_adapter_simplex.prm");
}

TEST(ElastodynamicsAdapterMMS, SimplexPolynomialInTimeUMFPACK)
{
  check_transient("gtests/parameters/elastodynamics_adapter_simplex.prm",
                  Adapter::LinearSolver::umfpack);
}

#  ifdef DEAL_II_WITH_MUMPS
TEST(ElastodynamicsAdapterMMS, BOTH_SimplexPolynomialInTimeMUMPS)
{
  check_transient("gtests/parameters/elastodynamics_adapter_simplex.prm",
                  Adapter::LinearSolver::mumps);
}
#  endif

TEST(ElastodynamicsAdapterMMS, BOTH_HypercubePolynomialInTime)
{
  check_transient("gtests/parameters/elastodynamics_adapter_hypercube.prm");
}
#endif
