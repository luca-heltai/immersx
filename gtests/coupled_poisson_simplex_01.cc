// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_parser.h>

#include <gtest/gtest.h>
#include <immersx/core/constraint.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>

#include "test_paths.h"
#ifdef DEAL_II_WITH_VTK
using namespace ImmersX;
using namespace dealii;
TEST(CoupledPoisson, BOTH_SimplexQuadraticSolution)
{
  ParameterAcceptor::clear();
  PoissonParameters<2>    bulk_parameters;
  PoissonParameters<1, 2> embedded_parameters("/Embedded Poisson/");
  // The fixture also carries executable-only output parameters.
  class ApplicationParameters : public ParameterAcceptor
  {
  public:
    ApplicationParameters()
      : ParameterAcceptor("/Coupled Poisson/")
    {
      add_parameter("Output directory", output);
    }
    std::string output;
  } application;
  FiniteElementSpaceParameters<1, 2> multiplier_parameters(
    "/Coupled Poisson/Multiplier space/", "FE_DGQ<1>(0)");
  LinearSolverParameters solver_parameters;
  initialize_parameters(
    TestPaths::parameter_path("gtests/parameters/coupled_poisson_simplex.prm"));
  PoissonSolver<2>    bulk(bulk_parameters);
  PoissonSolver<1, 2> embedded(embedded_parameters);
  bulk.setup();
  embedded.setup();
  EXPECT_TRUE(bulk.dof_handler().get_fe().reference_cell().is_simplex());
  LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector> adapter(
    solver_parameters, MPI_COMM_WORLD);
  const auto registered_bulk     = adapter.add(bulk, "bulk");
  const auto registered_embedded = adapter.add(embedded, "embedded");
  const auto u                   = registered_bulk.fields().solution;
  const auto v                   = registered_embedded.fields().solution;
  FiniteElementSpace<1, 2> multiplier_space(embedded.triangulation(),
                                            multiplier_parameters);
  const auto               multiplier_view = multiplier_space.view();
  const auto               lambda          = multiplier_view.field("lambda");
  const auto coupling = adapter.add(make_continuity_constraint(u, v, lambda));
  auto       state    = adapter.make_state();
  adapter.solve(state);
  auto residual = adapter.make_state();
  adapter.evaluate_residual(state, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-9);
  EXPECT_LT(adapter.field(state, coupling.fields().multiplier).l2_norm(),
            1.e-8);
  FunctionParser<2> exact;
  exact.initialize("x,y", "1+x*x", {});
  const auto check = [&](const auto &field) {
    auto physical = adapter.field(state, field);
    field.constraints().distribute(physical);
    auto expected = physical;
    VectorTools::interpolate(field.mapping(),
                             field.dof_handler(),
                             exact,
                             expected);
    expected -= physical;
    EXPECT_LT(expected.linfty_norm(), 1.e-9);
  };
  check(u);
  check(v);
}
#endif
