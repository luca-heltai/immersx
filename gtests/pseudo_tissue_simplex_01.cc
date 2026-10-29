// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/coupled_elasticity.h>

#include "test_paths.h"
using namespace ImmersX;
using namespace dealii;
TEST(ElasticityValidation, BOTH_SimplexVesselwisePseudoExchange)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> parameters;
  CoupledElasticityProblem<3>    problem(parameters);
  initialize_parameters(
    TestPaths::parameter_path("gtests/parameters/pseudocoupling_simplex.prm"));
  problem.run_timestep0();
  problem.update_inclusions_data({.001});
  problem.run_timestep();
  auto displacement = problem.solution.block(0);
  auto multiplier   = problem.solution.block(1);
  auto pressure     = problem.coupling_pressure;
  EXPECT_GT(displacement.l2_norm(), 1.e-6);
  EXPECT_GT(pressure.l2_norm(), 1.e-6);
  problem.update_inclusions_data({.002});
  problem.run_timestep();
  displacement *= 2.;
  multiplier *= 2.;
  pressure *= 2.;
  displacement -= problem.solution.block(0);
  multiplier -= problem.solution.block(1);
  pressure -= problem.coupling_pressure;
  EXPECT_LT(displacement.linfty_norm(), 1.e-9);
  EXPECT_LT(multiplier.linfty_norm(), 1.e-9);
  EXPECT_LT(pressure.linfty_norm(), 1.e-9);
}
