// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <immersx/physics/lifted_poisson.h>

#include <sstream>

#include "test_paths.h"
#ifdef DEAL_II_WITH_VTK
using namespace ImmersX;
using namespace dealii;
TEST(LiftedPoissonApplication, BOTH_SimplexQuadraticSolution)
{
  ParameterAcceptor::clear();
  LiftedPoissonParameters parameters;
  ParameterAcceptor::initialize(
    TestPaths::parameter_path("gtests/parameters/lifted_poisson_simplex.prm"));
  parameters.output.output_directory = TestPaths::output_directory(
    "lifted-poisson-simplex-" +
    std::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)));
  run_lifted_poisson(parameters);
  MPI_Barrier(MPI_COMM_WORLD);
  std::ifstream input(parameters.output.output_directory + "/diagnostics.csv");
  ASSERT_TRUE(input.good());
  std::string line;
  std::getline(input, line);
  ASSERT_TRUE(static_cast<bool>(std::getline(input, line)));
  std::istringstream  stream(line);
  std::vector<double> values;
  std::string         column;
  while (std::getline(stream, column, ','))
    values.push_back(std::stod(column));
  ASSERT_EQ(values.size(), 9u);
  EXPECT_GT(values[1], 0.);
  EXPECT_GT(values[5], 1.);
  EXPECT_LT(values[4], 1.e-9);
  EXPECT_LT(values[6], 1.e-7);
  EXPECT_LT(values[7], 1.e-9);
  EXPECT_LT(values[8], 1.e-8);
}
#endif
