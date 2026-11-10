// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/config.h>

#include <gtest/gtest.h>
#ifdef DEAL_II_WITH_VTK
#  include <immersx/io/utils.h>
#  include <immersx/physics/lifted_poisson.h>

#  include <filesystem>
#  include <fstream>
#  include <sstream>

#  include "test_paths.h"

using namespace ImmersX;
using namespace dealii;
namespace
{
  std::string
  lifted_robin_input(const std::string &output,
                     const bool         simplex,
                     const std::string &coefficient = "2+x")
  {
    return "subsection Lifted Poisson\nset Refinement cycles = 1\nset Compute errors = true\n"
           "set Dirichlet boundary ids = " +
           std::string(simplex ? "" : "0,2,3,4,5") +
           "\nset Robin boundary ids = " + (simplex ? "0" : "1") +
           "\nset Modal data = " + (simplex ? "1" : "1+x") +
           "\nsubsection Background\nset Grid generator = " +
           (simplex ? TestPaths::data_filename("tests/simplex_cube.msh") :
                      "hyper_cube") +
           "\nset Triangulation type = fullydistributed\nset Initial refinement = " +
           (simplex ? "0" : "2") +
           "\nset Grid generator arguments = 0:1:true\nend\n"
           "subsection Representative\nset Grid generator = " +
           TestPaths::data_filename("tests/simplex_interior_line_3d.vtk") +
           "\nend\nsubsection Background space\nset Finite element = " +
           (simplex ? "FE_SimplexP<3>(1)" : "FE_Q<3>(1)") +
           "\nend\nsubsection Lift\nset Thickness = .1\nsubsection Cross section\nset Maximum basis degree = 0\nset Selected modes = 0\nend\nend\n"
           "subsection Co-refinement\nset Max refinement level = 0\nset Space pre-refinement cycles = 0\nset Space post-refinement cycles = 0\nset Embedded pre-refinement cycles = 0\nset Embedded post-refinement cycles = 0\nend\n"
           "subsection Boundary\nset Function expression = " +
           (simplex ? "1" : "1+x") +
           "\nend\nsubsection Exact solution\nset Function expression = " +
           (simplex ? "1" : "1+x") +
           "\nend\nsubsection Robin coefficient\nset Function expression = " +
           coefficient +
           "\nend\nsubsection Robin boundary\nset Function expression = " +
           (simplex ? "2+x" : "1+(2+x)*(1+x)") +
           "\nend\nsubsection Solver\nset Solver = direct\nend\nsubsection Output\nset Output directory = " +
           output + "\nend\nend\n";
  }
  void
  check_lifted_robin(const bool simplex)
  {
    ParameterAcceptor::clear();
    LiftedPoissonParameters parameters;
    const auto              output = TestPaths::output_directory(
      "lifted-robin-" + std::to_string(simplex) + "-" +
      std::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)));
    initialize_parameters_from_string(lifted_robin_input(output, simplex));
    run_lifted_poisson(parameters);
    std::ifstream input(output + "/diagnostics.csv");
    ASSERT_TRUE(input.good());
    std::string line;
    std::getline(input, line);
    std::getline(input, line);
    std::istringstream  row(line);
    std::vector<double> values;
    while (std::getline(row, line, ','))
      values.push_back(std::stod(line));
    ASSERT_EQ(values.size(), 9u);
    for (const auto value : values)
      EXPECT_TRUE(std::isfinite(value));
    EXPECT_LT(values[4], 1.e-9); // full coupled algebraic residual
    EXPECT_GT(values[5], 1.);    // nonzero bulk solution
    EXPECT_LT(values[6], 1.e-8); // exact multiplier is zero
    EXPECT_LT(values[7], 1.e-9); // independent L2 manufactured error
    EXPECT_LT(values[8], 1.e-8); // independent H1 manufactured error
  }
} // namespace
TEST(LiftedPoisson, BOTH_RobinMixedBoundaryManufacturedApplication)
{
  check_lifted_robin(false);
}
TEST(LiftedPoisson, BOTH_RobinSimplexManufacturedApplication)
{
  check_lifted_robin(true);
}
TEST(LiftedPoisson, BOTH_RobinRejectsInvalidAndOverlappingData)
{
  for (const auto &coefficient : {"x-1.25", "exp(1000*x)"})
    {
      ParameterAcceptor::clear();
      LiftedPoissonParameters parameters;
      initialize_parameters_from_string(
        lifted_robin_input(TestPaths::output_directory("lifted-robin-invalid"),
                           false,
                           coefficient));
      EXPECT_THROW(run_lifted_poisson(parameters), dealii::ExceptionBase);
    }
  ParameterAcceptor::clear();
  LiftedPoissonParameters parameters;
  initialize_parameters_from_string(
    lifted_robin_input(TestPaths::output_directory("lifted-robin-overlap"),
                       false));
  parameters.dirichlet_ids.insert(1);
  EXPECT_THROW(run_lifted_poisson(parameters), dealii::ExceptionBase);
}
#endif
