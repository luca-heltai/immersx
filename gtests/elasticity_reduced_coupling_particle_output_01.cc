#include <gtest/gtest.h>

#include <filesystem>

#include "elasticity.h"

using namespace dealii;

TEST(ElasticityReducedCouplingParticleOutput, WritesReducedParticles)
{
  ParameterAcceptor::clear();

  ElasticityProblemParameters<2, 3> par;
  par.use_reduced_coupling = true;
  par.domain_type          = "generate";
  par.name_of_grid         = "hyper_cube";
  par.arguments_for_grid   = "-1: 1: false";
  par.initial_refinement   = 1;
  par.output_directory     = "tests_debug_output";
  par.output_name          = "elasticity_reduced_particles";
  par.reduced_coupling_parameters.tensor_product_space_parameters
    .reduced_grid_name = SOURCE_DIR "/data/tests/one_cylinder.vtk";
  par.reduced_coupling_parameters.coupling_rhs_expressions = {"1", "0", "0"};

  initialize_parameters();
  ParameterAcceptor::parse_all_parameters();

  // Parsing applies registered defaults; set the test-specific values after it.
  par.use_reduced_coupling = true;
  par.reduced_coupling_parameters.tensor_product_space_parameters
    .reduced_grid_name = SOURCE_DIR "/data/tests/one_cylinder.vtk";
  par.reduced_coupling_parameters.coupling_rhs_expressions = {"1", "0", "0"};

  ElasticityProblem<2, 3> problem(par);
  problem.make_grid();
  problem.setup_fe();
  ASSERT_TRUE(problem.using_reduced_coupling());
  ASSERT_NO_THROW(problem.setup_dofs());
  ASSERT_TRUE(problem.reduced_coupling != nullptr);

  std::filesystem::create_directories(par.output_directory);
  const std::string filename =
    par.output_directory + "/" + par.output_name + "_particles.vtu";
  ASSERT_NO_THROW(problem.output_immersed_particles(filename));
  EXPECT_TRUE(std::filesystem::exists(filename));
}
