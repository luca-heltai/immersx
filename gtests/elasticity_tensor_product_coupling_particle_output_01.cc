#include <gtest/gtest.h>
#include <immersx/physics/elasticity.h>

#include <filesystem>

using namespace ImmersX;
#include "test_paths.h"

using namespace dealii;

TEST(ElasticityCouplingParticleOutput, WritesTensorProductParticles)
{
  ParameterAcceptor::clear();

  ElasticityProblemParameters<2, 3> par;
  par.coupling_type      = CouplingType::TensorProduct;
  par.domain_type        = "generate";
  par.name_of_grid       = "hyper_cube";
  par.arguments_for_grid = "-1: 1: false";
  par.initial_refinement = 1;
  par.output_directory =
    ImmersX::TestPaths::output_directory("elasticity-tensor-product-particles");
  par.output_name = "elasticity_tensor_product_particles";
  par.tensor_product_coupling_parameters.tensor_product_space_parameters
    .reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
  par.tensor_product_coupling_parameters.coupling_rhs_expressions = {"1",
                                                                     "0",
                                                                     "0"};

  initialize_parameters();
  ParameterAcceptor::parse_all_parameters();

  // Parsing applies registered defaults; set the test-specific values after it.
  par.coupling_type = CouplingType::TensorProduct;
  par.tensor_product_coupling_parameters.tensor_product_space_parameters
    .reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
  par.tensor_product_coupling_parameters.coupling_rhs_expressions = {"1",
                                                                     "0",
                                                                     "0"};

  ElasticityProblem<2, 3> problem(par);
  problem.make_grid();
  problem.setup_fe();
  ASSERT_TRUE(problem.uses_tensor_product_coupling());
  ASSERT_NO_THROW(problem.setup_dofs());
  ASSERT_TRUE(problem.tensor_product_coupling != nullptr);

  std::filesystem::create_directories(par.output_directory);
  const std::string filename =
    par.output_directory + "/" + par.output_name + "_particles.vtu";
  ASSERT_NO_THROW(problem.output_immersed_particles(filename));
  EXPECT_TRUE(std::filesystem::exists(filename));
}
