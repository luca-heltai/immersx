#include <gtest/gtest.h>
#include <immersx/physics/elasticity.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace ImmersX;
#include "test_paths.h"

using namespace dealii;

namespace
{
  // Names of the point data arrays of a VTU file.
  std::vector<std::string>
  data_array_names(const std::string &filename)
  {
    std::ifstream            file(filename);
    std::stringstream        buffer;
    std::vector<std::string> names;
    buffer << file.rdbuf();
    const std::string text = buffer.str();
    const std::string key  = "<DataArray";
    for (auto pos = text.find(key); pos != std::string::npos;
         pos      = text.find(key, pos + 1))
      {
        const auto end   = text.find('>', pos);
        const auto name  = text.find("Name=\"", pos);
        const auto close = text.find('"', name + 6);
        if (name < end)
          names.push_back(text.substr(name + 6, close - name - 6));
      }
    return names;
  }

  bool
  has(const std::vector<std::string> &names, const std::string &name)
  {
    return std::find(names.begin(), names.end(), name) != names.end();
  }
} // namespace

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
  ASSERT_NO_THROW(problem.output_wall_particles(filename));
  EXPECT_TRUE(std::filesystem::exists(filename));
  const auto names = data_array_names(filename);
  EXPECT_TRUE(has(names, "id"));
  EXPECT_TRUE(has(names, "lambda"));
  // ElasticityProblem computes no wall pressure.
  EXPECT_FALSE(has(names, "pressure_at_immersed_surface"));
}

TEST(ElasticityCouplingParticleOutput, EmptyPropertyListWritesNoFile)
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
  par.output_name = "elasticity_tensor_product_no_particles";
  initialize_parameters();
  ParameterAcceptor::parse_all_parameters();
  par.coupling_type = CouplingType::TensorProduct;
  par.tensor_product_coupling_parameters.tensor_product_space_parameters
    .reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
  par.tensor_product_coupling_parameters.coupling_rhs_expressions = {"1",
                                                                     "0",
                                                                     "0"};
  par.wall_particle_output.properties.clear();

  ElasticityProblem<2, 3> problem(par);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_dofs();

  std::filesystem::create_directories(par.output_directory);
  const std::string filename =
    par.output_directory + "/" + par.output_name + "_particles.vtu";
  std::filesystem::remove(filename);
  MPI_Barrier(MPI_COMM_WORLD);
  ASSERT_NO_THROW(problem.output_wall_particles(filename));
  EXPECT_FALSE(std::filesystem::exists(filename));
}

TEST(ElasticityCouplingParticleOutput, BOTH_SolvedStateCarriesMultiplierAndWall)
{
  ParameterAcceptor::clear();

  ElasticityProblemParameters<2, 3> par;
  par.coupling_type      = CouplingType::TensorProduct;
  par.domain_type        = "generate";
  par.name_of_grid       = "hyper_cube";
  par.arguments_for_grid = "-1: 1: false";
  par.initial_refinement = 2;
  par.output_directory =
    ImmersX::TestPaths::output_directory("elasticity-tensor-product-particles");
  par.output_name = "elasticity_tensor_product_solved_particles";
  initialize_parameters();
  ParameterAcceptor::parse_all_parameters();
  par.coupling_type                           = CouplingType::TensorProduct;
  par.default_material_properties.Lame_mu     = 1;
  par.default_material_properties.Lame_lambda = 1;
  par.dirichlet_ids                           = {0, 1, 2, 3, 4, 5};
  par.tensor_product_coupling_parameters.tensor_product_space_parameters
    .reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
  par.tensor_product_coupling_parameters.coupling_rhs_expressions = {"1",
                                                                     "0",
                                                                     "0"};
  par.wall_particle_output.properties = {"lifted_value",
                                         "modal_coefficients",
                                         "bulk_field",
                                         "pressure_at_immersed_surface"};

  ElasticityProblem<2, 3> problem(par);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_dofs();
  problem.assemble_elasticity_system();
  problem.assemble_coupling();
  ASSERT_NO_THROW(problem.solve_static());

  std::filesystem::create_directories(par.output_directory);
  const std::string filename =
    par.output_directory + "/" + par.output_name + "_particles.vtu";

  // Without a computed pressure the column is omitted.
  ASSERT_NO_THROW(problem.output_wall_particles(filename));
  auto names = data_array_names(filename);
  EXPECT_TRUE(has(names, "lambda"));
  EXPECT_TRUE(has(names, "lambda_0"));
  EXPECT_TRUE(has(names, "displacement"));
  EXPECT_FALSE(has(names, "pressure_at_immersed_surface"));

  // With one, the retained particles are reused with an extra column.
  problem.vessel_pressure_at_wall_points.assign(
    problem.tensor_product_coupling->get_locally_owned_qpoints().size(), 1.);
  problem.has_vessel_pressure_at_wall_points = true;
  ASSERT_NO_THROW(problem.output_wall_particles(filename));
  names = data_array_names(filename);
  EXPECT_TRUE(has(names, "lambda"));
  EXPECT_TRUE(has(names, "pressure_at_immersed_surface"));
}
