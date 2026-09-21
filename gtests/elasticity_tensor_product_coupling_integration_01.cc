#include <gtest/gtest.h>
#include <immersx/physics/elasticity.h>

#include <cmath>
#include <filesystem>

using namespace ImmersX;
#include "test_paths.h"

using namespace dealii;

namespace
{
  void
  configure_tensor_product_parameters(
    ElasticityProblemParameters<2, 3> &par,
    const std::string &dirichlet_expression = "0; 0; 0.01*sin(2*pi*50*t)")
  {
    initialize_parameters();
    ParameterAcceptor::prm.parse_input_from_string(
      "subsection Functions\n"
      "  subsection Dirichlet boundary conditions\n"
      "    set Function expression = " +
      dirichlet_expression +
      "\n"
      "  end\n"
      "  subsection Right hand side\n"
      "    set Function expression = 0; 0; 0\n"
      "  end\n"
      "  subsection Exact solution\n"
      "    set Function expression = 0; 0; 0\n"
      "  end\n"
      "end\n");
    ParameterAcceptor::parse_all_parameters();

    par.coupling_type                           = CouplingType::TensorProduct;
    par.domain_type                             = "generate";
    par.name_of_grid                            = "hyper_cube";
    par.arguments_for_grid                      = "-1: 1: false";
    par.initial_refinement                      = 1;
    par.default_material_properties.Lame_mu     = 1;
    par.default_material_properties.Lame_lambda = 1;
    par.dirichlet_ids                           = {0, 1, 2, 3, 4, 5};
    par.tensor_product_coupling_parameters.tensor_product_space_parameters
      .reduced_grid_name =
      ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
    par.tensor_product_coupling_parameters.coupling_rhs_expressions = {"1",
                                                                       "0",
                                                                       "0"};
  }
} // namespace

TEST(ElasticityCouplingIntegration, SetupCreatesTensorProductMultiplierBlock)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  configure_tensor_product_parameters(par);
  ElasticityProblem<2, 3> problem(par);

  problem.make_grid();
  problem.setup_fe();
  ASSERT_NO_THROW(problem.setup_dofs());
  ASSERT_TRUE(problem.tensor_product_coupling != nullptr);
  EXPECT_GT(problem.solution.block(1).size(), 0);
  EXPECT_EQ(problem.solution.n_blocks(), 2u);
}

TEST(ElasticityCouplingIntegration, AssemblyProducesTensorProductRhs)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  configure_tensor_product_parameters(par);
  ElasticityProblem<2, 3> problem(par);

  problem.make_grid();
  problem.setup_fe();
  problem.setup_dofs();
  problem.assemble_elasticity_system();
  problem.assemble_coupling();

  EXPECT_GT(problem.system_rhs.block(1).l2_norm(), 0.0);
}

TEST(ElasticityCouplingIntegrationValidation, StaticSolveCompletes)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  configure_tensor_product_parameters(par);
  ElasticityProblem<2, 3> problem(par);

  problem.make_grid();
  problem.setup_fe();
  problem.setup_dofs();
  problem.assemble_elasticity_system();
  problem.assemble_coupling();
  ASSERT_NO_THROW(problem.solve_static());

  EXPECT_TRUE(std::isfinite(problem.solution.block(0).l2_norm()));
  EXPECT_TRUE(std::isfinite(problem.solution.block(1).l2_norm()));
}

TEST(ElasticityCouplingIntegrationValidation,
     DynamicStrongConstraintsRemainFinite)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  configure_tensor_product_parameters(par);

  par.output_directory =
    TestPaths::output_directory("elasticity-issue-203-diagnostics");
  std::filesystem::create_directories(par.output_directory);
  par.time_parameters.initial_time                                = 0.;
  par.time_parameters.final_time                                  = 2.e-3;
  par.time_parameters.time_step                                   = 1.e-3;
  par.default_material_properties.rho                             = 1.;
  par.tensor_product_coupling_parameters.coupling_rhs_expressions = {
    "sin(2*pi*t)", "sin(2*pi*t)", "sin(2*pi*t)"};
  par.check_model_consistency();

  ElasticityProblem<2, 3> problem(par);
  ASSERT_NO_THROW(problem.run());

  EXPECT_TRUE(std::isfinite(problem.solution.block(0).l2_norm()));
  EXPECT_TRUE(std::isfinite(problem.solution.block(1).l2_norm()));
  EXPECT_LT(problem.solution.block(0).linfty_norm(), 1.e3);
}


TEST(ElasticityCouplingIntegrationValidation,
     BOTH_DynamicStrongConstraintsWithLocalRefinementRemainFinite)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  configure_tensor_product_parameters(par, "0; 0; 0");

  par.output_directory =
    TestPaths::output_directory("elasticity-issue-203-small");
  std::filesystem::create_directories(par.output_directory);
  par.time_parameters.initial_time    = 0.;
  par.time_parameters.final_time      = 2.e-2;
  par.time_parameters.time_step       = 1.e-3;
  par.default_material_properties.rho = 1.;
  par.tensor_product_coupling_parameters.refinement_parameters
    .refinement_strategy = "space";
  // On this small fixture, level two is the smallest cap that creates a
  // locally refined background mesh and therefore hanging-node constraints.
  par.tensor_product_coupling_parameters.refinement_parameters
    .max_refinement_level                                         = 2;
  par.tensor_product_coupling_parameters.coupling_rhs_expressions = {
    "sin(2*pi*t)", "sin(2*pi*t)", "sin(2*pi*t)"};
  par.check_model_consistency();

  ElasticityProblem<2, 3> problem(par);
  ASSERT_NO_THROW(problem.run());

  EXPECT_TRUE(std::isfinite(problem.solution.block(0).l2_norm()));
  EXPECT_TRUE(std::isfinite(problem.solution.block(1).l2_norm()));
  EXPECT_GT(problem.solution.block(0).linfty_norm(), 1.e-12);
  EXPECT_LT(problem.solution.block(0).linfty_norm(), 1.e3);
}
