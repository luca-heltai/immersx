#include <gtest/gtest.h>

#include <cmath>

#include "elasticity.h"
#include "test_paths.h"

using namespace dealii;

namespace
{
  void
  configure_tensor_product_parameters(ElasticityProblemParameters<2, 3> &par)
  {
    initialize_parameters();
    ParameterAcceptor::prm.parse_input_from_string(R"(
      subsection Functions
        subsection Right hand side
          set Function expression = 0; 0; 0
        end
        subsection Exact solution
          set Function expression = 0; 0; 0
        end
      end
    )");
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

TEST(ElasticityCouplingIntegration, StaticSolveCompletes)
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
