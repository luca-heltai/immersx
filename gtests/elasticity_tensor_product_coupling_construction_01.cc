#include <gtest/gtest.h>
#include <immersx/physics/elasticity.h>

using namespace ImmersX;
#include <immersx/physics/elasticity_problem_parameters.h>

using namespace dealii;

TEST(ElasticityCouplingConstruction, BuildsWithTensorProductCoupling)
{
  ParameterAcceptor::clear();
  initialize_parameters();

  ElasticityProblemParameters<2> par;
  par.coupling_type = CouplingType::TensorProduct;
  // minimal settings to make grid generation cheap
  par.domain_type        = "generate";
  par.name_of_grid       = "hyper_cube";
  par.arguments_for_grid = "-1: 1: false";
  par.initial_refinement = 1;

  ElasticityProblem<2> problem(par);

  // Should at least build the grid and FE objects without throwing.
  EXPECT_NO_THROW(problem.make_grid());
  EXPECT_NO_THROW(problem.setup_fe());
}
