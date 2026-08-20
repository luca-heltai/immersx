#include <gtest/gtest.h>
#include <immersx/physics/elasticity_problem_parameters.h>

using namespace ImmersX;
#include <immersx/io/utils.h>

#include "test_paths.h"

using namespace dealii;

TEST(ElasticityCouplingParameters, ParseParameterFile)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 2> par;

  EXPECT_NO_THROW(initialize_parameters(ImmersX::TestPaths::data_filename(
    "tests/elasticity_tensor_product_coupling_2d.prm")));
  EXPECT_EQ(par.coupling_type, CouplingType::TensorProduct);
  EXPECT_EQ(
    par.tensor_product_coupling_parameters.coupling_rhs_expressions.size(), 2u);
  EXPECT_FALSE(par.tensor_product_coupling_parameters
                 .tensor_product_space_parameters.reduced_grid_name.empty());
}
