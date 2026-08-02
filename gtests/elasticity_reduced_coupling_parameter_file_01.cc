#include <gtest/gtest.h>

#include "elasticity_problem_parameters.h"
#include "utils.h"

using namespace dealii;

TEST(ElasticityReducedCouplingParameters, ParseParameterFile)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;

  EXPECT_NO_THROW(initialize_parameters(
    SOURCE_DIR "/data/tests/elasticity_reduced_coupling_2d.prm"));
  EXPECT_TRUE(par.use_reduced_coupling);
  EXPECT_EQ(par.reduced_coupling_parameters.coupling_rhs_expressions.size(),
            3u);
  EXPECT_FALSE(par.reduced_coupling_parameters.tensor_product_space_parameters
                 .reduced_grid_name.empty());
}
