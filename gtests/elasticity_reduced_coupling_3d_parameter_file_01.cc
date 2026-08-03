#include <gtest/gtest.h>

#include "elasticity.h"
#include "elasticity_problem_parameters.h"
#include "utils.h"

#ifdef DEAL_II_WITH_VTK

using namespace dealii;

TEST(ElasticityReducedCoupling, MPI_ThreeDimensionalParameterFile)
{
  ParameterAcceptor::clear();

  ElasticityProblemParameters<3> par;
  ASSERT_NO_THROW(initialize_parameters(
    SOURCE_DIR "/data/tests/elasticity_reduced_coupling_3d.prm"));
  ASSERT_TRUE(par.use_reduced_coupling);

  ElasticityProblem<3> problem(par);
  ASSERT_NO_THROW(problem.run());

  ASSERT_EQ(problem.solution.n_blocks(), 2u);
  EXPECT_NEAR(problem.solution.block(0).linfty_norm(), 1.067e-1, 5.e-3);
}

#endif
