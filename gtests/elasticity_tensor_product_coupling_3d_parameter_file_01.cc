#include <gtest/gtest.h>
#include <immersx/physics/elasticity.h>

using namespace ImmersX;
#include <immersx/io/utils.h>
#include <immersx/physics/elasticity_problem_parameters.h>

#include "test_paths.h"

#ifdef DEAL_II_WITH_VTK

using namespace dealii;

TEST(ElasticityCoupling, MPI_ThreeDimensionalParameterFile)
{
  ParameterAcceptor::clear();

  ElasticityProblemParameters<3> par;
  ASSERT_NO_THROW(initialize_parameters(ImmersX::TestPaths::data_filename(
    "tests/elasticity_tensor_product_coupling_3d.prm")));
  ASSERT_EQ(par.coupling_type, CouplingType::TensorProduct);

  ElasticityProblem<3> problem(par);
  ASSERT_NO_THROW(problem.run());

  ASSERT_EQ(problem.solution.n_blocks(), 2u);
  EXPECT_NEAR(problem.solution.block(0).linfty_norm(), 1.067e-1, 5.e-3);
}

#endif
