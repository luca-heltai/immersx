#include <gtest/gtest.h>
#include <immersx/coupling/inclusions.h>
#include <immersx/io/utils.h>
#include <immersx/physics/coupled_elasticity.h>

#include <filesystem>

#include "test_paths.h"

using namespace ImmersX;
using namespace dealii;

namespace
{
  // Initialize point inclusions along x with the given vessel ids and return
  // the number of vessels found by check_vessels().
  unsigned int
  n_vessels_of_inclusions(const std::vector<unsigned int> &vessel_ids)
  {
    Inclusions<3> inclusions;
    inclusions.set_n_q_points(4);
    inclusions.set_n_coefficients(1);
    // cx, cy, cz, dx, dy, dz, r, vessel id
    for (unsigned int i = 0; i < vessel_ids.size(); ++i)
      inclusions.inclusions.push_back(
        {0.1 * i, 0, 0, 1, 0, 0, 0.05, double(vessel_ids[i])});
    inclusions.initialize();
    return inclusions.get_n_vessels();
  }
} // namespace

TEST(Inclusion3, CheckVesselsConsecutiveIds)
{
  EXPECT_EQ(n_vessels_of_inclusions({0, 0, 1, 1, 2}), 3u);
}

TEST(Inclusion3, CheckVesselsUnorderedIds)
{
  EXPECT_EQ(n_vessels_of_inclusions({2, 0, 1, 0}), 3u);
}

TEST(Inclusion3, CheckVesselsRejectsGap)
{
  EXPECT_ANY_THROW(n_vessels_of_inclusions({0, 2}));
}

TEST(Inclusion3, CheckVesselsRejectsMissingZero)
{
  EXPECT_ANY_THROW(n_vessels_of_inclusions({1, 2}));
}

// CoupledElasticityProblem, run as app_coupled_elasticity does without a 1D
// model, must give the same solution as ElasticityProblem::run() on the same
// input file.
TEST(CoupledElasticity, BOTH_SameSolutionAsElasticity)
{
  const auto parameter_file = TestPaths::parameter_path(
    "gtests/parameters/coupled_elasticity_vs_elasticity.prm");
  ASSERT_TRUE(std::filesystem::is_regular_file(parameter_file));

  // l2 norms of the displacement and of the multiplier.
  std::vector<double> elasticity;
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<3> par;
    ElasticityProblem<3>           problem(par);
    initialize_parameters(parameter_file);
    problem.run();
    elasticity = {problem.solution.block(0).l2_norm(),
                  problem.solution.block(1).l2_norm()};
  }

  std::vector<double> coupled;
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<3> par;
    CoupledElasticityProblem<3>    problem(par);
    initialize_parameters(parameter_file);
    problem.run_timestep0();
    problem.run_timestep();
    coupled = {problem.solution.block(0).l2_norm(),
               problem.solution.block(1).l2_norm()};
  }

  const std::vector<std::string> names = {"displacement", "multiplier"};
  for (unsigned int i = 0; i < names.size(); ++i)
    {
      ASSERT_GT(elasticity[i], 0.0) << names[i];
      EXPECT_NEAR(coupled[i], elasticity[i], 1e-8 * elasticity[i]) << names[i];
    }
}
