// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_lib.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/fiber_reinforced_elastodynamics.h>

#include "test_paths.h"
#ifdef DEAL_II_WITH_VTK
using namespace ImmersX;
using namespace dealii;
namespace
{
  template <int dim>
  void
  forced_translation(const bool ida)
  {
    ParameterAcceptor::clear();
    FiberReinforcedElastodynamicsParameters<dim> parameters;
    initialize_parameters(
      TestPaths::parameter_path("gtests/parameters/fiber_reinforced_simplex_" +
                                std::to_string(dim) + "d.prm"));
    FiberReinforcedElastodynamics<dim> driver(parameters);
    driver.setup();
    if (ida)
      driver.run_execution();
    else
      {
        driver.set_initial_conditions();
        driver.advance_one_timestep();
        driver.advance_one_timestep();
      }
    EXPECT_NEAR(driver.current_time(), .02, 1.e-12);
    EXPECT_TRUE(driver.matrix_problem().fe().reference_cell().is_simplex());
    const double expected_u = ida ? 1.0004 : 1.0006;
    const auto   check      = [&](const auto &problem) {
      auto                expected = problem.displacement();
      std::vector<double> data(dim, 0.);
      data[0] = expected_u;
      Functions::ConstantFunction<dim> u(data);
      VectorTools::interpolate(problem.mapping(),
                               problem.dof_handler(),
                               u,
                               expected);
      expected -= problem.displacement();
      EXPECT_LT(expected.linfty_norm(), 1.e-8);
      data[0] = .04;
      Functions::ConstantFunction<dim> v(data);
      VectorTools::interpolate(problem.mapping(),
                               problem.dof_handler(),
                               v,
                               expected);
      expected -= problem.velocity();
      EXPECT_LT(expected.linfty_norm(), 1.e-8);
    };
    check(driver.matrix_problem());
    check(driver.fiber_problem());
    EXPECT_LT(driver.multiplier().l2_norm(), 1.e-8);
  }
} // namespace
TEST(FiberReinforcedElastodynamicsValidation, BOTH_SimplexForcedTranslation)
{
  forced_translation<2>(false);
  forced_translation<3>(false);
}
#  ifdef DEAL_II_WITH_SUNDIALS
TEST(FiberReinforcedElastodynamicsValidation, BOTH_SimplexForcedTranslationIDA)
{
  forced_translation<2>(true);
  forced_translation<3>(true);
}
#  endif
#endif
