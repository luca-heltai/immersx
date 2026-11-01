// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/fiber_reinforced_elasticity_freq.h>

#include "test_paths.h"
using namespace ImmersX;
using namespace dealii;
namespace
{
  template <int dim, int sd>
  void
  harmonic_translation(const bool shared)
  {
    ParameterAcceptor::clear();
    FiberReinforcedElasticityFreqParameters<dim, sd> parameters;
    initialize_parameters(TestPaths::parameter_path(
      "gtests/parameters/fiber_harmonic_simplex_" + std::to_string(dim) +
      std::to_string(sd) + ".prm"));
    parameters.formulation = shared ? "shared" : "independent";
    FiberReinforcedElasticityFreq<dim, sd> driver(parameters);
    driver.setup();
    EXPECT_NEAR(driver.reinforcement_measure(), 1., 1.e-12);
    driver.solve_frequency(.25);
    const double omega = .5 * numbers::PI;
    // Equal acceleration forcing keeps translation exact for a sheet whose
    // integration points cross background-cell interfaces.
    const auto amplitude =
      dim < sd ? 1. / std::complex<double>(-2. * omega * omega, .8 * omega) :
                 1. / std::complex<double>(-5. * omega * omega, 2.9 * omega);
    const auto check = [&](const auto &problem) {
      const auto                          &fe = problem.dof_handler().get_fe();
      std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
      for (const auto &cell : problem.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(indices);
            for (unsigned int i = 0; i < indices.size(); ++i)
              if (problem.locally_owned_dofs().is_element(indices[i]))
                {
                  const auto expected =
                    (fe.system_to_component_index(i).first + 1.) * amplitude;
                  EXPECT_NEAR(problem.real_displacement()[indices[i]],
                              expected.real(),
                              1.e-9);
                  EXPECT_NEAR(problem.imaginary_displacement()[indices[i]],
                              expected.imag(),
                              1.e-9);
                }
          }
    };
    check(driver.matrix_problem());
    if (!shared)
      {
        check(driver.fiber_problem());
        const double multiplier =
          std::hypot(driver.real_multiplier().l2_norm(),
                     driver.imaginary_multiplier().l2_norm());
        if constexpr (dim < sd)
          EXPECT_LT(multiplier, 1.e-8);
        else
          EXPECT_GT(multiplier, 1.e-5);
        EXPECT_LT(driver.continuity_norm(), 1.e-9);
      }
    EXPECT_LT(driver.residual_norm(), 1.e-9);
  }
} // namespace
TEST(FiberReinforcedElasticityFreq, BOTH_SimplexVolumeTranslation)
{
  harmonic_translation<2, 2>(false);
  harmonic_translation<3, 3>(false);
}
TEST(FiberReinforcedElasticityFreq, BOTH_SimplexSheetTranslation)
{
  harmonic_translation<2, 3>(false);
}
TEST(FiberReinforcedElasticityFreq, BOTH_SimplexSharedTranslation)
{
  harmonic_translation<2, 2>(true);
  harmonic_translation<2, 3>(true);
  harmonic_translation<3, 3>(true);
}
