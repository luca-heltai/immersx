#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "inclusions.h"

using namespace dealii;

TEST(TestInclusionsBasis2, CheckIntegrals) // NOLINT
{
  // cx, cy, r
  Inclusions<2> ref;
  ref.n_q_points         = 100;
  ref.n_coefficients     = 3;
  ref.coefficient_offset = 0;
  ref.inclusions.push_back({{0, 0, 1.0}});
  ref.initialize();
  double integral[4] = {0, 0, 0, 0};
  for (unsigned int q = 0; q < ref.n_particles(); ++q)
    {
      const auto &fe_values = ref.get_fe_values(q);
      const auto &ds        = ref.get_JxW(q);
      integral[0] += ds;
      integral[1] += fe_values[0] * ds;
      integral[2] += fe_values[1] * ds;
      integral[3] += fe_values[2] * ds;
    }

  ASSERT_NEAR(integral[0], 2 * numbers::PI, 1e-10);
  ASSERT_NEAR(integral[1], 2 * numbers::PI, 1e-10);
  ASSERT_NEAR(integral[2], 0, 1e-10);
  ASSERT_NEAR(integral[3], 0, 1e-10);
}

TEST(TestInclusionsBasis2, CheckScaling) // NOLINT
{
  // cx, cy, r
  Inclusions<2> ref;
  double        radius   = 9;
  ref.n_q_points         = 100;
  ref.n_coefficients     = 13;
  ref.coefficient_offset = 0;
  ref.inclusions.push_back({{0, 0, radius}});
  ref.initialize();
  std::vector<double> integral(ref.n_coefficients + 1, 0.0);
  for (unsigned int q = 0; q < ref.n_particles(); ++q)
    {
      const auto &fe_values = ref.get_fe_values(q);
      const auto &ds        = ref.get_JxW(q);
      integral[0] += ds;
      for (unsigned int i = 0; i < ref.n_coefficients; ++i)
        integral[i + 1] += fe_values[i] * fe_values[i] * ds;
    }
  // Take the square root of the integral of the basis functions
  // and check that they are equal to sqrt(|D|) where |D| is the length of the
  // inclusion
  auto D = 2 * numbers::PI * radius;
  for (unsigned int i = 0; i < ref.n_coefficients + 1; ++i)
    {
      EXPECT_NEAR(integral[i], D, 1e-10);
    }
}
