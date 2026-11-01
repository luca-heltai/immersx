// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>
using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  std::string
  issues(const GenerationResult &result)
  {
    std::string text;
    for (const auto &issue : result.validation.issues)
      text += issue.message + '\n';
    return text;
  }
  void
  one_cache(const GenerationResult &result)
  {
    std::uint64_t constructions = 0, boundary_builds = 0;
    for (const auto &statistics : result.growth_statistics)
      {
        constructions +=
          statistics.work[GrowthOperation::background_cache_constructions];
        boundary_builds +=
          statistics.work[GrowthOperation::boundary_index_builds];
        EXPECT_EQ(
          statistics.work[GrowthOperation::background_cache_constructions],
          statistics.initialization_work
            [GrowthOperation::background_cache_constructions]);
      }
    EXPECT_EQ(constructions, 1u);
    EXPECT_EQ(boundary_builds, 1u);
  }
} // namespace
TEST(VascularGeneration, BOTH_OneBackgroundContextAcrossArterialAndVenousGrowth)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  bg.refine_global(2);
  Parameters p;
  p.seed_position                  = {.2, .5, .5};
  p.seed_direction                 = {1., 0., 0.};
  p.seed_radius                    = .003;
  p.viscosity                      = .004;
  p.reference_flow                 = 2e-9;
  p.reference_conditions.pressures = {{0, 12000.}, {2, 4000.}};
  p.venous.enabled                 = true;
  p.venous.seed_position           = {.8, .5, .5};
  p.venous.seed_radius             = .004;
  p.tissue_demand                  = "1+x+2*y";
  p.terminal_count                 = 3;
  p.seed_length                    = .2;
  p.venous.terminal_separation     = .04;
  const auto result = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid()) << issues(result);

  EXPECT_EQ(Utilities::MPI::sum(result.terminal_pairs.size(), MPI_COMM_WORLD),
            3u);
  one_cache(result);
}
TEST(VascularGeneration, BOTH_CalibrationReusesBackgroundAcrossAttempts)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  bg.refine_global(1);
  Parameters p;
  p.seed_position                  = {.2, .5, .5};
  p.seed_direction                 = {1., 0., 0.};
  p.seed_length                    = .3;
  p.seed_radius                    = .01;
  p.viscosity                      = .004;
  p.minimum_length                 = 0.;
  p.density_beta                   = 0.;
  p.terminal_count                 = 3;
  p.calibration.minimum_terminals  = 1;
  p.calibration.maximum_terminals  = 5;
  p.calibration.volume_fraction    = numbers::PI * .01 * .01 * .3;
  p.calibration.relative_tolerance = 1e-10;
  const auto result = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid()) << issues(result);
  EXPECT_EQ(result.calibration.status, CalibrationStatus::converged);
  EXPECT_GE(result.calibration.attempts.size(), 2u);
  EXPECT_EQ(Utilities::MPI::sum(result.territories.size(), MPI_COMM_WORLD), 1u);
  double demand = 0.;
  for (const auto &territory : result.territories)
    demand += territory.demand;
  EXPECT_NEAR(Utilities::MPI::sum(demand, MPI_COMM_WORLD), 1., 2e-13);
  one_cache(result);
}
