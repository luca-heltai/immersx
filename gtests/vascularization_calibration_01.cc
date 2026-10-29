// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <cmath>
#include <limits>
#include <set>

using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  void
  synthetic(Parameters &p)
  {
    p.seed_position                 = Point<3>(.2, .5, .5);
    p.seed_direction                = Point<3>(1., 0., 0.);
    p.seed_radius                   = .01;
    p.seed_length                   = .3;
    p.viscosity                     = .004;
    p.calibration.minimum_terminals = 1;
    p.calibration.maximum_terminals = 5;
  }
  double
  seed_volume()
  {
    return numbers::PI * .01 * .01 * .3;
  }
} // namespace

TEST(VascularCalibration, FindsAnalyticSeedVolumeWithoutChangingParameters)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.terminal_count                 = 3;
  p.calibration.volume_fraction    = seed_volume();
  p.calibration.relative_tolerance = 1.e-10;
  p.coverage.service_radius        = 2.;
  p.calibration.minimum_coverage   = 1.;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_EQ(result.calibration.status, CalibrationStatus::converged);
  ASSERT_EQ(result.calibration.attempts.size(), 2u);
  EXPECT_EQ(result.calibration.attempts[0].terminal_count, 3u);
  EXPECT_EQ(result.calibration.attempts[1].terminal_count, 1u);
  EXPECT_EQ(result.territories.size(), 1u);
  EXPECT_NEAR(result.tissue.volume_fraction / seed_volume(), 1., 1.e-13);
  EXPECT_NEAR(result.tissue.arterial_coverage, 1., 1.e-13);
  EXPECT_EQ(p.terminal_count, 3u);
  EXPECT_DOUBLE_EQ(p.seed_radius, .01);
  EXPECT_EQ(result.calibration.sampled_points,
            result.calibration.attempts[0].sampled_points);
}

TEST(VascularCalibration, ReachesBranchedVolumeTargetFromDifferentInitialCount)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.terminal_count = 8;
  const auto reference =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(reference.validation.valid());
  double target = 0.;
  for (const auto &[id, e] : reference.network.vessels)
    target +=
      numbers::PI * e.radii[0] * e.radii[0] *
      reference.network.nodes.at(e.nodes[0])
        .position.distance(reference.network.nodes.at(e.nodes[1]).position);
  p.terminal_count                = 2;
  p.calibration.maximum_terminals = 16;
  p.calibration.volume_fraction   = target;
  const auto calibrated =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(calibrated.validation.valid());
  EXPECT_EQ(calibrated.calibration.status, CalibrationStatus::converged);
  EXPECT_LE(std::abs(calibrated.tissue.volume_fraction / target - 1.), .05);
  EXPECT_NE(calibrated.territories.size(), 2u);
  EXPECT_LE(calibrated.calibration.attempts.size(), 12u);
  double achieved = 0.;
  for (const auto &[id, e] : calibrated.network.vessels)
    achieved +=
      numbers::PI * e.radii[0] * e.radii[0] *
      calibrated.network.nodes.at(e.nodes[0])
        .position.distance(calibrated.network.nodes.at(e.nodes[1]).position);
  EXPECT_NEAR(calibrated.tissue.volume_fraction / achieved, 1., 1.e-13);
}

TEST(VascularCalibration, NonmonotoneSearchDoesNotPruneUntestedCounts)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.terminal_count              = 3;
  p.calibration.volume_fraction = 1.e-10;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_EQ(result.calibration.status, CalibrationStatus::exhausted);
  std::set<unsigned int> counts;
  std::uint64_t          samples = 0;
  for (const auto &trial : result.calibration.attempts)
    {
      EXPECT_TRUE(trial.generation_validation.valid());
      EXPECT_GT(trial.tissue.volume_fraction,
                p.calibration.volume_fraction *
                  (1. + p.calibration.relative_tolerance));
      EXPECT_TRUE(counts.insert(trial.terminal_count).second);
      samples += trial.sampled_points;
    }
  EXPECT_EQ(counts, (std::set<unsigned int>{1, 2, 3, 4, 5}));
  EXPECT_EQ(result.calibration.sampled_points, samples);
  EXPECT_TRUE(validate_data(result.network, 1).valid());
}

TEST(VascularCalibration, BudgetExhaustionCannotBeReportedAsSuccess)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.terminal_count                 = 2;
  p.calibration.volume_fraction    = seed_volume();
  p.calibration.relative_tolerance = 1.e-10;
  p.calibration.attempt_budget     = 1;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_EQ(result.calibration.status, CalibrationStatus::exhausted);
  ASSERT_EQ(result.calibration.attempts.size(), 1u);
  EXPECT_EQ(result.calibration.attempts[0].terminal_count, 2u);
  EXPECT_EQ(result.territories.size(), 2u);
  EXPECT_GT(std::abs(result.tissue.volume_fraction / seed_volume() - 1.),
            p.calibration.relative_tolerance);
}

TEST(VascularCalibration, FixedImportedInsideVolumeProvesIncompatibility)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  NetworkData initial;
  initial.nodes[0].position = Point<3>(-.2, .5, .5);
  initial.nodes[0].port     = Port::arterial_inlet;
  initial.nodes[1].position = Point<3>(.3, .5, .5);
  Vessel e;
  e.nodes              = {{0, 1}};
  e.radii              = {{.01, .01}};
  e.viscosity          = .004;
  initial.vessels[4]   = e;
  initial.terminals[7] = {1, invalid_id, System::arterial, 0, 1.};
  Parameters p;
  synthetic(p);
  p.calibration.volume_fraction = seed_volume() / 2.;
  const auto result =
    generate(background, MappingQ1<3>(), initial, p, MPI_COMM_WORLD);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_EQ(result.calibration.status, CalibrationStatus::incompatible);
  EXPECT_TRUE(result.calibration.attempts.empty());
  EXPECT_EQ(result.sampled_points, 0u);
  EXPECT_NEAR(result.tissue.inside_vessel_volume / seed_volume(), 1., 1.e-13);
  EXPECT_EQ(result.network.nodes.size(), 2u);
  EXPECT_EQ(result.network.vessels.size(), 1u);
  EXPECT_EQ(result.network.vessels.at(4).radii, e.radii);
  EXPECT_EQ(initial.terminals.at(7).node, 1u);
  EXPECT_EQ(initial.nodes.at(0).position, Point<3>(-.2, .5, .5));
}

TEST(VascularCalibration, VolumeMatchDoesNotOverrideMissingCoverage)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.calibration.maximum_terminals = 1;
  p.calibration.volume_fraction   = seed_volume();
  p.calibration.minimum_coverage  = .5;
  p.coverage.service_radius       = .001;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_EQ(result.calibration.status, CalibrationStatus::exhausted);
  EXPECT_NEAR(result.tissue.volume_fraction / seed_volume(), 1., 1.e-13);
  EXPECT_DOUBLE_EQ(result.tissue.arterial_coverage, 0.);
  EXPECT_EQ(result.calibration.attempts.size(), 1u);
}

TEST(VascularCalibration, TrialsAreReproducibleAndRespectCountBounds)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.terminal_count                = 100;
  p.calibration.minimum_terminals = 2;
  p.calibration.maximum_terminals = 4;
  p.calibration.volume_fraction   = 1.e-10;
  const auto a = generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  const auto b = generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_EQ(a.calibration.attempts.size(), 3u);
  ASSERT_EQ(a.calibration.attempts.size(), b.calibration.attempts.size());
  EXPECT_EQ(a.calibration.attempts[0].terminal_count, 4u);
  for (unsigned int i = 0; i < a.calibration.attempts.size(); ++i)
    {
      const auto &x = a.calibration.attempts[i], &y = b.calibration.attempts[i];
      EXPECT_GE(x.terminal_count, 2u);
      EXPECT_LE(x.terminal_count, 4u);
      EXPECT_EQ(x.terminal_count, y.terminal_count);
      EXPECT_DOUBLE_EQ(x.tissue.volume_fraction, y.tissue.volume_fraction);
      EXPECT_EQ(x.sampled_points, y.sampled_points);
    }
  for (const auto &[id, e] : a.network.vessels)
    EXPECT_EQ(e.radii, b.network.vessels.at(id).radii);
}

TEST(VascularCalibration, FailedGrowthRetainsTrialDiagnostics)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.minimum_radius                = .01;
  p.calibration.minimum_terminals = 2;
  p.calibration.maximum_terminals = 3;
  p.calibration.volume_fraction   = seed_volume();
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_EQ(result.calibration.status, CalibrationStatus::exhausted);
  ASSERT_EQ(result.calibration.attempts.size(), 2u);
  for (const auto &trial : result.calibration.attempts)
    {
      EXPECT_FALSE(trial.generation_validation.valid());
      EXPECT_EQ(trial.sampled_points, 0u);
    }
}

TEST(VascularCalibration, DefaultsAndInvalidControls)
{
  CalibrationParameters p;
  EXPECT_DOUBLE_EQ(p.relative_tolerance, .05);
  EXPECT_EQ(p.attempt_budget, 12u);
  p.minimum_terminals = 0;
  EXPECT_THROW(p.validate(), ExceptionBase);
  p.minimum_terminals = 2;
  p.maximum_terminals = 1;
  EXPECT_THROW(p.validate(), ExceptionBase);
  p.maximum_terminals = 2;
  p.volume_fraction   = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(p.validate(), ExceptionBase);
  Parameters parameters;
  synthetic(parameters);
  parameters.calibration.minimum_coverage = .5;
  EXPECT_THROW(parameters.validate(), ExceptionBase);
}
