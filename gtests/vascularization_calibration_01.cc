// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/distributed/tria.h>

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

namespace
{
  void
  calibration_background(parallel::distributed::Triangulation<3> &background)
  {
    GridGenerator::subdivided_hyper_rectangle(background,
                                              {8, 1, 1},
                                              Point<3>(0., 0., 0.),
                                              Point<3>(8., 1., 1.));
  }

  void
  distributed_controls(Parameters &p)
  {
    synthetic(p);
    p.seed_position                 = Point<3>(6.5, .5, .5);
    p.seed_direction                = Point<3>(-1., 0., 0.);
    p.optimization_steps            = 3;
    p.calibration.maximum_terminals = 3;
  }
} // namespace

TEST(VascularCalibration, BOTH_CollectiveSearchFindsAnalyticCrossPartitionSeed)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  parallel::distributed::Triangulation<3> background(comm);
  calibration_background(background);
  Parameters p;
  distributed_controls(p);
  p.seed_length                    = 5.;
  p.terminal_count                 = 3;
  p.reference_flow                 = 2.e-9;
  p.tissue_demand                  = "1+x";
  const double target              = numbers::PI * .01 * .01 * 5. / 8.;
  p.calibration.volume_fraction    = target;
  p.calibration.relative_tolerance = 1.e-10;
  p.coverage.service_radius        = 10.;
  p.calibration.minimum_coverage   = 1.;
  const auto result = generate(background, MappingQ1<3>(), {}, p, comm);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_EQ(result.calibration.status, CalibrationStatus::converged);
  ASSERT_EQ(result.calibration.attempts.size(), 2u);
  EXPECT_EQ(result.calibration.attempts[0].terminal_count, 3u);
  EXPECT_EQ(result.calibration.attempts[1].terminal_count, 1u);
  EXPECT_EQ(Utilities::MPI::sum(
              static_cast<unsigned int>(result.territories.size()), comm),
            1u);
  EXPECT_NEAR(result.tissue.volume_fraction / target, 1., 1.e-13);
  EXPECT_NEAR(result.tissue.arterial_coverage, 1., 1.e-13);
  unsigned int vessels    = 0;
  double       total_flow = 0.;
  for (const auto &[id, v] : result.network.vessels)
    if (v.owner == rank)
      {
        ++vessels;
        EXPECT_EQ(v.radii, (std::array<double, 2>{{.01, .01}}));
        EXPECT_NEAR(length(result.network, id), 5., 1.e-13);
      }
  for (const auto &[id, t] : result.network.terminals)
    if (t.owner == rank)
      total_flow += result.target_terminal_flows.at(id);
  EXPECT_EQ(Utilities::MPI::sum(vessels, comm), 1u);
  EXPECT_NEAR(Utilities::MPI::sum(total_flow, comm) / p.reference_flow,
              1.,
              1.e-13);
  EXPECT_EQ(result.calibration.sampled_points,
            result.calibration.attempts[0].sampled_points);
  EXPECT_GT(result.calibration.sampled_points, 0u);
  EXPECT_EQ(Utilities::MPI::min(result.calibration.sampled_points, comm),
            result.calibration.sampled_points);
  EXPECT_EQ(Utilities::MPI::max(result.calibration.sampled_points, comm),
            result.calibration.sampled_points);
  EXPECT_EQ(p.terminal_count, 3u);
  EXPECT_DOUBLE_EQ(p.seed_radius, .01);
}

TEST(VascularCalibration, BOTH_CollectiveNonmonotoneSearchExhaustsAndRepeats)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> background(comm);
  calibration_background(background);
  Parameters p;
  distributed_controls(p);
  p.terminal_count              = 3;
  p.calibration.volume_fraction = 1.e-12;
  const auto result   = generate(background, MappingQ1<3>(), {}, p, comm);
  const auto repeated = generate(background, MappingQ1<3>(), {}, p, comm);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_EQ(result.calibration.status, CalibrationStatus::exhausted);
  ASSERT_EQ(result.calibration.attempts.size(), 3u);
  ASSERT_EQ(repeated.calibration.attempts.size(), 3u);
  std::set<unsigned int> counts;
  std::uint64_t          samples = 0;
  for (unsigned int i = 0; i < result.calibration.attempts.size(); ++i)
    {
      const auto &attempt = result.calibration.attempts[i];
      const auto &again   = repeated.calibration.attempts[i];
      EXPECT_TRUE(attempt.generation_validation.valid());
      EXPECT_TRUE(counts.insert(attempt.terminal_count).second);
      EXPECT_GT(attempt.tissue.volume_fraction,
                p.calibration.volume_fraction *
                  (1. + p.calibration.relative_tolerance));
      EXPECT_EQ(attempt.terminal_count, again.terminal_count);
      EXPECT_DOUBLE_EQ(attempt.tissue.volume_fraction,
                       again.tissue.volume_fraction);
      EXPECT_EQ(attempt.sampled_points, again.sampled_points);
      EXPECT_EQ(Utilities::MPI::min(attempt.tissue.volume_fraction, comm),
                attempt.tissue.volume_fraction);
      EXPECT_EQ(Utilities::MPI::max(attempt.tissue.volume_fraction, comm),
                attempt.tissue.volume_fraction);
      samples += attempt.sampled_points;
    }
  EXPECT_EQ(counts, (std::set<unsigned int>{1, 2, 3}));
  EXPECT_EQ(result.calibration.sampled_points, samples);
  EXPECT_EQ(result.network.vessels.size(), repeated.network.vessels.size());
  EXPECT_TRUE(
    validate_data(result.network, Utilities::MPI::n_mpi_processes(comm))
      .valid());
  for (const auto &[id, v] : result.network.vessels)
    {
      ASSERT_TRUE(repeated.network.vessels.count(id));
      EXPECT_EQ(v.nodes, repeated.network.vessels.at(id).nodes);
      EXPECT_EQ(v.radii, repeated.network.vessels.at(id).radii);
    }
  for (const auto &[id, n] : result.network.nodes)
    {
      ASSERT_TRUE(repeated.network.nodes.count(id));
      EXPECT_EQ(n.position, repeated.network.nodes.at(id).position);
    }
}

TEST(VascularCalibration, BOTH_ImportedVolumeAndEmptyBackgroundRanks)
{
  const auto comm   = MPI_COMM_WORLD;
  const auto rank   = Utilities::MPI::this_mpi_process(comm);
  const auto source = Utilities::MPI::n_mpi_processes(comm) - 1;
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::hyper_cube(background);
  NetworkData input;
  if (rank == source)
    {
      input.nodes[0] = {Point<3>(-.2, .5, .5),
                        source,
                        Port::arterial_inlet,
                        Provenance::imported};
      input.nodes[1] = {Point<3>(.3, .5, .5),
                        source,
                        Port::none,
                        Provenance::imported};
      Vessel v;
      v.nodes                                = {{0, 1}};
      v.radii                                = {{.01, .01}};
      v.viscosity                            = .004;
      v.owner                                = source;
      v.physical_properties["Young modulus"] = 12345.;
      input.vessels[4]                       = v;
      input.terminals[7] = {1, invalid_id, System::arterial, source, 2.};
    }
  Parameters p;
  synthetic(p);
  // Empty ranks must validate this as imported data, not an empty seed.
  p.seed_radius                   = 0.;
  p.viscosity                     = 0.;
  p.terminal_count                = 1;
  p.calibration.maximum_terminals = 1;
  p.calibration.volume_fraction   = seed_volume() / 2.;
  const auto incompatible =
    generate(background, MappingQ1<3>(), input, p, comm);
  EXPECT_FALSE(incompatible.validation.valid());
  EXPECT_EQ(incompatible.calibration.status, CalibrationStatus::incompatible);
  EXPECT_TRUE(incompatible.calibration.attempts.empty());
  EXPECT_EQ(incompatible.calibration.sampled_points, 0u);
  EXPECT_NEAR(incompatible.tissue.inside_vessel_volume / seed_volume(),
              1.,
              1.e-13);
  EXPECT_EQ(incompatible.network.vessels.size(), input.vessels.size());
  p.calibration.volume_fraction    = 2. * seed_volume();
  p.calibration.relative_tolerance = 1.e-10;
  p.reference_flow                 = 1.e-9;
  p.tissue_demand                  = "1+x";
  p.coverage.service_radius        = 2.;
  p.calibration.minimum_coverage   = 1.;
  const auto result = generate(background, MappingQ1<3>(), input, p, comm);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_EQ(result.calibration.status, CalibrationStatus::converged);
  EXPECT_EQ(result.calibration.attempts.size(), 1u);
  EXPECT_NEAR(result.tissue.volume_fraction / (2. * seed_volume()), 1., 1.e-13);
  EXPECT_NEAR(result.tissue.background_volume, 1., 1.e-13);
  EXPECT_NEAR(result.tissue.arterial_coverage, 1., 1.e-13);
  EXPECT_EQ(Utilities::MPI::sum(
              static_cast<unsigned int>(result.territories.size()), comm),
            1u);
  unsigned int frozen = 0, fresh = 0;
  for (const auto &[id, v] : result.network.vessels)
    if (v.owner == rank)
      {
        if (id == 4)
          {
            ++frozen;
            EXPECT_EQ(v.nodes, (std::array<Id, 2>{{0, 1}}));
            EXPECT_EQ(v.radii, (std::array<double, 2>{{.01, .01}}));
            EXPECT_DOUBLE_EQ(v.viscosity, .004);
            EXPECT_DOUBLE_EQ(v.physical_properties.at("Young modulus"), 12345.);
          }
        else
          {
            ++fresh;
            EXPECT_DOUBLE_EQ(v.radii[0], .01);
            EXPECT_DOUBLE_EQ(v.viscosity, .004);
          }
      }
  EXPECT_EQ(Utilities::MPI::sum(frozen, comm), 1u);
  EXPECT_EQ(Utilities::MPI::sum(fresh, comm), 1u);
  if (rank == source)
    {
      EXPECT_EQ(input.nodes.at(0).position, Point<3>(-.2, .5, .5));
      EXPECT_EQ(input.vessels.at(4).owner, source);
      EXPECT_DOUBLE_EQ(input.terminals.at(7).demand, 2.);
    }
}

TEST(VascularCalibration, BOTH_CollectiveVolumeMatchStillRequiresCoverage)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> background(comm);
  calibration_background(background);
  Parameters p;
  distributed_controls(p);
  p.calibration.maximum_terminals = 1;
  p.calibration.volume_fraction   = seed_volume() / 8.;
  p.calibration.minimum_coverage  = .5;
  p.coverage                      = {.001, 1};
  const auto result = generate(background, MappingQ1<3>(), {}, p, comm);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_EQ(result.calibration.status, CalibrationStatus::exhausted);
  ASSERT_EQ(result.calibration.attempts.size(), 1u);
  EXPECT_TRUE(result.calibration.attempts[0].generation_validation.valid());
  EXPECT_NEAR(result.tissue.volume_fraction / (seed_volume() / 8.), 1., 1.e-13);
  EXPECT_DOUBLE_EQ(result.tissue.arterial_coverage, 0.);
  p.calibration.volume_fraction = 0.;
  const auto coverage_only = generate(background, MappingQ1<3>(), {}, p, comm);
  EXPECT_FALSE(coverage_only.validation.valid());
  EXPECT_EQ(coverage_only.calibration.status, CalibrationStatus::exhausted);
  EXPECT_EQ(coverage_only.calibration.attempts.size(), 1u);
}

TEST(VascularCalibration, BOTH_CollectiveFailedGrowthKeepsTrialDiagnostics)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> background(comm);
  calibration_background(background);
  Parameters p;
  distributed_controls(p);
  p.minimum_radius                = p.seed_radius;
  p.terminal_count                = 2;
  p.calibration.minimum_terminals = 2;
  p.calibration.maximum_terminals = 3;
  p.calibration.volume_fraction   = seed_volume() / 8.;
  const auto result = generate(background, MappingQ1<3>(), {}, p, comm);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_EQ(result.calibration.status, CalibrationStatus::exhausted);
  ASSERT_EQ(result.calibration.attempts.size(), 2u);
  std::set<unsigned int> counts;
  for (const auto &attempt : result.calibration.attempts)
    {
      EXPECT_FALSE(attempt.generation_validation.valid());
      EXPECT_EQ(attempt.sampled_points, 0u);
      counts.insert(attempt.terminal_count);
    }
  EXPECT_EQ(counts, (std::set<unsigned int>{2, 3}));
  EXPECT_EQ(result.calibration.sampled_points, 0u);
}

TEST(VascularCalibration, MPI_InitialCountsAndInvalidControlsFailCollectively)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  ASSERT_GT(size, 1u);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.calibration.volume_fraction = seed_volume();
  p.terminal_count              = rank == size - 1 ? 2 : 1;
  EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, comm),
               ExceptionBase);
  p.terminal_count = 1;
  if (rank == size - 1)
    p.minimum_radius = -.1;
  EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, comm),
               ExceptionBase);
  p.minimum_radius = 0.;
  NetworkData input;
  if (rank == size - 1)
    input.nodes[10] = {Point<3>(std::numeric_limits<double>::quiet_NaN(),
                                .5,
                                .5),
                       rank,
                       Port::arterial_inlet,
                       Provenance::imported};
  const auto invalid = generate(background, MappingQ1<3>(), input, p, comm);
  EXPECT_FALSE(invalid.validation.valid());
  EXPECT_TRUE(invalid.calibration.attempts.empty());
}
