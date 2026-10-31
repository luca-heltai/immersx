// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_description.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <set>

using namespace dealii;
using namespace ImmersX::Vascularization;

namespace
{
  void
  synthetic(Parameters &p)
  {
    p.terminal_count_multiplier_factor = "none";
    p.length_scale                     = "meter";
    p.minimum_length                   = 0.;
    p.density_beta                     = 0.;
    p.coverage.service_radius          = 0.;
    p.venous.terminal_separation       = 0.;

    p.seed_position  = Point<3>(.3, .5, .5);
    p.seed_direction = Point<3>(1., 0., 0.);
    p.seed_radius    = .01;
    p.seed_length    = .4;
    p.viscosity      = .004;
  }
} // namespace

TEST(VascularGeneration, CubeSeedAnalyticalGate)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.refine_global(2);
  const MappingQ1<3> mapping;
  Parameters         p;
  synthetic(p);
  const NetworkData initial;
  const auto result = generate(background, mapping, initial, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  ASSERT_EQ(result.network.vessels.size(), 1u);
  ASSERT_EQ(result.network.terminals.size(), 1u);
  EXPECT_EQ(result.network.nodes.at(0).port, Port::arterial_inlet);
  EXPECT_DOUBLE_EQ(result.network.vessels.at(0).radii[0], p.seed_radius);
  EXPECT_DOUBLE_EQ(result.network.vessels.at(0).radii[1], p.seed_radius);
  EXPECT_NEAR(volume(result.network, 0) / (numbers::PI * .01 * .01 * .4),
              1.,
              1.e-14);
  EXPECT_NEAR(resistance(result.network, 0) /
                (8. * .004 * .4 / (numbers::PI * 1.e-8)),
              1.,
              1.e-14);
  EXPECT_TRUE(initial.nodes.empty());
  EXPECT_TRUE(initial.vessels.empty());
}

TEST(VascularGeneration, RejectsWallEscapeWithInteriorCentreline)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.seed_position[1] = .005;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_TRUE(result.network.vessels.empty());
}

TEST(VascularGeneration, NonconvexDomainRejectsSegmentBetweenInteriorEndpoints)
{
  Triangulation<3> full, background;
  GridGenerator::subdivided_hyper_rectangle(full,
                                            {3, 3, 1},
                                            Point<3>(0., 0., 0.),
                                            Point<3>(3., 3., 1.));
  std::set<Triangulation<3>::active_cell_iterator> remove;
  for (const auto &cell : full.active_cell_iterators())
    if (cell->center()[0] > 1. && cell->center()[0] < 2. &&
        cell->center()[1] > 1.)
      remove.insert(cell);
  GridGenerator::create_triangulation_with_removed_cells(full,
                                                         remove,
                                                         background);
  Parameters p;
  synthetic(p);
  p.seed_position = Point<3>(.5, 2.5, .5);
  p.seed_length   = 2.;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_TRUE(result.network.vessels.empty());
}

TEST(VascularGeneration, TetrahedralBoundaryBoxIsOnlyBroadPhase)
{
  Triangulation<3> background;
  GridGenerator::reference_cell(background, ReferenceCells::Tetrahedron);
  const MappingFE<3> mapping(FE_SimplexP<3>(1));
  Parameters         p;
  synthetic(p);
  p.seed_position   = Point<3>(.15, .15, .15);
  p.seed_length     = .1;
  p.seed_radius     = .02;
  const auto result = generate(background, mapping, {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_NEAR(length(result.network, 0), .1, 1.e-14);
}

TEST(VascularGeneration, DeterministicInteriorDirectionAndInvalidPhysiology)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.seed_position  = Point<3>(.5, .5, .5);
  p.seed_direction = Point<3>();
  p.seed_length    = .1;
  const auto a = generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  const auto b = generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(a.validation.valid());
  ASSERT_TRUE(b.validation.valid());
  EXPECT_EQ(a.network.nodes.at(1).position, b.network.nodes.at(1).position);
  p.seed_radius = 0.;
  EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD),
               ExceptionBase);
}

TEST(VascularGeneration, BilinearWallContainment)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.begin_active()->vertex(7)[2] = 1.4;
  Parameters p;
  synthetic(p);
  p.seed_position = Point<3>(.6, .6, 1.05);
  p.seed_length   = .1;
  p.seed_radius   = .02;
  // The warped upper face is z=1+0.4*x*y. Its bounding box overlaps
  // both proposals, while only the second proposal's wall reaches the face.
  const auto accepted =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_TRUE(accepted.validation.valid());
  p.seed_position[2] = 1.14;
  const auto rejected =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(rejected.validation.valid());
  EXPECT_TRUE(rejected.network.vessels.empty());
}

TEST(VascularGeneration, ExhaustedGeometryBudgetDoesNotAccept)
{
  Triangulation<3> background;
  GridGenerator::reference_cell(background, ReferenceCells::Tetrahedron);
  const MappingFE<3> mapping(FE_SimplexP<3>(1));
  Parameters         p;
  synthetic(p);
  p.seed_position     = Point<3>(.15, .15, .15);
  p.seed_length       = .1;
  p.seed_radius       = .02;
  p.geometry_work     = 1;
  const auto rejected = generate(background, mapping, {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(rejected.validation.valid());
  EXPECT_TRUE(rejected.network.vessels.empty());
  p.geometry_work     = 100000;
  const auto accepted = generate(background, mapping, {}, p, MPI_COMM_WORLD);
  EXPECT_TRUE(accepted.validation.valid());
}

TEST(VascularGeneration,
     BOTH_PublicSeedStartsAtItsBackgroundOwnerAndCrossesPartitions)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {8, 1, 1},
                                            Point<3>(0., 0., 0.),
                                            Point<3>(8., 1., 1.));
  Parameters p;
  synthetic(p);
  p.seed_position  = Point<3>(6.5, .5, .5);
  p.seed_direction = Point<3>(-1., 0., 0.);
  p.seed_length    = 5.;
  p.reference_flow = 1.e-9;
  p.tissue_demand  = "1+x";
  p.coverage       = {10., 1};
  const NetworkData input;
  const auto result = generate(background, MappingQ1<3>(), input, p, comm);
  ASSERT_TRUE(result.validation.valid());
  unsigned int node_owner = size, end_owner = size;
  for (const auto &cell : background.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        if (cell->bounding_box().point_inside(p.seed_position))
          node_owner = rank;
        if (cell->bounding_box().point_inside(Point<3>(1.5, .5, .5)))
          end_owner = rank;
      }
  node_owner               = Utilities::MPI::min(node_owner, comm);
  end_owner                = Utilities::MPI::min(end_owner, comm);
  unsigned int owned_ports = 0, owned_vessels = 0, owned_terminals = 0;
  for (const auto &[id, n] : result.network.nodes)
    if (n.owner == rank && n.port == Port::arterial_inlet)
      {
        ++owned_ports;
        EXPECT_EQ(id, 0u);
        EXPECT_EQ(n.position, p.seed_position);
        EXPECT_EQ(rank, node_owner);
      }
  for (const auto &[id, v] : result.network.vessels)
    if (v.owner == rank)
      {
        ++owned_vessels;
        EXPECT_EQ(id, 0u);
        EXPECT_EQ(v.radii,
                  (std::array<double, 2>{{p.seed_radius, p.seed_radius}}));
        EXPECT_NEAR(length(result.network, id), 5., 1.e-13);
      }
  for (const auto &[id, t] : result.network.terminals)
    if (t.owner == rank)
      {
        ++owned_terminals;
        EXPECT_EQ(rank, end_owner);
        EXPECT_NEAR(t.demand, 2.5, 1.e-13);
        EXPECT_EQ(result.network.nodes.at(t.node).position,
                  Point<3>(1.5, .5, .5));
        EXPECT_NEAR(result.target_terminal_flows.at(id) / p.reference_flow,
                    1.,
                    1.e-13);
      }
  EXPECT_EQ(Utilities::MPI::sum(owned_ports, comm), 1u);
  EXPECT_EQ(Utilities::MPI::sum(owned_vessels, comm), 1u);
  EXPECT_EQ(Utilities::MPI::sum(owned_terminals, comm), 1u);
  if (size > 1)
    {
      EXPECT_NE(node_owner, 0u);
      EXPECT_NE(node_owner, end_owner);
    }
  EXPECT_NEAR(result.tissue.inside_vessel_volume / (numbers::PI * .0001 * 5.),
              1.,
              1.e-13);
  EXPECT_NEAR(result.tissue.volume_fraction / (numbers::PI * .0001 * 5. / 8.),
              1.,
              1.e-13);
  EXPECT_NEAR(result.tissue.arterial_coverage, 1., 1.e-13);
  EXPECT_TRUE(input.nodes.empty());
  EXPECT_TRUE(input.vessels.empty());
}

TEST(VascularGeneration, BOTH_PublicSeedFeedsTheSameBatchedGrowthEngine)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {4, 1, 1},
                                            Point<3>(0., 0., 0.),
                                            Point<3>(4., 1., 1.));
  Parameters p;
  synthetic(p);
  p.seed_position      = Point<3>(3.6, .5, .5);
  p.seed_direction     = Point<3>(-1., 0., 0.);
  p.seed_length        = .4;
  p.terminal_count     = 6;
  p.minimum_radius     = .0001;
  p.minimum_length     = .002;
  p.minimum_angle      = 10.;
  p.sample_budget      = 500;
  p.optimization_steps = 3;
  p.reference_flow     = 1.e-9;
  p.tissue_demand      = "1+x";
  const auto a         = generate(background, MappingQ1<3>(), {}, p, comm);
  ASSERT_TRUE(a.validation.valid());
  const auto b = generate(background, MappingQ1<3>(), {}, p, comm);
  ASSERT_TRUE(b.validation.valid());
  unsigned int ports = 0, vessels = 0, terminals = 0;
  double       flow = 0.;
  for (const auto &[id, n] : a.network.nodes)
    {
      EXPECT_EQ(n.position, b.network.nodes.at(id).position);
      ports += n.owner == rank && n.port == Port::arterial_inlet;
    }
  for (const auto &[id, v] : a.network.vessels)
    {
      EXPECT_EQ(v.nodes, b.network.vessels.at(id).nodes);
      EXPECT_EQ(v.radii, b.network.vessels.at(id).radii);
      vessels += v.owner == rank;
      if (id == 0)
        {
          EXPECT_DOUBLE_EQ(v.radii[0], p.seed_radius);
        }
    }
  for (const auto &[id, t] : a.network.terminals)
    if (t.owner == rank)
      {
        ++terminals;
        flow += a.target_terminal_flows.at(id);
      }
  EXPECT_EQ(Utilities::MPI::sum(ports, comm), 1u);
  EXPECT_EQ(Utilities::MPI::sum(vessels, comm), 2 * p.terminal_count - 1);
  EXPECT_EQ(Utilities::MPI::sum(terminals, comm), p.terminal_count);
  EXPECT_NEAR(Utilities::MPI::sum(flow, comm) / p.reference_flow, 1., 1.e-13);
  EXPECT_EQ(a.accepted_insertions, p.terminal_count - 1);
  EXPECT_GT(a.sampled_points, 0u);
  EXPECT_EQ(a.sampled_points, b.sampled_points);
}

TEST(VascularGeneration, BOTH_PublicSeedOnSurfaceWithEmptyBackgroundRanks)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.seed_position           = Point<3>(0., .5, .5);
  p.seed_inward_port_normal = Point<3>(1., 0., 0.);
  const auto open           = generate(background, MappingQ1<3>(), {}, p, comm);
  ASSERT_TRUE(open.validation.valid());
  EXPECT_NEAR(open.tissue.inside_vessel_volume /
                (numbers::PI * .0001 * p.seed_length),
              1.,
              1.e-13);
  p.seed_inward_port_normal = Point<3>();
  const auto closed         = generate(background, MappingQ1<3>(), {}, p, comm);
  EXPECT_FALSE(closed.validation.valid());
  EXPECT_TRUE(closed.network.vessels.empty());
  p.seed_position    = Point<3>(1.1, .5, .5);
  const auto outside = generate(background, MappingQ1<3>(), {}, p, comm);
  EXPECT_FALSE(outside.validation.valid());
  EXPECT_TRUE(outside.network.vessels.empty());
}

TEST(VascularGeneration, MPI_SeedControlDisagreementIsRejectedCollectively)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  ASSERT_GE(size, 2u);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  if (rank == size - 1)
    p.seed_position[0] += .01;
  EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, comm),
               ExceptionBase);
  synthetic(p);
  if (rank == size - 1)
    p.seed_radius = 0.;
  EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, comm),
               ExceptionBase);
  synthetic(p);
  if (rank == size - 1)
    p.calibration.volume_fraction = .001;
  EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, comm),
               ExceptionBase);
}

TEST(VascularGeneration, BOTH_PublicTetrahedralSeedWithEmptyBackgroundRanks)
{
  const auto       comm = MPI_COMM_WORLD;
  Triangulation<3> serial;
  GridGenerator::reference_cell(serial, ReferenceCells::Tetrahedron);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, comm);
  parallel::fullydistributed::Triangulation<3> background(comm);
  background.create_triangulation(description);
  Parameters p;
  synthetic(p);
  p.seed_position = Point<3>(.15, .15, .15);
  p.seed_length   = .1;
  p.seed_radius   = .02;
  p.coverage      = {2., 2};
  const auto result =
    generate(background, MappingFE<3>(FE_SimplexP<3>(1)), {}, p, comm);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_NEAR(result.tissue.background_volume, 1. / 6., 1.e-13);
  EXPECT_NEAR(result.tissue.inside_vessel_volume / (numbers::PI * .0004 * .1),
              1.,
              1.e-13);
  EXPECT_NEAR(result.tissue.arterial_coverage, 1., 1.e-13);
}

TEST(VascularGeneration, BOTH_GrowthStatisticsCountActualOwnedWork)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  bg.refine_global(2);
  Parameters p;
  p.terminal_count_multiplier_factor = "none";
  p.length_scale                     = "meter";
  p.minimum_length                   = 0.;
  p.density_beta                     = 0.;
  p.coverage.service_radius          = 0.;
  p.venous.terminal_separation       = 0.;

  p.seed_position   = Point<3>(.2, .5, .5);
  p.seed_direction  = Point<3>(1., 0., 0.);
  p.seed_radius     = .003;
  p.seed_length     = .15;
  p.viscosity       = .004;
  p.reference_flow  = 2.e-9;
  p.terminal_count  = 12;
  const auto result = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  ASSERT_FALSE(result.growth_statistics.empty());
  EXPECT_LE(result.growth_statistics.size(), 4u);
  std::uint64_t sampled = 0, accepted = 0, first_owned = 0, final_owned = 0;
  for (const auto &s : result.growth_statistics)
    {
      EXPECT_EQ(s.system, System::arterial);
      EXPECT_EQ(s.requested_terminals, 12u);
      EXPECT_GT(s.batches, 0u);
      EXPECT_LE(s.first_terminals, s.last_terminals);
      EXPECT_LE(s.last_terminals, 12u);
      EXPECT_LE(s.accepted_insertions, s.proposed_insertions);
      EXPECT_LE(s.active_writer_batches, s.batches);
      EXPECT_LE(s.committed_batches + s.rejected_batches, s.batches);
      EXPECT_GE(s.owned_vessels_maximum, s.owned_vessels_first);
      EXPECT_GE(s.owned_vessels_maximum, s.owned_vessels_last);
      EXPECT_GE(s.sampling_seconds, 0.);
      EXPECT_GE(s.search_seconds, 0.);
      EXPECT_GE(s.optimization_seconds, 0.);
      EXPECT_GE(s.selection_seconds, 0.);
      EXPECT_GE(s.commit_seconds, 0.);
      EXPECT_GE(s.redistribution_seconds, 0.);
      EXPECT_GE(s.physiology_seconds, 0.);
      EXPECT_GE(s.certification_seconds, 0.);
      EXPECT_LE(s.sampling_seconds + s.search_seconds + s.optimization_seconds +
                  s.selection_seconds + s.commit_seconds,
                s.total_seconds + 1.e-6);
      EXPECT_LE(s.redistribution_seconds + s.physiology_seconds +
                  s.certification_seconds,
                s.commit_seconds + 1.e-6);
      sampled += s.sampled_points;
      accepted += s.accepted_insertions;
      if (s.first_terminals == 1)
        {
          first_owned = s.owned_vessels_first;
          EXPECT_GT(s.initialization_seconds, 0.);
        }
      if (s.last_terminals == 12)
        final_owned = s.owned_vessels_last;
    }
  // A single real seed is owned once, including when other ranks initially
  // have no vessels. Final counts exclude all native geometric ghosts.
  EXPECT_EQ(Utilities::MPI::sum(first_owned, MPI_COMM_WORLD), 1u);
  EXPECT_EQ(Utilities::MPI::sum(final_owned, MPI_COMM_WORLD), 23u);
  EXPECT_EQ(Utilities::MPI::sum(accepted, MPI_COMM_WORLD), 11u);
  EXPECT_EQ(Utilities::MPI::sum(sampled, MPI_COMM_WORLD),
            result.sampled_points);
}

TEST(VascularGeneration, BOTH_StatisticsRetainAllCalibrationTrials)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  bg.refine_global(1);
  Parameters p;
  p.seed_position                 = Point<3>(.2, .5, .5);
  p.seed_direction                = Point<3>(1., 0., 0.);
  p.seed_radius                   = .003;
  p.seed_length                   = .15;
  p.viscosity                     = .004;
  p.terminal_count                = 4;
  p.calibration.volume_fraction   = 1.e-7;
  p.calibration.minimum_terminals = 2;
  p.calibration.maximum_terminals = 8;
  p.calibration.attempt_budget    = 3;
  const auto result = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_EQ(result.calibration.status, CalibrationStatus::exhausted);
  std::set<unsigned int> measured, attempted;
  for (const auto &s : result.growth_statistics)
    measured.insert(s.requested_terminals);
  for (const auto &a : result.calibration.attempts)
    {
      EXPECT_TRUE(a.generation_validation.valid());
      attempted.insert(a.terminal_count);
    }
  EXPECT_EQ(attempted.size(), 3u);
  EXPECT_EQ(measured, attempted);
}
