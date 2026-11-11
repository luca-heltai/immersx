// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <cmath>

using namespace dealii;
using namespace ImmersX::Vascularization;

namespace
{
  void
  configure(Parameters &p)
  {
    p.terminal_count_multiplier_factor = "none";
    p.length_scale                     = "meter";
    p.enforce_geometry                 = false;
    p.minimum_length                   = 0.;
    p.density_beta                     = 0.;
    p.coverage.service_radius          = 0.;
    p.seed_position                    = Point<3>(.3, .5, .5);
    p.seed_direction                   = Point<3>(1., 0., 0.);
    p.seed_radius                      = .01;
    p.seed_length                      = .4;
    p.viscosity                        = .004;
    p.terminal_count                   = 6;
    p.sample_budget                    = 500;
    p.candidate_count                  = 8;
    p.optimization_steps               = 2;
    p.reference_flow                   = 1.e-9;
    p.tissue_demand                    = "1+x";
  }

  void
  same_network(const NetworkData &a, const NetworkData &b)
  {
    ASSERT_EQ(a.nodes.size(), b.nodes.size());
    ASSERT_EQ(a.vessels.size(), b.vessels.size());
    ASSERT_EQ(a.terminals.size(), b.terminals.size());
    for (const auto &[id, node] : a.nodes)
      {
        ASSERT_TRUE(b.nodes.count(id));
        EXPECT_EQ(node.position, b.nodes.at(id).position);
        EXPECT_EQ(node.owner, b.nodes.at(id).owner);
        EXPECT_EQ(node.port, b.nodes.at(id).port);
      }
    for (const auto &[id, vessel] : a.vessels)
      {
        ASSERT_TRUE(b.vessels.count(id));
        EXPECT_EQ(vessel.nodes, b.vessels.at(id).nodes);
        EXPECT_EQ(vessel.radii, b.vessels.at(id).radii);
        EXPECT_EQ(vessel.owner, b.vessels.at(id).owner);
        EXPECT_EQ(vessel.system, b.vessels.at(id).system);
        EXPECT_EQ(vessel.provenance, b.vessels.at(id).provenance);
      }
    for (const auto &[id, terminal] : a.terminals)
      {
        ASSERT_TRUE(b.terminals.count(id));
        EXPECT_EQ(terminal.node, b.terminals.at(id).node);
        EXPECT_EQ(terminal.territory, b.terminals.at(id).territory);
        EXPECT_EQ(terminal.demand, b.terminals.at(id).demand);
      }
  }
} // namespace

TEST(VascularGeneration, BOTH_EmptyControlPreservesGeneration)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_cube(bg, 2);
  Parameters p("/Generation empty control/");
  configure(p);
  const auto baseline = generate(bg, MappingQ1<3>(), {}, p, comm);
  ASSERT_TRUE(baseline.validation.valid());
  const auto controlled =
    generate(bg, MappingQ1<3>(), {}, p, comm, GenerationControl{});
  ASSERT_TRUE(controlled.validation.valid());
  EXPECT_FALSE(controlled.controlled_stop);
  same_network(baseline.network, controlled.network);
}

TEST(VascularGeneration, BOTH_SnapshotsPreserveAcceptedGrowth)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_cube(bg, 2);
  Parameters p("/Generation snapshots/");
  configure(p);
  const auto baseline = generate(bg, MappingQ1<3>(), {}, p, comm);
  ASSERT_TRUE(baseline.validation.valid());
  unsigned int      snapshots = 0;
  GenerationControl control;
  control.poll = [](const GenerationProgress &progress) {
    return GenerationAction{progress.accepted_terminals >= 2, false};
  };
  control.checkpoint = [&](const GenerationProgress &progress,
                           NetworkData               snapshot,
                           bool                      stopped) {
    ++snapshots;
    EXPECT_FALSE(stopped);
    EXPECT_GE(progress.accepted_terminals, 2u);
    for (const auto &[id, vessel] : snapshot.vessels)
      {
        (void)id;
        EXPECT_GT(vessel.radii[0], 0.);
        EXPECT_GT(vessel.radii[1], 0.);
        EXPECT_TRUE(std::isfinite(vessel.radii[0]));
        EXPECT_TRUE(std::isfinite(vessel.radii[1]));
      }
  };
  const auto result = generate(bg, MappingQ1<3>(), {}, p, comm, control);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_FALSE(result.controlled_stop);
  EXPECT_GE(snapshots, 2u);
  EXPECT_EQ(Utilities::MPI::min(snapshots, comm),
            Utilities::MPI::max(snapshots, comm));
  same_network(baseline.network, result.network);
}

TEST(VascularGeneration, BOTH_ControlledStopBypassesFurtherCalibration)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto last = Utilities::MPI::n_mpi_processes(comm) - 1;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_cube(bg, 2);
  Parameters p("/Generation cooperative calibration stop/");
  configure(p);
  p.terminal_count                = 12;
  p.calibration.volume_fraction   = .01;
  p.calibration.minimum_terminals = 4;
  p.calibration.maximum_terminals = 20;
  p.calibration.attempt_budget    = 4;
  unsigned int      snapshots = 0, stopped_snapshots = 0;
  std::uint64_t     stopped_count = 0;
  GenerationControl control;
  control.poll = [rank, last](const GenerationProgress &progress) {
    return GenerationAction{rank == 0 && progress.accepted_terminals >= 2,
                            rank == last && progress.accepted_terminals >= 5};
  };
  control.checkpoint = [&](const GenerationProgress &progress,
                           NetworkData               snapshot,
                           bool                      stopped) {
    ++snapshots;
    stopped_snapshots += stopped;
    if (stopped)
      stopped_count = progress.accepted_terminals;
    std::uint64_t owned = 0;
    for (const auto &entry : snapshot.vessels)
      owned += entry.second.owner == rank;
    EXPECT_GT(Utilities::MPI::sum(owned, comm), 0u);
  };
  const auto result = generate(bg, MappingQ1<3>(), {}, p, comm, control);
  EXPECT_TRUE(result.controlled_stop);
  EXPECT_FALSE(result.validation.valid());
  ASSERT_EQ(result.calibration.attempts.size(), 1u);
  EXPECT_EQ(result.calibration.status, CalibrationStatus::exhausted);
  EXPECT_GE(stopped_count, 5u);
  EXPECT_LT(stopped_count, p.terminal_count);
  EXPECT_EQ(stopped_snapshots, 1u);
  EXPECT_GE(snapshots, 2u);
  EXPECT_EQ(Utilities::MPI::min(stopped_count, comm),
            Utilities::MPI::max(stopped_count, comm));
}

TEST(VascularGeneration, MPI_ControlPresenceDisagreementIsRejected)
{
  const auto comm = MPI_COMM_WORLD;
  ASSERT_GE(Utilities::MPI::n_mpi_processes(comm), 2u);
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_cube(bg, 2);
  Parameters p("/Generation control disagreement/");
  configure(p);
  GenerationControl control;
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    control.poll = [](const GenerationProgress &) {
      return GenerationAction{};
    };
  EXPECT_ANY_THROW(generate(bg, MappingQ1<3>(), {}, p, comm, control));
}
