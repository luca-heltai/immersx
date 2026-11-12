// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>
#include <immersx/vascularization/io.h>

#include <chrono>
#include <cmath>
#include <tuple>

#include "../source/vascularization/network_serialization.h"
#include "test_paths.h"

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
        EXPECT_EQ(node.provenance, b.nodes.at(id).provenance);
        EXPECT_EQ(node.inward_port_normal, b.nodes.at(id).inward_port_normal);
      }
    for (const auto &[id, vessel] : a.vessels)
      {
        ASSERT_TRUE(b.vessels.count(id));
        EXPECT_EQ(vessel.nodes, b.vessels.at(id).nodes);
        EXPECT_EQ(vessel.radii, b.vessels.at(id).radii);
        EXPECT_EQ(vessel.viscosity, b.vessels.at(id).viscosity);
        EXPECT_EQ(vessel.physical_properties,
                  b.vessels.at(id).physical_properties);
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
        EXPECT_EQ(terminal.system, b.terminals.at(id).system);
        EXPECT_EQ(terminal.owner, b.terminals.at(id).owner);
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
    std::uint64_t owned_vessels = 0, owned_terminals = 0;
    const auto    rank = Utilities::MPI::this_mpi_process(comm);
    for (const auto &[id, vessel] : snapshot.vessels)
      {
        owned_vessels += vessel.owner == rank;
        (void)id;
        EXPECT_GT(vessel.radii[0], 0.);
        EXPECT_GT(vessel.radii[1], 0.);
        EXPECT_TRUE(std::isfinite(vessel.radii[0]));
        EXPECT_TRUE(std::isfinite(vessel.radii[1]));
      }
    for (const auto &[id, terminal] : snapshot.terminals)
      owned_terminals += terminal.owner == rank;
    EXPECT_EQ(Utilities::MPI::sum(owned_terminals, comm),
              progress.accepted_terminals);
    EXPECT_EQ(Utilities::MPI::sum(owned_vessels, comm),
              2 * progress.accepted_terminals - 1);
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

TEST(VascularGeneration, BOTH_RejectedBatchRetainsLastAcceptedState)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_cube(bg, 2);
  Parameters p("/Generation rejected batch stop/");
  configure(p);
  p.enforce_geometry = true;
  // Bootstrap's .4-long seed is admissible, but splitting it cannot produce
  // two segments each at least .3 long. The first growth batch is rejected.
  p.minimum_length        = .3;
  std::uint64_t     polls = 0, snapshots = 0;
  GenerationControl control;
  control.poll = [&](const GenerationProgress &progress) {
    ++polls;
    EXPECT_EQ(progress.accepted_terminals, 1u);
    return GenerationAction{false, progress.batches >= 1};
  };
  control.checkpoint = [&](const GenerationProgress &progress,
                           NetworkData               snapshot,
                           bool                      stopped) {
    ++snapshots;
    EXPECT_TRUE(stopped);
    EXPECT_EQ(progress.batches, 1u);
    EXPECT_EQ(progress.accepted_terminals, 1u);
    std::uint64_t owned = 0;
    const auto    rank  = Utilities::MPI::this_mpi_process(comm);
    for (const auto &[id, vessel] : snapshot.vessels)
      owned += vessel.owner == rank;
    EXPECT_EQ(Utilities::MPI::sum(owned, comm), 1u);
  };
  const auto result = generate(bg, MappingQ1<3>(), {}, p, comm, control);
  EXPECT_TRUE(result.controlled_stop);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_EQ(polls, 2u);
  EXPECT_EQ(snapshots, 1u);
  EXPECT_EQ(result.accepted_insertions, 0u);
}

TEST(VascularGeneration, BOTH_ObserverFailureStopsCollectively)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_cube(bg, 2);
  Parameters p("/Generation observer failure/");
  configure(p);
  const auto        rank = Utilities::MPI::this_mpi_process(comm);
  GenerationControl control;
  control.poll = [rank](const GenerationProgress &) {
    AssertThrow(rank != 0, ExcMessage("Injected rank-zero observer failure"));
    return GenerationAction{};
  };
  const auto result = generate(bg, MappingQ1<3>(), {}, p, comm, control);
  EXPECT_TRUE(result.controlled_stop);
  EXPECT_FALSE(result.validation.valid());
}

TEST(VascularGeneration, BOTH_NativeSnapshotRetainsAcceptedPhysicalRegistry)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_cube(bg, 2);
  const MappingQ1<3> mapping;
  Parameters         p("/Generation native checkpoint/");
  configure(p);
  p.terminal_count                 = 8;
  p.venous.enabled                 = true;
  p.portal.enabled                 = true;
  p.venous.terminal_separation     = .01;
  p.portal.terminal_separation     = .02;
  p.reference_conditions.pressures = {{0, 1500.}, {10, 0.}, {20, 1000.}};
  NetworkData original;
  if (rank == 0)
    for (const auto &[id, system, port, root, tip] :
         {std::make_tuple(Id{0},
                          System::arterial,
                          Port::arterial_inlet,
                          Point<3>(.1, .3, .3),
                          Point<3>(.3, .3, .3)),
          std::make_tuple(Id{10},
                          System::venous,
                          Port::venous_outlet,
                          Point<3>(.9, .5, .5),
                          Point<3>(.7, .5, .5)),
          std::make_tuple(Id{20},
                          System::portal,
                          Port::portal_inlet,
                          Point<3>(.1, .7, .7),
                          Point<3>(.3, .7, .7))})
      {
        original.nodes[id]                    = {root, 0, port};
        original.nodes[id].inward_port_normal = tip - root;
        original.nodes[id + 1]                = {tip, 0, Port::none};
        Vessel vessel;
        vessel.nodes                                   = {{id, id + 1}};
        vessel.radii                                   = {{.01, .01}};
        vessel.viscosity                               = p.viscosity;
        vessel.system                                  = system;
        vessel.physical_properties["fixture_property"] = .125;
        original.vessels[id]                           = vessel;
        original.terminals[id] = {id + 1, invalid_id, system, 0, 1.};
      }
  std::string token;
  if (rank == 0)
    token = std::to_string(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  token = Utilities::MPI::broadcast(comm, token, 0);
  const auto directory =
    ImmersX::TestPaths::output_path("native-growth-checkpoint") / token;
  // Compare physical records independently of repartition and ghost selection.
  const auto physical = [&](const NetworkData &network) {
    NetworkData owned, result;
    for (const auto &[id, node] : network.nodes)
      if (node.owner == rank)
        owned.nodes.emplace(id, node);
    for (const auto &[id, vessel] : network.vessels)
      if (vessel.owner == rank)
        owned.vessels.emplace(id, vessel);
    for (const auto &[id, terminal] : network.terminals)
      if (terminal.owner == rank)
        owned.terminals.emplace(id, terminal);
    for (const auto &part : Utilities::MPI::all_gather(comm, owned))
      {
        result.nodes.insert(part.nodes.begin(), part.nodes.end());
        result.vessels.insert(part.vessels.begin(), part.vessels.end());
        result.terminals.insert(part.terminals.begin(), part.terminals.end());
      }
    for (auto &[id, node] : result.nodes)
      node.owner = 0;
    for (auto &[id, vessel] : result.vessels)
      vessel.owner = 0;
    for (auto &[id, terminal] : result.terminals)
      terminal.owner = 0;
    return result;
  };
  GenerationControl control;
  control.poll = [](const GenerationProgress &progress) {
    return GenerationAction{false,
                            progress.system == System::venous &&
                              progress.accepted_terminals >= 2};
  };
  unsigned int saved = 0;
  control.checkpoint = [&](const GenerationProgress &progress,
                           NetworkData               snapshot,
                           bool                      stopped) {
    EXPECT_TRUE(stopped);
    EXPECT_LT(progress.accepted_terminals, p.terminal_count);
    ++saved;
    save_network(directory, snapshot, {}, {}, comm);
    const auto restored = load_network(directory, bg, mapping, 0., comm);
    EXPECT_TRUE(restored.territories.empty());
    EXPECT_TRUE(restored.terminal_pairs.empty());
    EXPECT_TRUE(restored.terminal_triplets.empty());
    same_network(physical(snapshot), physical(restored.network));
  };
  const auto result = generate(bg, mapping, original, p, comm, control);
  EXPECT_TRUE(result.controlled_stop);
  ASSERT_EQ(saved, 1u);
  const auto restored = load_network(directory, bg, mapping, 0., comm);
  const auto resumed  = resume_companions(
    bg, mapping, original, restored.network, p, comm, System::venous);
  for (const auto &issue : resumed.validation.issues)
    ADD_FAILURE() << issue.entity << ": " << issue.message;
  ASSERT_TRUE(resumed.validation.valid());
  EXPECT_EQ(resumed.effective_terminal_count, p.terminal_count);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(
                                  resumed.terminal_triplets.size()),
                                comm),
            p.terminal_count);
  const auto before = physical(restored.network);
  const auto after  = physical(resumed.network);
  for (const auto &[id, vessel] : before.vessels)
    if (vessel.system != System::venous)
      {
        ASSERT_TRUE(after.vessels.count(id));
        EXPECT_EQ(vessel.nodes, after.vessels.at(id).nodes);
        EXPECT_EQ(vessel.radii, after.vessels.at(id).radii);
        EXPECT_EQ(vessel.physical_properties,
                  after.vessels.at(id).physical_properties);
      }
}
