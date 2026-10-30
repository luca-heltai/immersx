// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/distribution.h>
#include <immersx/vascularization/io.h>

#include <chrono>
#include <cmath>

#include "test_paths.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  NetworkData
  physical(bool extra_vein = false)
  {
    NetworkData n;
    for (const auto &[id, p] : std::map<Id, Point<3>>{{0, {.05, .5, .5}},
                                                      {1, {.25, .5, .5}},
                                                      {2, {.4, .25, .5}},
                                                      {3, {.4, .75, .5}},
                                                      {10, {.95, .5, .5}},
                                                      {11, {.75, .5, .5}},
                                                      {12, {.6, .25, .5}},
                                                      {13, {.6, .75, .5}}})
      n.nodes[id].position = p;
    n.nodes[0].port       = Port::arterial_inlet;
    n.nodes[10].port      = Port::venous_outlet;
    n.nodes[3].provenance = Provenance::generated;
    for (const auto &[id, ends] :
         std::map<Id, std::array<Id, 2>>{{100, {{0, 1}}},
                                         {101, {{1, 2}}},
                                         {102, {{1, 3}}},
                                         {110, {{10, 11}}},
                                         {111, {{11, 12}}},
                                         {112, {{11, 13}}}})
      {
        Vessel v;
        v.nodes        = ends;
        const double r = id == 100 || id == 110 ? .005 : .003;
        v.radii        = {{r, r}};
        v.viscosity    = .004;
        v.system       = id < 110 ? System::arterial : System::venous;
        v.physical_properties["index"] = id;
        if (id == 102 || id == 112)
          v.provenance = Provenance::generated;
        n.vessels[id] = v;
      }
    n.terminals[200] = {2, 20, System::arterial, 0, 1.};
    n.terminals[201] = {3, 21, System::arterial, 0, 2.};
    n.terminals[210] = {12, 20, System::venous, 0, 1.};
    n.terminals[211] = {13, 21, System::venous, 0, 2.};
    if (extra_vein)
      {
        n.nodes[20]      = {Point<3>(.95, .85, .5), 0, Port::venous_outlet};
        n.nodes[21]      = {Point<3>(.6, .85, .5), 0};
        auto v           = n.vessels.at(112);
        v.nodes          = {{20, 21}};
        n.vessels[120]   = v;
        n.terminals[220] = {21, invalid_id, System::venous, 0, 1.};
      }
    return n;
  }
  NetworkArchive
  initial_archive(const Triangulation<3> &bg, bool extra = false)
  {
    const auto     comm = MPI_COMM_WORLD;
    const auto     rank = Utilities::MPI::this_mpi_process(comm);
    NetworkArchive a;
    if (rank == 0)
      a.network = physical(extra);
    a.network = redistribute(bg, MappingQ1<3>(), a.network, 0., comm);
    for (const auto &[id, t] : a.network.terminals)
      if (t.owner == rank && t.system == System::arterial)
        {
          a.territories.push_back(
            {t.territory, a.network.nodes.at(t.node).position, rank, t.demand});
          a.terminal_pairs.push_back(
            {t.territory, id, id + 10, t.territory == 20 ? 4.e12 : 5.e12});
        }
    return a;
  }
  void
  controls(Parameters &p)
  {
    p.seed_length                    = .08;
    p.selected_terminals             = {201};
    p.terminal_count                 = 1;
    p.reference_flow                 = 2.e-9;
    p.coverage                       = {2., 2};
    p.venous.enabled                 = true;
    p.venous.selected_terminals      = {211};
    p.venous.terminal_separation     = .04;
    p.reference_conditions.pressures = {{0, 12000.}, {10, 4000.}};
  }
  std::string
  issues(const GenerationResult &r)
  {
    std::string text;
    for (const auto &i : r.validation.issues)
      text += i.message + "\n";
    return text;
  }
  double
  cylinder_resistance(const NetworkData &n, Id id)
  {
    const auto  &v = n.vessels.at(id);
    const double l =
      n.nodes.at(v.nodes[0]).position.distance(n.nodes.at(v.nodes[1]).position);
    return 8. * v.viscosity * l / (numbers::PI * std::pow(v.radii[0], 4));
  }
} // namespace
TEST(VascularCompletion, BOTH_ReloadedSharedTrunksRetainBedsInTheDesignSolve)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 2, 2},
                                            Point<3>(),
                                            Point<3>(1., 1., 1.));
  auto        a = initial_archive(bg);
  std::string token;
  if (rank == 0)
    token = std::to_string(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  token = Utilities::MPI::broadcast(comm, token, 0);
  const auto directory =
    ImmersX::TestPaths::output_path("vascular-completion") / token;
  save_network(directory, a.network, a.territories, a.terminal_pairs, comm);
  a = load_network(directory, bg, MappingQ1<3>(), 0., comm);
  Parameters p;
  controls(p);
  const auto r = generate(
    bg, MappingQ1<3>(), a.network, p, comm, a.territories, a.terminal_pairs);
  ASSERT_TRUE(r.validation.valid()) << issues(r);
  const auto expected = physical();
  for (const auto &[id, node] : r.network.nodes)
    if (node.owner == rank && expected.nodes.count(id))
      {
        const auto &old = expected.nodes.at(id);
        EXPECT_EQ(node.position, old.position);
        EXPECT_EQ(node.port, old.port);
        EXPECT_EQ(node.provenance, old.provenance);
        EXPECT_EQ(node.inward_port_normal, old.inward_port_normal);
      }
  double       added_resistance = 0., added_volume = 0.;
  unsigned int old_vessels = 0, new_vessels = 0;
  for (const auto &[id, v] : r.network.vessels)
    if (v.owner == rank)
      {
        if (expected.vessels.count(id))
          {
            ++old_vessels;
            const auto &e = expected.vessels.at(id);
            EXPECT_EQ(v.nodes, e.nodes);
            EXPECT_EQ(v.radii, e.radii);
            EXPECT_DOUBLE_EQ(v.viscosity, e.viscosity);
            EXPECT_EQ(v.provenance, e.provenance);
            EXPECT_EQ(v.physical_properties, e.physical_properties);
          }
        else
          {
            ++new_vessels;
            added_resistance += cylinder_resistance(r.network, id);
            added_volume +=
              numbers::PI * v.radii[0] * v.radii[0] * length(r.network, id);
          }
      }
  EXPECT_EQ(Utilities::MPI::sum(old_vessels, comm), 6u);
  EXPECT_EQ(Utilities::MPI::sum(new_vessels, comm), 2u);
  added_resistance = Utilities::MPI::sum(added_resistance, comm);
  added_volume     = Utilities::MPI::sum(added_volume, comm);
  const double shared =
    cylinder_resistance(expected, 100) + cylinder_resistance(expected, 110);
  const double old_path = cylinder_resistance(expected, 101) + 4.e12 +
                          cylinder_resistance(expected, 111);
  const double old_flow =
    (8000. - shared * p.reference_flow) / (shared + old_path);
  const double new_bed =
    (8000. - shared * (old_flow + p.reference_flow)) / p.reference_flow -
    cylinder_resistance(expected, 102) - cylinder_resistance(expected, 112) -
    added_resistance;
  EXPECT_GT(old_flow, 0.);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<Id>(r.territories.size()), comm),
            2u);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<Id>(r.terminal_pairs.size()), comm),
            2u);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<Id>(r.target_terminal_flows.size()),
                                comm),
            2u);
  for (const auto &pair : r.terminal_pairs)
    {
      const double flow = r.reference.territories.at(pair.territory).flow;
      if (pair.territory == 20)
        {
          EXPECT_DOUBLE_EQ(pair.resistance, 4.e12);
          EXPECT_EQ(pair.arterial, 200u);
          EXPECT_EQ(pair.venous, 210u);
          EXPECT_NEAR(flow / old_flow, 1., 1.e-9);
        }
      else
        {
          EXPECT_NE(pair.territory, 21u);
          EXPECT_NEAR(pair.resistance / new_bed, 1., 1.e-10);
          EXPECT_NEAR(flow / p.reference_flow, 1., 1.e-9);
        }
    }
  double old_volume = 0.;
  for (const auto &[id, v] : expected.vessels)
    old_volume += volume(expected, id);
  EXPECT_NEAR(r.tissue.inside_vessel_volume / (old_volume + added_volume),
              1.,
              1.e-12);
  for (const auto &[id, t] : r.network.terminals)
    {
      EXPECT_NE(id, 201u);
      EXPECT_NE(id, 211u);
    }
  p.calibration.volume_fraction   = r.tissue.volume_fraction * 1.01;
  p.calibration.minimum_coverage  = .9;
  p.calibration.minimum_terminals = 1;
  p.calibration.maximum_terminals = 1;
  p.calibration.attempt_budget    = 1;
  const auto calibrated           = generate(
    bg, MappingQ1<3>(), a.network, p, comm, a.territories, a.terminal_pairs);
  ASSERT_TRUE(calibrated.validation.valid()) << issues(calibrated);
  EXPECT_EQ(calibrated.calibration.status, CalibrationStatus::converged);
  EXPECT_EQ(Utilities::MPI::sum(
              static_cast<Id>(calibrated.terminal_pairs.size()), comm),
            2u);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<Id>(a.terminal_pairs.size()), comm),
            2u);
}
TEST(VascularCompletion, BOTH_HalfExtendedInputBedsAreRejectedCollectively)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 2, 2},
                                            Point<3>(),
                                            Point<3>(1., 1., 1.));
  const auto a = initial_archive(bg, true);
  Parameters p;
  controls(p);
  p.venous.selected_terminals          = {220};
  p.reference_conditions.pressures[20] = 4000.;
  const auto r                         = generate(
    bg, MappingQ1<3>(), a.network, p, comm, a.territories, a.terminal_pairs);
  EXPECT_FALSE(r.validation.valid());
  bool specific = false;
  for (const auto &issue : r.validation.issues)
    specific = specific || issue.message.find("must be extended together") !=
                             std::string::npos;
  EXPECT_EQ(Utilities::MPI::max(static_cast<unsigned int>(specific), comm), 1u)
    << issues(r);
}
TEST(VascularCompletion,
     BOTH_InputMetadataAndVenousModeAreValidatedCollectively)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 2, 2},
                                            Point<3>(),
                                            Point<3>(1., 1., 1.));
  auto       a = initial_archive(bg);
  Parameters p;
  controls(p);
  p.venous.enabled = false;
  EXPECT_THROW(
    generate(
      bg, MappingQ1<3>(), a.network, p, comm, a.territories, a.terminal_pairs),
    ExceptionBase);
  p.venous.enabled = true;
  if (!a.terminal_pairs.empty())
    a.terminal_pairs.front().resistance = -1.;
  EXPECT_THROW(
    generate(
      bg, MappingQ1<3>(), a.network, p, comm, a.territories, a.terminal_pairs),
    ExceptionBase);
}

TEST(VascularCompletion,
     BOTH_UnpairedArterialTerritoriesFollowTheirSurvivingLeaf)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 2, 2},
                                            Point<3>(),
                                            Point<3>(1., 1., 1.));
  NetworkData            input;
  std::vector<Territory> territories;
  if (rank == 0)
    {
      const auto full = physical();
      for (Id id : {0, 1, 2, 3})
        input.nodes.emplace(id, full.nodes.at(id));
      for (Id id : {100, 101, 102})
        input.vessels.emplace(id, full.vessels.at(id));
      for (Id id : {200, 201})
        input.terminals.emplace(id, full.terminals.at(id));
      territories = {{20, input.nodes.at(2).position, 0, 1.},
                     {21, input.nodes.at(3).position, 0, 2.}};
    }
  Parameters p;
  p.seed_length        = .08;
  p.selected_terminals = {200};
  p.terminal_count     = 1;
  p.reference_flow     = 2.e-9;
  const auto r = generate(bg, MappingQ1<3>(), input, p, comm, territories, {});
  ASSERT_TRUE(r.validation.valid()) << issues(r);
  EXPECT_TRUE(r.terminal_pairs.empty());
  EXPECT_EQ(Utilities::MPI::sum(static_cast<Id>(r.territories.size()), comm),
            2u);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<Id>(r.target_terminal_flows.size()),
                                comm),
            1u);
  unsigned int retained = 0;
  for (const auto &t : r.territories)
    {
      EXPECT_NE(t.id, 20u);
      EXPECT_EQ(t.owner, rank);
      if (t.id == 21)
        {
          ++retained;
          EXPECT_DOUBLE_EQ(t.demand, 2.);
          EXPECT_EQ(t.position, Point<3>(.4, .75, .5));
          ASSERT_TRUE(r.network.terminals.count(201));
          EXPECT_EQ(r.network.terminals.at(201).owner, rank);
        }
    }
  EXPECT_EQ(Utilities::MPI::sum(retained, comm), 1u);
}
