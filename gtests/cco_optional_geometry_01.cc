// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>
#include <immersx/vascularization/io.h>
#include <immersx/vascularization/reference_flow.h>

#include <chrono>
#include <cmath>
#include <limits>

#include "../source/vascularization/growth_batch.h"
#include "../source/vascularization/growth_candidate.h"
#include "../source/vascularization/growth_commit.h"
#include "../source/vascularization/growth_start.h"
#include "test_paths.h"

using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  namespace vi = ImmersX::Vascularization::internal;
  unsigned int
  owner()
  {
    return Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1;
  }
  void
  output(const std::string      &stage,
         const NetworkData      &network,
         const ValidationReport &validation = {})
  {
#ifdef DEAL_II_WITH_VTK
    static const std::string run_id = []() {
      std::string token;
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        token = std::to_string(
          std::chrono::high_resolution_clock::now().time_since_epoch().count());
      return Utilities::MPI::broadcast(MPI_COMM_WORLD, token, 0);
    }();
#  ifdef DEBUG
    const std::string configuration = "debug";
#  else
    const std::string configuration = "release";
#  endif
    const auto test = ::testing::UnitTest::GetInstance()->current_test_info();
    write_diagnostic_vtk(ImmersX::TestPaths::output_directory(
                           "cco-optional-geometry/" + configuration + "-mpi" +
                           std::to_string(
                             Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) +
                           "/run-" + run_id + "/" + test->name() + "/" + stage),
                         network,
                         validation,
                         MPI_COMM_WORLD);
#else
    (void)stage;
    (void)network;
    (void)validation;
#endif
  }
  vi::GrowthCandidateContext
  context()
  {
    vi::GrowthCandidateContext c;
    vi::GrowthPathBranch       b;
    b.edge      = {10,
                   {{0, 1}},
                   {{Point<3>(.2, .5, .5), Point<3>(.8, .5, .5)}},
                   .01};
    b.owner     = owner();
    b.viscosity = .004;
    b.demand    = 1.;
    c.path.push_back(b);
    c.root        = 10;
    c.next_node   = 2;
    c.next_vessel = 11;
    c.root_radius = .01;
    c.root_demand = c.minimum_demand = c.total_demand = 1.;
    c.volume_sum                                      = .6;
    c.dissipation_sum                                 = .004 * .6;
    c.forest_volume = numbers::PI * .01 * .01 * .6;
    return c;
  }
  NetworkData
  initial_network()
  {
    NetworkData n;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == owner())
      {
        n.nodes[0] = {Point<3>(.2, .5, .5), owner(), Port::arterial_inlet};
        n.nodes[1] = {Point<3>(.8, .5, .5), owner()};
        Vessel v;
        v.nodes          = {{0, 1}};
        v.owner          = owner();
        v.radii          = {{.01, .01}};
        v.viscosity      = .004;
        v.provenance     = Provenance::generated;
        n.vessels[10]    = v;
        n.terminals[101] = {1, 201, System::arterial, owner(), 1.};
      }
    return n;
  }
  NetworkData
  materialize(const vi::GrowthCandidate &candidate)
  {
    NetworkData n;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != owner())
      return n;
    for (const auto &e : candidate.changed)
      {
        for (unsigned int k = 0; k < 2; ++k)
          n.nodes[e.nodes[k]] = {e.points[k],
                                 owner(),
                                 e.nodes[k] == 0 ? Port::arterial_inlet :
                                                   Port::none,
                                 Provenance::generated};
        Vessel v;
        v.nodes         = e.nodes;
        v.owner         = owner();
        v.radii         = {{e.radius, e.radius}};
        v.viscosity     = .004;
        v.provenance    = Provenance::generated;
        n.vessels[e.id] = v;
      }
    n.terminals[101] = {1, 201, System::arterial, owner(), 1.};
    n.terminals[102] = {3, 202, System::arterial, owner(), 1.};
    return n;
  }
  void
  unchecked(Parameters &p)
  {
    p.enforce_geometry                 = false;
    p.length_scale                     = "meter";
    p.terminal_count_multiplier_factor = "none";
    // These optional geometric thresholds must not become fluid-radius limits.
    p.minimum_length = 100.;
    p.minimum_radius = 100.;
    p.minimum_angle  = 119.;
    p.clearance      = 100.;
    p.reference_flow = 2.e-8;
  }
} // namespace

TEST(VascularCCO, BOTH_UncheckedSkipsGeometricPredicate)
{
  Parameters p("/Optional geometry predicate/");
  unchecked(p);
  output("input", initial_network());
  unsigned int calls     = 0;
  const auto   predicate = [&](const std::vector<vi::Edge> &) {
    ++calls;
    return false;
  };
  const auto c = context();
  const auto proposal =
    vi::propose_growth(c,
                       p,
                       Point<3>(.5, .5, .5),
                       Point<3>(.5, .8, .5),
                       std::numeric_limits<double>::infinity(),
                       1.,
                       predicate);
  EXPECT_EQ(calls, 0u);
  ASSERT_TRUE(proposal.has_value());
  const auto network = materialize(*proposal);
  output("unchecked-proposal", network);
  EXPECT_TRUE(
    validate_data(network, Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
      .valid());
  if (!network.vessels.empty())
    {
      const double parent   = std::pow(network.vessels.at(10).radii[0], 3);
      const double children = std::pow(network.vessels.at(11).radii[0], 3) +
                              std::pow(network.vessels.at(12).radii[0], 3);
      EXPECT_NEAR(children / parent, 1., 1.e-13);
    }
  p.enforce_geometry = true;
  p.minimum_radius = p.minimum_length = 0.;
  const auto rejected =
    vi::propose_growth(c,
                       p,
                       Point<3>(.5, .5, .5),
                       Point<3>(.5, .8, .5),
                       std::numeric_limits<double>::infinity(),
                       1.,
                       predicate);
  EXPECT_GT(calls, 0u);
  EXPECT_FALSE(rejected.has_value());
  // No geometric distance or intersection oracle is invoked by this test.
}

TEST(VascularCCO, BOTH_UncheckedPreservesNativeReferenceFlow)
{
  Parameters p("/Optional geometry reference/");
  unchecked(p);
  output("input", initial_network());
  const auto proposal =
    vi::propose_growth(context(),
                       p,
                       Point<3>(.5, .5, .5),
                       Point<3>(.5, .8, .5),
                       std::numeric_limits<double>::infinity(),
                       1.,
                       [](const auto &) {
                         ADD_FAILURE() << "Unchecked geometry callback invoked";
                         return false;
                       });
  ASSERT_TRUE(proposal.has_value());
  auto network = materialize(*proposal);
  output("physical-input", network);
  ReferenceConditions bc;
  if (!network.nodes.empty())
    bc.pressures = {{0, 1200.}, {1, 200.}, {3, 200.}};
  const auto reference = solve_reference(network, {}, bc, {}, MPI_COMM_WORLD);
  output("reference-result", network, reference.validation);
  ASSERT_TRUE(reference.validation.valid());
  if (!reference.vessels.empty())
    {
      const double r       = .01 / std::cbrt(2.);
      const double root_r  = 8. * .004 * .3 / (numbers::PI * std::pow(.01, 4));
      const double child_r = 8. * .004 * .3 / (numbers::PI * std::pow(r, 4));
      const double q       = 1000. / (root_r + child_r / 2.);
      EXPECT_NEAR(reference.vessels.at(10).flow / q, 1., 1.e-11);
      EXPECT_NEAR(reference.vessels.at(11).flow / q, .5, 1.e-11);
      EXPECT_NEAR(reference.vessels.at(12).flow / q, .5, 1.e-11);
    }
  EXPECT_LT(reference.relative_flow_defect, 1.e-10);
}

TEST(VascularCCO, BOTH_UncheckedExteriorSeedKeepsPhysicalCommit)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_cube(bg, 2);
  MappingQ1<3> mapping;
  Parameters   p("/Optional geometry seed/");
  unchecked(p);
  p.seed_position  = Point<3>(-.2, .5, .5);
  p.seed_direction = Point<3>(-1., 0., 0.);
  p.seed_length    = .2;
  p.seed_radius    = .01;
  p.viscosity      = .004;
  const auto start = vi::start_seed(bg, mapping, p, MPI_COMM_WORLD);
  output("seed-result", start.network, start.validation);
  ASSERT_TRUE(start.prepared);
  const auto committed = vi::commit_growth_splits(bg,
                                                  mapping,
                                                  start.network,
                                                  start.mutable_owned,
                                                  start.fixed_owned_roots,
                                                  {},
                                                  p,
                                                  MPI_COMM_WORLD);
  output("fluid-commit", committed.network, committed.validation);
  EXPECT_TRUE(committed.committed);
  EXPECT_TRUE(committed.validation.valid());
}

TEST(VascularCCO, BOTH_UncheckedRetainsInvalidFluidRejections)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_cube(bg, 2);
  MappingQ1<3> mapping;
  Parameters   p("/Optional geometry invalid fluid/");
  unchecked(p);
  for (unsigned int trial = 0; trial < 2; ++trial)
    {
      NetworkData          network;
      std::set<Id>         mutable_ids;
      std::map<Id, double> roots;
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == owner())
        {
          network.nodes[0] = {Point<3>(.2, .5, .5),
                              owner(),
                              Port::arterial_inlet};
          network.nodes[1] = {Point<3>(.8, .5, .5), owner()};
          Vessel v;
          v.owner             = owner();
          v.nodes             = {{0, 1}};
          v.radii             = trial == 0 ? std::array<double, 2>{{.01, .01}} :
                                             std::array<double, 2>{{1.e-200, 1.e-200}};
          v.viscosity         = trial == 0 ? 0. : .004;
          v.provenance        = Provenance::generated;
          network.vessels[10] = v;
          network.terminals[101] = {1, 201, System::arterial, owner(), 1.};
          mutable_ids.insert(10);
          roots[10] = v.radii[0];
        }
      ValidationReport invalid;
      invalid.issues.push_back(
        {"test input",
         10,
         trial == 0 ? "Zero viscosity" : "Nonrepresentable resistance"});
      output("invalid-input-" + std::to_string(trial), network, invalid);
      const auto result = vi::commit_growth_splits(
        bg, mapping, network, mutable_ids, roots, {}, p, MPI_COMM_WORLD);
      EXPECT_FALSE(result.committed);
      EXPECT_FALSE(result.validation.valid());
    }
}

TEST(VascularCCO, BOTH_UncheckedRejectsNonphysicalSegments)
{
  Parameters p("/Optional geometry positive segments/");
  unchecked(p);
  output("input", initial_network());
  unsigned int calls    = 0;
  const auto   callback = [&](const auto &) {
    ++calls;
    return true;
  };
  const auto c    = context();
  const auto zero = vi::propose_growth(c,
                                       p,
                                       Point<3>(.2, .5, .5),
                                       Point<3>(.5, .8, .5),
                                       std::numeric_limits<double>::infinity(),
                                       1.,
                                       callback);
  EXPECT_FALSE(zero.has_value());
  const auto nonfinite = vi::propose_growth(
    c,
    p,
    Point<3>(.5, .5, .5),
    Point<3>(std::numeric_limits<double>::quiet_NaN(), .8, .5),
    std::numeric_limits<double>::infinity(),
    1.,
    callback);
  EXPECT_FALSE(nonfinite.has_value());
  EXPECT_EQ(calls, 0u);
}

TEST(VascularCCO, MPI_GeometryModeMustMatchBeforeBatchBranches)
{
  const auto comm = MPI_COMM_WORLD;
  ASSERT_GT(Utilities::MPI::n_mpi_processes(comm), 1u);
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_cube(bg, 2);
  MappingQ1<3> mapping;
  Parameters   p("/Optional geometry MPI consensus/");
  unchecked(p);
  p.enforce_geometry = Utilities::MPI::this_mpi_process(comm) != 0;
  vi::BackgroundGeometry geometry(bg, mapping);
  output("empty-input", {});
  EXPECT_ANY_THROW(vi::select_growth_batch(geometry, {}, {}, p, comm, [](Id) {
    return std::array<double, 2>{};
  }));
}

TEST(VascularCCO, BOTH_UncheckedIntegratedThreeSystemGrowth)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_cube(bg, 2);
  MappingQ1<3> mapping;
  Parameters   p("/Optional geometry three-system integration/");
  unchecked(p);
  p.seed_position      = Point<3>(.2, .5, .5);
  p.seed_direction     = Point<3>(1., 0., 0.);
  p.seed_length        = .2;
  p.seed_radius        = .01;
  p.viscosity          = .004;
  p.terminal_count     = 2;
  p.sample_budget      = 32;
  p.candidate_count    = 4;
  p.optimization_steps = 2;
  p.venous.enabled = p.portal.enabled = true;
  p.portal.seed_position              = Point<3>(1.2, .5, .5);
  p.venous.seed_position              = Point<3>(1.3, .6, .5);
  p.portal.seed_radius                = .015;
  p.venous.seed_radius                = .02;
  p.portal.terminal_separation        = .015;
  p.venous.terminal_separation        = .01;
  p.reference_conditions.pressures    = {{0, 10000.}, {2, 500.}, {3, 2000.}};
  const auto result = generate(bg, mapping, {}, p, MPI_COMM_WORLD);
  output("generation-result", result.network, result.validation);
  for (const auto &issue : result.validation.issues)
    ADD_FAILURE() << issue.entity << ": " << issue.message;
  ASSERT_TRUE(result.validation.valid());
  EXPECT_TRUE(result.reference.validation.valid());
  EXPECT_EQ(result.effective_terminal_count, 2u);
  const auto n = Utilities::MPI::sum(
    static_cast<unsigned int>(result.terminal_triplets.size()), MPI_COMM_WORLD);
  EXPECT_EQ(n, 2u);
  EXPECT_LT(result.reference.relative_flow_defect, 1.e-9);
}

TEST(VascularCCO, BOTH_UncheckedExteriorAttachmentIntegratesTissue)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_cube(bg, 2);
  MappingQ1<3> mapping;
  Parameters   p("/Optional geometry exterior tissue bootstrap/");
  unchecked(p);
  p.selected_terminals            = {101};
  p.seed_direction                = Point<3>(1., 0., 0.);
  p.seed_length                   = .1;
  p.viscosity                     = .004;
  p.terminal_count                = 1;
  p.tissue_demand                 = "1+3*x";
  p.coverage.service_radius       = 10.;
  p.calibration.minimum_coverage  = .5;
  p.calibration.minimum_terminals = p.calibration.maximum_terminals = 1;
  p.calibration.attempt_budget                                      = 1;
  p.venous.enabled = p.portal.enabled = true;
  p.portal.seed_position              = Point<3>(1.2, .5, .5);
  p.venous.seed_position              = Point<3>(1.3, .6, .5);
  p.portal.seed_radius                = .015;
  p.venous.seed_radius                = .02;
  p.portal.terminal_separation        = .015;
  p.venous.terminal_separation        = .01;
  p.reference_conditions.pressures    = {{0, 10000.}, {3, 500.}, {4, 2000.}};
  NetworkData initial;
  if (Utilities::MPI::this_mpi_process(comm) == owner())
    {
      initial.nodes[0] = {Point<3>(-.6, .5, .5), owner(), Port::arterial_inlet};
      initial.nodes[1] = {Point<3>(-.2, .5, .5), owner()};
      Vessel vessel;
      vessel.nodes           = {{0, 1}};
      vessel.radii           = {{.01, .01}};
      vessel.viscosity       = .004;
      vessel.owner           = owner();
      initial.vessels[10]    = vessel;
      initial.terminals[101] = {1, 201, System::arterial, owner(), 1.};
    }
  output("exterior-input", initial);
  const auto result = generate(bg, mapping, initial, p, comm);
  output("integrated-result", result.network, result.validation);
  for (const auto &issue : result.validation.issues)
    ADD_FAILURE() << issue.entity << ": " << issue.message;
  ASSERT_TRUE(result.validation.valid());
  EXPECT_TRUE(result.reference.validation.valid());
  EXPECT_EQ(result.calibration.status, CalibrationStatus::converged);
  EXPECT_EQ(result.calibration.attempts.size(), 1u);
  EXPECT_EQ(result.effective_terminal_count, 1u);
  EXPECT_TRUE(result.tissue.coverage_evaluated);
  EXPECT_NEAR(result.tissue.background_volume, 1., 1.e-13);
  EXPECT_NEAR(result.tissue.coverage_quadrature_volume, 1., 1.e-13);
  EXPECT_NEAR(result.tissue.arterial_coverage, 1., 1.e-13);
  std::array<double, 3> demand{}, terminal_flow{};
  unsigned int          leaves = 0;
  const auto            rank   = Utilities::MPI::this_mpi_process(comm);
  for (const auto &[id, terminal] : result.network.terminals)
    if (terminal.owner == rank)
      {
        ++leaves;
        demand[static_cast<unsigned int>(terminal.system)] += terminal.demand;
      }
  for (const auto &[id, flow] : result.reference.terminal_outflows)
    terminal_flow[static_cast<unsigned int>(
      result.network.terminals.at(id).system)] += flow;
  // The demand integral over the unit cube is independent of its exterior
  // centre.
  const std::array<double, 3> fractions{{.5, -1., .5}};
  for (unsigned int system = 0; system < 3; ++system)
    {
      EXPECT_NEAR(Utilities::MPI::sum(demand[system], comm), 2.5, 1.e-13);
      EXPECT_NEAR(Utilities::MPI::sum(terminal_flow[system], comm) /
                    p.reference_flow,
                  fractions[system],
                  1.e-12);
    }
  EXPECT_EQ(Utilities::MPI::sum(leaves, comm), 3u);
  EXPECT_EQ(Utilities::MPI::sum(
              static_cast<unsigned int>(result.terminal_triplets.size()), comm),
            1u);
  EXPECT_NEAR(result.tissue.triple_coverage, 1., 1.e-13);
  EXPECT_LT(result.reference.relative_flow_defect, 1.e-9);
}
