// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>
#include <immersx/vascularization/reference_flow.h>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  void
  edge(NetworkData &n,
       Id           id,
       Id           a,
       Id           b,
       double       mu     = .0035,
       double       radius = .001,
       System       system = System::arterial)
  {
    Vessel e;
    e.nodes       = {{a, b}};
    e.radii       = {{radius, radius}};
    e.viscosity   = mu;
    e.system      = system;
    n.vessels[id] = e;
  }
  NetworkData
  chain()
  {
    NetworkData n;
    n.nodes[10].position = Point<3>(0., 0., 0.);
    n.nodes[20].position = Point<3>(.03, 0., 0.);
    n.nodes[30].position = Point<3>(.05, 0., 0.);
    n.nodes[10].port     = Port::arterial_inlet;
    n.nodes[30].port     = Port::arterial_outlet;
    edge(n, 100, 10, 20);
    n.vessels[100].radii = {{.001, .0008}};
    edge(n, 101, 20, 30);
    n.vessels[101].radii = {{.0008, .0006}};
    return n;
  }
  double
  taper(double l, double r0, double r1, double mu = .0035)
  {
    return 8. * mu * l / (3. * numbers::PI * (r1 - r0)) *
           (std::pow(r0, -3) - std::pow(r1, -3));
  }
  NetworkData
  av()
  {
    NetworkData n;
    n.nodes[0].position = Point<3>(0., 0., 0.);
    n.nodes[1].position = Point<3>(.03, 0., 0.);
    n.nodes[2].position = Point<3>(.1, .02, 0.);
    n.nodes[3].position = Point<3>(.04, .02, 0.);
    n.nodes[0].port     = Port::arterial_inlet;
    n.nodes[2].port     = Port::venous_outlet;
    edge(n, 100, 0, 1);
    edge(n, 101, 2, 3, .004, .0012, System::venous);
    n.terminals[11] = {1, 99, System::arterial, 0, 2.};
    n.terminals[12] = {3, 99, System::venous, 0, 2.};
    return n;
  }
  std::vector<TerminalPair>
  bed()
  {
    return {{99, 11, 12, 6000. / 2.e-9}};
  }
  ReferenceResult
  solve(const NetworkData               &n,
        const ReferenceConditions       &bc,
        const std::vector<TerminalPair> &pairs = {},
        const ReferenceParameters       &p     = {})
  {
    return solve_reference(n, pairs, bc, p, MPI_COMM_WORLD);
  }
} // namespace

TEST(VascularReferenceFlow, TaperedSeriesHasAnalyticPressureFlowAndShear)
{
  const auto   n  = chain();
  const double r0 = taper(.03, .001, .0008), r1 = taper(.02, .0008, .0006);
  const double q     = 8000. / (r0 + r1);
  const auto   state = solve(n, {{{10, 12000.}, {30, 4000.}}, {}});
  ASSERT_TRUE(state.validation.valid());
  EXPECT_NEAR(state.vessels.at(100).resistance / r0, 1., 1.e-13);
  EXPECT_NEAR(state.vessels.at(101).resistance / r1, 1., 1.e-13);
  EXPECT_NEAR(state.vessels.at(100).flow / q, 1., 1.e-12);
  EXPECT_NEAR(state.vessels.at(101).flow / q, 1., 1.e-12);
  EXPECT_NEAR(state.pressures.at(20), 12000. - q * r0, 1.e-8);
  EXPECT_NEAR(state.boundary_inflows.at(10) / q, 1., 1.e-12);
  EXPECT_NEAR(state.boundary_inflows.at(30) / q, -1., 1.e-12);
  EXPECT_NEAR(state.dissipated_power / (8000. * q), 1., 1.e-12);
  for (const auto &[id, e] : n.vessels)
    for (unsigned int k = 0; k < 2; ++k)
      EXPECT_NEAR(state.vessels.at(id).wall_shear[k] /
                    (4. * .0035 * q / (numbers::PI * std::pow(e.radii[k], 3))),
                  1.,
                  1.e-12);
  const double half_r = taper(.015, .001, .0009);
  EXPECT_NEAR(reference_pressure(n, state, 100, .5),
              12000. - q * half_r,
              1.e-8);
  EXPECT_DOUBLE_EQ(reference_pressure(n, state, 100, 0.), 12000.);
  EXPECT_DOUBLE_EQ(reference_pressure(n, state, 101, 1.), 4000.);
  EXPECT_LT(state.relative_flow_defect, 1.e-11);
  EXPECT_THROW(reference_pressure(n, state, 100, -.1), ExceptionBase);
}

TEST(VascularReferenceFlow, ParallelPathsConserveAnalyticalFlow)
{
  NetworkData n;
  n.nodes[0].position = Point<3>(0., 0., 0.);
  n.nodes[1].position = Point<3>(1., 1., 0.);
  n.nodes[2].position = Point<3>(1., -1., 0.);
  n.nodes[3].position = Point<3>(2., 0., 0.);
  n.nodes[0].port     = Port::arterial_inlet;
  n.nodes[3].port     = Port::arterial_outlet;
  edge(n, 0, 0, 1, .004, .005);
  edge(n, 1, 1, 3, .004, .005);
  edge(n, 2, 0, 2, .008, .005);
  edge(n, 3, 2, 3, .008, .005);
  const double r =
    8. * .004 * std::sqrt(2.) / (numbers::PI * std::pow(.005, 4));
  const double q     = 8000. / (2. * r);
  const auto   state = solve(n, {{{0, 12000.}, {3, 4000.}}, {}});
  ASSERT_TRUE(state.validation.valid());
  EXPECT_NEAR(state.vessels.at(0).flow / q, 1., 1.e-12);
  EXPECT_NEAR(state.vessels.at(1).flow / q, 1., 1.e-12);
  EXPECT_NEAR(state.vessels.at(2).flow / q, .5, 1.e-12);
  EXPECT_NEAR(state.vessels.at(3).flow / q, .5, 1.e-12);
  EXPECT_NEAR(state.boundary_inflows.at(0) / q, 1.5, 1.e-12);
  EXPECT_NEAR(state.pressures.at(1), 8000., 1.e-8);
  EXPECT_NEAR(state.pressures.at(2), 8000., 1.e-8);
  EXPECT_NEAR(state.dissipated_power / (8000. * q * 1.5), 1., 1.e-12);
}

TEST(VascularReferenceFlow, PrescribedFlowAndPressureGaugeAreIndependent)
{
  const auto   n = chain();
  const double q = 2.e-9,
               r = taper(.03, .001, .0008) + taper(.02, .0008, .0006);
  ReferenceConditions bc{{{30, 5000.}}, {{10, q}}};
  const auto          state = solve(n, bc);
  ASSERT_TRUE(state.validation.valid());
  EXPECT_NEAR(state.pressures.at(10), 5000. + q * r, 1.e-8);
  EXPECT_NEAR(state.vessels.at(100).flow / q, 1., 1.e-11);
  EXPECT_NEAR(state.dissipated_power / (q * q * r), 1., 1.e-11);
  bc.pressures.at(30) += 1.e9;
  const auto shifted = solve(n, bc);
  ASSERT_TRUE(shifted.validation.valid());
  EXPECT_DOUBLE_EQ(shifted.vessels.at(100).flow, state.vessels.at(100).flow);
  EXPECT_DOUBLE_EQ(shifted.vessels.at(101).flow, state.vessels.at(101).flow);
  EXPECT_NEAR(shifted.pressures.at(10) - state.pressures.at(10), 1.e9, 2.e-7);
  EXPECT_DOUBLE_EQ(shifted.pressures.at(30), bc.pressures.at(30));
  const auto equilibrium = solve(n, {{{10, 5000.}, {30, 5000.}}, {}});
  ASSERT_TRUE(equilibrium.validation.valid());
  EXPECT_DOUBLE_EQ(equilibrium.dissipated_power, 0.);
  EXPECT_DOUBLE_EQ(equilibrium.maximum_flow_defect, 0.);
  EXPECT_EQ(equilibrium.iterations, 0u);
}

TEST(VascularReferenceFlow, ArteriovenousBedReportsPerfusionAndNoAddedVolume)
{
  const auto   n     = av();
  const auto   pairs = bed();
  const double ra    = 8. * .0035 * .03 / (numbers::PI * std::pow(.001, 4));
  const double rv    = 8. * .004 * .06 / (numbers::PI * std::pow(.0012, 4));
  const double q     = 8000. / (ra + pairs[0].resistance + rv);
  const double v     = volume(n, 100) + volume(n, 101);
  const auto   state = solve(n, {{{0, 12000.}, {2, 4000.}}, {}}, pairs);
  ASSERT_TRUE(state.validation.valid());
  EXPECT_NEAR(state.vessels.at(100).flow / q, 1., 1.e-12);
  EXPECT_NEAR(state.vessels.at(101).flow / q, -1., 1.e-12);
  EXPECT_NEAR(state.pressures.at(1), 12000. - q * ra, 1.e-8);
  EXPECT_NEAR(state.pressures.at(3), 4000. + q * rv, 1.e-8);
  EXPECT_NEAR(state.territories.at(99).flow / q, 1., 1.e-12);
  EXPECT_NEAR(state.territories.at(99).flow_per_demand / q, .5, 1.e-12);
  EXPECT_NEAR(state.terminal_outflows.at(11) / q, 1., 1.e-9);
  EXPECT_NEAR(state.terminal_outflows.at(12) / q, -1., 1.e-9);
  EXPECT_NEAR(state.dissipated_power / (8000. * q), 1., 1.e-12);
  EXPECT_DOUBLE_EQ(volume(n, 100) + volume(n, 101), v);
  EXPECT_EQ(n.nodes.size(), 4u);
  EXPECT_EQ(n.vessels.size(), 2u);
}

TEST(VascularReferenceFlow,
     MultipleInputsAndOutletsHaveAnalyticalTotalResistance)
{
  NetworkData                 n;
  const std::vector<Point<3>> points{{0., 0., 0.},
                                     {0., 2., 0.},
                                     {1., 1., 0.},
                                     {2., .5, 0.},
                                     {2., 1.5, 0.},
                                     {5., 0., 0.},
                                     {5., 2., 0.},
                                     {4., 1., 0.},
                                     {3., .5, 0.},
                                     {3., 1.5, 0.}};
  for (Id i = 0; i < points.size(); ++i)
    n.nodes[i].position = points[i];
  for (Id id : {0, 1})
    n.nodes[id].port = Port::arterial_inlet;
  for (Id id : {5, 6})
    n.nodes[id].port = Port::venous_outlet;
  const std::vector<std::array<Id, 2>> ends{{{0, 2}},
                                            {{1, 2}},
                                            {{2, 3}},
                                            {{2, 4}},
                                            {{5, 7}},
                                            {{6, 7}},
                                            {{7, 8}},
                                            {{7, 9}}};
  const double                         r = 1.e12, radius = .001;
  for (unsigned int i = 0; i < ends.size(); ++i)
    {
      const auto [a, b] = ends[i];
      const double mu   = r * numbers::PI * std::pow(radius, 4) /
                        (8. * points[a].distance(points[b]));
      edge(n,
           100 + i,
           a,
           b,
           mu,
           radius,
           i < 4 ? System::arterial : System::venous);
    }
  n.terminals[31] = {3, 1001, System::arterial, 0, 1.};
  n.terminals[32] = {4, 1002, System::arterial, 0, 2.};
  n.terminals[41] = {8, 1001, System::venous, 0, 1.};
  n.terminals[42] = {9, 1002, System::venous, 0, 2.};
  const std::vector<TerminalPair> pairs{{1001, 31, 41, 2. * r},
                                        {1002, 32, 42, 2. * r}};
  const auto                      state =
    solve(n, {{{0, 12000.}, {1, 12000.}, {5, 4000.}, {6, 4000.}}, {}}, pairs);
  ASSERT_TRUE(state.validation.valid());
  const double q = 8000. / (3. * r);
  EXPECT_NEAR((state.boundary_inflows.at(0) + state.boundary_inflows.at(1)) / q,
              1.,
              1.e-12);
  EXPECT_NEAR((state.boundary_inflows.at(5) + state.boundary_inflows.at(6)) / q,
              -1.,
              1.e-12);
  EXPECT_NEAR(state.territories.at(1001).flow / q, .5, 1.e-12);
  EXPECT_NEAR(state.territories.at(1002).flow / q, .5, 1.e-12);
  EXPECT_NEAR(state.territories.at(1002).flow_per_demand / q, .25, 1.e-12);
  EXPECT_NEAR(state.pressures.at(2), 12000. - 8000. / 6., 1.e-8);
  EXPECT_NEAR(state.pressures.at(7), 4000. + 8000. / 6., 1.e-8);
  EXPECT_LT(state.relative_flow_defect, 1.e-10);
}

TEST(VascularReferenceFlow, GeneratedNetworkAndNativeRefinementPreservePhysics)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  p.terminal_count_multiplier_factor = "none";

  p.seed_position  = Point<3>(.2, .5, .5);
  p.seed_direction = Point<3>(1., 0., 0.);
  p.seed_radius    = .005;
  p.seed_length    = .3;
  p.viscosity      = .004;
  p.terminal_count = 8;
  const auto generated =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(generated.validation.valid());
  ReferenceConditions bc;
  bc.inflows[0] = 1.e-9;
  for (const auto &[id, t] : generated.network.terminals)
    bc.pressures[t.node] = 4000.;
  const auto state = solve(generated.network, bc);
  ASSERT_TRUE(state.validation.valid());
  double terminal_flow = 0.;
  for (const auto &[id, q] : state.terminal_outflows)
    {
      EXPECT_GT(q, 0.);
      terminal_flow += q;
    }
  EXPECT_NEAR(terminal_flow / 1.e-9, 1., 1.e-9);
  Triangulation<1, 3> mesh;
  const auto          refs =
    create_triangulation(generated.network, generated.discretization, mesh);
  const auto coarse = mesh.n_active_cells();
  mesh.refine_global(2);
  EXPECT_EQ(mesh.n_active_cells(), 4 * coarse);
  std::map<Id, double> drops;
  for (const auto &cell : mesh.active_cell_iterators())
    {
      const auto   ref = physical_reference(cell->id(), refs);
      const double begin =
        reference_pressure(generated.network, state, ref.vessel, ref.begin);
      const double end =
        reference_pressure(generated.network, state, ref.vessel, ref.end);
      const auto  &physical = generated.network.vessels.at(ref.vessel);
      const double cell_resistance =
        8. * physical.viscosity * cell->measure() /
        (numbers::PI * std::pow(physical.radii[0], 4));
      const double expected =
        state.vessels.at(ref.vessel).flow * cell_resistance;
      // Absolute pressure values are doubles. Their subtraction incurs a
      // roundoff bound set by the pressure gauge, not by the tiny cell drop.
      const double rounding = 4. * std::numeric_limits<double>::epsilon() *
                              std::max(std::abs(begin), std::abs(end));
      EXPECT_NEAR(begin - end,
                  expected,
                  rounding + 1.e-12 * std::abs(expected));
      drops[ref.vessel] += begin - end;
    }
  const auto again = solve(generated.network, bc);
  ASSERT_TRUE(again.validation.valid());
  for (const auto &[id, e] : state.vessels)
    {
      const auto  &physical = generated.network.vessels.at(id);
      const double rounding =
        4. * std::numeric_limits<double>::epsilon() *
        std::max(std::abs(state.pressures.at(physical.nodes[0])),
                 std::abs(state.pressures.at(physical.nodes[1])));
      EXPECT_NEAR(drops.at(id),
                  e.flow * e.resistance,
                  rounding + 1.e-12 * std::abs(e.flow * e.resistance));
      EXPECT_DOUBLE_EQ(e.flow, again.vessels.at(id).flow);
    }
}

TEST(VascularReferenceFlow,
     InvalidBoundaryConditionsAndMissingGaugesAreRejected)
{
  auto n = chain();
  EXPECT_FALSE(
    solve(n, {{{10, 12000.}, {30, 4000.}}, {{10, 1.e-9}}}).validation.valid());
  EXPECT_FALSE(solve(n, {{}, {{10, 1.e-9}, {30, -1.e-9}}}).validation.valid());
  EXPECT_FALSE(solve(n, {{{999, 0.}}, {}}).validation.valid());
  EXPECT_FALSE(
    solve(n, {{{30, 0.}}, {{10, std::numeric_limits<double>::quiet_NaN()}}})
      .validation.valid());
  EXPECT_FALSE(
    solve(n, {{{20, 6000.}, {30, 4000.}}, {{10, 1.e-9}}}).validation.valid());
  n.nodes[99].position = Point<3>(0., 1., 0.);
  n.nodes[99].port     = Port::arterial_outlet;
  EXPECT_FALSE(
    solve(n, {{{30, 0.}, {99, 0.}}, {{10, 1.e-9}}}).validation.valid());
  const auto a = av();
  auto bc      = ReferenceConditions{{{0, 12000.}, {2, 4000.}, {1, 8000.}}, {}};
  EXPECT_FALSE(solve(a, bc, bed()).validation.valid());
}

TEST(VascularReferenceFlow, InvalidPairsAreRejectedBeforeAssembly)
{
  auto       n     = av();
  const auto pairs = bed();
  ASSERT_TRUE(validate_terminal_pairs(n, pairs).valid());
  for (const double r : {0.,
                         -1.,
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()})
    {
      auto bad          = pairs;
      bad[0].resistance = r;
      EXPECT_FALSE(validate_terminal_pairs(n, bad).valid());
    }
  auto duplicate = pairs;
  duplicate.push_back(pairs[0]);
  EXPECT_FALSE(validate_terminal_pairs(n, duplicate).valid());
  auto bad        = pairs;
  bad[0].arterial = 999;
  EXPECT_FALSE(validate_terminal_pairs(n, bad).valid());
  bad = pairs;
  std::swap(bad[0].arterial, bad[0].venous);
  EXPECT_FALSE(validate_terminal_pairs(n, bad).valid());
  n.terminals.at(12).demand = 3.;
  EXPECT_FALSE(validate_terminal_pairs(n, pairs).valid());
  n.terminals.at(12).demand    = 2.;
  n.terminals.at(12).territory = 100;
  EXPECT_FALSE(validate_terminal_pairs(n, pairs).valid());
  n.terminals.at(12).territory = 99;
  const auto old               = n.nodes.at(3).position;
  n.nodes.at(3).position       = n.nodes.at(1).position;
  EXPECT_FALSE(validate_terminal_pairs(n, pairs).valid());
  n.nodes.at(3).position = old;
  n.nodes[4].position    = Point<3>(.04, 0., 0.);
  edge(n, 102, 1, 4);
  EXPECT_FALSE(validate_terminal_pairs(n, pairs).valid());
}

TEST(VascularReferenceFlow, ReversedFlowAndUnperfusedBedsAreInvalid)
{
  const auto n = av();
  EXPECT_FALSE(
    solve(n, {{{0, 4000.}, {2, 4000.}}, {}}, bed()).validation.valid());
  const auto reverse = solve(n, {{{0, 3000.}, {2, 4000.}}, {}}, bed());
  EXPECT_FALSE(reverse.validation.valid());
  EXPECT_LT(reverse.territories.at(99).flow, 0.);
  EXPECT_LT(reverse.vessels.at(100).flow, 0.);
  EXPECT_GT(reverse.vessels.at(101).flow, 0.);
}

TEST(VascularReferenceFlow, FiniteSolverBudgetAndUnrepresentableResistance)
{
  NetworkData n;
  for (Id i = 0; i < 10; ++i)
    n.nodes[i].position = Point<3>(.1 * i, 0., 0.);
  for (Id i = 0; i < 9; ++i)
    edge(n, 100 + i, i, i + 1, .004 * (i + 1));
  n.nodes[0].port = Port::arterial_inlet;
  n.nodes[9].port = Port::arterial_outlet;
  ReferenceParameters p;
  p.maximum_iterations = 1;
  const auto exhausted = solve(n, {{{9, 0.}}, {{0, 1.e-9}}}, {}, p);
  EXPECT_FALSE(exhausted.validation.valid());
  EXPECT_EQ(exhausted.iterations, 1u);
  auto small                  = chain();
  small.vessels.at(100).radii = {{1.e-100, 1.e-100}};
  EXPECT_FALSE(
    solve(small, {{{10, 12000.}, {30, 4000.}}, {}}).validation.valid());
  EXPECT_DOUBLE_EQ(small.vessels.at(100).radii[0], 1.e-100);
  p.maximum_iterations = 0;
  EXPECT_THROW(p.validate(), ExceptionBase);
  p.maximum_iterations        = 10;
  p.linear_relative_tolerance = 0.;
  EXPECT_THROW(p.validate(), ExceptionBase);
}

TEST(VascularReferenceFlow, ResistiveLiftingPreservesTinyDropsBesideLargeBeds)
{
  NetworkData n;
  for (Id i = 0; i < 3; ++i)
    n.nodes[i].position = Point<3>(.01 * i, 0., 0.);
  n.nodes[0].port = Port::arterial_inlet;
  edge(n, 0, 0, 1, .004, .01);
  edge(n, 1, 1, 2, .004, .01);
  for (Id i = 0; i < 6; ++i)
    n.nodes[10 + i].position = Point<3>(.1 - .01 * i, .02, 0.);
  n.nodes[10].port = Port::venous_outlet;
  for (Id i = 0; i < 5; ++i)
    edge(n, 10 + i, 10 + i, 11 + i, .004, .01, System::venous);
  n.terminals[100]                 = {2, 500, System::arterial, 0, 1.};
  n.terminals[200]                 = {15, 500, System::venous, 0, 1.};
  const double bed_resistance      = 4.e12;
  double       expected_resistance = bed_resistance;
  for (const auto &[id, e] : n.vessels)
    expected_resistance +=
      8. * .004 * length(n, id) / (numbers::PI * std::pow(.01, 4));
  const double q = 8000. / expected_resistance;
  const auto   r = solve(n,
                         {{{0, 12000.}, {10, 4000.}}, {}},
                         {{500, 100, 200, bed_resistance}});
  ASSERT_TRUE(r.validation.valid());
  EXPECT_NEAR(r.territories.at(500).flow / q, 1., 1.e-12);
  for (const auto &[id, e] : n.vessels)
    {
      const double direction = e.system == System::arterial ? 1. : -1.;
      EXPECT_NEAR(r.vessels.at(id).flow / (direction * q), 1., 1.e-10);
    }
  EXPECT_LE(r.relative_flow_defect, 1.e-9);
}

namespace
{
  NetworkData
  reference_partition(const NetworkData &full, unsigned int active_ranks = 0)
  {
    const auto  rank   = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto  size   = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const auto  active = active_ranks == 0 ? size : active_ranks;
    NetworkData result;
    const auto  add_node = [&](Id id) {
      auto node        = full.nodes.at(id);
      node.owner       = id % active;
      result.nodes[id] = node;
    };
    for (const auto &[id, node] : full.nodes)
      if (id % active == rank)
        add_node(id);
    for (const auto &[id, v] : full.vessels)
      if ((id + 1) % active == rank)
        {
          auto owned         = v;
          owned.owner        = rank;
          result.vessels[id] = owned;
          for (Id node : v.nodes)
            add_node(node);
        }
    for (const auto &[id, t] : full.terminals)
      if (t.node % active == rank)
        {
          auto owned           = t;
          owned.owner          = rank;
          result.terminals[id] = owned;
          add_node(t.node);
        }
    return result;
  }

  NetworkData
  large_id_chain()
  {
    constexpr Id base     = (Id{1} << 60) + 37;
    const auto   original = chain();
    NetworkData  n;
    for (const auto &[id, node] : original.nodes)
      n.nodes[base + id] = node;
    for (const auto &[id, v] : original.vessels)
      {
        auto moved           = v;
        moved.nodes          = {{base + v.nodes[0], base + v.nodes[1]}};
        n.vessels[base + id] = moved;
      }
    return n;
  }
} // namespace

TEST(VascularReferenceFlow, BOTH_DistributedTaperedSeriesUsesPhysicalLargeIds)
{
  constexpr Id        base = (Id{1} << 60) + 37;
  const auto          comm = MPI_COMM_WORLD;
  const auto          rank = Utilities::MPI::this_mpi_process(comm);
  const auto          size = Utilities::MPI::n_mpi_processes(comm);
  const auto          full = large_id_chain();
  const auto          n    = reference_partition(full);
  ReferenceConditions bc;
  if (rank == size - 1)
    bc.pressures[base + 10] = 12000.;
  if (rank == 0)
    bc.pressures[base + 30] = 4000.;
  const auto state = solve_reference(n, {}, bc, {}, comm);
  ASSERT_TRUE(state.validation.valid());
  const double r0 = taper(.03, .001, .0008), r1 = taper(.02, .0008, .0006);
  const double q             = 8000. / (r0 + r1);
  unsigned int owned_vessels = 0;
  for (const auto &[id, v] : n.vessels)
    {
      ASSERT_TRUE(state.vessels.count(id));
      ++owned_vessels;
      EXPECT_NEAR(state.vessels.at(id).flow / q, 1., 1.e-12);
      EXPECT_NEAR(state.vessels.at(id).resistance /
                    (id == base + 100 ? r0 : r1),
                  1.,
                  1.e-13);
      EXPECT_EQ(v.radii, full.vessels.at(id).radii);
      const double expected = id == base + 100 ?
                                12000. - q * taper(.015, .001, .0009) :
                                12000. - q * (r0 + taper(.01, .0008, .0007));
      EXPECT_NEAR(reference_pressure(n, state, id, .5), expected, 1.e-8);
      for (unsigned int end = 0; end < 2; ++end)
        EXPECT_NEAR(state.vessels.at(id).wall_shear[end] /
                      (4. * v.viscosity * q /
                       (numbers::PI * std::pow(v.radii[end], 3))),
                    1.,
                    1.e-12);
    }
  EXPECT_EQ(Utilities::MPI::sum(owned_vessels, comm), 2u);
  if (state.pressures.count(base + 20))
    {
      EXPECT_NEAR(state.pressures.at(base + 20), 12000. - q * r0, 1.e-8);
    }
  EXPECT_NEAR(state.dissipated_power / (8000. * q), 1., 1.e-12);
  EXPECT_LT(state.relative_flow_defect, 1.e-11);
  EXPECT_NEAR(state.boundary_flow_balance / q, 0., 1.e-11);
  EXPECT_EQ(Utilities::MPI::sum(
              static_cast<unsigned int>(state.boundary_inflows.size()), comm),
            2u);
  for (const auto &[id, flow] : state.boundary_inflows)
    EXPECT_NEAR(flow / q, id == base + 10 ? 1. : -1., 1.e-12);
}

TEST(VascularReferenceFlow,
     BOTH_SparseBedLookupPreservesStiffPerfusionAndEmptyRanks)
{
  const auto  comm = MPI_COMM_WORLD;
  const auto  rank = Utilities::MPI::this_mpi_process(comm);
  const auto  size = Utilities::MPI::n_mpi_processes(comm);
  NetworkData full;
  for (Id i = 0; i < 3; ++i)
    full.nodes[i].position = Point<3>(.01 * i, 0., 0.);
  full.nodes[0].port = Port::arterial_inlet;
  edge(full, 0, 0, 1, .004, .01);
  edge(full, 1, 1, 2, .004, .01);
  for (Id i = 0; i < 6; ++i)
    full.nodes[10 + i].position = Point<3>(.1 - .01 * i, .02, 0.);
  full.nodes[10].port = Port::venous_outlet;
  for (Id i = 0; i < 5; ++i)
    edge(full, 10 + i, 10 + i, 11 + i, .004, .01, System::venous);
  full.terminals[100] = {2, 500, System::arterial, 0, 1.};
  full.terminals[200] = {15, 500, System::venous, 0, 1.};
  const auto n        = reference_partition(full, std::max(1u, size - 1));
  std::vector<TerminalPair> pairs;
  if (rank == size - 1)
    pairs.push_back({500, 100, 200, 4.e12});
  if (size > 1 && rank == size - 1)
    {
      EXPECT_TRUE(n.nodes.empty());
      EXPECT_TRUE(n.vessels.empty());
      EXPECT_TRUE(n.terminals.empty());
    }
  const ReferenceConditions bc{{{0, 12000.}, {10, 4000.}}, {}};
  const auto                state = solve_reference(n, pairs, bc, {}, comm);
  ASSERT_TRUE(state.validation.valid());
  const double resistance =
    4.e12 + 7. * (8. * .004 * .01 / (numbers::PI * std::pow(.01, 4)));
  const double q = 8000. / resistance;
  for (const auto &[id, v] : n.vessels)
    {
      const double sign = v.system == System::arterial ? 1. : -1.;
      EXPECT_NEAR(state.vessels.at(id).flow / (sign * q), 1., 1.e-10);
    }
  if (rank == size - 1)
    {
      EXPECT_NEAR(state.territories.at(500).flow / q, 1., 1.e-12);
      EXPECT_GT(state.territories.at(500).flow, 0.);
      EXPECT_NEAR(state.pressures.at(2),
                  12000. - q * (2. * 8. * .004 * .01 /
                                (numbers::PI * std::pow(.01, 4))),
                  1.e-8);
      EXPECT_NEAR(state.pressures.at(15),
                  4000. + q * (5. * 8. * .004 * .01 /
                               (numbers::PI * std::pow(.01, 4))),
                  1.e-8);
    }
  EXPECT_EQ(Utilities::MPI::sum(
              static_cast<unsigned int>(state.territories.size()), comm),
            1u);
  for (const auto &[id, t] : n.terminals)
    EXPECT_NEAR(state.terminal_outflows.at(id) / q,
                t.system == System::arterial ? 1. : -1.,
                1.e-9);
  EXPECT_NEAR(state.dissipated_power / (8000. * q), 1., 1.e-12);
  EXPECT_LE(state.relative_flow_defect, 1.e-9);
}

TEST(VascularReferenceFlow, BOTH_DisconnectedAnchoredNetworksKeepSparseResults)
{
  const auto          comm = MPI_COMM_WORLD;
  const auto          size = Utilities::MPI::n_mpi_processes(comm);
  NetworkData         full;
  ReferenceConditions bc;
  for (Id component = 0; component < 16; ++component)
    {
      const auto original = chain();
      const Id   shift    = component * 1000;
      for (const auto &[id, node] : original.nodes)
        full.nodes[shift + id] = node;
      for (const auto &[id, v] : original.vessels)
        {
          auto moved               = v;
          moved.nodes              = {{shift + v.nodes[0], shift + v.nodes[1]}};
          full.vessels[shift + id] = moved;
        }
      bc.pressures[shift + 10] = 12000.;
      bc.pressures[shift + 30] = 4000.;
    }
  const auto n     = reference_partition(full);
  const auto state = solve_reference(n, {}, bc, {}, comm);
  ASSERT_TRUE(state.validation.valid());
  const double q = 8000. / (taper(.03, .001, .0008) + taper(.02, .0008, .0006));
  EXPECT_NEAR(state.dissipated_power / (16. * 8000. * q), 1., 1.e-12);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(state.vessels.size()),
                                comm),
            32u);
  if (size > 1)
    {
      EXPECT_LT(Utilities::MPI::min(
                  static_cast<unsigned int>(state.pressures.size()), comm),
                full.nodes.size());
    }
  for (const auto &[id, v] : state.vessels)
    EXPECT_NEAR(v.flow / q, 1., 1.e-12);
  const auto repeated = solve_reference(n, {}, bc, {}, comm);
  ASSERT_TRUE(repeated.validation.valid());
  for (const auto &[id, v] : state.vessels)
    EXPECT_DOUBLE_EQ(v.flow, repeated.vessels.at(id).flow);
}

TEST(VascularReferenceFlow, BOTH_ReferenceFailuresAreCollective)
{
  const auto          comm = MPI_COMM_WORLD;
  const auto          rank = Utilities::MPI::this_mpi_process(comm);
  const auto          size = Utilities::MPI::n_mpi_processes(comm);
  auto                n    = reference_partition(chain());
  ReferenceConditions bc{{{10, 12000.}, {30, 4000.}}, {}};
  auto                bad_bc = bc;
  if (rank == size - 1)
    bad_bc.pressures[999] = 0.;
  EXPECT_FALSE(solve_reference(n, {}, bad_bc, {}, comm).validation.valid());
  bad_bc = bc;
  if (rank == size - 1)
    bad_bc.inflows[10] = 1.e-9;
  EXPECT_FALSE(solve_reference(n, {}, bad_bc, {}, comm).validation.valid());
  EXPECT_FALSE(
    solve_reference(n, {}, {{}, {{10, 1.e-9}, {30, -1.e-9}}}, {}, comm)
      .validation.valid());
  if (rank == size - 1)
    n.nodes[99] = {Point<3>(0., 1., 0.),
                   rank,
                   Port::arterial_outlet,
                   Provenance::imported};
  bad_bc               = bc;
  bad_bc.pressures[99] = 4000.;
  EXPECT_FALSE(solve_reference(n, {}, bad_bc, {}, comm).validation.valid());
  n = reference_partition(chain());
  if (rank == size - 1)
    n.nodes[99] = {Point<3>(std::numeric_limits<double>::quiet_NaN(), 1., 0.),
                   rank,
                   Port::arterial_outlet,
                   Provenance::imported};
  EXPECT_FALSE(solve_reference(n, {}, bc, {}, comm).validation.valid());
  const auto                a = reference_partition(av());
  std::vector<TerminalPair> pairs;
  if (rank == size - 1)
    {
      pairs = bed();
      pairs.push_back(pairs[0]);
    }
  const ReferenceConditions av_bc{{{0, 12000.}, {2, 4000.}}, {}};
  EXPECT_FALSE(solve_reference(a, pairs, av_bc, {}, comm).validation.valid());
  pairs.clear();
  if (rank == size - 1)
    pairs = {{99, 999, 12, 1.e12}};
  EXPECT_FALSE(solve_reference(a, pairs, av_bc, {}, comm).validation.valid());
}

TEST(VascularReferenceFlow, BOTH_ParallelBudgetAndZeroEquilibrium)
{
  const auto  comm = MPI_COMM_WORLD;
  NetworkData full;
  for (Id i = 0; i < 10; ++i)
    full.nodes[i].position = Point<3>(.1 * i, 0., 0.);
  for (Id i = 0; i < 9; ++i)
    edge(full, 100 + i, i, i + 1, .004 * (i + 1));
  full.nodes[0].port    = Port::arterial_inlet;
  full.nodes[9].port    = Port::arterial_outlet;
  const auto          n = reference_partition(full);
  ReferenceParameters p;
  p.maximum_iterations = 1;
  const auto exhausted =
    solve_reference(n, {}, {{{9, 0.}}, {{0, 1.e-9}}}, p, comm);
  EXPECT_FALSE(exhausted.validation.valid());
  EXPECT_EQ(exhausted.iterations, 1u);
  const auto zero =
    solve_reference(n, {}, {{{0, 5000.}, {9, 5000.}}, {}}, {}, comm);
  ASSERT_TRUE(zero.validation.valid());
  EXPECT_DOUBLE_EQ(zero.dissipated_power, 0.);
  EXPECT_DOUBLE_EQ(zero.maximum_flow_defect, 0.);
  EXPECT_EQ(zero.iterations, 0u);
}

TEST(VascularReferenceFlow,
     MPI_InconsistentReferenceControlsAndGhostsAreRejected)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  ASSERT_GT(size, 1u);
  auto                n = reference_partition(chain());
  ReferenceConditions bc{{{10, 12000.}, {30, 4000.}}, {}};
  ReferenceParameters p;
  if (rank == size - 1)
    p.maximum_iterations = 0;
  EXPECT_THROW(solve_reference(n, {}, bc, p, comm), ExceptionBase);
  p = {};
  if (rank == size - 1)
    p.maximum_iterations = 20;
  EXPECT_THROW(solve_reference(n, {}, bc, p, comm), ExceptionBase);
  if (rank == size - 1)
    bc.pressures[10] = 13000.;
  EXPECT_FALSE(solve_reference(n, {}, bc, {}, comm).validation.valid());
  bc.pressures[10] = 12000.;
  if (rank == 101 % size)
    n.nodes.at(10).position[1] = .01;
  EXPECT_FALSE(solve_reference(n, {}, bc, {}, comm).validation.valid());
}

TEST(VascularReferenceFlow, BOTH_PublicGeneratedTreeRealizesNonzeroDemandFlows)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {8, 1, 1},
                                            Point<3>(0., 0., 0.),
                                            Point<3>(8., 1., 1.));
  Parameters p;
  p.terminal_count_multiplier_factor = "none";

  p.seed_position      = Point<3>(6.5, .5, .5);
  p.seed_direction     = Point<3>(-1., 0., 0.);
  p.seed_length        = .5;
  p.seed_radius        = .005;
  p.viscosity          = .004;
  p.terminal_count     = 4;
  p.optimization_steps = 3;
  p.reference_flow     = 2.e-9;
  p.tissue_demand      = "1+x";
  const auto generated = generate(background, MappingQ1<3>(), {}, p, comm);
  ASSERT_TRUE(generated.validation.valid());
  const auto bc = demand_flow_conditions(
    generated.network, {{0, 12000.}}, p.reference_flow, {}, comm);
  const auto state = solve_reference(generated.network, {}, bc, {}, comm);
  ASSERT_TRUE(state.validation.valid());
  double perfusion = 0., reaction = 0.;
  for (const auto &[id, q] : state.terminal_outflows)
    {
      EXPECT_GT(q, 0.);
      EXPECT_NEAR(q / generated.target_terminal_flows.at(id), 1., 1.e-9);
      perfusion += q;
    }
  for (const auto &[id, q] : state.boundary_inflows)
    if (id == 0)
      reaction += q;
  EXPECT_NEAR(Utilities::MPI::sum(perfusion, comm) / p.reference_flow,
              1.,
              1.e-9);
  EXPECT_NEAR(Utilities::MPI::sum(reaction, comm) / p.reference_flow,
              1.,
              1.e-9);
  EXPECT_GT(state.dissipated_power, 0.);
  EXPECT_LE(state.relative_flow_defect, 1.e-9);
  for (const auto &[id, pressure] : state.pressures)
    {
      EXPECT_GT(pressure, 0.);
      EXPECT_LE(pressure, 12000. + 1.e-8);
    }
}
