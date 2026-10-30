// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>
#include <immersx/vascularization/reference_flow.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <set>

using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  void
  synthetic(Parameters &p)
  {
    p.seed_position  = Point<3>(.2, .5, .5);
    p.seed_direction = Point<3>(1., 0., 0.);
    p.seed_radius    = .01;
    p.seed_length    = .3;
    p.viscosity      = .004;
    p.reference_flow = 2.e-9;
    p.terminal_count = 12;
  }
  struct Metrics
  {
    double volume = 0., power = 0.;
  };
  Metrics
  oracle(const GenerationResult &result,
         const NetworkData      &initial,
         const Parameters       &p)
  {
    const auto  &n = result.network;
    std::set<Id> territories;
    for (const auto &t : result.territories)
      territories.insert(t.id);
    std::map<Id, double> terminal_demand;
    double               total = 0.;
    for (const auto &[id, t] : n.terminals)
      if (territories.count(t.territory))
        {
          terminal_demand[t.node] = t.demand;
          total += t.demand;
        }
    EXPECT_EQ(terminal_demand.size(), result.territories.size());
    std::map<Id, std::vector<Id>> children;
    std::map<Id, Id>              incoming;
    for (const auto &[id, e] : n.vessels)
      if (!initial.vessels.count(id))
        {
          children[e.nodes[0]].push_back(id);
          incoming[e.nodes[1]] = id;
        }
    std::map<Id, double>          demands;
    std::map<Id, Id>              roots;
    std::set<Id>                  root_ids;
    std::function<double(Id, Id)> visit = [&](Id id, Id root) {
      const auto &e = n.vessels.at(id);
      double      d = 0.;
      roots[id]     = root;
      if (children[e.nodes[1]].empty())
        d = terminal_demand.at(e.nodes[1]);
      else
        {
          EXPECT_EQ(children[e.nodes[1]].size(), 2u);
          double murray = 0.;
          for (Id child : children[e.nodes[1]])
            {
              d += visit(child, root);
              murray += std::pow(n.vessels.at(child).radii[0],
                                 p.arterial_murray_exponent);
            }
          EXPECT_NEAR(std::pow(e.radii[0], p.arterial_murray_exponent) / murray,
                      1.,
                      1.e-12);
        }
      demands[id] = d;
      return d;
    };
    for (const auto &[id, e] : n.vessels)
      if (!initial.vessels.count(id) && !incoming.count(e.nodes[0]))
        {
          root_ids.insert(id);
          visit(id, id);
        }
    Metrics metrics;
    for (const auto &[id, d] : demands)
      {
        const auto  &e        = n.vessels.at(id);
        const auto  &root     = n.vessels.at(roots.at(id));
        const double expected = std::pow(d / demands.at(roots.at(id)),
                                         1. / p.arterial_murray_exponent);
        EXPECT_NEAR(e.radii[0] / root.radii[0], expected, 1.e-12);
        EXPECT_DOUBLE_EQ(e.radii[0], e.radii[1]);
        const double l = n.nodes.at(e.nodes[0])
                           .position.distance(n.nodes.at(e.nodes[1]).position);
        const double q = p.reference_flow * d / total;
        metrics.volume += numbers::PI * e.radii[0] * e.radii[0] * l;
        metrics.power += q * q * 8. * e.viscosity * l /
                         (numbers::PI * std::pow(e.radii[0], 4));
      }
    EXPECT_EQ(result.target_terminal_flows.size(), terminal_demand.size());
    EXPECT_EQ(result.target_attachment_flows.size(), root_ids.size());
    for (const auto &[id, t] : n.terminals)
      if (territories.count(t.territory))
        EXPECT_NEAR(result.target_terminal_flows.at(id) /
                      (p.reference_flow * t.demand / total),
                    1.,
                    1.e-12);
    for (Id id : root_ids)
      EXPECT_NEAR(result.target_attachment_flows.at(id) /
                    (p.reference_flow * demands.at(id) / total),
                  1.,
                  1.e-12);
    return metrics;
  }
  NetworkData
  forest()
  {
    NetworkData n;
    for (const auto &[id, pos] : std::map<Id, Point<3>>{{10, {.05, .3, .5}},
                                                        {20, {.2, .3, .5}},
                                                        {30, {.05, .7, .5}},
                                                        {40, {.2, .7, .5}},
                                                        {50, {.6, .1, .5}},
                                                        {60, {.7, .1, .5}}})
      n.nodes[id].position = pos;
    for (Id id : {10, 30, 50})
      n.nodes[id].port = Port::arterial_inlet;
    for (const auto &[id, ends] : std::map<Id, std::array<Id, 2>>{
           {100, {{10, 20}}}, {200, {{30, 40}}}, {300, {{50, 60}}}})
      {
        Vessel e;
        e.nodes       = ends;
        e.radii       = {{id == 100 ? .01 : .008, id == 100 ? .01 : .008}};
        e.viscosity   = .004;
        n.vessels[id] = e;
      }
    n.terminals[17] = {20, 1, System::arterial, 0, 5.};
    n.terminals[27] = {40, 2, System::arterial, 0, 7.};
    n.terminals[37] = {60, 3, System::arterial, 0, 11.};
    n.vessels[100].physical_properties["density"] = 1050.;
    return n;
  }
} // namespace

TEST(VascularDemand, WeightedMurrayAndGlobalDesignFlowsHaveIndependentOracles)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.tissue_demand            = "1+3*x+2*y";
  p.arterial_murray_exponent = 2.7;
  p.terminal_count           = 20;
  p.dissipation_weight       = .35;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_EQ(result.territories.size(), 20u);
  EXPECT_EQ(result.target_terminal_flows.size(), 20u);
  const auto metrics = oracle(result, {}, p);
  EXPECT_NEAR(result.tissue.inside_vessel_volume / metrics.volume, 1., 1.e-12);
  EXPECT_GT(metrics.power, 0.);
  double flow = 0., smallest = 1.e300, largest = 0.;
  for (const auto &[id, t] : result.network.terminals)
    {
      const auto pos = result.network.nodes.at(t.node).position;
      EXPECT_NEAR(t.demand, 1. + 3. * pos[0] + 2. * pos[1], 1.e-13);
      smallest = std::min(smallest, t.demand);
      largest  = std::max(largest, t.demand);
      flow += result.target_terminal_flows.at(id);
    }
  EXPECT_GT(largest, smallest + 1.);
  EXPECT_NEAR(flow / p.reference_flow, 1., 1.e-12);
  EXPECT_DOUBLE_EQ(result.network.vessels.at(0).radii[0], p.seed_radius);
}

TEST(VascularDemand, CommonDemandAndFlowScalesDoNotChangePhysicalGeneration)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.dissipation_weight = .4;
  const auto a    = generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  p.tissue_demand = "4";
  p.reference_flow *= 3.;
  const auto b = generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(a.validation.valid());
  ASSERT_TRUE(b.validation.valid());
  EXPECT_EQ(a.sampled_points, b.sampled_points);
  for (const auto &[id, node] : a.network.nodes)
    EXPECT_EQ(node.position, b.network.nodes.at(id).position);
  for (const auto &[id, e] : a.network.vessels)
    EXPECT_EQ(e.radii, b.network.vessels.at(id).radii);
  for (const auto &[id, q] : a.target_terminal_flows)
    EXPECT_NEAR(b.target_terminal_flows.at(id) / q, 3., 1.e-12);
  for (const auto &[id, t] : b.network.terminals)
    EXPECT_DOUBLE_EQ(t.demand, 4.);
}

TEST(VascularDemand, AHeavyNewLeafCannotShrinkAnOldLeafBelowMinimumRadius)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.terminal_count  = 2;
  p.minimum_radius  = .004;
  p.tissue_demand   = "1+999999*(x>0.7)";
  p.terminal_weight = "x>0.8";
  p.sample_budget   = 64;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_EQ(result.sampled_points, 64u);
  EXPECT_EQ(result.accepted_insertions, 0u);
  EXPECT_EQ(result.network.vessels.size(), 1u);
  EXPECT_DOUBLE_EQ(result.network.vessels.at(0).radii[0], p.seed_radius);
}

TEST(VascularDemand, ReferenceSolveRealizesTheDemandSplitWithoutDoubleInletData)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.tissue_demand      = "1+x+2*z";
  p.dissipation_weight = .2;
  const auto generated =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(generated.validation.valid());
  const auto bc = demand_flow_conditions(
    generated.network, {{0, 12000.}}, p.reference_flow, {}, MPI_COMM_WORLD);
  EXPECT_EQ(bc.pressures.size(), 1u);
  EXPECT_EQ(bc.inflows.count(0), 0u);
  const auto state =
    solve_reference(generated.network, {}, bc, {}, MPI_COMM_WORLD);
  ASSERT_TRUE(state.validation.valid());
  EXPECT_NEAR(state.boundary_inflows.at(0) / p.reference_flow, 1., 1.e-9);
  for (const auto &[id, q] : generated.target_terminal_flows)
    EXPECT_NEAR(state.terminal_outflows.at(id) / q, 1., 1.e-9);
  for (const auto &[id, pressure] : state.pressures)
    EXPECT_GT(pressure, 0.);
  oracle(generated, {}, p);
}

TEST(VascularDemand, ImportedForestsRetainPhysicalDataAndUseGlobalDemandShares)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  const auto initial = forest();
  Parameters p;
  synthetic(p);
  p.selected_terminals = {17, 27};
  p.seed_length        = .15;
  p.tissue_demand      = "1+2*y";
  p.terminal_count     = 10;
  p.dissipation_weight = .4;
  const auto result =
    generate(background, MappingQ1<3>(), initial, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  oracle(result, initial, p);
  EXPECT_EQ(result.target_terminal_flows.count(37), 0u);
  EXPECT_DOUBLE_EQ(result.network.terminals.at(37).demand, 11.);
  std::vector<Id> selected;
  for (const auto &[id, q] : result.target_terminal_flows)
    selected.push_back(id);
  auto bc = demand_flow_conditions(result.network,
                                   {{10, 12000.}, {30, 12000.}, {50, 12000.}},
                                   p.reference_flow,
                                   selected,
                                   MPI_COMM_WORLD);
  // The untouched terminal keeps its separately supplied pressure load.
  bc.pressures[60] = 11000.;
  const auto actual =
    solve_reference(result.network, {}, bc, {}, MPI_COMM_WORLD);
  ASSERT_TRUE(actual.validation.valid());
  for (const auto &[id, q] : result.target_attachment_flows)
    EXPECT_NEAR(actual.vessels.at(id).flow / q, 1., 1.e-9);
  EXPECT_GT(actual.terminal_outflows.at(37), 0.);
  for (const auto &[id, node] : initial.nodes)
    EXPECT_EQ(node.position, result.network.nodes.at(id).position);
  for (const auto &[id, e] : initial.vessels)
    {
      EXPECT_EQ(e.nodes, result.network.vessels.at(id).nodes);
      EXPECT_EQ(e.radii, result.network.vessels.at(id).radii);
      EXPECT_EQ(e.physical_properties,
                result.network.vessels.at(id).physical_properties);
    }
}

TEST(VascularDemand,
     LocalOptimizationReducesAnIndependentlyIntegratedMixedObjective)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  const auto initial = forest();
  Parameters p;
  synthetic(p);
  p.selected_terminals = {17, 27};
  p.seed_length        = .15;
  p.tissue_demand      = "1+2*y";
  p.terminal_count     = 2;
  p.dissipation_weight = .4;
  p.optimization_steps = 0;
  const auto base =
    generate(background, MappingQ1<3>(), initial, p, MPI_COMM_WORLD);
  ASSERT_TRUE(base.validation.valid());
  const auto old   = oracle(base, initial, p);
  p.terminal_count = 3;
  const auto plain =
    generate(background, MappingQ1<3>(), initial, p, MPI_COMM_WORLD);
  p.optimization_steps = 8;
  const auto optimized =
    generate(background, MappingQ1<3>(), initial, p, MPI_COMM_WORLD);
  ASSERT_TRUE(plain.validation.valid());
  ASSERT_TRUE(optimized.validation.valid());
  for (unsigned int i = 0; i < plain.territories.size(); ++i)
    EXPECT_EQ(plain.territories[i].position,
              optimized.territories.at(i).position);
  const auto   a = oracle(plain, initial, p), b = oracle(optimized, initial, p);
  const double cost_a = .6 * a.volume / old.volume + .4 * a.power / old.power;
  const double cost_b = .6 * b.volume / old.volume + .4 * b.power / old.power;
  EXPECT_LE(cost_b, cost_a * (1. + 1.e-12));
}

TEST(VascularDemand, InvalidDemandAndMissingPhysicalFlowAreRejected)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  synthetic(p);
  p.reference_flow     = 0.;
  p.dissipation_weight = .4;
  EXPECT_THROW(p.validate(), ExceptionBase);
  p.dissipation_weight = 0.;
  p.reference_flow     = -1.;
  EXPECT_THROW(p.validate(), ExceptionBase);
  p.reference_flow = 0.;
  for (const auto expression : {"0", "-1", "1e300*1e300"})
    {
      p.tissue_demand = expression;
      EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD),
                   ExceptionBase);
    }
  p.tissue_demand  = "1+x";
  p.terminal_count = 1;
  const auto unspecified =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(unspecified.validation.valid());
  EXPECT_TRUE(unspecified.target_terminal_flows.empty());
}

TEST(VascularDemand, ArterialAndVenousDemandConditionsYieldMatchedPositiveBeds)
{
  NetworkData n;
  n.nodes[0].position = Point<3>(.1, .3, .5);
  n.nodes[1].position = Point<3>(.3, .3, .5);
  n.nodes[2].position = Point<3>(.3, .5, .5);
  n.nodes[3].position = Point<3>(.9, .3, .5);
  n.nodes[4].position = Point<3>(.7, .3, .5);
  n.nodes[5].position = Point<3>(.7, .5, .5);
  n.nodes[0].port     = Port::arterial_inlet;
  n.nodes[3].port     = Port::venous_outlet;
  for (Id i = 0; i < 4; ++i)
    {
      Vessel e;
      e.nodes =
        i < 2 ? std::array<Id, 2>{{0, i + 1}} : std::array<Id, 2>{{3, i + 2}};
      e.radii      = {{.002, .002}};
      e.viscosity  = .004;
      e.system     = i < 2 ? System::arterial : System::venous;
      n.vessels[i] = e;
    }
  n.terminals[10]   = {1, 77, System::arterial, 0, 1.};
  n.terminals[11]   = {2, 78, System::arterial, 0, 3.};
  n.terminals[20]   = {4, 77, System::venous, 0, 1.};
  n.terminals[21]   = {5, 78, System::venous, 0, 3.};
  const double flow = 2.e-9;
  const auto   bc   = demand_flow_conditions(
    n, {{0, 12000.}, {3, 4000.}}, flow, {}, MPI_COMM_WORLD);
  EXPECT_NEAR(bc.inflows.at(1) / flow, -.25, 1.e-13);
  EXPECT_NEAR(bc.inflows.at(2) / flow, -.75, 1.e-13);
  EXPECT_NEAR(bc.inflows.at(4) / flow, .25, 1.e-13);
  EXPECT_NEAR(bc.inflows.at(5) / flow, .75, 1.e-13);
  const auto design = solve_reference(n, {}, bc, {}, MPI_COMM_WORLD);
  ASSERT_TRUE(design.validation.valid());
  std::vector<TerminalPair> pairs;
  for (const auto &[a, v] : std::map<Id, Id>{{10, 20}, {11, 21}})
    {
      const auto  &ta = n.terminals.at(a), &tv = n.terminals.at(v);
      const double q = -bc.inflows.at(ta.node);
      const double drop =
        design.pressures.at(ta.node) - design.pressures.at(tv.node);
      ASSERT_GT(drop, 0.);
      pairs.push_back({ta.territory, a, v, drop / q});
    }
  const auto coupled = solve_reference(
    n, pairs, {{{0, 12000.}, {3, 4000.}}, {}}, {}, MPI_COMM_WORLD);
  ASSERT_TRUE(coupled.validation.valid());
  EXPECT_NEAR(coupled.territories.at(77).flow / flow, .25, 1.e-9);
  EXPECT_NEAR(coupled.territories.at(78).flow / flow, .75, 1.e-9);
  const auto selected = demand_flow_conditions(
    n, {{0, 12000.}, {3, 4000.}}, flow, {11, 21}, MPI_COMM_WORLD);
  EXPECT_EQ(selected.inflows.size(), 2u);
  EXPECT_NEAR(selected.inflows.at(2) / flow, -1., 1.e-13);
  EXPECT_THROW(
    demand_flow_conditions(n, {{0, 12000.}}, flow, {999}, MPI_COMM_WORLD),
    ExceptionBase);
  EXPECT_THROW(
    demand_flow_conditions(n, {{0, 12000.}}, flow, {10, 10}, MPI_COMM_WORLD),
    ExceptionBase);
  EXPECT_THROW(
    demand_flow_conditions(n, {{1, 12000.}}, flow, {}, MPI_COMM_WORLD),
    ExceptionBase);
  EXPECT_THROW(demand_flow_conditions(n, {{0, 12000.}}, 0., {}, MPI_COMM_WORLD),
               ExceptionBase);
}

namespace
{
  constexpr Id demand_base = (Id{1} << 60) + 75;

  NetworkData
  distributed_demand_network()
  {
    const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto active =
      std::max(1u, Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1);
    NetworkData                   full, local;
    const std::array<Point<3>, 6> positions{{{.1, .3, .5},
                                             {.3, .3, .5},
                                             {.3, .5, .5},
                                             {.9, .3, .5},
                                             {.7, .3, .5},
                                             {.7, .5, .5}}};
    for (unsigned int i = 0; i < positions.size(); ++i)
      {
        auto &n    = full.nodes[demand_base + i];
        n.position = positions[i];
        n.owner    = i % active;
      }
    full.nodes.at(demand_base).port     = Port::arterial_inlet;
    full.nodes.at(demand_base + 3).port = Port::venous_outlet;
    for (unsigned int i = 0; i < 4; ++i)
      {
        Vessel v;
        v.nodes     = i < 2 ?
                        std::array<Id, 2>{{demand_base, demand_base + i + 1}} :
                        std::array<Id, 2>{{demand_base + 3, demand_base + i + 2}};
        v.radii     = {{.002, .002}};
        v.viscosity = .004;
        v.system    = i < 2 ? System::arterial : System::venous;
        v.owner     = (i + 1) % active;
        full.vessels[demand_base + 100 + i] = v;
      }
    full.terminals[demand_base + 10] = {
      demand_base + 1, demand_base + 77, System::arterial, 1 % active, 1.};
    full.terminals[demand_base + 11] = {
      demand_base + 2, demand_base + 78, System::arterial, 2 % active, 3.};
    full.terminals[demand_base + 20] = {
      demand_base + 4, demand_base + 77, System::venous, 4 % active, 1.};
    full.terminals[demand_base + 21] = {
      demand_base + 5, demand_base + 78, System::venous, 5 % active, 3.};
    for (const auto &[id, n] : full.nodes)
      if (n.owner == rank)
        local.nodes.emplace(id, n);
    for (const auto &[id, v] : full.vessels)
      if (v.owner == rank)
        {
          local.vessels.emplace(id, v);
          for (Id node : v.nodes)
            local.nodes.emplace(node, full.nodes.at(node));
        }
    for (const auto &[id, t] : full.terminals)
      if (t.owner == rank)
        {
          local.terminals.emplace(id, t);
          local.nodes.emplace(t.node, full.nodes.at(t.node));
        }
    return local;
  }

  std::map<Id, double>
  demand_pressures(bool replicated = false)
  {
    if (replicated || Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1)
      return {{demand_base, 12000.}, {demand_base + 3, 4000.}};
    return {};
  }
} // namespace

TEST(VascularDemand, BOTH_GlobalConditionsYieldActualMatchedPositiveBeds)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  const auto n    = distributed_demand_network();
  if (size > 1 && rank == size - 1)
    {
      EXPECT_TRUE(n.nodes.empty());
      EXPECT_TRUE(n.terminals.empty());
    }
  constexpr double q         = 2.e-9;
  const auto       pressures = demand_pressures();
  const auto       bc = demand_flow_conditions(n, pressures, q, {}, comm);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(bc.pressures.size()),
                                comm),
            2u);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(bc.inflows.size()),
                                comm),
            4u);
  double arterial = 0., venous = 0.;
  for (const auto &[id, t] : n.terminals)
    {
      const double expected = t.demand / 4.;
      const double sign     = t.system == System::arterial ? -1. : 1.;
      EXPECT_NEAR(bc.inflows.at(t.node) / (sign * q), expected, 1.e-13);
      if (t.system == System::arterial)
        arterial -= bc.inflows.at(t.node);
      else if (t.system == System::venous)
        venous += bc.inflows.at(t.node);
    }
  EXPECT_NEAR(Utilities::MPI::sum(arterial, comm) / q, 1., 1.e-13);
  EXPECT_NEAR(Utilities::MPI::sum(venous, comm) / q, 1., 1.e-13);
  for (const auto &[id, p] : bc.pressures)
    {
      EXPECT_EQ(n.nodes.at(id).owner, rank);
      EXPECT_FALSE(bc.inflows.count(id));
    }
  const auto design = solve_reference(n, {}, bc, {}, comm);
  ASSERT_TRUE(design.validation.valid());
  std::vector<TerminalPair> pairs;
  for (const auto &[id, t] : n.terminals)
    if (t.system == System::arterial)
      {
        const auto  &v    = n.terminals.at(id + 10);
        const double flow = -bc.inflows.at(t.node);
        const double resistance =
          (design.pressures.at(t.node) - design.pressures.at(v.node)) / flow;
        const double vessel_length =
          id == demand_base + 10 ? .2 : std::sqrt(.08);
        const double expected =
          8000. / flow -
          2. * 8. * .004 * vessel_length / (numbers::PI * std::pow(.002, 4));
        EXPECT_GT(resistance, 0.);
        EXPECT_NEAR(resistance / expected, 1., 1.e-12);
        pairs.push_back({t.territory, id, id + 10, resistance});
      }
  const auto actual = solve_reference(n, pairs, {pressures, {}}, {}, comm);
  ASSERT_TRUE(actual.validation.valid());
  EXPECT_EQ(Utilities::MPI::sum(
              static_cast<unsigned int>(actual.territories.size()), comm),
            2u);
  for (const auto &[id, t] : n.terminals)
    if (t.system == System::arterial)
      {
        EXPECT_NEAR(actual.territories.at(t.territory).flow / q,
                    t.demand / 4.,
                    1.e-9);
      }
  EXPECT_LE(actual.relative_flow_defect, 1.e-9);
}

TEST(VascularDemand, BOTH_ExplicitSelectionAndRepeatedPortControlsStayOwned)
{
  const auto            comm = MPI_COMM_WORLD;
  const auto            rank = Utilities::MPI::this_mpi_process(comm);
  const auto            n    = distributed_demand_network();
  constexpr double      q    = 2.e-9;
  const std::vector<Id> selected{demand_base + 11, demand_base + 21};
  const auto            bc =
    demand_flow_conditions(n, demand_pressures(true), q, selected, comm);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(bc.pressures.size()),
                                comm),
            2u);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(bc.inflows.size()),
                                comm),
            2u);
  for (const auto &[id, p] : bc.pressures)
    EXPECT_EQ(n.nodes.at(id).owner, rank);
  const auto actual = solve_reference(n, {}, bc, {}, comm);
  ASSERT_TRUE(actual.validation.valid());
  for (const auto &[id, t] : n.terminals)
    {
      const bool chosen = id == demand_base + 11 || id == demand_base + 21;
      EXPECT_EQ(bc.inflows.count(t.node), chosen ? 1u : 0u);
      const double sign = t.system == System::arterial ? 1. : -1.;
      EXPECT_NEAR(actual.terminal_outflows.at(id) / (sign * q),
                  chosen ? 1. : 0.,
                  1.e-12);
    }
}

TEST(VascularDemand, BOTH_ExtremeDemandScalesPreserveRepresentableFlows)
{
  const auto comm = MPI_COMM_WORLD;
  auto       n    = distributed_demand_network();
  for (auto &[id, t] : n.terminals)
    t.demand = 1.e308;
  const auto equal =
    demand_flow_conditions(n, demand_pressures(), 2.e-9, {}, comm);
  for (const auto &[id, q] : equal.inflows)
    EXPECT_NEAR(std::abs(q) / 1.e-9, 1., 1.e-13);
  for (auto &[id, t] : n.terminals)
    t.demand =
      id == demand_base + 10 || id == demand_base + 20 ? 1.e-300 : 1.e300;
  const auto scaled =
    demand_flow_conditions(n, demand_pressures(), 1.e300, {}, comm);
  for (const auto &[id, t] : n.terminals)
    {
      const double expected =
        id == demand_base + 10 || id == demand_base + 20 ? 1.e-300 : 1.e300;
      EXPECT_GT(std::abs(scaled.inflows.at(t.node)), 0.);
      EXPECT_NEAR(std::abs(scaled.inflows.at(t.node)) / expected, 1., 1.e-13);
    }
  EXPECT_THROW(demand_flow_conditions(n, demand_pressures(), 1.e-300, {}, comm),
               ExceptionBase);
}

TEST(VascularDemand,
     BOTH_InvalidSelectionsPortsAndTerminalAuthorityFailCollectively)
{
  const auto       comm = MPI_COMM_WORLD;
  auto             n    = distributed_demand_network();
  constexpr double q    = 2.e-9;
  EXPECT_THROW(
    demand_flow_conditions(n, demand_pressures(), q, {demand_base + 999}, comm),
    ExceptionBase);
  EXPECT_THROW(demand_flow_conditions(n,
                                      demand_pressures(),
                                      q,
                                      {demand_base + 10, demand_base + 10},
                                      comm),
               ExceptionBase);
  EXPECT_THROW(
    demand_flow_conditions(n, {{demand_base + 1, 12000.}}, q, {}, comm),
    ExceptionBase);
  EXPECT_THROW(
    demand_flow_conditions(n, {{demand_base + 999, 12000.}}, q, {}, comm),
    ExceptionBase);
  auto bad = n;
  if (bad.terminals.count(demand_base + 10))
    bad.terminals.at(demand_base + 10).demand = 0.;
  EXPECT_THROW(demand_flow_conditions(bad, demand_pressures(), q, {}, comm),
               ExceptionBase);
  bad = n;
  if (bad.terminals.count(demand_base + 10))
    bad.terminals[demand_base + 999] = bad.terminals.at(demand_base + 10);
  EXPECT_THROW(demand_flow_conditions(bad, demand_pressures(), q, {}, comm),
               ExceptionBase);
  bad = n;
  bad.terminals.clear();
  EXPECT_THROW(demand_flow_conditions(bad, demand_pressures(), q, {}, comm),
               ExceptionBase);
  EXPECT_THROW(demand_flow_conditions(n, demand_pressures(), 0., {}, comm),
               ExceptionBase);
}

TEST(VascularDemand,
     MPI_InconsistentDemandControlsAndDuplicateOwnersFailCollectively)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  ASSERT_GT(size, 1u);
  const auto n = distributed_demand_network();
  EXPECT_THROW(demand_flow_conditions(n,
                                      demand_pressures(),
                                      rank == size - 1 ? 3.e-9 : 2.e-9,
                                      {},
                                      comm),
               ExceptionBase);
  const std::vector<Id> selected =
    rank == size - 1 ? std::vector<Id>{demand_base + 10} : std::vector<Id>{};
  EXPECT_THROW(
    demand_flow_conditions(n, demand_pressures(), 2.e-9, selected, comm),
    ExceptionBase);
  auto pressures = demand_pressures(true);
  if (rank == size - 1)
    pressures.at(demand_base) = 13000.;
  EXPECT_THROW(demand_flow_conditions(n, pressures, 2.e-9, {}, comm),
               ExceptionBase);
  NetworkData duplicate;
  duplicate.nodes[demand_base + 2000 + rank] = {Point<3>(rank, .5, .5),
                                                rank,
                                                Port::none,
                                                Provenance::imported};
  duplicate.terminals[demand_base + 3000]    = {
    demand_base + 2000 + rank, invalid_id, System::arterial, rank, 1.};
  EXPECT_THROW(demand_flow_conditions(duplicate, {}, 2.e-9, {}, comm),
               ExceptionBase);
  NetworkData foreign;
  if (rank == 0 || rank == size - 1)
    foreign.nodes[demand_base] = {Point<3>(.5, .5, .5),
                                  0,
                                  Port::none,
                                  Provenance::imported};
  if (rank == size - 1)
    foreign.terminals[demand_base + 10] = {
      demand_base, invalid_id, System::arterial, rank, 1.};
  EXPECT_THROW(demand_flow_conditions(foreign, {}, 2.e-9, {}, comm),
               ExceptionBase);
}
