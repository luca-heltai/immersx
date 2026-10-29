// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <cmath>
#include <functional>
#include <set>

using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  NetworkData
  imported_forest()
  {
    NetworkData n;
    n.nodes[10] = {Point<3>(-.2, .3, .5),
                   0,
                   Port::arterial_inlet,
                   Provenance::imported};
    n.nodes[20] = {Point<3>(.2, .3, .5), 0, Port::none, Provenance::imported};
    n.nodes[80] = {Point<3>(-.2, .7, .5),
                   0,
                   Port::arterial_inlet,
                   Provenance::generated};
    n.nodes[90] = {Point<3>(.2, .7, .5), 0, Port::none, Provenance::generated};
    Vessel a;
    a.nodes                                = {{10, 20}};
    a.radii                                = {{.009, .008}};
    a.viscosity                            = .003;
    a.physical_properties["Young modulus"] = 12345.;
    n.vessels[400]                         = a;
    a.nodes                                = {{80, 90}};
    a.radii                                = {{.013, .012}};
    a.viscosity                            = .005;
    a.provenance                           = Provenance::generated;
    n.vessels[900]                         = a;
    n.terminals[701]                       = {20, 501, System::arterial, 0, 2.};
    n.terminals[702]                       = {90, 502, System::arterial, 0, 3.};
    return n;
  }

  void
  controls(Parameters &p)
  {
    p.seed_length        = .12;
    p.seed_direction     = Point<3>(1., 0., 0.);
    p.terminal_count     = 12;
    p.optimization_steps = 5;
    p.candidate_count    = 8;
    p.sample_budget      = 1000;
  }

  void
  unchanged(const NetworkData &initial, const NetworkData &result)
  {
    for (const auto &[id, node] : initial.nodes)
      {
        ASSERT_TRUE(result.nodes.count(id));
        const auto &actual = result.nodes.at(id);
        EXPECT_EQ(actual.position, node.position);
        EXPECT_EQ(actual.owner, node.owner);
        EXPECT_EQ(actual.port, node.port);
        EXPECT_EQ(actual.provenance, node.provenance);
      }
    for (const auto &[id, e] : initial.vessels)
      {
        ASSERT_TRUE(result.vessels.count(id));
        const auto &actual = result.vessels.at(id);
        EXPECT_EQ(actual.nodes, e.nodes);
        EXPECT_EQ(actual.radii, e.radii);
        EXPECT_EQ(actual.viscosity, e.viscosity);
        EXPECT_EQ(actual.system, e.system);
        EXPECT_EQ(actual.provenance, e.provenance);
        EXPECT_EQ(actual.owner, e.owner);
        EXPECT_EQ(actual.physical_properties, e.physical_properties);
      }
  }
} // namespace

TEST(VascularImported, ExtendsOnlySelectedTerminalAndInheritsProperties)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  const auto initial = imported_forest();
  Parameters p;
  controls(p);
  p.selected_terminals = {702};
  const auto result =
    generate(background, MappingQ1<3>(), initial, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  unchanged(initial, result.network);
  unchanged(imported_forest(), initial);
  ASSERT_EQ(result.attachment_roots.size(), 1u);
  EXPECT_TRUE(result.attachment_roots.count(702));
  const Id root = result.attachment_roots.at(702);
  EXPECT_EQ(result.network.vessels.at(root).nodes[0], 90u);
  EXPECT_DOUBLE_EQ(result.network.vessels.at(root).radii[0], .012);
  EXPECT_EQ(result.network.terminals.size(), p.terminal_count + 1);
  EXPECT_TRUE(result.network.terminals.count(701));
  EXPECT_FALSE(result.network.terminals.count(702));
  EXPECT_EQ(result.network.terminals.at(701).demand, 2.);
  for (const auto &[id, e] : result.network.vessels)
    if (!initial.vessels.count(id))
      {
        EXPECT_GT(id, initial.vessels.rbegin()->first);
        EXPECT_EQ(e.viscosity, .005);
        EXPECT_EQ(e.physical_properties,
                  initial.vessels.at(900).physical_properties);
        EXPECT_EQ(e.provenance, Provenance::generated);
      }
}

TEST(VascularImported, MultipleInletsKeepIndependentFixedRootMurrayLaws)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  const auto initial = imported_forest();
  Parameters p;
  controls(p);
  p.terminal_count     = 18;
  p.dissipation_weight = .3;
  p.reference_flow     = 2.e-9;
  const auto result =
    generate(background, MappingQ1<3>(), initial, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  unchanged(initial, result.network);
  ASSERT_EQ(result.attachment_roots.size(), 2u);
  EXPECT_EQ(result.network.terminals.size(), p.terminal_count);
  EXPECT_EQ(result.territories.size(), p.terminal_count);
  std::map<Id, std::vector<Id>> children;
  for (const auto &[id, e] : result.network.vessels)
    if (!initial.vessels.count(id))
      children[e.nodes[0]].push_back(id);
  unsigned int                    total = 0;
  std::set<Id>                    visited;
  std::function<unsigned int(Id)> leaves = [&](Id id) {
    EXPECT_TRUE(visited.insert(id).second);
    const auto &e   = result.network.vessels.at(id);
    const auto &out = children[e.nodes[1]];
    if (out.empty())
      return 1u;
    EXPECT_EQ(out.size(), 2u);
    double       murray = 0.;
    unsigned int count  = 0;
    for (Id child : out)
      {
        count += leaves(child);
        murray += std::pow(result.network.vessels.at(child).radii[0], 3.);
      }
    EXPECT_NEAR(murray / std::pow(e.radii[1], 3.), 1., 2.e-13);
    return count;
  };
  for (const auto &[terminal, root] : result.attachment_roots)
    {
      const auto original = terminal == 701 ? 400 : 900;
      EXPECT_DOUBLE_EQ(result.network.vessels.at(root).radii[0],
                       initial.vessels.at(original).radii[1]);
      total += leaves(root);
    }
  EXPECT_EQ(total, p.terminal_count);
  EXPECT_EQ(visited.size(),
            result.network.vessels.size() - initial.vessels.size());
}

TEST(VascularImported, DiscoversEndpointsAndKeepsVenousPortsAndBranches)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  auto initial = imported_forest();
  initial.terminals.clear();
  initial.nodes[100]    = {Point<3>(.8, .9, .5),
                           0,
                           Port::venous_outlet,
                           Provenance::imported};
  initial.nodes[110]    = {Point<3>(.6, .9, .5),
                           0,
                           Port::none,
                           Provenance::imported};
  auto vein             = initial.vessels.at(400);
  vein.nodes            = {{100, 110}};
  vein.system           = System::venous;
  initial.vessels[1000] = vein;
  initial.nodes[120]    = {Point<3>(-.2, .95, .5),
                           0,
                           Port::arterial_inlet,
                           Provenance::imported};
  initial.nodes[130]    = {Point<3>(.2, .95, .5),
                           0,
                           Port::arterial_outlet,
                           Provenance::imported};
  auto final_branch     = initial.vessels.at(400);
  final_branch.nodes    = {{120, 130}};
  initial.vessels[1100] = final_branch;
  Parameters p;
  controls(p);
  p.terminal_count = 8;
  const auto result =
    generate(background, MappingQ1<3>(), initial, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  unchanged(initial, result.network);
  EXPECT_EQ(result.attachment_roots.size(), 2u);
  for (const auto &[id, root] : result.attachment_roots)
    EXPECT_TRUE(result.network.vessels.at(root).nodes[0] == 20 ||
                result.network.vessels.at(root).nodes[0] == 90);
  unsigned int venous = 0;
  for (const auto &[id, t] : result.network.terminals)
    if (t.system == System::venous)
      {
        ++venous;
        EXPECT_EQ(t.node, 110u);
      }
  EXPECT_EQ(venous, 1u);
  EXPECT_EQ(result.network.nodes.at(130).port, Port::arterial_outlet);
}

TEST(VascularImported,
     ExteriorInputIsPreservedAndExplicitExteriorExtensionFails)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  auto initial                  = imported_forest();
  initial.nodes.at(80).position = Point<3>(.5, -.2, .5);
  initial.nodes.at(90).position = Point<3>(.5, -.1, .5);
  Parameters p;
  controls(p);
  p.selected_terminals = {702};
  const auto rejected =
    generate(background, MappingQ1<3>(), initial, p, MPI_COMM_WORLD);
  EXPECT_FALSE(rejected.validation.valid());
  EXPECT_EQ(rejected.network.vessels.size(), initial.vessels.size());
  unchanged(initial, rejected.network);
  p.selected_terminals.clear();
  const auto accepted =
    generate(background, MappingQ1<3>(), initial, p, MPI_COMM_WORLD);
  ASSERT_TRUE(accepted.validation.valid());
  unchanged(initial, accepted.network);
  EXPECT_EQ(accepted.attachment_roots.size(), 1u);
  EXPECT_TRUE(accepted.attachment_roots.count(701));
  for (const auto &[id, e] : accepted.network.vessels)
    if (!initial.vessels.count(id))
      for (Id node : e.nodes)
        for (unsigned int d = 0; d < 3; ++d)
          {
            EXPECT_GT(accepted.network.nodes.at(node).position[d], e.radii[0]);
            EXPECT_LT(accepted.network.nodes.at(node).position[d],
                      1. - e.radii[0]);
          }
}

TEST(VascularImported, RejectsUnknownDuplicateAndNonterminalSelections)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  auto       initial = imported_forest();
  Parameters p;
  controls(p);
  for (const auto &selection :
       {std::vector<Id>{999}, std::vector<Id>{701, 701}})
    {
      p.selected_terminals = selection;
      EXPECT_FALSE(
        generate(background, MappingQ1<3>(), initial, p, MPI_COMM_WORLD)
          .validation.valid());
    }
  p.selected_terminals           = {701};
  initial.terminals.at(701).node = 10;
  EXPECT_FALSE(generate(background, MappingQ1<3>(), initial, p, MPI_COMM_WORLD)
                 .validation.valid());
}

TEST(VascularImported, FrozenLargeObstacleBlocksInitialExtension)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  auto initial          = imported_forest();
  initial.nodes[100]    = {Point<3>(.26, .3, .4),
                           0,
                           Port::venous_outlet,
                           Provenance::imported};
  initial.nodes[110]    = {Point<3>(.26, .3, .6),
                           0,
                           Port::none,
                           Provenance::imported};
  auto obstacle         = initial.vessels.at(400);
  obstacle.nodes        = {{100, 110}};
  obstacle.radii        = {{.04, .04}};
  obstacle.system       = System::venous;
  initial.vessels[1000] = obstacle;
  Parameters p;
  controls(p);
  p.selected_terminals = {701};
  const auto result =
    generate(background, MappingQ1<3>(), initial, p, MPI_COMM_WORLD);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_EQ(result.network.vessels.size(), initial.vessels.size());
  unchanged(initial, result.network);
}

TEST(VascularImported, RestartFreezesPreviouslyGeneratedData)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  controls(p);
  p.seed_radius    = .007;
  p.viscosity      = .004;
  p.seed_position  = Point<3>(.2, .5, .5);
  p.terminal_count = 4;
  const auto initial =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(initial.validation.valid());
  p.selected_terminals = {initial.network.terminals.begin()->first};
  p.seed_direction     = Point<3>();
  p.seed_radius        = 0.;
  p.viscosity          = 0.;
  p.terminal_count     = 8;
  const auto result =
    generate(background, MappingQ1<3>(), initial.network, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  unchanged(initial.network, result.network);
  EXPECT_EQ(result.attachment_roots.size(), 1u);
}
