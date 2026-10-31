// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/distribution.h>
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

namespace
{
  constexpr Id distributed_base = (Id{1} << 60) + 317;

  NetworkData
  distributed_input(bool with_terminals = true)
  {
    const auto  rank   = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto  source = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1;
    NetworkData n;
    if (rank != source)
      return n;
    for (unsigned int i = 0; i < 2; ++i)
      {
        const Id node     = distributed_base + 2 * i;
        n.nodes[node]     = {Point<3>(i == 0 ? 9.2 : -.2, i == 0 ? .3 : .7, .5),
                             source,
                             Port::arterial_inlet,
                         i == 0 ? Provenance::imported : Provenance::generated};
        n.nodes[node + 1] = {Point<3>(i == 0 ? 6.5 : 1.5, i == 0 ? .3 : .7, .5),
                             source,
                             Port::none,
                             Provenance::imported};
        Vessel v;
        v.nodes     = {{node, node + 1}};
        v.radii     = {{.012 + .002 * i, .008 + .002 * i}};
        v.viscosity = .003 + .002 * i;
        v.physical_properties["Young modulus"] = 12345. + i;
        v.owner                                = source;
        v.provenance = i == 0 ? Provenance::imported : Provenance::generated;
        n.vessels[distributed_base + 100 + i] = v;
        if (with_terminals)
          n.terminals[distributed_base + 200 + i] = {node + 1,
                                                     distributed_base + 300 + i,
                                                     System::arterial,
                                                     source,
                                                     2. + i};
      }
    return n;
  }

  void
  distributed_background(parallel::distributed::Triangulation<3> &bg)
  {
    GridGenerator::subdivided_hyper_rectangle(bg,
                                              {8, 1, 1},
                                              Point<3>(0., 0., 0.),
                                              Point<3>(8., 1., 1.));
  }
} // namespace

TEST(VascularImported,
     BOTH_PublicSelectedAttachmentCrossesPartitionsWithLargeIds)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  parallel::distributed::Triangulation<3> bg(comm);
  distributed_background(bg);
  const auto input = distributed_input();
  Parameters p;
  controls(p);
  p.seed_direction     = Point<3>();
  p.seed_length        = 5.5;
  p.terminal_count     = 1;
  p.reference_flow     = 1.e-9;
  p.tissue_demand      = "1+x";
  p.selected_terminals = {distributed_base + 200};
  const auto generated = generate(bg, MappingQ1<3>(), input, p, comm);
  ASSERT_TRUE(generated.validation.valid());
  unsigned int attachments = 0, frozen = 0, fresh = 0, old_terminals = 0,
               new_terminals    = 0;
  unsigned int attachment_owner = size, root_owner = size;
  for (const auto &[id, root] : generated.attachment_roots)
    {
      ++attachments;
      EXPECT_EQ(id, distributed_base + 200);
      EXPECT_GT(root, distributed_base + 101);
      attachment_owner = rank;
    }
  for (const auto &[id, v] : generated.network.vessels)
    if (v.owner == rank)
      {
        if (id == distributed_base + 100 || id == distributed_base + 101)
          {
            ++frozen;
            const unsigned int i = id - distributed_base - 100;
            EXPECT_EQ(v.nodes,
                      (std::array<Id, 2>{{distributed_base + 2 * i,
                                          distributed_base + 2 * i + 1}}));
            EXPECT_EQ(v.radii,
                      (std::array<double, 2>{
                        {.012 + .002 * i, .008 + .002 * i}}));
            EXPECT_DOUBLE_EQ(v.viscosity, .003 + .002 * i);
            EXPECT_DOUBLE_EQ(v.physical_properties.at("Young modulus"),
                             12345. + i);
            EXPECT_EQ(v.provenance,
                      i == 0 ? Provenance::imported : Provenance::generated);
          }
        else
          {
            ++fresh;
            root_owner = rank;
            EXPECT_EQ(v.nodes[0], distributed_base + 1);
            EXPECT_DOUBLE_EQ(v.radii[0], .008);
            EXPECT_DOUBLE_EQ(v.radii[1], .008);
            EXPECT_DOUBLE_EQ(v.viscosity, .003);
            EXPECT_DOUBLE_EQ(v.physical_properties.at("Young modulus"), 12345.);
            EXPECT_NEAR(length(generated.network, id), 5.5, 1.e-13);
            EXPECT_LT(generated.network.nodes.at(v.nodes[1])
                        .position.distance(Point<3>(1., .3, .5)),
                      1.e-13);
          }
      }
  for (const auto &[id, t] : generated.network.terminals)
    if (t.owner == rank)
      {
        EXPECT_NE(id, distributed_base + 200);
        if (id == distributed_base + 201)
          {
            ++old_terminals;
            EXPECT_EQ(t.territory, distributed_base + 301);
            EXPECT_DOUBLE_EQ(t.demand, 3.);
          }
        else
          {
            ++new_terminals;
            EXPECT_DOUBLE_EQ(t.demand, 2.);
            EXPECT_GT(t.territory, distributed_base + 301);
            EXPECT_NEAR(generated.target_terminal_flows.at(id) /
                          p.reference_flow,
                        1.,
                        1.e-13);
          }
      }
  EXPECT_EQ(Utilities::MPI::sum(attachments, comm), 1u);
  EXPECT_EQ(Utilities::MPI::sum(frozen, comm), 2u);
  EXPECT_EQ(Utilities::MPI::sum(fresh, comm), 1u);
  EXPECT_EQ(Utilities::MPI::sum(old_terminals, comm), 1u);
  EXPECT_EQ(Utilities::MPI::sum(new_terminals, comm), 1u);
  attachment_owner = Utilities::MPI::min(attachment_owner, comm);
  root_owner       = Utilities::MPI::min(root_owner, comm);
  if (size > 1)
    {
      EXPECT_NE(root_owner, attachment_owner);
    }
  unchanged(distributed_input(), input);
}

TEST(VascularImported, BOTH_PublicMultipleAttachmentsFeedConcurrentGrowth)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  parallel::distributed::Triangulation<3> bg(comm);
  distributed_background(bg);
  const auto input = distributed_input(false);
  Parameters p;
  controls(p);
  p.seed_direction     = Point<3>();
  p.seed_length        = .6;
  p.terminal_count     = 6;
  p.reference_flow     = 2.e-9;
  p.tissue_demand      = "1+x";
  const auto generated = generate(bg, MappingQ1<3>(), input, p, comm);
  ASSERT_TRUE(generated.validation.valid());
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(
                                  generated.attachment_roots.size()),
                                comm),
            2u);
  EXPECT_EQ(Utilities::MPI::sum(
              static_cast<unsigned int>(generated.territories.size()), comm),
            6u);
  EXPECT_EQ(generated.accepted_insertions, 4u);
  unsigned int frozen = 0, fresh = 0, terminal_count = 0;
  double       total_flow = 0.;
  for (const auto &[id, v] : generated.network.vessels)
    if (v.owner == rank)
      {
        if (id == distributed_base + 100 || id == distributed_base + 101)
          ++frozen;
        else
          {
            ++fresh;
            EXPECT_GT(id, distributed_base + 101);
            EXPECT_EQ(v.provenance, Provenance::generated);
          }
      }
  for (const auto &[id, t] : generated.network.terminals)
    if (t.owner == rank)
      {
        ++terminal_count;
        total_flow += generated.target_terminal_flows.at(id);
      }
  EXPECT_EQ(Utilities::MPI::sum(frozen, comm), 2u);
  EXPECT_EQ(Utilities::MPI::sum(fresh, comm), 10u);
  EXPECT_EQ(Utilities::MPI::sum(terminal_count, comm), 6u);
  EXPECT_NEAR(Utilities::MPI::sum(total_flow, comm) / p.reference_flow,
              1.,
              2.e-13);
  const auto repeated = generate(bg, MappingQ1<3>(), input, p, comm);
  ASSERT_TRUE(repeated.validation.valid());
  unchanged(generated.network, repeated.network);
  EXPECT_EQ(generated.sampled_points, repeated.sampled_points);
  unchanged(distributed_input(false), input);
}

TEST(VascularImported, BOTH_PublicInvalidAttachmentIsRejectedCollectively)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  parallel::distributed::Triangulation<3> bg(comm);
  distributed_background(bg);
  auto       input = distributed_input();
  Parameters p;
  controls(p);
  p.terminal_count     = 2;
  p.selected_terminals = {distributed_base + 200, distributed_base + 200};
  EXPECT_FALSE(generate(bg, MappingQ1<3>(), input, p, comm).validation.valid());
  p.selected_terminals = {distributed_base + 999};
  EXPECT_FALSE(generate(bg, MappingQ1<3>(), input, p, comm).validation.valid());
  p.selected_terminals.clear();
  p.terminal_count = 1;
  EXPECT_FALSE(generate(bg, MappingQ1<3>(), input, p, comm).validation.valid());
  p.terminal_count = 2;
  // One real root is accepted in the first round; the other points back into
  // its frozen incoming vessel. Failure must discard the whole startup.
  p.seed_direction       = Point<3>(-1., 0., 0.);
  const auto rolled_back = generate(bg, MappingQ1<3>(), input, p, comm);
  EXPECT_FALSE(rolled_back.validation.valid());
  unchanged(input, rolled_back.network);
  EXPECT_EQ(rolled_back.network.vessels.size(), input.vessels.size());
  EXPECT_TRUE(rolled_back.attachment_roots.empty());
  p.seed_direction = Point<3>();
  if (rank == size - 1)
    input.terminals[distributed_base + 202] =
      input.terminals.at(distributed_base + 200);
  EXPECT_FALSE(generate(bg, MappingQ1<3>(), input, p, comm).validation.valid());
  input = distributed_input();
  if (rank == size - 1)
    input.vessels.at(distributed_base + 100).radii[1] = -1.;
  EXPECT_FALSE(generate(bg, MappingQ1<3>(), input, p, comm).validation.valid());
}

TEST(VascularImported, BOTH_PublicExteriorLeafKeepsRemoteIncidence)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  parallel::distributed::Triangulation<3> bg(comm);
  distributed_background(bg);
  auto     input = distributed_input();
  const Id port = distributed_base + 500, leaf = port + 1, vessel = port + 2,
           terminal = port + 3;
  if (rank == 0 || rank == size - 1)
    input.nodes[leaf] = {Point<3>(-2., .9, .5),
                         size - 1,
                         Port::none,
                         Provenance::imported};
  if (rank == 0)
    {
      input.nodes[port] = {Point<3>(.5, .9, .5),
                           0,
                           Port::arterial_inlet,
                           Provenance::imported};
      Vessel v;
      v.nodes               = {{port, leaf}};
      v.radii               = {{.01, .01}};
      v.viscosity           = .004;
      v.owner               = 0;
      input.vessels[vessel] = v;
    }
  if (rank == size - 1)
    input.terminals[terminal] = {
      leaf, invalid_id, System::arterial, size - 1, 7.};
  const auto relevant = redistribute(bg, MappingQ1<3>(), input, 0., comm);
  if (rank == size - 1)
    {
      EXPECT_TRUE(relevant.vessels.count(vessel));
      EXPECT_TRUE(relevant.nodes.count(port));
    }
  Parameters p;
  controls(p);
  p.seed_direction     = Point<3>();
  p.terminal_count     = 2;
  const auto generated = generate(bg, MappingQ1<3>(), input, p, comm);
  ASSERT_TRUE(generated.validation.valid());
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(
                                  generated.attachment_roots.size()),
                                comm),
            2u);
  if (rank == size - 1)
    {
      EXPECT_TRUE(generated.network.vessels.count(vessel));
      EXPECT_DOUBLE_EQ(generated.network.terminals.at(terminal).demand, 7.);
    }
  p.selected_terminals = {terminal};
  const auto rejected  = generate(bg, MappingQ1<3>(), input, p, comm);
  EXPECT_FALSE(rejected.validation.valid());
  unchanged(input, rejected.network);
}

TEST(VascularImported, BOTH_PublicImportedStartupWithEmptyBackgroundRanks)
{
  const auto comm   = MPI_COMM_WORLD;
  const auto rank   = Utilities::MPI::this_mpi_process(comm);
  const auto source = Utilities::MPI::n_mpi_processes(comm) - 1;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::hyper_cube(bg);
  NetworkData input;
  if (rank == source)
    {
      input.nodes[10] = {Point<3>(.1, .5, .5),
                         source,
                         Port::arterial_inlet,
                         Provenance::imported};
      input.nodes[20] = {Point<3>(.3, .5, .5),
                         source,
                         Port::none,
                         Provenance::imported};
      Vessel v;
      v.nodes              = {{10, 20}};
      v.radii              = {{.01, .008}};
      v.viscosity          = .004;
      v.owner              = source;
      input.vessels[100]   = v;
      input.terminals[200] = {20, 300, System::arterial, source, 2.};
    }
  Parameters p;
  controls(p);
  p.seed_length        = .15;
  p.terminal_count     = 3;
  p.selected_terminals = {200};
  p.reference_flow     = 1.e-9;
  const auto result    = generate(bg, MappingQ1<3>(), input, p, comm);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_EQ(Utilities::MPI::sum(
              static_cast<unsigned int>(result.attachment_roots.size()), comm),
            1u);
  EXPECT_EQ(Utilities::MPI::sum(
              static_cast<unsigned int>(result.territories.size()), comm),
            3u);
  EXPECT_EQ(result.accepted_insertions, 2u);
  EXPECT_NEAR(result.tissue.background_volume, 1., 1.e-13);
  for (const auto &[id, v] : result.network.vessels)
    if (id == 100)
      {
        EXPECT_EQ(v.radii, (std::array<double, 2>{{.01, .008}}));
        EXPECT_EQ(v.provenance, Provenance::imported);
      }
}
