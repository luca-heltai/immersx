// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/global_generation.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <set>

#include "../source/vascularization/network_serialization.h"
#include "test_paths.h"

using namespace dealii;
using namespace ImmersX::Vascularization;

namespace
{
  constexpr Id base = (Id{1} << 60) + 41;

  unsigned int
  rank()
  {
    return Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  }

  unsigned int
  size()
  {
    return Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  }

  NetworkData
  fixture()
  {
    NetworkData n;
    if (rank() != size() - 1)
      return n;
    // Reversed root/endpoint ordering forces intersections in the first draft.
    const std::array<double, 3> x{{2., 3., 1.}};
    const std::array<Port, 3>   ports{
        {Port::arterial_inlet, Port::venous_outlet, Port::portal_inlet}};
    for (unsigned int s = 0; s < 3; ++s)
      {
        const Id id     = base + 10 * s;
        n.nodes[id]     = {Point<3>(x[s], .05, .5),
                           rank(),
                           ports[s],
                           Provenance::imported};
        n.nodes[id + 1] = {Point<3>(x[s], .1, .5),
                           rank(),
                           Port::none,
                           Provenance::imported};
        Vessel v;
        v.nodes                          = {{id, id + 1}};
        v.radii                          = {{.012, .01}};
        v.viscosity                      = .004;
        v.system                         = static_cast<System>(s);
        v.owner                          = rank();
        v.physical_properties["density"] = 1050. + s;
        n.vessels[id + 2]                = v;
        n.terminals[id + 3] = {id + 1, invalid_id, v.system, rank(), 1.};
      }
    return n;
  }

  NetworkData
  gathered(const NetworkData &n)
  {
    NetworkData owned;
    for (const auto &[id, node] : n.nodes)
      if (node.owner == rank())
        owned.nodes[id] = node;
    for (const auto &[id, v] : n.vessels)
      if (v.owner == rank())
        owned.vessels[id] = v;
    for (const auto &[id, t] : n.terminals)
      if (t.owner == rank())
        owned.terminals[id] = t;
    NetworkData all;
    // This exhaustive oracle gathers only the small analytical test fixture.
    for (const auto &part : Utilities::MPI::all_gather(MPI_COMM_WORLD, owned))
      {
        for (const auto &[id, node] : part.nodes)
          EXPECT_TRUE(all.nodes.emplace(id, node).second);
        for (const auto &[id, v] : part.vessels)
          EXPECT_TRUE(all.vessels.emplace(id, v).second);
        for (const auto &[id, t] : part.terminals)
          EXPECT_TRUE(all.terminals.emplace(id, t).second);
      }
    return all;
  }

  std::filesystem::path
  export_stage(const NetworkData &n,
               const std::string &label,
               const unsigned int index,
               const bool         certified,
               const unsigned int count)
  {
#ifdef DEBUG
    const std::string variant = "debug";
#else
    const std::string variant = "release";
#endif
    static std::map<std::string, std::string> runs;
    if (index == 0)
      {
        std::string unique;
        if (rank() == 0)
          unique = std::to_string(std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count());
        runs[label] = Utilities::MPI::broadcast(MPI_COMM_WORLD, unique, 0);
      }
    const auto directory = ImmersX::TestPaths::output_path(
      "global-forest/" + label + "/" + variant + "-mpi" +
      std::to_string(size()) + "/run-" + runs.at(label) + "/stage-" +
      std::to_string(index));
    if (rank() == 0)
      {
        std::filesystem::create_directories(directory.parent_path());
        AssertThrow(std::filesystem::create_directory(directory),
                    ExcMessage(
                      "Refusing to overwrite a global-forest test stage."));
      }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto open = [&](const std::string &name) {
      std::ofstream stream(directory /
                           (name + "-rank" + std::to_string(rank()) + ".csv"));
      stream.exceptions(std::ios::badbit | std::ios::failbit);
      stream << std::setprecision(17);
      return stream;
    };
    auto nodes = open("nodes");
    nodes << "id,x,y,z,port,provenance,owner\n";
    for (const auto &[id, node] : n.nodes)
      if (node.owner == rank())
        nodes << id << "," << node.position[0] << "," << node.position[1] << ","
              << node.position[2] << "," << static_cast<unsigned int>(node.port)
              << "," << static_cast<unsigned int>(node.provenance) << ","
              << node.owner << "\n";
    nodes.close();
    auto vessels = open("vessels");
    vessels
      << "id,node0,node1,radius0,radius1,viscosity,system,provenance,owner\n";
    for (const auto &[id, v] : n.vessels)
      if (v.owner == rank())
        vessels << id << "," << v.nodes[0] << "," << v.nodes[1] << ","
                << v.radii[0] << "," << v.radii[1] << "," << v.viscosity << ","
                << static_cast<unsigned int>(v.system) << ","
                << static_cast<unsigned int>(v.provenance) << "," << v.owner
                << "\n";
    vessels.close();
    auto terminals = open("terminals");
    terminals << "id,node,territory,system,demand,owner\n";
    for (const auto &[id, t] : n.terminals)
      if (t.owner == rank())
        terminals << id << "," << t.node << "," << t.territory << ","
                  << static_cast<unsigned int>(t.system) << "," << t.demand
                  << "," << t.owner << "\n";
    terminals.close();
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank() == 0)
      {
        std::ofstream ready(directory / "stage.json");
        ready.exceptions(std::ios::badbit | std::ios::failbit);
        ready << "{\"ranks\":" << size()
              << ",\"effective_common_terminal_count\":" << count
              << ",\"geometry_certified\":" << (certified ? "true" : "false")
              << ",\"generation_complete\":false,\"label\":\"" << label
              << "\"}\n";
        ready.close();
      }
    MPI_Barrier(MPI_COMM_WORLD);
    return directory;
  }

  void
  controls(Parameters &p, GlobalGenerationParameters &c)
  {
    p.length_scale                      = "meter";
    p.terminal_count_multiplier_factor  = "none";
    p.terminal_count                    = 4;
    p.calibration.maximum_terminals     = 4;
    p.minimum_length                    = .001;
    p.minimum_radius                    = .0001;
    p.minimum_angle                     = 5.;
    p.density_beta                      = 0.;
    p.reference_flow                    = 1.e-6;
    p.portal_flow_fraction              = .5;
    p.venous.enabled                    = true;
    p.portal.enabled                    = true;
    p.venous.terminal_separation        = .08;
    p.portal.terminal_separation        = .08;
    p.coverage.service_radius           = 0.;
    p.reference_conditions.pressures    = {{base, 2000.},
                                           {base + 10, 0.},
                                           {base + 20, 1000.}};
    c.initial_terminals                 = 1;
    c.topology_trials                   = 2;
    c.pool_attempt_budget               = 10000;
    c.geometry.minimum_length_m         = .001;
    c.geometry.maximum_length_m         = 5.;
    c.geometry.clearance_m              = .001;
    c.geometry.maximum_outer_iterations = 30;
    c.geometry.maximum_inner_iterations = 500;
  }

  void
  preserve_original(const NetworkData &all)
  {
    for (unsigned int s = 0; s < 3; ++s)
      {
        const Id id = base + 10 * s;
        ASSERT_TRUE(all.vessels.count(id + 2));
        const auto &v = all.vessels.at(id + 2);
        EXPECT_EQ(v.nodes, (std::array<Id, 2>{{id, id + 1}}));
        EXPECT_EQ(v.radii, (std::array<double, 2>{{.012, .01}}));
        EXPECT_DOUBLE_EQ(v.viscosity, .004);
        EXPECT_EQ(v.provenance, Provenance::imported);
        EXPECT_EQ(v.system, static_cast<System>(s));
        EXPECT_EQ(v.physical_properties,
                  (std::map<std::string, double>{{"density", 1050. + s}}));
        const std::array<Port, 3> ports{
          {Port::arterial_inlet, Port::venous_outlet, Port::portal_inlet}};
        EXPECT_EQ(all.nodes.at(id).port, ports[s]);
        EXPECT_EQ(all.nodes.at(id + 1).port, Port::none);
        for (Id node : {id, id + 1})
          {
            EXPECT_EQ(all.nodes.at(node).provenance, Provenance::imported);
            EXPECT_EQ(all.nodes.at(node).inward_port_normal, Point<3>());
          }
        EXPECT_EQ(all.nodes.at(id).position,
                  Point<3>(s == 0 ? 2. :
                           s == 1 ? 3. :
                                    1.,
                           .05,
                           .5));
        EXPECT_EQ(all.nodes.at(id + 1).position,
                  Point<3>(s == 0 ? 2. :
                           s == 1 ? 3. :
                                    1.,
                           .1,
                           .5));
      }
  }

  void
  shared_pool(const NetworkData &all, unsigned int expected)
  {
    std::map<Id, std::array<const Terminal *, 3>> common;
    for (const auto &[id, t] : all.terminals)
      {
        EXPECT_NE(t.territory, invalid_id);
        EXPECT_TRUE(common[t.territory][static_cast<unsigned int>(t.system)] ==
                    nullptr);
        common[t.territory][static_cast<unsigned int>(t.system)] = &t;
      }
    ASSERT_EQ(common.size(), expected);
    for (const auto &[id, ts] : common)
      {
        for (unsigned int s = 0; s < 3; ++s)
          ASSERT_NE(ts[s], nullptr);
        EXPECT_NE(ts[0]->node, ts[1]->node);
        EXPECT_NE(ts[0]->node, ts[2]->node);
        EXPECT_NE(ts[1]->node, ts[2]->node);
        EXPECT_DOUBLE_EQ(ts[0]->demand, ts[1]->demand);
        EXPECT_DOUBLE_EQ(ts[0]->demand, ts[2]->demand);
        EXPECT_GT(ts[0]->demand, 0.);
      }
  }

  void
  independent_murray(const NetworkData &all)
  {
    std::map<Id, std::vector<Id>> outgoing;
    std::map<Id, double>          demand;
    for (const auto &[id, v] : all.vessels)
      outgoing[v.nodes[0]].push_back(id);
    for (const auto &[id, t] : all.terminals)
      demand[t.node] = t.demand;
    std::set<Id>              visiting;
    std::function<double(Id)> subtree = [&](Id node) {
      const auto known = demand.find(node);
      if (known != demand.end())
        return known->second;
      if (!visiting.insert(node).second)
        {
          ADD_FAILURE() << "Cycle in generated tree at node " << node;
          return 0.;
        }
      double sum = 0.;
      for (Id id : outgoing[node])
        sum += subtree(all.vessels.at(id).nodes[1]);
      visiting.erase(node);
      demand[node] = sum;
      return sum;
    };
    for (unsigned int system = 0; system < 3; ++system)
      {
        const double total = subtree(base + 10 * system + 1);
        ASSERT_GT(total, 0.);
        for (const auto &[id, v] : all.vessels)
          if (v.system == static_cast<System>(system) &&
              v.provenance == Provenance::generated)
            {
              const double expected =
                .01 * .01 * .01 * subtree(v.nodes[1]) / total;
              for (double radius : v.radii)
                EXPECT_NEAR(std::pow(radius, 3) / expected, 1., 2.e-12);
              EXPECT_GT(all.nodes.at(v.nodes[0])
                          .position.distance(all.nodes.at(v.nodes[1]).position),
                        0.);
            }
      }
  }

  bool
  first_draft_crosses(const NetworkData &all)
  {
    std::array<const Vessel *, 3> edges{};
    for (const auto &[id, v] : all.vessels)
      if (v.provenance == Provenance::generated)
        edges[static_cast<unsigned int>(v.system)] = &v;
    if (!edges[0] || !edges[1] || !edges[2])
      return false;
    const auto  &a = *edges[0], &p = *edges[2];
    const auto   a0 = all.nodes.at(a.nodes[0]).position;
    const auto   p0 = all.nodes.at(p.nodes[0]).position;
    const auto   a1 = all.nodes.at(a.nodes[1]).position;
    const auto   p1 = all.nodes.at(p.nodes[1]).position;
    const double t  = (a0[0] - p0[0]) / (a0[0] - p0[0] + p1[0] - a1[0]);
    return t > 0. && t < 1. &&
           ((1. - t) * a0 + t * a1).distance((1. - t) * p0 + t * p1) < 1.e-12;
  }
} // namespace

TEST(VascularForest, BOTH_GlobalCommonPoolDoublesAndRepairsAllForests)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 2, 1},
                                            Point<3>(),
                                            Point<3>(4., 2., 1.));
  Parameters                 p;
  GlobalGenerationParameters c;
  controls(p, c);
  unsigned int              index = 0;
  std::vector<unsigned int> drafts, certified;
  struct FixedEndpoint
  {
    Id       node;
    Point<3> position;
    System   system;
    double   demand;
  };
  std::map<Id, FixedEndpoint> fixed;
  c.stage_callback = [&](const GenerationResult &r, GlobalStage stage) {
    export_stage(r.network,
                 "doubling",
                 index++,
                 stage == GlobalStage::certified,
                 r.effective_terminal_count);
    EXPECT_FALSE(r.validation.valid());
    ASSERT_TRUE(r.global_generation);
    const auto all = gathered(r.network);
    preserve_original(all);
    shared_pool(all, r.effective_terminal_count);
    for (const auto &[id, endpoint] : fixed)
      EXPECT_TRUE(all.terminals.count(id));
    for (const auto &[id, t] : all.terminals)
      {
        const auto position = all.nodes.at(t.node).position;
        const auto found    = fixed.find(id);
        if (found != fixed.end())
          {
            EXPECT_EQ(found->second.node, t.node);
            EXPECT_EQ(found->second.position, position);
            EXPECT_EQ(found->second.system, t.system);
            EXPECT_DOUBLE_EQ(found->second.demand, t.demand);
          }
        else
          fixed[id] = {t.node, position, t.system, t.demand};
      }
    for (const auto &territory : r.territories)
      {
        EXPECT_EQ(territory.owner, rank());
        bool found = false;
        for (const auto &[id, t] : all.terminals)
          if (t.system == System::arterial && t.territory == territory.id)
            {
              found = true;
              EXPECT_EQ(t.owner, territory.owner);
              EXPECT_EQ(all.nodes.at(t.node).position, territory.position);
              EXPECT_DOUBLE_EQ(t.demand, territory.demand);
            }
        EXPECT_TRUE(found);
      }
    if (stage == GlobalStage::draft)
      {
        drafts.push_back(r.effective_terminal_count);
        if (r.effective_terminal_count == 1 && drafts.size() == 1)
          {
            EXPECT_TRUE(first_draft_crosses(all));
          }
      }
    else
      {
        certified.push_back(r.effective_terminal_count);
        ASSERT_TRUE(r.global_geometry);
        EXPECT_TRUE(r.global_geometry->certified);
        independent_murray(all);
      }
  };
  const auto result =
    generate_global(bg, MappingQ1<3>(), fixture(), p, c, MPI_COMM_WORLD);
  export_stage(result.network,
               "doubling-final",
               0,
               result.global_geometry && result.global_geometry->certified,
               result.effective_terminal_count);
  std::string messages;
  for (const auto &issue : result.validation.issues)
    messages += issue.message + "\n";
  ASSERT_TRUE(result.validation.valid()) << messages;
  EXPECT_EQ(drafts, (std::vector<unsigned int>{1, 2, 4}));
  EXPECT_EQ(certified, drafts);
  EXPECT_EQ(result.requested_terminal_count, 4u);
  EXPECT_EQ(result.effective_terminal_count, 4u);
  EXPECT_EQ(Utilities::MPI::sum(result.terminal_triplets.size(),
                                MPI_COMM_WORLD),
            4u);
  ASSERT_TRUE(result.global_generation);
  EXPECT_EQ(result.global_generation->upfront_targets, 4u);
  EXPECT_EQ(result.global_generation->history.size(), 3u);
  preserve_original(gathered(result.network));
  shared_pool(gathered(result.network), 4);
}

TEST(VascularForest, BOTH_GlobalViewBudgetRejectsWithoutPartialSuccess)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 2, 1},
                                            Point<3>(),
                                            Point<3>(4., 2., 1.));
  Parameters                 p;
  GlobalGenerationParameters c;
  controls(p, c);
  c.maximum_group_records = 1;
  unsigned int callbacks  = 0;
  c.stage_callback        = [&](const GenerationResult &r, GlobalStage stage) {
    export_stage(r.network,
                 "budget",
                 callbacks++,
                 stage == GlobalStage::certified,
                 r.effective_terminal_count);
  };
  const auto result =
    generate_global(bg, MappingQ1<3>(), fixture(), p, c, MPI_COMM_WORLD);
  export_stage(
    result.network, "budget-final", 0, false, result.effective_terminal_count);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_EQ(callbacks, 0u);
  EXPECT_EQ(result.effective_terminal_count, 0u);
  preserve_original(gathered(result.network));
}

TEST(VascularForest, MPI_GlobalRejectsDifferentPoolControls)
{
  ASSERT_GT(size(), 1u);
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 2, 1},
                                            Point<3>(),
                                            Point<3>(4., 2., 1.));
  Parameters                 p;
  GlobalGenerationParameters c;
  controls(p, c);
  c.random_seed += rank();
  const auto input = fixture();
  export_stage(input, "different-controls", 0, false, 0);
  EXPECT_THROW(generate_global(bg, MappingQ1<3>(), input, p, c, MPI_COMM_WORLD),
               dealii::ExceptionBase);
}
