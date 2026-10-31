// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/distribution.h>

#include <algorithm>
#include <array>

#include "../source/vascularization/growth_geometry.h"
#include "../source/vascularization/growth_owned_rows.h"
#include "../source/vascularization/network_serialization.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
using ImmersX::Vascularization::internal::BackgroundGeometry;
using ImmersX::Vascularization::internal::GrowthOwnedRows;
using ImmersX::Vascularization::internal::GrowthOwnershipDelta;
using ImmersX::Vascularization::internal::GrowthSplit;
using ImmersX::Vascularization::internal::prepare_growth_ownership;
namespace
{
  constexpr Id row_base = Id(1) << 54;
  NetworkData
  native_row_seed(double extent, unsigned int unrelated, bool shared)
  {
    NetworkData n;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
      return n;
    n.nodes[row_base + 10]          = {{0., .5, .5},
                                       0,
                                       Port::arterial_inlet,
                                       Provenance::imported,
                                       {1., 0., 0.}};
    n.nodes[row_base + 20]          = {{.975 * extent, .5, .5},
                                       0,
                                       Port::none,
                                       Provenance::generated};
    n.nodes[row_base + 90].position = {-.5, .1, .1};
    Vessel v;
    v.nodes     = {{row_base + 10, row_base + 20}};
    v.radii     = {{.01, .015}};
    v.viscosity = .004;
    v.physical_properties["retained SI property"] = 17.25;
    n.vessels[row_base + 100]                     = v;
    n.terminals[row_base + 200]                   = {
      row_base + 20, row_base + 300, System::arterial, 0, 3.};
    for (unsigned int i = 0; i < unrelated; ++i)
      {
        const Id     a = row_base + 1000 + 2 * i, b = a + 1;
        const double y                = .1 + .1 * i / std::max(1u, unrelated);
        n.nodes[a].position           = {.1 * extent, y, .2};
        n.nodes[b].position           = {.15 * extent, y, .2};
        auto e                        = v;
        e.nodes                       = {{a, b}};
        n.vessels[row_base + 500 + i] = e;
      }
    if (shared)
      {
        n.nodes[row_base + 40].position = {-.225 * extent, .25, .5};
        auto e                          = v;
        e.nodes                         = {{row_base + 40, row_base + 20}};
        n.vessels[row_base + 400]       = e;
      }
    return n;
  }
  NetworkData
  owned_projection(const NetworkData &full)
  {
    const auto  rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    NetworkData expected;
    for (const auto &[id, n] : full.nodes)
      if (n.owner == rank)
        expected.nodes.emplace(id, n);
    for (const auto &[id, v] : full.vessels)
      if (v.owner == rank)
        {
          expected.vessels.emplace(id, v);
          for (Id node : v.nodes)
            expected.nodes.emplace(node, full.nodes.at(node));
        }
    for (const auto &[id, t] : full.terminals)
      if (t.owner == rank)
        {
          expected.terminals.emplace(id, t);
          expected.nodes.emplace(t.node, full.nodes.at(t.node));
        }
    return expected;
  }
  unsigned int
  native_writer(const NetworkData &rows, Id vessel)
  {
    const auto rank  = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto size  = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const auto found = rows.vessels.find(vessel);
    return Utilities::MPI::min(
      found != rows.vessels.end() && found->second.owner == rank ? rank : size,
      MPI_COMM_WORLD);
  }
  GrowthSplit
  row_split(const NetworkData &rows, double extent, bool second = false)
  {
    GrowthSplit s;
    s.vessel     = row_base + (second ? 101 : 100);
    s.owner      = native_writer(rows, s.vessel);
    s.old_nodes  = second ? std::array<Id, 2>{{row_base + 100, row_base + 20}} :
                            std::array<Id, 2>{{row_base + 10, row_base + 20}};
    s.old_points = {{Point<3>(second ? .225 * extent : 0., .5, .5),
                     Point<3>(.975 * extent, .5, .5)}};
    s.junction   = row_base + (second ? 110 : 100);
    s.tip        = s.junction + 1;
    s.continuation      = s.tip;
    s.leaf              = s.junction + 2;
    s.terminal          = s.leaf;
    s.territory         = s.junction + 3;
    s.junction_position = {extent * (second ? .5 : .225), .5, .5};
    s.tip_position      = {extent * (second ? .7 : .8), second ? .25 : .7, .5};
    s.demand            = second ? .5 : 1.25;
    return s;
  }
  GrowthOwnershipDelta
  row_delta(GrowthOwnedRows    &rows,
            const GrowthSplit  &s,
            BackgroundGeometry &geometry)
  {
    std::vector<GrowthSplit> local;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) + 1 ==
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
      local.push_back(s);
    return prepare_growth_ownership(rows.network(),
                                    local,
                                    geometry,
                                    MPI_COMM_WORLD);
  }
  std::map<Id, double>
  native_updates(const NetworkData &rows, const std::map<Id, double> &global)
  {
    std::map<Id, double> result;
    for (const auto &[id, d] : global)
      if (rows.terminals.count(id))
        result.emplace(id, d);
    return result;
  }
  NetworkData
  full_row_step(const Triangulation<3>     &bg,
                const MappingQ1<3>         &mapping,
                NetworkData                 full,
                const GrowthSplit          &s,
                const std::map<Id, double> &global)
  {
    const auto rank    = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    bool       applied = true;
    if (rank == s.owner)
      applied = ImmersX::Vascularization::internal::apply_growth_split(full, s);
    EXPECT_EQ(Utilities::MPI::min(static_cast<unsigned int>(applied),
                                  MPI_COMM_WORLD),
              1u);
    for (const auto &[id, d] : global)
      {
        const auto found = full.terminals.find(id);
        if (found != full.terminals.end() && found->second.owner == rank)
          found->second.demand = d;
      }
    return redistribute(bg, mapping, full, 0., MPI_COMM_WORLD);
  }
} // namespace
TEST(VascularForest,
     BOTH_NativeRowsJournalMatchesFullOwnershipAndRestoresOriginalNodes)
{
  for (unsigned int cells : {1u, 4u})
    for (unsigned int unrelated : {0u, 64u})
      for (bool shared : {false, true})
        {
          parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
          GridGenerator::subdivided_hyper_rectangle(bg,
                                                    {cells, 1, 1},
                                                    Point<3>(),
                                                    Point<3>(cells, 1., 1.));
          MappingQ1<3>    mapping;
          auto            full = redistribute(bg,
                                   mapping,
                                   native_row_seed(cells, unrelated, shared),
                                   0.,
                                   MPI_COMM_WORLD);
          GrowthWork      work;
          GrowthOwnedRows rows(full, MPI_COMM_WORLD, &work);
          EXPECT_EQ(Utilities::pack(rows.network(), false),
                    Utilities::pack(owned_projection(full), false));
          BackgroundGeometry geometry(bg, mapping);
          for (bool second : {false, true})
            {
              const auto split = row_split(rows.network(), cells, second);
              const auto delta = row_delta(rows, split, geometry);
              ASSERT_TRUE(delta.validation.valid());
              const std::map<Id, double> global =
                second ? std::map<Id, double>{{row_base + 200, 1.},
                                              {row_base + 102, 1.5}} :
                         std::map<Id, double>{{row_base + 200, 1.75}};
              const auto demands = native_updates(rows.network(), global);
              const auto expected_full =
                full_row_step(bg, mapping, full, split, global);
              const auto expected =
                Utilities::pack(owned_projection(expected_full), false);
              const auto original = Utilities::pack(rows.network(), false);
              std::map<Id, const Node *>     nodes;
              std::map<Id, const Vessel *>   vessels;
              std::map<Id, const Terminal *> terminals;
              for (const auto &[id, n] : rows.network().nodes)
                nodes[id] = &n;
              for (const auto &[id, v] : rows.network().vessels)
                vessels[id] = &v;
              for (const auto &[id, t] : rows.network().terminals)
                terminals[id] = &t;
              const auto before   = work;
              const auto prepared = rows.prepare(delta, demands);
              ASSERT_EQ(Utilities::MPI::min(static_cast<unsigned int>(
                                              prepared.valid()),
                                            MPI_COMM_WORLD),
                        1u);
              EXPECT_EQ(Utilities::pack(rows.network(), false), expected);
              EXPECT_THROW(rows.prepare(delta, demands), ExceptionBase);
              rows.discard();
              EXPECT_EQ(Utilities::pack(rows.network(), false), original);
              for (const auto &[id, address] : nodes)
                {
                  const auto found = rows.network().nodes.find(id);
                  EXPECT_NE(found, rows.network().nodes.end());
                  if (found != rows.network().nodes.end())
                    {
                      EXPECT_EQ(&found->second, address);
                    }
                }
              for (const auto &[id, address] : vessels)
                {
                  const auto found = rows.network().vessels.find(id);
                  EXPECT_NE(found, rows.network().vessels.end());
                  if (found != rows.network().vessels.end())
                    {
                      EXPECT_EQ(&found->second, address);
                    }
                }
              for (const auto &[id, address] : terminals)
                {
                  const auto found = rows.network().terminals.find(id);
                  EXPECT_NE(found, rows.network().terminals.end());
                  if (found != rows.network().terminals.end())
                    {
                      EXPECT_EQ(&found->second, address);
                    }
                }
              const auto transaction = work - before;
              EXPECT_LE(
                Utilities::MPI::sum(
                  transaction[GrowthOperation::validation_record_visits],
                  MPI_COMM_WORLD),
                64u);
              EXPECT_LE(Utilities::MPI::sum(
                          transaction[GrowthOperation::transaction_records],
                          MPI_COMM_WORLD),
                        128u);
              EXPECT_GT(Utilities::MPI::sum(
                          transaction[GrowthOperation::rollback_records],
                          MPI_COMM_WORLD),
                        0u);
              const auto retry = rows.prepare(delta, demands);
              ASSERT_EQ(Utilities::MPI::min(static_cast<unsigned int>(
                                              retry.valid()),
                                            MPI_COMM_WORLD),
                        1u);
              rows.adopt();
              EXPECT_EQ(Utilities::pack(rows.network(), false), expected);
              for (const auto &[id, address] : vessels)
                if (!delta.changed_owned.count(id) &&
                    !delta.released_owned.count(id))
                  {
                    const auto found = rows.network().vessels.find(id);
                    EXPECT_NE(found, rows.network().vessels.end());
                    if (found != rows.network().vessels.end())
                      {
                        EXPECT_EQ(&found->second, address);
                      }
                  }
              full = expected_full;
            }
        }
}
TEST(VascularForest,
     BOTH_NativeRowsRejectLocalErrorsWithCollectiveRollbackAndRetry)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {4, 1, 1},
                                            Point<3>(),
                                            Point<3>(4., 1., 1.));
  MappingQ1<3> mapping;
  const auto   full =
    redistribute(bg, mapping, native_row_seed(4., 8, true), 0., MPI_COMM_WORLD);
  GrowthOwnedRows    rows(full, MPI_COMM_WORLD);
  BackgroundGeometry geometry(bg, mapping);
  const auto         split = row_split(rows.network(), 4.);
  const auto         delta = row_delta(rows, split, geometry);
  ASSERT_TRUE(delta.validation.valid());
  const auto original = Utilities::pack(rows.network(), false);
  for (unsigned int kind : {0u, 1u})
    {
      auto patch   = delta;
      auto demands = native_updates(rows.network(), {{row_base + 200, 1.75}});
      if (kind == 0 && demands.count(row_base + 200))
        demands[row_base + 200] = 0.;
      else if (kind == 1)
        {
          const auto node = patch.owned.nodes.find(row_base + 10);
          if (node != patch.owned.nodes.end())
            node->second.provenance = Provenance::generated;
        }
      const auto validation = rows.prepare(patch, demands);
      EXPECT_EQ(Utilities::MPI::min(static_cast<unsigned int>(
                                      validation.valid()),
                                    MPI_COMM_WORLD),
                0u);
      rows.discard();
      EXPECT_EQ(Utilities::pack(rows.network(), false), original);
    }
  const auto validation =
    rows.prepare(delta,
                 native_updates(rows.network(), {{row_base + 200, 1.75}}));
  ASSERT_EQ(Utilities::MPI::min(static_cast<unsigned int>(validation.valid()),
                                MPI_COMM_WORLD),
            1u);
  rows.adopt();
  const auto expected =
    full_row_step(bg, mapping, full, split, {{row_base + 200, 1.75}});
  EXPECT_EQ(Utilities::pack(rows.network(), false),
            Utilities::pack(owned_projection(expected), false));
}
