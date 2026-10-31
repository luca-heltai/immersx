// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <gtest/gtest.h>
#include <immersx/vascularization/distribution.h>
#include <immersx/vascularization/generation.h>

#include <functional>

#include "../source/vascularization/forest_state.h"
#include "../source/vascularization/growth_batch.h"
#include "../source/vascularization/growth_commit.h"
#include "../source/vascularization/growth_loop.h"
#include "../source/vascularization/growth_paths.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
using namespace ImmersX::Vascularization::internal;
namespace
{
  constexpr Id base = (Id{1} << 60) + 13;
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
  std::string
  issues(const ForestState &s)
  {
    std::string text;
    for (const auto &i : s.validation.issues)
      text += i.message + "\n";
    return text;
  }
  void
  background(parallel::distributed::Triangulation<3> &bg)
  {
    GridGenerator::subdivided_hyper_rectangle(bg,
                                              {8, 1, 1},
                                              Point<3>(),
                                              Point<3>(4., 2., 2.));
  }
  NetworkData
  forest(bool pair_systems = false)
  {
    NetworkData n;
    if (rank() != size() - 1)
      return n;
    // Two real attachment roots, each with three weighted leaves. Input
    // prefixes deliberately have generated provenance: mutability is explicit.
    for (unsigned int tree = 0; tree < 2; ++tree)
      {
        const Id       off  = base + 100 * tree;
        const double   y    = .5 + tree;
        const Point<3> p[7] = {{.1, y, .5},
                               {.3, y, .5},
                               {1.2, y, .5},
                               {2.1, y, .5},
                               {3.7, y - .2, .5},
                               {3.5, y + .2, .5},
                               {3.8, y, .9}};
        for (unsigned int j = 0; j < 7; ++j)
          {
            n.nodes[off + j].position = p[j];
            n.nodes[off + j].owner    = rank();
          }
        n.nodes[off].port = tree == 1 && pair_systems ? Port::venous_outlet :
                                                        Port::arterial_inlet;
        const std::array<Id, 2> ends[6] = {{{off, off + 1}},
                                           {{off + 1, off + 2}},
                                           {{off + 2, off + 3}},
                                           {{off + 2, off + 4}},
                                           {{off + 3, off + 5}},
                                           {{off + 3, off + 6}}};
        for (unsigned int j = 0; j < 6; ++j)
          {
            Vessel v;
            v.nodes                  = ends[j];
            const double root_radius = .006 + .001 * tree;
            v.radii     = {{j == 0 ? .009 : root_radius, root_radius}};
            v.viscosity = .004 + j * .0001;
            v.owner     = rank();
            v.system =
              tree == 1 && pair_systems ? System::venous : System::arterial;
            v.provenance                     = Provenance::generated;
            v.physical_properties["density"] = 1050. + tree;
            n.vessels[off + 10 + j]          = v;
          }
        for (unsigned int j = 0; j < 3; ++j)
          n.terminals[off + 30 + j] = {off + 4 + j,
                                       off + 40 + j,
                                       n.vessels.at(off + 11).system,
                                       rank(),
                                       1. + 2 * j + tree};
      }
    return n;
  }
  std::set<Id>
  mutable_ids(const NetworkData &n)
  {
    std::set<Id> ids;
    for (const auto &[id, v] : n.vessels)
      if (v.owner == rank())
        {
          const auto local = id >= base + 100 ? id - (base + 100) : id - base;
          if (local >= 11 && local <= 15)
            ids.insert(id);
        }
    return ids;
  }
  std::map<Id, double>
  roots(const NetworkData &n)
  {
    std::map<Id, double> r;
    for (unsigned int i = 0; i < 2; ++i)
      {
        const Id id = base + 100 * i + 11;
        if (n.vessels.count(id) && n.vessels.at(id).owner == rank())
          r[id] = .006 + i * .001;
      }
    return r;
  }
  ForestState
  evaluate(const NetworkData &n, const Parameters &p)
  {
    return evaluate_forest(n, mutable_ids(n), roots(n), p, MPI_COMM_WORLD);
  }
  void
  oracle(const NetworkData &n,
         const ForestState &s,
         const Parameters  &p,
         bool               pair_systems = false)
  {
    EXPECT_TRUE(s.validation.valid()) << issues(s);
    double        local_volume[2] = {0., 0.}, local_power[2] = {0., 0.};
    double        local_moment[2][2] = {{0., 0.}, {0., 0.}};
    std::uint64_t local_count        = 0;
    for (const auto &[id, b] : s.owned)
      {
        ++local_count;
        const unsigned int tree  = id >= base + 100 ? 1 : 0;
        const Id           off   = base + 100 * tree;
        const auto         local = id - off;
        // Public fixture topology gives independent exact descendant demands:
        // first side leaf 1+t, continuation leaves 3+t and 5+t.
        const double       root_demand = 9. + 3. * tree;
        const double       d           = local == 11 ? root_demand :
                                         local == 12 ? 8. + 2. * tree :
                                         local == 13 ? 1. + tree :
                                         local == 14 ? 3. + tree :
                                                       5. + tree;
        const unsigned int system      = pair_systems && tree == 1 ? 1 : 0;
        const double       gamma =
          system == 0 ? p.arterial_murray_exponent : p.venous.murray_exponent;
        const double root_radius = .006 + .001 * tree;
        local_moment[tree][0] +=
          length(n, id) * std::pow(d / s.demand_scale, 2. / gamma);
        if (p.dissipation_weight > 0.)
          local_moment[tree][1] +=
            n.vessels.at(id).viscosity * length(n, id) *
            std::pow(d / s.demand_scale, 2. - 4. / gamma);
        EXPECT_NEAR(b.root_minimum_demand * s.demand_scale, 1. + tree, 1.e-12);
        const double minimum = local == 11 ? 1. + tree :
                               local == 12 ? 3. + tree :
                                             d;
        EXPECT_NEAR(b.minimum_demand * s.demand_scale, minimum, 1.e-12);
        EXPECT_EQ(b.root, off + 11);
        EXPECT_NEAR(b.demand * s.demand_scale / d, 1., 1.e-12);
        EXPECT_NEAR(b.root_demand * s.demand_scale / root_demand, 1., 1.e-12);
        EXPECT_NEAR(b.radius /
                      (root_radius * std::pow(d / root_demand, 1. / gamma)),
                    1.,
                    1.e-12);
        if (local == 11)
          {
            EXPECT_DOUBLE_EQ(b.radius, root_radius);
          }
        EXPECT_EQ(b.terminals, local == 11 ? 3u : local == 12 ? 2u : 1u);
        if (local == 11)
          EXPECT_EQ(b.parent, invalid_id);
        else
          EXPECT_EQ(b.parent, off + (local == 12 || local == 13 ? 11 : 12));
        if (p.reference_flow > 0.)
          {
            const double total = pair_systems ? root_demand : 21.;
            const double q     = p.reference_flow * d / total;
            EXPECT_NEAR(b.flow / q, 1., 1.e-12);
            const double l = length(n, id);
            const double r = 8. * n.vessels.at(id).viscosity * l /
                             (numbers::PI * std::pow(b.radius, 4));
            EXPECT_NEAR(b.resistance / r, 1., 1.e-12);
            local_power[system] += q * q * r;
          }
        local_volume[system] +=
          numbers::PI * b.radius * b.radius * length(n, id);
      }
    EXPECT_EQ(Utilities::MPI::sum(local_count, MPI_COMM_WORLD), 10u);
    for (unsigned int i = 0; i < 2; ++i)
      {
        const double volume =
                       Utilities::MPI::sum(local_volume[i], MPI_COMM_WORLD),
                     power =
                       Utilities::MPI::sum(local_power[i], MPI_COMM_WORLD);
        EXPECT_NEAR(s.volume[i], volume, 1.e-15);
        EXPECT_NEAR(s.power[i], power, 1.e-20);
        if (p.dissipation_weight > 0. && power > 0.)
          {
            const double numerator = power / p.reference_flow /
                                     p.reference_flow * s.total_demand[i] *
                                     s.total_demand[i];
            EXPECT_NEAR(s.power_numerator[i] / numerator, 1., 1.e-12);
          }
        else
          {
            EXPECT_DOUBLE_EQ(s.power_numerator[i], 0.);
          }
      }
    for (unsigned int tree = 0; tree < 2; ++tree)
      for (unsigned int moment = 0; moment < 2; ++moment)
        {
          const double total =
            Utilities::MPI::sum(local_moment[tree][moment], MPI_COMM_WORLD);
          for (const auto &[id, b] : s.owned)
            if (id >= base + 100 * tree + 11 && id <= base + 100 * tree + 15)
              {
                EXPECT_NEAR(moment == 0 ? b.root_volume_moment :
                                          b.root_resistance_moment,
                            total,
                            1.e-14);
              }
        }
    using Contribution = std::pair<Id, std::array<double, 2>>;
    std::map<unsigned int, std::vector<Contribution>> send;
    for (const auto &[id, b] : s.owned)
      {
        const auto  &v     = n.vessels.at(id);
        const double gamma = v.system == System::arterial ?
                               p.arterial_murray_exponent :
                               p.venous.murray_exponent;
        for (unsigned int end = 0; end < 2; ++end)
          send[n.nodes.at(v.nodes[end]).owner].push_back(
            {v.nodes[end],
             {{(end == 0 ? 1. : -1.) * b.flow,
               (end == 0 ? 1. : -1.) * std::pow(b.radius, gamma)}}});
      }
    const auto contributions =
      Utilities::MPI::some_to_some(MPI_COMM_WORLD, send);
    std::map<Id, std::array<double, 2>> sums;
    for (const auto &[source, values] : contributions)
      for (const auto &[node, value] : values)
        {
          sums[node][0] += value[0];
          sums[node][1] += value[1];
        }
    for (unsigned int tree = 0; tree < 2; ++tree)
      for (Id local : {Id{2}, Id{3}})
        {
          const Id node = base + 100 * tree + local;
          if (n.nodes.count(node) && n.nodes.at(node).owner == rank())
            {
              if (p.reference_flow > 0.)
                {
                  EXPECT_NEAR(sums.at(node)[0] / p.reference_flow, 0., 1.e-12);
                }
              const double gamma = pair_systems && tree == 1 ?
                                     p.venous.murray_exponent :
                                     p.arterial_murray_exponent;
              EXPECT_NEAR(sums.at(node)[1] /
                            std::pow(.006 + .001 * tree, gamma),
                          0.,
                          1.e-12);
            }
        }
    for (const auto &[id, q] : s.terminal_flows)
      {
        const auto &t = n.terminals.at(id);
        EXPECT_EQ(t.owner, rank());
        const double total =
          pair_systems ? (t.system == System::arterial ? 9. : 12.) : 21.;
        EXPECT_NEAR(q / (p.reference_flow * t.demand / total), 1., 1.e-12);
      }
  }
} // namespace
TEST(VascularForest,
     BOTH_WeightedDemandsAcrossPartitionsNormalizeEveryOwnedBranch)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  const auto n = redistribute(bg, MappingQ1<3>(), forest(), 0., MPI_COMM_WORLD);
  Parameters p;
  p.arterial_murray_exponent = 2.7;
  p.reference_flow           = 2.e-9;
  p.dissipation_weight       = .3;
  const auto s               = evaluate(n, p);
  oracle(n, s, p);
  const std::uint64_t count = s.terminal_flows.size();
  EXPECT_EQ(Utilities::MPI::sum(count, MPI_COMM_WORLD), 6u);
  EXPECT_EQ(s.upward_levels, 3u);
  EXPECT_EQ(s.downward_levels, 3u);
  for (const auto &[id, v] : n.vessels)
    {
      const unsigned int tree   = id >= base + 100 ? 1 : 0;
      const double       radius = .006 + .001 * tree;
      EXPECT_EQ(v.radii,
                (std::array<double, 2>{
                  {id == base + 100 * tree + 10 ? .009 : radius, radius}}));
    }
}
TEST(VascularForest,
     BOTH_ArteriesAndVeinsUseIndependentExponentsAndTheSameTotalSupply)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  const auto n =
    redistribute(bg, MappingQ1<3>(), forest(true), 0., MPI_COMM_WORLD);
  Parameters p;
  p.arterial_murray_exponent = 2.7;
  p.venous.murray_exponent   = 3.2;
  p.reference_flow           = 2.e-9;
  const auto s               = evaluate(n, p);
  oracle(n, s, p, true);
}
TEST(VascularForest,
     BOTH_CommonDemandScaleAndNumericalSubdivisionDoNotChangePhysics)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  auto       n = redistribute(bg, MappingQ1<3>(), forest(), 0., MPI_COMM_WORLD);
  Parameters p;
  p.reference_flow = 2.e-9;
  const auto a     = evaluate(n, p);
  for (auto &[id, t] : n.terminals)
    if (t.owner == rank())
      t.demand *= 1.e120;
  n            = redistribute(bg, MappingQ1<3>(), n, 0., MPI_COMM_WORLD);
  const auto b = evaluate(n, p);
  EXPECT_TRUE(a.validation.valid()) << issues(a);
  EXPECT_TRUE(b.validation.valid()) << issues(b);
  for (const auto &[id, branch] : a.owned)
    {
      EXPECT_NEAR(branch.radius / b.owned.at(id).radius, 1., 1.e-13);
      EXPECT_NEAR(branch.flow / b.owned.at(id).flow, 1., 1.e-13);
      auto physical                 = n;
      physical.vessels.at(id).radii = {{branch.radius, branch.radius}};
      const auto cells = subdivide(physical, id, {0., .02, .3, .7, 1.});
      double     v = 0., r = 0.;
      for (const auto &cell : cells)
        {
          v += volume(physical, id, cell.begin, cell.end);
          r += resistance(physical, id, cell.begin, cell.end);
        }
      EXPECT_NEAR(v / (numbers::PI * branch.radius * branch.radius *
                       length(n, id)),
                  1.,
                  1.e-13);
      EXPECT_NEAR(r / branch.resistance, 1., 1.e-13);
    }
}
TEST(VascularForest,
     BOTH_NoReferenceFlowLeavesSupplyUnspecifiedAndFrozenPrefixesUntouched)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  const auto n = redistribute(bg, MappingQ1<3>(), forest(), 0., MPI_COMM_WORLD);
  Parameters p;
  const auto s = evaluate(n, p);
  oracle(n, s, p);
  EXPECT_TRUE(s.terminal_flows.empty());
  EXPECT_DOUBLE_EQ(s.power[0], 0.);
  for (unsigned int i = 0; i < 2; ++i)
    EXPECT_EQ(s.owned.count(base + 100 * i + 10), 0u);
}
TEST(VascularForest, BOTH_ChangingOneLeafUpdatesAllRadiiInItsRootButNoOtherRoot)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  auto       n = redistribute(bg, MappingQ1<3>(), forest(), 0., MPI_COMM_WORLD);
  Parameters p;
  p.reference_flow  = 2.e-9;
  const auto before = evaluate(n, p);
  if (n.terminals.count(base + 30) && n.terminals.at(base + 30).owner == rank())
    n.terminals.at(base + 30).demand = 100.;
  n                = redistribute(bg, MappingQ1<3>(), n, 0., MPI_COMM_WORLD);
  const auto after = evaluate(n, p);
  EXPECT_TRUE(before.validation.valid()) << issues(before);
  EXPECT_TRUE(after.validation.valid()) << issues(after);
  for (const auto &[id, b] : before.owned)
    {
      const auto &a = after.owned.at(id);
      if (id >= base + 100)
        EXPECT_NEAR(a.radius / b.radius, 1., 1.e-13);
      else if (id == base + 11)
        EXPECT_DOUBLE_EQ(a.radius, b.radius);
      else if (id == base + 13)
        EXPECT_GT(a.radius, b.radius);
      else
        EXPECT_LT(a.radius, b.radius);
    }
  p.minimum_radius        = .005;
  const auto incompatible = evaluate(n, p);
  EXPECT_FALSE(incompatible.validation.valid());
}
TEST(VascularForest, BOTH_EmptyRanksAndAnEmptyMutableSetParticipate)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  const auto n = redistribute(bg, MappingQ1<3>(), forest(), 0., MPI_COMM_WORLD);
  Parameters p;
  const auto empty = evaluate_forest(n, {}, {}, p, MPI_COMM_WORLD);
  EXPECT_TRUE(empty.validation.valid());
  EXPECT_TRUE(empty.owned.empty());
  auto ids     = mutable_ids(n);
  auto control = roots(n);
  for (auto it = ids.begin(); it != ids.end();)
    if (*it >= base + 100)
      it = ids.erase(it);
    else
      ++it;
  control.erase(base + 111);
  const auto one = evaluate_forest(n, ids, control, p, MPI_COMM_WORLD);
  EXPECT_TRUE(one.validation.valid()) << issues(one);
  std::uint64_t count = one.owned.size();
  EXPECT_EQ(Utilities::MPI::sum(count, MPI_COMM_WORLD), 5u);
}
TEST(VascularForest,
     BOTH_InvalidRootMarkersAndMissingLeafRolesAreRejectedCollectively)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  auto       n = redistribute(bg, MappingQ1<3>(), forest(), 0., MPI_COMM_WORLD);
  Parameters p;
  auto       controls = roots(n);
  if (n.vessels.count(base + 13) && n.vessels.at(base + 13).owner == rank())
    controls[base + 13] = .006;
  EXPECT_FALSE(evaluate_forest(n, mutable_ids(n), controls, p, MPI_COMM_WORLD)
                 .validation.valid());
  EXPECT_FALSE(evaluate_forest(n, mutable_ids(n), {}, p, MPI_COMM_WORLD)
                 .validation.valid());
  controls = roots(n);
  for (auto &[id, radius] : controls)
    radius = .009;
  EXPECT_FALSE(evaluate_forest(n, mutable_ids(n), controls, p, MPI_COMM_WORLD)
                 .validation.valid());
  if (size() > 1)
    {
      if (rank() == 0)
        p.arterial_murray_exponent = 2.9;
      EXPECT_FALSE(evaluate(n, p).validation.valid());
      p.arterial_murray_exponent = 3.;
    }
  if (n.terminals.count(base + 30) && n.terminals.at(base + 30).owner == rank())
    n.terminals.erase(base + 30);
  n = redistribute(bg, MappingQ1<3>(), n, 0., MPI_COMM_WORLD);
  EXPECT_FALSE(evaluate(n, p).validation.valid());
}
TEST(VascularForest,
     BOTH_ConcurrentAncestorAndDescendantSplitsAggregateTheNewJointDemand)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  auto n = redistribute(bg, MappingQ1<3>(), forest(), 0., MPI_COMM_WORLD);
  for (unsigned int split = 0; split < 2; ++split)
    {
      const Id id = base + (split == 0 ? 12 : 14);
      if (!n.vessels.count(id) || n.vessels.at(id).owner != rank())
        continue;
      const auto old      = n.vessels.at(id);
      const Id   junction = base + 50 + 2 * split, tip = junction + 1,
               continuation          = base + 70 + 2 * split;
      const Point<3> middle          = .5 * (n.nodes.at(old.nodes[0]).position +
                                    n.nodes.at(old.nodes[1]).position);
      n.nodes[junction].position     = middle;
      n.nodes[junction].owner        = rank();
      n.nodes[tip].position          = middle + Point<3>(0., 0., .2);
      n.nodes[tip].owner             = rank();
      n.vessels.at(id).nodes[1]      = junction;
      auto c                         = old;
      c.nodes                        = {{junction, old.nodes[1]}};
      n.vessels[continuation]        = c;
      auto leaf                      = old;
      leaf.nodes                     = {{junction, tip}};
      n.vessels[continuation + 1]    = leaf;
      n.terminals[base + 90 + split] = {
        tip, base + 95 + split, System::arterial, rank(), 2. + 2 * split};
    }
  n        = redistribute(bg, MappingQ1<3>(), n, 0., MPI_COMM_WORLD);
  auto ids = mutable_ids(n);
  for (Id id : {base + 70, base + 71, base + 72, base + 73})
    if (n.vessels.count(id) && n.vessels.at(id).owner == rank())
      ids.insert(id);
  Parameters p;
  p.reference_flow           = 2.e-9;
  p.arterial_murray_exponent = 2.7;
  const auto s = evaluate_forest(n, ids, roots(n), p, MPI_COMM_WORLD);
  EXPECT_TRUE(s.validation.valid()) << issues(s);
  const std::map<Id, double> expected = {{base + 11, 15.},
                                         {base + 12, 14.},
                                         {base + 13, 1.},
                                         {base + 14, 7.},
                                         {base + 15, 5.},
                                         {base + 70, 12.},
                                         {base + 71, 2.},
                                         {base + 72, 3.},
                                         {base + 73, 4.}};
  for (const auto &[id, d] : expected)
    if (s.owned.count(id))
      {
        const auto &b = s.owned.at(id);
        EXPECT_NEAR(b.demand * s.demand_scale / d, 1., 1.e-12);
        EXPECT_NEAR(b.radius / (.006 * std::pow(d / 15., 1. / 2.7)),
                    1.,
                    1.e-12);
        EXPECT_NEAR(b.flow / (p.reference_flow * d / 27.), 1., 1.e-12);
        if (id == base + 14)
          {
            EXPECT_EQ(b.parent, base + 70);
          }
        else if (id == base + 72)
          {
            EXPECT_EQ(b.parent, base + 14);
          }
      }
  EXPECT_EQ(s.upward_levels, 5u);
  EXPECT_EQ(s.downward_levels, 5u);
  std::uint64_t count = s.owned.size();
  EXPECT_EQ(Utilities::MPI::sum(count, MPI_COMM_WORLD), 14u);
}
TEST(VascularForest, BOTH_AConnectedRootDoesNotHideACyclicMutableComponent)
{
  NetworkData n;
  if (rank() == size() - 1)
    {
      const Point<3> points[8] = {{.2, .2, .5},
                                  {.4, .2, .5},
                                  {1., .5, .5},
                                  {2., .5, .5},
                                  {1.5, 1.5, .5},
                                  {1., .5, .9},
                                  {2., .5, .9},
                                  {1.5, 1.5, .9}};
      for (Id i = 0; i < 8; ++i)
        {
          n.nodes[base + i].position = points[i];
          n.nodes[base + i].owner    = rank();
        }
      n.nodes[base].port               = Port::arterial_inlet;
      const std::array<Id, 2> edges[7] = {
        {{0, 1}}, {{2, 3}}, {{3, 4}}, {{4, 2}}, {{2, 5}}, {{3, 6}}, {{4, 7}}};
      for (Id i = 0; i < 7; ++i)
        {
          Vessel v;
          v.nodes                  = {{base + edges[i][0], base + edges[i][1]}};
          v.owner                  = rank();
          v.radii                  = {{.006, .006}};
          v.viscosity              = .004;
          n.vessels[base + 20 + i] = v;
        }
      for (Id i : {Id{1}, Id{5}, Id{6}, Id{7}})
        n.terminals[base + 40 + i] = {
          base + i, base + 50 + i, System::arterial, rank(), 1.};
    }
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  n = redistribute(bg, MappingQ1<3>(), n, 0., MPI_COMM_WORLD);
  std::set<Id> ids;
  for (const auto &[id, v] : n.vessels)
    if (v.owner == rank())
      ids.insert(id);
  std::map<Id, double> r;
  if (n.vessels.count(base + 20) && n.vessels.at(base + 20).owner == rank())
    r[base + 20] = .006;
  Parameters p;
  const auto s = evaluate_forest(n, ids, r, p, MPI_COMM_WORLD);
  EXPECT_FALSE(s.validation.valid());
  EXPECT_LE(s.upward_levels, 2u);
}

namespace
{
  std::vector<GrowthSplit>
  concurrent_splits(const NetworkData &n)
  {
    std::map<unsigned int, std::vector<GrowthSplit>> send;
    for (unsigned int i = 0; i < 2; ++i)
      {
        const Id id = base + (i == 0 ? 12 : 14);
        if (!n.vessels.count(id) || n.vessels.at(id).owner != rank())
          continue;
        const auto &v = n.vessels.at(id);
        GrowthSplit s;
        s.vessel     = id;
        s.owner      = rank();
        s.old_nodes  = v.nodes;
        s.old_points = {
          {n.nodes.at(v.nodes[0]).position, n.nodes.at(v.nodes[1]).position}};
        s.junction          = base + 50 + 2 * i;
        s.tip               = s.junction + 1;
        s.junction_position = .5 * (s.old_points[0] + s.old_points[1]);
        s.tip_position      = s.junction_position + Point<3>(0., 0., .2);
        s.continuation      = base + 70 + 2 * i;
        s.leaf              = s.continuation + 1;
        s.terminal          = base + 90 + i;
        s.territory         = base + 95 + i;
        s.demand            = 2. + 2 * i;
        // The producer and writer are different processes in MPI. Only these
        // two compact proposals move, rather than the physical forest.
        send[(rank() + 1) % size()].push_back(s);
      }
    const auto received = Utilities::MPI::some_to_some(MPI_COMM_WORLD, send);
    std::vector<GrowthSplit> result;
    for (const auto &[owner, values] : received)
      result.insert(result.end(), values.begin(), values.end());
    return result;
  }
} // namespace
TEST(VascularForest,
     BOTH_CommitJointSplitsOnlyAfterOwnerPhysiologyAndGeometryChecks)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  const auto n = redistribute(bg, MappingQ1<3>(), forest(), 0., MPI_COMM_WORLD);
  Parameters p;
  p.reference_flow    = 2.e-9;
  p.minimum_angle     = 1.;
  const auto proposed = concurrent_splits(n);
  const auto result   = commit_growth_splits(bg,
                                           MappingQ1<3>(),
                                           n,
                                           mutable_ids(n),
                                           roots(n),
                                           proposed,
                                           p,
                                           MPI_COMM_WORLD);
  EXPECT_TRUE(result.committed);
  EXPECT_TRUE(result.validation.valid());
  for (const auto &i : result.validation.issues)
    ADD_FAILURE() << i.message;
  const std::uint64_t count = result.mutable_owned.size();
  EXPECT_EQ(Utilities::MPI::sum(count, MPI_COMM_WORLD), 14u);
  EXPECT_EQ(result.physiology.upward_levels, 5u);
  for (const auto &[id, b] : result.physiology.owned)
    {
      const auto &v = result.network.vessels.at(id);
      EXPECT_EQ(v.owner, rank());
      EXPECT_EQ(v.radii, (std::array<double, 2>{{b.radius, b.radius}}));
      const double root_demand = id >= base + 100 ? 12. : 15.;
      EXPECT_NEAR(b.root_demand * result.physiology.demand_scale,
                  root_demand,
                  1.e-13);
      EXPECT_NEAR(b.flow / (p.reference_flow * b.demand *
                            result.physiology.demand_scale / 27.),
                  1.,
                  1.e-12);
      EXPECT_EQ(v.physical_properties.at("density"),
                id >= base + 100 ? 1051. : 1050.);
      if (id == base + 14)
        {
          EXPECT_EQ(b.parent, base + 70);
        }
    }
  for (unsigned int tree = 0; tree < 2; ++tree)
    {
      const Id id = base + 100 * tree + 10;
      if (result.network.vessels.count(id))
        {
          EXPECT_EQ(result.network.vessels.at(id).nodes,
                    (std::array<Id, 2>{
                      {base + 100 * tree, base + 100 * tree + 1}}));
          EXPECT_EQ(result.network.vessels.at(id).radii,
                    (std::array<double, 2>{{.009, .006 + .001 * tree}}));
        }
    }
  // Neither proposal staging nor the successful return updates the input.
  if (n.vessels.count(base + 12))
    {
      EXPECT_EQ(n.vessels.at(base + 12).nodes[1], base + 3);
    }
  EXPECT_EQ(n.nodes.count(base + 50), 0u);
}
TEST(VascularForest, BOTH_AFailedFinalGeometryOrRadiusCheckRollsBackEveryRank)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  const auto n = redistribute(bg, MappingQ1<3>(), forest(), 0., MPI_COMM_WORLD);
  Parameters p;
  p.minimum_angle = 1.;
  auto proposed   = concurrent_splits(n);
  for (auto &s : proposed)
    if (s.vessel == base + 14)
      s.tip_position[2] = -.1;
  const auto outside = commit_growth_splits(bg,
                                            MappingQ1<3>(),
                                            n,
                                            mutable_ids(n),
                                            roots(n),
                                            proposed,
                                            p,
                                            MPI_COMM_WORLD);
  EXPECT_FALSE(outside.committed);
  EXPECT_FALSE(outside.validation.valid());
  EXPECT_TRUE(outside.network.vessels.empty());
  p.minimum_radius = .0028;
  // The old side leaf is admissible before the two insertions, but falls
  // below this radius when their combined demand is normalized at the root.
  EXPECT_TRUE(evaluate(n, p).validation.valid());
  const auto thin = commit_growth_splits(bg,
                                         MappingQ1<3>(),
                                         n,
                                         mutable_ids(n),
                                         roots(n),
                                         concurrent_splits(n),
                                         p,
                                         MPI_COMM_WORLD);
  EXPECT_FALSE(thin.committed);
  EXPECT_FALSE(thin.validation.valid());
  EXPECT_TRUE(thin.network.vessels.empty());
  EXPECT_EQ(n.nodes.count(base + 50), 0u);
}
TEST(VascularForest, BOTH_StaleDuplicateAndFrozenWritesNeverChangeTheSnapshot)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  const auto n = redistribute(bg, MappingQ1<3>(), forest(), 0., MPI_COMM_WORLD);
  Parameters p;
  if (size() > 1)
    {
      if (rank() == 0)
        p.clearance = .001;
      EXPECT_FALSE(commit_growth_splits(bg,
                                        MappingQ1<3>(),
                                        n,
                                        mutable_ids(n),
                                        roots(n),
                                        concurrent_splits(n),
                                        p,
                                        MPI_COMM_WORLD)
                     .committed);
      p.clearance = 0.;
    }
  auto proposed = concurrent_splits(n);
  for (auto &s : proposed)
    if (s.vessel == base + 12)
      s.old_points[0][0] += .01;
  EXPECT_FALSE(commit_growth_splits(bg,
                                    MappingQ1<3>(),
                                    n,
                                    mutable_ids(n),
                                    roots(n),
                                    proposed,
                                    p,
                                    MPI_COMM_WORLD)
                 .committed);
  proposed = concurrent_splits(n);
  if (!proposed.empty())
    proposed.push_back(proposed.front());
  EXPECT_FALSE(commit_growth_splits(bg,
                                    MappingQ1<3>(),
                                    n,
                                    mutable_ids(n),
                                    roots(n),
                                    proposed,
                                    p,
                                    MPI_COMM_WORLD)
                 .committed);
  proposed = concurrent_splits(n);
  // Remove the writer's permission independently of its provenance.
  auto ids = mutable_ids(n);
  ids.erase(base + 12);
  EXPECT_FALSE(
    commit_growth_splits(
      bg, MappingQ1<3>(), n, ids, roots(n), proposed, p, MPI_COMM_WORLD)
      .committed);
  EXPECT_EQ(n.nodes.count(base + 50), 0u);
}

TEST(VascularForest, BOTH_FixedRootControlsFollowTheMigratingPhysicalRoot)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  const auto n = redistribute(bg, MappingQ1<3>(), forest(), 0., MPI_COMM_WORLD);
  std::vector<GrowthSplit> proposed;
  unsigned int             previous_owner = size();
  if (n.vessels.count(base + 11) && n.vessels.at(base + 11).owner == rank())
    {
      previous_owner = rank();
      const auto &v  = n.vessels.at(base + 11);
      GrowthSplit s;
      s.vessel     = base + 11;
      s.owner      = rank();
      s.old_nodes  = v.nodes;
      s.old_points = {
        {n.nodes.at(v.nodes[0]).position, n.nodes.at(v.nodes[1]).position}};
      s.junction          = base + 50;
      s.tip               = base + 51;
      s.junction_position = Point<3>(3.9, .1, 1.5);
      s.tip_position      = Point<3>(3.9, .3, 1.5);
      s.continuation      = base + 70;
      s.leaf              = base + 71;
      s.terminal          = base + 90;
      s.territory         = base + 95;
      s.demand            = 2.;
      proposed.push_back(s);
    }
  previous_owner = Utilities::MPI::min(previous_owner, MPI_COMM_WORLD);
  Parameters p;
  p.minimum_angle   = 1.;
  const auto result = commit_growth_splits(bg,
                                           MappingQ1<3>(),
                                           n,
                                           mutable_ids(n),
                                           roots(n),
                                           proposed,
                                           p,
                                           MPI_COMM_WORLD);
  EXPECT_TRUE(result.committed);
  for (const auto &i : result.validation.issues)
    ADD_FAILURE() << i.message;
  unsigned int new_owner = size();
  if (result.fixed_owned_roots.count(base + 11))
    {
      new_owner = rank();
      EXPECT_DOUBLE_EQ(result.fixed_owned_roots.at(base + 11), .006);
      EXPECT_DOUBLE_EQ(result.network.vessels.at(base + 11).radii[0], .006);
      EXPECT_EQ(result.physiology.owned.at(base + 11).root, base + 11);
      EXPECT_EQ(result.physiology.owned.at(base + 11).terminals, 4u);
    }
  new_owner = Utilities::MPI::min(new_owner, MPI_COMM_WORLD);
  EXPECT_LT(new_owner, size());
  if (size() > 1)
    {
      EXPECT_NE(new_owner, previous_owner);
    }
}

TEST(VascularForest,
     BOTH_RemoteAncestralPathsOptimizeAndCommitFromLocalBackground)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  const auto n = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
  Parameters p;
  p.minimum_angle      = 1.;
  p.reference_flow     = 2.e-9;
  p.dissipation_weight = .3;
  const auto     state = evaluate(n, p);
  const Point<3> terminal(3.15, .85, .95);
  unsigned int   producer = geometry.containing(terminal) ? rank() : size();
  producer                = Utilities::MPI::min(producer, MPI_COMM_WORLD);
  std::map<Id, unsigned int> requests;
  for (Id id : {base + 14, base + 15})
    {
      unsigned int owner =
        n.vessels.count(id) && n.vessels.at(id).owner == rank() ? rank() :
                                                                  size();
      owner = Utilities::MPI::min(owner, MPI_COMM_WORLD);
      if (rank() == producer)
        requests[id] = owner;
    }
  const auto paths = fetch_growth_candidates(
    n, state, requests, base + 50, base + 70, p, MPI_COMM_WORLD);
  EXPECT_TRUE(paths.validation.valid());
  for (const auto &i : paths.validation.issues)
    ADD_FAILURE() << i.message;
  EXPECT_EQ(paths.retained_branches, rank() == producer ? 4u : 0u);
  EXPECT_EQ(paths.rounds, 3u);
  std::vector<GrowthSplit>       splits;
  std::optional<GrowthCandidate> candidate;
  if (rank() == producer && paths.contexts.count(base + 14))
    {
      const auto &x = paths.contexts.at(base + 14);
      EXPECT_EQ(x.path.size(), 3u);
      EXPECT_EQ(x.path.front().edge.id, base + 14);
      EXPECT_EQ(x.path.back().edge.id, base + 11);
      EXPECT_NEAR(x.minimum_demand * x.demand_scale, 1., 1.e-12);
      const auto cheap_geometry = [&](const std::vector<Edge> &changed) {
        for (std::size_t i = 0; i < changed.size(); ++i)
          {
            std::array<Point<3>, 2> normals;
            for (const auto &branch : x.path)
              for (unsigned int end = 0; end < 2; ++end)
                for (unsigned int other = 0; other < 2; ++other)
                  if (changed[i].nodes[end] == branch.edge.nodes[other])
                    normals[end] = branch.port_normals[other];
            if (!geometry.boundary_separated({changed[i].points[0],
                                              changed[i].points[1]},
                                             changed[i].radius + p.clearance,
                                             p.geometry_depth,
                                             p.geometry_work,
                                             normals))
              return false;
            for (std::size_t j = 0; j < i; ++j)
              if (!compatible(
                    changed[i], changed[j], p.minimum_angle, p.clearance))
                return false;
          }
        return true;
      };
      candidate = optimize_growth(x, p, terminal, 2., cheap_geometry);
      EXPECT_TRUE(candidate.has_value());
      if (candidate)
        {
          const auto &selected = x.path.front();
          GrowthSplit s;
          s.vessel            = candidate->vessel;
          s.owner             = selected.owner;
          s.old_nodes         = selected.edge.nodes;
          s.old_points        = selected.edge.points;
          s.junction          = base + 50;
          s.tip               = base + 51;
          s.junction_position = candidate->junction;
          s.tip_position      = candidate->terminal;
          s.continuation      = base + 70;
          s.leaf              = base + 71;
          s.terminal          = base + 90;
          s.territory         = base + 95;
          s.demand            = candidate->terminal_demand;
          splits.push_back(s);
        }
    }
  const auto result = commit_growth_splits(
    bg, mapping, n, mutable_ids(n), roots(n), splits, p, MPI_COMM_WORLD);
  EXPECT_TRUE(result.committed);
  for (const auto &i : result.validation.issues)
    ADD_FAILURE() << i.message;
  const std::uint64_t count = result.mutable_owned.size();
  EXPECT_EQ(Utilities::MPI::sum(count, MPI_COMM_WORLD), 12u);
  if (candidate && result.committed)
    {
      // Independent final physical volume and Q^2 R sums include the other
      // root's changed flow share, although its topology was never fetched.
      const double expected =
        (1. - p.dissipation_weight) * result.physiology.volume[0] /
          state.volume[0] +
        p.dissipation_weight * result.physiology.power[0] / state.power[0];
      EXPECT_NEAR(candidate->cost, expected, 1.e-12);
      for (const auto &e : candidate->changed)
        if (result.network.vessels.count(e.id))
          {
            EXPECT_NEAR(result.network.vessels.at(e.id).radii[0],
                        e.radius,
                        1.e-14);
          }
    }
  EXPECT_EQ(n.nodes.count(base + 50), 0u);
}
TEST(VascularForest, BOTH_InvalidOwnerAndCyclicPathMetadataFailCollectively)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  const auto n = redistribute(bg, MappingQ1<3>(), forest(), 0., MPI_COMM_WORLD);
  Parameters p;
  auto       state = evaluate(n, p);
  std::map<Id, unsigned int> requests;
  if (rank() == size() - 1)
    requests[base + 999] = 0;
  EXPECT_FALSE(fetch_growth_candidates(
                 n, state, requests, base + 50, base + 70, p, MPI_COMM_WORLD)
                 .validation.valid());
  unsigned int owner = state.owned.count(base + 14) ? rank() : size();
  owner              = Utilities::MPI::min(owner, MPI_COMM_WORLD);
  requests.clear();
  if (rank() == size() - 1)
    requests[base + 14] = owner;
  if (size() > 1)
    {
      auto wrong = requests;
      if (!wrong.empty())
        wrong.begin()->second = (owner + 1) % size();
      EXPECT_FALSE(fetch_growth_candidates(
                     n, state, wrong, base + 50, base + 70, p, MPI_COMM_WORLD)
                     .validation.valid());
    }
  if (state.owned.count(base + 14))
    {
      state.owned.at(base + 14).parent       = base + 14;
      state.owned.at(base + 14).parent_owner = rank();
    }
  const auto cyclic = fetch_growth_candidates(
    n, state, requests, base + 50, base + 70, p, MPI_COMM_WORLD);
  EXPECT_FALSE(cyclic.validation.valid());
  EXPECT_TRUE(cyclic.contexts.empty());
  requests.clear();
  const auto empty = fetch_growth_candidates(
    n, state, requests, base + 50, base + 70, p, MPI_COMM_WORLD);
  EXPECT_TRUE(empty.validation.valid());
  EXPECT_TRUE(empty.contexts.empty());
  EXPECT_EQ(empty.retained_branches, 0u);
}

namespace
{
  NetworkData
  single_seed()
  {
    NetworkData n;
    if (rank() == size() - 1)
      {
        n.nodes[base]     = {Point<3>(.2, 1., 1.),
                             rank(),
                             Port::arterial_inlet,
                             Provenance::generated};
        n.nodes[base + 1] = {Point<3>(.7, 1., 1.),
                             rank(),
                             Port::none,
                             Provenance::generated};
        Vessel v;
        v.nodes                = {{base, base + 1}};
        v.radii                = {{.006, .006}};
        v.viscosity            = .004;
        v.provenance           = Provenance::generated;
        v.owner                = rank();
        n.vessels[base + 10]   = v;
        n.terminals[base + 30] = {
          base + 1, base + 40, System::arterial, rank(), 1.};
      }
    return n;
  }
  std::set<Id>
  seed_ids(const NetworkData &n)
  {
    std::set<Id> result;
    if (n.vessels.count(base + 10) && n.vessels.at(base + 10).owner == rank())
      result.insert(base + 10);
    return result;
  }
  std::map<Id, double>
  seed_roots(const NetworkData &n)
  {
    std::map<Id, double> result;
    for (Id id : seed_ids(n))
      result[id] = .006;
    return result;
  }
} // namespace
TEST(VascularForest,
     BOTH_BatchedGrowthPropagatesFromOneRealSeedAndRepeatsExactly)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  const auto   n = redistribute(bg, mapping, single_seed(), 0., MPI_COMM_WORLD);
  Parameters   p;
  p.seed_length           = .2;
  p.terminal_count        = 12;
  p.sample_budget         = 100;
  p.candidate_count       = 4;
  p.optimization_steps    = 5;
  p.reference_flow        = 2.e-9;
  p.terminal_weight       = "1+x";
  p.terminal_weight_bound = 5.;
  p.tissue_demand         = "2+y";
  const auto run =
    grow_forest(bg, mapping, n, seed_ids(n), seed_roots(n), p, MPI_COMM_WORLD);
  EXPECT_TRUE(run.validation.valid());
  for (const auto &i : run.validation.issues)
    ADD_FAILURE() << i.message;
  EXPECT_TRUE(run.accepted.committed);
  EXPECT_EQ(Utilities::MPI::sum(run.accepted_insertions, MPI_COMM_WORLD), 11u);
  const unsigned int active   = run.accepted_insertions > 0 ? 1 : 0;
  const auto active_producers = Utilities::MPI::sum(active, MPI_COMM_WORLD);
  if (size() > 1)
    {
      EXPECT_GT(active_producers, 1u);
      EXPECT_GT(run.maximum_batch_insertions, 1u);
    }
  for (const auto &[id, t] : run.accepted.network.terminals)
    if (t.owner == rank() && id != base + 30)
      {
        EXPECT_DOUBLE_EQ(t.demand,
                         2. +
                           run.accepted.network.nodes.at(t.node).position[1]);
      }
  std::uint64_t leaves = 0, branches = run.accepted.mutable_owned.size(),
                roots_count = run.accepted.fixed_owned_roots.size();
  double flow               = 0.;
  for (const auto &[id, b] : run.accepted.physiology.owned)
    {
      if (b.children.empty())
        ++leaves;
      EXPECT_EQ(b.root, base + 10);
      const auto &v = run.accepted.network.vessels.at(id);
      EXPECT_EQ(v.owner, rank());
      EXPECT_NEAR(v.radii[0] /
                    (.006 * std::pow(b.demand / b.root_demand, 1. / 3.)),
                  1.,
                  1.e-12);
      EXPECT_DOUBLE_EQ(v.viscosity, .004);
    }
  for (const auto &[id, q] : run.accepted.physiology.terminal_flows)
    flow += q;
  EXPECT_EQ(Utilities::MPI::sum(leaves, MPI_COMM_WORLD), 12u);
  EXPECT_EQ(Utilities::MPI::sum(branches, MPI_COMM_WORLD), 23u);
  EXPECT_EQ(Utilities::MPI::sum(roots_count, MPI_COMM_WORLD), 1u);
  EXPECT_NEAR(Utilities::MPI::sum(flow, MPI_COMM_WORLD) / p.reference_flow,
              1.,
              1.e-12);
  const auto repeated =
    grow_forest(bg, mapping, n, seed_ids(n), seed_roots(n), p, MPI_COMM_WORLD);
  EXPECT_TRUE(repeated.validation.valid());
  EXPECT_EQ(run.batches, repeated.batches);
  EXPECT_EQ(run.sampled_points, repeated.sampled_points);
  EXPECT_EQ(run.accepted.network.nodes.size(),
            repeated.accepted.network.nodes.size());
  for (const auto &[id, node] : run.accepted.network.nodes)
    {
      const auto it = repeated.accepted.network.nodes.find(id);
      EXPECT_NE(it, repeated.accepted.network.nodes.end());
      if (it != repeated.accepted.network.nodes.end())
        {
          EXPECT_EQ(node.position, it->second.position);
          EXPECT_EQ(node.owner, it->second.owner);
        }
    }
  if (n.nodes.count(base + 1))
    {
      EXPECT_EQ(n.nodes.at(base + 1).position, Point<3>(.7, 1., 1.));
    }
}
TEST(VascularForest,
     BOTH_BatchedGrowthIncludesEmptyRanksAndReportsFiniteFailure)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg, 0., 4.);
  MappingQ1<3> mapping;
  const auto   n = redistribute(bg, mapping, single_seed(), 0., MPI_COMM_WORLD);
  Parameters   p;
  p.seed_length        = .2;
  p.terminal_count     = 4;
  p.sample_budget      = 100;
  p.candidate_count    = 4;
  p.optimization_steps = 3;
  const auto run =
    grow_forest(bg, mapping, n, seed_ids(n), seed_roots(n), p, MPI_COMM_WORLD);
  EXPECT_TRUE(run.validation.valid());
  for (const auto &i : run.validation.issues)
    ADD_FAILURE() << i.message;
  EXPECT_EQ(Utilities::MPI::sum(run.accepted_insertions, MPI_COMM_WORLD), 3u);
  if (bg.n_locally_owned_active_cells() == 0)
    {
      EXPECT_EQ(run.sampled_points, 0u);
      EXPECT_EQ(run.proposed_insertions, 0u);
    }
  p.terminal_weight = "0";
  p.sample_budget   = 3;
  const auto failed =
    grow_forest(bg, mapping, n, seed_ids(n), seed_roots(n), p, MPI_COMM_WORLD);
  EXPECT_FALSE(failed.validation.valid());
  EXPECT_TRUE(failed.accepted.committed);
  EXPECT_EQ(failed.batches, 3u);
  EXPECT_EQ(failed.accepted_insertions, 0u);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<std::uint64_t>(
                                  failed.accepted.mutable_owned.size()),
                                MPI_COMM_WORLD),
            1u);
  p.minimum_radius = .006;
  const auto incompatible =
    grow_forest(bg, mapping, n, seed_ids(n), seed_roots(n), p, MPI_COMM_WORLD);
  EXPECT_FALSE(incompatible.validation.valid());
  EXPECT_EQ(incompatible.batches, 0u);
  EXPECT_EQ(incompatible.sampled_points, 0u);
  p.minimum_radius = 0.;
  if (size() > 1)
    {
      if (rank() == 0)
        ++p.terminal_count;
      EXPECT_FALSE(
        grow_forest(
          bg, mapping, n, seed_ids(n), seed_roots(n), p, MPI_COMM_WORLD)
          .validation.valid());
    }
}

TEST(VascularForest, BOTH_PreliminaryVotesCannotAdoptAnAncestorRadiusCollision)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  auto         input      = forest();
  const double old_radius = .006 * std::pow(8. / 9., 1. / 3.);
  if (rank() == size() - 1)
    {
      const double y          = .5 + old_radius + .001 + .0001;
      input.nodes[base + 200] = {Point<3>(1.4, y, .5), rank()};
      input.nodes[base + 201] = {Point<3>(1.9, y, .5), rank()};
      Vessel v;
      v.nodes                   = {{base + 200, base + 201}};
      v.radii                   = {{.001, .001}};
      v.viscosity               = .004;
      v.owner                   = rank();
      input.vessels[base + 210] = v;
    }
  const auto raw = redistribute(bg, mapping, input, 0., MPI_COMM_WORLD);
  Parameters p;
  p.minimum_angle     = 1.;
  const auto baseline = commit_growth_splits(
    bg, mapping, raw, mutable_ids(raw), roots(raw), {}, p, MPI_COMM_WORLD);
  EXPECT_TRUE(baseline.committed);
  for (const auto &i : baseline.validation.issues)
    ADD_FAILURE() << i.message;
  const auto                &n = baseline.committed ? baseline.network : raw;
  auto                       candidates = concurrent_splits(n);
  std::vector<GrowthSplit>   splits;
  std::vector<BatchProposal> proposals;
  for (auto s : candidates)
    if (s.vessel == base + 14)
      {
        s.demand = 100.;
        splits.push_back(s);
        BatchProposal proposal;
        proposal.id                  = 0;
        proposal.owners[s.vessel]    = s.owner;
        proposal.old_nodes[s.vessel] = s.old_nodes;
        proposal.changed             = {{s.vessel,
                                         {{s.old_nodes[0], s.junction}},
                                         {{s.old_points[0], s.junction_position}},
                                         .006 * std::pow(103. / 109., 1. / 3.)},
                                        {s.continuation,
                                         {{s.junction, s.old_nodes[1]}},
                                         {{s.junction_position, s.old_points[1]}},
                                         .006 * std::pow(3. / 109., 1. / 3.)},
                                        {s.leaf,
                                         {{s.junction, s.tip}},
                                         {{s.junction_position, s.tip_position}},
                                         .006 * std::pow(100. / 109., 1. / 3.)}};
        proposals.push_back(proposal);
      }
  BackgroundGeometry geometry(bg, mapping);
  const auto         preliminary = select_growth_batch(
    geometry,
    n,
    proposals,
    p,
    MPI_COMM_WORLD,
    [&](Id id) { return n.vessels.at(id).radii[0]; },
    false);
  for (const auto &[id, selected] : preliminary.local)
    EXPECT_TRUE(selected);
  const auto result = commit_growth_splits(
    bg, mapping, n, mutable_ids(n), roots(n), splits, p, MPI_COMM_WORLD);
  EXPECT_FALSE(result.committed);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_TRUE(result.network.vessels.empty());
  // The unchanged parent grows into the frozen obstacle only after the joint
  // demand update; the three replacement rays alone do not expose this.
  if (n.vessels.count(base + 12))
    {
      EXPECT_NEAR(n.vessels.at(base + 12).radii[0], old_radius, 1.e-15);
    }
}

namespace
{
  NetworkData
  paired_seed()
  {
    NetworkData n;
    if (rank() != size() - 1)
      return n;
    const double x[4] = {.8, 1., 2.6, 3.4};
    // Four frozen arterial supply branches and one real venous outlet.
    for (unsigned int i = 0; i < 4; ++i)
      {
        const Id off = base + 100 * i;
        n.nodes[off] = {Point<3>(x[i], .2, .7), rank(), Port::arterial_inlet};
        n.nodes[off + 1] = {Point<3>(x[i], .7, .7), rank()};
        Vessel v;
        v.nodes               = {{off, off + 1}};
        v.radii               = {{.001, .001}};
        v.viscosity           = .004;
        v.owner               = rank();
        n.vessels[off + 10]   = v;
        n.terminals[off + 30] = {
          off + 1, off + 40, System::arterial, rank(), 1. + i};
      }
    n.nodes[base + 10000] = {Point<3>(.2, 1.5, .7),
                             rank(),
                             Port::venous_outlet,
                             Provenance::generated};
    n.nodes[base + 10001] = {Point<3>(.8, .74, .7),
                             rank(),
                             Port::none,
                             Provenance::generated};
    Vessel v;
    v.nodes                   = {{base + 10000, base + 10001}};
    v.radii                   = {{.003, .003}};
    v.viscosity               = .004;
    v.system                  = System::venous;
    v.provenance              = Provenance::generated;
    v.owner                   = rank();
    n.vessels[base + 10010]   = v;
    n.terminals[base + 10030] = {
      base + 10001, base + 40, System::venous, rank(), 1.};
    return n;
  }
  std::vector<Territory>
  common_territories(const NetworkData &n)
  {
    std::vector<Territory> result;
    for (const auto &[id, t] : n.terminals)
      if (t.owner == rank() && t.system == System::arterial)
        result.push_back(
          {t.territory, n.nodes.at(t.node).position, rank(), t.demand});
    return result;
  }
  std::set<Id>
  venous_seed_ids(const NetworkData &n)
  {
    std::set<Id> result;
    if (n.vessels.count(base + 10010) &&
        n.vessels.at(base + 10010).owner == rank())
      result.insert(base + 10010);
    return result;
  }
  std::map<Id, double>
  venous_seed_roots(const NetworkData &n)
  {
    std::map<Id, double> result;
    for (Id id : venous_seed_ids(n))
      result[id] = .003;
    return result;
  }
  void
  venous_controls(Parameters &p)
  {
    p.seed_length                     = .2;
    p.terminal_count                  = 4;
    p.sample_budget                   = 200;
    p.candidate_count                 = 4;
    p.optimization_steps              = 5;
    p.minimum_angle                   = 10.;
    p.venous.enabled                  = true;
    p.venous.murray_exponent          = 3.2;
    p.venous.terminal_separation      = .04;
    p.venous.microvascular_resistance = 1.e13;
    // The offset point must never reevaluate the arterial tissue demand.
    p.tissue_demand = "1000";
  }
} // namespace
TEST(VascularForest, BOTH_VenousGrowthClaimsCommonTerritoriesAcrossTheirOwners)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  const auto   n = redistribute(bg, mapping, paired_seed(), 0., MPI_COMM_WORLD);
  const auto   common = common_territories(n);
  Parameters   p;
  venous_controls(p);
  const auto run = grow_forest(bg,
                               mapping,
                               n,
                               venous_seed_ids(n),
                               venous_seed_roots(n),
                               p,
                               MPI_COMM_WORLD,
                               4,
                               System::venous,
                               common);
  EXPECT_TRUE(run.validation.valid());
  for (const auto &i : run.validation.issues)
    ADD_FAILURE() << i.message;
  EXPECT_EQ(Utilities::MPI::sum(run.accepted_insertions, MPI_COMM_WORLD), 3u);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<std::uint64_t>(
                                  run.owned_venous_terminals.size()),
                                MPI_COMM_WORLD),
            4u);
  EXPECT_TRUE(run.accepted.physiology.terminal_flows.empty());
  EXPECT_EQ(Utilities::MPI::sum(static_cast<std::uint64_t>(
                                  run.accepted.fixed_owned_roots.size()),
                                MPI_COMM_WORLD),
            1u);
  for (const auto &[id, branch] : run.accepted.physiology.owned)
    {
      EXPECT_EQ(branch.root, base + 10010);
      EXPECT_EQ(run.accepted.network.vessels.at(id).system, System::venous);
    }
  std::array<unsigned int, 4> owners;
  for (unsigned int i = 0; i < 4; ++i)
    {
      const Id     tid = base + 100 * i + 30;
      unsigned int owner =
        n.terminals.count(tid) && n.terminals.at(tid).owner == rank() ? rank() :
                                                                        size();
      owners[i] = Utilities::MPI::min(owner, MPI_COMM_WORLD);
    }
  using Received = std::pair<Id, std::pair<Id, std::pair<Point<3>, double>>>;
  std::map<unsigned int, std::vector<Received>> send;
  for (Id tid : run.accepted.physiology.owned_leaf_terminals)
    {
      const auto &t     = run.accepted.network.terminals.at(tid);
      const auto  index = (t.territory - base - 40) / 100;
      EXPECT_LT(index, 4u);
      if (index < 4)
        send[owners[index]].push_back(
          {t.territory,
           {tid, {run.accepted.network.nodes.at(t.node).position, t.demand}}});
    }
  const auto received = Utilities::MPI::some_to_some(MPI_COMM_WORLD, send);
  std::map<Id, Territory> centres;
  for (const auto &t : common)
    centres[t.id] = t;
  std::set<Id> served;
  for (const auto &[owner, entries] : received)
    for (const auto &[territory, terminal] : entries)
      {
        EXPECT_TRUE(served.insert(territory).second);
        EXPECT_TRUE(centres.count(territory));
        if (centres.count(territory))
          {
            EXPECT_EQ(run.owned_venous_terminals.at(territory), terminal.first);
            EXPECT_DOUBLE_EQ(terminal.second.second,
                             centres.at(territory).demand);
            EXPECT_NEAR(terminal.second.first.distance(
                          centres.at(territory).position) /
                          p.venous.terminal_separation,
                        1.,
                        1.e-12);
            if (territory == base + 140)
              {
                EXPECT_NEAR(terminal.second.first[0], 1.04, 1.e-14);
                if (size() == 4)
                  {
                    EXPECT_NE(owner, rank());
                  }
              }
          }
      }
  EXPECT_EQ(served.size(), common.size());
  for (const auto &[id, v] : n.vessels)
    if (v.owner == rank() && v.system == System::arterial)
      {
        EXPECT_TRUE(run.accepted.network.vessels.count(id));
        if (run.accepted.network.vessels.count(id))
          {
            const auto &after = run.accepted.network.vessels.at(id);
            EXPECT_EQ(after.nodes, v.nodes);
            EXPECT_EQ(after.radii, v.radii);
          }
      }
}
TEST(VascularForest,
     BOTH_InvalidCommonTerritoryClaimsFailWithoutPartialAdoption)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  const auto   n = redistribute(bg, mapping, paired_seed(), 0., MPI_COMM_WORLD);
  const auto   common = common_territories(n);
  Parameters   p;
  venous_controls(p);
  auto wrong = common;
  for (auto &t : wrong)
    if (t.id == base + 40)
      t.demand = 2.;
  EXPECT_FALSE(grow_forest(bg,
                           mapping,
                           n,
                           venous_seed_ids(n),
                           venous_seed_roots(n),
                           p,
                           MPI_COMM_WORLD,
                           4,
                           System::venous,
                           wrong)
                 .validation.valid());
  wrong = common;
  for (auto &t : wrong)
    if (t.id == base + 40)
      t.owner = size();
  EXPECT_FALSE(grow_forest(bg,
                           mapping,
                           n,
                           venous_seed_ids(n),
                           venous_seed_roots(n),
                           p,
                           MPI_COMM_WORLD,
                           4,
                           System::venous,
                           wrong)
                 .validation.valid());
  wrong = common;
  for (auto &t : wrong)
    if (t.id == base + 140)
      t.id = base + 40;
  EXPECT_FALSE(grow_forest(bg,
                           mapping,
                           n,
                           venous_seed_ids(n),
                           venous_seed_roots(n),
                           p,
                           MPI_COMM_WORLD,
                           4,
                           System::venous,
                           wrong)
                 .validation.valid());
  auto duplicate_input = paired_seed();
  if (rank() == size() - 1)
    {
      duplicate_input.nodes[base + 11000] = {Point<3>(.2, 1.8, 1.2),
                                             rank(),
                                             Port::venous_outlet,
                                             Provenance::generated};
      duplicate_input.nodes[base + 11001] = {Point<3>(.8, .7, .74),
                                             rank(),
                                             Port::none,
                                             Provenance::generated};
      Vessel v;
      v.nodes                                 = {{base + 11000, base + 11001}};
      v.radii                                 = {{.003, .003}};
      v.viscosity                             = .004;
      v.system                                = System::venous;
      v.provenance                            = Provenance::generated;
      v.owner                                 = rank();
      duplicate_input.vessels[base + 11010]   = v;
      duplicate_input.terminals[base + 11030] = {
        base + 11001, base + 40, System::venous, rank(), 1.};
    }
  const auto duplicate =
    redistribute(bg, mapping, duplicate_input, 0., MPI_COMM_WORLD);
  auto ids           = venous_seed_ids(duplicate);
  auto root_controls = venous_seed_roots(duplicate);
  if (duplicate.vessels.count(base + 11010) &&
      duplicate.vessels.at(base + 11010).owner == rank())
    {
      ids.insert(base + 11010);
      root_controls[base + 11010] = .003;
    }
  const auto double_claim = grow_forest(bg,
                                        mapping,
                                        duplicate,
                                        ids,
                                        root_controls,
                                        p,
                                        MPI_COMM_WORLD,
                                        4,
                                        System::venous,
                                        common_territories(duplicate));
  // Both roots are geometrically valid, but distinct endpoints cannot claim
  // the same tissue territory twice.
  EXPECT_TRUE(double_claim.accepted.committed);
  EXPECT_FALSE(double_claim.validation.valid());
  EXPECT_EQ(double_claim.batches, 0u);
  EXPECT_EQ(n.nodes.count(base + 10002), 0u);
}

TEST(VascularForest, BOTH_FullCommitWorkCounts)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  const auto   raw = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
  Parameters   p;
  p.minimum_angle = 1.;
  GrowthWork work;
  const auto result = commit_growth_splits(bg,
                                           mapping,
                                           raw,
                                           mutable_ids(raw),
                                           roots(raw),
                                           {},
                                           p,
                                           MPI_COMM_WORLD,
                                           nullptr,
                                           &work);
  ASSERT_TRUE(result.committed);
  const auto total = [&](GrowthOperation operation) {
    return Utilities::MPI::sum(work[operation], MPI_COMM_WORLD);
  };
  std::uint64_t owned = 0;
  for (const auto &[id, v] : raw.vessels)
    if (v.owner == rank())
      ++owned;
  owned = Utilities::MPI::sum(owned, MPI_COMM_WORLD);
  const auto changed =
    Utilities::MPI::sum(static_cast<std::uint64_t>(mutable_ids(raw).size()),
                        MPI_COMM_WORLD);
  // The independent full commit relocates every midpoint twice. This gate
  // exposes that cost even when ordinary growth uses an incremental backend.
  EXPECT_EQ(total(GrowthOperation::vessel_midpoint_localizations), 2 * owned);
  EXPECT_EQ(total(GrowthOperation::certified_vessels), changed);
  EXPECT_EQ(total(GrowthOperation::materialized_vessels), changed);
  EXPECT_GT(total(GrowthOperation::physical_copy_attempts), owned);
}

TEST(VascularForest, BOTH_PersistentCenterlineIndicesFollowCommittedDeltas)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  const auto initial = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
  GrowthWork work;
  GrowthIndices indices(
    initial, mutable_ids(initial), roots(initial), MPI_COMM_WORLD, &work);
  const auto check = [&](const NetworkData  &network,
                         const std::set<Id> &mutable_owned) {
    std::map<Id, std::array<Point<3>, 2>> expected, actual;
    for (const auto &[id, v] : network.vessels)
      if (v.owner == rank())
        expected[id] = {{network.nodes.at(v.nodes[0]).position,
                         network.nodes.at(v.nodes[1]).position}};
    for (const auto &[segment, id] : indices.owned)
      actual[id] = {{segment.first, segment.second}};
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(indices.owned.size(), expected.size());
    for (const auto system : {System::arterial, System::venous})
      {
        std::set<Id> wanted, found;
        for (Id id : mutable_owned)
          if (network.vessels.at(id).system == system)
            wanted.insert(id);
        for (const auto &[segment, id] : indices.mutable_tree(system))
          found.insert(id);
        EXPECT_EQ(found, wanted);
        EXPECT_EQ(indices.mutable_tree(system).size(), wanted.size());
      }
    for (const auto p : {Point<3>(.023, .42, .373), Point<3>(2.78, .31, .92)})
      {
        const BoundingBox<3> box(
          std::make_pair(Point<3>(p - Point<3>(.4, .4, .4)),
                         p + Point<3>(.4, .4, .4)));
        std::set<Id> wanted, found;
        for (const auto &[id, points] : expected)
          if (boost::geometry::intersects(Segment<3>(points[0], points[1]),
                                          box))
            wanted.insert(id);
        for (const auto &[segment, id] :
             indices.owned | bgi::adaptors::queried(bgi::intersects(box)))
          found.insert(id);
        EXPECT_EQ(found, wanted);
        std::vector<std::pair<double, Id>> distances, hits;
        for (const auto &[id, points] : expected)
          distances.emplace_back(
            point_distance_square(p, Segment<3>(points[0], points[1])), id);
        for (const auto &[segment, id] :
             nearest_centerlines(indices.owned, p, 3, nullptr))
          hits.emplace_back(point_distance_square(p, segment), id);
        std::sort(distances.begin(), distances.end());
        std::sort(hits.begin(), hits.end());
        distances.resize(std::min<std::size_t>(3, distances.size()));
        EXPECT_EQ(hits, distances);
      }
  };
  check(initial, mutable_ids(initial));
  Parameters p;
  p.minimum_angle = 1.;
  BackgroundGeometry geometry(bg, mapping);
  const auto         committed = commit_growth_splits(bg,
                                              mapping,
                                              initial,
                                              mutable_ids(initial),
                                              roots(initial),
                                              concurrent_splits(initial),
                                              p,
                                              MPI_COMM_WORLD,
                                              &geometry);
  ASSERT_TRUE(committed.committed);
  indices.update(committed.network,
                 committed.mutable_owned,
                 committed.geometry_changes);
  check(committed.network, committed.mutable_owned);
  EXPECT_EQ(work[GrowthOperation::network_index_builds], 3u);
  p.minimum_radius    = 1.;
  const auto rejected = commit_growth_splits(bg,
                                             mapping,
                                             committed.network,
                                             committed.mutable_owned,
                                             committed.fixed_owned_roots,
                                             {},
                                             p,
                                             MPI_COMM_WORLD,
                                             &geometry);
  EXPECT_FALSE(rejected.committed);
  // A rejected physical transaction never publishes a centerline delta.
  EXPECT_TRUE(rejected.geometry_changes.changed_owned.empty());
  EXPECT_TRUE(rejected.geometry_changes.released_owned.empty());
  check(committed.network, committed.mutable_owned);
}

TEST(VascularForest, CanonicalNearestStopsAfterCutoffTies)
{
  constexpr unsigned int ties = 20;
  for (unsigned int population : {1000u, 2000u, 4000u})
    {
      std::vector<GrowthIndices::Leaf> leaves;
      for (unsigned int i = 0; i < ties; ++i)
        leaves.push_back(
          {{Point<3>(), Point<3>(1., 1. + i, 0.)}, base + 100 + i});
      for (unsigned int i = 0; i < population; ++i)
        leaves.push_back(
          {{Point<3>(100. + 2 * i, 0., 0.), Point<3>(101. + 2 * i, 1., 0.)},
           base + 1000 + i});
      auto tree = pack_rtree(leaves);
      for (unsigned int history = 0; history < 2; ++history)
        {
          GrowthWork work;
          const auto nearest =
            nearest_centerlines(tree, Point<3>(-1., 0., 0.), 1, &work);
          ASSERT_EQ(nearest.size(), 1u);
          EXPECT_EQ(nearest.front().second, base + 100);
          EXPECT_GE(work[GrowthOperation::network_index_hits], ties);
          EXPECT_LE(work[GrowthOperation::network_index_hits], ties + 1);
          // A different native tree layout must preserve the physical-id tie.
          if (history == 0)
            for (auto it = leaves.rbegin(); it != leaves.rend(); ++it)
              {
                ASSERT_EQ(tree.remove(*it), 1u);
                tree.insert(*it);
              }
        }
    }
}
