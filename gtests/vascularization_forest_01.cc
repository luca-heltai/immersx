// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/distribution.h>
#include <immersx/vascularization/generation.h>

#include <functional>

#include "../source/vascularization/forest_state.h"
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
