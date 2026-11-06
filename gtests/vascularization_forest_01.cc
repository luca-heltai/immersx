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
#include <stdexcept>

#include "../source/vascularization/demand_mass.h"
#include "../source/vascularization/forest_state.h"
#include "../source/vascularization/growth_batch.h"
#include "../source/vascularization/growth_commit.h"
#include "../source/vascularization/growth_demand.h"
#include "../source/vascularization/growth_demand_callbacks.h"
#include "../source/vascularization/growth_demand_journal.h"
#include "../source/vascularization/growth_demand_state.h"
#include "../source/vascularization/growth_execution.h"
#include "../source/vascularization/growth_loop.h"
#include "../source/vascularization/growth_ownership_delta.h"
#include "../source/vascularization/network_serialization.h"
#include "vascularization_forest_oracle.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
using namespace ImmersX::Vascularization::internal;
namespace
{
  void
  physical_controls(Parameters &p)
  {
    p.length_scale               = "meter";
    p.minimum_length             = 0.;
    p.density_beta               = 0.;
    p.coverage.service_radius    = 0.;
    p.venous.terminal_separation = 0.;
  }
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
          {
            EXPECT_EQ(b.parent, invalid_id);
          }
        else
          {
            EXPECT_EQ(b.parent, off + (local == 12 || local == 13 ? 11 : 12));
          }
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
  physical_controls(p);
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
  physical_controls(p);
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
  physical_controls(p);
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
  physical_controls(p);
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
  physical_controls(p);
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
        {
          EXPECT_NEAR(a.radius / b.radius, 1., 1.e-13);
        }
      else if (id == base + 11)
        {
          EXPECT_DOUBLE_EQ(a.radius, b.radius);
        }
      else if (id == base + 13)
        {
          EXPECT_GT(a.radius, b.radius);
        }
      else
        {
          EXPECT_LT(a.radius, b.radius);
        }
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
  physical_controls(p);
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
  physical_controls(p);
  auto controls = roots(n);
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
  physical_controls(p);
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
  physical_controls(p);
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
  physical_controls(p);
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
  physical_controls(p);
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
  physical_controls(p);
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
  physical_controls(p);
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
  physical_controls(p);
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
  physical_controls(p);
  auto                       state = evaluate(n, p);
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
TEST(VascularForest, BOTH_UnassignedFrozenTerritoryDoesNotResetFreshIds)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  auto         input = single_seed();
  if (rank() == size() - 1)
    {
      input.nodes[base + 2]      = {Point<3>(.2, 1.8, 1.5),
                                    rank(),
                                    Port::venous_outlet,
                                    Provenance::imported};
      input.nodes[base + 3]      = {Point<3>(.7, 1.8, 1.5),
                                    rank(),
                                    Port::none,
                                    Provenance::imported};
      auto vein                  = input.vessels.at(base + 10);
      vein.nodes                 = {{base + 2, base + 3}};
      vein.system                = System::venous;
      vein.provenance            = Provenance::imported;
      input.vessels[base + 11]   = vein;
      input.terminals[base + 31] = {
        base + 3, invalid_id, System::venous, rank(), 1.};
    }
  const auto initial = redistribute(bg, mapping, input, 0., MPI_COMM_WORLD);
  Parameters p;
  physical_controls(p);
  p.seed_length        = .2;
  p.terminal_count     = 3;
  p.candidate_count    = 4;
  p.optimization_steps = 5;
  p.sample_budget      = 200;
  const auto run       = grow_forest(bg,
                               mapping,
                               initial,
                               seed_ids(initial),
                               seed_roots(initial),
                               p,
                               MPI_COMM_WORLD);
  for (const auto &issue : run.validation.issues)
    ADD_FAILURE() << issue.message;
  ASSERT_TRUE(run.validation.valid());
  unsigned int                            leaves = 0;
  std::map<unsigned int, std::vector<Id>> send;
  for (const auto &[id, terminal] : run.accepted.network.terminals)
    if (terminal.owner == rank() && terminal.system == System::arterial)
      {
        ++leaves;
        EXPECT_GE(terminal.territory, base + 40);
        EXPECT_NE(terminal.territory, invalid_id);
        send[terminal.territory % size()].push_back(terminal.territory);
      }
  std::set<Id> territories;
  for (const auto &[source, ids] :
       Utilities::MPI::some_to_some(MPI_COMM_WORLD, send))
    for (Id id : ids)
      EXPECT_TRUE(territories.insert(id).second);
  EXPECT_EQ(Utilities::MPI::sum(leaves, MPI_COMM_WORLD), 3u);
}

TEST(VascularForest, BOTH_ExhaustedIdentifierNamespacesNeverWrap)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  for (unsigned int kind = 0; kind < 4; ++kind)
    {
      SCOPED_TRACE(kind);
      auto input = single_seed();
      if (rank() == size() - 1)
        {
          if (kind == 0)
            {
              auto node  = input.nodes.extract(base + 1);
              node.key() = invalid_id - 1;
              input.nodes.insert(std::move(node));
              input.vessels.at(base + 10).nodes[1] = invalid_id - 1;
              input.terminals.at(base + 30).node   = invalid_id - 1;
            }
          else if (kind == 1)
            {
              auto vessel  = input.vessels.extract(base + 10);
              vessel.key() = invalid_id - 1;
              input.vessels.insert(std::move(vessel));
            }
          else if (kind == 2)
            {
              auto terminal  = input.terminals.extract(base + 30);
              terminal.key() = invalid_id - 1;
              input.terminals.insert(std::move(terminal));
            }
          else if (kind == 3)
            input.terminals.at(base + 30).territory = invalid_id - 1;
        }
      const auto initial = redistribute(bg, mapping, input, 0., MPI_COMM_WORLD);
      std::set<Id>         mutable_ids;
      std::map<Id, double> roots;
      for (const auto &[id, vessel] : initial.vessels)
        if (vessel.owner == rank())
          {
            mutable_ids.insert(id);
            roots[id] = .006;
          }
      Parameters p;
      physical_controls(p);
      p.seed_length        = .2;
      p.terminal_count     = 1;
      const auto unchanged = grow_forest(
        bg, mapping, initial, mutable_ids, roots, p, MPI_COMM_WORLD);
      ASSERT_TRUE(unchanged.validation.valid());
      p.terminal_count     = 2;
      const auto exhausted = grow_forest(
        bg, mapping, initial, mutable_ids, roots, p, MPI_COMM_WORLD);
      EXPECT_FALSE(exhausted.validation.valid());
      bool reported = false;
      for (const auto &issue : exhausted.validation.issues)
        reported =
          reported || issue.message.find("identifier space is exhausted") !=
                        std::string::npos;
      EXPECT_TRUE(reported);
      EXPECT_EQ(exhausted.sampled_points, 0u);
      ASSERT_TRUE(exhausted.accepted.committed);
      const auto &after = exhausted.accepted.network;
      EXPECT_EQ(after.nodes.size(), initial.nodes.size());
      EXPECT_EQ(after.vessels.size(), initial.vessels.size());
      EXPECT_EQ(after.terminals.size(), initial.terminals.size());
      for (const auto &[id, node] : initial.nodes)
        {
          ASSERT_TRUE(after.nodes.count(id));
          EXPECT_EQ(after.nodes.at(id).position, node.position);
          EXPECT_EQ(after.nodes.at(id).owner, node.owner);
        }
      for (const auto &[id, vessel] : initial.vessels)
        {
          ASSERT_TRUE(after.vessels.count(id));
          EXPECT_EQ(after.vessels.at(id).nodes, vessel.nodes);
          EXPECT_EQ(after.vessels.at(id).radii, vessel.radii);
          EXPECT_EQ(after.vessels.at(id).viscosity, vessel.viscosity);
        }
      for (const auto &[id, terminal] : initial.terminals)
        {
          ASSERT_TRUE(after.terminals.count(id));
          EXPECT_EQ(after.terminals.at(id).node, terminal.node);
          EXPECT_EQ(after.terminals.at(id).territory, terminal.territory);
          EXPECT_EQ(after.terminals.at(id).demand, terminal.demand);
        }
    }
}

TEST(VascularForest,
     BOTH_BatchedGrowthPropagatesFromOneRealSeedAndRepeatsExactly)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  const auto   n = redistribute(bg, mapping, single_seed(), 0., MPI_COMM_WORLD);
  Parameters   p;
  physical_controls(p);
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
  physical_controls(p);
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
  physical_controls(p);
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
  physical_controls(p);
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
  physical_controls(p);
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
  physical_controls(p);
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

TEST(VascularForest, BOTH_ObstacleRemovalPreservesCoincidentPhysicalVessels)
{
  using Box = BoundingBox<3>;
  const Box original(
    std::make_pair(Point<3>(.1, .2, .3), Point<3>(.4, .5, .6)));
  const Box moved(std::make_pair(Point<3>(.2, .3, .4), Point<3>(.5, .6, .7)));
  GrowthExecution::ObstacleTree tree;
  tree.insert({original, base});
  tree.insert({original, base + 1});
  EXPECT_EQ(tree.remove({original, base}), 1u);
  ASSERT_EQ(tree.size(), 1u);
  EXPECT_EQ(tree.begin()->second, base + 1);
  tree.insert({moved, base});
  EXPECT_EQ(tree.remove({original, base}), 0u);
  EXPECT_EQ(tree.size(), 2u);
  EXPECT_EQ(tree.remove({moved, base}), 1u);
  ASSERT_EQ(tree.size(), 1u);
  EXPECT_EQ(tree.begin()->second, base + 1);
}

TEST(VascularForest, BOTH_CenterlineRemovalPreservesCoincidentPhysicalVessels)
{
  NetworkData n;
  const Id    first  = base + 10 * rank();
  n.nodes[first]     = {Point<3>(.2, .3, .4), rank()};
  n.nodes[first + 1] = {Point<3>(.4, .5, .6), rank()};
  n.nodes[first + 2] = n.nodes[first];
  n.nodes[first + 3] = n.nodes[first + 1];
  Vessel vessel;
  vessel.nodes     = {{first, first + 1}};
  vessel.radii     = {{.001, .001}};
  vessel.viscosity = .004;
  vessel.owner     = rank();
  vessel.system    = System::arterial;
  n.vessels.emplace(first + 4, vessel);
  vessel.nodes = {{first + 2, first + 3}};
  n.vessels.emplace(first + 5, vessel);
  const std::set<Id> mutable_owned{first + 4, first + 5};
  GrowthIndices      indices(n, mutable_owned, {}, MPI_COMM_WORLD, nullptr);
  indices.erase(first + 4);
  ASSERT_EQ(indices.owned.size(), 1u);
  EXPECT_EQ(indices.owned.begin()->second, first + 5);
  ASSERT_EQ(indices.mutable_tree(System::arterial).size(), 1u);
  EXPECT_EQ(indices.mutable_tree(System::arterial).begin()->second, first + 5);
  // Reinsert the same physical id at a different location.
  n.nodes[first].position[2] += .1;
  indices.update(n, mutable_owned, {{first + 4}, {}});
  EXPECT_EQ(indices.owned.size(), 2u);
  indices.erase(first + 4);
  EXPECT_EQ(indices.owned.size(), 1u);
  EXPECT_EQ(indices.owned.begin()->second, first + 5);
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
  physical_controls(p);
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
      GrowthIndices::Tree tree(leaves.begin(), leaves.end());
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

TEST(VascularForest, BOTH_DemandUpdateDirectoryTransport)
{
  GrowthDemandUpdate update;
  update.snapshot_revision        = 17;
  update.base_quadrature_revision = 3;
  update.quadrature_revision      = 4;
  update.terminals.push_back({base + rank(), .125 + rank()});
  update.canonical_total_demand[rank() % 3] = 7. + rank();
  std::map<unsigned int, std::vector<GrowthDemandUpdate>> send;
  send[(base + rank()) % size()].push_back(update);
  const auto    received = Utilities::MPI::some_to_some(MPI_COMM_WORLD, send);
  std::uint64_t count    = 0;
  for (const auto &[source, updates] : received)
    for (const auto &value : updates)
      {
        ++count;
        EXPECT_EQ(value.snapshot_revision, 17u);
        EXPECT_EQ(value.base_quadrature_revision, 3u);
        EXPECT_EQ(value.quadrature_revision, 4u);
        EXPECT_EQ(value.terminals.size(), 1u);
        if (value.terminals.size() != 1)
          continue;
        EXPECT_EQ(value.terminals.front().terminal, base + source);
        EXPECT_EQ(value.terminals.front().terminal % size(), rank());
        EXPECT_EQ(value.terminals.front().demand, .125 + source);
        for (unsigned int system = 0; system < 3; ++system)
          {
            EXPECT_EQ(value.canonical_total_demand[system].has_value(),
                      source % 3 == system);
            if (value.canonical_total_demand[system])
              {
                EXPECT_EQ(*value.canonical_total_demand[system], 7. + source);
              }
          }
      }
  EXPECT_EQ(Utilities::MPI::sum(count, MPI_COMM_WORLD), size());
}

TEST(VascularForest, BOTH_SharedBackgroundSurvivesRepeatedFullCommits)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  auto       network  = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
  auto       ids      = mutable_ids(network);
  auto       controls = roots(network);
  Parameters p;
  physical_controls(p);
  p.minimum_angle = 1.;
  GrowthWork         work;
  BackgroundGeometry geometry(bg, mapping, &work);
  for (unsigned int stage = 0; stage < 3; ++stage)
    {
      auto result = commit_growth_splits(bg,
                                         mapping,
                                         network,
                                         ids,
                                         controls,
                                         {},
                                         p,
                                         MPI_COMM_WORLD,
                                         &geometry,
                                         &work);
      ASSERT_TRUE(result.committed);
      network  = std::move(result.network);
      ids      = std::move(result.mutable_owned);
      controls = std::move(result.fixed_owned_roots);
    }
  EXPECT_EQ(work[GrowthOperation::background_cache_constructions], 1u);
  EXPECT_EQ(work[GrowthOperation::background_cell_visits], bg.n_active_cells());
  EXPECT_EQ(work[GrowthOperation::boundary_index_builds], 1u);
}

TEST(VascularForest, ExactMassRetainsTinyContributionsAfterReassignment)
{
  const auto tiny  = demand_mass_units(1.e-300);
  const auto large = demand_mass_units(1.e300);
  EXPECT_DOUBLE_EQ(physical_demand_mass(tiny + large), 1.e300);
  EXPECT_DOUBLE_EQ(physical_demand_mass(tiny + large - large), 1.e-300);
  EXPECT_DOUBLE_EQ(physical_demand_mass(tiny + large - tiny), 1.e300);
  EXPECT_DOUBLE_EQ(physical_demand_mass(tiny - tiny), 0.);
  for (const double mass : {0.,
                            std::numeric_limits<double>::denorm_min(),
                            std::numeric_limits<double>::min(),
                            1.e-300,
                            .1,
                            1.,
                            1.e300,
                            std::numeric_limits<double>::max()})
    EXPECT_DOUBLE_EQ(physical_demand_mass(demand_mass_units(mass)), mass);
}
TEST(VascularForest, ExactMassRoundsTiesToEvenWithoutPartitionDependence)
{
  const double step = std::ldexp(1., -52), half = step / 2.;
  const auto   one     = demand_mass_units(1.);
  const auto   epsilon = demand_mass_units(step);
  const auto   tie     = demand_mass_units(half);
  const auto   subnormal =
    demand_mass_units(std::numeric_limits<double>::denorm_min());
  EXPECT_DOUBLE_EQ(physical_demand_mass(one + tie), 1.);
  EXPECT_DOUBLE_EQ(physical_demand_mass(one + tie + subnormal), 1. + step);
  EXPECT_DOUBLE_EQ(physical_demand_mass(one + epsilon + tie), 1. + 2. * step);
  EXPECT_DOUBLE_EQ(physical_demand_mass(one + epsilon + tie - subnormal),
                   1. + step);
  EXPECT_DOUBLE_EQ(physical_demand_mass((one + tie) + (epsilon - tie)),
                   1. + step);
  EXPECT_TRUE(std::isinf(physical_demand_mass(
    demand_mass_units(std::numeric_limits<double>::max()) * 2)));
}
TEST(VascularForest, ExactMassRejectsInvalidInputs)
{
  for (double value : {-1.,
                       std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(demand_mass_units(value), dealii::ExceptionBase);
  EXPECT_THROW(physical_demand_mass(boost::multiprecision::cpp_int(-1)),
               dealii::ExceptionBase);
}

TEST(VascularForest, BOTH_IntegratedBootstrapReplacesAllNativeSeedDemands)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  auto         n = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
  const auto   leaves =
    collect_growth_leaf_terminals(n, mutable_ids(n), MPI_COMM_WORLD);
  ASSERT_TRUE(leaves.validation.valid());
  Parameters p;
  physical_controls(p);
  const auto before = evaluate(n, p);
  ASSERT_TRUE(before.validation.valid()) << issues(before);
  EXPECT_EQ(leaves.owned, before.owned_leaf_terminals);
  GrowthDemandJointUpdate weights;
  weights.update.snapshot_revision = 7;
  for (Id id : leaves.owned)
    weights.new_terminals.push_back({id, 2. + double(id % 7)});
  const auto valid =
    initialize_growth_demands(n, leaves.owned, weights, 7, MPI_COMM_WORLD);
  ASSERT_TRUE(valid.valid());
  for (Id id : leaves.owned)
    EXPECT_EQ(n.terminals.at(id).demand, 2. + double(id % 7));
  const auto after = evaluate(n, p);
  EXPECT_TRUE(after.validation.valid());
  double local = 0.;
  for (Id id : leaves.owned)
    local += n.terminals.at(id).demand;
  EXPECT_NEAR(after.total_demand[0] * after.demand_scale,
              Utilities::MPI::sum(local, MPI_COMM_WORLD),
              1.e-12);
  auto missing = weights;
  for (auto &entry : missing.new_terminals)
    entry.demand += 100.;
  for (auto it = missing.new_terminals.begin();
       it != missing.new_terminals.end();)
    if (it->terminal == base + 30)
      it = missing.new_terminals.erase(it);
    else
      ++it;
  const auto rejected =
    initialize_growth_demands(n, leaves.owned, missing, 7, MPI_COMM_WORLD);
  EXPECT_FALSE(rejected.valid());
  for (Id id : leaves.owned)
    EXPECT_EQ(n.terminals.at(id).demand, 2. + double(id % 7));
  auto stale                     = weights;
  stale.update.snapshot_revision = 6;
  EXPECT_FALSE(
    initialize_growth_demands(n, leaves.owned, stale, 7, MPI_COMM_WORLD)
      .valid());
}

TEST(VascularForest, BOTH_SparseDemandPreviewMatchesFullRecompute)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  auto         n = redistribute(bg, mapping, forest(true), 0., MPI_COMM_WORLD);
  Parameters   p;
  physical_controls(p);
  p.dissipation_weight = .4;
  p.reference_flow     = 2.e-9;
  const auto accepted  = evaluate(n, p);
  ASSERT_TRUE(accepted.validation.valid()) << issues(accepted);
  GrowthDemandState state(n, accepted, p, 19, MPI_COMM_WORLD);
  ASSERT_TRUE(state.validation().valid());
  GrowthDemandUpdate update;
  update.snapshot_revision = 19;
  const std::map<Id, double> changed{{base + 31, 7.}, {base + 32, .125}};
  for (const auto &[id, value] : changed)
    if (id % size() == rank())
      update.terminals.push_back({id, value});
  const auto preview = state.preview(update);
  ASSERT_TRUE(preview.validation.valid());
  EXPECT_EQ(Utilities::MPI::sum(preview.marked_branches, MPI_COMM_WORLD), 4u);
  EXPECT_EQ(Utilities::MPI::sum(preview.recomputed_branches, MPI_COMM_WORLD),
            4u);
  for (auto &[id, terminal] : n.terminals)
    if (terminal.owner == rank() && changed.count(id))
      terminal.demand = changed.at(id);
  const auto expected = evaluate(n, p);
  ASSERT_TRUE(expected.validation.valid()) << issues(expected);
  for (const auto &[id, branch] : expected.owned)
    {
      const auto &value =
        preview.owned.count(id) ? preview.owned.at(id) : state.summary(id);
      const double gamma = n.vessels.at(id).system == System::arterial ?
                             p.arterial_murray_exponent :
                             p.venous.murray_exponent;
      const auto   mass  = physical_demand_mass(value.mass);
      EXPECT_NEAR(static_cast<double>(mass),
                  branch.demand * expected.demand_scale,
                  2.e-14 * static_cast<double>(mass));
      const long double volume =
        branch.volume_moment / std::pow(branch.demand, 2. / gamma);
      const long double resistance =
        branch.resistance_moment / std::pow(branch.demand, 2. - 4. / gamma);
      EXPECT_NEAR(static_cast<double>(value.volume_length),
                  static_cast<double>(volume),
                  2.e-13 * static_cast<double>(volume));
      EXPECT_NEAR(static_cast<double>(value.resistance_length),
                  static_cast<double>(resistance),
                  2.e-13 * static_cast<double>(resistance));
      EXPECT_NEAR(static_cast<double>(value.minimum_ratio),
                  std::pow(branch.minimum_demand / branch.demand, 1. / gamma),
                  2.e-14);
      EXPECT_EQ(value.leaves, branch.terminals);
    }
  for (unsigned int system = 0; system < 2; ++system)
    {
      EXPECT_NEAR(static_cast<double>(preview.totals[3 * system]),
                  expected.total_demand[system] * expected.demand_scale,
                  2.e-13);
      EXPECT_NEAR(static_cast<double>(preview.totals[3 * system + 1]),
                  expected.volume[system],
                  2.e-13 * expected.volume[system]);
      const double coefficient = expected.power_numerator[system] *
                                 expected.demand_scale * expected.demand_scale;
      EXPECT_NEAR(static_cast<double>(preview.totals[3 * system + 2]),
                  coefficient,
                  2.e-13 * coefficient);
    }
  // Previews are independent overlays: no accepted aggregate was changed.
  const auto repeated = state.preview(update);
  ASSERT_TRUE(repeated.validation.valid());
  EXPECT_EQ(repeated.recomputed_branches, preview.recomputed_branches);
  for (const auto &[id, value] : preview.owned)
    EXPECT_EQ(repeated.owned.at(id).mass, value.mass);
  GrowthDemandUpdate empty;
  empty.snapshot_revision = 19;
  const auto untouched    = state.preview(empty);
  ASSERT_TRUE(untouched.validation.valid());
  EXPECT_TRUE(untouched.owned.empty());
  EXPECT_EQ(untouched.marked_branches, 0u);
  auto stale              = update;
  stale.snapshot_revision = 18;
  EXPECT_FALSE(state.preview(stale).validation.valid());
  auto duplicate = update;
  if (!duplicate.terminals.empty())
    duplicate.terminals.push_back(duplicate.terminals.front());
  EXPECT_FALSE(state.preview(duplicate).validation.valid());
}

TEST(VascularForest, BOTH_DemandPreviewRetainsSmallUnchangedRootContribution)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  auto         n = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
  for (auto &[id, terminal] : n.terminals)
    if (terminal.owner == rank() && id < base + 100)
      terminal.demand = 1.e100;
  Parameters p;
  physical_controls(p);
  p.dissipation_weight = .5;
  p.reference_flow     = 2.e-9;
  const auto accepted  = evaluate(n, p);
  ASSERT_TRUE(accepted.validation.valid()) << issues(accepted);
  GrowthDemandState state(n, accepted, p, 2, MPI_COMM_WORLD);
  ASSERT_TRUE(state.validation().valid());
  GrowthDemandUpdate no_change;
  no_change.snapshot_revision = 2;
  const auto accepted_preview = state.preview(no_change);
  ASSERT_TRUE(accepted_preview.validation.valid());
  if (accepted.owned.count(base + 11))
    {
      auto other = state.root_view(base + 11, accepted_preview).other_local;
      for (unsigned int owner = 0; owner < size(); ++owner)
        if (owner != rank())
          for (unsigned int j = 0; j < 6; ++j)
            other[j] += accepted_preview.rank_totals.at(owner)[j];
      // Removing the dominant root by subtraction would erase this value.
      EXPECT_GT(other[2], 0.);
      EXPECT_NEAR(static_cast<double>(other[0]), 12., 1.e-14);
    }
  GrowthDemandUpdate update;
  update.snapshot_revision = 2;
  for (Id id = base + 30; id <= base + 32; ++id)
    if (id % size() == rank())
      update.terminals.push_back({id, 1.e-100});
  const auto preview = state.preview(update);
  ASSERT_TRUE(preview.validation.valid());
  EXPECT_EQ(Utilities::MPI::sum(preview.recomputed_branches, MPI_COMM_WORLD),
            5u);
  for (auto &[id, terminal] : n.terminals)
    if (terminal.owner == rank() && id < base + 100)
      terminal.demand = 1.e-100;
  const auto expected = evaluate(n, p);
  ASSERT_TRUE(expected.validation.valid()) << issues(expected);
  const double coefficient =
    expected.power_numerator[0] * expected.demand_scale * expected.demand_scale;
  EXPECT_GT(coefficient, 0.);
  EXPECT_NEAR(static_cast<double>(preview.totals[2]),
              coefficient,
              3.e-13 * coefficient);
  EXPECT_LT(preview.totals[2], state.totals()[2] * 1.e-100L);
  if (accepted.owned.count(base + 11))
    {
      const auto root  = state.root_view(base + 11, preview);
      auto       other = root.other_local;
      for (unsigned int owner = 0; owner < size(); ++owner)
        if (owner != rank())
          for (unsigned int j = 0; j < 6; ++j)
            other[j] += preview.rank_totals.at(owner)[j];
      EXPECT_NEAR(static_cast<double>(other[2]),
                  coefficient,
                  3.e-13 * coefficient);
      EXPECT_EQ(root.summary.mass, state.summary(base + 11, preview).mass);
      EXPECT_EQ(state.topology(base + 11).parent, invalid_id);
      EXPECT_EQ(state.children(base + 11, preview).size(), 2u);
      boost::multiprecision::cpp_int child_mass;
      for (const auto &[id, child] : state.children(base + 11, preview))
        child_mass += child.mass;
      EXPECT_EQ(child_mass, root.summary.mass);
    }
  EXPECT_NEAR(static_cast<double>(preview.totals[1]),
              expected.volume[0],
              3.e-13 * expected.volume[0]);
}

TEST(VascularForest, BOTH_DemandCandidateUsesUpdatedWeightsAndAcceptedReference)
{
  for (const System system : {System::arterial, System::portal})
    for (const double flow : {2.e-9, 1.e-180})
      for (const double new_demand : {2.5, 1.e-30})
        {
          parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
          background(bg);
          MappingQ1<3> mapping;
          auto n = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
          if (system == System::portal)
            {
              for (auto &[id, v] : n.vessels)
                v.system = System::portal;
              for (auto &[id, t] : n.terminals)
                t.system = System::portal;
              for (auto &[id, node] : n.nodes)
                if (node.port == Port::arterial_inlet)
                  node.port = Port::portal_inlet;
            }
          const unsigned int slot = system == System::portal ? 2u : 0u;
          Parameters         p;
          physical_controls(p);
          p.portal.enabled         = system == System::portal;
          p.portal.murray_exponent = 2.6;
          p.portal_flow_fraction   = .7;
          p.minimum_angle          = 1.;
          p.reference_flow         = flow;
          p.dissipation_weight     = .3;
          const auto accepted      = evaluate(n, p);
          ASSERT_TRUE(accepted.validation.valid()) << issues(accepted);
          GrowthDemandState state(n, accepted, p, 7, MPI_COMM_WORLD);
          ASSERT_TRUE(state.validation().valid());
          GrowthDemandUpdate update;
          update.snapshot_revision            = 7;
          const double final_demand           = 18.75 + new_demand;
          update.canonical_total_demand[slot] = final_demand;
          const std::map<Id, double> changed{{base + 30, .5},
                                             {base + 131, 2.25}};
          for (const auto &[id, demand] : changed)
            if (id % size() == rank())
              update.terminals.push_back({id, demand});
          const auto preview = state.preview(update);
          ASSERT_TRUE(preview.validation.valid());
          const auto                 producer = size() - 1;
          std::map<Id, unsigned int> requests;
          for (Id id : {base + 14, base + 15})
            {
              const auto owner =
                Utilities::MPI::min(n.vessels.count(id) &&
                                        n.vessels.at(id).owner == rank() ?
                                      rank() :
                                      size(),
                                    MPI_COMM_WORLD);
              if (rank() == producer)
                requests[id] = owner;
            }
          GrowthWork work;
          const auto paths =
            state.candidate_paths(n,
                                  preview,
                                  requests,
                                  base + 50,
                                  base + 70,
                                  rank() == producer ? new_demand : 0.,
                                  1.,
                                  &work);
          ASSERT_TRUE(paths.validation.valid());
          EXPECT_EQ(paths.retained_branches, rank() == producer ? 4u : 0u);
          EXPECT_EQ(work[GrowthOperation::validation_record_visits], 0u);
          EXPECT_EQ(work[GrowthOperation::physical_copy_attempts], 0u);
          std::optional<GrowthCandidate> candidate;
          std::vector<GrowthSplit>       splits;
          if (rank() == producer)
            {
              const auto &context = paths.contexts.at(base + 14);
              ASSERT_TRUE(context.physiology.has_value());
              EXPECT_EQ(context.physiology->final_system_demand, final_demand);
              EXPECT_EQ(context.physiology->reference_volume,
                        accepted.volume[slot]);
              const auto &selected = context.path.front();
              const auto  junction =
                .5 * (selected.edge.points[0] + selected.edge.points[1]);
              candidate =
                propose_growth(context,
                               p,
                               junction,
                               junction + Point<3>(0., 0., .2),
                               std::numeric_limits<double>::infinity(),
                               new_demand,
                               [](const std::vector<Edge> &) { return true; });
              EXPECT_TRUE(candidate.has_value());
              if (candidate)
                {
                  GrowthSplit split;
                  split.vessel            = candidate->vessel;
                  split.owner             = selected.owner;
                  split.old_nodes         = selected.edge.nodes;
                  split.old_points        = selected.edge.points;
                  split.junction          = base + 50;
                  split.tip               = base + 51;
                  split.junction_position = candidate->junction;
                  split.tip_position      = candidate->terminal;
                  split.continuation      = base + 70;
                  split.leaf              = base + 71;
                  split.terminal          = base + 90;
                  split.territory         = base + 95;
                  split.demand            = candidate->terminal_demand;
                  splits.push_back(split);
                }
            }
          // The independent full transaction consumes exactly the same absolute
          // Di.
          for (auto &[id, terminal] : n.terminals)
            if (terminal.owner == rank() && changed.count(id))
              terminal.demand = changed.at(id);
          const auto committed = commit_growth_splits(bg,
                                                      mapping,
                                                      n,
                                                      mutable_ids(n),
                                                      roots(n),
                                                      splits,
                                                      p,
                                                      MPI_COMM_WORLD);
          ASSERT_TRUE(committed.committed);
          ASSERT_TRUE(committed.physiology.validation.valid());
          for (const auto &[id, terminal] : committed.network.terminals)
            if (id == base + 90 && terminal.owner == rank())
              {
                EXPECT_EQ(terminal.demand, new_demand);
                const double expected_flow =
                  flow *
                  (system == System::portal ? p.portal_flow_fraction : 1.) *
                  (new_demand / final_demand);
                EXPECT_GT(expected_flow, 0.);
                EXPECT_NEAR(committed.physiology.terminal_flows.at(id) /
                              expected_flow,
                            1.,
                            2.e-13);
              }
          for (const auto &[id, branch] : committed.physiology.owned)
            if (id == base + 71)
              {
                const double expected_radius =
                  .006 * std::pow(new_demand / (8.5 + new_demand),
                                  1. / (system == System::portal ?
                                          p.portal.murray_exponent :
                                          p.arterial_murray_exponent));
                EXPECT_GT(branch.radius, 0.);
                EXPECT_NEAR(branch.radius / expected_radius, 1., 2.e-13);
              }
          if (candidate)
            {
              const double power_ratio =
                committed.physiology.power_numerator[slot] /
                std::pow(committed.physiology.total_demand[slot], 2) /
                (accepted.power_numerator[slot] /
                 std::pow(accepted.total_demand[slot], 2));
              const double expected = (1. - p.dissipation_weight) *
                                        committed.physiology.volume[slot] /
                                        accepted.volume[slot] +
                                      p.dissipation_weight * power_ratio;
              if (flow == 1.e-180)
                {
                  EXPECT_EQ(accepted.power[slot], 0.);
                }
              EXPECT_NEAR(candidate->cost, expected, 2.e-12);
            }
          std::map<unsigned int, std::map<Id, double>> send;
          for (const auto &[id, branch] : committed.physiology.owned)
            send[producer].emplace(id, branch.radius);
          const auto received =
            Utilities::MPI::some_to_some(MPI_COMM_WORLD, send);
          if (candidate)
            for (const auto &edge : candidate->changed)
              {
                bool found = false;
                for (const auto &[owner, radii] : received)
                  if (radii.count(edge.id))
                    {
                      EXPECT_NEAR(edge.radius / radii.at(edge.id), 1., 2.e-12);
                      found = true;
                    }
                EXPECT_TRUE(found);
              }
        }
}

TEST(VascularForest, BOTH_DemandEdgePreviewIncludesOtherRootsAndFrozenInput)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  auto         n = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
  Parameters   p;
  physical_controls(p);
  p.arterial_murray_exponent = 2.7;
  p.reference_flow           = 2.e-9;
  p.dissipation_weight       = .3;
  const auto accepted        = evaluate(n, p);
  ASSERT_TRUE(accepted.validation.valid());
  GrowthDemandState state(n, accepted, p, 11, MPI_COMM_WORLD);
  ASSERT_TRUE(state.validation().valid());
  GrowthDemandUpdate update;
  update.snapshot_revision = 11;
  const std::map<Id, double> changed{{base + 30, .5}, {base + 131, 2.25}};
  for (const auto &[id, value] : changed)
    if (id % size() == rank())
      update.terminals.push_back({id, value});
  const auto preview = state.preview(update);
  ASSERT_TRUE(preview.validation.valid());
  const unsigned int      producer = size() - 1;
  const std::array<Id, 5> queried{
    {base + 10, base + 13, base + 14, base + 113, base + 114}};
  std::map<Id, unsigned int> requests;
  for (Id id : queried)
    {
      const auto owner =
        Utilities::MPI::min(n.vessels.count(id) &&
                                n.vessels.at(id).owner == rank() ?
                              rank() :
                              size(),
                            MPI_COMM_WORLD);
      if (rank() == producer)
        requests.emplace(id, owner);
    }
  const auto edges = state.physical_edges(n, preview, requests);
  ASSERT_TRUE(edges.validation.valid());
  EXPECT_EQ(edges.records.size(), rank() == producer ? queried.size() : 0u);
  for (auto &[id, terminal] : n.terminals)
    if (terminal.owner == rank() && changed.count(id))
      terminal.demand = changed.at(id);
  const auto expected = evaluate(n, p);
  ASSERT_TRUE(expected.validation.valid());
  std::map<unsigned int, std::map<Id, std::array<double, 2>>> send;
  for (Id id : queried)
    if (n.vessels.count(id) && n.vessels.at(id).owner == rank())
      {
        const auto found = expected.owned.find(id);
        if (found == expected.owned.end())
          send[producer].emplace(id,
                                 std::array<double, 2>{
                                   {std::max(n.vessels.at(id).radii[0],
                                             n.vessels.at(id).radii[1]),
                                    0.}});
        else
          send[producer].emplace(id,
                                 std::array<double, 2>{
                                   {found->second.radius,
                                    found->second.root_demand *
                                      expected.demand_scale}});
      }
  const auto received = Utilities::MPI::some_to_some(MPI_COMM_WORLD, send);
  for (const auto &[owner, records] : received)
    for (const auto &[id, values] : records)
      {
        const auto &edge = edges.records.at(id);
        EXPECT_NEAR(edge.edge.radius, values[0], 2.e-13 * values[0]);
        EXPECT_NEAR(static_cast<double>(edge.root_demand),
                    values[1],
                    2.e-13 * values[1]);
        if (id == base + 10)
          {
            EXPECT_EQ(edge.root, invalid_id);
            EXPECT_EQ(edge.edge.radius, .009);
          }
        else
          {
            EXPECT_NE(edge.root, invalid_id);
            EXPECT_GT(edge.demand, 0.);
          }
      }
  // Provenance is not mutability. The same generated input rows stay frozen
  // when the accepted mutable set is empty.
  const auto frozen_forest = evaluate_forest(n, {}, {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(frozen_forest.validation.valid());
  GrowthDemandState frozen(n, frozen_forest, p, 11, MPI_COMM_WORLD);
  ASSERT_TRUE(frozen.validation().valid());
  GrowthDemandUpdate no_change;
  no_change.snapshot_revision = 11;
  const auto frozen_preview   = frozen.preview(no_change);
  ASSERT_TRUE(frozen_preview.validation.valid());
  const auto frozen_edges = frozen.physical_edges(n, frozen_preview, requests);
  ASSERT_TRUE(frozen_edges.validation.valid());
  if (rank() == producer)
    {
      EXPECT_EQ(frozen_edges.records.at(base + 13).root, invalid_id);
      EXPECT_EQ(frozen_edges.records.at(base + 13).edge.radius, .006);
    }
  auto stale = preview;
  --stale.snapshot_revision;
  EXPECT_FALSE(state.physical_edges(n, stale, requests).validation.valid());
}

TEST(VascularForest, BOTH_OneRootDemandUpdateDoesNotVisitOtherRoots)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  for (unsigned int count : {8u, 32u, 128u, 512u})
    {
      NetworkData n;
      if (rank() == size() - 1)
        for (unsigned int i = 0; i < count; ++i)
          {
            const Id first = base + 10000 + 2 * i, vessel = base + 1000 + 7 * i;
            const double x     = .25 + 3.2 * i / count;
            n.nodes[first]     = {Point<3>(x, .5, .5),
                                  rank(),
                                  Port::arterial_inlet};
            n.nodes[first + 1] = {Point<3>(x + .001, .5, .5), rank()};
            Vessel v;
            v.nodes      = {{first, first + 1}};
            v.radii      = {{.0001, .0001}};
            v.viscosity  = .004;
            v.owner      = rank();
            v.system     = System::arterial;
            v.provenance = Provenance::generated;
            n.vessels.emplace(vessel, v);
            n.terminals.emplace(base + 20000 + i,
                                Terminal{first + 1,
                                         base + 30000 + i,
                                         System::arterial,
                                         rank(),
                                         1.});
          }
      n = redistribute(bg, mapping, n, 0., MPI_COMM_WORLD);
      std::set<Id>         mutable_owned;
      std::map<Id, double> fixed;
      for (const auto &[id, vessel] : n.vessels)
        if (vessel.owner == rank())
          {
            mutable_owned.insert(id);
            fixed.emplace(id, .0001);
          }
      Parameters p;
      physical_controls(p);
      p.dissipation_weight = .3;
      p.reference_flow     = 2.e-9;
      const auto accepted =
        evaluate_forest(n, mutable_owned, fixed, p, MPI_COMM_WORLD);
      ASSERT_TRUE(accepted.validation.valid());
      GrowthDemandState state(n, accepted, p, 23, MPI_COMM_WORLD);
      ASSERT_TRUE(state.validation().valid());
      GrowthDemandUpdate update;
      update.snapshot_revision  = 23;
      const Id changed_terminal = base + 20000 + count - 1;
      const Id changed_root     = base + 1000 + 7 * (count - 1);
      if (rank() == 0)
        update.terminals.push_back({changed_terminal, 2.});
      const auto preview = state.preview(update);
      ASSERT_TRUE(preview.validation.valid());
      EXPECT_EQ(Utilities::MPI::sum(preview.recomputed_branches,
                                    MPI_COMM_WORLD),
                1u);
      EXPECT_LE(Utilities::MPI::sum(static_cast<std::uint64_t>(
                                      preview.root_sums.size()),
                                    MPI_COMM_WORLD),
                static_cast<std::uint64_t>(std::numeric_limits<Id>::digits +
                                           1));
      EXPECT_EQ(preview.totals[0], static_cast<long double>(count + 1));
      if (accepted.owned.count(changed_root))
        {
          auto other = state.root_view(changed_root, preview).other_local;
          for (unsigned int owner = 0; owner < size(); ++owner)
            if (owner != rank())
              for (unsigned int j = 0; j < 6; ++j)
                other[j] += preview.rank_totals[owner][j];
          EXPECT_EQ(other[0], static_cast<long double>(count - 1));
        }
      if (n.terminals.count(changed_terminal) &&
          n.terminals.at(changed_terminal).owner == rank())
        n.terminals.at(changed_terminal).demand = 2.;
      const auto expected =
        evaluate_forest(n, mutable_owned, fixed, p, MPI_COMM_WORLD);
      ASSERT_TRUE(expected.validation.valid());
      EXPECT_NEAR(static_cast<double>(preview.totals[1] / expected.volume[0]),
                  1.,
                  2.e-13);
      const double coefficient = expected.power_numerator[0] *
                                 expected.demand_scale * expected.demand_scale;
      EXPECT_NEAR(static_cast<double>(preview.totals[2] / coefficient),
                  1.,
                  2.e-13);
    }
}

TEST(VascularForest, BOTH_ScaledRadiusMinimumIncludesCoarseInternalBranch)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  for (unsigned int refinement = 0; refinement < 2; ++refinement)
    {
      for (const auto &cell : bg.active_cell_iterators())
        if (cell->is_locally_owned() && cell->center()[0] >= 1.)
          cell->set_refine_flag();
      bg.execute_coarsening_and_refinement();
    }
  double local_min = std::numeric_limits<double>::infinity(), local_max = 0.;
  double local_transition = 0.;
  for (const auto &cell : bg.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        local_min = std::min(local_min, cell->diameter());
        local_max = std::max(local_max, cell->diameter());
        if (cell->center()[0] > .5 && cell->center()[0] < 1.)
          local_transition = std::max(local_transition, cell->diameter());
      }
  const double fine   = Utilities::MPI::min(local_min, MPI_COMM_WORLD);
  const double coarse = Utilities::MPI::max(local_max, MPI_COMM_WORLD);
  const double transition =
    Utilities::MPI::max(local_transition, MPI_COMM_WORLD);
  ASSERT_NEAR(fine * 4., coarse, 2.e-14 * coarse);
  ASSERT_NEAR(transition * 2., coarse, 2.e-14 * coarse);
  MappingQ1<3> mapping;
  const auto   n = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
  Parameters   p;
  physical_controls(p);
  const auto accepted = evaluate(n, p);
  ASSERT_TRUE(accepted.validation.valid());
  GrowthGeometryMeasures measures;
  for (const auto &[id, branch] : accepted.owned)
    {
      const bool   root  = branch.parent == invalid_id;
      const double scale = root ? coarse : fine;
      // Only the root traverses x<1. All other mutable segments lie in the
      // uniformly twice-refined region, so these integrals are analytical.
      const double normalized =
        root ? .2 / coarse + .5 / transition + .2 / fine : length(n, id) / fine;
      measures.emplace(id,
                       GrowthGeometryMeasure{normalized, scale, scale, true});
    }
  GrowthDemandState state(n, accepted, p, 31, MPI_COMM_WORLD, &measures);
  ASSERT_TRUE(state.validation().valid());
  double leaf_min = std::numeric_limits<double>::infinity();
  for (const auto &[id, branch] : accepted.owned)
    {
      if (branch.children.empty())
        leaf_min =
          std::min(leaf_min, branch.radius / branch.root_radius / fine);
      if (branch.parent == invalid_id)
        {
          EXPECT_NEAR(static_cast<double>(
                        state.summary(id).minimum_scaled_ratio),
                      1. / coarse,
                      2.e-13 / coarse);
          EXPECT_NEAR(static_cast<double>(state.summary(id).minimum_ratio),
                      std::pow(branch.minimum_demand / branch.demand,
                               1. / p.arterial_murray_exponent),
                      2.e-13);
        }
    }
  EXPECT_GT(Utilities::MPI::min(leaf_min, MPI_COMM_WORLD), 1. / coarse);
  auto incomplete = measures;
  for (auto &[id, value] : incomplete)
    if (id == base + 11)
      value.complete = false;
  GrowthDemandState invalid(n, accepted, p, 31, MPI_COMM_WORLD, &incomplete);
  EXPECT_FALSE(invalid.validation().valid());
}

TEST(VascularForest, BOTH_DemandStateTracksPowerWithoutObjectiveWeight)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  const auto   n = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
  Parameters   p;
  physical_controls(p);
  p.reference_flow     = 2.e-9;
  p.dissipation_weight = 0.;
  const auto accepted  = evaluate(n, p);
  ASSERT_TRUE(accepted.validation.valid());
  ASSERT_GT(accepted.power[0], 0.);
  GrowthDemandState state(n, accepted, p, 5, MPI_COMM_WORLD);
  ASSERT_TRUE(state.validation().valid());
  const auto       &totals = state.totals();
  const long double power =
    p.reference_flow * p.reference_flow * totals[2] / totals[0] / totals[0];
  EXPECT_NEAR(static_cast<double>(power / accepted.power[0]), 1., 2.e-13);
}

TEST(VascularForest,
     BOTH_PhysicalDemandJournalStagesConcurrentEditsAndRollsBack)
{
  for (unsigned int scenario : {0u, 1u, 2u})
    {
      const bool                              concurrent = scenario == 1;
      parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
      background(bg);
      MappingQ1<3> mapping;
      auto         n = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
      Parameters   p;
      physical_controls(p);
      p.reference_flow     = 2.e-9;
      p.dissipation_weight = .3;
      const auto accepted  = evaluate(n, p);
      ASSERT_TRUE(accepted.validation.valid());
      GrowthDemandState state(n, accepted, p, 7, MPI_COMM_WORLD);
      ASSERT_TRUE(state.validation().valid());
      GrowthWork                                work;
      GrowthDemandJournal                       execution(state, &work);
      std::map<Id, GrowthDemandSummary>         original;
      std::map<Id, const GrowthDemandSummary *> addresses;
      for (const auto &[id, b] : accepted.owned)
        {
          original.emplace(id, state.summary(id));
          addresses.emplace(id, &state.summary(id));
        }
      std::vector<GrowthSplit> written;
      for (unsigned int j = 0; j < (concurrent ? 2u : 1u); ++j)
        {
          const Id   id     = base + (scenario == 2 ? 14 : 11 + j);
          const auto vessel = n.vessels.find(id);
          if (vessel == n.vessels.end() || vessel->second.owner != rank())
            continue;
          GrowthSplit s;
          s.vessel            = id;
          s.owner             = rank();
          s.old_nodes         = vessel->second.nodes;
          s.old_points        = {{n.nodes.at(s.old_nodes[0]).position,
                                  n.nodes.at(s.old_nodes[1]).position}};
          s.junction          = base + 1000 + 10 * j;
          s.tip               = s.junction + 1;
          s.continuation      = s.junction;
          s.leaf              = s.junction + 1;
          s.terminal          = s.junction;
          s.territory         = s.junction;
          s.junction_position = .5 * (s.old_points[0] + s.old_points[1]);
          s.tip_position      = s.junction_position + Point<3>(.1, .15, .1);
          s.demand            = 2.5 + j;
          written.push_back(s);
        }
      std::vector<GrowthSplit> splits;
      for (const auto &part :
           Utilities::MPI::all_gather(MPI_COMM_WORLD, written))
        splits.insert(splits.end(), part.begin(), part.end());
      ASSERT_EQ(splits.size(), concurrent ? 2u : 1u);
      std::vector<GrowthSplit> local;
      if (rank() == 0)
        local = splits;
      BackgroundGeometry geometry(bg, mapping);
      const auto         delta =
        prepare_growth_ownership(n, local, geometry, MPI_COMM_WORLD);
      ASSERT_TRUE(delta.validation.valid());
      GrowthGeometryMeasures measures;
      for (const auto &[id, vessel] : delta.written.vessels)
        measures.emplace(
          id, GrowthGeometryMeasure{length(delta.written, id), 1., 1., true});
      GrowthDemandJointUpdate joint;
      joint.update.snapshot_revision = 7;
      if (rank() == size() - 1)
        {
          joint.update.terminals = {{base + 30, .5},
                                    {base + 31, .75},
                                    {base + 131, 2.25}};
          for (const auto &s : splits)
            joint.new_terminals.push_back({s.terminal, s.demand});
        }
      auto full = n;
      for (auto &[id, terminal] : full.terminals)
        if (terminal.owner == rank())
          {
            if (id == base + 30)
              terminal.demand = .5;
            if (id == base + 31)
              terminal.demand = .75;
            if (id == base + 131)
              terminal.demand = 2.25;
          }
      for (const auto &s : splits)
        if (rank() == s.owner)
          {
            ASSERT_TRUE(apply_growth_split(full, s));
          }
      full              = redistribute(bg, mapping, full, 0., MPI_COMM_WORLD);
      auto mutable_full = mutable_ids(full);
      for (const auto &[id, vessel] : full.vessels)
        if (vessel.owner == rank() && delta.vessel_owners.count(id))
          mutable_full.insert(id);
      const auto oracle =
        evaluate_forest(full, mutable_full, roots(full), p, MPI_COMM_WORLD);
      ASSERT_TRUE(oracle.validation.valid()) << issues(oracle);
      std::array<double, 2> minimum_resistance{
        {std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()}},
        maximum_resistance{{0., 0.}};
      for (const auto &[id, branch] : oracle.owned)
        {
          const unsigned int tree = branch.root == base + 11 ? 0 : 1;
          minimum_resistance[tree] =
            std::min(minimum_resistance[tree], branch.resistance);
          maximum_resistance[tree] =
            std::max(maximum_resistance[tree], branch.resistance);
        }
      for (unsigned int tree = 0; tree < 2; ++tree)
        {
          minimum_resistance[tree] =
            Utilities::MPI::min(minimum_resistance[tree], MPI_COMM_WORLD);
          maximum_resistance[tree] =
            Utilities::MPI::max(maximum_resistance[tree], MPI_COMM_WORLD);
        }
      auto check = [&](const GrowthDemandPreview &preview) {
        ASSERT_TRUE(preview.validation.valid());
        std::map<Id, unsigned int> requests;
        for (Id id : mutable_full)
          requests.emplace(id, rank());
        const auto edges = state.physical_edges(full, preview, requests);
        ASSERT_TRUE(edges.validation.valid());
        for (const auto &[id, branch] : oracle.owned)
          {
            const auto &actual = edges.records.at(id);
            EXPECT_NEAR(static_cast<double>(
                          actual.demand /
                          (branch.demand * oracle.demand_scale)),
                        1.,
                        2.e-13);
            EXPECT_NEAR(actual.edge.radius / branch.radius, 1., 2.e-13);
            EXPECT_EQ(state.topology(id).parent, branch.parent);
            EXPECT_EQ(state.topology(id).parent_owner, branch.parent_owner);
            EXPECT_EQ(state.topology(id).children, branch.children);
            EXPECT_EQ(state.topology(id).terminal, branch.terminal);
            if (branch.parent == invalid_id)
              {
                const auto        &summary = state.summary(id, preview);
                const unsigned int tree    = branch.root == base + 11 ? 0 : 1;
                const long double  factor =
                  8.L / numbers::PI /
                  std::pow(static_cast<long double>(branch.root_radius), 4);
                EXPECT_NEAR(static_cast<double>(
                              factor * summary.minimum_resistance_length /
                              minimum_resistance[tree]),
                            1.,
                            2.e-13);
                EXPECT_NEAR(static_cast<double>(
                              factor * summary.maximum_resistance_length /
                              maximum_resistance[tree]),
                            1.,
                            2.e-13);
                EXPECT_DOUBLE_EQ(physical_demand_mass(
                                   summary.minimum_leaf_mass),
                                 tree == 0 ? .5 : 2.);
              }
            if (branch.terminal != invalid_id)
              {
                EXPECT_EQ(state.topology(id).terminal_owner,
                          branch.terminal_owner);
              }
          }
        EXPECT_NEAR(static_cast<double>(preview.totals[1] / oracle.volume[0]),
                    1.,
                    2.e-13);
        const long double power = p.reference_flow * p.reference_flow *
                                  preview.totals[2] / preview.totals[0] /
                                  preview.totals[0];
        EXPECT_NEAR(static_cast<double>(power / oracle.power[0]), 1., 2.e-13);
      };
      const auto &initial = execution.prepare(delta, joint, measures);
      check(initial);
      const std::map<Id, double> native_expected{{base + 30, .5},
                                                 {base + 31, .75},
                                                 {base + 131, 2.25}};
      for (const auto &[id, demand] : native_expected)
        {
          const auto terminal = n.terminals.find(id);
          const bool owned =
            terminal != n.terminals.end() && terminal->second.owner == rank();
          EXPECT_EQ(execution.native_terminal_demands().count(id),
                    owned ? 1u : 0u);
          if (owned)
            {
              EXPECT_DOUBLE_EQ(execution.native_terminal_demands().at(id),
                               demand);
            }
        }
      EXPECT_EQ(Utilities::MPI::sum(execution.native_terminal_demands().size(),
                                    MPI_COMM_WORLD),
                3u);
      EXPECT_EQ(work[GrowthOperation::physiology_branch_visits],
                initial.recomputed_branches);
      EXPECT_GE(work[GrowthOperation::transaction_records],
                initial.recomputed_branches);
      execution.discard();
      EXPECT_TRUE(execution.native_terminal_demands().empty());
      EXPECT_GT(Utilities::MPI::sum(work[GrowthOperation::rollback_records],
                                    MPI_COMM_WORLD),
                0u);
      for (const auto &[id, value] : original)
        {
          const auto &actual = state.summary(id);
          EXPECT_EQ(&actual, addresses.at(id));
          EXPECT_EQ(actual.mass, value.mass);
          EXPECT_EQ(actual.minimum_leaf_mass, value.minimum_leaf_mass);
          EXPECT_EQ(actual.minimum_resistance_length,
                    value.minimum_resistance_length);
          EXPECT_EQ(actual.maximum_resistance_length,
                    value.maximum_resistance_length);
          EXPECT_EQ(actual.volume_length, value.volume_length);
          EXPECT_EQ(actual.resistance_length, value.resistance_length);
          EXPECT_EQ(actual.minimum_ratio, value.minimum_ratio);
          EXPECT_EQ(actual.minimum_scaled_ratio, value.minimum_scaled_ratio);
          EXPECT_EQ(actual.leaves, value.leaves);
        }
      GrowthDemandUpdate old;
      old.snapshot_revision = 7;
      const auto restored   = state.preview(old);
      ASSERT_TRUE(restored.validation.valid());
      EXPECT_EQ(restored.totals, state.totals());
      auto malformed = measures;
      if (!malformed.empty())
        malformed.begin()->second.complete = false;
      EXPECT_FALSE(
        execution.prepare(delta, joint, malformed).validation.valid());
      execution.discard();
      check(execution.prepare(delta, joint, measures));
      execution.adopt();
      EXPECT_TRUE(execution.native_terminal_demands().empty());
      old.snapshot_revision = 8;
      const auto adopted    = state.preview(old);
      check(adopted);
      EXPECT_EQ(adopted.totals, state.totals());
      // A later absolute update must reach the newly remapped native leaf.
      GrowthDemandUpdate later;
      later.snapshot_revision = 8;
      if (rank() == 0)
        {
          later.terminals.push_back({splits.front().terminal, 1.75});
          later.terminals.push_back({base + 31, .875});
        }
      EXPECT_TRUE(state.preview(later).validation.valid());
    }
}

TEST(VascularForest,
     BOTH_ExecutionCommitsOneJointSnapshotAndRestoresRejectedStages)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  auto         n = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
  Parameters   p;
  physical_controls(p);
  p.reference_flow     = 2.e-9;
  p.dissipation_weight = .3;
  p.minimum_angle      = 1.;
  auto accepted        = evaluate(n, p);
  ASSERT_TRUE(accepted.validation.valid());
  for (const auto &[id, branch] : accepted.owned)
    n.vessels.at(id).radii = {{branch.radius, branch.radius}};
  n        = redistribute(bg, mapping, n, 0., MPI_COMM_WORLD);
  accepted = evaluate(n, p);
  ASSERT_TRUE(accepted.validation.valid());
  const auto meter = [](const GrowthGeometrySegments &segments) {
    GrowthGeometryMeasures result;
    for (const auto &[id, s] : segments)
      result.emplace(
        id, GrowthGeometryMeasure{s.first.distance(s.second), 1., 1., true});
    return result;
  };
  GrowthGeometrySegments initial_segments;
  for (const auto &[id, v] : n.vessels)
    if (v.owner == rank())
      initial_segments.emplace(id,
                               Segment<3>{n.nodes.at(v.nodes[0]).position,
                                          n.nodes.at(v.nodes[1]).position});
  const auto         initial_measures = meter(initial_segments);
  GrowthWork         caller, work;
  BackgroundGeometry geometry(bg, mapping, &caller);
  GrowthExecution    execution(n,
                            accepted,
                            p,
                            7,
                            geometry,
                            initial_measures,
                            meter,
                               {},
                            0.,
                            MPI_COMM_WORLD,
                            &work);
  ASSERT_TRUE(execution.validation().valid());
  auto first = execution.accepted_snapshot();
  ASSERT_TRUE(first.current());
  EXPECT_EQ(first.revision(), 7u);
  EXPECT_EQ(first.mutable_count(), 10u);
  EXPECT_EQ(first.terminal_count(), 6u);
  EXPECT_EQ(first.obstacles().size(), first.network().vessels.size());
  const auto original        = Utilities::pack(first.network(), false);
  const auto original_totals = first.demand().totals();
  const auto boxes           = [](const GrowthExecution::ObstacleTree &tree) {
    std::map<Id, std::pair<Point<3>, Point<3>>> result;
    for (const auto &[box, id] : tree)
      result.emplace(id, box.get_boundary_points());
    return result;
  };
  const auto               original_boxes = boxes(first.obstacles());
  std::vector<GrowthSplit> written;
  const auto               selected = n.vessels.find(base + 13);
  if (selected != n.vessels.end() && selected->second.owner == rank())
    {
      GrowthSplit split;
      split.vessel       = base + 13;
      split.owner        = rank();
      split.old_nodes    = selected->second.nodes;
      split.old_points   = {{n.nodes.at(split.old_nodes[0]).position,
                             n.nodes.at(split.old_nodes[1]).position}};
      split.junction     = base + 1000;
      split.tip          = base + 1001;
      split.continuation = base + 1000;
      split.leaf         = base + 1001;
      split.terminal     = base + 1000;
      split.territory    = base + 1002;
      split.junction_position =
        split.old_points[0] + .16 * (split.old_points[1] - split.old_points[0]);
      split.tip_position = split.junction_position + Point<3>(.1, .15, .1);
      split.demand       = 1.7;
      written.push_back(split);
    }
  std::vector<GrowthSplit> global;
  for (const auto &part : Utilities::MPI::all_gather(MPI_COMM_WORLD, written))
    global.insert(global.end(), part.begin(), part.end());
  ASSERT_EQ(global.size(), 1u);
  std::vector<GrowthSplit> local;
  if (rank() == size() - 1)
    local = global;
  GrowthDemandJointUpdate joint;
  joint.update.snapshot_revision = 7;
  if (rank() == 0)
    joint.update.terminals = {{base + 30, .5}, {base + 131, 2.25}};
  if (rank() == size() - 1)
    joint.new_terminals = {{global.front().terminal, 2.5}};
  unsigned int          prepared = 0, adopted = 0, discarded = 0;
  bool                  staged_tissue = false, fail_tissue = false;
  GrowthDemandCallbacks callbacks;
  callbacks.prepare_adopt = [&](std::uint64_t quad, std::uint64_t revision) {
    EXPECT_EQ(quad, 0u);
    EXPECT_EQ(revision, 8u);
    ++prepared;
    staged_tissue = true;
    if (fail_tissue && rank() == 0)
      throw std::runtime_error("Injected tissue adoption failure");
  };
  callbacks.adopt = [&]() {
    EXPECT_TRUE(staged_tissue);
    staged_tissue = false;
    ++adopted;
  };
  callbacks.discard = [&]() {
    staged_tissue = false;
    ++discarded;
  };
  auto bad = local;
  for (auto &split : bad)
    split.tip_position[2] = -.05;
  const auto rejected = execution.commit(bad, joint, callbacks);
  EXPECT_FALSE(rejected.committed);
  EXPECT_FALSE(first.current());
  EXPECT_THROW(first.network(), ExceptionBase);
  EXPECT_EQ(prepared, 0u);
  EXPECT_EQ(adopted, 0u);
  EXPECT_EQ(discarded, 1u);
  EXPECT_FALSE(staged_tissue);
  EXPECT_EQ(geometry.counts, &caller);
  auto after_wall = execution.accepted_snapshot();
  EXPECT_EQ(after_wall.revision(), 7u);
  EXPECT_EQ(Utilities::pack(after_wall.network(), false), original);
  EXPECT_EQ(after_wall.demand().totals(), original_totals);
  EXPECT_EQ(boxes(after_wall.obstacles()), original_boxes);
  fail_tissue        = true;
  const auto aborted = execution.commit(local, joint, callbacks);
  EXPECT_FALSE(aborted.committed);
  EXPECT_EQ(prepared, 1u);
  EXPECT_EQ(adopted, 0u);
  EXPECT_EQ(discarded, 2u);
  EXPECT_FALSE(staged_tissue);
  auto after_tissue = execution.accepted_snapshot();
  EXPECT_EQ(after_tissue.revision(), 7u);
  EXPECT_EQ(Utilities::pack(after_tissue.network(), false), original);
  EXPECT_EQ(after_tissue.demand().totals(), original_totals);
  EXPECT_EQ(boxes(after_tissue.obstacles()), original_boxes);
  fail_tissue          = false;
  const auto committed = execution.commit(local, joint, callbacks);
  ASSERT_TRUE(committed.committed);
  EXPECT_EQ(prepared, 2u);
  EXPECT_EQ(adopted, 1u);
  EXPECT_EQ(discarded, 2u);
  EXPECT_FALSE(staged_tissue);
  EXPECT_EQ(geometry.counts, &caller);
  const auto current = execution.accepted_snapshot();
  EXPECT_EQ(current.revision(), 8u);
  EXPECT_EQ(current.mutable_count(), 12u);
  EXPECT_EQ(current.terminal_count(), 7u);
  EXPECT_EQ(current.obstacles().size(), current.network().vessels.size());
  const auto actual = execution.materialize();
  auto       full   = n;
  for (auto &[id, t] : full.terminals)
    if (t.owner == rank())
      {
        if (id == base + 30)
          t.demand = .5;
        if (id == base + 131)
          t.demand = 2.25;
      }
  auto true_local = local;
  for (auto &split : true_local)
    split.demand = 2.5;
  const auto oracle = commit_growth_splits(
    bg, mapping, full, mutable_ids(n), roots(n), true_local, p, MPI_COMM_WORLD);
  std::string failures;
  for (const auto &issue : oracle.validation.issues)
    failures += issue.message + "\n";
  ASSERT_TRUE(oracle.committed) << failures;
  EXPECT_EQ(actual.mutable_owned, oracle.mutable_owned);
  EXPECT_EQ(actual.fixed_owned_roots, oracle.fixed_owned_roots);
  EXPECT_EQ(Utilities::pack(actual.network.nodes, false),
            Utilities::pack(oracle.network.nodes, false));
  EXPECT_EQ(Utilities::pack(actual.network.terminals, false),
            Utilities::pack(oracle.network.terminals, false));
  ASSERT_EQ(actual.network.vessels.size(), oracle.network.vessels.size());
  for (const auto &[id, v] : actual.network.vessels)
    {
      const auto found = oracle.network.vessels.find(id);
      ASSERT_NE(found, oracle.network.vessels.end());
      const auto &expected = found->second;
      EXPECT_EQ(v.nodes, expected.nodes);
      EXPECT_EQ(v.owner, expected.owner);
      EXPECT_EQ(v.physical_properties, expected.physical_properties);
      EXPECT_EQ(v.system, expected.system);
      EXPECT_EQ(v.provenance, expected.provenance);
      EXPECT_DOUBLE_EQ(v.viscosity, expected.viscosity);
      for (unsigned int end = 0; end < 2; ++end)
        EXPECT_NEAR(v.radii[end] / expected.radii[end], 1., 2.e-12);
    }
  EXPECT_NEAR(committed.volume[0] / oracle.physiology.volume[0], 1., 2.e-12);
  EXPECT_NEAR(committed.power[0] / oracle.physiology.power[0], 1., 2.e-12);
  GrowthWork      direct_work;
  GrowthExecution direct(n,
                         accepted,
                         p,
                         7,
                         geometry,
                         initial_measures,
                         meter,
                         {},
                         0.,
                         MPI_COMM_WORLD,
                         &direct_work);
  ASSERT_TRUE(direct.commit(local, joint).committed);
  EXPECT_EQ(Utilities::pack(execution.materialize().network, false),
            Utilities::pack(direct.materialize().network, false));
}

TEST(VascularForest, BOTH_ExecutionObstacleOverlayContainsEveryTrialCollision)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  auto         n = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
  Parameters   p;
  physical_controls(p);
  p.minimum_angle = 1.;
  auto accepted   = evaluate(n, p);
  ASSERT_TRUE(accepted.validation.valid());
  for (const auto &[id, branch] : accepted.owned)
    n.vessels.at(id).radii = {{branch.radius, branch.radius}};
  n                = redistribute(bg, mapping, n, 0., MPI_COMM_WORLD);
  accepted         = evaluate(n, p);
  const auto meter = [](const GrowthGeometrySegments &segments) {
    GrowthGeometryMeasures result;
    for (const auto &[id, s] : segments)
      result.emplace(
        id, GrowthGeometryMeasure{s.first.distance(s.second), 1., 1., true});
    return result;
  };
  GrowthGeometrySegments     segments;
  std::map<Id, unsigned int> native;
  for (const auto &[id, v] : n.vessels)
    if (v.owner == rank())
      {
        segments.emplace(id,
                         Segment<3>{n.nodes.at(v.nodes[0]).position,
                                    n.nodes.at(v.nodes[1]).position});
        native.emplace(id, rank());
      }
  std::map<Id, unsigned int> owners;
  for (const auto &part : Utilities::MPI::all_gather(MPI_COMM_WORLD, native))
    owners.insert(part.begin(), part.end());
  GrowthWork         work;
  BackgroundGeometry geometry(bg, mapping);
  GrowthExecution    execution(n,
                            accepted,
                            p,
                            7,
                            geometry,
                            meter(segments),
                            meter,
                               {},
                            0.,
                            MPI_COMM_WORLD,
                            &work);
  ASSERT_TRUE(execution.validation().valid());
  const auto                 snapshot = execution.accepted_snapshot();
  const unsigned int         producer = size() - 1;
  std::map<Id, unsigned int> requests;
  if (rank() == producer)
    requests.emplace(base + 13, owners.at(base + 13));
  for (unsigned int scenario = 0; scenario < 4; ++scenario)
    {
      GrowthDemandUpdate update;
      update.snapshot_revision = 7;
      if (rank() == 0 && scenario != 0)
        {
          update.terminals = {{base + 30, .5}, {base + 131, 2.25}};
          if (scenario == 2)
            update.terminals = {{base + 30, 1.e6}, {base + 131, .01}};
          if (scenario == 3)
            update.terminals = {{base + 30, .01},
                                {base + 31, .01},
                                {base + 32, .01},
                                {base + 131, 1.e6}};
        }
      const auto preview = snapshot.demand().preview(update);
      ASSERT_TRUE(preview.validation.valid());
      const double new_demand = scenario == 3 ? .005 : 2.5;
      const auto   paths = snapshot.demand().candidate_paths(snapshot.network(),
                                                           preview,
                                                           requests,
                                                           base + 1000,
                                                           base + 1000,
                                                           new_demand,
                                                           1.,
                                                           &work);
      ASSERT_TRUE(paths.validation.valid());
      const std::optional<Point<3>> target =
        rank() == producer ? std::optional<Point<3>>(Point<3>(3., .8, .7)) :
                             std::nullopt;
      const auto before = work;
      const auto obstacles =
        execution.candidate_obstacles(snapshot,
                                      preview,
                                      paths.contexts,
                                      target,
                                      rank() == producer ? new_demand : 0.,
                                      producer);
      ASSERT_TRUE(obstacles.validation.valid());
      EXPECT_TRUE(snapshot.current());
      EXPECT_EQ(work[GrowthOperation::materialized_vessels],
                before[GrowthOperation::materialized_vessels]);
      EXPECT_EQ(work[GrowthOperation::validation_record_visits],
                before[GrowthOperation::validation_record_visits]);
      auto full = n;
      for (const auto &part :
           Utilities::MPI::all_gather(MPI_COMM_WORLD, update.terminals))
        for (const auto &change : part)
          {
            const auto found = full.terminals.find(change.terminal);
            if (found != full.terminals.end())
              found->second.demand = change.demand;
          }
      const auto oracle = evaluate(full, p);
      ASSERT_TRUE(oracle.validation.valid());
      std::map<Id, GrowthDemandEdge> local_edges;
      for (const auto &[id, v] : full.vessels)
        if (v.owner == rank())
          {
            const auto       branch = oracle.owned.find(id);
            GrowthDemandEdge value;
            value.edge = {id,
                          v.nodes,
                          {{full.nodes.at(v.nodes[0]).position,
                            full.nodes.at(v.nodes[1]).position}},
                          branch == oracle.owned.end() ?
                            std::max(v.radii[0], v.radii[1]) :
                            branch->second.radius};
            if (branch != oracle.owned.end())
              value.root = branch->second.root;
            local_edges.emplace(id, value);
          }
      std::map<Id, GrowthDemandEdge> all_edges;
      for (const auto &part :
           Utilities::MPI::all_gather(MPI_COMM_WORLD, local_edges))
        all_edges.insert(part.begin(), part.end());
      if (rank() == producer)
        {
          ASSERT_EQ(paths.contexts.size(), 1u);
          const auto &context     = paths.contexts.at(base + 13);
          const auto  trial_radii = candidate_radii(context, new_demand);
          ASSERT_TRUE(trial_radii);
          for (const auto &[id, record] : obstacles.records)
            {
              EXPECT_EQ(obstacles.owners.at(id), owners.at(id));
              EXPECT_EQ(record.root, all_edges.at(id).root);
              EXPECT_NEAR(record.edge.radius / all_edges.at(id).edge.radius,
                          1.,
                          2.e-12);
            }
          const auto &ends = context.path.front().edge.points;
          const auto  candidate =
            propose_growth(context,
                           p,
                           .5 * (ends[0] + ends[1]),
                           *target,
                           std::numeric_limits<double>::infinity(),
                           new_demand,
                           [](const auto &) { return true; });
          ASSERT_TRUE(candidate);
          for (const auto &changed : candidate->changed)
            {
              const auto &mask = obstacles.covered_ancestors.at(base + 13);
              if (mask.count(changed.id))
                continue;
              for (const auto &[id, baseline] : all_edges)
                {
                  if (id == base + 13 || id == changed.id)
                    continue;
                  auto edge   = baseline.edge;
                  edge.radius = (*trial_radii)(id, baseline.root, edge.radius);
                  if (!compatible(changed, edge, p.minimum_angle, 0.))
                    {
                      EXPECT_TRUE(obstacles.records.count(id))
                        << scenario << " " << id;
                    }
                }
            }
          if (scenario == 3)
            {
              EXPECT_TRUE(obstacles.covered_ancestors.at(base + 13).empty());
            }
        }
      else
        {
          EXPECT_TRUE(obstacles.records.empty());
          EXPECT_TRUE(obstacles.owners.empty());
          EXPECT_TRUE(obstacles.covered_ancestors.empty());
        }
      auto stale              = preview;
      stale.snapshot_revision = 6;
      EXPECT_FALSE(
        execution
          .candidate_obstacles(
            snapshot, stale, paths.contexts, target, new_demand, producer)
          .validation.valid());
    }
  GrowthDemandUpdate unchanged;
  unchanged.snapshot_revision = 7;
  const auto preview          = snapshot.demand().preview(unchanged);
  const auto empty =
    execution.candidate_obstacles(snapshot, preview, {}, {}, 0., producer);
  EXPECT_TRUE(empty.validation.valid());
  EXPECT_TRUE(empty.records.empty());
}

TEST(VascularForest,
     BOTH_ExecutionBoundsTouchOnlyChangedRootsAndKeepRemoteFlowMinimum)
{
  for (unsigned int count : {8u, 32u, 128u})
    {
      parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
      GridGenerator::subdivided_hyper_rectangle(bg,
                                                {4, 1, 1},
                                                Point<3>(),
                                                Point<3>(1., 1., 1.));
      MappingQ1<3> mapping;
      NetworkData  input;
      if (rank() == size() - 1)
        for (unsigned int i = 0; i < count; ++i)
          {
            const Id     id              = base + 4 * i;
            const double y               = (i + 1.) / (count + 1.);
            input.nodes[id].position     = Point<3>(.1, y, .5);
            input.nodes[id].port         = Port::arterial_inlet;
            input.nodes[id].owner        = rank();
            input.nodes[id + 1].position = Point<3>(.9, y, .5);
            input.nodes[id + 1].owner    = rank();
            Vessel vessel;
            vessel.nodes      = {{id, id + 1}};
            vessel.owner      = rank();
            vessel.radii      = {{1.e-5, 1.e-5}};
            vessel.viscosity  = .004;
            vessel.system     = System::arterial;
            vessel.provenance = Provenance::generated;
            input.vessels.emplace(id, vessel);
            input.terminals.emplace(id,
                                    Terminal{id + 1,
                                             id,
                                             System::arterial,
                                             rank(),
                                             i == count - 1 ? 1.e-300 : 1.});
          }
      const auto   n = redistribute(bg, mapping, input, 0., MPI_COMM_WORLD);
      std::set<Id> mutable_owned;
      std::map<Id, double>   fixed;
      GrowthGeometrySegments segments;
      for (const auto &[id, v] : n.vessels)
        if (v.owner == rank())
          {
            mutable_owned.insert(id);
            fixed.emplace(id, 1.e-5);
            segments.emplace(id,
                             Segment<3>{n.nodes.at(v.nodes[0]).position,
                                        n.nodes.at(v.nodes[1]).position});
          }
      Parameters p;
      physical_controls(p);
      p.reference_flow = 2.e-9;
      const auto initial =
        evaluate_forest(n, mutable_owned, fixed, p, MPI_COMM_WORLD);
      ASSERT_TRUE(initial.validation.valid()) << issues(initial);
      const auto meter = [](const GrowthGeometrySegments &segments) {
        GrowthGeometryMeasures result;
        for (const auto &[id, s] : segments)
          result.emplace(
            id,
            GrowthGeometryMeasure{s.first.distance(s.second), 1., 1., true});
        return result;
      };
      GrowthWork         work;
      BackgroundGeometry geometry(bg, mapping);
      GrowthExecution    execution(n,
                                initial,
                                p,
                                0,
                                geometry,
                                meter(segments),
                                meter,
                                   {},
                                0.,
                                MPI_COMM_WORLD,
                                &work);
      ASSERT_TRUE(execution.validation().valid());
      std::vector<GrowthSplit> local;
      const auto               found = n.vessels.find(base);
      if (found != n.vessels.end() && found->second.owner == rank())
        {
          GrowthSplit split;
          split.vessel            = base;
          split.owner             = rank();
          split.old_nodes         = found->second.nodes;
          split.old_points        = {{n.nodes.at(split.old_nodes[0]).position,
                                      n.nodes.at(split.old_nodes[1]).position}};
          split.junction          = base + 10000;
          split.tip               = base + 10001;
          split.continuation      = base + 10000;
          split.leaf              = base + 10001;
          split.terminal          = base + 10000;
          split.territory         = base + 10000;
          split.junction_position = Point<3>(.35, 1. / (count + 1.), .5);
          split.tip_position      = Point<3>(.35, 1. / (count + 1.) + .002, .7);
          split.demand            = 1.;
          local.push_back(split);
        }
      GrowthDemandJointUpdate joint;
      joint.update.snapshot_revision = 0;
      if (rank() == 0)
        joint.new_terminals = {{base + 10000, 1.e20}};
      const auto before   = work;
      const auto rejected = execution.commit(local, joint);
      EXPECT_FALSE(rejected.committed);
      EXPECT_EQ(execution.accepted_snapshot().revision(), 0u);
      const auto failed_visits =
        Utilities::MPI::sum(work[GrowthOperation::physiology_branch_visits] -
                              before[GrowthOperation::physiology_branch_visits],
                            MPI_COMM_WORLD);
      EXPECT_LE(failed_visits, 8u) << count;
      if (rank() == 0)
        joint.new_terminals = {{base + 10000, 1.}};
      const auto retry_before = work;
      const auto actual       = execution.commit(local, joint);
      ASSERT_TRUE(actual.committed);
      const auto visits = Utilities::MPI::sum(
        work[GrowthOperation::physiology_branch_visits] -
          retry_before[GrowthOperation::physiology_branch_visits],
        MPI_COMM_WORLD);
      EXPECT_LE(visits, 8u) << count;
      const auto expected = commit_growth_splits(
        bg, mapping, n, mutable_owned, fixed, local, p, MPI_COMM_WORLD);
      ASSERT_TRUE(expected.committed);
      const auto materialized = execution.materialize();
      EXPECT_EQ(materialized.mutable_owned, expected.mutable_owned);
      EXPECT_EQ(materialized.fixed_owned_roots, expected.fixed_owned_roots);
      EXPECT_EQ(Utilities::pack(materialized.network.terminals, false),
                Utilities::pack(expected.network.terminals, false));
      ASSERT_EQ(materialized.network.vessels.size(),
                expected.network.vessels.size());
      for (const auto &[id, v] : materialized.network.vessels)
        {
          const auto &oracle = expected.network.vessels.at(id);
          EXPECT_EQ(v.owner, oracle.owner);
          EXPECT_EQ(v.nodes, oracle.nodes);
          EXPECT_NEAR(v.radii[0] / oracle.radii[0], 1., 2.e-12);
        }
    }
}

TEST(VascularForest, BOTH_ExecutionReusesAncestorCapsAndAdvancesShrinkEvents)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  background(bg);
  MappingQ1<3> mapping;
  auto         n = redistribute(bg, mapping, forest(), 0., MPI_COMM_WORLD);
  Parameters   p;
  physical_controls(p);
  p.minimum_angle = 1.;
  auto initial    = evaluate(n, p);
  ASSERT_TRUE(initial.validation.valid());
  for (const auto &[id, branch] : initial.owned)
    n.vessels.at(id).radii = {{branch.radius, branch.radius}};
  n                = redistribute(bg, mapping, n, 0., MPI_COMM_WORLD);
  initial          = evaluate(n, p);
  const auto meter = [](const GrowthGeometrySegments &segments) {
    GrowthGeometryMeasures result;
    for (const auto &[id, s] : segments)
      result.emplace(
        id, GrowthGeometryMeasure{s.first.distance(s.second), 1., 1., true});
    return result;
  };
  GrowthGeometrySegments segments;
  for (const auto &[id, v] : n.vessels)
    if (v.owner == rank())
      segments.emplace(id,
                       Segment<3>{n.nodes.at(v.nodes[0]).position,
                                  n.nodes.at(v.nodes[1]).position});
  GrowthWork         work;
  BackgroundGeometry geometry(bg, mapping);
  GrowthExecution    execution(n,
                            initial,
                            p,
                            0,
                            geometry,
                            meter(segments),
                            meter,
                               {},
                            0.,
                            MPI_COMM_WORLD,
                            &work);
  ASSERT_TRUE(execution.validation().valid());
  auto     expected_network = n;
  auto     expected_mutable = mutable_ids(n);
  auto     expected_roots   = roots(n);
  const Id selected[]      = {base + 14, base + 2000, base + 3001, base + 4000};
  const Point<3> offsets[] = {Point<3>(.1, .1, .15),
                              Point<3>(.05, .08, .1),
                              Point<3>(.04, .05, .08),
                              Point<3>(.08, -.02, -.05)};
  const auto     width     = [&]() {
    double       result   = 0.;
    unsigned int count    = 0;
    const auto   snapshot = execution.accepted_snapshot();
    for (const auto &[box, id] : snapshot.obstacles())
      if (id == base + 13)
        {
          result = box.get_boundary_points().second[2] -
                   box.get_boundary_points().first[2];
          ++count;
        }
    EXPECT_EQ(Utilities::MPI::sum(count, MPI_COMM_WORLD), 1u);
    return Utilities::MPI::sum(result, MPI_COMM_WORLD);
  };
  double before_shrink = 0., after_shrink = 0.;
  for (unsigned int step = 0; step < 4; ++step)
    {
      const Id                 next     = base + 2000 + step * 1000;
      const auto               snapshot = execution.accepted_snapshot();
      const auto              &rows     = snapshot.network();
      std::vector<GrowthSplit> local;
      const auto               found = rows.vessels.find(selected[step]);
      if (found != rows.vessels.end())
        {
          GrowthSplit split;
          split.vessel       = selected[step];
          split.owner        = rank();
          split.old_nodes    = found->second.nodes;
          split.old_points   = {{rows.nodes.at(split.old_nodes[0]).position,
                                 rows.nodes.at(split.old_nodes[1]).position}};
          split.junction     = next;
          split.tip          = next + 1;
          split.continuation = next;
          split.leaf         = next + 1;
          split.terminal     = next;
          split.territory    = next + 2;
          split.junction_position =
            .5 * (split.old_points[0] + split.old_points[1]);
          split.tip_position = split.junction_position + offsets[step];
          split.demand       = .001;
          local.push_back(split);
        }
      GrowthDemandJointUpdate joint;
      joint.update.snapshot_revision = step;
      if (rank() == 0)
        {
          joint.new_terminals = {{next, .001}};
          if (step == 2)
            joint.update.terminals = {{base + 32, 50.}};
        }
      if (step == 2)
        before_shrink = width();
      const auto before = work;
      const auto actual = execution.commit(local, joint);
      ASSERT_TRUE(actual.committed);
      if (step == 1)
        {
          const auto certified =
            Utilities::MPI::sum(work[GrowthOperation::certified_vessels] -
                                  before[GrowthOperation::certified_vessels],
                                MPI_COMM_WORLD);
          EXPECT_EQ(certified, 3u);
        }
      if (step == 2)
        {
          after_shrink = width();
          EXPECT_LT(after_shrink, before_shrink);
        }
      if (step == 3)
        {
          EXPECT_DOUBLE_EQ(width(), after_shrink);
        }
      if (step == 2)
        for (auto &[id, terminal] : expected_network.terminals)
          if (id == base + 32 && terminal.owner == rank())
            terminal.demand = 50.;
      const auto expected = commit_growth_splits(bg,
                                                 mapping,
                                                 expected_network,
                                                 expected_mutable,
                                                 expected_roots,
                                                 local,
                                                 p,
                                                 MPI_COMM_WORLD);
      ASSERT_TRUE(expected.committed);
      const auto materialized = execution.materialize();
      ASSERT_EQ(materialized.network.vessels.size(),
                expected.network.vessels.size());
      for (const auto &[id, vessel] : materialized.network.vessels)
        {
          const auto &full = expected.network.vessels.at(id);
          EXPECT_EQ(vessel.owner, full.owner);
          EXPECT_EQ(vessel.nodes, full.nodes);
          EXPECT_NEAR(vessel.radii[0] / full.radii[0], 1., 2.e-12);
        }
      expected_network = expected.network;
      expected_mutable = expected.mutable_owned;
      expected_roots   = expected.fixed_owned_roots;
    }
}

TEST(VascularForest, BOTH_RejectedBestRetriesAnotherParentAtTheSameTerritory)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {2, 1, 1},
                                            Point<3>(),
                                            Point<3>(1., 1., 1.));
  for (const auto &cell : bg.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[0] < .5)
      cell->set_refine_flag();
  bg.execute_coarsening_and_refinement();
  MappingQ1<3> mapping;

  const Id                bad = base + 10010, good = base + 11010;
  const std::array<Id, 3> territories{{base + 40, base + 140, base + 240}};
  const std::array<Point<3>, 2> start{
    {Point<3>(.65, .4, .5), Point<3>(.95, .25, .5)}};
  const std::array<Point<3>, 2> end{
    {Point<3>(.65, .6, .5), Point<3>(.95, .75, .5)}};
  const std::array<double, 2> radii{{.0002, .001}};
  const Point<3>              target(.78, .5, .5);
  constexpr double            hmin = .75, hcoarse = 1.5, offset = .00015;
  // Exact diameters of the refined (.25,.5,.5) and coarse (.5,1,1) boxes.
  double local_min = std::numeric_limits<double>::max(), local_max = 0.;
  for (const auto &cell : bg.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        local_min = std::min(local_min, cell->diameter());
        local_max = std::max(local_max, cell->diameter());
      }
  EXPECT_NEAR(Utilities::MPI::min(local_min, MPI_COMM_WORLD), hmin, 1.e-14);
  EXPECT_NEAR(Utilities::MPI::max(local_max, MPI_COMM_WORLD), hcoarse, 1.e-14);
  for (unsigned int direction = 0; direction < 6; ++direction)
    {
      Point<3> tip = target;
      tip[direction / 2] += (direction % 2 ? 1. : -1.) * offset;
      const auto delta = [&](unsigned int tree) {
        const auto   midpoint = .5 * (start[tree] + end[tree]);
        const double half     = .5 * start[tree].distance(end[tree]);
        return radii[tree] * radii[tree] *
               (half +
                std::pow(2., -2. / 3.) * (half + midpoint.distance(tip)) -
                2. * half);
      };
      EXPECT_LT(delta(0), delta(1)); // Common positive forest-volume divisor.
      EXPECT_GT(.1, .1 * hmin);
      EXPECT_LT(.1 / hcoarse, .1);
      EXPECT_GT(.25 / hcoarse, .1);
      EXPECT_GT((.5 * (start[1] + end[1])).distance(tip) / hcoarse, .1);
    }

  NetworkData            input;
  std::vector<Territory> common;
  if (rank() == size() - 1)
    {
      for (unsigned int tree = 0; tree < 2; ++tree)
        {
          const Id id          = tree == 0 ? bad : good;
          input.nodes[id - 10] = {start[tree],
                                  rank(),
                                  Port::venous_outlet,
                                  Provenance::generated};
          input.nodes[id - 9]  = {end[tree],
                                  rank(),
                                  Port::none,
                                  Provenance::generated};
          Vessel vessel;
          vessel.nodes             = {{id - 10, id - 9}};
          vessel.radii             = {{radii[tree], radii[tree]}};
          vessel.viscosity         = .004;
          vessel.system            = System::venous;
          vessel.provenance        = Provenance::generated;
          vessel.owner             = rank();
          input.vessels[id]        = vessel;
          input.terminals[id + 20] = {
            id - 9, territories[tree], System::venous, rank(), 1.};
          common.push_back({territories[tree], end[tree], rank(), 1.});
        }
      common.push_back({territories[2], target, rank(), 1.});
    }
  const auto   native = redistribute(bg, mapping, input, 0., MPI_COMM_WORLD);
  std::set<Id> mutable_owned;
  std::map<Id, double> roots;
  for (unsigned int tree = 0; tree < 2; ++tree)
    {
      const Id id = tree == 0 ? bad : good;
      if (native.vessels.count(id) && native.vessels.at(id).owner == rank())
        {
          mutable_owned.insert(id);
          roots[id] = radii[tree];
        }
    }
  Parameters p;
  physical_controls(p);
  p.length_scale                     = "local mesh size";
  p.terminal_count_multiplier_factor = "none";
  p.terminal_count                   = 3;
  p.minimum_length                   = .1;
  p.minimum_radius                   = 0.;
  p.clearance                        = 0.;
  p.minimum_angle                    = 1.;
  p.candidate_count                  = 2;
  p.optimization_steps               = 0;
  p.sample_budget                    = 12;
  p.reference_flow                   = 2.e-9;
  p.dissipation_weight               = 0.;
  p.venous.enabled                   = true;
  p.venous.murray_exponent           = 3.;
  p.venous.terminal_separation       = 1.e-4;
  const auto  run                    = grow_forest(bg,
                               mapping,
                               native,
                               mutable_owned,
                               roots,
                               p,
                               MPI_COMM_WORLD,
                               3,
                               System::venous,
                               common);
  std::string messages;
  for (const auto &issue : run.validation.issues)
    messages += issue.message + "\n";
  ASSERT_TRUE(run.validation.valid()) << messages;
  ASSERT_TRUE(run.accepted.committed);
  EXPECT_EQ(Utilities::MPI::sum(run.accepted_insertions, MPI_COMM_WORLD), 1u);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<std::uint64_t>(
                                  run.accepted.fixed_owned_roots.size()),
                                MPI_COMM_WORLD),
            2u);
  EXPECT_EQ(
    Utilities::MPI::sum(static_cast<std::uint64_t>(
                          run.accepted.physiology.owned_leaf_terminals.size()),
                        MPI_COMM_WORLD),
    3u);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<std::uint64_t>(
                                  run.owned_venous_terminals.size()),
                                MPI_COMM_WORLD),
            3u);
  const auto                 &after = run.accepted.network;
  std::array<unsigned int, 3> claims{{0, 0, 0}};
  for (const auto &[id, terminal] : after.terminals)
    if (terminal.owner == rank())
      {
        const auto entry =
          std::find(territories.begin(), territories.end(), terminal.territory);
        EXPECT_NE(entry, territories.end());
        if (entry == territories.end())
          continue;
        const unsigned int index = entry - territories.begin();
        ++claims[index];
        EXPECT_DOUBLE_EQ(terminal.demand, 1.);
        const auto position = after.nodes.at(terminal.node).position;
        if (index < 2)
          EXPECT_LT(position.distance(end[index]), 1.e-13);
        else
          EXPECT_NEAR(position.distance(target), offset, 1.e-13);
      }
  for (unsigned int count : claims)
    EXPECT_EQ(Utilities::MPI::sum(count, MPI_COMM_WORLD), 1u);
  for (const auto &[id, vessel] : after.vessels)
    if (vessel.owner == rank())
      {
        const auto a = after.nodes.at(vessel.nodes[0]).position;
        const auto b = after.nodes.at(vessel.nodes[1]).position;
        EXPECT_GE(std::min(a[0], b[0]), .5);
        EXPECT_GE(a.distance(b) / hcoarse + 1.e-13, .1);
        EXPECT_TRUE(std::isfinite(vessel.radii[0]));
        EXPECT_GT(vessel.radii[0], 0.);
        if (id == bad)
          {
            EXPECT_EQ(vessel.nodes, native.vessels.at(bad).nodes);
            EXPECT_EQ(vessel.radii, native.vessels.at(bad).radii);
            EXPECT_LT(a.distance(start[0]), 1.e-13);
            EXPECT_LT(b.distance(end[0]), 1.e-13);
          }
        if (id == good)
          {
            EXPECT_LT(b.distance(.5 * (start[1] + end[1])), 1.e-13);
          }
      }
}
