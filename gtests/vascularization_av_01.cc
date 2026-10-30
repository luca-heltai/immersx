// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_description.h>

#include <boost/serialization/map.hpp>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <set>

#include "../source/vascularization/generation_pairs.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  void
  controls(Parameters &p)
  {
    p.seed_position                  = Point<3>(.2, .5, .5);
    p.seed_direction                 = Point<3>(1., 0., 0.);
    p.seed_radius                    = .003;
    p.seed_length                    = .2;
    p.viscosity                      = .004;
    p.terminal_count                 = 8;
    p.reference_flow                 = 2.e-9;
    p.tissue_demand                  = "1+x+2*y";
    p.arterial_murray_exponent       = 2.7;
    p.venous.enabled                 = true;
    p.venous.seed_position           = Point<3>(.8, .5, .5);
    p.venous.seed_radius             = .004;
    p.venous.murray_exponent         = 3.2;
    p.venous.terminal_separation     = .04;
    p.reference_conditions.pressures = {{0, 12000.}, {2, 4000.}};
  }
  std::string
  issues(const GenerationResult &r)
  {
    std::string s;
    for (const auto &i : r.validation.issues)
      s += i.message + "\n";
    return s;
  }
  void
  oracle(const GenerationResult &r,
         const Parameters       &p,
         const NetworkData      &initial = {})
  {
    ASSERT_TRUE(r.validation.valid()) << issues(r);
    ASSERT_TRUE(r.reference.validation.valid());
    ASSERT_EQ(r.terminal_pairs.size(), r.territories.size());
    EXPECT_TRUE(validate_terminal_pairs(r.network, r.terminal_pairs).valid());
    std::map<Id, Territory> territories;
    double                  demand = 0.;
    for (const auto &t : r.territories)
      {
        EXPECT_TRUE(territories.emplace(t.id, t).second);
        demand += t.demand;
      }
    std::set<Id> paired_terminals, paired_territories;
    double       total = 0.;
    for (const auto &pair : r.terminal_pairs)
      {
        ASSERT_TRUE(paired_territories.insert(pair.territory).second);
        EXPECT_TRUE(paired_terminals.insert(pair.arterial).second);
        EXPECT_TRUE(paired_terminals.insert(pair.venous).second);
        const auto &a = r.network.terminals.at(pair.arterial),
                   &v = r.network.terminals.at(pair.venous);
        EXPECT_EQ(a.system, System::arterial);
        EXPECT_EQ(v.system, System::venous);
        const auto &pa = r.network.nodes.at(a.node).position,
                   &pv = r.network.nodes.at(v.node).position;
        const auto &t  = territories.at(pair.territory);
        EXPECT_EQ(pa, t.position);
        EXPECT_DOUBLE_EQ(a.demand, t.demand);
        EXPECT_DOUBLE_EQ(v.demand, t.demand);
        EXPECT_NEAR(pa.distance(pv) / p.venous.terminal_separation, 1., 1.e-12);
        EXPECT_GT(pair.resistance, 0.);
        const double q = r.reference.territories.at(pair.territory).flow;
        EXPECT_GT(q, 0.);
        total += q;
        const double drop =
          r.reference.pressures.at(a.node) - r.reference.pressures.at(v.node);
        EXPECT_NEAR(drop / (q * pair.resistance), 1., 1.e-10);
        if (p.reference_flow > 0.)
          EXPECT_NEAR(q / (p.reference_flow * t.demand / demand), 1., 1.e-8);
      }
    EXPECT_EQ(paired_territories.size(), territories.size());
    if (p.reference_flow > 0.)
      EXPECT_NEAR(total / p.reference_flow, 1., 1.e-8);
    for (const auto &[id, e] : r.network.vessels)
      {
        const double q = r.reference.vessels.at(id).flow;
        if (e.system == System::arterial)
          EXPECT_GE(q, 0.);
        else if (e.system == System::venous)
          EXPECT_LE(q, 0.);
      }
    // Independent rooted traversal of new physical vessels, separate from CCO
    // caches.
    std::map<Id, std::vector<Id>> children;
    std::map<Id, Id>              incoming;
    for (const auto &[id, e] : r.network.vessels)
      if (!initial.vessels.count(id))
        {
          children[e.nodes[0]].push_back(id);
          incoming[e.nodes[1]] = id;
        }
    std::map<Id, double> leaf_demand;
    for (const auto &[id, t] : r.network.terminals)
      if (territories.count(t.territory))
        leaf_demand[t.node] = t.demand;
    std::function<double(Id, double)> visit = [&](Id id, double gamma) {
      const auto &e = r.network.vessels.at(id);
      double      d = 0.;
      if (children[e.nodes[1]].empty())
        d = leaf_demand.at(e.nodes[1]);
      else
        {
          double sum = 0.;
          EXPECT_EQ(children[e.nodes[1]].size(), 2u);
          for (Id child : children[e.nodes[1]])
            {
              d += visit(child, gamma);
              sum += std::pow(r.network.vessels.at(child).radii[0], gamma);
            }
          EXPECT_NEAR(std::pow(e.radii[0], gamma) / sum, 1., 1.e-12);
        }
      return d;
    };
    double arterial_flow = 0., venous_flow = 0.;
    for (const auto &[id, e] : r.network.vessels)
      if (!initial.vessels.count(id) && !incoming.count(e.nodes[0]))
        {
          const double gamma = e.system == System::arterial ?
                                 p.arterial_murray_exponent :
                                 p.venous.murray_exponent;
          const double d     = visit(id, gamma);
          if (p.reference_flow > 0.)
            {
              EXPECT_NEAR(r.target_attachment_flows.at(id) /
                            (p.reference_flow * d / demand),
                          1.,
                          1.e-12);
              EXPECT_NEAR(std::abs(r.reference.vessels.at(id).flow) /
                            r.target_attachment_flows.at(id),
                          1.,
                          1.e-8);
            }
          if (e.system == System::arterial)
            arterial_flow += r.reference.vessels.at(id).flow;
          else if (e.system == System::venous)
            venous_flow -= r.reference.vessels.at(id).flow;
        }
    EXPECT_NEAR(arterial_flow / venous_flow, 1., 1.e-9);
    // Independently minimize distance over the four segment boundaries and
    // the stationary interior point. Check every cross-system vessel pair.
    const auto point_segment =
      [](const Point<3> &x, const Point<3> &a, const Point<3> &b) {
        const auto   u = b - a;
        const double t = std::clamp(((x - a) * u) / (u * u), 0., 1.);
        return x.distance_square(Point<3>(a + t * u));
      };
    for (const auto &[ai, a] : r.network.vessels)
      if (a.system == System::arterial)
        for (const auto &[vi, v] : r.network.vessels)
          if (v.system == System::venous)
            {
              const auto p0         = r.network.nodes.at(a.nodes[0]).position,
                         p1         = r.network.nodes.at(a.nodes[1]).position;
              const auto q0         = r.network.nodes.at(v.nodes[0]).position,
                         q1         = r.network.nodes.at(v.nodes[1]).position;
              double       distance = std::min({point_segment(p0, q0, q1),
                                                point_segment(p1, q0, q1),
                                                point_segment(q0, p0, p1),
                                                point_segment(q1, p0, p1)});
              const auto   u = p1 - p0, w = q1 - q0, z = p0 - q0;
              const double aa = u * u, b = u * w, c = w * w, d = u * z,
                           e = w * z, det = aa * c - b * b;
              if (det > 0.)
                {
                  const double t = (b * e - c * d) / det,
                               s = (aa * e - b * d) / det;
                  if (t >= 0. && t <= 1. && s >= 0. && s <= 1.)
                    distance =
                      std::min(distance,
                               Point<3>(p0 + t * u)
                                 .distance_square(Point<3>(q0 + s * w)));
                }
              const double radii = std::max(a.radii[0], a.radii[1]) +
                                   std::max(v.radii[0], v.radii[1]) +
                                   p.clearance;
              EXPECT_GT(distance, radii * radii);
            }
    double volume = 0.;
    for (const auto &[id, e] : r.network.vessels)
      volume += ImmersX::Vascularization::volume(r.network, id);
    EXPECT_NEAR(r.tissue.inside_vessel_volume / volume, 1., 1.e-12);
  }
} // namespace
TEST(VascularAV, SharedWeightedTerritoriesAndPositiveDerivedMicrovascularBeds)
{
  Triangulation<3> bg;
  GridGenerator::hyper_cube(bg);
  Parameters p;
  controls(p);
  p.dissipation_weight = .3;
  const auto r         = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(r.validation.valid()) << issues(r);
  ASSERT_EQ(r.territories.size(), 8u);
  ASSERT_EQ(r.network.terminals.size(), 16u);
  EXPECT_EQ(r.network.nodes.at(0).port, Port::arterial_inlet);
  EXPECT_EQ(r.network.nodes.at(2).port, Port::venous_outlet);
  for (const auto &t : r.territories)
    EXPECT_NEAR(t.demand, 1. + t.position[0] + 2. * t.position[1], 1.e-13);
  oracle(r, p);
  const auto again = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(again.validation.valid()) << issues(again);
  EXPECT_EQ(r.sampled_points, again.sampled_points);
  for (const auto &[id, n] : r.network.nodes)
    EXPECT_EQ(n.position, again.network.nodes.at(id).position);
}
TEST(VascularAV, ADirectBedHasTheAnalyticSeriesFlowAndAddsNoVolume)
{
  Triangulation<3> bg;
  GridGenerator::hyper_cube(bg);
  Parameters p;
  controls(p);
  p.terminal_count                  = 1;
  p.reference_flow                  = 0.;
  p.venous.microvascular_resistance = 4.e12;
  const auto r = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(r.validation.valid()) << issues(r);
  ASSERT_EQ(r.network.vessels.size(), 2u);
  double resistance_sum = p.venous.microvascular_resistance;
  for (const auto &[id, e] : r.network.vessels)
    resistance_sum += resistance(r.network, id);
  EXPECT_NEAR(r.reference.territories.begin()->second.flow /
                (8000. / resistance_sum),
              1.,
              1.e-12);
  EXPECT_DOUBLE_EQ(r.terminal_pairs.at(0).resistance,
                   p.venous.microvascular_resistance);
  EXPECT_TRUE(r.target_terminal_flows.empty());
  oracle(r, p);
  p.reference_flow        = 4.e-9;
  const auto incompatible = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(incompatible.validation.valid());
}
TEST(VascularAV, NonpositivePressureHeadAndUnreachablePartnersCannotBeAdopted)
{
  Triangulation<3> bg;
  GridGenerator::hyper_cube(bg);
  Parameters p;
  controls(p);
  p.terminal_count                 = 1;
  p.reference_conditions.pressures = {{0, 4000.}, {2, 12000.}};
  auto r = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(r.validation.valid());
  p.reference_conditions.pressures = {{0, 12000.}, {2, 4000.}};
  p.venous.terminal_separation     = 3.;
  r = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(r.validation.valid());
  EXPECT_TRUE(r.terminal_pairs.empty());
  p.venous.terminal_separation = 0.;
  EXPECT_THROW(p.validate(), ExceptionBase);
  p.venous.terminal_separation = .04;
  p.venous.murray_exponent     = 0.;
  EXPECT_THROW(p.validate(), ExceptionBase);
}
TEST(VascularAV, TwoInletsAndTwoVenousOutletsKeepAllImportedPhysicalAttributes)
{
  Triangulation<3> bg;
  GridGenerator::hyper_cube(bg);
  NetworkData input;
  for (unsigned int i = 0; i < 4; ++i)
    {
      const bool artery = i < 2;
      const Id   a = 10 * i, b = a + 1;
      input.nodes[a].position =
        Point<3>(artery ? .05 : .95, i % 2 ? .7 : .3, .5);
      input.nodes[b].position =
        Point<3>(artery ? .15 : .85, i % 2 ? .7 : .3, .5);
      input.nodes[a].port = artery ? Port::arterial_inlet : Port::venous_outlet;
      Vessel e;
      e.nodes     = {{a, b}};
      e.system    = artery ? System::arterial : System::venous;
      e.radii     = {{.004, .003}};
      e.viscosity = .004;
      e.physical_properties["density"] = 1050. + i;
      input.vessels[100 + i]           = e;
      input.terminals[200 + i]         = {b, 300 + i, e.system, 0, 5. + i};
    }
  Parameters p;
  controls(p);
  p.seed_direction                 = Point<3>();
  p.seed_length                    = .15;
  p.selected_terminals             = {200, 201};
  p.venous.selected_terminals      = {202, 203};
  p.reference_conditions.pressures = {{0, 12000.},
                                      {10, 12000.},
                                      {20, 4000.},
                                      {30, 4000.}};
  const auto r = generate(bg, MappingQ1<3>(), input, p, MPI_COMM_WORLD);
  ASSERT_TRUE(r.validation.valid()) << issues(r);
  ASSERT_EQ(r.attachment_roots.size(), 4u);
  for (const auto &[id, n] : input.nodes)
    {
      EXPECT_EQ(n.position, r.network.nodes.at(id).position);
      EXPECT_EQ(n.port, r.network.nodes.at(id).port);
      EXPECT_EQ(n.provenance, r.network.nodes.at(id).provenance);
    }
  for (const auto &[id, e] : input.vessels)
    {
      const auto &actual = r.network.vessels.at(id);
      EXPECT_EQ(e.nodes, actual.nodes);
      EXPECT_EQ(e.radii, actual.radii);
      EXPECT_EQ(e.system, actual.system);
      EXPECT_EQ(e.viscosity, actual.viscosity);
      EXPECT_EQ(e.physical_properties, actual.physical_properties);
    }
  for (const auto &[id, t] : input.terminals)
    EXPECT_EQ(r.network.terminals.count(id), 0u);
  oracle(r, p, input);
  for (Id id : {200, 201, 202, 203})
    EXPECT_DOUBLE_EQ(r.network.vessels.at(r.attachment_roots.at(id)).radii[0],
                     .003);
}
TEST(VascularAV, ExplicitExteriorOrDuplicateVenousAttachmentsAreRejected)
{
  Triangulation<3> bg;
  GridGenerator::hyper_cube(bg);
  NetworkData input;
  input.nodes[0].position  = Point<3>(.1, .3, .5);
  input.nodes[0].port      = Port::arterial_inlet;
  input.nodes[1].position  = Point<3>(.2, .3, .5);
  input.nodes[10].position = Point<3>(1.2, .7, .5);
  input.nodes[10].port     = Port::venous_outlet;
  input.nodes[11].position = Point<3>(1.1, .7, .5);
  Vessel a;
  a.nodes             = {{0, 1}};
  a.radii             = {{.003, .003}};
  a.viscosity         = .004;
  input.vessels[0]    = a;
  a.nodes             = {{10, 11}};
  a.system            = System::venous;
  input.vessels[10]   = a;
  input.terminals[40] = {1, 70, System::arterial, 0, 1.};
  input.terminals[50] = {11, 80, System::venous, 0, 1.};
  Parameters p;
  controls(p);
  p.terminal_count            = 1;
  p.venous.selected_terminals = {50};
  auto r = generate(bg, MappingQ1<3>(), input, p, MPI_COMM_WORLD);
  EXPECT_FALSE(r.validation.valid());
  EXPECT_TRUE(r.terminal_pairs.empty());
  EXPECT_EQ(r.network.nodes.at(11).position, input.nodes.at(11).position);
  input.nodes[10].position    = Point<3>(.9, .7, .5);
  input.nodes[11].position    = Point<3>(.8, .7, .5);
  p.venous.selected_terminals = {50, 50};
  r = generate(bg, MappingQ1<3>(), input, p, MPI_COMM_WORLD);
  EXPECT_FALSE(r.validation.valid());
  EXPECT_TRUE(r.terminal_pairs.empty());
}
TEST(VascularAV, CalibratedCountsKeepTheSamePhysicalPortIdsAndUseDualCoverage)
{
  Triangulation<3> bg;
  GridGenerator::hyper_cube(bg);
  Parameters p;
  controls(p);
  p.terminal_count = 3;
  const auto small = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(small.validation.valid()) << issues(small);
  p.terminal_count = 8;
  const auto large = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(large.validation.valid()) << issues(large);
  EXPECT_EQ(small.network.nodes.at(2).position,
            large.network.nodes.at(2).position);
  p.calibration.volume_fraction   = small.tissue.volume_fraction;
  p.calibration.minimum_terminals = 3;
  p.calibration.maximum_terminals = 3;
  const auto calibrated = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(calibrated.validation.valid()) << issues(calibrated);
  EXPECT_EQ(calibrated.calibration.status, CalibrationStatus::converged);
  EXPECT_EQ(calibrated.terminal_pairs.size(), 3u);
  oracle(calibrated, p);
}
TEST(VascularAV, VenousPlanarOutletIsClippedAndCrossSystemCollisionIsRejected)
{
  Triangulation<3> bg;
  GridGenerator::hyper_cube(bg);
  Parameters p;
  controls(p);
  p.terminal_count                 = 5;
  p.venous.seed_position           = Point<3>(1., .8, .5);
  p.venous.seed_inward_port_normal = Point<3>(-1., 0., 0.);
  const auto r = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(r.validation.valid()) << issues(r);
  EXPECT_EQ(r.network.nodes.at(2).inward_port_normal,
            p.venous.seed_inward_port_normal);
  oracle(r, p);
  p.venous.seed_position           = p.seed_position;
  p.venous.seed_inward_port_normal = Point<3>();
  const auto collision = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(collision.validation.valid());
  EXPECT_TRUE(collision.terminal_pairs.empty());
}
TEST(VascularAV, CalibrationRequiresCoverageByBothSystems)
{
  Triangulation<3> bg;
  GridGenerator::hyper_cube(bg);
  Parameters p;
  controls(p);
  p.terminal_count            = 1;
  p.coverage.service_radius   = .3;
  p.coverage.quadrature_order = 3;
  const auto plain = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(plain.validation.valid()) << issues(plain);
  ASSERT_GT(plain.tissue.arterial_coverage, plain.tissue.dual_coverage);
  ASSERT_GT(plain.tissue.dual_coverage, 0.);
  p.calibration.minimum_terminals = 1;
  p.calibration.maximum_terminals = 1;
  p.calibration.attempt_budget    = 1;
  p.calibration.minimum_coverage =
    .5 * (plain.tissue.arterial_coverage + plain.tissue.dual_coverage);
  const auto failed = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(failed.validation.valid());
  EXPECT_EQ(failed.calibration.status, CalibrationStatus::exhausted);
  ASSERT_EQ(failed.calibration.attempts.size(), 1u);
  EXPECT_GE(failed.tissue.arterial_coverage, p.calibration.minimum_coverage);
  EXPECT_LT(failed.tissue.dual_coverage, p.calibration.minimum_coverage);
}

namespace
{
  GenerationResult
  distributed_pair_fixture()
  {
    const auto         rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto         size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const unsigned int active = std::min(size, 2u);
    GenerationResult   r;
    for (unsigned int i = 0; i < 2; ++i)
      {
        const Id           base    = (Id{1} << 60) + 100 * i;
        const unsigned int a_owner = i % active, v_owner = (i + 1) % active;
        const double       demand = 1. + 2. * i;
        if (rank == a_owner)
          {
            r.network.nodes[base]     = {Point<3>(0., i, 0.),
                                         rank,
                                         Port::arterial_inlet};
            r.network.nodes[base + 1] = {Point<3>(1., i, 0.), rank};
            Vessel v;
            v.nodes                        = {{base, base + 1}};
            v.radii                        = {{.001, .001}};
            v.viscosity                    = .004;
            v.owner                        = rank;
            r.network.vessels[base]        = v;
            r.network.terminals[base + 10] = {
              base + 1, base + 20, System::arterial, rank, demand};
            r.territories.push_back(
              {base + 20, Point<3>(1., i, 0.), rank, demand});
            r.target_terminal_flows[base + 10] = demand * 1.e-9;
          }
        if (rank == v_owner)
          {
            r.network.nodes[base + 2] = {Point<3>(3., i, 0.),
                                         rank,
                                         Port::venous_outlet};
            r.network.nodes[base + 3] = {Point<3>(2., i, 0.), rank};
            Vessel v;
            v.nodes                        = {{base + 2, base + 3}};
            v.radii                        = {{.0015, .0015}};
            v.viscosity                    = .004;
            v.system                       = System::venous;
            v.owner                        = rank;
            r.network.vessels[base + 1]    = v;
            r.network.terminals[base + 11] = {
              base + 3, base + 20, System::venous, rank, demand};
            r.target_terminal_flows[base + 11] = demand * 1.e-9;
          }
      }
    // This pre-existing component must retain its independent tissue pressure.
    if (rank == 0)
      {
        r.network.nodes[500] = {Point<3>(0., 4., 0.),
                                rank,
                                Port::arterial_inlet};
        r.network.nodes[501] = {Point<3>(1., 4., 0.), rank};
        Vessel v;
        v.nodes                  = {{500, 501}};
        v.radii                  = {{.001, .001}};
        v.viscosity              = .004;
        r.network.vessels[500]   = v;
        r.network.terminals[600] = {501, 700, System::arterial, rank, 999.};
      }
    return r;
  }
  void
  pairing_controls(Parameters &p)
  {
    p.venous.enabled = true;
    p.reference_flow = 4.e-9;
    // Pressure input can reside on a rank that owns no vascular entity.
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) + 1 ==
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
      {
        for (unsigned int i = 0; i < 2; ++i)
          {
            const Id base                          = (Id{1} << 60) + 100 * i;
            p.reference_conditions.pressures[base] = 12000.;
            p.reference_conditions.pressures[base + 2] = 4000.;
          }
        p.reference_conditions.pressures[500] = 9000.;
        p.reference_conditions.pressures[501] = 5000.;
      }
  }
} // namespace
TEST(VascularAV, BOTH_DistributedPairingDerivesBedsFromRemoteEndpointPressures)
{
  Parameters p;
  pairing_controls(p);
  auto       r              = distributed_pair_fixture();
  const auto original_nodes = r.network.nodes;
  ImmersX::Vascularization::internal::pair_territories(r, p, MPI_COMM_WORLD);
  ASSERT_TRUE(r.validation.valid()) << issues(r);
  ASSERT_TRUE(r.reference.validation.valid());
  EXPECT_EQ(r.terminal_pairs.size(), r.territories.size());
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(
                                  r.terminal_pairs.size()),
                                MPI_COMM_WORLD),
            2u);
  const double ra = 8. * .004 / (numbers::PI * std::pow(.001, 4));
  const double rv = 8. * .004 / (numbers::PI * std::pow(.0015, 4));
  for (const auto &pair : r.terminal_pairs)
    {
      const unsigned int i        = (pair.territory - (Id{1} << 60) - 20) / 100;
      const double       desired  = (1. + 2. * i) * 1.e-9;
      const double       expected = 8000. / desired - ra - rv;
      EXPECT_NEAR(pair.resistance / expected, 1., 1.e-12);
      EXPECT_NEAR(r.reference.territories.at(pair.territory).flow / desired,
                  1.,
                  1.e-10);
      if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
        {
          EXPECT_EQ(r.network.terminals.count(pair.venous), 0u);
        }
    }
  for (const auto &[id, v] : r.network.vessels)
    {
      const double q = r.reference.vessels.at(id).flow;
      if (id == 500)
        {
          EXPECT_NEAR(q / (4000. / ra), 1., 1.e-12);
        }
      else if (v.system == System::arterial)
        {
          EXPECT_GT(q, 0.);
        }
      else if (v.system == System::venous)
        {
          EXPECT_LT(q, 0.);
        }
    }
  EXPECT_EQ(r.network.nodes.size(), original_nodes.size());
  for (const auto &[id, n] : original_nodes)
    EXPECT_EQ(r.network.nodes.at(id).position, n.position);
}
TEST(VascularAV, BOTH_DistributedSuppliedBedsCheckActualFlowAgainstDemand)
{
  Parameters p;
  pairing_controls(p);
  p.venous.microvascular_resistance = 4.e12;
  p.reference_flow                  = 0.;
  auto r                            = distributed_pair_fixture();
  ImmersX::Vascularization::internal::pair_territories(r, p, MPI_COMM_WORLD);
  ASSERT_TRUE(r.validation.valid()) << issues(r);
  const double resistance_sum =
    4.e12 + 8. * .004 / numbers::PI *
              (1. / std::pow(.001, 4) + 1. / std::pow(.0015, 4));
  for (const auto &pair : r.terminal_pairs)
    {
      EXPECT_DOUBLE_EQ(pair.resistance, 4.e12);
      EXPECT_NEAR(r.reference.territories.at(pair.territory).flow /
                    (8000. / resistance_sum),
                  1.,
                  1.e-12);
    }
  p.reference_flow = 4.e-9;
  r                = distributed_pair_fixture();
  ImmersX::Vascularization::internal::pair_territories(r, p, MPI_COMM_WORLD);
  EXPECT_FALSE(r.validation.valid());
  EXPECT_TRUE(r.reference.validation.valid());
}
TEST(VascularAV,
     BOTH_DistributedPairingRejectsRemoteMissingOrConflictingTerritories)
{
  Parameters p;
  pairing_controls(p);
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const Id   base = Id{1} << 60;
  for (unsigned int mode = 0; mode < 7; ++mode)
    {
      auto r = distributed_pair_fixture();
      if (mode == 0)
        r.network.terminals.erase(base + 11);
      else if (mode == 1 && r.network.terminals.count(base + 11))
        r.network.terminals.at(base + 11).demand = 2.;
      else if (mode == 2 && rank == 0)
        r.territories.push_back(r.territories.front());
      else if (mode == 3 && rank == 0)
        r.target_terminal_flows.at(base + 10) = 0.;
      else if (mode == 4 && r.network.terminals.count(base + 11))
        r.network.terminals[base + 12] = r.network.terminals.at(base + 11);
      else if (mode == 5 && rank == 0)
        r.network.nodes.at(base + 1).owner =
          Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      else if (mode == 6 &&
               rank + 1 == Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        p.reference_conditions.inflows[base + 1] = -1.e-9;
      ImmersX::Vascularization::internal::pair_territories(r,
                                                           p,
                                                           MPI_COMM_WORLD);
      EXPECT_FALSE(r.validation.valid()) << mode;
    }
}

namespace
{
  void
  distributed_av_oracle(const GenerationResult &r, const Parameters &p)
  {
    const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    ASSERT_TRUE(r.validation.valid()) << issues(r);
    ASSERT_TRUE(r.reference.validation.valid());
    const auto global_count = [&](std::size_t n) {
      return Utilities::MPI::sum(static_cast<Id>(n), MPI_COMM_WORLD);
    };
    EXPECT_EQ(global_count(r.territories.size()), p.terminal_count);
    EXPECT_EQ(global_count(r.terminal_pairs.size()), p.terminal_count);
    double demand = 0.;
    for (const auto &t : r.territories)
      {
        EXPECT_EQ(t.owner, rank);
        demand += t.demand;
      }
    demand = Utilities::MPI::sum(demand, MPI_COMM_WORLD);
    std::map<Id, double> local_demand;
    for (const auto &t : r.territories)
      local_demand[t.id] = t.demand;
    for (const auto &pair : r.terminal_pairs)
      {
        EXPECT_GT(pair.resistance, 0.);
        EXPECT_NE(pair.arterial, pair.venous);
        const double q = r.reference.territories.at(pair.territory).flow;
        EXPECT_GT(q, 0.);
        EXPECT_NEAR(q / (p.reference_flow * local_demand.at(pair.territory) /
                         demand),
                    1.,
                    1.e-8);
      }
    std::array<Id, 2> counts{{0, 0}};
    double            a_flow = 0., v_flow = 0.;
    for (const auto &[id, t] : r.network.terminals)
      if (t.owner == rank)
        {
          const double q = r.reference.terminal_outflows.at(id);
          if (t.system == System::arterial)
            {
              ++counts[0];
              a_flow += q;
            }
          else if (t.system == System::venous)
            {
              ++counts[1];
              v_flow -= q;
            }
        }
    EXPECT_EQ(global_count(counts[0]), p.terminal_count);
    EXPECT_EQ(global_count(counts[1]), p.terminal_count);
    EXPECT_NEAR(Utilities::MPI::sum(a_flow, MPI_COMM_WORLD) / p.reference_flow,
                1.,
                1.e-8);
    EXPECT_NEAR(Utilities::MPI::sum(v_flow, MPI_COMM_WORLD) / p.reference_flow,
                1.,
                1.e-8);
    for (const auto &[id, vessel] : r.network.vessels)
      if (vessel.owner == rank)
        {
          const auto &flow = r.reference.vessels.at(id);
          EXPECT_NEAR(flow.resistance / resistance(r.network, id), 1., 1.e-12);
          const double dp = r.reference.pressures.at(vessel.nodes[0]) -
                            r.reference.pressures.at(vessel.nodes[1]);
          EXPECT_NEAR(dp / (flow.resistance * flow.flow), 1., 1.e-7);
          if (vessel.system == System::arterial)
            {
              EXPECT_GT(flow.flow, 0.);
            }
          else if (vessel.system == System::venous)
            {
              EXPECT_LT(flow.flow, 0.);
            }
        }
  }
} // namespace
TEST(VascularAV, BOTH_PublicDistributedStartupFromOneRealArterialAndVenousPort)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {4, 2, 2},
                                            Point<3>(),
                                            Point<3>(1., 1., 1.));
  Parameters p;
  controls(p);
  p.seed_position        = Point<3>(.15, .25, .5);
  p.seed_length          = .6;
  p.venous.seed_position = Point<3>(.85, .75, .5);
  p.terminal_count       = 4;
  p.coverage             = {2., 2};
  const auto r           = generate(background, MappingQ1<3>(), {}, p, comm);
  distributed_av_oracle(r, p);
  ASSERT_TRUE(r.validation.valid()) << issues(r);
  unsigned int a_ports = 0, v_ports = 0;
  for (const auto &[id, n] : r.network.nodes)
    if (n.owner == rank)
      {
        if (n.port == Port::arterial_inlet)
          {
            ++a_ports;
            EXPECT_EQ(id, 0u);
            EXPECT_EQ(n.position, p.seed_position);
          }
        else if (n.port == Port::venous_outlet)
          {
            ++v_ports;
            EXPECT_EQ(id, 2u);
            EXPECT_EQ(n.position, p.venous.seed_position);
          }
      }
  EXPECT_EQ(Utilities::MPI::sum(a_ports, comm), 1u);
  EXPECT_EQ(Utilities::MPI::sum(v_ports, comm), 1u);
  EXPECT_NEAR(r.tissue.dual_coverage, 1., 1.e-12);
  p.calibration.volume_fraction   = r.tissue.volume_fraction * 1.02;
  p.calibration.minimum_coverage  = .9;
  p.calibration.minimum_terminals = p.terminal_count;
  p.calibration.maximum_terminals = p.terminal_count;
  p.calibration.attempt_budget    = 1;
  const auto again = generate(background, MappingQ1<3>(), {}, p, comm);
  ASSERT_TRUE(again.validation.valid()) << issues(again);
  EXPECT_EQ(again.calibration.status, CalibrationStatus::converged);
  EXPECT_EQ(again.calibration.attempts.size(), 1u);
  EXPECT_GE(again.tissue.dual_coverage, .9);
  EXPECT_EQ(r.sampled_points, again.sampled_points);
  EXPECT_EQ(r.network.nodes.size(), again.network.nodes.size());
  for (const auto &[id, n] : r.network.nodes)
    EXPECT_EQ(n.position, again.network.nodes.at(id).position);
}
TEST(VascularAV, BOTH_PublicDistributedMultipleImportedOutletsPreserveInput)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {4, 2, 2},
                                            Point<3>(),
                                            Point<3>(1., 1., 1.));
  NetworkData initial;
  if (rank == 0)
    for (unsigned int i = 0; i < 4; ++i)
      {
        const bool   artery = i < 2;
        const Id     a = 10 * i, b = a + 1;
        const double y = artery ? (i % 2 ? .7 : .3) : .2 + .05 * (i - 2);
        const double z = artery ? .5 : .45 + .1 * (i - 2);
        initial.nodes[a].position = Point<3>(artery ? -.05 : 1.05, y, z);
        initial.nodes[b].position = Point<3>(artery ? .15 : .85, y, z);
        initial.nodes[a].port =
          artery ? Port::arterial_inlet : Port::venous_outlet;
        Vessel v;
        v.nodes     = {{a, b}};
        v.system    = artery ? System::arterial : System::venous;
        v.radii     = {{.004, .003}};
        v.viscosity = .004;
        v.physical_properties["density"] = 1050. + i;
        initial.vessels[100 + i]         = v;
        initial.terminals[200 + i]       = {b, 300 + i, v.system, 0, 5. + i};
      }
  Parameters p;
  controls(p);
  p.seed_direction                 = {};
  p.seed_length                    = .15;
  p.terminal_count                 = 2;
  p.selected_terminals             = {200, 201};
  p.venous.selected_terminals      = {202, 203};
  p.reference_conditions.pressures = {{0, 12000.},
                                      {10, 12000.},
                                      {20, 4000.},
                                      {30, 4000.}};
  const auto r = generate(background, MappingQ1<3>(), initial, p, comm);
  distributed_av_oracle(r, p);
  ASSERT_TRUE(r.validation.valid()) << issues(r);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<Id>(r.attachment_roots.size()),
                                comm),
            4u);
  std::map<Id, Id> roots;
  for (const auto &entries :
       Utilities::MPI::all_gather(comm, r.attachment_roots))
    for (const auto &[input, root] : entries)
      EXPECT_TRUE(roots.emplace(root, input).second);
  unsigned int fixed_nodes = 0;
  for (const auto &[id, node] : r.network.nodes)
    if (node.owner == rank && id < 32 && id % 10 <= 1)
      {
        const unsigned int i      = id / 10;
        const bool         artery = i < 2, port = id % 10 == 0;
        const double       y = artery ? (i % 2 ? .7 : .3) : .2 + .05 * (i - 2);
        const double       z = artery ? .5 : .45 + .1 * (i - 2);
        const double x = artery ? (port ? -.05 : .15) : (port ? 1.05 : .85);
        ++fixed_nodes;
        EXPECT_EQ(node.position, Point<3>(x, y, z));
        EXPECT_EQ(node.provenance, Provenance::imported);
        EXPECT_EQ(node.port,
                  port ? (artery ? Port::arterial_inlet : Port::venous_outlet) :
                         Port::none);
      }
  EXPECT_EQ(Utilities::MPI::sum(fixed_nodes, comm), 8u);
  unsigned int inherited = 0;
  for (const auto &[id, vessel] : r.network.vessels)
    if (vessel.owner == rank && roots.count(id))
      {
        const unsigned int i = roots.at(id) - 200;
        ++inherited;
        EXPECT_EQ(vessel.nodes[0], 10 * i + 1);
        EXPECT_EQ(vessel.radii, (std::array<double, 2>{{.003, .003}}));
        EXPECT_DOUBLE_EQ(vessel.viscosity, .004);
        EXPECT_DOUBLE_EQ(vessel.physical_properties.at("density"), 1050. + i);
        EXPECT_EQ(vessel.system, i < 2 ? System::arterial : System::venous);
      }
  EXPECT_EQ(Utilities::MPI::sum(inherited, comm), 4u);
  unsigned int imported = 0;
  for (const auto &[id, v] : r.network.vessels)
    if (v.owner == rank && id >= 100 && id < 104)
      {
        const unsigned int i = id - 100;
        ++imported;
        EXPECT_EQ(v.nodes, (std::array<Id, 2>{{10 * i, 10 * i + 1}}));
        EXPECT_EQ(v.radii, (std::array<double, 2>{{.004, .003}}));
        EXPECT_DOUBLE_EQ(v.viscosity, .004);
        EXPECT_DOUBLE_EQ(v.physical_properties.at("density"), 1050. + i);
        EXPECT_EQ(v.provenance, Provenance::imported);
      }
  EXPECT_EQ(Utilities::MPI::sum(imported, comm), 4u);
  for (const auto &[id, t] : r.network.terminals)
    EXPECT_FALSE(id >= 200 && id < 204);
  if (rank == 0)
    {
      EXPECT_EQ(initial.nodes.at(0).position, Point<3>(-.05, .3, .5));
      EXPECT_EQ(initial.terminals.size(), 4u);
    }
  p.venous.selected_terminals = {202, 202};
  auto rejected = generate(background, MappingQ1<3>(), initial, p, comm);
  EXPECT_FALSE(rejected.validation.valid());
  EXPECT_TRUE(rejected.terminal_pairs.empty());
  p.venous.selected_terminals = {202, 203};
  if (rank == 0)
    initial.nodes.at(21).position[0] = 1.1;
  rejected = generate(background, MappingQ1<3>(), initial, p, comm);
  EXPECT_FALSE(rejected.validation.valid());
  EXPECT_TRUE(rejected.terminal_pairs.empty());
}
TEST(VascularAV, BOTH_PublicArterioVenousTetrahedronWithEmptyBackgroundRanks)
{
  const auto       comm = MPI_COMM_WORLD;
  Triangulation<3> serial;
  GridGenerator::reference_cell(serial, ReferenceCells::Tetrahedron);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, comm);
  parallel::fullydistributed::Triangulation<3> background(comm);
  background.create_triangulation(description);
  Parameters p;
  controls(p);
  p.seed_position              = Point<3>(.15, .15, .15);
  p.seed_length                = .1;
  p.venous.seed_position       = Point<3>(.15, .4, .15);
  p.venous.terminal_separation = .03;
  p.terminal_count             = 1;
  p.coverage                   = {2., 2};
  const auto r =
    generate(background, MappingFE<3>(FE_SimplexP<3>(1)), {}, p, comm);
  distributed_av_oracle(r, p);
  ASSERT_TRUE(r.validation.valid()) << issues(r);
  EXPECT_NEAR(r.tissue.background_volume, 1. / 6., 1.e-13);
  EXPECT_NEAR(r.tissue.dual_coverage, 1., 1.e-12);
}

TEST(VascularAV, BOTH_PublicVenousStartupRejectsUnreachableOutletCollectively)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {4, 1, 1},
                                            Point<3>(),
                                            Point<3>(1., 1., 1.));
  Parameters p;
  controls(p);
  p.terminal_count       = 1;
  p.venous.seed_position = Point<3>(2., .5, .5);
  auto r                 = generate(background, MappingQ1<3>(), {}, p, comm);
  EXPECT_FALSE(r.validation.valid());
  EXPECT_TRUE(r.terminal_pairs.empty());
  p.venous.seed_position       = Point<3>(.8, .5, .5);
  p.venous.terminal_separation = 3.;
  r = generate(background, MappingQ1<3>(), {}, p, comm);
  EXPECT_FALSE(r.validation.valid());
  EXPECT_TRUE(r.terminal_pairs.empty());
  p.venous.terminal_separation = .04;
  p.selected_terminals         = {77};
  r = generate(background, MappingQ1<3>(), {}, p, comm);
  EXPECT_FALSE(r.validation.valid());
  EXPECT_TRUE(r.terminal_pairs.empty());
}
TEST(VascularAV, MPI_PublicVenousStartupRejectsMismatchedControls)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  ASSERT_GT(Utilities::MPI::n_mpi_processes(comm), 1u);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {4, 1, 1},
                                            Point<3>(),
                                            Point<3>(1., 1., 1.));
  Parameters p;
  controls(p);
  p.terminal_count = 1;
  if (rank == 0)
    p.venous.enabled = false;
  EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, comm),
               ExceptionBase);
  p.venous.enabled = true;
  if (rank == 0)
    p.venous.selected_terminals = {17};
  EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, comm),
               ExceptionBase);
}
