// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
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
