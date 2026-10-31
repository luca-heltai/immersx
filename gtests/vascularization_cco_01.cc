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
    p.terminal_count_multiplier_factor = "none";
    p.length_scale                     = "meter";
    p.minimum_length                   = 0.;
    p.density_beta                     = 0.;
    p.coverage.service_radius          = 0.;
    p.venous.terminal_separation       = 0.;

    p.seed_position      = Point<3>(.2, .5, .5);
    p.seed_direction     = Point<3>(1., 0., 0.);
    p.seed_radius        = .01;
    p.seed_length        = .3;
    p.viscosity          = .004;
    p.terminal_count     = 16;
    p.optimization_steps = 5;
    p.candidate_count    = 6;
    p.sample_budget      = 1000;
    p.clearance          = .001;
  }

  // Independent line-line stationary point plus all four constrained edges.
  double
  oracle_distance(Point<3> a, Point<3> b, Point<3> c, Point<3> d)
  {
    const auto u = b - a, v = d - c, w = a - c;
    const auto point_segment = [](Point<3> p, Point<3> x, Point<3> y) {
      const auto   delta = y - x;
      const double t =
        std::clamp((p - x) * delta / delta.norm_square(), 0., 1.);
      return p.distance(x + t * delta);
    };
    double      result = std::min({point_segment(a, c, d),
                                   point_segment(b, c, d),
                                   point_segment(c, a, b),
                                   point_segment(d, a, b)});
    long double aa = 0., bb = 0., cc = 0., dd = 0., ee = 0.;
    for (unsigned int k = 0; k < 3; ++k)
      {
        aa += static_cast<long double>(u[k]) * u[k];
        bb += static_cast<long double>(u[k]) * v[k];
        cc += static_cast<long double>(v[k]) * v[k];
        dd += static_cast<long double>(u[k]) * w[k];
        ee += static_cast<long double>(v[k]) * w[k];
      }
    const auto det = aa * cc - bb * bb;
    if (det > 0.)
      {
        const double s = (bb * ee - cc * dd) / det,
                     t = (aa * ee - bb * dd) / det;
        if (s >= 0. && s <= 1. && t >= 0. && t <= 1.)
          result = std::min(result, (a + s * u).distance(c + t * v));
      }
    return result;
  }
} // namespace

TEST(VascularCCO, TreeTopologyAndMurrayHaveIndependentOracles)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.refine_global(2);
  Parameters p;
  controls(p);
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  const auto &n = result.network;
  EXPECT_EQ(n.terminals.size(), p.terminal_count);
  EXPECT_EQ(n.nodes.size(), 2 * p.terminal_count);
  EXPECT_EQ(n.vessels.size(), 2 * p.terminal_count - 1);
  EXPECT_EQ(result.accepted_insertions, p.terminal_count - 1);
  std::map<Id, std::vector<Id>> children;
  for (const auto &[id, e] : n.vessels)
    children[e.nodes[0]].push_back(id);
  std::set<Id>                    visited;
  std::function<unsigned int(Id)> leaves = [&](Id id) {
    EXPECT_TRUE(visited.insert(id).second);
    const auto &e   = n.vessels.at(id);
    const auto &out = children[e.nodes[1]];
    if (out.empty())
      return 1u;
    EXPECT_EQ(out.size(), 2u);
    double       sum   = 0.;
    unsigned int count = 0;
    for (Id child : out)
      {
        count += leaves(child);
        sum +=
          std::pow(n.vessels.at(child).radii[0], p.arterial_murray_exponent);
      }
    EXPECT_NEAR(sum / std::pow(e.radii[1], p.arterial_murray_exponent),
                1.,
                2.e-13);
    return count;
  };
  EXPECT_EQ(leaves(0), p.terminal_count);
  EXPECT_EQ(visited.size(), n.vessels.size());
  EXPECT_DOUBLE_EQ(n.vessels.at(0).radii[0], p.seed_radius);
  const double terminal_radius =
    p.seed_radius *
    std::pow(1. / p.terminal_count, 1. / p.arterial_murray_exponent);
  for (const auto &[id, e] : n.vessels)
    if (children[e.nodes[1]].empty())
      EXPECT_NEAR(e.radii[0] / terminal_radius, 1., 2.e-14);
}

TEST(VascularCCO, WholeTreeWallsAnglesAndSkewClearances)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  controls(p);
  p.terminal_count = 24;
  p.random_seed    = 71;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  const auto &n = result.network;
  for (auto a = n.vessels.begin(); a != n.vessels.end(); ++a)
    {
      const auto &e = a->second;
      for (Id node : e.nodes)
        for (unsigned int k = 0; k < 3; ++k)
          {
            EXPECT_GT(n.nodes.at(node).position[k], e.radii[0] + p.clearance);
            EXPECT_LT(n.nodes.at(node).position[k],
                      1. - e.radii[0] - p.clearance);
          }
      for (auto b = n.vessels.begin(); b != a; ++b)
        {
          const auto &f        = b->second;
          bool        incident = false;
          for (unsigned int i = 0; i < 2; ++i)
            for (unsigned int j = 0; j < 2; ++j)
              if (e.nodes[i] == f.nodes[j])
                {
                  incident     = true;
                  const auto u = n.nodes.at(e.nodes[1 - i]).position -
                                 n.nodes.at(e.nodes[i]).position;
                  const auto v = n.nodes.at(f.nodes[1 - j]).position -
                                 n.nodes.at(f.nodes[j]).position;
                  EXPECT_LE(u * v / (u.norm() * v.norm()),
                            std::cos(p.minimum_angle * numbers::PI / 180.) +
                              1.e-13);
                }
          if (!incident)
            EXPECT_GT(oracle_distance(n.nodes.at(e.nodes[0]).position,
                                      n.nodes.at(e.nodes[1]).position,
                                      n.nodes.at(f.nodes[0]).position,
                                      n.nodes.at(f.nodes[1]).position),
                      e.radii[0] + f.radii[0] + p.clearance - 1.e-13);
        }
    }
}

TEST(VascularCCO, ReproducibleForFixedSeed)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  controls(p);
  p.terminal_count = 8;
  const auto a = generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  const auto b = generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(a.validation.valid());
  ASSERT_TRUE(b.validation.valid());
  EXPECT_EQ(a.sampled_points, b.sampled_points);
  for (const auto &[id, node] : a.network.nodes)
    EXPECT_EQ(node.position, b.network.nodes.at(id).position);
  for (const auto &[id, e] : a.network.vessels)
    {
      EXPECT_EQ(e.nodes, b.network.vessels.at(id).nodes);
      EXPECT_EQ(e.radii, b.network.vessels.at(id).radii);
    }
}

TEST(VascularCCO, DistinguishesRadiusIncompatibilityAndSearchBudget)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  controls(p);
  p.minimum_radius = .009;
  const auto incompatible =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(incompatible.validation.valid());
  EXPECT_EQ(incompatible.sampled_points, 0u);
  p.minimum_radius  = 0.;
  p.terminal_weight = "0";
  p.sample_budget   = 17;
  const auto exhausted =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  EXPECT_FALSE(exhausted.validation.valid());
  EXPECT_EQ(exhausted.sampled_points, 17u);
  EXPECT_EQ(exhausted.network.terminals.size(), 1u);
  EXPECT_DOUBLE_EQ(exhausted.network.vessels.at(0).radii[0], p.seed_radius);
}

TEST(VascularCCO, SpatialWeightsAndMixedObjective)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  controls(p);
  p.terminal_count           = 10;
  p.terminal_weight          = "x<0.5";
  p.dissipation_weight       = .4;
  p.reference_flow           = 2.e-9;
  p.arterial_murray_exponent = 2.7;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  for (std::size_t i = 1; i < result.territories.size(); ++i)
    EXPECT_LT(result.territories[i].position[0], .5);
}

TEST(VascularCCO, OptimizationReducesIndependentlyIntegratedVolume)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  Parameters p;
  controls(p);
  p.terminal_count     = 2;
  p.optimization_steps = 0;
  const auto initial =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  p.optimization_steps = 10;
  const auto optimized =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(initial.validation.valid());
  ASSERT_TRUE(optimized.validation.valid());
  EXPECT_EQ(initial.territories.at(1).position,
            optimized.territories.at(1).position);
  double v0 = 0., v1 = 0.;
  for (const auto &[id, e] : initial.network.vessels)
    v0 += volume(initial.network, id);
  for (const auto &[id, e] : optimized.network.vessels)
    v1 += volume(optimized.network, id);
  EXPECT_LT(v1, v0);
}

TEST(VascularCCO,
     LocalRefinementChangesTerminalDensityWithoutChangingBackground)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.refine_global(1);
  for (const auto &cell : background.active_cell_iterators())
    if (cell->center()[0] < .5)
      cell->set_refine_flag();
  background.execute_coarsening_and_refinement();
  const auto cells = background.n_active_cells();
  Parameters p;
  controls(p);
  p.terminal_count          = 32;
  unsigned int uniform_left = 0, adaptive_left = 0;
  for (unsigned int seed : {3u, 7u, 17u})
    for (double beta : {0., 3.})
      {
        p.random_seed  = seed;
        p.density_beta = beta;
        const auto result =
          generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
        ASSERT_TRUE(result.validation.valid());
        for (std::size_t i = 1; i < result.territories.size(); ++i)
          if (result.territories[i].position[0] < .5)
            {
              if (beta == 0.)
                ++uniform_left;
              else if (beta == 3.)
                ++adaptive_left;
            }
      }
  EXPECT_GT(adaptive_left, uniform_left);
  EXPECT_EQ(background.n_active_cells(), cells);
}
