// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <cmath>

#include "../source/vascularization/growth_batch.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
using namespace ImmersX::Vascularization::internal;
namespace
{
  void
  scaled_seed(Parameters &p)
  {
    p.terminal_count_multiplier_factor = "none";
    p.seed_position                    = Point<3>(.1, .5, .5);
    p.seed_direction                   = Point<3>(1, 0, 0);
    p.seed_radius                      = .001;
    p.viscosity                        = .004;
  }
  std::string
  messages(const GenerationResult &r)
  {
    std::string result;
    for (const auto &issue : r.validation.issues)
      result += issue.message + "\n";
    return result;
  }
  void
  refined_half(parallel::distributed::Triangulation<3> &bg)
  {
    GridGenerator::subdivided_hyper_rectangle(bg,
                                              {2, 1, 1},
                                              Point<3>(),
                                              Point<3>(1, 1, 1));
    for (const auto &cell : bg.active_cell_iterators())
      if (cell->is_locally_owned() && cell->center()[0] < .5)
        cell->set_refine_flag();
    bg.execute_coarsening_and_refinement();
  }
  Edge
  edge(Id id, Point<3> a, Point<3> b, double radius = .005)
  {
    return {id, {{id + 1, id + 2}}, {{a, b}}, radius};
  }
  unsigned int
  vote(BackgroundGeometry      &geometry,
       const Parameters        &p,
       const std::vector<Edge> &edges)
  {
    std::vector<BatchProposal> proposed;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1)
      {
        BatchProposal proposal;
        proposal.id      = 0;
        proposal.changed = edges;
        proposed.push_back(proposal);
      }
    const auto result =
      select_growth_batch(geometry, {}, proposed, p, MPI_COMM_WORLD, [](Id) {
        return 0.;
      });
    unsigned int accepted = 0;
    for (const auto &[id, valid] : result.local)
      accepted += valid;
    return Utilities::MPI::sum(accepted, MPI_COMM_WORLD);
  }
} // namespace
TEST(VascularGeneration, MeshAdaptiveParameterDefaults)
{
  Parameters p("/Mesh adaptive defaults/");
  EXPECT_EQ(p.length_scale, "local mesh size");
  EXPECT_EQ(p.terminal_count, 1u);
  EXPECT_EQ(p.terminal_count_multiplier_factor, "background cell count");
  EXPECT_DOUBLE_EQ(p.seed_length, 1.);
  EXPECT_DOUBLE_EQ(p.minimum_length, .1);
  EXPECT_DOUBLE_EQ(p.density_beta, 3.);
  EXPECT_DOUBLE_EQ(p.clearance, 0.);
  EXPECT_DOUBLE_EQ(p.coverage.service_radius, 1.);
  EXPECT_DOUBLE_EQ(p.minimum_radius, 0.);
  EXPECT_DOUBLE_EQ(p.venous.terminal_separation, .25);
  EXPECT_DOUBLE_EQ(p.discretization.target_factor, 1.);
  EXPECT_DOUBLE_EQ(p.discretization.lower_factor, .5);
  EXPECT_DOUBLE_EQ(p.discretization.upper_factor, 2.);
  EXPECT_DOUBLE_EQ(p.calibration.volume_fraction, 0.);
  EXPECT_DOUBLE_EQ(p.calibration.minimum_coverage, 0.);
  EXPECT_EQ(p.calibration.maximum_terminals, 0u);
  for (const auto &policy :
       {"meter", "local mesh size", "min mesh size", "max mesh size"})
    {
      ParameterAcceptor::prm.parse_input_from_string(
        "subsection Mesh adaptive defaults\n set Length scale = " +
        std::string(policy) + "\nend\n");
      EXPECT_EQ(p.length_scale, policy);
    }
  EXPECT_THROW(
    ParameterAcceptor::prm.parse_input_from_string(
      "subsection Mesh adaptive defaults\n set Length scale = cell size\nend\n"),
    ExceptionBase);
}
TEST(VascularGeneration, BOTH_GenerationUsesEachConfiguredLengthPolicy)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg, 0., 4.);
  bg.refine_global(1);
  for (const auto &policy :
       {"meter", "local mesh size", "min mesh size", "max mesh size"})
    {
      Parameters p;
      scaled_seed(p);
      p.length_scale  = policy;
      p.seed_position = Point<3>(.2, 2, 2);
      p.seed_length   = .2;
      const auto r    = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
      ASSERT_TRUE(r.validation.valid()) << messages(r);
      const double h = std::string(policy) == "meter" ? 1. : 2. * std::sqrt(3.);
      double       local_length = 0., local_volume = 0.;
      for (const auto &[id, v] : r.network.vessels)
        if (v.owner == Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
          {
            local_length +=
              r.network.nodes.at(v.nodes[0])
                .position.distance(r.network.nodes.at(v.nodes[1]).position);
            local_volume += volume(r.network, id);
            EXPECT_DOUBLE_EQ(v.radii[0], .001);
            EXPECT_DOUBLE_EQ(v.radii[1], .001);
            EXPECT_NEAR(r.normalized_geometry.at(id).normalized_length,
                        .2,
                        1.e-12);
          }
      EXPECT_NEAR(Utilities::MPI::sum(local_length, MPI_COMM_WORLD),
                  .2 * h,
                  1.e-12);
      EXPECT_NEAR(Utilities::MPI::sum(local_volume, MPI_COMM_WORLD),
                  numbers::PI * 1.e-6 * .2 * h,
                  1.e-16);
      EXPECT_DOUBLE_EQ(p.seed_length, .2);
      EXPECT_EQ(p.terminal_count, 1u);
      EXPECT_EQ(r.base_terminal_count, 1u);
      EXPECT_EQ(r.initial_terminal_count, 1u);
      EXPECT_EQ(r.requested_terminal_count, 1u);
      EXPECT_EQ(r.effective_terminal_count, 1u);
    }
}
TEST(VascularGeneration, BOTH_RefinementHalvesThePhysicalSeedWithoutRetuning)
{
  for (unsigned int level : {2u, 3u})
    {
      parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
      GridGenerator::hyper_cube(bg);
      bg.refine_global(level);
      Parameters p;
      scaled_seed(p);
      const auto r = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
      ASSERT_TRUE(r.validation.valid()) << messages(r);
      const double h      = std::sqrt(3.) / std::pow(2., level);
      double       length = 0., total_resistance = 0.;
      for (const auto &[id, v] : r.network.vessels)
        if (v.owner == Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
          {
            length +=
              r.network.nodes.at(v.nodes[0])
                .position.distance(r.network.nodes.at(v.nodes[1]).position);
            total_resistance += resistance(r.network, id);
            EXPECT_DOUBLE_EQ(v.radii[0], .001);
            EXPECT_NEAR(r.normalized_geometry.at(id).normalized_length,
                        1.,
                        1.e-12);
          }
      EXPECT_NEAR(Utilities::MPI::sum(length, MPI_COMM_WORLD), h, 1.e-12);
      EXPECT_NEAR(Utilities::MPI::sum(total_resistance, MPI_COMM_WORLD) /
                    (8. * .004 * h / (numbers::PI * 1.e-12)),
                  1.,
                  1.e-12);
    }
}
TEST(VascularGeneration, BOTH_MinimumLengthUsesTheWholeCrossedBranch)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  refined_half(bg);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  p.minimum_radius = 0.;
  p.minimum_length = .7;
  const auto branch =
    edge(Id{1} << 57, Point<3>(.1, .25, .25), Point<3>(.9, .25, .25));
  // .4/.75+.4/1.5=.8. Midpoint-only and coarse-end-only conversions disagree.
  EXPECT_EQ(vote(geometry, p, {branch}), 1u);
  p.minimum_length = .9;
  EXPECT_EQ(vote(geometry, p, {branch}), 0u);
}
TEST(VascularGeneration, BOTH_MinimumRadiusChecksTheCoarsePartOfTheBranch)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  refined_half(bg);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  p.minimum_length = 0.;
  p.minimum_radius = .009;
  const auto branch =
    edge(Id{1} << 57, Point<3>(.1, .25, .25), Point<3>(.9, .25, .25), .015);
  EXPECT_EQ(vote(geometry, p, {branch}), 1u);
  p.minimum_radius = .011;
  EXPECT_EQ(vote(geometry, p, {branch}), 0u);
}
TEST(VascularGeneration, BOTH_ClearanceIsSymmetricAndDependsOnEachWholeBranch)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  refined_half(bg);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  p.minimum_length  = 0.;
  p.clearance       = .08;
  const Id   base   = Id{1} << 57;
  const auto fine_a = edge(base, Point<3>(.2, .35, .5), Point<3>(.4, .35, .5));
  const auto crossing_a =
    edge(base, Point<3>(.2, .35, .5), Point<3>(.8, .35, .5));
  const auto b = edge(base + 10, Point<3>(.2, .44, .5), Point<3>(.4, .44, .5));
  EXPECT_EQ(vote(geometry, p, {fine_a, b}), 1u);
  EXPECT_EQ(vote(geometry, p, {b, fine_a}), 1u);
  EXPECT_EQ(vote(geometry, p, {crossing_a, b}), 0u);
  EXPECT_EQ(vote(geometry, p, {b, crossing_a}), 0u);
}
TEST(VascularGeneration,
     BOTH_WallClearanceUsesTheBranchScaleRatherThanTheGlobalMaximum)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  refined_half(bg);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p;
  p.minimum_length = 0.;
  p.clearance      = .08;
  const Id base    = Id{1} << 57;
  EXPECT_EQ(vote(geometry,
                 p,
                 {edge(base, Point<3>(.08, .35, .5), Point<3>(.4, .35, .5))}),
            1u);
  EXPECT_EQ(vote(geometry,
                 p,
                 {edge(base, Point<3>(.08, .35, .5), Point<3>(.8, .35, .5))}),
            0u);
}
TEST(VascularGeneration,
     BOTH_MurrayCapacityConvertsTheMinimumRadiusBeforeRejecting)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  bg.refine_global(2);
  Parameters p;
  scaled_seed(p);
  p.seed_position  = Point<3>(.2, .5, .5);
  p.seed_length    = .3;
  p.minimum_length = .01;
  p.minimum_radius = .0015;
  p.terminal_count = 2;
  const auto r     = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(r.validation.valid()) << messages(r);
  EXPECT_EQ(r.effective_terminal_count, 2u);
  for (const auto &[id, m] : r.normalized_geometry)
    EXPECT_GE(m.minimum_normalized_radius, p.minimum_radius);
}

TEST(VascularGeneration,
     BOTH_VenousSeedAndPairUseTheirRespectiveAttachmentAndTerritoryScales)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  refined_half(bg);
  Parameters p;
  scaled_seed(p);
  p.seed_position                  = Point<3>(.35, .25, .25);
  p.seed_length                    = .1;
  p.minimum_length                 = .01;
  p.venous.enabled                 = true;
  p.venous.seed_position           = Point<3>(.8, .25, .25);
  p.venous.seed_radius             = .0015;
  p.reference_flow                 = 1.e-9;
  p.reference_conditions.pressures = {{0, 12000.}, {2, 4000.}};
  const auto r = generate(bg, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(r.validation.valid()) << messages(r);
  double coordinates[6]{0., 0., 0., 0., 0., 0.};
  for (const auto &[id, terminal] : r.network.terminals)
    if (terminal.owner == Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
      {
        const unsigned int offset = terminal.system == System::arterial ? 0 : 3;
        for (unsigned int d = 0; d < 3; ++d)
          coordinates[offset + d] =
            r.network.nodes.at(terminal.node).position[d];
      }
  Utilities::MPI::sum(coordinates, MPI_COMM_WORLD, coordinates);
  const Point<3> artery(coordinates[0], coordinates[1], coordinates[2]);
  const Point<3> vein(coordinates[3], coordinates[4], coordinates[5]);
  EXPECT_NEAR(artery[0], .425, 1.e-13);
  EXPECT_GT(vein[0], .5);
  EXPECT_NEAR(artery.distance(vein), .25 * .75, 1.e-13);
  unsigned int local_stems = 0;
  for (const auto &[id, v] : r.network.vessels)
    if (v.owner == Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) &&
        v.system == System::venous &&
        r.network.nodes.at(v.nodes[0]).port == Port::venous_outlet)
      {
        ++local_stems;
        EXPECT_NEAR(length(r.network, id), .1 * 1.5, 1.e-13);
        EXPECT_DOUBLE_EQ(v.radii[0], .0015);
        EXPECT_DOUBLE_EQ(v.radii[1], .0015);
      }
  EXPECT_EQ(Utilities::MPI::sum(local_stems, MPI_COMM_WORLD), 1u);
}
