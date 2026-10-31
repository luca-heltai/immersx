// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_description.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <limits>

using namespace dealii;
using namespace ImmersX::Vascularization;

namespace
{
  void
  configure(Parameters &p)
  {
    p.seed_position  = Point<3>(.2, .5, .5);
    p.seed_direction = Point<3>(1., 0., 0.);
    p.seed_radius    = .001;
    p.seed_length    = .15;
    p.viscosity      = .004;
    p.random_seed    = 23;
  }

  void
  expect_count(const GenerationResult &result, unsigned int expected)
  {
    ASSERT_TRUE(result.validation.valid());
    const auto   rank      = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    unsigned int terminals = 0;
    for (const auto &[id, terminal] : result.network.terminals)
      terminals += terminal.owner == rank;
    EXPECT_EQ(Utilities::MPI::sum(terminals, MPI_COMM_WORLD), expected);
    EXPECT_EQ(result.accepted_insertions, expected - 1);
    for (const auto &statistics : result.growth_statistics)
      EXPECT_EQ(statistics.requested_terminals, expected);
  }
} // namespace

TEST(VascularGeneration, TerminalCountMultiplierParameterSelection)
{
  Parameters p("/Terminal multiplier parsing/");
  EXPECT_EQ(p.terminal_count_multiplier_factor, "background cell count");
  ParameterAcceptor::prm.parse_input_from_string(
    "subsection Terminal multiplier parsing\n"
    " subsection Sampling\n"
    "  set Terminal count multiplier factor = background vertex count\n"
    " end\nend\n");
  EXPECT_EQ(p.terminal_count_multiplier_factor, "background vertex count");
  EXPECT_THROW(ParameterAcceptor::prm.parse_input_from_string(
                 "subsection Terminal multiplier parsing\n"
                 " subsection Sampling\n"
                 "  set Terminal count multiplier factor = unknown\n"
                 " end\nend\n"),
               ExceptionBase);
}

TEST(VascularGeneration, BOTH_TerminalCountMultiplierUsesGlobalCounts)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {2, 1, 1},
                                            Point<3>(),
                                            Point<3>(1., 1., 1.));
  Parameters p;
  configure(p);
  p.terminal_count = 2;
  // Two cubes share a face: 12 distinct vertices, not 16 cell vertices.
  for (const auto &[factor, count] :
       std::vector<std::pair<std::string, unsigned int>>{
         {"none", 2},
         {"background cell count", 4},
         {"background vertex count", 24}})
    {
      p.terminal_count_multiplier_factor = factor;
      expect_count(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD),
                   count);
      EXPECT_EQ(p.terminal_count, 2u);
      EXPECT_EQ(p.terminal_count_multiplier_factor, factor);
    }
}

TEST(VascularGeneration, BOTH_TerminalCountMultiplierFollowsLocalRefinement)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {2, 1, 1},
                                            Point<3>(),
                                            Point<3>(1., 1., 1.));
  for (const auto &cell : background.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[0] < .5)
      cell->set_refine_flag();
  background.execute_coarsening_and_refinement();
  // Splitting one of the two cubes adds 7 active cells and 19 vertices.
  ASSERT_EQ(background.n_global_active_cells(), 9u);
  Parameters p;
  configure(p);
  p.terminal_count_multiplier_factor = "background cell count";
  expect_count(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD), 9);
  p.terminal_count_multiplier_factor = "background vertex count";
  expect_count(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD), 31);
}

TEST(VascularGeneration, BOTH_TerminalCountMultiplierTetrahedronAndEmptyRanks)
{
  Triangulation<3> serial;
  GridGenerator::reference_cell(serial, ReferenceCells::Tetrahedron);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<3> background(MPI_COMM_WORLD);
  background.create_triangulation(description);
  Parameters p;
  configure(p);
  p.seed_position = Point<3>(.15, .15, .15);
  p.seed_length   = .1;
  const MappingFE<3> mapping(FE_SimplexP<3>(1));
  p.terminal_count_multiplier_factor = "background cell count";
  expect_count(generate(background, mapping, {}, p, MPI_COMM_WORLD), 1);
  p.terminal_count_multiplier_factor = "background vertex count";
  expect_count(generate(background, mapping, {}, p, MPI_COMM_WORLD), 4);
}

TEST(VascularGeneration,
     BOTH_TerminalCountMultiplierKeepsAbsoluteCalibrationLimits)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background);
  Parameters p;
  configure(p);
  p.terminal_count_multiplier_factor = "background vertex count";
  p.calibration.minimum_terminals    = 2;
  p.calibration.maximum_terminals    = 3;
  p.calibration.minimum_coverage     = .1;
  p.coverage.service_radius          = 2.;
  const auto result =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  expect_count(result, 3);
  EXPECT_EQ(result.calibration.status, CalibrationStatus::converged);
  ASSERT_EQ(result.calibration.attempts.size(), 1u);
  EXPECT_EQ(result.calibration.attempts[0].terminal_count, 3u);
  EXPECT_EQ(p.terminal_count, 1u);
  EXPECT_EQ(p.calibration.maximum_terminals, 3u);
}

TEST(VascularGeneration, BOTH_TerminalCountMultiplierRejectsInvalidControls)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background);
  Parameters p;
  configure(p);
  p.terminal_count_multiplier_factor = "unknown";
  EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD),
               ExceptionBase);
  p.terminal_count_multiplier_factor = "background vertex count";
  p.terminal_count                   = std::numeric_limits<unsigned int>::max();
  EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD),
               ExceptionBase);
  p.terminal_count = 0;
  EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD),
               ExceptionBase);
}

TEST(VascularGeneration,
     MPI_TerminalCountMultiplierDisagreementFailsCollectively)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background);
  Parameters p;
  configure(p);
  ASSERT_GE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    p.terminal_count_multiplier_factor = "none";
  // The numeric factor is one in both modes; the mode must still agree.
  EXPECT_THROW(generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD),
               ExceptionBase);
}
