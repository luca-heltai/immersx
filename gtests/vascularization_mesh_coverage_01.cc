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
#include <immersx/vascularization/mesh_scale.h>
#include <immersx/vascularization/metrics.h>
using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  void
  refined_half(parallel::distributed::Triangulation<3> &background)
  {
    GridGenerator::subdivided_hyper_rectangle(background,
                                              {2, 1, 1},
                                              Point<3>(),
                                              Point<3>(1, 1, 1));
    for (const auto &cell : background.active_cell_iterators())
      if (cell->is_locally_owned() && cell->center()[0] < .5)
        cell->set_refine_flag();
    background.execute_coarsening_and_refinement();
  }
  NetworkData
  centre(const Point<3> &position)
  {
    NetworkData n;
    const auto  rank  = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto  owner = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1;
    if (rank == owner)
      {
        const Id id     = Id{1} << 57;
        n.nodes[id]     = {position, owner, Port::none, Provenance::imported};
        n.terminals[id] = {id, id + 1, System::arterial, owner, 1.};
      }
    return n;
  }
  TissueMetrics
  covered(const Triangulation<3> &bg,
          const NetworkData      &n,
          double                  c,
          const std::string      &policy)
  {
    const auto           scale = mesh_length_scale(bg, policy, MPI_COMM_WORLD);
    const MappingQ1<3>   mapping;
    DiscretizationResult d;
    return measure_tissue(bg, mapping, n, d, {c, 1}, MPI_COMM_WORLD, scale);
  }
} // namespace
TEST(VascularMetrics, BOTH_LocalCoverageUsesCoarseTissueReachFromAFineTerminal)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  refined_half(bg);
  const auto n = centre(Point<3>(.25, .5, .5));
  for (const auto &policy :
       {"meter", "local mesh size", "min mesh size", "max mesh size"})
    {
      const auto   m        = covered(bg, n, .6, policy);
      const double expected = std::string(policy) == "min mesh size" ? .5 : 1.;
      EXPECT_NEAR(m.arterial_coverage, expected, 1.e-13);
      EXPECT_DOUBLE_EQ(m.venous_coverage, 0.);
      EXPECT_DOUBLE_EQ(m.dual_coverage, 0.);
      EXPECT_NEAR(m.background_volume, 1., 1.e-13);
      EXPECT_NEAR(m.coverage_quadrature_volume, 1., 1.e-13);
    }
}
TEST(VascularMetrics,
     BOTH_LocalCoverageDoesNotUseCoarseTerminalReachInFineTissue)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  refined_half(bg);
  const auto n = centre(Point<3>(.75, .5, .5));
  for (const auto &policy :
       {"meter", "local mesh size", "min mesh size", "max mesh size"})
    {
      const auto   m        = covered(bg, n, .4, policy);
      const double expected = std::string(policy) == "max mesh size" ? .75 : .5;
      EXPECT_NEAR(m.arterial_coverage, expected, 1.e-13);
    }
}
TEST(VascularMetrics,
     MPI_LocalCoverageRejectsInconsistentScaleControlsCollectively)
{
  ASSERT_GE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  auto scale = mesh_length_scale(bg, "local mesh size", MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    scale.policy = LengthScalePolicy::meter;
  const auto n = centre(Point<3>(.5, .5, .5));
  EXPECT_THROW(
    measure_tissue(bg, MappingQ1<3>(), n, {}, {.6, 1}, MPI_COMM_WORLD, scale),
    ExceptionBase);
}

TEST(VascularMetrics,
     BOTH_LocalCoverageKeepsArterialVenousAndDualPhysicalUnions)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  refined_half(bg);
  auto       n     = centre(Point<3>(.25, .5, .5));
  const auto rank  = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto owner = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1;
  if (rank == owner)
    {
      const Id id     = (Id{1} << 57) + 2;
      n.nodes[id]     = {Point<3>(.75, .5, .5),
                         owner,
                         Port::none,
                         Provenance::imported};
      n.terminals[id] = {id, id + 1, System::venous, owner, 1.};
    }
  const auto m = covered(bg, n, .6, "local mesh size");
  EXPECT_NEAR(m.arterial_coverage, 1., 1.e-13);
  EXPECT_NEAR(m.venous_coverage, .5, 1.e-13);
  EXPECT_NEAR(m.dual_coverage, .5, 1.e-13);
  if (rank == owner)
    {
      const Id id     = (Id{1} << 57) + 4;
      n.nodes[id]     = {Point<3>(.25, .5, .5),
                         owner,
                         Port::none,
                         Provenance::imported};
      n.terminals[id] = {id, id + 1, System::arterial, owner, 1.};
    }
  const auto duplicated = covered(bg, n, .6, "local mesh size");
  EXPECT_DOUBLE_EQ(duplicated.arterial_coverage, m.arterial_coverage);
  EXPECT_DOUBLE_EQ(duplicated.venous_coverage, m.venous_coverage);
  EXPECT_DOUBLE_EQ(duplicated.dual_coverage, m.dual_coverage);
}

TEST(VascularMetrics, BOTH_LocalCoverageTetrahedronWithEmptyBackgroundRanks)
{
  Triangulation<3> serial;
  GridGenerator::reference_cell(serial, ReferenceCells::Tetrahedron);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<3> bg(MPI_COMM_WORLD);
  bg.create_triangulation(description);
  const MappingFE<3> mapping(FE_SimplexP<3>(1));
  const auto scale = mesh_length_scale(bg, "local mesh size", MPI_COMM_WORLD);
  const auto n     = centre(Point<3>(.25, .25, .25));
  const auto m =
    measure_tissue(bg, mapping, n, {}, {1.e-6, 1}, MPI_COMM_WORLD, scale);
  EXPECT_NEAR(m.background_volume, 1. / 6., 1.e-13);
  EXPECT_NEAR(m.coverage_quadrature_volume, 1. / 6., 1.e-13);
  EXPECT_NEAR(m.arterial_coverage, 1., 1.e-13);
}
