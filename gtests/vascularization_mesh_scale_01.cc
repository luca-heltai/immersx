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

#include <cmath>
#include <limits>

using namespace dealii;
using namespace ImmersX::Vascularization;

namespace
{
  NetworkData
  one_vessel(Point<3> a, Point<3> b, double r0 = .03, double r1 = .06)
  {
    NetworkData n;
    const auto  rank  = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto  owner = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1;
    if (rank == owner)
      {
        const Id base     = Id{1} << 58;
        n.nodes[base]     = {a, owner, Port::none, Provenance::imported};
        n.nodes[base + 1] = {b, owner, Port::none, Provenance::imported};
        Vessel vessel;
        vessel.nodes        = {{base, base + 1}};
        vessel.radii        = {{r0, r1}};
        vessel.viscosity    = .004;
        vessel.owner        = owner;
        n.vessels[base + 2] = vessel;
      }
    return n;
  }

  void
  refined_pair(parallel::distributed::Triangulation<3> &background)
  {
    GridGenerator::subdivided_hyper_rectangle(background,
                                              {2, 1, 1},
                                              Point<3>(),
                                              Point<3>(1., 1., 1.));
    for (const auto &cell : background.active_cell_iterators())
      if (cell->is_locally_owned() && cell->center()[0] < .5)
        cell->set_refine_flag();
    background.execute_coarsening_and_refinement();
  }
} // namespace

TEST(VascularGeneration, BOTH_LengthScalesUniformMeshAndRefinement)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background);
  const double h = std::sqrt(3.);
  for (const auto &policy :
       {"local mesh size", "min mesh size", "max mesh size"})
    {
      const auto scale = mesh_length_scale(background, policy, MPI_COMM_WORLD);
      EXPECT_NEAR(scale.mesh_minimum, h, 1.e-14);
      EXPECT_NEAR(scale.mesh_maximum, h, 1.e-14);
      EXPECT_NEAR(scale.value(h), h, 1.e-14);
      EXPECT_NEAR(scale.physical_length(.4, h), .4 * h, 1.e-14);
    }
  const auto meter = mesh_length_scale(background, "meter", MPI_COMM_WORLD);
  EXPECT_DOUBLE_EQ(meter.value(0.), 1.);
  EXPECT_DOUBLE_EQ(meter.physical_length(.4, 0.), .4);
  background.refine_global(1);
  const auto finer =
    mesh_length_scale(background, "local mesh size", MPI_COMM_WORLD);
  EXPECT_NEAR(finer.physical_length(.4, finer.mesh_minimum), .2 * h, 1.e-14);
}

TEST(VascularGeneration, BOTH_LengthScalesIntegrateTheWholeCrossedBranch)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  refined_pair(background);
  const auto network =
    one_vessel(Point<3>(.1, .25, .25), Point<3>(.9, .25, .25));
  for (const auto &policy :
       {"meter", "local mesh size", "min mesh size", "max mesh size"})
    {
      const auto scale = mesh_length_scale(background, policy, MPI_COMM_WORLD);
      EXPECT_NEAR(scale.mesh_minimum, .75, 1.e-14);
      EXPECT_NEAR(scale.mesh_maximum, 1.5, 1.e-14);
      const auto result = measure_scaled_vessels(
        background, MappingQ1<3>(), network, scale, MPI_COMM_WORLD);
      EXPECT_EQ(result.size(), network.vessels.size());
      for (const auto &[id, m] : result)
        {
          const double expected = std::string(policy) == "local mesh size" ?
                                    .4 / .75 + .4 / 1.5 :
                                    .8 / scale.value(1.);
          EXPECT_NEAR(m.normalized_length, expected, 2.e-13);
          EXPECT_NEAR(m.inside_length, .8, 2.e-13);
          EXPECT_DOUBLE_EQ(m.exterior_length, 0.);
          EXPECT_TRUE(m.complete);
          EXPECT_TRUE(m.has_normalized_part);
          if (std::string(policy) == "local mesh size")
            {
              EXPECT_NEAR(m.minimum_scale, .75, 1.e-14);
              EXPECT_NEAR(m.maximum_scale, 1.5, 1.e-14);
              EXPECT_NEAR(m.minimum_normalized_radius, .045 / 1.5, 1.e-13);
              EXPECT_NEAR(m.maximum_normalized_radius, .045 / .75, 1.e-13);
            }
          EXPECT_DOUBLE_EQ(network.vessels.at(id).radii[0], .03);
          EXPECT_DOUBLE_EQ(network.vessels.at(id).radii[1], .06);
        }
    }
}

TEST(VascularGeneration, BOTH_LengthScalesDoNotApplyTheFinestCellEverywhere)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  refined_pair(background);
  const auto scale =
    mesh_length_scale(background, "local mesh size", MPI_COMM_WORLD);
  const auto network =
    one_vessel(Point<3>(.6, .75, .75), Point<3>(.9, .75, .75));
  const auto result = measure_scaled_vessels(
    background, MappingQ1<3>(), network, scale, MPI_COMM_WORLD);
  for (const auto &[id, m] : result)
    {
      EXPECT_NEAR(m.normalized_length, .3 / 1.5, 1.e-13);
      EXPECT_NEAR(m.minimum_scale, 1.5, 1.e-14);
    }
}

TEST(VascularGeneration, BOTH_LengthScalesChooseTheFineSideOfAnMPIInterface)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  refined_pair(background);
  const auto scale =
    mesh_length_scale(background, "local mesh size", MPI_COMM_WORLD);
  const auto network = one_vessel(Point<3>(.5, .1, .25), Point<3>(.5, .9, .25));
  const auto result  = measure_scaled_vessels(
    background, MappingQ1<3>(), network, scale, MPI_COMM_WORLD);
  for (const auto &[id, m] : result)
    EXPECT_NEAR(m.normalized_length, .8 / .75, 2.e-13);
}

TEST(VascularGeneration, BOTH_LengthScalesKeepExteriorInputPhysicalAndExplicit)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background);
  const auto network = one_vessel(Point<3>(-.5, .5, .5), Point<3>(1.5, .5, .5));
  for (const auto &policy : {"meter", "local mesh size"})
    {
      const auto scale  = mesh_length_scale(background, policy, MPI_COMM_WORLD);
      const auto result = measure_scaled_vessels(
        background, MappingQ1<3>(), network, scale, MPI_COMM_WORLD);
      for (const auto &[id, m] : result)
        {
          EXPECT_NEAR(m.inside_length, 1., 1.e-13);
          EXPECT_NEAR(m.exterior_length, 1., 1.e-13);
          if (std::string(policy) == "local mesh size")
            {
              EXPECT_FALSE(m.complete);
              EXPECT_NEAR(m.normalized_length, 1. / std::sqrt(3.), 1.e-13);
            }
          else if (std::string(policy) == "meter")
            {
              EXPECT_TRUE(m.complete);
              EXPECT_NEAR(m.normalized_length, 2., 1.e-13);
            }
          EXPECT_DOUBLE_EQ(
            network.nodes.at(network.vessels.at(id).nodes[0]).position[0], -.5);
          EXPECT_DOUBLE_EQ(network.vessels.at(id).radii[0], .03);
        }
    }
}

TEST(VascularGeneration, BOTH_LengthScalesTetrahedronWithEmptyBackgroundRanks)
{
  Triangulation<3> serial;
  GridGenerator::reference_cell(serial, ReferenceCells::Tetrahedron);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<3> background(MPI_COMM_WORLD);
  background.create_triangulation(description);
  const MappingFE<3> mapping(FE_SimplexP<3>(1));
  const auto         scale =
    mesh_length_scale(background, "local mesh size", MPI_COMM_WORLD);
  EXPECT_NEAR(scale.mesh_minimum, std::sqrt(2.), 1.e-14);
  EXPECT_NEAR(scale.mesh_maximum, std::sqrt(2.), 1.e-14);
  const auto network = one_vessel(Point<3>(.1, .1, .1), Point<3>(.3, .1, .1));
  const auto result =
    measure_scaled_vessels(background, mapping, network, scale, MPI_COMM_WORLD);
  for (const auto &[id, m] : result)
    EXPECT_NEAR(m.normalized_length, .2 / std::sqrt(2.), 1.e-13);
}

TEST(VascularGeneration,
     BOTH_LengthScalesRejectInvalidModesAndUnrepresentableCoefficients)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background);
  EXPECT_THROW(mesh_length_scale(background, "unknown", MPI_COMM_WORLD),
               ExceptionBase);
  const auto scale =
    mesh_length_scale(background, "local mesh size", MPI_COMM_WORLD);
  EXPECT_THROW(scale.value(0.), ExceptionBase);
  EXPECT_DOUBLE_EQ(scale.physical_length(0., 0.), 0.);
  EXPECT_THROW(scale.physical_length(std::numeric_limits<double>::max(), 2.),
               ExceptionBase);
  background.clear();
  EXPECT_THROW(mesh_length_scale(background, "local mesh size", MPI_COMM_WORLD),
               ExceptionBase);
}

TEST(VascularGeneration, MPI_LengthScaleModeDisagreementFailsCollectively)
{
  ASSERT_GE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background);
  const std::string mode =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 ? "meter" :
                                                            "local mesh size";
  EXPECT_THROW(mesh_length_scale(background, mode, MPI_COMM_WORLD),
               ExceptionBase);
}
