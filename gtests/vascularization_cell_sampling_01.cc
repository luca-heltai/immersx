// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "../source/vascularization/growth_cell_sampling.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
using namespace ImmersX::Vascularization::internal;

TEST(VascularGeneration,
     BOTH_CellMixturePreservesPhysicalVolumeDensityAcrossOwners)
{
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(bg,
                                            {2, 1, 1},
                                            Point<3>(),
                                            Point<3>(2., 1., 1.));
  GridTools::transform(
    [](const Point<3> &p) {
      return Point<3>(p[0] <= 1. ? p[0] : 3. * p[0] - 2., p[1], p[2]);
    },
    bg);
  MappingQ1<3> mapping;
  for (double beta : {0., 3., -3.})
    {
      GrowthWork            work;
      BackgroundCellSampler sampler(bg, mapping, beta, MPI_COMM_WORLD, &work);
      EXPECT_EQ(
        Utilities::MPI::sum(work[GrowthOperation::background_cell_visits],
                            MPI_COMM_WORLD),
        2u);
      const auto      initial = work.counts;
      std::mt19937_64 choice(17 + rank);
      std::mt19937    physical(71 + rank);
      unsigned int    count = 0, left = 0;
      double          sum   = 0.;
      bool            valid = true;
      try
        {
          for (unsigned int i = 0; i < 4096; ++i)
            if (const auto p = sampler.sample(choice, physical))
              {
                ++count;
                left += (*p)[0] < 1.;
                sum += (*p)[0];
                EXPECT_GE((*p)[0], 0.);
                EXPECT_LE((*p)[0], 4.);
                EXPECT_GE((*p)[1], 0.);
                EXPECT_LE((*p)[1], 1.);
                EXPECT_GE((*p)[2], 0.);
                EXPECT_LE((*p)[2], 1.);
              }
        }
      catch (const std::exception &error)
        {
          ADD_FAILURE() << error.what();
          valid = false;
        }
      ASSERT_EQ(Utilities::MPI::min(unsigned(valid), MPI_COMM_WORLD), 1u);
      const auto n     = Utilities::MPI::sum(count, MPI_COMM_WORLD);
      const auto first = Utilities::MPI::sum(left, MPI_COMM_WORLD);
      ASSERT_GE(n, 4096u);
      // Volumes are 1 and 3: cell masses are V, 1, V^2 respectively.
      const double probability = beta == 0. ? .25 : beta == 3. ? .5 : .1;
      const double sigma = std::sqrt(n * probability * (1. - probability));
      EXPECT_NEAR(first, n * probability, 6. * sigma);
      const double expected = .5 * probability + 2.5 * (1. - probability);
      EXPECT_NEAR(Utilities::MPI::sum(sum, MPI_COMM_WORLD) / n,
                  expected,
                  12. / std::sqrt(double(n)));
      EXPECT_EQ(work.counts, initial);
    }
}

TEST(VascularGeneration,
     BOTH_NativeCellSamplerUsesPhysicalRatherThanReferenceDensity)
{
  const auto       rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  Triangulation<3> serial;
  GridGenerator::hyper_cube(serial);
  // z=w(1+3uv): volume=7/4, physical centroid=(4/7,4/7,1).
  serial.begin_active()->vertex(7)[2] = 4.;
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<3> bg(MPI_COMM_WORLD);
  bg.create_triangulation(description);
  MappingQ1<3>          mapping;
  BackgroundCellSampler sampler(bg, mapping, 3., MPI_COMM_WORLD);
  std::mt19937_64       choice(31 + rank);
  std::mt19937          physical(47 + rank);
  unsigned int          count = 0;
  Tensor<1, 3>          sum;
  bool                  valid = true;
  try
    {
      for (unsigned int i = 0; i < 16384; ++i)
        if (const auto p = sampler.sample(choice, physical))
          {
            ++count;
            sum += *p;
            EXPECT_GE((*p)[2], 0.);
            EXPECT_LE((*p)[2], 1. + 3. * (*p)[0] * (*p)[1] + 1e-10);
          }
    }
  catch (const std::exception &error)
    {
      ADD_FAILURE() << error.what();
      valid = false;
    }
  ASSERT_EQ(Utilities::MPI::min(unsigned(valid), MPI_COMM_WORLD), 1u);
  const auto n = Utilities::MPI::sum(count, MPI_COMM_WORLD);
  ASSERT_EQ(n, 16384u);
  const double expected[] = {4. / 7., 4. / 7., 1.};
  for (unsigned int d = 0; d < 3; ++d)
    EXPECT_NEAR(Utilities::MPI::sum(sum[d], MPI_COMM_WORLD) / n,
                expected[d],
                3. * (d == 2 ? 4. : 1.) / std::sqrt(double(n)));
}

TEST(VascularGeneration, BOTH_NativeCellSamplerHandlesTetrahedronAndEmptyRanks)
{
  const auto       rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  Triangulation<3> serial;
  GridGenerator::reference_cell(serial, ReferenceCells::Tetrahedron);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<3> bg(MPI_COMM_WORLD);
  bg.create_triangulation(description);
  MappingFE<3>          mapping(FE_SimplexP<3>(1));
  BackgroundCellSampler sampler(bg, mapping, 3., MPI_COMM_WORLD);
  std::mt19937_64       choice(53 + rank);
  std::mt19937          physical(61 + rank);
  unsigned int          count = 0;
  Tensor<1, 3>          sum;
  bool                  valid = true;
  try
    {
      for (unsigned int i = 0; i < 4096; ++i)
        if (const auto p = sampler.sample(choice, physical))
          {
            ++count;
            sum += *p;
            for (unsigned int d = 0; d < 3; ++d)
              EXPECT_GE((*p)[d], -1e-11);
            EXPECT_LE((*p)[0] + (*p)[1] + (*p)[2], 1. + 1e-11);
          }
    }
  catch (const std::exception &error)
    {
      ADD_FAILURE() << error.what();
      valid = false;
    }
  ASSERT_EQ(Utilities::MPI::min(unsigned(valid), MPI_COMM_WORLD), 1u);
  const auto n = Utilities::MPI::sum(count, MPI_COMM_WORLD);
  ASSERT_EQ(n, 4096u);
  for (unsigned int d = 0; d < 3; ++d)
    EXPECT_NEAR(Utilities::MPI::sum(sum[d], MPI_COMM_WORLD) / n,
                .25,
                3. / std::sqrt(double(n)));
}

TEST(VascularGeneration, BOTH_CellSamplerRejectsInvalidControlsCollectively)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  MappingQ1<3> mapping;
  EXPECT_THROW(BackgroundCellSampler(bg,
                                     mapping,
                                     std::numeric_limits<double>::infinity(),
                                     MPI_COMM_WORLD),
               ExceptionBase);
  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
    {
      EXPECT_THROW(BackgroundCellSampler(bg,
                                         mapping,
                                         Utilities::MPI::this_mpi_process(
                                           MPI_COMM_WORLD) == 0 ?
                                           0. :
                                           3.,
                                         MPI_COMM_WORLD),
                   ExceptionBase);
    }
}
