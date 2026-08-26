// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/coupling/particle_coupling.h>

#include <array>
#include <cstdint>
#include <vector>

using namespace dealii;
using namespace ImmersX;


TEST(DistributedLiftedQuadrature,
     MPI_CrossPartitionSourceStencilExchange) // NOLINT
{
  const MPI_Comm comm = MPI_COMM_WORLD;
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(comm), 2u);

  parallel::distributed::Triangulation<3> target_tria(comm);
  GridGenerator::hyper_cube(target_tria, 0., 1.);
  target_tria.refine_global(2);

  std::vector<double> local_center;
  for (const auto &cell : target_tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const auto center = cell->center();
        local_center      = {center[0], center[1], center[2]};
        break;
      }
  ASSERT_EQ(local_center.size(), 3u);

  const auto centers = Utilities::MPI::all_gather(comm, local_center);
  ASSERT_EQ(centers.size(), 2u);
  ASSERT_EQ(centers[0].size(), 3u);
  ASSERT_EQ(centers[1].size(), 3u);

  const unsigned int rank = Utilities::MPI::this_mpi_process(comm);
  RepresentationQuadraturePoint<3, double> source_point;
  const auto                              &opposite_center = centers[1 - rank];
  source_point.point =
    Point<3>(opposite_center[0], opposite_center[1], opposite_center[2]);
  source_point.representative_point = source_point.point;
  source_point.weight               = 1.;
  source_point.stable_id            = 100 + rank;
  source_point.dof_indices          = {rank};
  source_point.basis_values         = {1.};

  ParticleCouplingParameters<3>  parameters("/Cross partition test/");
  DistributedLiftedQuadrature<3> distributed(parameters);
  MappingQ1<3>                   mapping;
  distributed.initialize(target_tria, mapping, {source_point});

  ASSERT_EQ(distributed.target_stencils().size(), 1u);
  const auto &stencil = distributed.target_stencils().begin()->second;
  EXPECT_EQ(stencil.stable_id, 100 + (1 - rank));
  EXPECT_EQ(stencil.source_rank, 1 - rank);
  EXPECT_EQ(stencil.source_dof_indices,
            (std::vector<types::global_dof_index>{1 - rank}));

  Vector<double> source_values(1);
  source_values[0]         = rank + 1.;
  const auto target_values = distributed.values_on_target(source_values);
  ASSERT_EQ(target_values.size(), 1u);
  EXPECT_DOUBLE_EQ(target_values.begin()->second, 2. - rank);

  std::map<types::particle_index, double> target_contribution;
  target_contribution.emplace(stencil.stable_id, 10. + rank);
  Vector<double> transposed(1);
  transposed = 0.;
  distributed.add_transpose_to_source(target_contribution, transposed);
  EXPECT_DOUBLE_EQ(transposed[0], 11. - rank);
}
