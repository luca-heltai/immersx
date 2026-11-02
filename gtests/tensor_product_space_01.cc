// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "test_paths.h"

#ifdef DEAL_II_WITH_VTK

#  include <immersx/coupling/immersed_repartitioner.h>

using namespace ImmersX;
#  include <immersx/coupling/tensor_product_space.h>
#  include <immersx/io/utils.h>

using namespace dealii;

TEST(TensorProductSpace, GridGeneration) // NOLINT
{
  static constexpr int reduced_dim  = 1;
  static constexpr int dim          = 2;
  static constexpr int spacedim     = 3;
  static constexpr int n_components = 1;

  // Setup parameters
  TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components> params;
  params.reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/mstree_100.vtk");

  // Create the tensor product space
  TensorProductSpace<reduced_dim, dim, spacedim, n_components> tps(params);

  // Initialize the tensor product space
  tps.initialize();

  // Verify the reduced grid was created
  const DoFHandler<reduced_dim, spacedim> &dof_handler = tps.get_dof_handler();
  ASSERT_GT(dof_handler.n_dofs(), 0);

  // Get quadrature points positions and check they are not empty
  const auto &qpoints = tps.get_locally_owned_qpoints();
  ASSERT_FALSE(qpoints.empty());
}

TEST(TensorProductSpace, MPI_ImmersedGridPartitioning) // NOLINT
{
  static constexpr int reduced_dim  = 1;
  static constexpr int dim          = 2;
  static constexpr int spacedim     = 3;
  static constexpr int n_components = 1;

  // Setup parameters
  TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components> params;

  params.reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/mstree_100.vtk");

  // Create a background grid (hypercube)
  parallel::distributed::Triangulation<spacedim> background_tria(
    MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background_tria, -0.2, 1.2);
  background_tria.refine_global(2);

  // Create the tensor product space
  TensorProductSpace<reduced_dim, dim, spacedim, n_components> tps(params);

  // Set the make_reduced_grid function to read from a VTK file and use
  // ImmersedRepartitioner

  ImmersedRepartitioner<reduced_dim, spacedim> immersed_partitioner(
    background_tria);
  tps.set_partitioner = [&](auto &tria) {
    tria.set_partitioner(immersed_partitioner,
                         TriangulationDescription::Settings());
  };

  // Initialize the tensor product space
  tps.initialize();

  // Verify the reduced grid was created
  const DoFHandler<reduced_dim, spacedim> &dof_handler = tps.get_dof_handler();
  ASSERT_GT(dof_handler.n_dofs(), 0);

  // Check how many ranks we have:
  const auto n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  // Check that, if n_ranks > 1, the number of local dofs is > 0, and < than
  // the total number of dofs
  if (n_ranks > 1)
    {
      const auto n_local_dofs = tps.get_dof_handler().n_locally_owned_dofs();
      ASSERT_GT(n_local_dofs, 0u);
      ASSERT_LT(n_local_dofs, tps.get_dof_handler().n_dofs());
      std::cout << "Rank " << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                << " has " << n_local_dofs << " local dofs out of "
                << tps.get_dof_handler().n_dofs() << " total dofs."
                << std::endl;
    }


  // Get quadrature points positions and check they are not empty
  const auto &qpoints = tps.get_locally_owned_qpoints();


  ASSERT_FALSE(qpoints.empty())
    << "No quadrature points found in the tensor product space for processor : "
    << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  // Test that we can get indices from qpoints
  if (!qpoints.empty())
    {
      const auto [cell_index, q_index, i] =
        tps.particle_id_to_cell_and_qpoint_indices(0);
      ASSERT_GE(cell_index, 0);
      ASSERT_GE(q_index, 0);
    }
}



TEST(TensorProductSpace, CrossSectionWeightsMatchReferenceMass)
{
  constexpr double                         thickness = 0.125;
  TensorProductSpaceParameters<1, 2, 3, 1> params;
  params.thickness = "0.125";
  params.reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
  params.section.inclusion_degree      = 1;
  params.section.selected_coefficients = {0, 1, 2};
  TensorProductSpace<1, 2, 3, 1> tps(params);
  tps.initialize();

  const auto &weights         = tps.get_locally_owned_weights();
  const auto &reduced_weights = tps.get_locally_owned_reduced_weights();
  const auto &cross           = tps.get_reference_cross_section();
  ASSERT_EQ(cross.n_selected_basis(), 3u);
  ASSERT_FALSE(reduced_weights.empty());
  ASSERT_GE(weights.size(), cross.n_quadrature_points());

  for (unsigned int i = 0; i < cross.n_selected_basis(); ++i)
    for (unsigned int j = 0; j < cross.n_selected_basis(); ++j)
      {
        double integral = 0.;
        for (unsigned int q = 0; q < cross.n_quadrature_points(); ++q)
          integral += cross.shape_value(i, q, 0) * cross.shape_value(j, q, 0) *
                      weights[q][0] / reduced_weights[0][0];
        const double expected = i == j ? cross.measure(thickness) *
                                           cross.get_basis_mean_squares()[i] :
                                         0.;
        EXPECT_NEAR(integral, expected, 1.e-10);
      }
}

#endif // DEAL_II_WITH_VTK
