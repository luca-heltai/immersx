// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "immersed_repartitioner.h"
#include "tensor_product_space.h"

using namespace dealii;

TEST(TensorProductSpace, GridGeneration) // NOLINT
{
  static constexpr int reduced_dim  = 1;
  static constexpr int dim          = 2;
  static constexpr int spacedim     = 3;
  static constexpr int n_components = 1;

  // Setup parameters
  TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components> params;
  params.refinement_level = 0;
  params.fe_degree        = 1;

  // Create the tensor product space
  TensorProductSpace<reduced_dim, dim, spacedim, n_components> tps(params);

  // Set the make_reduced_grid function to read from a VTK file
  tps.make_reduced_grid = [&](auto &tria) {
    // First create a serial triangulation with the VTK file
    const std::string filename = SOURCE_DIR "/data/tests/mstree_100.vtk";
    GridIn<reduced_dim, spacedim>        gridin;
    Triangulation<reduced_dim, spacedim> serial_tria;
    gridin.attach_triangulation(serial_tria);
    std::ifstream f(filename);
    ASSERT_TRUE(f.good()) << "Failed to open file: " << filename;
    gridin.read_vtk(f);
    tria.copy_triangulation(serial_tria);
  };

  // Initialize the tensor product space
  tps.initialize();

  // Verify the reduced grid was created
  const DoFHandler<reduced_dim, spacedim> &dof_handler = tps.get_dof_handler();
  ASSERT_GT(dof_handler.n_dofs(), 0);

  // Get quadrature points positions and check they are not empty
  auto qpoints = tps.get_locally_owned_qpoints_positions();
  ASSERT_FALSE(qpoints.empty());
}

TEST(TensorProductSpace, ImmersedGridPartitioning) // NOLINT
{
  static constexpr int reduced_dim  = 1;
  static constexpr int dim          = 2;
  static constexpr int spacedim     = 3;
  static constexpr int n_components = 1;

  // Setup parameters
  TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components> params;
  params.refinement_level = 0;
  params.fe_degree        = 1;

  // Create a background grid (hypercube)
  parallel::distributed::Triangulation<spacedim> background_tria(
    MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background_tria, -1.2, 1.2);
  background_tria.refine_global(5);

  // Create the tensor product space
  TensorProductSpace<reduced_dim, dim, spacedim, n_components> tps(params);

  // Set the make_reduced_grid function to read from a VTK file and use
  // ImmersedRepartitioner
  tps.make_reduced_grid = [&](auto &tria) {
    // First create a serial triangulation with the VTK file
    const std::string filename = SOURCE_DIR "/data/tests/mstree_100.vtk";
    GridIn<reduced_dim, spacedim>        gridin;
    Triangulation<reduced_dim, spacedim> serial_tria;
    gridin.attach_triangulation(serial_tria);
    std::ifstream f(filename);
    ASSERT_TRUE(f.good()) << "Failed to open file: " << filename;
    gridin.read_vtk(f);

    // Create an unpartitioned fully distributed triangulation
    parallel::fullydistributed::Triangulation<reduced_dim, spacedim>
      serial_tria_fully_distributed(MPI_COMM_WORLD);
    serial_tria_fully_distributed.copy_triangulation(serial_tria);

    // Now use ImmersedRepartitioner to partition the grid
    ImmersedRepartitioner<reduced_dim, spacedim> repartitioner(background_tria);

    // Apply the repartitioner to create a partitioned grid
    auto partition = repartitioner.partition(serial_tria_fully_distributed);

    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(serial_tria_fully_distributed,
                                            partition);
    tria.create_triangulation(construction_data);
  };

  // Initialize the tensor product space
  tps.initialize();

  // Verify the reduced grid was created
  const DoFHandler<reduced_dim, spacedim> &dof_handler = tps.get_dof_handler();
  ASSERT_GT(dof_handler.n_dofs(), 0);

  // Get quadrature points positions and check they are not empty
  auto qpoints = tps.get_locally_owned_qpoints_positions();
  ASSERT_FALSE(qpoints.empty())
    << "No quadrature points found in the tensor product space for processor : "
    << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  // Test that we can get indices from qpoints
  if (!qpoints.empty())
    {
      auto [cell_index, q_index, i] =
        tps.qpoint_index_to_cell_and_qpoint_indices(0);
      ASSERT_GE(cell_index, 0);
      ASSERT_GE(q_index, 0);
    }
}