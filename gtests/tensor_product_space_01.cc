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

#ifdef DEAL_II_WITH_VTK

#include "immersed_repartitioner.h"
#include "tensor_product_space.h"
#include "utils.h"

using namespace dealii;

TEST(TensorProductSpace, GridGeneration) // NOLINT
{
  static constexpr int reduced_dim  = 1;
  static constexpr int dim          = 2;
  static constexpr int spacedim     = 3;
  static constexpr int n_components = 1;

  // Setup parameters
  TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components> params;
  params.reduced_grid_name = SOURCE_DIR "/data/tests/mstree_100.vtk";

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

  params.reduced_grid_name = SOURCE_DIR "/data/tests/mstree_100.vtk";

  // Create a background grid (hypercube)
  parallel::distributed::Triangulation<spacedim> background_tria(
    MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background_tria, -0.2, 1.2);
  background_tria.refine_global(5);

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



TEST(TensorProductSpace, OrthoNormality) // NOLINT
{
  static constexpr int reduced_dim  = 1;
  static constexpr int dim          = 2;
  static constexpr int spacedim     = 3;
  static constexpr int n_components = 1;

  // Setup parameters
  TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components> params;
  params.thickness = 0.125;

  params.reduced_grid_name = SOURCE_DIR "/data/tests/mstree_100.vtk";
  // Create the tensor product space
  TensorProductSpace<reduced_dim, dim, spacedim, n_components> tps(params);

  // Initialize the tensor product space
  tps.initialize();

  // Loop over the first cross-section quadrature points, and integrate phi_i *
  // phi_j. This should give delta_ij We can do this by looping over all
  // quadrature points, and checking that the integral is 0 for i != j
  const auto &weigths         = tps.get_locally_owned_weights();
  const auto &reduced_weights = tps.get_locally_owned_reduced_weights();

  const auto &cross = tps.get_reference_cross_section();

  for (size_t i = 0; i < cross.n_selected_basis(); ++i)
    for (size_t j = 0; i < cross.n_selected_basis(); ++i)
      {
        double integral = 0.0;
        for (size_t q = 0;
             q < tps.get_reference_cross_section().n_quadrature_points();
             ++q)
          {
            auto phi_i = cross.shape_value(i, q, 0);
            auto phi_j = cross.shape_value(j, q, 0);

            integral += phi_i * phi_j * weigths[q][0] / reduced_weights[0][0];
          }
        if (i != j)
          {
            ASSERT_NEAR(integral, 0.0, 1e-10)
              << "Integral of phi_" << i << " and phi_" << j
              << " should be zero.";
          }
        else
          {
            ASSERT_NEAR(integral,
                        tps.get_reference_cross_section().measure(
                          params.thickness),
                        1e-10)
              << "Integral of phi_" << i << " and phi_" << j
              << " should be one.";
          }
      }
}

#endif // DEAL_II_WITH_VTK