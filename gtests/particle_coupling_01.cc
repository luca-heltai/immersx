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
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "immersed_repartitioner.h"
#include "particle_coupling.h"
#include "tensor_product_space.h"

TEST(ParticleCoupling, MPI_OutputParticles) // NOLINT
{
  static constexpr int reduced_dim  = 1;
  static constexpr int dim          = 2;
  static constexpr int spacedim     = 3;
  static constexpr int n_components = 1;

  // Setup parameters
  TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components> params;

  // Create a background grid (hypercube)
  parallel::distributed::Triangulation<spacedim> background_tria(
    MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background_tria, -0.2, 1.2);
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


  ParticleCouplingParameters<spacedim> par;
  ParticleCoupling<spacedim>           particle_coupling(par);
  // Initialize the particle handler with the background triangulation
  particle_coupling.initialize_particle_handler(background_tria);

  // Now add particles to the particle handler
  const auto &qpoints = tps.get_locally_owned_qpoints();
  const auto &weights = tps.get_locally_owned_weights();
  particle_coupling.insert_points(qpoints, weights);
  // Output the particles to a file
  const std::string filename = "particles_test.vtu";
  particle_coupling.output_particles(filename);

  // Check that the file was written correctly
  std::ifstream file(filename);
  ASSERT_TRUE(file.good()) << "Failed to open file: " << filename;
  file.close();

  // Check that the particles were inserted correctly
  const auto &particles = particle_coupling.get_particles();

  auto n_total_points = Utilities::MPI::sum(qpoints.size(), MPI_COMM_WORLD);

  ASSERT_EQ(particles.n_global_particles(), n_total_points)
    << "Number of particles does not match the number of quadrature points.";

  // Output the background grid and its partitioning
  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(background_tria,
                                           "particles_test_background");
  grid_out.write_mesh_per_processor_as_vtu(
    tps.get_dof_handler().get_triangulation(), "particles_test_reduced");
}

TEST(ParticleCoupling, MPI_GlobalCells) // NOLINT
{
  static constexpr int reduced_dim  = 1;
  static constexpr int dim          = 2;
  static constexpr int spacedim     = 3;
  static constexpr int n_components = 1;

  // Setup parameters
  TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components> params;

  // Create a background grid (hypercube)
  parallel::distributed::Triangulation<spacedim> background_tria(
    MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background_tria, -0.2, 1.2);
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

  tps.initialize();

  // Initialize particle coupling
  ParticleCouplingParameters<spacedim> par;
  ParticleCoupling<spacedim>           particle_coupling(par);
  particle_coupling.initialize_particle_handler(background_tria);

  // Add particles to the particle handler
  const auto &qpoints   = tps.get_locally_owned_qpoints();
  const auto &weights   = tps.get_locally_owned_weights();
  auto proc_to_qindices = particle_coupling.insert_points(qpoints, weights);

  tps.update_local_dof_indices(proc_to_qindices);

  // Loop over local particles, and check that the local indices are all within
  // the index set of the relevant indices
  const auto &relevant_indices = tps.locally_relevant_indices();

  std::cout << "Relevant indices: ";
  relevant_indices.print(std::cout);
  std::cout << std::endl;

  for (const auto particle : particle_coupling.get_particles())
    {
      const auto qpoint_index = particle.get_id();
      const auto [cell_index, q_index_gamma, q_index_section] =
        tps.particle_id_to_cell_and_qpoint_indices(qpoint_index);

      ASSERT_TRUE(relevant_indices.is_element(cell_index))
        << "Qpoint index " << qpoint_index << ", corresponding to cell index "
        << cell_index << ", q_index_gamma " << q_index_gamma
        << ", and q_index_section " << q_index_section << " is not within the "
        << "locally relevant indices.";
    }
}
