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
#include <immersx/coupling/particle_coupling.h>

#include <algorithm>
#include <filesystem>

#include "test_paths.h"

#ifdef DEAL_II_WITH_VTK

#  include <immersx/coupling/immersed_repartitioner.h>

using namespace ImmersX;
#  include <immersx/coupling/tensor_product_space.h>

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
  background_tria.refine_global(2);
  params.reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/mstree_100.vtk");

  // Create the tensor product space
  TensorProductSpace<reduced_dim, dim, spacedim, n_components> tps(params);

  ImmersedRepartitioner<reduced_dim, spacedim> immersed_partitioner(
    background_tria);
  tps.set_partitioner = [&](auto &tria) {
    tria.set_partitioner(immersed_partitioner,
                         TriangulationDescription::Settings());
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
  const auto  proc_to_qindices =
    particle_coupling.insert_points(qpoints, weights);
  tps.update_local_dof_indices(proc_to_qindices);
  // Output the particles to a file
  const auto output_directory =
    ImmersX::TestPaths::output_path("particle-coupling");
  std::filesystem::create_directories(output_directory);
  const std::string filename =
    (output_directory / "particles_test.vtu").string();
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

  const auto &relevant_indices = tps.locally_relevant_indices();
  for (const auto &particle : particle_coupling.get_particles())
    {
      const auto [cell_index, q_index_gamma, q_index_section] =
        tps.particle_id_to_cell_and_qpoint_indices(particle.get_id());
      EXPECT_TRUE(relevant_indices.is_element(cell_index));
    }
}


#endif // DEAL_II_WITH_VTK

TEST(ParticleCoupling, BOTH_SerialBackgroundKeepsPointProperties)
{
  using namespace dealii;
  using namespace ImmersX;
  Triangulation<2> background;
  GridGenerator::hyper_cube(background);
  background.refine_global(1);
  const MappingQ1<2>                       mapping;
  const std::vector<Point<2>>              points     = {Point<2>(0.125, 0.125),
                                                         Point<2>(0.75, 0.25),
                                                         Point<2>(0.4, 0.8)};
  const std::vector<std::vector<double>>   properties = {{0.125, -3.},
                                                         {0.75, 7.},
                                                         {0.5, 11.}};
  const std::vector<types::particle_index> ids        = {13, 27, 41};
  for (const bool explicit_mapping : {false, true})
    {
      SCOPED_TRACE(explicit_mapping);
      ParticleCouplingParameters<2> parameters("/Serial particle coupling/");
      ParticleCoupling<2>           coupling(parameters);
      if (explicit_mapping)
        coupling.initialize_particle_handler(background, mapping, 2);
      else
        coupling.initialize_particle_handler(background, 2);
      const auto bounding_boxes = coupling.get_global_bounding_boxes();
      ASSERT_EQ(bounding_boxes.size(), 1u);
      EXPECT_FALSE(bounding_boxes.front().empty());
      const auto &particles = coupling.get_particles();
      // Each rank keeps the complete point cloud on its serial background.
      const auto located = coupling.insert_points(points, properties, ids);
      ASSERT_EQ(located.size(), 1u);
      EXPECT_EQ(located.at(0).n_elements(), points.size());
      EXPECT_EQ(particles.n_global_particles(), points.size());
      for (const auto &particle : particles)
        {
          const auto id = std::find(ids.begin(), ids.end(), particle.get_id());
          ASSERT_NE(id, ids.end());
          const auto index = static_cast<std::size_t>(id - ids.begin());
          EXPECT_LT(particle.get_location().distance(points[index]), 1.e-14);
          ASSERT_EQ(particle.get_properties().size(), 2u);
          EXPECT_DOUBLE_EQ(particle.get_properties()[0], properties[index][0]);
          EXPECT_DOUBLE_EQ(particle.get_properties()[1], properties[index][1]);
          EXPECT_TRUE(located.at(0).is_element(index));
        }
      const auto empty = coupling.insert_points({});
      for (const auto &[owner, indices] : empty)
        {
          (void)owner;
          EXPECT_EQ(indices.n_elements(), 0u);
        }
      EXPECT_EQ(particles.n_global_particles(), points.size());
    }
}
