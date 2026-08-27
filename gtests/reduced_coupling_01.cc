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

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>

#include <set>

#include "test_paths.h"

#ifdef DEAL_II_WITH_VTK

#  include <immersx/coupling/reduced_coupling.h>

using namespace ImmersX;
#  include <immersx/io/utils.h>

using namespace dealii;

TEST(ReducedCoupling, MPI_Constructor) // NOLINT
{
  ParameterAcceptor::clear();
  static constexpr int reduced_dim  = 1;
  static constexpr int dim          = 2;
  static constexpr int spacedim     = 3;
  static constexpr int n_components = 1;

  // Create a background grid (hypercube)
  parallel::distributed::Triangulation<spacedim> background_tria(
    MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background_tria, -0.2, 1.2);
  background_tria.refine_global(2);

  ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components> par;

  initialize_parameters("",
                        ImmersX::TestPaths::output_directory(
                          "reduced-coupling/reduced_coupling_01.prm"));

  par.tensor_product_space_parameters.reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/mstree_100.vtk");

  ReducedCoupling<reduced_dim, dim, spacedim, n_components> coupling(
    background_tria, par);

  // Initialize everything
  coupling.initialize();
}



TEST(ReducedCoupling, CheckMatrices) // NOLINT
{
  ParameterAcceptor::clear();
  static constexpr int reduced_dim  = 1;
  static constexpr int dim          = 2;
  static constexpr int spacedim     = 3;
  static constexpr int n_components = 1;

  // Create a background grid (hypercube)
  parallel::distributed::Triangulation<spacedim> background_tria(
    MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background_tria, -0.2, 1.2);
  background_tria.refine_global(2);

  ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components> par;

  par.tensor_product_space_parameters.reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
  // This should be the scaling factor for the coupling
  par.coupling_rhs_expressions                  = {"1"};
  par.tensor_product_space_parameters.thickness = "0.01";

  ReducedCoupling<reduced_dim, dim, spacedim, n_components> coupling(
    background_tria, par);

  // Initialize everything
  coupling.initialize();

  FE_Q<spacedim>       fe(1);
  DoFHandler<spacedim> dh(background_tria);
  dh.distribute_dofs(fe);

  IndexSet owned_dofs    = dh.locally_owned_dofs();
  IndexSet relevant_dofs = DoFTools::extract_locally_relevant_dofs(dh);

  AffineConstraints<double> constraints(owned_dofs, relevant_dofs);
  constraints.close();

  DynamicSparsityPattern dsp(dh.n_dofs(), coupling.get_dof_handler().n_dofs());
  coupling.assemble_coupling_sparsity(dsp, dh, constraints);

  LinearAlgebraTrilinos::MPI::SparseMatrix coupling_matrix;
  coupling_matrix.reinit(owned_dofs,
                         coupling.get_dof_handler().locally_owned_dofs(),
                         dsp,
                         MPI_COMM_WORLD);
  coupling.assemble_coupling_matrix(coupling_matrix, dh, constraints);

  // Now build a vector
  LinearAlgebraTrilinos::MPI::Vector back_vector;
  LinearAlgebraTrilinos::MPI::Vector immersed_vector;

  back_vector.reinit(owned_dofs, MPI_COMM_WORLD);
  immersed_vector.reinit(coupling.get_dof_handler().locally_owned_dofs(),
                         MPI_COMM_WORLD);

  VectorTools::interpolate(dh,
                           Functions::ConstantFunction<spacedim>(1.0),
                           back_vector);

  auto res = immersed_vector;

  coupling_matrix.Tvmult(res, back_vector);

  // Now try assembling the rhs
  coupling.assemble_reduced_rhs(immersed_vector);

  // Now take the difference and check the L2 norm. It should be zero.
  res -= immersed_vector;
  const double norm = res.l2_norm();
  ASSERT_NEAR(norm, 0.0, 1e-10)
    << "The L2 norm of the difference between the two vectors is: " << norm;
}


TEST(ReducedCoupling, PositiveDimensionalAssemblySeparatesRepresentativeCells)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);
  ParameterAcceptor::clear();

  static constexpr int reduced_dim  = 1;
  static constexpr int dim          = 2;
  static constexpr int spacedim     = 3;
  static constexpr int n_components = 1;

  parallel::distributed::Triangulation<spacedim> background_tria(
    MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background_tria, -0.1, 3.1);

  ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components> par;
  par.tensor_product_space_parameters.reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/simple_1d_grid.vtk");
  par.tensor_product_space_parameters.n_q_points               = 2;
  par.tensor_product_space_parameters.thickness                = "0.2";
  par.tensor_product_space_parameters.section.inclusion_type   = "hyper_ball";
  par.tensor_product_space_parameters.section.refinement_level = 1;
  par.tensor_product_space_parameters.section.inclusion_degree = 0;
  par.tensor_product_space_parameters.section.selected_coefficients = {0};
  par.tensor_product_space_parameters.section.quadrature_type       = "gauss";
  par.tensor_product_space_parameters.section.n_q_points            = 4;
  par.coupling_rhs_expressions                                      = {"0"};

  ReducedCoupling<reduced_dim, dim, spacedim, n_components> coupling(
    background_tria, par);
  coupling.initialize();

  FE_Q<spacedim>       background_fe(1);
  DoFHandler<spacedim> background_dh(background_tria);
  background_dh.distribute_dofs(background_fe);
  AffineConstraints<double> constraints;
  constraints.close();

  DynamicSparsityPattern dsp(background_dh.n_dofs(),
                             coupling.get_dof_handler().n_dofs());
  coupling.assemble_coupling_sparsity(dsp, background_dh, constraints);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);

  SparseMatrix<double> coupling_matrix;
  coupling_matrix.reinit(sparsity);
  coupling.assemble_coupling_matrix(coupling_matrix,
                                    background_dh,
                                    constraints);

  SparseMatrix<double> expected_matrix;
  expected_matrix.reinit(sparsity);
  std::set<types::global_cell_index>   representative_cells;
  std::vector<types::global_dof_index> background_dof_indices(
    background_fe.n_dofs_per_cell());
  const auto &immersed_fe = coupling.get_dof_handler().get_fe();
  std::vector<types::global_dof_index> immersed_dof_indices(
    immersed_fe.n_dofs_per_cell());

  for (const auto &particle : coupling.get_particles())
    {
      const typename DoFHandler<spacedim>::cell_iterator cell(
        *particle.get_surrounding_cell(), &background_dh);
      cell->get_dof_indices(background_dof_indices);
      const auto [immersed_cell_id, immersed_q, section_q] =
        coupling.particle_id_to_representative_indices(particle.get_id());
      representative_cells.insert(immersed_cell_id);
      immersed_dof_indices      = coupling.get_dof_indices(immersed_cell_id);
      const auto immersed_point = coupling.get_quadrature().point(immersed_q);
      for (unsigned int i = 0; i < background_fe.n_dofs_per_cell(); ++i)
        for (unsigned int j = 0; j < immersed_fe.n_dofs_per_cell(); ++j)
          {
            const auto background_component =
              background_fe.system_to_component_index(i).first;
            const auto immersed_component =
              immersed_fe.system_to_component_index(j).first;
            expected_matrix.add(
              background_dof_indices[i],
              immersed_dof_indices[j],
              background_fe.shape_value(i, particle.get_reference_location()) *
                coupling.get_reference_cross_section().shape_value(
                  immersed_component, section_q, background_component) *
                immersed_fe.shape_value(j, immersed_point) *
                particle.get_properties()[0]);
          }
    }

  ASSERT_GT(representative_cells.size(), 1u);
  for (unsigned int i = 0; i < coupling_matrix.m(); ++i)
    for (unsigned int j = 0; j < coupling_matrix.n(); ++j)
      EXPECT_DOUBLE_EQ(coupling_matrix.el(i, j), expected_matrix.el(i, j));
}


TEST(ReducedCoupling, MPI_ConstructorP1) // NOLINT
{
  ParameterAcceptor::clear();
  static constexpr int reduced_dim  = 1;
  static constexpr int dim          = 2;
  static constexpr int spacedim     = 3;
  static constexpr int n_components = 1;

  // Create a background grid (hypercube)
  parallel::distributed::Triangulation<spacedim> background_tria(
    MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background_tria, -0.2, 1.2);
  background_tria.refine_global(2);

  ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components> par;

  initialize_parameters("",
                        ImmersX::TestPaths::output_directory(
                          "reduced-coupling/reduced_coupling_01_p1.prm"));

  par.tensor_product_space_parameters.reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/mstree_100.vtk");
  par.tensor_product_space_parameters.section.inclusion_degree = 1;
  par.coupling_rhs_expressions = {"1", "0", "0"};

  ReducedCoupling<reduced_dim, dim, spacedim, n_components> coupling(
    background_tria, par);

  // Initialize everything
  coupling.initialize();
}

#endif // DEAL_II_WITH_VTK
