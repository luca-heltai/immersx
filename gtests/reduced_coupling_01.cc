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

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>

#ifdef DEAL_II_WITH_VTK

#include "reduced_coupling.h"

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
  background_tria.refine_global(5);

  ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components> par;

  ParameterAcceptor::initialize("", "reduced_coupling_01.prm");

  par.tensor_product_space_parameters.reduced_grid_name =
    SOURCE_DIR "/data/tests/mstree_100.vtk";

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
  background_tria.refine_global(5);

  ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components> par;

  par.tensor_product_space_parameters.reduced_grid_name =
    SOURCE_DIR "/data/tests/one_cylinder.vtk";
  // This should be the scaling factor for the coupling
  par.coupling_rhs_expressions                  = {"1"};
  par.tensor_product_space_parameters.thickness = 0.01;

  ReducedCoupling<reduced_dim, dim, spacedim, n_components> coupling(
    background_tria, par);

  // Initialize everything
  coupling.initialize();

  FE_Q<spacedim>       fe(1);
  DoFHandler<spacedim> dh(background_tria);
  dh.distribute_dofs(fe);

  IndexSet owned_dofs = dh.locally_owned_dofs();
  IndexSet relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dh, relevant_dofs);

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
  background_tria.refine_global(5);

  ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components> par;

  ParameterAcceptor::initialize("", "reduced_coupling_01.prm");

  par.tensor_product_space_parameters.reduced_grid_name =
    SOURCE_DIR "/data/tests/mstree_100.vtk";
  par.tensor_product_space_parameters.section.inclusion_degree = 1;
  par.coupling_rhs_expressions = {"1", "0", "0"};

  ReducedCoupling<reduced_dim, dim, spacedim, n_components> coupling(
    background_tria, par);

  // Initialize everything
  coupling.initialize();
}

#endif // DEAL_II_WITH_VTK