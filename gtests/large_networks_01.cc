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

#include <deal.II/base/config.h>

#include <deal.II/base/function_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/matrix_tools.h>

#include <gtest/gtest.h>

#include "test_paths.h"

#ifdef DEAL_II_WITH_VTK

#  include <immersx/io/vtk_utils.h>

using namespace dealii;
using namespace ImmersX;

TEST(LargeNetworks, TerminalNodes)
{
  const auto vtk_filename =
    TestPaths::binary_path("gtests/fixtures/branching_network.vtk").string();
  Triangulation<1, 3> tria;
  VTKUtils::read_vtk(vtk_filename, tria, false);
  ASSERT_EQ(tria.n_active_cells(), 3u);
  ASSERT_EQ(tria.n_vertices(), 4u);

  Vector<double> path_lengths(tria.n_vertices());
  VTKUtils::read_vertex_data(vtk_filename, "path_distance", path_lengths);
  EXPECT_DOUBLE_EQ(path_lengths[0], 0.);
  EXPECT_DOUBLE_EQ(path_lengths[1], 1.);
  EXPECT_DOUBLE_EQ(path_lengths[2], 2.);
  EXPECT_DOUBLE_EQ(path_lengths[3], 2.);

  const auto boundary_points = GridTools::get_all_vertices_at_boundary(tria);
  EXPECT_EQ(boundary_points.size(), 3u);
  const auto boundary_ids = tria.get_boundary_ids();
  ASSERT_EQ(boundary_ids.size(), 2u);
  for (const auto &cell : tria.active_cell_iterators())
    for (const auto f : cell->face_indices())
      if (cell->face(f)->at_boundary())
        {
          EXPECT_EQ(cell->face(f)->boundary_id(),
                    cell->vertex(f) == Point<3>() ? 0u : 1u);
        }
}

TEST(LargeNetworksValidation, SolvePoisson)
{
  Triangulation<1, 3> tria;
  VTKUtils::read_vtk(
    TestPaths::binary_path("gtests/fixtures/branching_network.vtk").string(),
    tria,
    false);

  FE_Q<1, 3>       fe(1);
  DoFHandler<1, 3> dh(tria);
  dh.distribute_dofs(fe);
  Vector<double> solution(dh.n_dofs());
  Vector<double> system_rhs(dh.n_dofs());

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dh, constraints);
  types::global_dof_index root_index = numbers::invalid_dof_index;
  for (const auto &cell : dh.active_cell_iterators())
    for (const auto v : cell->vertex_indices())
      if (cell->vertex(v) == Point<3>())
        root_index = cell->vertex_dof_index(v, 0);
  ASSERT_NE(root_index, numbers::invalid_dof_index);
  constraints.add_line(root_index);
  constraints.set_inhomogeneity(root_index, 1.);
  constraints.close();

  DynamicSparsityPattern dsp(dh.n_dofs(), dh.n_dofs());
  DoFTools::make_sparsity_pattern(dh, dsp, constraints, false);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  SparseMatrix<double> system_matrix(sparsity_pattern);
  MatrixCreator::create_laplace_matrix(dh,
                                       QGauss<1>(fe.degree + 1),
                                       system_matrix,
                                       Functions::ConstantFunction<3>(1.),
                                       system_rhs,
                                       {},
                                       constraints);
  SparseDirectUMFPACK solver;
  solver.initialize(system_matrix);
  solver.vmult(solution, system_rhs);
  constraints.distribute(solution);

  // On three unit edges, -u'' = 1, u(root) = 1 and zero leaf flux give
  // u(junction) = 7/2 and u(leaves) = 4. Q1 has these exact nodal values.
  for (const auto &cell : dh.active_cell_iterators())
    for (const auto v : cell->vertex_indices())
      {
        const auto  &point    = cell->vertex(v);
        const double expected = point[2] == 0. ? 1. : point[0] == 0. ? 3.5 : 4.;
        EXPECT_NEAR(solution[cell->vertex_dof_index(v, 0)], expected, 1.e-12);
      }
}

#else

TEST(LargeNetworks, RequiresVTK)
{
  GTEST_SKIP() << "LargeNetworks tests require deal.II VTK support.";
}

#endif // DEAL_II_WITH_VTK
