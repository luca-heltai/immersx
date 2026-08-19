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

#include <deal.II/base/patterns.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h> // Added include
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h> // Added include
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>     // Added include
#include <deal.II/numerics/matrix_tools.h> // Added include

#include <gtest/gtest.h>

#include <cmath>
#include <set>

#ifdef DEAL_II_WITH_VTK

#  include "reduced_field_utils.h"
#  include "vtk_utils.h"

using namespace dealii;

TEST(LargeNetworks, TerminalNodes)
{
  std::string         vtk_filename = SOURCE_DIR "/data/tests/mstree_1000.vtk";
  Triangulation<1, 3> tria;

  VTKUtils::read_vtk(vtk_filename, tria, false);

  Vector<double> path_lengths(tria.n_vertices());
  VTKUtils::read_vertex_data(vtk_filename, "path_distance", path_lengths);

  // By construction, the root of the tree (not necessarily at the boundary,
  // i.e., it may have more than one edge) has path length 0.
  // The other vertices have path length > 0.
  ASSERT_DOUBLE_EQ(path_lengths(0), 0.0);

  const auto boundary_points = GridTools::get_all_vertices_at_boundary(tria);

  auto boundary_ids = tria.get_boundary_ids();
  ASSERT_GT(boundary_points.size(), 0U);
  ASSERT_EQ(boundary_ids.size(), 1U);

  const auto expected_boundary_id = *boundary_ids.begin();
  for (const auto &cell : tria.active_cell_iterators())
    for (const auto f : cell->face_indices())
      if (cell->face(f)->at_boundary())
        EXPECT_EQ(cell->face(f)->boundary_id(), expected_boundary_id);

  const unsigned int n_terminal_nodes        = boundary_points.size();
  const unsigned int n_distinct_boundary_ids = boundary_ids.size();

  std::cout << "Number of terminal nodes: " << n_terminal_nodes
            << ", number of vertices: " << tria.n_vertices()
            << ", number of boundary ids: " << n_distinct_boundary_ids
            << std::endl;
}


TEST(LargeNetworks, SolvePoisson)
{
  std::string         vtk_filename = SOURCE_DIR "/data/tests/mstree_1000.vtk";
  Triangulation<1, 3> tria;

  VTKUtils::read_vtk(vtk_filename, tria, false);

  Vector<double> path_lengths(tria.n_vertices());
  VTKUtils::read_vertex_data(vtk_filename, "path_distance", path_lengths);

  FE_Q<1, 3>       fe(1);
  DoFHandler<1, 3> dh(tria);
  dh.distribute_dofs(fe);
  Vector<double> solution(dh.n_dofs());
  Vector<double> system_rhs(dh.n_dofs());

  // Build poisson matrix
  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dh, constraints);
  // Constrain the root vertex (vertex 0) to 1.0
  // First, get the DoF index for vertex 0.
  // In 1D, with FE_Q(1), each vertex has one DoF.
  // We need to find the cell that owns vertex 0 and then get its DoF index.
  // However, a simpler way for FE_Q(1) is to assume DoF indices match vertex
  // indices if DoFs are ordered by vertex. For a more general approach, one
  // would iterate cells. For this specific case (FE_Q(1) on a 1D mesh), we can
  // often assume the DoF index for vertex `v` is `v` if there's only one
  // component. Let's verify this assumption or find a robust way.

  // A robust way to get the DoF index for a specific vertex:
  types::global_dof_index root_index     = numbers::invalid_unsigned_int;
  bool                    found_vertex_0 = false;
  for (const auto &cell : dh.active_cell_iterators())
    {
      for (const auto &v : cell->vertex_indices())
        if (cell->vertex_index(v) == 0)
          {
            root_index     = cell->vertex_dof_index(v, 0);
            found_vertex_0 = true;
            break;
          }
      if (found_vertex_0)
        break;
    }
  ASSERT_TRUE(found_vertex_0) << "Could not find DoF for vertex 0.";
  constraints.add_line(root_index);
  constraints.set_inhomogeneity(root_index, 1.0);
  constraints.close();

  SparsityPattern        sparsity_pattern;
  DynamicSparsityPattern dsp(dh.n_dofs(), dh.n_dofs());
  DoFTools::make_sparsity_pattern(dh, dsp, constraints, false);
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix(sparsity_pattern);
  MatrixCreator::create_laplace_matrix(dh,
                                       QGauss<1>(fe.degree + 1),
                                       system_matrix,
                                       Functions::ConstantFunction<3>(1),
                                       system_rhs,
                                       {},
                                       constraints);
  SparseDirectUMFPACK solver;
  solver.initialize(system_matrix);
  solver.vmult(solution, system_rhs);
  constraints.distribute(solution);

  auto path_output = solution;

  DataOut<1, 3> data_out;
  data_out.attach_dof_handler(dh);
  ReducedFieldUtils::data_to_dealii_vector(tria, path_lengths, dh, path_output);
  data_out.add_data_vector(path_output, "path_lengths");
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
}

#else

using namespace dealii;

TEST(LargeNetworks, RequiresVTK)
{
  GTEST_SKIP() << "LargeNetworks tests require deal.II VTK support.";
}

#endif // DEAL_II_WITH_VTK
