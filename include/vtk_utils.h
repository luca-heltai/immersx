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

#ifndef rdl_vtk_utils_h
#define rdl_vtk_utils_h

#include <deal.II/base/config.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <iostream>
#include <map>
#include <unordered_map>


#ifdef DEAL_II_WITH_VTK

#  include <deal.II/grid/tria.h>

#  include <deal.II/lac/la_parallel_vector.h>
#  include <deal.II/lac/vector.h>

using namespace dealii;

namespace VTKUtils
{


  /**
   * @brief Read a VTK mesh file and populate a deal.II Triangulation.
   *
   * This function reads the mesh from the specified VTK file and fills the
   * given Triangulation object. If cleanup is true, overlapping points in the
   * VTK file are merged using VTK's cleaning utilities.
   *
   * @param vtk_filename The name of the input VTK file.
   * @param tria The Triangulation object to populate.
   * @param cleanup If true, merge overlapping points in the VTK file (default: true).
   */
  template <int dim, int spacedim>
  void
  read_vtk(const std::string            &vtk_filename,
           Triangulation<dim, spacedim> &tria,
           const bool                    cleanup = true);

  /**
   * @brief Read cell data (scalar or vector) from a VTK file and store it in
   * the output vector.
   *
   * This function reads the specified cell data array (scalar or vector) from
   * the given VTK file and stores it in the provided output vector. For vector
   * data, all components are stored in row-major order (cell0_comp0,
   * cell0_comp1, ..., cell1_comp0, ...).
   *
   * @param vtk_filename The name of the input VTK file.
   * @param cell_data_name The name of the cell data array to read.
   * @param output_vector The vector to store the cell data values.
   */
  void
  read_cell_data(const std::string &vtk_filename,
                 const std::string &cell_data_name,
                 Vector<double>    &output_vector);

  /**
   * @brief Read vertex data from a VTK file and store it in the output vector.
   *
   * This function reads the specified vertex data array (scalar or vector) from
   * the given VTK file and stores it in the provided output vector.
   *
   * @param vtk_filename The name of the input VTK file.
   * @param vertex_data_name The name of the vertex data array to read.
   * @param output_vector The vector to store the vertex data values.
   */
  void
  read_vertex_data(const std::string &vtk_filename,
                   const std::string &vertex_data_name,
                   Vector<double>    &output_vector);


  /**
   * @brief Read all field data from a VTK file and store it in the output vector.
   *
   * This function reads all field data arrays (scalar or vector, cell or point
   * data) from the given VTK file and stores it in the provided output vector.
   *
   * The data is output in the following way:
   * - first all vertex data (point data) in the order they are found in the
   *   VTK file, with all components stored in row-major order (vertex0_comp0,
   *   vertex0_comp1, ..., vertex1_comp0, ...)
   * - then all cell data (cell data) in the order they are found in the VTK
   * file, with all components stored in row-major order (cell0_comp0,
   * cell0_comp1, ...).
   *
   * This is equivalent to calling read_vertex_data() for each vertex data
   * field, and then read_cell_data() for each cell data field, and
   * concatenating the resulting vectors in a single long vector.
   *
   * @param vtk_filename The name of the input VTK file.
   * @param output_vector The vector to store the vertex data values.
   */
  void
  read_data(const std::string &vtk_filename, Vector<double> &output_vector);

  /**
   * Map vtk fields to a FiniteElement object.
   *
   * This function reads the vtk file and constructs a suitable FiniteElement
   * object that can be later used to store the data field values contained in
   * the vtk file. The function returns a pair containing the FESystem object
   * with one block for each field found in the vtk file, and a vector of
   * strings with the names of the fields.
   *
   * VTK point data is stored in blocks of FE_Q elements or FE_System(FE_Q,
   * n_comps), while cell data is stored in FE_DGQ or FE_System(FE_DGQ,
   * n_comps). The number of components is determined by the number of
   * components in the data field.
   *
   * @param vtk_filename The name of the input VTK file
   */
  template <int dim, int spacedim>
  std::pair<std::unique_ptr<FiniteElement<dim, spacedim>>,
            std::vector<std::string>>
  vtk_to_finite_element(const std::string &vtk_filename);


  /**
   * Translate a vtk data file (obtained through read_data()) to a dealii vector
   * type, associated with the given DoFHandler object.
   *
   * The input data refers to the serial triangulation (obtained through the
   * read_vtk() method). The DoFHandler dh may be serial or parallel.
   *
   * The DoFHandler must be already initialized with the finite element obtained
   * through the vtk_to_finite_element() method. The grid underlying the
   * DoFHandler dh must be generated by the serial_tria for this method to work.
   *
   * @tparam dim
   * @tparam spacedim
   * @tparam VectorType
   * @param serial_tria
   * @param data
   * @param dh
   * @param output_vector
   */
  template <int dim, int spacedim, typename VectorType>
  void
  data_to_dealii_vector(const Triangulation<dim, spacedim> &serial_tria,
                        const Vector<double>               &data,
                        const DoFHandler<dim, spacedim>    &dh,
                        VectorType                         &output_vector);


  /**
   * @brief Read a VTK mesh and all data fields into a DoFHandler and output
   * vector.
   *
   * This function reads the mesh from the specified VTK file, populates the
   * Triangulation associated to the given DoFHandler, and queries all cell and
   * vertex data fields. For each data field, a suitable FESystem is constructed
   * (using FE_DGQ for cell data and FE_Q for vertex data, with the correct
   * number of components). DoFs are distributed and renumbered block-wise. All
   * data is read into the output_vector, and the names of the fields are stored
   * in data_names.
   *
   * @param vtk_filename The name of the input VTK file.
   * @param dof_handler The DoFHandler to distribute DoFs on the mesh.
   * @param output_vector The vector to store all data field values.
   * @param data_names The vector to store the names of all data fields found in
   * the VTK file.
   */
  template <int dim, int spacedim>
  void
  read_vtk(const std::string         &vtk_filename,
           DoFHandler<dim, spacedim> &dof_handler,
           Vector<double>            &output_vector,
           std::vector<std::string>  &data_names);


  // Custom comparator for Point<dim>.
  template <int dim>
  struct PointComparator
  {
    bool
    operator()(const Point<dim> &p1, const Point<dim> &p2) const
    {
      const double tol = (p1.norm() + p2.norm()) * .5e-7;
      // Compare lexicographically
      for (unsigned int i = 0; i < dim; ++i)
        {
          if (p1[i] < p2[i] - tol)
            return true;
          if (p2[i] < p1[i] - tol)
            return false;
        }
      return false; // Points are considered equal
    }
  };


  /**
   * Map a serial vector to a distributed vector.
   *
   * This function transfers data from a serial vector (the one returned by
   * read_vtk() above) to a distributed vector. The underlying assumption is
   * that the serial vector contains data for all vertices in the mesh, and that
   * the global cell indices are preserved in the distributed vector.
   *
   * @param serial_dof_handler The serial DoFHandler.
   * @param parallel_dof_handler The parallel DoFHandler.
   * @param serial_vec The serial vector containing the data.
   * @param parallel_vec The distributed vector to be filled.
   */
  template <int dim, int spacedim>
  void
  serial_vector_to_distributed_vector(
    const DoFHandler<dim, spacedim>            &serial_dh,
    const DoFHandler<dim, spacedim>            &parallel_dh,
    const Vector<double>                       &serial_vec,
    LinearAlgebra::distributed::Vector<double> &parallel_vec);


  /**
   * Map distributed vertex indices to serial vertex indices.
   *
   * The returned vector has size parallel_tria.n_vertices(). For each locally
   * owned vertex, it contains the corresponding vertex index of the serial
   * Triangulation. If a vertex is not locally owned, the corresponding serial
   * index is `numbers::invalid_unsigned_int`
   *
   * The parallel Triangulation must have been generated from the serial one for
   * this function to be any meaningful at all.
   *
   * @param serial_tria The serial Triangulation
   * @param parallel_tria The parallel Triangulation
   */
  template <int dim, int spacedim>
  std::vector<types::global_vertex_index>
  distributed_to_serial_vertex_indices(
    const Triangulation<dim, spacedim> &serial_tria,
    const Triangulation<dim, spacedim> &parallel_tria);

  /**
   * Fill a distributed vector from a serial vector using a mapping of
   * points to DoF indices.
   *
   * This function transfers data from a serial vector (the one returned by
   * read_vtk() above) to a distributed vector based on the mapping of points to
   * DoF indices for both the serial and parallel DoFHandler objects.
   *
   * @param parallel_dof_handler The parallel DoFHandler.
   * @param serial_vec The serial vector containing the data.
   * @param mapping The mapping used for support points.
   * @param parallel_vec The distributed vector to be filled.
   * @param parallel_map The mapping of points to DoF indices for the parallel DoFHandler.
   * @param comm The MPI communicator.
   */
  template <int dim>
  void
  fill_distributed_vector_from_serial(
    const IndexSet       &owned_dofs,
    const Vector<double> &serial_vec,
    const std::map<Point<dim>, types::global_dof_index, PointComparator<dim>>
                                               &serial_map,
    LinearAlgebra::distributed::Vector<double> &parallel_vec,
    const std::map<Point<dim>, types::global_dof_index, PointComparator<dim>>
            &parallel_map,
    MPI_Comm comm);

  // Template functions
  template <int dim, int spacedim, typename VectorType>
  void
  data_to_dealii_vector(const Triangulation<dim, spacedim> &serial_tria,
                        const Vector<double>               &data,
                        const DoFHandler<dim, spacedim>    &dh,
                        VectorType                         &output_vector)
  {
    AssertDimension(dh.n_dofs(), output_vector.size());
    const auto &fe = dh.get_fe();

    const auto dist_to_serial_vertices =
      distributed_to_serial_vertex_indices(serial_tria, dh.get_triangulation());

    const auto &locally_owned_dofs = dh.locally_owned_dofs();

    types::global_dof_index dofs_offset        = 0;
    unsigned int            vertex_comp_offset = 0;
    unsigned int            cell_comp_offset   = 0;
    for (unsigned int field = 0; field < fe.n_blocks(); ++field)
      {
        const auto        &field_fe = fe.base_element(field);
        const unsigned int n_comps  = field_fe.n_components();
        if (field_fe.n_dofs_per_vertex() > 0)
          {
            // This is a vertex data field
            const types::global_dof_index n_local_dofs =
              n_comps * serial_tria.n_vertices();
            for (const auto &cell : dh.active_cell_iterators())
              if (cell->is_locally_owned())
                for (const auto v : cell->vertex_indices())
                  {
                    const types::global_dof_index serial_vertex_index =
                      dist_to_serial_vertices[cell->vertex_index(v)];
                    if (serial_vertex_index != numbers::invalid_unsigned_int)
                      for (unsigned int c = 0; c < n_comps; ++c)
                        {
                          const types::global_dof_index dof_index =
                            cell->vertex_dof_index(v, vertex_comp_offset + c);
                          Assert(locally_owned_dofs.is_element(dof_index),
                                 ExcInternalError());
                          output_vector[dof_index] =
                            data[dofs_offset + n_comps * serial_vertex_index +
                                 c];
                        }
                  }
            dofs_offset += n_local_dofs;
            vertex_comp_offset += n_comps;
          }
        else if (field_fe.template n_dofs_per_object<dim>() > 0)
          {
            // this is a cell data field
            const types::global_dof_index n_local_dofs =
              n_comps * serial_tria.n_global_active_cells();

            // Assumption: serial and parallel meshes have the same ordering of
            // cells.
            auto serial_cell   = serial_tria.begin_active();
            auto parallel_cell = dh.begin_active();
            for (; parallel_cell != dh.end(); ++parallel_cell)
              if (parallel_cell->is_locally_owned())
                {
                  // Advanced serial cell until we reach the same cell index of
                  // the parallel cell
                  while (serial_cell->id() < parallel_cell->id())
                    ++serial_cell;
                  const auto serial_cell_index =
                    serial_cell->global_active_cell_index();
                  for (unsigned int c = 0; c < n_comps; ++c)
                    {
                      const types::global_dof_index dof_index =
                        parallel_cell->dof_index(cell_comp_offset + c);
                      Assert(locally_owned_dofs.is_element(dof_index),
                             ExcInternalError());
                      output_vector[dof_index] =
                        data[dofs_offset + n_comps * serial_cell_index + c];
                    }
                }
            dofs_offset += n_local_dofs;
            cell_comp_offset += n_comps;
          }
      }
  }
} // namespace VTKUtils
#endif // DEAL_II_WITH_VTK


#endif