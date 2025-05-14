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
    const Triangulation<dim, spacedim>               &serial_tria,
    const parallel::TriangulationBase<dim, spacedim> &parallel_tria);

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
} // namespace VTKUtils

#endif // DEAL_II_WITH_VTK


#endif