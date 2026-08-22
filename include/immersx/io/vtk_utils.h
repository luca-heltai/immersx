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

#ifndef rdl_vtk_utils_h
#define rdl_vtk_utils_h

#include <deal.II/base/config.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <immersx/core/reduced_field_catalog.h>
#include <immersx/coupling/point_cloud.h>

#include <limits>
#include <string>
#include <vector>

namespace ImmersX
{
#ifdef DEAL_II_WITH_VTK

  using namespace dealii;

  namespace VTKUtils
  /**
   * @namespace VTKUtils
   * @brief Utility functions for interfacing between VTK mesh/data files and deal.II data structures.
   *
   * This namespace provides a collection of functions to read VTK mesh files
   * and associated data fields, and to map them into deal.II Triangulation,
   * DoFHandler, and vector objects. The utilities support reading mesh
   * geometry, cell and vertex data, and converting VTK metadata into the
   * generic reduced field catalog used by the core.
   *
   * Main functionalities include:
   * - Reading VTK mesh files and populating deal.II Triangulation objects.
   * - Reading cell and vertex data arrays from VTK files into deal.II vectors.
   * - Mapping VTK data fields to the generic reduced-field finite-element
   * layout.
   *
   * These utilities are intended to facilitate the import of VTK-based mesh and
   * data into deal.II-based finite element workflows, supporting both serial
   * and parallel computations.
   */
  {
    /** Read a particle-only VTK, VTU, or PVTU unstructured grid.
     * PointData is copied directly; vertex-cell data is reordered to its point.
     */
    template <int spacedim>
    void
    read_vtk_point_cloud(const std::string    &vtk_filename,
                         PointCloud<spacedim> &point_cloud,
                         const unsigned int    requested_piece =
                           std::numeric_limits<unsigned int>::max(),
                         const unsigned int n_requested_pieces = 1);

    /** Convenience overload returning the catalog and field names separately.
     */
    template <int spacedim>
    void
    read_vtk_point_cloud(const std::string                &vtk_filename,
                         std::vector<Point<spacedim>>     &points,
                         std::vector<std::vector<double>> &properties,
                         FieldCatalog                     &catalog,
                         std::vector<std::string>         &property_names);

    /** Convenience overload with field-major flattened property storage. */
    template <int spacedim>
    void
    read_vtk_point_cloud(const std::string            &vtk_filename,
                         std::vector<Point<spacedim>> &points,
                         Vector<double>               &properties,
                         FieldCatalog                 &catalog,
                         std::vector<std::string>     &property_names);

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
     * the given VTK file and stores it in the provided output vector. For
     * vector data, all components are stored in row-major order (cell0_comp0,
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
     * This function reads the specified vertex data array (scalar or vector)
     * from the given VTK file and stores it in the provided output vector.
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
     * This function reads all field data arrays (scalar or vector, cell or
     * point data) from the given VTK file and stores it in the provided output
     * vector.
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
    std::unique_ptr<FiniteElement<dim, spacedim>>
    vtk_to_finite_element(const std::string &vtk_filename,
                          FieldCatalog      &catalog);

    /**
     * @brief Read a VTK mesh and all data fields into a DoFHandler and output
     * vector.
     *
     * This function reads the mesh from the specified VTK file, populates the
     * Triangulation associated to the given DoFHandler, and queries all cell
     * and vertex data fields. For each data field, a suitable FESystem is
     * constructed (using FE_DGQ for cell data and FE_Q for vertex data, with
     * the correct number of components). DoFs are distributed and renumbered
     * block-wise. All data is read into the output_vector, and the names of the
     * fields are stored in data_names.
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

    template <int dim, int spacedim>
    void
    read_vtk(const std::string         &vtk_filename,
             DoFHandler<dim, spacedim> &dof_handler,
             Vector<double>            &output_vector,
             FieldCatalog              &catalog);
  }    // namespace VTKUtils
#endif // DEAL_II_WITH_VTK


} // namespace ImmersX

#endif
