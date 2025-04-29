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
   * @brief Read point data from a VTK file and store it in the output vector.
   *
   * This function reads the specified point data array (scalar or vector) from
   * the given VTK file and stores it in the provided output vector.
   *
   * @param vtk_filename The name of the input VTK file.
   * @param point_data_name The name of the point data array to read.
   * @param output_vector The vector to store the point data values.
   */
  void
  read_point_data(const std::string &vtk_filename,
                  const std::string &point_data_name,
                  Vector<double>    &output_vector);

  /**
   * @brief Read a VTK mesh and all data fields into a DoFHandler and output
   * vector.
   *
   * This function reads the mesh from the specified VTK file, populates the
   * Triangulation associated to the given DoFHandler, and queries all cell and
   * point data fields. For each data field, a suitable FESystem is constructed
   * (using FE_DGQ for cell data and FE_Q for point data, with the correct
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

} // namespace VTKUtils

#endif // DEAL_II_WITH_VTK
#endif