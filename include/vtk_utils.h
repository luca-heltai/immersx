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

#  include <deal.II/lac/vector.h>

using namespace dealii;

namespace VTKUtils
{
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
} // namespace VTKUtils

#endif // DEAL_II_WITH_VTK
#endif