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

/**
 * @brief Header file for the ReferenceInclusion class.
 *
 * This class represents the reference domain for the cross section of an
 * inclusion.
 *
 * @tparam dim The intrinsic dimension of the reference domain.
 * @tparam spacedim The dimension of the space in which the inclusions are embedded.
 */
#ifndef rdlm_reference_inclusion
#define rdlm_reference_inclusion

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <fstream>

using namespace dealii;

template <int dim, int spacedim, int n_components = 1>
class ReferenceInclusionParameters : public ParameterAcceptor
{
public:
  enum class InclusionType
  {
    hyper_ball = 0x1, //< Ball inclusion
    hyper_cube = 0x2, //< Square inclusion
  };

  ReferenceInclusionParameters(const unsigned int)
    : ParameterAcceptor("Reference inclusion")
  {
    add_parameter("Maximum inclusion degree", inclusion_degree);
    add_parameter(
      "Selected indices",
      selected_coefficients,
      "This allows one to select a subset of the components of the "
      "basis functions used for the representation of the data "
      "(boundary data or forcing data). Notice that these indices are "
      "w.r.t. to the total number of components of the problem, that "
      "is, dimension of the space P^{inclusion_degree} number of Fourier coefficients x number of vector "
      "components. In particular any entry of this list must be in "
      "the set [0,inclusion_degree*n_components). ");
    add_parameter("Inclusion type", inclusion_type);
    add_parameter("Refinement level", refinement_level);
  }

  unsigned int              refinement_level = 1;
  InclusionType             inclusion_type   = InclusionType::ball;
  unsigned int              inclusion_degree = 0;
  std::vector<unsigned int> selected_coefficients;
};

/**
 * @brief Class for handling a Reference inclusion in a Reduced method.
 *
 * This class provides functionality for handling many non-matching inclusions
 * in an immersed boundary method. It stores a fully initialized Lagrange finite
 * element space.
 */
template <int dim, int spacedim, int n_components = 1>
class ReferenceInclusion
{
public:
  /**
   * @brief Construct a new Reference Inclusion object
   *
   * @param par The parameters for the reference inclusion.
   */
  ReferenceInclusion(
    const ReferenceInclusionParameters<dim, spacedim, n_components> &par);

  /**
   * @brief Initialize the grid for the reference inclusion.
   */
  void
  make_grid();

  /**
   * @brief Set the up dofs object
   */
  void
  setup_dofs();

private:
  const ReferenceInclusionParameters<dim, spacedim, n_components> &par;

  Triangulation<dim, spacedim> triangulation;
  QGauss<dim>                  quadrature_formula;
  FESystem<dim, spacedim>      fe;

  DoFHandler<dim, spacedim> dof_handler;

  std::vector<Vector<double>>          basis_functions;
  std::array<Vector<double>, spacedim> x;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> mass_matrix;
};

#endif