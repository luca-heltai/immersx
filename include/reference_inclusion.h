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
#include <deal.II/base/patterns.h>
#include <deal.II/base/polynomials_p.h>
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

template <int dim, int spacedim = dim, int n_components = 1>
class ReferenceInclusionParameters : public ParameterAcceptor
{
public:
  ReferenceInclusionParameters();

  unsigned int                      refinement_level = 1;
  std::string                       inclusion_type   = "hyper_ball";
  unsigned int                      inclusion_degree = 0;
  mutable std::vector<unsigned int> selected_coefficients;
};

/**
 * @brief Class for handling a Reference inclusion in a Reduced method.
 *
 * This class provides functionality for handling many non-matching inclusions
 * in an immersed boundary method. It stores a fully initialized Lagrange finite
 * element space.
 */
template <int dim, int spacedim = dim, int n_components = 1>
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
   * @brief Get the global quadrature object
   *
   * @return const Quadrature<spacedim>&
   */
  const Quadrature<spacedim> &
  get_global_quadrature() const;

  /**
   * @brief Get the selected basis functions object
   *
   * @return const std::vector<Vector<double>>&
   */
  const std::vector<Vector<double>> &
  get_basis_functions() const;

  /**
   * @brief Get the mass matrix object
   *
   * @return const SparseMatrix<double>&
   */
  const SparseMatrix<double> &
  get_mass_matrix() const;

  /**
   * @brief Return the maximum number of selectable basis functions.
   *
   * @return unsigned int
   */
  unsigned int
  max_n_basis() const;

private:
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

  void
  compute_basis();

  const ReferenceInclusionParameters<dim, spacedim, n_components> &par;

  PolynomialsP<spacedim> polynomials;

  Triangulation<dim, spacedim> triangulation;
  QGauss<dim>                  quadrature_formula;
  FESystem<dim, spacedim>      fe;

  DoFHandler<dim, spacedim> dof_handler;

  std::vector<Vector<double>>          basis_functions;
  std::vector<Vector<double>>          selected_basis_functions;
  std::array<Vector<double>, spacedim> x;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> mass_matrix;

  Quadrature<spacedim> global_quadrature;
};

#endif