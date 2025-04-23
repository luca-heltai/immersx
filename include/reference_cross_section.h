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
 * @file reference_cross_section.h
 * Defines the ReferenceCrossSection and ReferenceCrossSectionParameters classes
 * for representing a reference domain and basis functions in reduced immersed
 * boundary methods.
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
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <fstream>

using namespace dealii;

/**
 * Parameter configuration for a ReferenceCrossSection.
 *
 * @tparam dim The intrinsic dimension of the reference domain.
 * @tparam spacedim The embedding space dimension.
 * @tparam n_components Number of components per field variable.
 */
template <int dim, int spacedim = dim, int n_components = 1>
struct ReferenceCrossSectionParameters : public ParameterAcceptor
{
  /// Constructor that registers parameters.
  ReferenceCrossSectionParameters();

  /// Refinement level of the mesh.
  unsigned int refinement_level = 1;

  /// Geometric type of inclusion ("hyper_ball", etc.).
  std::string inclusion_type = "hyper_ball";

  /// Degree of the polynomial basis for inclusion.
  unsigned int inclusion_degree = 0;

  /// List of selected coefficient indices for reduced modeling.
  mutable std::vector<unsigned int> selected_coefficients;
};

/**
 * Handles the construction and management of a reference inclusion geometry and
 * its basis.
 *
 * Used in reduced basis immersed boundary methods, this class initializes and
 * stores a full finite element space, computes basis functions, and provides
 * access to quadrature rules and mass matrices.
 *
 * @tparam dim The intrinsic dimension of the inclusion.
 * @tparam spacedim The embedding dimension.
 * @tparam n_components Number of components per field variable.
 */
template <int dim, int spacedim = dim, int n_components = 1>
class ReferenceCrossSection
{
public:
  /// Constructs the ReferenceCrossSection from parameters.
  ReferenceCrossSection(
    const ReferenceCrossSectionParameters<dim, spacedim, n_components> &par);

  /// Returns the global quadrature object in the embedding space.
  const Quadrature<spacedim> &
  get_global_quadrature() const;

  /// Returns the list of selected basis functions.
  const std::vector<Vector<double>> &
  get_basis_functions() const;

  /// the component comp of the ith selected basis function, at the quadrature
  /// point index q,
  const double &
  shape_value(const unsigned int i,
              const unsigned int q,
              const unsigned int comp) const;

  /// Returns the mass matrix corresponding to selected basis functions.
  const SparseMatrix<double> &
  get_mass_matrix() const;

  /// Returns the number of selected basis functions.
  unsigned int
  n_selected_basis() const;

  /// Returns the total number of available basis functions.
  unsigned int
  max_n_basis() const;

  Quadrature<spacedim>
  get_transformed_quadrature(const Point<spacedim>     &new_origin,
                             const Tensor<1, spacedim> &new_vertical,
                             const double               scale) const;

  unsigned int
  n_quadrature_points() const;

private:
  /// Builds the mesh for the reference inclusion domain.
  void
  make_grid();

  /// Initializes the degrees of freedom and finite element space.
  void
  setup_dofs();

  /// Computes and stores all basis functions.
  void
  compute_basis();

  /// Reference to parameter configuration.
  const ReferenceCrossSectionParameters<dim, spacedim, n_components> &par;

  /// Polynomial space used for modal basis generation.
  PolynomialsP<spacedim> polynomials;

  /// Triangulation of the reference inclusion domain.
  Triangulation<dim, spacedim> triangulation;

  /// Quadrature formula for integration on the reference domain.
  QGauss<dim> quadrature_formula;

  /// Finite element space for the inclusion.
  FESystem<dim, spacedim> fe;

  // Mapping space for the inclusion.
  MappingQ<dim, spacedim> mapping;

  /// Degree of freedom handler for the inclusion.
  DoFHandler<dim, spacedim> dof_handler;

  /// List of all computed basis functions.
  std::vector<Vector<double>> basis_functions;

  /// Subset of selected basis functions.
  std::vector<Vector<double>> selected_basis_functions;

  /// Sparsity pattern used for assembling the mass matrix.
  SparsityPattern sparsity_pattern;

  /// Mass matrix associated with the selected basis functions.
  SparseMatrix<double> mass_matrix;

  /// Quadrature rule in the embedding space.
  Quadrature<spacedim> global_quadrature;

  /// The basis functions computed on the quadrature points.
  FullMatrix<double> basis_functions_on_qpoints;
};

#endif