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

#ifndef tensor_product_space_h
#define tensor_product_space_h

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>

#include "reference_cross_section.h"

using namespace dealii;

/**
 * A structure to hold parameters for a tensor product space.
 *
 * This structure is used to define the parameters required for constructing a
 * tensor product space, including the dimensionality, refinement level, finite
 * element degree, and reference cross-section parameters.
 *
 * @tparam reduced_dim The reduced dimensionality of the tensor product space.
 * @tparam dim The full dimensionality of the tensor product space.
 * @tparam spacedim The spatial dimensionality of the embedding space.
 * @tparam n_components The number of components in the tensor product space.
 */
template <int reduced_dim, int dim, int spacedim, int n_components>
struct TensorProductSpaceParameters : public ParameterAcceptor
{
  /**
   * Default constructor.
   *
   * Initializes the parameters for the tensor product space with default
   * values.
   */
  TensorProductSpaceParameters();

  /**
   * The dimensionality of the cross-section.
   *
   * This is computed as the difference between the full dimensionality
   * (`dim`) and the reduced dimensionality (`reduced_dim`).
   */
  static constexpr int cross_section_dim = dim - reduced_dim;

  /**
   * Parameters for the reference cross-section.
   *
   * This member holds the parameters for the reference cross-section
   * of the tensor product space. The cross-section is defined in a
   * space of dimensionality `cross_section_dim`.
   */
  ReferenceCrossSectionParameters<cross_section_dim, spacedim, n_components>
    section;

  /**
   * The refinement level of the mesh.
   *
   * Specifies the number of refinement levels applied to the mesh
   * used in the tensor product space. Default value is 0.
   */
  unsigned int refinement_level = 0;

  /**
   * The degree of the finite element basis functions.
   *
   * Specifies the polynomial degree of the finite element basis
   * functions used in the tensor product space. Default value is 1.
   */
  unsigned int fe_degree = 1;
};


/**
 * A class representing a tensor product space combining a lower-dimensional
 * triangulation and a reference cross-section.
 *
 * @tparam reduced_dim The dimension of the reduced triangulation.
 * @tparam dim The dimension of the full-order object.
 * @tparam spacedim The dimension of the ambient space
 * @tparam n_components The number of components of the problem.
 */
/**
 * @class TensorProductSpace
 * A class representing a tensor product space for reduced-dimensional problems.
 *
 * This class provides functionality to work with tensor product spaces,
 * including the initialization of reduced grids, handling degrees of freedom
 * (DoFs), and managing reference cross-sections for reduced-dimensional
 * domains.
 *
 * @tparam reduced_dim The reduced dimension of the space.
 * @tparam dim The full dimension of the space.
 * @tparam spacedim The spatial dimension.
 * @tparam n_components The number of components in the system.
 */
template <int reduced_dim, int dim, int spacedim, int n_components>
class TensorProductSpace
{
public:
  /**
   * Constructor for the TensorProductSpace class.
   *
   * Initializes the tensor product space using the provided parameters.
   *
   * @param par The parameters defining the tensor product space.
   */
  TensorProductSpace(
    const TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components>
      &par);

  /**
   * The dimension of the cross-section of the reduced domain.
   *
   * This is computed as the difference between the full dimension and the
   * reduced dimension.
   */
  static constexpr int cross_section_dim = dim - reduced_dim;

  /**
   * A function to create the reduced grid.
   *
   * This function is used to generate the triangulation for the
   * reduced-dimensional domain.
   */
  std::function<void(Triangulation<reduced_dim, spacedim> &)> make_reduced_grid;

  /**
   * Initializes the tensor product space.
   *
   * This function sets up the necessary components, such as the DoFHandler and
   * finite element system.
   */
  void
  initialize();

  /**
   * Retrieves the reference cross-section.
   *
   * @return A constant reference to the reference cross-section object.
   */
  const ReferenceCrossSection<dim - reduced_dim, spacedim, n_components> &
  get_reference_cross_section() const;

  /**
   * Retrieves the DoFHandler for the reduced domain.
   *
   * @return A constant reference to the DoFHandler object.
   */
  const DoFHandler<reduced_dim, spacedim> &
  get_dof_handler() const;

private:
  /**
   * Sets up the degrees of freedom (DoFs) for the reduced domain.
   *
   * This function initializes the DoFHandler and associates it with the finite
   * element system.
   */
  void
  setup_dofs();

  /**
   * The parameters defining the tensor product space.
   *
   * This object contains all the necessary configuration for the tensor product
   * space.
   */
  const TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components>
    &par;

  /**
   * The reference cross-section for the reduced domain.
   *
   * This object represents the cross-section of the reduced-dimensional domain.
   */
  ReferenceCrossSection<cross_section_dim, spacedim, n_components>
    reference_cross_section;

  /**
   * The triangulation representing the reduced domain.
   *
   * This object holds the mesh for the reduced-dimensional domain.
   */
  Triangulation<reduced_dim, spacedim> triangulation;

  /**
   * The finite element system used for the reduced domain.
   *
   * This object defines the finite element basis functions for the
   * reduced-dimensional domain.
   */
  FESystem<reduced_dim, spacedim> fe;

  /**
   * The DoFHandler for the reduced domain.
   *
   * This object manages the degrees of freedom for the reduced-dimensional
   * domain.
   */
  DoFHandler<reduced_dim, spacedim> dof_handler;
};


#endif // tensor_product_space_h
