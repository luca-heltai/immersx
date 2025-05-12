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

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/utilities.h>

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
   * The degree of the finite element basis functions.
   *
   * Specifies the polynomial degree of the finite element basis
   * functions used in the tensor product space. Default value is 1.
   */
  unsigned int fe_degree = 1;

  /**
   * Number of quadrature points to be used in the reduced domain.
   *
   * This parameter controls the accuracy of the numerical integration
   * in the reduced domain. If left to zero, the number of quadrature
   * points will be set to the minimum required for the finite element
   * degree.
   */
  unsigned int n_q_points = 0;

  /**
   * Thickness of the inclusion.
   */
  double thickness = 0.01;
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
            &par,
    MPI_Comm mpi_communicator = MPI_COMM_WORLD);

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
  std::function<void(
    parallel::fullydistributed::Triangulation<reduced_dim, spacedim> &)>
    make_reduced_grid;

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

  const std::vector<Point<spacedim>> &
  get_locally_owned_qpoints() const;

  const std::vector<std::vector<double>> &
  get_locally_owned_weights() const;

  const std::vector<Point<spacedim>> &
  get_locally_owned_reduced_qpoints() const;

  const std::vector<std::vector<double>> &
  get_locally_owned_reduced_weights() const;

  const std::vector<std::vector<double>> &
  get_locally_owned_section_measure() const;

  /**
   * Update the relevant local dof_indices.
   *
   * After inserting global particles, this function updates the indices that
   * are required to assemble the coupling matrix.
   */
  void
  update_local_dof_indices(
    const std::map<unsigned int, IndexSet> &remote_q_point_indices);


  const std::vector<types::global_dof_index> &
  get_dof_indices(const types::global_cell_index cell_index) const;


  /**
   * Convert a global particle id to a global cell index, and the local
   * quadrature indices on the reduce triangulation and on the cross-section.
   *
   * @param particle_id The global particle id.
   * @return std::tuple<unsigned int, unsigned int, unsigned int> cell_index,
   * q_index, qpoint_index_in_section
   */
  std::tuple<unsigned int, unsigned int, unsigned int>
  particle_id_to_cell_and_qpoint_indices(const unsigned int qpoint_index) const;


  /**
   * Return the indices of the quadrature points that are locally owned by the
   * reduced domain.
   *
   * @return IndexSet
   */
  IndexSet
  locally_owned_qpoints() const;


  /**
   * Return the indices of the cells that are required to assemble the coupling
   * matrix.
   *
   * @return IndexSet
   */
  IndexSet
  locally_relevant_indices() const;

  /**
   * Retrieve the quadrature formula used in the reduced domain.
   *
   * @return A constant reference to the quadrature formula.
   */
  auto
  get_quadrature() const -> const QGauss<reduced_dim> &;

  void
  compute_points_and_weights();

  const parallel::fullydistributed::Triangulation<reduced_dim, spacedim> &
  get_triangulation() const;

  double
  get_scaling(const unsigned int) const;

private:
  /**
   * Sets up the degrees of freedom (DoFs) for the reduced domain.
   *
   * This function initializes the DoFHandler and associates it with the
   * finite element system.
   */
  void
  setup_dofs();

  /**
   * Given a map of processor to local quadrature point indices, return a map of
   * processor to the corresponding global cell indices.
   */
  std::map<unsigned int, IndexSet>
  local_q_point_indices_to_global_cell_indices(
    const std::map<unsigned int, IndexSet> &local_q_point_indices) const;

  /**
   * The MPI communicator for parallel processing.
   *
   * This object is used to manage communication between different processes
   * in a parallel environment.
   */
  MPI_Comm mpi_communicator;

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
  parallel::fullydistributed::Triangulation<reduced_dim, spacedim>
    triangulation;

  /**
   * The finite element system used for the reduced domain.
   *
   * This object defines the finite element basis functions for the
   * reduced-dimensional domain.
   */
  FESystem<reduced_dim, spacedim> fe;

  /**
   * The quadrature formula used for integration in the reduced domain.
   */
  QGauss<reduced_dim> quadrature_formula;

  /**
   * The DoFHandler for the reduced domain.
   *
   * This object manages the degrees of freedom for the reduced-dimensional
   * domain.
   */
  DoFHandler<reduced_dim, spacedim> dof_handler;

  /**
   * Mapping from global cell index to dof indices.
   */
  std::map<types::global_cell_index, std::vector<types::global_dof_index>>
    global_cell_to_dof_indices;

  std::vector<Point<spacedim>>     all_qpoints;
  std::vector<std::vector<double>> all_weights;

  std::vector<Point<spacedim>>     reduced_qpoints;
  std::vector<std::vector<double>> reduced_weights;
};


// Template specializations for the TensorProductSpaceParameters



#endif // tensor_product_space_h
