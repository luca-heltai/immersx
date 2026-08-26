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

#ifndef tensor_product_space_h
#define tensor_product_space_h

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/utilities.h>

#include <immersx/core/input_field_selector.h>
#include <immersx/core/reduced_field_catalog.h>
#include <immersx/core/reduced_field_values.h>
#include <immersx/core/symbolic_field_evaluator.h>
#include <immersx/coupling/point_cloud.h>
#include <immersx/coupling/reference_cross_section.h>
#include <immersx/coupling/tensor_product_lift.h>

#include <limits>

namespace ImmersX
{
  using namespace dealii;

  /**
   * A structure to hold parameters for a tensor product space.
   *
   * This structure is used to define the parameters required for constructing a
   * tensor product space, including the dimensionality, refinement level,
   * finite element degree, and reference cross-section parameters.
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
     * Type of 1D quadrature used to build the reduced-domain rule.
     *
     * This is passed to dealii::QuadratureSelector and then repeated through
     * QIterated on each reduced cell.
     */
    std::string quadrature_type = "gauss";

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
     * Number of times to repeat the reduced-domain quadrature formula.
     *
     * A value of 1 keeps the standard Gauss rule. Larger values build an
     * iterated Gauss quadrature on each reduced cell.
     */
    unsigned int n_quadrature_repetitions = 1;

    /** Constant or symbolic expression for the reduced cross-section thickness.
     */
    std::string thickness = "0.01";

    /** Comma-separated scalar reduced-field selectors exposed to expressions.
     */
    std::string input_file_fields = "";

    /**
     * @brief Name of the grid to read from a file.
     */
    std::string reduced_grid_name = "";

    /** Legacy ASCII inclusion geometry, consumed only by the input adapter. */
    std::string inclusions_file = "";

    /** Legacy ASCII coefficient data, consumed only by the input adapter. */
    std::string data_file = "";

    /** Number of legacy scalar Fourier modes per component. */
    unsigned int legacy_n_coefficients = 1;

    /** Optional legacy coefficient selection retained for parameter migration.
     */
    std::vector<unsigned int> legacy_selected_coefficients;

    /** Coefficients used for every legacy record when no data file is given. */
    std::vector<double> legacy_reference_inclusion_data;

    /** Programmatic point input for zero-dimensional representative domains. */
    PointCloud<spacedim> point_cloud;
  };


  /**
   * A class representing a tensor product space combining a representative
   * domain (a positive-dimensional triangulation or a zero-dimensional
   * point/particle collection) and a reference cross-section.
   *
   * @tparam reduced_dim The dimension of the reduced triangulation.
   * @tparam dim The dimension of the full-order object.
   * @tparam spacedim The dimension of the ambient space
   * @tparam n_components The number of components of the problem.
   */
  /**
   * @class TensorProductSpace
   * A class representing a tensor product space for reduced-dimensional
   * problems. Zero-dimensional representative domains are intentionally stored
   * as points/particles rather than degenerate cells, preserving their
   * distinction from genuine embedded one-dimensional representative domains.
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
     * @param mpi_communicator Communicator used for distributed setup.
     */
    TensorProductSpace(const TensorProductSpaceParameters<reduced_dim,
                                                          dim,
                                                          spacedim,
                                                          n_components> &par,
                       MPI_Comm mpi_communicator = MPI_COMM_WORLD);

    /**
     * The dimension of the cross-section of the reduced domain.
     *
     * This is computed as the difference between the full dimension and the
     * reduced dimension.
     */
    static constexpr int cross_section_dim = dim - reduced_dim;

    /**
     * Preprocess serial triangulation before setting up the partitioner.
     */
    std::function<void(Triangulation<reduced_dim, spacedim> &)>
      preprocess_serial_triangulation = [](auto &) {};

    /**
     * Modify the partitioner for the triangulation.
     *
     * This function is used to generate a fully distributed triangulation from
     * a serial triangulation. The user can overload the default behavior to
     * partition the grid differently. This function is called after the serial
     * triangulation is generated, and before the triangulation is copied to the
     * fully distributed triangulation.
     */
    std::function<void(
      parallel::fullydistributed::Triangulation<reduced_dim, spacedim> &)>
      set_partitioner = [](auto &) {};

    /**
     * Initializes the tensor product space.
     *
     * This function sets up the necessary components, such as the DoFHandler
     * and finite element system.
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
     * Build reduced grid and read optional reduced properties from file.
     */
    void
    make_reduced_grid_and_properties();

    /**
     * Retrieves the DoFHandler for the reduced domain.
     *
     * @return A constant reference to the DoFHandler object.
     */
    const DoFHandler<reduced_dim, spacedim> &
    get_dof_handler() const;

    /**
     * Return lifted quadrature points owned by this rank.
     */
    const std::vector<Point<spacedim>> &
    get_locally_owned_qpoints() const;

    /**
     * Return lifted quadrature weights owned by this rank.
     */
    const std::vector<std::vector<double>> &
    get_locally_owned_weights() const;

    /**
     * Return reduced-manifold quadrature points owned by this rank.
     */
    const std::vector<Point<spacedim>> &
    get_locally_owned_reduced_qpoints() const;

    /**
     * Return reduced-manifold quadrature weights owned by this rank.
     */
    const std::vector<std::vector<double>> &
    get_locally_owned_reduced_weights() const;

    /**
     * Return reduced cross-section measures associated with local points.
     */
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


    /**
     * Return DoF indices associated with one reduced cell.
     */
    const std::vector<types::global_dof_index> &
    get_dof_indices(const types::global_cell_index cell_index) const;


    /**
     * Convert a global particle id to a global cell index, and the local
     * quadrature indices on the reduce triangulation and on the cross-section.
     *
     * @param qpoint_index The global quadrature-point index.
     * @return std::tuple<unsigned int, unsigned int, unsigned int> cell_index,
     * q_index, qpoint_index_in_section
     */
    std::tuple<unsigned int, unsigned int, unsigned int>
    particle_id_to_cell_and_qpoint_indices(
      const unsigned int qpoint_index) const;

    std::tuple<unsigned int, unsigned int, unsigned int>
    particle_id_to_representative_indices(
      const unsigned int qpoint_index) const;

    /** Record stable ParticleHandler id to representative mapping after
     * insertion. */
    void
    register_particle_id_mapping();

    /**
     * Return the indices of the quadrature points that are locally owned by the
     * reduced domain.
     *
     * @return IndexSet
     */
    IndexSet
    locally_owned_qpoints() const;


    /**
     * Return the indices of the cells that are required to assemble the
     * coupling matrix.
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
    get_quadrature() const -> const Quadrature<reduced_dim> &;

    /**
     * Compute and cache lifted/reduced quadrature points and weights.
     */
    void
    compute_points_and_weights();

    /**
     * Return distributed reduced triangulation.
     */
    const parallel::fullydistributed::Triangulation<reduced_dim, spacedim> &
    get_triangulation() const;

    /**
     * Return the scaling associated with one reduced cell.
     */
    double
    get_scaling(const unsigned int) const;

    /**
     * Return interpolated reduced-grid properties.
     */
    const LinearAlgebra::distributed::Vector<double> &
    get_properties() const;

    /**
     * Return DoFHandler used to represent reduced-grid properties.
     */
    const DoFHandler<reduced_dim, spacedim> &
    get_properties_dh() const;

    /**
     * Mutable access to the property DoFHandler.
     */
    DoFHandler<reduced_dim, spacedim> &
    get_properties_dh();

    /**
     * Return names of reduced-grid property fields.
     */
    const std::vector<std::string> &
    get_properties_names() const;

    /** Return stable metadata for imported reduced fields. */
    const FieldCatalog &
    get_properties_catalog() const;

    const std::vector<InputFieldBinding> &
    get_properties_bindings() const;

    /** Return the parsed thickness expression, or an empty string for a
     * constant.
     */
    const std::string &
    get_thickness_expression() const;

    const SymbolicFieldEvaluator &
    get_thickness_evaluator() const;

    /** Set the time used when evaluating the Thickness expression. */
    void
    set_time(const double time);

    /**
     * Mutable access to property field names.
     */
    std::vector<std::string> &
    get_properties_names();

    unsigned int
    n_representative_dofs() const;
    IndexSet
    locally_owned_representative_dofs() const;
    IndexSet
    locally_relevant_representative_dofs() const;
    unsigned int
    n_representative_entities() const;
    IndexSet
    locally_owned_representative_entities() const;
    const std::vector<types::global_dof_index> &
    get_representative_dof_indices(types::global_dof_index entity_id) const;
    unsigned int
    n_representative_q_points_per_entity() const;
    unsigned int
    n_representative_dofs_per_entity() const;

  protected:
    /**
     * Sets up the degrees of freedom (DoFs) for the reduced domain.
     *
     * This function initializes the DoFHandler and associates it with the
     * finite element system.
     */
    void
    setup_dofs();

    /**
     * Given a map of processor to local quadrature point indices, return a map
     * of processor to the corresponding global cell indices.
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
     * This object contains all the necessary configuration for the tensor
     * product space.
     */
    const TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components>
      &par;

    /**
     * The reference cross-section for the reduced domain.
     *
     * This object represents the cross-section of the reduced-dimensional
     * domain.
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
    Quadrature<reduced_dim> quadrature_formula;

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

    /**
     * All quadrature points lifted in ambient coordinates.
     */
    std::vector<Point<spacedim>> all_qpoints;
    /**
     * Weights associated with all lifted quadrature points.
     */
    std::vector<std::vector<double>> all_weights;

    /**
     * Quadrature points on the reduced manifold.
     */
    std::vector<Point<spacedim>> reduced_qpoints;
    /**
     * Weights associated with reduced-manifold quadrature points.
     */
    std::vector<std::vector<double>> reduced_weights;

    /**
     * The properties of the inclusion.
     */
    LinearAlgebra::distributed::Vector<double> properties;

    /**
     * The finite element system used for the properties of the inclusion.
     */
    DoFHandler<reduced_dim, spacedim> properties_dh;

    /**
     * The names of the properties stored in the input file.
     */
    std::vector<std::string> properties_names;

    FieldCatalog                   properties_catalog;
    std::vector<InputFieldBinding> properties_bindings;
    std::string                    thickness_expression;
    double                         constant_thickness = 0.01;
    double                         evaluation_time    = 0.;
    SymbolicFieldEvaluator         thickness_evaluator;
  };


  // Template specializations for the TensorProductSpaceParameters



  /** Point-backed specialization for zero-dimensional representative domains.
   */
  template <int dim, int spacedim, int n_components>
  class TensorProductSpace<0, dim, spacedim, n_components>
  {
  public:
    static constexpr int cross_section_dim = dim;


    TensorProductSpace(
      const TensorProductSpaceParameters<0, dim, spacedim, n_components> &par,
      MPI_Comm mpi_communicator = MPI_COMM_WORLD);

    void
    initialize();
    void
    prepare();
    void
    make_reduced_grid_and_properties();
    void
    set_point_cloud(const PointCloud<spacedim> &point_cloud);
    void
    initialize_representative_particle_handler(
      const parallel::TriangulationBase<spacedim> &background_tria,
      const Mapping<spacedim>                     &mapping,
      const std::vector<std::vector<BoundingBox<spacedim>>>
        &global_bounding_boxes);
    const Particles::ParticleHandler<spacedim> &
    get_representative_particles() const;
    void
    register_particle_id_mapping();

    const ReferenceCrossSection<dim, spacedim, n_components> &
    get_reference_cross_section() const;

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

    void
    update_local_dof_indices(const std::map<unsigned int, IndexSet> &);
    const std::vector<types::global_dof_index> &
    get_dof_indices(types::global_cell_index entity_id) const;
    std::tuple<unsigned int, unsigned int, unsigned int>
    particle_id_to_cell_and_qpoint_indices(unsigned int particle_id) const;
    std::tuple<unsigned int, unsigned int, unsigned int>
    particle_id_to_representative_indices(unsigned int particle_id) const;
    IndexSet
    locally_owned_qpoints() const;
    IndexSet
    locally_relevant_indices() const;
    void
    compute_points_and_weights();
    double
    get_scaling(unsigned int) const;

    unsigned int
    n_representative_dofs() const;
    IndexSet
    locally_owned_representative_dofs() const;
    IndexSet
    locally_relevant_representative_dofs() const;
    unsigned int
    n_representative_entities() const;
    IndexSet
    locally_owned_representative_entities() const;
    const std::vector<types::global_dof_index> &
    get_representative_dof_indices(types::global_dof_index entity_id) const;
    unsigned int
    n_representative_q_points_per_entity() const
    {
      return 1;
    }
    unsigned int
    n_representative_dofs_per_entity() const;

    const std::vector<std::string> &
    get_properties_names() const;
    std::vector<std::string> &
    get_properties_names();
    const FieldCatalog &
    get_properties_catalog() const;
    /** Selected imported scalar properties, indexed by entity then binding. */
    const std::vector<std::vector<double>> &
    get_properties() const;
    const std::vector<InputFieldBinding> &
    get_properties_bindings() const;
    const std::string &
    get_thickness_expression() const;
    const SymbolicFieldEvaluator &
    get_thickness_evaluator() const;
    void
    set_time(double time);

    /** Values bound to symbolic expressions for one local representative
     * entity.
     */
    std::vector<double>
    get_entity_property_values(unsigned int entity_id) const;
    double
    get_entity_thickness(unsigned int entity_id) const;
    const Point<spacedim> &
    get_entity_position(unsigned int entity_id) const;
    Tensor<1, spacedim>
    get_entity_orientation(unsigned int entity_id) const;

  protected:
    MPI_Comm mpi_communicator;
    const TensorProductSpaceParameters<0, dim, spacedim, n_components> &par;
    ReferenceCrossSection<dim, spacedim, n_components> reference_cross_section;
    PointCloud<spacedim>                               point_cloud;
    std::map<types::global_cell_index, std::vector<types::global_dof_index>>
                                     representative_entity_to_dof_indices;
    std::vector<Point<spacedim>>     all_qpoints;
    std::vector<std::vector<double>> all_weights;
    std::vector<Point<spacedim>>     reduced_qpoints;
    std::vector<std::vector<double>> reduced_weights;
    std::vector<std::vector<double>> section_measure;
    std::map<types::particle_index,
             std::tuple<unsigned int, unsigned int, unsigned int>>
                                         particle_id_to_representative;
    Particles::ParticleHandler<spacedim> representative_particles;
    bool                      representative_handler_initialized = false;
    unsigned int              n_global_representative_entities   = 0;
    std::vector<unsigned int> source_entity_ids;
    // These vectors contain only the source-local input until the
    // representative particle handler is initialized. Runtime geometry and
    // metadata then live exclusively in representative_particles.
    std::vector<std::vector<double>>       representative_properties;
    IndexSet                               relevant_representative_entities;
    std::vector<unsigned int>              lifted_entity_ids;
    std::vector<unsigned int>              lifted_section_indices;
    std::vector<std::vector<unsigned int>> all_lifted_entity_ids;
    std::vector<std::vector<unsigned int>> all_lifted_section_indices;
    std::vector<std::string>               properties_names;
    FieldCatalog                           properties_catalog;
    std::vector<InputFieldBinding>         properties_bindings;
    std::vector<std::vector<double>>       entity_properties;
    std::vector<double>                    entity_thickness;
    std::string                            thickness_expression;
    double                                 constant_thickness = 0.01;
    double                                 evaluation_time    = 0.;
    SymbolicFieldEvaluator                 thickness_evaluator;
  };

} // namespace ImmersX

#endif // tensor_product_space_h
