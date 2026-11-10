// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception (the "License") as published by the Free
// Software Foundation; either version 3.0 of the License, or (at your option)
// any later version.
//
// ---------------------------------------------------------------------

#ifndef immersx_domain_h
#define immersx_domain_h

#include <deal.II/base/config.h>

#include <deal.II/base/observer_pointer.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>

#include <immersx/core/integration_support.h>
#include <immersx/coupling/tensor_product_lift.h>
#include <immersx/io/imported_finite_element_fields.h>
#include <mpi.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ImmersX
{
  template <int reduced_dim, int surface_dim, int spacedim, int n_components>
  class LiftedIntegrationSupport;

  inline std::string
  domain_subsection(const std::string &subsection)
  {
    if (subsection.empty())
      return "/Domain/";

    std::string normalized = subsection;
    if (normalized.front() != '/')
      normalized.insert(normalized.begin(), '/');
    if (normalized.back() != '/')
      normalized.push_back('/');
    return normalized + "Domain/";
  }

  /** Parameters used to construct an ImmersX computational domain.
   *
   * `name_of_grid` may either be the name understood by
   * `GridGenerator::generate_from_name_and_arguments()` or an existing mesh
   * file. In the latter case the file format is selected from its extension
   * and the second string is ignored.
   */
  template <int dim, int spacedim = dim>
  class DomainParameters : public dealii::ParameterAcceptor
  {
  public:
    explicit DomainParameters(const std::string &subsection = "/Domain/");

    std::string  name_of_grid       = "hyper_cube";
    std::string  arguments_for_grid = "-1: 1: false";
    double       grid_scale         = 1.;
    unsigned int initial_refinement = 0;
    std::string  triangulation_type = "distributed";

    /** For Domain<3,3>: auto detects volume versus closed surface input. */
    std::string mesh_input = "auto";

    /** Gmsh target edge length in input coordinates, before Grid scale. */
    double surface_mesh_size = 1.;

    /** Angle in degrees separating patches when parametrizing a surface. */
    double surface_feature_angle = 40.;

    /** Tetrahedra, or four conforming hexahedra per generated tetrahedron. */
    std::string surface_volume_cells = "tetrahedra";

    /** Read and retain named data arrays from a VTK input mesh. */
    bool read_vtk_fields = false;

    /** Optional VTK cell-data array used to set boundary ids. */
    std::string vtk_boundary_id_field;

    /** Optional VTK cell-data array used to set material ids. */
    std::string vtk_material_id_field;
  };


  /** An MPI-aware computational domain with an owned or observed triangulation.
   */
  template <int dim, int spacedim = dim>
  class Domain
  {
  public:
    using ImportedFields = ImportedFiniteElementFields<dim, spacedim>;

    using DistributedTriangulation =
      dealii::parallel::distributed::Triangulation<dim, spacedim>;
    using FullyDistributedTriangulation =
      dealii::parallel::fullydistributed::Triangulation<dim, spacedim>;
    using TriangulationType =
      dealii::parallel::TriangulationBase<dim, spacedim>;

    explicit Domain(const DomainParameters<dim, spacedim> &parameters,
                    MPI_Comm communicator = MPI_COMM_WORLD);

    /** Observe a triangulation owned by another object.
     *
     * The triangulation and parameters must outlive this Domain. The MPI
     * communicator is taken from the triangulation. A populated mesh is kept
     * unchanged and any requested VTK fields are imported onto that mesh;
     * the input mesh must match it. An empty mesh can instead be initialized
     * with make_grid() or prepare_grid()/finalize_grid().
     */
    Domain(const DomainParameters<dim, spacedim> &parameters,
           TriangulationType                     &triangulation);

    ~Domain();

    Domain(const Domain &) = delete;
    Domain &
    operator=(const Domain &) = delete;

    /** Prepare and finalize a mesh without coupled local refinement. */
    void
    make_grid();

    /** Read the mesh and initial fields before coupled refinement.
     *
     * A fullydistributed domain is prepared on a replicated serial mesh.
     * Set serial=true to do the same for an embedded distributed domain.
     * Call finalize_grid() only after all mesh adjustments. This must happen
     * before attaching application DoFHandlers or requesting field views.
     * Preparing an already populated domain throws rather than discarding it.
     */
    void
    prepare_grid(bool serial = false);

    /** Prepare an existing coarse mesh with its material and boundary ids.
     * Imported VTK fields are read before refinement and transferred with it.
     * The final triangulation must still be empty. Call finalize_grid() after
     * all mesh changes, as for prepare_grid(true).
     */
    void
    prepare_grid(const dealii::Triangulation<dim, spacedim> &coarse_mesh);

    /** Evaluate an imported component on a serial preparation cell.
     * Geometry policies may use retained data before final field views exist.
     * The coefficients follow every refinement through Domain's transfer hooks.
     */
    double
    prepared_field_value(const std::string        &name,
                         const dealii::CellId     &cell,
                         const dealii::Point<dim> &unit_point,
                         unsigned int              component = 0) const;

    /** Partition the prepared mesh and transfer its refined input fields. */
    void
    finalize_grid();

    /** Finalize a fullydistributed mesh with a one-time ownership policy.
     * Apply the policy before transferring retained input fields. The policy
     * is evaluated collectively and is not retained by the Domain.
     */
    void
    finalize_grid(
      const dealii::RepartitioningPolicyTools::Base<dim, spacedim> &policy);

    bool
    grid_is_prepared() const;

    /** Mesh used during preparation; serial or parallel::distributed. */
    dealii::Triangulation<dim, spacedim> &
    refinement_triangulation();

    TriangulationType &
    triangulation();

    const TriangulationType &
    triangulation() const;

    bool
    uses_fully_distributed_triangulation() const;

    MPI_Comm
    communicator() const;

    /** Integrate over all active cells owned by this domain. */
    IntegrationSupport<dim, spacedim>
    support() const
    {
      return IntegrationSupport<dim, spacedim>(triangulation());
    }

    /** Integrate over active cells with the selected material id. */
    IntegrationSupport<dim, spacedim>
    support(const dealii::types::material_id material_id) const
    {
      return IntegrationSupport<dim, spacedim>(triangulation(), material_id);
    }

    /** Build a geometric integration support from a tensor-product lift. */
    template <int surface_dim, int n_components>
    LiftedIntegrationSupport<dim, surface_dim, spacedim, n_components>
    lift(const TensorProductLift<dim, surface_dim, spacedim, n_components>
           &descriptor) const;

    template <int surface_dim, int n_components>
    LiftedIntegrationSupport<dim, surface_dim, spacedim, n_components>
    lift(const TensorProductLift<dim, surface_dim, spacedim, n_components>
                                          &descriptor,
         FixedThicknessEvaluator<spacedim> thickness_evaluator) const;

    /** Integrate over every boundary face of this domain. */
    BoundaryIntegrationSupport<dim, spacedim>
    boundary_support() const
    {
      return BoundaryIntegrationSupport<dim, spacedim>(triangulation());
    }

    /** Integrate over boundary faces with the selected boundary id. */
    BoundaryIntegrationSupport<dim, spacedim>
    boundary_support(const dealii::types::boundary_id boundary_id) const
    {
      return BoundaryIntegrationSupport<dim, spacedim>(triangulation(),
                                                       boundary_id);
    }

    /** Whether VTK data fields have been imported for this domain. */
    bool
    has_vtk_fields() const;

    /** Access the imported VTK fields, if enabled in the parameters. */
    const ImportedFields &
    vtk_fields() const;

    /** Return one scalar component of a named imported VTK field. */
    typename ImportedFields::FieldView
    field(const std::string &name, unsigned int component = 0) const;

  private:
    void
    finalize_grid_impl(
      const dealii::RepartitioningPolicyTools::Base<dim, spacedim> *policy);

    static std::unique_ptr<TriangulationType>
    make_triangulation_storage(
      const DomainParameters<dim, spacedim> &parameters,
      MPI_Comm                               communicator);

    void
    prepare_grid_impl(bool                                        serial,
                      const dealii::Triangulation<dim, spacedim> *coarse_mesh);

    void
    import_vtk_fields_if_requested();

    struct SerialFields;

    const DomainParameters<dim, spacedim>     &parameters_;
    MPI_Comm                                   communicator_;
    std::unique_ptr<TriangulationType>         triangulation_storage_;
    dealii::ObserverPointer<TriangulationType> triangulation_;
    std::unique_ptr<ImportedFields>            vtk_fields_;
    std::unique_ptr<dealii::Triangulation<dim, spacedim>> serial_triangulation_;
    std::unique_ptr<SerialFields>                         serial_fields_;
    bool grid_prepared_ = false;
  };
} // namespace ImmersX

#include <immersx/core/lifted_integration_support.h>

#endif // immersx_domain_h
