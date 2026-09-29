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

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>

#include <immersx/io/imported_finite_element_fields.h>
#include <mpi.h>

#include <memory>
#include <string>
#include <variant>

namespace ImmersX
{
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

    /** Read and retain named data arrays from a VTK input mesh. */
    bool read_vtk_fields = false;

    /** Optional VTK cell-data array used to set boundary ids. */
    std::string vtk_boundary_id_field;

    /** Optional VTK cell-data array used to set material ids. */
    std::string vtk_material_id_field;
  };


  /** An MPI-aware computational domain owning its deal.II triangulation. */
  template <int dim, int spacedim = dim>
  class Domain
  {
  public:
    using ImportedFields = ImportedFiniteElementFields<dim, spacedim>;

    using DistributedTriangulation =
      dealii::parallel::distributed::Triangulation<dim, spacedim>;
    using FullyDistributedTriangulation =
      dealii::parallel::fullydistributed::Triangulation<dim, spacedim>;
    using TriangulationVariant =
      std::variant<DistributedTriangulation, FullyDistributedTriangulation>;
    using TriangulationType =
      dealii::parallel::TriangulationBase<dim, spacedim>;

    explicit Domain(const DomainParameters<dim, spacedim> &parameters,
                    MPI_Comm communicator = MPI_COMM_WORLD);

    ~Domain();

    Domain(const Domain &) = delete;
    Domain &
    operator=(const Domain &) = delete;

    /** Read or generate the mesh and apply scaling and refinement. */
    void
    make_grid();

    TriangulationType &
    triangulation();

    const TriangulationType &
    triangulation() const;

    bool
    uses_fully_distributed_triangulation() const;

    MPI_Comm
    communicator() const;

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
    static TriangulationVariant
    make_triangulation_storage(
      const DomainParameters<dim, spacedim> &parameters,
      MPI_Comm                               communicator);

    void
    import_vtk_fields_if_requested();

    const DomainParameters<dim, spacedim> &parameters_;
    MPI_Comm                               communicator_;
    TriangulationVariant                   triangulation_storage_;
    TriangulationType                     *triangulation_ = nullptr;
    std::unique_ptr<ImportedFields>        vtk_fields_;
  };
} // namespace ImmersX

#endif // immersx_domain_h
