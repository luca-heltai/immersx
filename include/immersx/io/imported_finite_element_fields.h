// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_imported_finite_element_fields_h
#define immersx_imported_finite_element_fields_h

#include <deal.II/base/index_set.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/reduced_field_catalog.h>
#include <immersx/core/representation.h>

#include <memory>
#include <string>

namespace ImmersX
{
  /**
   * A reusable collection of frozen finite-element fields imported from VTK.
   *
   * The target triangulation is supplied by the owning Problem and must already
   * describe the same mesh as the VTK file.  The object owns the imported
   * DoFHandler, finite element, catalog, and distributed coefficients, while
   * its field views share those coefficients without copying them.
   */
  template <int dim, int spacedim = dim>
  class ImportedFiniteElementFields
  {
  public:
    using VectorType = ImmersXLA::MPI::Vector;
    using TriangulationType =
      dealii::parallel::TriangulationBase<dim, spacedim>;
    using DoFHandlerType = dealii::DoFHandler<dim, spacedim>;
    using Representation = FiniteElementRepresentation<dim, spacedim>;

    /** A non-owning named scalar component view of one imported field. */
    class FieldView
    {
    public:
      FieldView() = delete;

      const Representation &
      representation() const
      {
        return representation_;
      }

      const VectorType &
      coefficients() const
      {
        return *coefficients_;
      }

      const ReducedFieldDescriptor &
      descriptor() const
      {
        return descriptor_;
      }

      const std::string &
      name() const
      {
        return descriptor_.name;
      }

      unsigned int
      component() const
      {
        return component_;
      }

    private:
      friend class ImportedFiniteElementFields;

      FieldView(const Representation             &representation,
                std::shared_ptr<const VectorType> coefficients,
                const ReducedFieldDescriptor     &descriptor,
                const unsigned int                component)
        : representation_(representation)
        , coefficients_(std::move(coefficients))
        , descriptor_(descriptor)
        , component_(component)
      {}

      Representation                    representation_;
      std::shared_ptr<const VectorType> coefficients_;
      ReducedFieldDescriptor            descriptor_;
      unsigned int                      component_;
    };

    /** Import all VTK fields onto an already prepared Problem triangulation. */
    ImportedFiniteElementFields(const std::string       &vtk_filename,
                                const TriangulationType &triangulation,
                                MPI_Comm communicator = MPI_COMM_WORLD);

    const TriangulationType &
    triangulation() const
    {
      return *triangulation_;
    }

    const DoFHandlerType &
    dof_handler() const
    {
      return dof_handler_;
    }

    const dealii::FiniteElement<dim, spacedim> &
    finite_element() const
    {
      return dof_handler_.get_fe();
    }

    const dealii::IndexSet &
    locally_owned_dofs() const
    {
      return locally_owned_dofs_;
    }

    const dealii::IndexSet &
    locally_relevant_dofs() const
    {
      return locally_relevant_dofs_;
    }

    const VectorType &
    coefficients() const
    {
      return *coefficients_;
    }

    const FieldCatalog &
    catalog() const
    {
      return catalog_;
    }

    /** Return a scalar component view without copying coefficient data. */
    FieldView
    field(const std::string &name, unsigned int component = 0) const;

    MPI_Comm
    mpi_communicator() const
    {
      return communicator_;
    }

  private:
    const TriangulationType                              *triangulation_;
    DoFHandlerType                                        dof_handler_;
    std::unique_ptr<dealii::FiniteElement<dim, spacedim>> finite_element_;
    dealii::IndexSet                                      locally_owned_dofs_;
    dealii::IndexSet                  locally_relevant_dofs_;
    dealii::AffineConstraints<double> constraints_;
    std::shared_ptr<VectorType>       coefficients_;
    FieldCatalog                      catalog_;
    MPI_Comm                          communicator_;
    Representation                    representation_;
  };

  /** Import a scalar field using the same generic storage type from a Problem.
   */
  template <int dim, int spacedim>
  using ImportedFieldView =
    typename ImportedFiniteElementFields<dim, spacedim>::FieldView;

} // namespace ImmersX

#endif // immersx_imported_finite_element_fields_h
