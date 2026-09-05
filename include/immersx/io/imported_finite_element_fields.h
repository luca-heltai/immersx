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

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/solution_transfer.h>

#include <boost/signals2/connection.hpp>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/reduced_field_catalog.h>

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
    using SpaceView      = FESpaceView<dim, spacedim>;
    using FieldType = Field<dim, spacedim, dealii::FEValuesExtractors::Scalar>;

  private:
    struct Storage
    {
      using SolutionTransferType =
        dealii::SolutionTransfer<dim, VectorType, spacedim>;

      Storage(const TriangulationType &tria, const MPI_Comm comm)
        : triangulation(&tria)
        , coefficients()
        , communicator(comm)
      {
        auto &mutable_tria = const_cast<TriangulationType &>(tria);
        using DistributedTriangulation =
          dealii::parallel::distributed::Triangulation<dim, spacedim>;
        if (auto *distributed =
              dynamic_cast<DistributedTriangulation *>(&mutable_tria))
          {
            pre_refinement_connection =
              distributed->signals.pre_distributed_refinement.connect(
                [this]() { prepare_for_refinement(); });
            post_refinement_connection =
              distributed->signals.post_distributed_refinement.connect(
                [this]() { complete_refinement(); });
          }
      }

      Storage(const Storage &) = delete;
      Storage &
      operator=(const Storage &) = delete;
      Storage(Storage &&)        = delete;
      Storage &
      operator=(Storage &&) = delete;

      void
      prepare_for_refinement();

      void
      complete_refinement();

      const TriangulationType                     *triangulation;
      std::unique_ptr<OwnedFESpace<dim, spacedim>> space;
      std::unique_ptr<SpaceView>                   space_view;
      std::shared_ptr<VectorType>                  coefficients;
      FieldCatalog                                 catalog;
      MPI_Comm                                     communicator;
      std::shared_ptr<VectorType>                  refinement_input;
      std::shared_ptr<SolutionTransferType>        refinement_transfer;
      boost::signals2::scoped_connection           pre_refinement_connection;
      boost::signals2::scoped_connection           post_refinement_connection;
    };

  public:
    /** A non-owning named scalar component view of one imported field. */
    class FieldView
    {
    public:
      FieldView() = delete;

      FieldType
      field() const
      {
        const auto component_offset =
          descriptor_.first_fe_component + component_;
        return storage_->space_view->field(
          name(), dealii::FEValuesExtractors::Scalar(component_offset));
      }

      const SpaceView &
      space() const
      {
        return *storage_->space_view;
      }

      const TriangulationType &
      triangulation() const
      {
        return *storage_->triangulation;
      }

      const VectorType &
      coefficients() const
      {
        return *storage_->coefficients;
      }

      std::shared_ptr<const VectorType>
      coefficients_handle() const
      {
        return storage_->coefficients;
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

      FieldView(std::shared_ptr<const Storage> storage,
                const ReducedFieldDescriptor  &descriptor,
                const unsigned int             component)
        : storage_(std::move(storage))
        , descriptor_(descriptor)
        , component_(component)
      {}

      std::shared_ptr<const Storage> storage_;
      ReducedFieldDescriptor         descriptor_;
      unsigned int                   component_;
    };

    /** Import all VTK fields onto an already prepared Problem triangulation. */
    ImportedFiniteElementFields(const std::string       &vtk_filename,
                                const TriangulationType &triangulation,
                                MPI_Comm communicator = MPI_COMM_WORLD);

    const TriangulationType &
    triangulation() const
    {
      return *storage_->triangulation;
    }

    const DoFHandlerType &
    dof_handler() const
    {
      return storage_->space->dof_handler();
    }

    const dealii::FiniteElement<dim, spacedim> &
    finite_element() const
    {
      return storage_->space->finite_element();
    }

    const dealii::IndexSet &
    locally_owned_dofs() const
    {
      return storage_->space->locally_owned_dofs();
    }

    const dealii::IndexSet &
    locally_relevant_dofs() const
    {
      return storage_->space->locally_relevant_dofs();
    }

    const VectorType &
    coefficients() const
    {
      return *storage_->coefficients;
    }

    const VectorType &
    coefficients(const std::string &name) const
    {
      (void)field(name);
      return coefficients();
    }

    const SpaceView &
    space() const
    {
      return *storage_->space_view;
    }

    const FieldCatalog &
    catalog() const
    {
      return storage_->catalog;
    }

    /** Return a scalar component view without copying coefficient data. */
    FieldView
    field(const std::string &name, unsigned int component = 0) const;

    MPI_Comm
    mpi_communicator() const
    {
      return storage_->communicator;
    }

  private:
    std::shared_ptr<Storage> storage_;
  };

  /** Import a scalar field using the same generic storage type from a Problem.
   */
  template <int dim, int spacedim>
  using ImportedFieldView =
    typename ImportedFiniteElementFields<dim, spacedim>::FieldView;

} // namespace ImmersX

#endif // immersx_imported_finite_element_fields_h
