// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception (the "License"); either version 3.0 of the
// License, or (at your option) any later version. The full text of the
// License is available in the LICENSE.md file at the top level of the
// ImmersX distribution.
//
// ---------------------------------------------------------------------

#ifndef immersx_fe_space_h
#define immersx_fe_space_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/fe_values_views.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/affine_constraints.h>

#include <immersx/core/field.h>

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ImmersX
{
  template <int dim, int spacedim>
  class FESpaceView;

  /** A named semantic field attached to a non-owning FE-space view.
   *
   * The field contains no coefficients or other state. Its FieldId is supplied
   * by a StateLayout when the field is registered. The overloads that do not
   * receive a StateLayout are useful for describing an FE observable before
   * an execution adapter assigns semantic storage; such fields have an
   * invalid FieldId until they are registered.
   */
  template <int dim, int spacedim, typename Extractor>
  class Field
  {
  public:
    using global_index = dealii::types::global_dof_index;

    using extractor_type = Extractor;
    using view_type  = dealii::FEValuesViews::View<dim, spacedim, Extractor>;
    using value_type = typename view_type::value_type;
    using space_type = FESpaceView<dim, spacedim>;

    Field(const space_type &space,
          std::string       name,
          const Extractor  &extractor,
          const FieldId     id = FieldId())
      : space_(&space)
      , name_(std::move(name))
      , id_(id)
      , extractor_(extractor)
    {
      AssertThrow(!name_.empty(),
                  dealii::ExcMessage("A field must have a name."));
    }

    FieldId
    field_id() const
    {
      return id_;
    }

    FieldId
    id() const
    {
      return field_id();
    }

    /** Return the same semantic field with execution storage assigned. */
    Field
    with_id(const FieldId id) const
    {
      Field result(space(), name_, extractor_, id);
      result.execution_layout_ = execution_layout_;
      return result;
    }

    /**
     * Return a view of this FE field with a separate execution numbering.
     *
     * The FE space and extractor remain unchanged, while the returned field
     * exposes the supplied compact vector layout.  This is useful for a
     * component of a mixed native vector: FE shape functions still use the
     * native DoFHandler, but an execution adapter can store the component in
     * its own vector.  An empty mapping retains the ordinary FE numbering.
     */
    Field
    reindexed(std::string               name,
              const dealii::IndexSet   &locally_owned,
              const dealii::IndexSet   &locally_relevant,
              std::vector<global_index> execution_indices,
              std::shared_ptr<const dealii::AffineConstraints<double>>
                execution_constraints = {}) const
    {
      AssertDimension(execution_indices.size(), dof_handler().n_dofs());
      if (execution_constraints == nullptr)
        {
          auto unconstrained =
            std::make_shared<dealii::AffineConstraints<double>>();
          unconstrained->close();
          execution_constraints = std::move(unconstrained);
        }
      Field result(space(), std::move(name), extractor_);
      result.execution_layout_ = std::make_shared<ExecutionLayout>();
      result.execution_layout_->locally_owned    = locally_owned;
      result.execution_layout_->locally_relevant = locally_relevant;
      result.execution_layout_->indices          = std::move(execution_indices);
      result.execution_layout_->constraints = std::move(execution_constraints);
      return result;
    }

    bool
    is_registered() const
    {
      return id_.is_valid();
    }

    const std::string &
    name() const
    {
      return name_;
    }

    const space_type &
    space() const
    {
      return *space_;
    }

    const Extractor &
    extractor() const
    {
      return extractor_;
    }

    const dealii::DoFHandler<dim, spacedim> &
    dof_handler() const
    {
      return space().dof_handler();
    }

    const dealii::Mapping<dim, spacedim> &
    mapping() const
    {
      return space().mapping();
    }

    const dealii::Triangulation<dim, spacedim> &
    triangulation() const
    {
      return dof_handler().get_triangulation();
    }

    const dealii::parallel::TriangulationBase<dim, spacedim> &
    distributed_triangulation() const
    {
      const auto *distributed = dynamic_cast<
        const dealii::parallel::TriangulationBase<dim, spacedim> *>(
        &dof_handler().get_triangulation());
      AssertThrow(distributed != nullptr,
                  dealii::ExcMessage(
                    "This operation requires a distributed triangulation."));
      return *distributed;
    }

    const dealii::FiniteElement<dim, spacedim> &
    finite_element() const
    {
      return space().finite_element();
    }

    unsigned int
    n_dofs_per_cell() const
    {
      return finite_element().n_dofs_per_cell();
    }

    const dealii::AffineConstraints<double> &
    constraints() const
    {
      return execution_layout_ != nullptr ? *execution_layout_->constraints :
                                            space().constraints();
    }

    const dealii::IndexSet &
    locally_owned_dofs() const
    {
      return execution_layout_ != nullptr ? execution_layout_->locally_owned :
                                            space().locally_owned_dofs();
    }

    const dealii::IndexSet &
    locally_relevant_dofs() const
    {
      return execution_layout_ != nullptr ?
               execution_layout_->locally_relevant :
               space().locally_relevant_dofs();
    }

    /** Map a native FE DoF number to the execution-vector DoF number. */
    global_index
    execution_index(const global_index native_index) const
    {
      if (execution_layout_ == nullptr)
        return native_index;
      AssertIndexRange(native_index, execution_layout_->indices.size());
      return execution_layout_->indices[native_index];
    }

    bool
    has_execution_index(const global_index native_index) const
    {
      return execution_index(native_index) !=
             std::numeric_limits<global_index>::max();
    }

    bool
    is_reindexed() const
    {
      return execution_layout_ != nullptr;
    }

    static constexpr unsigned int
    dimension()
    {
      return dim;
    }

    static constexpr unsigned int
    space_dimension()
    {
      return dimension();
    }

    static constexpr unsigned int
    spacedimension()
    {
      return spacedim;
    }

    static constexpr unsigned int
    spacedim_dimension()
    {
      return spacedimension();
    }

  private:
    struct ExecutionLayout
    {
      dealii::IndexSet                                         locally_owned;
      dealii::IndexSet                                         locally_relevant;
      std::vector<global_index>                                indices;
      std::shared_ptr<const dealii::AffineConstraints<double>> constraints;
    };

    const space_type                *space_;
    std::string                      name_;
    FieldId                          id_;
    Extractor                        extractor_;
    std::shared_ptr<ExecutionLayout> execution_layout_;
  };

  template <int dim, int spacedim, typename Extractor>
  class FESubspaceView
  {
  public:
    using space_type = FESpaceView<dim, spacedim>;
    using field_type = Field<dim, spacedim, Extractor>;

    FESubspaceView(const space_type &space, const Extractor &extractor)
      : space_(&space)
      , extractor_(extractor)
    {}

    const space_type &
    space() const
    {
      return *space_;
    }

    const Extractor &
    extractor() const
    {
      return extractor_;
    }

    field_type
    field(const std::string &name) const
    {
      return field_type(space(), name, extractor_);
    }

    field_type
    field(StateLayout &layout, const std::string &name) const
    {
      const auto id = register_field(layout, name);
      return field_type(space(), name, extractor_, id);
    }

  private:
    FieldId
    register_field(StateLayout &layout, const std::string &name) const
    {
      FieldDescriptor descriptor;
      descriptor.name             = name;
      descriptor.locally_owned    = space().locally_owned_dofs();
      descriptor.locally_relevant = space().locally_relevant_dofs();
      return layout.add_field(std::move(descriptor));
    }

    const space_type *space_;
    Extractor         extractor_;
  };

  /** A thin, non-owning view of an existing deal.II finite-element space. */
  template <int dim, int spacedim = dim>
  class FESpaceView
  {
  public:
    using DoFHandlerType = dealii::DoFHandler<dim, spacedim>;
    using MappingType    = dealii::Mapping<dim, spacedim>;

    FESpaceView(const DoFHandlerType                    &dof_handler,
                const MappingType                       &mapping,
                const dealii::AffineConstraints<double> &constraints,
                const dealii::IndexSet *locally_relevant = nullptr)
      : dof_handler_(&dof_handler)
      , mapping_(&mapping)
      , constraints_(&constraints)
      , locally_relevant_(locally_relevant)
    {}

    const DoFHandlerType &
    dof_handler() const
    {
      return *dof_handler_;
    }

    const MappingType &
    mapping() const
    {
      return *mapping_;
    }

    const dealii::FiniteElement<dim, spacedim> &
    finite_element() const
    {
      return dof_handler().get_fe();
    }

    const dealii::AffineConstraints<double> &
    constraints() const
    {
      return *constraints_;
    }

    const dealii::IndexSet &
    locally_owned_dofs() const
    {
      return dof_handler().locally_owned_dofs();
    }

    const dealii::IndexSet &
    locally_relevant_dofs() const
    {
      return locally_relevant_ != nullptr ? *locally_relevant_ :
                                            locally_owned_dofs();
    }

    MPI_Comm
    mpi_communicator() const
    {
      return dof_handler().get_triangulation().get_mpi_communicator();
    }

    auto
    field(const std::string &name) const
    {
      return field(name, dealii::FEValuesExtractors::Scalar(0));
    }

    auto
    field(StateLayout &layout, const std::string &name) const
    {
      return field(layout, name, dealii::FEValuesExtractors::Scalar(0));
    }

    auto
    field(const FieldId id, const std::string &name) const
    {
      return field(id, name, dealii::FEValuesExtractors::Scalar(0));
    }

    template <typename Extractor>
    Field<dim, spacedim, Extractor>
    field(const std::string &name, const Extractor &extractor) const
    {
      return Field<dim, spacedim, Extractor>(*this, name, extractor);
    }

    template <typename Extractor>
    Field<dim, spacedim, Extractor>
    field(StateLayout       &layout,
          const std::string &name,
          const Extractor   &extractor) const
    {
      const auto id = register_field(layout, name);
      return Field<dim, spacedim, Extractor>(*this, name, extractor, id);
    }

    template <typename Extractor>
    Field<dim, spacedim, Extractor>
    field(const FieldId      id,
          const std::string &name,
          const Extractor   &extractor) const
    {
      return Field<dim, spacedim, Extractor>(*this, name, extractor, id);
    }

    template <typename Extractor>
    FESubspaceView<dim, spacedim, Extractor>
    operator[](const Extractor &extractor) const
    {
      return {*this, extractor};
    }

  private:
    FieldId
    register_field(StateLayout &layout, const std::string &name) const
    {
      FieldDescriptor descriptor;
      descriptor.name             = name;
      descriptor.locally_owned    = locally_owned_dofs();
      descriptor.locally_relevant = locally_relevant_dofs();
      return layout.add_field(std::move(descriptor));
    }

    const DoFHandlerType                    *dof_handler_;
    const MappingType                       *mapping_;
    const dealii::AffineConstraints<double> *constraints_;
    const dealii::IndexSet                  *locally_relevant_;
  };

  /** An owning finite-element discretization with a non-owning view API. */
  template <int dim, int spacedim = dim>
  class OwnedFESpace
  {
  public:
    using TriangulationType =
      dealii::parallel::TriangulationBase<dim, spacedim>;
    using DoFHandlerType = dealii::DoFHandler<dim, spacedim>;

    OwnedFESpace(const TriangulationType &triangulation,
                 std::unique_ptr<dealii::FiniteElement<dim, spacedim>> fe)
      : triangulation_(&triangulation)
      , dof_handler_(const_cast<TriangulationType &>(triangulation))
      , finite_element_(std::move(fe))
    {
      AssertThrow(finite_element_ != nullptr,
                  dealii::ExcMessage(
                    "An owned FE space needs a finite element."));
      distribute_dofs();
    }

    void
    distribute_dofs()
    {
      dof_handler_.distribute_dofs(*finite_element_);
      locally_owned_dofs_ = dof_handler_.locally_owned_dofs();
      locally_relevant_dofs_ =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_);
      constraints_.clear();
      constraints_.close();
    }

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
      return *finite_element_;
    }

    const dealii::AffineConstraints<double> &
    constraints() const
    {
      return constraints_;
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

    template <typename Mapping>
    FESpaceView<dim, spacedim>
    view(const Mapping &mapping) const
    {
      return FESpaceView<dim, spacedim>(dof_handler_,
                                        mapping,
                                        constraints_,
                                        &locally_relevant_dofs_);
    }

  private:
    const TriangulationType                              *triangulation_;
    DoFHandlerType                                        dof_handler_;
    std::unique_ptr<dealii::FiniteElement<dim, spacedim>> finite_element_;
    dealii::IndexSet                                      locally_owned_dofs_;
    dealii::IndexSet                  locally_relevant_dofs_;
    dealii::AffineConstraints<double> constraints_;
  };

  template <int dim, int spacedim = dim>
  FESpaceView<dim, spacedim>
  fe_space(const dealii::DoFHandler<dim, spacedim> &dof_handler,
           const dealii::Mapping<dim, spacedim>    &mapping,
           const dealii::AffineConstraints<double> &constraints,
           const dealii::IndexSet                  *locally_relevant = nullptr)
  {
    return FESpaceView<dim, spacedim>(dof_handler,
                                      mapping,
                                      constraints,
                                      locally_relevant);
  }

  template <int dim, int spacedim = dim>
  FESpaceView<dim, spacedim>
  fe_space(const dealii::DoFHandler<dim, spacedim> &dof_handler,
           const dealii::Mapping<dim, spacedim>    &mapping,
           const dealii::AffineConstraints<double> &constraints,
           const dealii::IndexSet                  &locally_relevant)
  {
    return FESpaceView<dim, spacedim>(dof_handler,
                                      mapping,
                                      constraints,
                                      &locally_relevant);
  }
} // namespace ImmersX

#endif // immersx_fe_space_h
