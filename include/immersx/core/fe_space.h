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
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/affine_constraints.h>

#include <immersx/core/field.h>

#include <string>
#include <utility>

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
  template <int dim,
            int spacedim,
            typename ValueType,
            typename Extractor = dealii::FEValuesExtractors::Scalar>
  class Field
  {
  public:
    using value_type     = ValueType;
    using extractor_type = Extractor;
    using space_type     = FESpaceView<dim, spacedim>;

    Field(const space_type &space,
          std::string       name,
          const Extractor  &extractor = Extractor(0),
          const FieldId     id        = FieldId())
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

    const dealii::AffineConstraints<double> &
    constraints() const
    {
      return space().constraints();
    }

    const dealii::IndexSet &
    locally_owned_dofs() const
    {
      return space().locally_owned_dofs();
    }

    const dealii::IndexSet &
    locally_relevant_dofs() const
    {
      return space().locally_relevant_dofs();
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
    const space_type *space_;
    std::string       name_;
    FieldId           id_;
    Extractor         extractor_;
  };

  template <int dim, int spacedim, typename Extractor, typename ValueType>
  class FESubspaceView
  {
  public:
    using space_type = FESpaceView<dim, spacedim>;
    using field_type = Field<dim, spacedim, ValueType, Extractor>;

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
    using ScalarField =
      Field<dim, spacedim, double, dealii::FEValuesExtractors::Scalar>;
    using VectorField = Field<dim,
                              spacedim,
                              dealii::Tensor<1, spacedim>,
                              dealii::FEValuesExtractors::Vector>;

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

    ScalarField
    field(const std::string &name) const
    {
      return ScalarField(*this, name);
    }

    ScalarField
    field(StateLayout &layout, const std::string &name) const
    {
      return field(layout, name, dealii::FEValuesExtractors::Scalar(0));
    }

    ScalarField
    field(const FieldId id, const std::string &name) const
    {
      return field(id, name, dealii::FEValuesExtractors::Scalar(0));
    }

    ScalarField
    field(const std::string                        &name,
          const dealii::FEValuesExtractors::Scalar &extractor) const
    {
      return ScalarField(*this, name, extractor);
    }

    ScalarField
    field(StateLayout                              &layout,
          const std::string                        &name,
          const dealii::FEValuesExtractors::Scalar &extractor) const
    {
      const auto id = register_field(layout, name);
      return ScalarField(*this, name, extractor, id);
    }

    ScalarField
    field(const FieldId                             id,
          const std::string                        &name,
          const dealii::FEValuesExtractors::Scalar &extractor) const
    {
      return ScalarField(*this, name, extractor, id);
    }

    VectorField
    field(const std::string                        &name,
          const dealii::FEValuesExtractors::Vector &extractor) const
    {
      return VectorField(*this, name, extractor);
    }

    VectorField
    field(StateLayout                              &layout,
          const std::string                        &name,
          const dealii::FEValuesExtractors::Vector &extractor) const
    {
      const auto id = register_field(layout, name);
      return VectorField(*this, name, extractor, id);
    }

    VectorField
    field(const FieldId                             id,
          const std::string                        &name,
          const dealii::FEValuesExtractors::Vector &extractor) const
    {
      return VectorField(*this, name, extractor, id);
    }

    FESubspaceView<dim, spacedim, dealii::FEValuesExtractors::Scalar, double>
    operator[](const dealii::FEValuesExtractors::Scalar &extractor) const
    {
      return {*this, extractor};
    }

    FESubspaceView<dim,
                   spacedim,
                   dealii::FEValuesExtractors::Vector,
                   dealii::Tensor<1, spacedim>>
    operator[](const dealii::FEValuesExtractors::Vector &extractor) const
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
