// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_coral_register_primitives_h
#define immersx_coral_register_primitives_h

#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe_values_extractors.h>

#include <coral.h>
#include <immersx/core/boundary_conditions.h>
#include <immersx/core/field_contributor.h>

#include <functional>
#include <memory>
#include <string>
#include <type_traits>

namespace ImmersX::Coral
{
  template <int dim, int spacedim>
  inline void
  register_primitive_domain_types()
  {
    using Parameters   = ImmersX::DomainParameters<dim, spacedim>;
    using DomainType   = ImmersX::Domain<dim, spacedim>;
    using DomainHandle = std::shared_ptr<DomainType>;

    register_parameter_type<Parameters>("ImmersX::DomainParameters<" +
                                        dimensions(dim, spacedim) + ">");
    coral::detail::set_type_alias<DomainHandle>(
      "ImmersX::OwnedDomain<" + dimensions(dim, spacedim) + ">");
    coral::NodeObject::register_output_type<DomainHandle>();

    coral::NodeObject::register_function(
      std::function<DomainHandle(const Parameters &)>(
        [](const Parameters &parameters) {
          return std::make_shared<DomainType>(parameters, MPI_COMM_WORLD);
        }),
      {"parameters"},
      coral::RegistryMetadata{"Create domain",
                              "Domain",
                              "Owned domain. " + dimensions(dim, spacedim),
                              "Create a generic owning computational domain."});

    coral::NodeObject::register_function(
      std::function<DomainHandle(DomainHandle &)>([](DomainHandle &domain) {
        domain->make_grid();
        return domain;
      }),
      {"domain"},
      coral::RegistryMetadata{"Generate domain",
                              "Generate domain",
                              "Owned domain. " + dimensions(dim, spacedim),
                              "Generate the mesh in an owned domain."});
  }

  template <int dim, int spacedim>
  inline void
  register_primitive_finite_element_space_types()
  {
    using Parameters   = ImmersX::FiniteElementSpaceParameters<dim, spacedim>;
    using Space        = ImmersX::FiniteElementSpace<dim, spacedim>;
    using SpaceHandle  = std::shared_ptr<Space>;
    using DomainHandle = std::shared_ptr<ImmersX::Domain<dim, spacedim>>;

    coral::NodeObject::register_function(
      std::function<SpaceHandle(const DomainHandle &, const Parameters &)>(
        [](const DomainHandle &domain, const Parameters &parameters) {
          return std::make_shared<Space>(domain->triangulation(), parameters);
        }),
      {"domain", "parameters"},
      coral::RegistryMetadata{
        "Create finite element space",
        "Create finite element space",
        "From domain. " + dimensions(dim, spacedim),
        "Create an owning finite element space directly on a domain."});

    coral::NodeObject::register_function(
      std::function<SpaceHandle(SpaceHandle &, unsigned int, double)>(
        [](SpaceHandle       &space,
           const unsigned int boundary_id,
           const double       value) {
          ImmersX::set_constant_dirichlet_boundary_condition(*space,
                                                             boundary_id,
                                                             value);
          return space;
        }),
      {"space", "boundary_id", "value"},
      coral::RegistryMetadata{
        "Set constant Dirichlet boundary condition",
        "Set constant Dirichlet boundary condition",
        "Finite element space. " + dimensions(dim, spacedim),
        "Configure one constant scalar Dirichlet boundary."});
  }

  template <int dim, int spacedim>
  inline void
  register_primitive_field_operations()
  {
    using Adapter       = LinearAdapterFor<dim, spacedim>;
    using AdapterHandle = std::shared_ptr<Adapter>;
    using ScalarField =
      ImmersX::Field<dim, spacedim, dealii::FEValuesExtractors::Scalar>;
    using Gradient     = std::decay_t<decltype(ImmersX::gradient(
      std::declval<const ScalarField &>()))>;
    using TestField    = std::decay_t<decltype(ImmersX::test(
      std::declval<const ScalarField &>()))>;
    using TestGradient = std::decay_t<decltype(ImmersX::gradient(
      std::declval<const TestField &>()))>;
    using Weak         = std::decay_t<decltype(ImmersX::weak_term(
      std::declval<const Gradient &>(), std::declval<const TestGradient &>()))>;

    coral::detail::set_type_alias<TestField>(
      "ImmersX::TestExpression<" + dimensions(dim, spacedim) + ",Scalar>");
    coral::detail::set_type_alias<TestGradient>(
      "ImmersX::TestGradient<" + dimensions(dim, spacedim) + ">");
    coral::detail::set_type_alias<Weak>("ImmersX::WeakTerm<" +
                                        dimensions(dim, spacedim) +
                                        ",ScalarGradient,ScalarTestGradient>");
    coral::NodeObject::register_output_type<TestField>();
    coral::NodeObject::register_output_type<TestGradient>();
    coral::NodeObject::register_output_type<Weak>();

    coral::NodeObject::register_function(
      std::function<TestField(const ScalarField &)>(
        [](const ScalarField &field) { return ImmersX::test(field); }),
      {"field"},
      coral::RegistryMetadata{
        "Test field",
        "Test field",
        "Scalar test field. " + dimensions(dim, spacedim),
        "Create the residual test expression for a scalar field."});

    coral::NodeObject::register_function(
      std::function<TestGradient(const ScalarField &)>(
        [](const ScalarField &field) {
          return ImmersX::gradient(ImmersX::test(field));
        }),
      {"field"},
      coral::RegistryMetadata{
        "Test gradient",
        "Test gradient",
        "Scalar test field. " + dimensions(dim, spacedim),
        "Take the gradient of a scalar test expression."});

    coral::NodeObject::register_function(
      std::function<
        ScalarField(AdapterHandle &, const ScalarField &, const std::string &)>(
        [](AdapterHandle     &adapter,
           const ScalarField &field,
           const std::string &prefix) {
          return (*adapter)
            .add(ImmersX::algebraic_field(field), prefix)
            .fields();
        }),
      {"adapter", "field", "prefix"},
      coral::RegistryMetadata{
        "Register algebraic field",
        "Register algebraic field",
        "Scalar field. " + dimensions(dim, spacedim),
        "Register an existing scalar Field in a linear execution."});

    coral::NodeObject::register_function(
      std::function<Weak(const Gradient &, const TestGradient &)>(
        [](const Gradient &gradient, const TestGradient &test_gradient) {
          return ImmersX::weak_term(gradient, test_gradient);
        }),
      {"trial_gradient", "test_gradient"},
      coral::RegistryMetadata{"Weak term",
                              "Weak term",
                              "Scalar gradient / scalar test gradient. " +
                                dimensions(dim, spacedim),
                              "Build a generic scalar gradient weak term."});

    coral::NodeObject::register_function(
      std::function<void(AdapterHandle &, const Weak &, const std::string &)>(
        [](AdapterHandle     &adapter,
           const Weak        &weak,
           const std::string &prefix) { (*adapter).add(weak, prefix); }),
      {"adapter", "term", "prefix"},
      coral::RegistryMetadata{"Add weak term to linear execution",
                              "Add weak term",
                              "LinearAdapter",
                              "Add a generic WeakTerm to a linear execution."});
  }

  template <int dim, int spacedim>
  inline void
  register_primitive_types()
  {
    register_primitive_domain_types<dim, spacedim>();
    register_owned_finite_element_space_types<dim, spacedim>();
    register_primitive_finite_element_space_types<dim, spacedim>();
    register_primitive_field_operations<dim, spacedim>();
  }
} // namespace ImmersX::Coral

#endif // immersx_coral_register_primitives_h
