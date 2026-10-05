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
#include <immersx/core/domain.h>
#include <immersx/core/field_contributor.h>
#include <immersx/core/known_term.h>

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
    using StaticField  = typename DomainType::ImportedFields::FieldView;

    register_parameter_type<Parameters>("ImmersX::DomainParameters<" +
                                        dimensions(dim, spacedim) + ">");
    coral::detail::set_type_alias<DomainHandle>(
      "ImmersX::OwnedDomain<" + dimensions(dim, spacedim) + ">");
    coral::NodeObject::register_output_type<DomainHandle>();
    coral::detail::set_type_alias<StaticField>("ImmersX::StaticScalarField<" +
                                               dimensions(dim, spacedim) + ">");
    coral::NodeObject::register_output_type<StaticField>();

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

    coral::NodeObject::register_function(
      std::function<StaticField(const DomainHandle &, const std::string &)>(
        [](const DomainHandle &domain, const std::string &name) {
          return domain->field(name);
        }),
      {"domain", "name"},
      coral::RegistryMetadata{
        "Static scalar field",
        "Static scalar field",
        "Domain field. " + dimensions(dim, spacedim),
        "Extract a named scalar field imported from the domain VTK mesh."});
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
  }

  template <int dim, int spacedim>
  inline void
  register_primitive_field_operations()
  {
    using Adapter       = LinearAdapterFor<dim, spacedim>;
    using AdapterHandle = std::shared_ptr<Adapter>;
    using FieldVector   = ImmersXLA::MPI::Vector;
    using ScalarField =
      ImmersX::Field<dim, spacedim, dealii::FEValuesExtractors::Scalar>;
    using TestExpression         = std::decay_t<decltype(ImmersX::test(
      std::declval<const ScalarField &>()))>;
    using Gradient               = std::decay_t<decltype(ImmersX::gradient(
      std::declval<const ScalarField &>()))>;
    using TestGradientExpression = std::decay_t<decltype(ImmersX::gradient(
      ImmersX::test(std::declval<const ScalarField &>())))>;
    using Weak                   = std::decay_t<decltype(ImmersX::weak_term(
      std::declval<const Gradient &>(),
      std::declval<const TestGradientExpression &>()))>;
    using ParsedRhs              = ImmersX::BoundaryFunction<spacedim>;
    using KnownSource            = ImmersX::KnownSource<double, spacedim>;
    using ParsedWeakTerm         = std::decay_t<
      decltype(ImmersX::known_term(std::declval<const KnownSource &>(),
                                   std::declval<const TestExpression &>()))>;
    using FrozenSource = std::decay_t<
      decltype(ImmersX::frozen(std::declval<const ScalarField &>(),
                               std::declval<const FieldVector &>()))>;
    using StaticWeakTerm = std::decay_t<
      decltype(ImmersX::weak_term(std::declval<const FrozenSource &>(),
                                  std::declval<const TestExpression &>()))>;

    coral::detail::set_type_alias<Weak>("ImmersX::WeakTerm<" +
                                        dimensions(dim, spacedim) +
                                        ",ScalarGradient,"
                                        "ScalarTestExpressionGradient>");
    coral::detail::set_type_alias<ParsedWeakTerm>("ImmersX::KnownTerm<" +
                                                  dimensions(dim, spacedim) +
                                                  ",ScalarParsedFunctionRhs>");
    coral::detail::set_type_alias<StaticWeakTerm>(
      "ImmersX::WeakTerm<" + dimensions(dim, spacedim) +
      ",ScalarStaticFieldRhs,ScalarTestExpression>");
    coral::NodeObject::register_output_type<Weak>();
    coral::NodeObject::register_output_type<ParsedWeakTerm>();
    coral::NodeObject::register_output_type<StaticWeakTerm>();

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
      std::function<Weak(const Gradient &, const TestGradientExpression &)>(
        [](const Gradient               &gradient,
           const TestGradientExpression &test_gradient) {
          return ImmersX::weak_term(gradient, test_gradient);
        }),
      {"trial_gradient", "test_gradient"},
      coral::RegistryMetadata{"Weak term",
                              "Weak term",
                              "Scalar gradient / scalar test expression. " +
                                dimensions(dim, spacedim),
                              "Build a generic scalar gradient weak term."});

    coral::NodeObject::register_function(
      std::function<ParsedWeakTerm(const ParsedRhs &, const TestExpression &)>(
        [](const ParsedRhs &rhs, const TestExpression &test_expression) {
          AssertThrow(rhs.value != nullptr,
                      dealii::ExcMessage(
                        "A parsed RHS must contain a function."));
          const auto  function = rhs.value;
          KnownSource source(
            [function](const typename KnownSource::Context &context) {
              ImmersX::detail::set_function_time(*function, context.time);
              return function->value(context.point);
            });
          return ImmersX::known_term(source, test_expression);
        }),
      {"rhs", "test"},
      coral::RegistryMetadata{
        "Weak term",
        "Weak term",
        "Scalar parsed-function RHS / scalar test. " +
          dimensions(dim, spacedim),
        "Build a known scalar RHS weak term from a parsed function."});

    coral::NodeObject::register_function(
      std::function<StaticWeakTerm(
        const typename ImmersX::Domain<dim, spacedim>::ImportedFields::FieldView
          &,
        const TestExpression &)>(
        [](const auto &rhs, const TestExpression &test_expression) {
          return ImmersX::weak_term(ImmersX::frozen(rhs.field(),
                                                    rhs.coefficients()),
                                    test_expression);
        }),
      {"rhs", "test"},
      coral::RegistryMetadata{
        "Weak term",
        "Weak term",
        "Scalar static-field RHS / scalar test. " + dimensions(dim, spacedim),
        "Build a known scalar RHS weak term from a field imported by a "
        "Domain."});

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

    coral::NodeObject::register_function(
      std::function<
        void(AdapterHandle &, const ParsedWeakTerm &, const std::string &)>(
        [](AdapterHandle        &adapter,
           const ParsedWeakTerm &term,
           const std::string    &prefix) { (*adapter).add(term, prefix); }),
      {"adapter", "term", "prefix"},
      coral::RegistryMetadata{
        "Add weak term to linear execution",
        "Add weak term",
        "Parsed-function RHS. " + dimensions(dim, spacedim),
        "Add a known parsed-function RHS to a linear execution."});

    coral::NodeObject::register_function(
      std::function<
        void(AdapterHandle &, const StaticWeakTerm &, const std::string &)>(
        [](AdapterHandle        &adapter,
           const StaticWeakTerm &term,
           const std::string    &prefix) { (*adapter).add(term, prefix); }),
      {"adapter", "term", "prefix"},
      coral::RegistryMetadata{
        "Add weak term to linear execution",
        "Add weak term",
        "Static-field RHS. " + dimensions(dim, spacedim),
        "Add a known imported-field RHS to a linear execution."});
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
