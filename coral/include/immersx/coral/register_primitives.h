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
#include <immersx/algebra/local_preconditioner.h>
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
  template <int spacedim, typename Value>
  inline ImmersX::KnownSource<Value, spacedim>
  known_source_from_boundary_function(
    const ImmersX::BoundaryFunction<spacedim> &boundary_function)
  {
    AssertThrow(boundary_function.value != nullptr,
                dealii::ExcMessage("A known source function cannot be null."));

    const auto function = boundary_function.value;
    return ImmersX::KnownSource<Value, spacedim>(
      [function](const ImmersX::KnownTermContext<spacedim> &context) {
        ImmersX::detail::set_function_time(*function, context.time);
        if constexpr (std::is_arithmetic_v<Value>)
          return function->value(context.point);
        else
          {
            Value result;
            for (unsigned int component = 0; component < spacedim; ++component)
              result[component] = function->value(context.point, component);
            return result;
          }
      });
  }

  template <int dim, int spacedim, typename Extractor>
  inline void
  register_primitive_known_term_operations(const std::string &field_kind)
  {
    using Adapter       = LinearAdapterFor<dim, spacedim>;
    using AdapterHandle = std::shared_ptr<Adapter>;
    using Field         = ImmersX::Field<dim, spacedim, Extractor>;
    using Value         = typename Field::value_type;
    using Test =
      std::decay_t<decltype(ImmersX::test(std::declval<const Field &>()))>;
    using Source = ImmersX::KnownSource<Value, spacedim>;
    using Term =
      std::decay_t<decltype(ImmersX::known_term(std::declval<const Source &>(),
                                                std::declval<const Test &>()))>;
    using BoundaryFunction = ImmersX::BoundaryFunction<spacedim>;

    coral::detail::set_type_alias<Source>("ImmersX::KnownSource<" +
                                          dimensions(dim, spacedim) + "," +
                                          field_kind + ">");
    coral::detail::set_type_alias<Term>("ImmersX::KnownTerm<" +
                                        dimensions(dim, spacedim) + "," +
                                        field_kind + ">");
    coral::NodeObject::register_output_type<Source>();
    coral::NodeObject::register_output_type<Term>();

    coral::NodeObject::register_function(
      std::function<Source(const BoundaryFunction &)>(
        [](const BoundaryFunction &function) {
          return known_source_from_boundary_function<spacedim, Value>(function);
        }),
      {"function"},
      coral::RegistryMetadata{
        "Known source",
        "Known source",
        field_kind + " boundary source. " + dimensions(dim, spacedim),
        "Create a generic context-aware known source from a boundary "
        "function."});

    coral::NodeObject::register_function(
      std::function<Term(const Source &, const Test &)>(
        [](const Source &source, const Test &test_expression) {
          return ImmersX::known_term(source, test_expression);
        }),
      {"source", "test"},
      coral::RegistryMetadata{
        "Known term",
        "Known term",
        field_kind + " test expression. " + dimensions(dim, spacedim),
        "Pair a generic known source with a residual test expression."});

    coral::NodeObject::register_function(
      std::function<Term(const Term &, const unsigned int)>(
        [](const Term &term, const unsigned int boundary_id) {
          return term.on_boundary(boundary_id);
        }),
      {"term", "boundary_id"},
      coral::RegistryMetadata{
        "Boundary known term",
        "Boundary known term",
        field_kind + " field. " + dimensions(dim, spacedim),
        "Restrict a generic known term to one boundary id."});

    coral::NodeObject::register_function(
      std::function<void(AdapterHandle &, const Term &, const std::string &)>(
        [](AdapterHandle     &adapter,
           const Term        &term,
           const std::string &prefix) { (*adapter).add(term, prefix); }),
      {"adapter", "term", "prefix"},
      coral::RegistryMetadata{
        "Add known term to linear execution",
        "Add known term",
        "LinearAdapter",
        "Add a generic known source term to a linear execution."});
  }

  template <int dim, int spacedim>
  inline void
  register_primitive_vector_weak_terms()
  {
    using Adapter       = LinearAdapterFor<dim, spacedim>;
    using AdapterHandle = std::shared_ptr<Adapter>;
    using Field =
      ImmersX::Field<dim, spacedim, dealii::FEValuesExtractors::Vector>;
    using SymmetricGradient = std::decay_t<decltype(ImmersX::symmetric_gradient(
      std::declval<const Field &>()))>;
    using Divergence        = std::decay_t<decltype(ImmersX::divergence(
      std::declval<const Field &>()))>;
    using Test =
      std::decay_t<decltype(ImmersX::test(std::declval<const Field &>()))>;
    using TestSymmetricGradient =
      std::decay_t<decltype(ImmersX::symmetric_gradient(
        std::declval<const Test &>()))>;
    using TestDivergence =
      std::decay_t<decltype(ImmersX::divergence(std::declval<const Test &>()))>;
    using SymmetricGradientWeak = std::decay_t<decltype(ImmersX::weak_term(
      std::declval<const SymmetricGradient &>(),
      std::declval<const TestSymmetricGradient &>()))>;
    using DivergenceWeak        = std::decay_t<
      decltype(ImmersX::weak_term(std::declval<const Divergence &>(),
                                  std::declval<const TestDivergence &>()))>;

    coral::detail::set_type_alias<SymmetricGradientWeak>(
      "ImmersX::WeakTerm<" + dimensions(dim, spacedim) +
      ",VectorSymmetricGradient,VectorTestExpressionSymmetricGradient>");
    coral::detail::set_type_alias<DivergenceWeak>(
      "ImmersX::WeakTerm<" + dimensions(dim, spacedim) +
      ",VectorDivergence,VectorTestExpressionDivergence>");
    coral::NodeObject::register_output_type<SymmetricGradientWeak>();
    coral::NodeObject::register_output_type<DivergenceWeak>();

    coral::NodeObject::register_function(
      std::function<SymmetricGradientWeak(const SymmetricGradient &,
                                          const TestSymmetricGradient &)>(
        [](const SymmetricGradient     &trial,
           const TestSymmetricGradient &test_expression) {
          return ImmersX::weak_term(trial, test_expression);
        }),
      {"trial", "test"},
      coral::RegistryMetadata{
        "Weak term",
        "Weak term",
        "Vector symmetric gradient. " + dimensions(dim, spacedim),
        "Build the generic symmetric-gradient part of a linear elasticity "
        "weak term."});
    coral::NodeObject::register_function(
      std::function<DivergenceWeak(const Divergence &, const TestDivergence &)>(
        [](const Divergence &trial, const TestDivergence &test_expression) {
          return ImmersX::weak_term(trial, test_expression);
        }),
      {"trial", "test"},
      coral::RegistryMetadata{
        "Weak term",
        "Weak term",
        "Vector divergence. " + dimensions(dim, spacedim),
        "Build the generic divergence part of a linear elasticity weak "
        "term."});

    coral::NodeObject::register_function(
      std::function<void(
        AdapterHandle &, const SymmetricGradientWeak &, const std::string &)>(
        [](AdapterHandle               &adapter,
           const SymmetricGradientWeak &term,
           const std::string &prefix) { (*adapter).add(term, prefix); }),
      {"adapter", "term", "prefix"},
      coral::RegistryMetadata{
        "Add weak term to linear execution",
        "Add weak term",
        "Vector symmetric gradient. " + dimensions(dim, spacedim),
        "Add a generic symmetric-gradient weak term to a linear execution."});
    coral::NodeObject::register_function(
      std::function<
        void(AdapterHandle &, const DivergenceWeak &, const std::string &)>(
        [](AdapterHandle        &adapter,
           const DivergenceWeak &term,
           const std::string    &prefix) { (*adapter).add(term, prefix); }),
      {"adapter", "term", "prefix"},
      coral::RegistryMetadata{
        "Add weak term to linear execution",
        "Add weak term",
        "Vector divergence. " + dimensions(dim, spacedim),
        "Add a generic divergence weak term to a linear execution."});
  }

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
          const auto registered_field =
            (*adapter).add(ImmersX::algebraic_field(field), prefix).fields();
          (*adapter).add_preconditioner(
            registered_field.id(),
            [](const auto &linearized_matrix, const auto &reinit_vector) {
              return make_amg_preconditioner(linearized_matrix, reinit_vector);
            });
          return registered_field;
        }),
      {"adapter", "field", "prefix"},
      coral::RegistryMetadata{
        "Register algebraic field",
        "Register algebraic field",
        "Scalar field. " + dimensions(dim, spacedim),
        "Register an existing scalar Field and its AMG local preconditioner "
        "in a linear execution."});

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

    using VectorField =
      ImmersX::Field<dim, spacedim, dealii::FEValuesExtractors::Vector>;
    coral::NodeObject::register_function(
      std::function<
        VectorField(AdapterHandle &, const VectorField &, const std::string &)>(
        [](AdapterHandle     &adapter,
           const VectorField &field,
           const std::string &prefix) {
          return (*adapter)
            .add(ImmersX::algebraic_field(field), prefix)
            .fields();
        }),
      {"adapter", "field", "prefix"},
      coral::RegistryMetadata{
        "Register algebraic field",
        "Register algebraic field",
        "Vector field. " + dimensions(dim, spacedim),
        "Register an existing vector Field in a linear execution."});

    register_primitive_vector_weak_terms<dim, spacedim>();
    register_primitive_known_term_operations<
      dim,
      spacedim,
      dealii::FEValuesExtractors::Scalar>("Scalar");
    register_primitive_known_term_operations<
      dim,
      spacedim,
      dealii::FEValuesExtractors::Vector>("Vector");
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
