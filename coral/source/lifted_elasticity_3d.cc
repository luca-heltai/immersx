// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of the ImmersX application, based on deal.II.
// ---------------------------------------------------------------------
#include <immersx/coral/register.h>
#include <immersx/coupling/lifted_modal_load.h>
#include <immersx/physics/static_elasticity_term.h>

namespace ImmersX::Coral
{
  void
  register_lifted_elasticity_operations()
  {
    using Adapter       = LinearAdapterFor<3, 3>;
    using AdapterHandle = std::shared_ptr<Adapter>;
    using Bulk     = ImmersX::Field<3, 3, dealii::FEValuesExtractors::Vector>;
    using Scalar   = ImmersX::Field<1, 3, dealii::FEValuesExtractors::Scalar>;
    using Space    = ImmersX::FiniteElementSpaceView<1, 3>;
    using Geometry = ImmersX::TensorProductLift<1, 2, 3, 3>;
    using ScalarValue = decltype(ImmersX::value(std::declval<Scalar>()));
    using Lifted      = decltype(ImmersX::lift(std::declval<ScalarValue>(),
                                          std::declval<const Geometry &>()));
    using LiftedTest  = decltype(ImmersX::test(std::declval<Lifted>()));
    using Weak =
      decltype(ImmersX::weak_term(ImmersX::value(std::declval<Bulk>()),
                                  std::declval<LiftedTest>()));
    using Sum         = ImmersX::ConstraintSum<Weak>;
    using Projection  = std::pair<std::shared_ptr<ImmersXLA::MPI::SparseMatrix>,
                                 std::shared_ptr<ImmersXLA::MPI::Vector>>;
    using Interaction = ImmersX::LiftedModalInteraction<Weak>;
    using Law         = ImmersX::IsotropicElasticStress<3>;
    using Energy =
      decltype(ImmersX::isotropic_elasticity_term(std::declval<Bulk>(),
                                                  std::declval<Law>()));
    register_parameter_type<Geometry>("ImmersX::TensorProductLift<1,2,3,3>");
    coral::detail::set_type_alias<Lifted>("ImmersX::ModalVectorLift");
    coral::detail::set_type_alias<LiftedTest>("ImmersX::ModalVectorLiftTest");
    coral::detail::set_type_alias<Weak>("ImmersX::ModalVectorWeakTerm");
    coral::detail::set_type_alias<Sum>("ImmersX::ModalVectorWeakSum");
    coral::detail::set_type_alias<Projection>("ImmersX::ModalProjection");
    coral::detail::set_type_alias<Interaction>(
      "ImmersX::LiftedModalInteraction");
    coral::detail::set_type_alias<Law>("ImmersX::IsotropicElasticStress<3>");
    coral::detail::set_type_alias<Energy>(
      "ImmersX::IsotropicElasticityTerm<3>");
    coral::NodeObject::register_output_type<Lifted>();
    coral::NodeObject::register_output_type<LiftedTest>();
    coral::NodeObject::register_output_type<Weak>();
    coral::NodeObject::register_output_type<Sum>();
    coral::NodeObject::register_output_type<Projection>();
    coral::NodeObject::register_output_type<Interaction>();
    coral::NodeObject::register_output_type<Law>();
    coral::NodeObject::register_output_type<Energy>();
    coral::NodeObject::register_function(
      std::function<Law(double, double)>([](double mu, double lambda) {
        return Law{mu, lambda};
      }),
      {"mu", "lambda"},
      method_metadata("Isotropic elastic law",
                      "Isotropic elastic law",
                      "3D",
                      "Small-strain Lamé constitutive law.",
                      "Elasticity",
                      "law"));
    coral::NodeObject::register_function(
      std::function<Energy(const Bulk &, const Law &)>(
        [](const Bulk &field, const Law &law) {
          return ImmersX::isotropic_elasticity_term(field, law);
        }),
      {"displacement", "law"},
      method_metadata(
        "Elasticity energy term",
        "Elasticity energy term",
        "3D",
        "Construct the affine elasticity weak form from FEValuesViews strain and divergence.",
        "WeakTerm",
        "elasticity"));
    coral::NodeObject::register_function(
      std::function<void(AdapterHandle &, const Energy &)>(
        [](AdapterHandle &adapter, const Energy &term) {
          adapter->add(term, "elasticity");
          adapter->add_preconditioner(
            term.field.id(), [](const auto &matrix, const auto &reinit) {
              return make_amg_preconditioner(matrix, reinit);
            });
        }),
      {"adapter", "term"},
      method_metadata(
        "Add elasticity energy",
        "Add elasticity energy",
        "3D",
        "Add the constitutive weak form and its local AMG policy to generic linear execution.",
        "LinearExecution",
        "add_elasticity"));
    coral::NodeObject::register_function(
      std::function<void(AdapterHandle &,
                         const Bulk &,
                         const ImmersX::ElasticStaticParameters<3> &)>(
        [](AdapterHandle                             &adapter,
           const Bulk                                &field,
           const ImmersX::ElasticStaticParameters<3> &parameters) {
          adapter->add(ImmersX::static_elasticity_robin(field, parameters),
                       "robin");
        }),
      {"adapter", "displacement", "parameters"},
      method_metadata(
        "Add elasticity Robin boundary",
        "Add elasticity Robin boundary",
        "3D",
        "Add sigma*n + beta*u = g using the static elasticity Robin parameters.",
        "LinearExecution",
        "add_robin"));
    coral::NodeObject::register_function(
      std::function<Scalar(const Space &, unsigned int, const std::string &)>(
        [](const Space       &space,
           unsigned int       component,
           const std::string &name) {
          return space.field(name,
                             dealii::FEValuesExtractors::Scalar(component));
        }),
      {"space", "component", "name"},
      method_metadata(
        "Modal coefficient field",
        "Modal coefficient field",
        "1D in 3D",
        "Select an independent scalar coefficient in a modal FESystem.",
        "Field",
        "modal_component"));
    coral::NodeObject::register_function(
      std::function<
        Lifted(const ScalarValue &, const Geometry &, unsigned int)>(
        [](const ScalarValue &value,
           const Geometry    &geometry,
           unsigned int       slot) {
          return ImmersX::lift(value, geometry).mode(slot);
        }),
      {"value", "geometry", "slot"},
      method_metadata(
        "Vector tensor-product lift",
        "Vector tensor-product lift",
        "1D to 2D in 3D",
        "Lift a selected scalar coefficient into its physical vector cross-section frame, preserving amplitude.",
        "Observable",
        "lift"));
    coral::NodeObject::register_function(
      std::function<LiftedTest(const Lifted &)>(
        [](const Lifted &lift) { return ImmersX::test(lift); }),
      {"lift"},
      method_metadata("Lifted vector test",
                      "Lifted vector test",
                      "1D to surface",
                      "Use the physical lifted basis as a residual test.",
                      "TestExpression",
                      "lift"));
    coral::NodeObject::register_function(
      std::function<Weak(const Bulk &, const LiftedTest &)>(
        [](const Bulk &field, const LiftedTest &target) {
          return ImmersX::weak_term(ImmersX::value(field), target);
        }),
      {"displacement", "test"},
      method_metadata(
        "Lifted vector weak term",
        "Lifted vector weak term",
        "3D displacement",
        "Pair background displacement with physical lifted modal tests.",
        "WeakTerm",
        "lifted_vector"));
    coral::NodeObject::register_function(
      std::function<Sum(const Weak &)>([](const Weak &term) {
        Sum sum;
        sum.add(term);
        return sum;
      }),
      {"term"},
      method_metadata("Modal weak sum",
                      "Modal weak sum",
                      "Unified Constraint",
                      "Start a signed sum of lifted modal weak terms.",
                      "ConstraintSum",
                      "modal"));
    coral::NodeObject::register_function(
      std::function<Sum(const Sum &, const Weak &)>(
        [](const Sum &terms, const Weak &term) {
          auto sum = terms;
          sum.add(term);
          return sum;
        }),
      {"sum", "term"},
      method_metadata("Append modal weak term",
                      "Append modal weak term",
                      "Unified Constraint",
                      "Add a component to the shared semantic multiplier row.",
                      "ConstraintSum",
                      "append_modal"));
    coral::NodeObject::register_function(
      std::function<
        Projection(const Lifted &, const Function<3> &, unsigned int)>(
        [](const Lifted      &lift,
           const Function<3> &data,
           unsigned int       component) {
          AssertThrow(data.value != nullptr,
                      dealii::ExcMessage("Modal data function is null."));
          return ImmersX::lifted_modal_load(lift, *data.value, component);
        }),
      {"lift", "data", "component"},
      method_metadata(
        "Project modal data",
        "Project modal data",
        "Section-weighted metric",
        "Assemble the production coefficient load and physical section-weighted metric.",
        "ModalData",
        "project"));
    coral::NodeObject::register_function(
      std::function<Projection(const Projection &, const Projection &)>(
        [](const Projection &left, const Projection &right) {
          auto matrix = std::make_shared<ImmersXLA::MPI::SparseMatrix>();
          matrix->copy_from(*left.first);
          matrix->add(1., *right.first);
          auto rhs = std::make_shared<ImmersXLA::MPI::Vector>(*left.second);
          *rhs += *right.second;
          return Projection{matrix, rhs};
        }),
      {"left", "right"},
      method_metadata(
        "Sum modal projections",
        "Sum modal projections",
        "Section-weighted metric",
        "Combine independent modal components in one FE coefficient space.",
        "ModalData",
        "sum"));
    coral::NodeObject::register_function(
      std::function<Interaction(const Sum &, const Projection &)>(
        [](const Sum &terms, const Projection &projection) {
          return ImmersX::lifted_modal_constraint(terms,
                                                  projection.first,
                                                  *projection.second);
        }),
      {"terms", "projection"},
      method_metadata(
        "Modal multiplier interaction",
        "Modal multiplier interaction",
        "Unified Constraint",
        "Build one nonzero-data constraint with its physical modal metric.",
        "Constraint",
        "modal"));
    coral::NodeObject::register_function(
      std::function<Scalar(AdapterHandle &, const Interaction &)>(
        [](AdapterHandle &adapter, const Interaction &interaction) {
          return adapter->add(interaction, "modal").fields().multiplier;
        }),
      {"adapter", "interaction"},
      method_metadata(
        "Add modal interaction",
        "Add modal interaction",
        "LinearAdapter",
        "Register one multiplier field and both variational rows for all modes.",
        "LinearExecution",
        "add_modal"));
    coral::NodeObject::register_function(
      std::function<Scalar(const Scalar &, const Scalar &)>(
        [](const Scalar &field, const Scalar &registered) {
          return field.with_id(registered.id());
        }),
      {"component", "registered"},
      method_metadata(
        "Registered modal component",
        "Registered modal component",
        "Output field",
        "Use the interaction's semantic state for a selected modal output component.",
        "Field",
        "registered_modal"));
  }
} // namespace ImmersX::Coral
