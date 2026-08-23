// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/lac/full_matrix.h>

#include <gtest/gtest.h>
#include <immersx/core/contributor.h>
#include <immersx/core/lifting.h>
#include <immersx/core/representation.h>
#include <immersx/core/state.h>
#include <immersx/coupling/coupling_operator.h>

#include <type_traits>

using namespace ImmersX;


using IdentityLine       = IdentityRepresentation<1, 2>;
using CylindricalSurface = TensorProductRepresentation<1, 2, 3, 1>;
using VectorField        = VectorFiniteElementRepresentation<2>;

static_assert(RepresentationConcept<IdentityLine>::value,
              "The direct FE representation must satisfy the interaction "
              "contract.");
static_assert(RepresentationConcept<CylindricalSurface>::value,
              "The tensor-product representation must satisfy the "
              "interaction contract.");
static_assert(RepresentationConcept<VectorField>::value,
              "The vector FE representation must satisfy the interaction "
              "contract.");
static_assert(std::is_same<typename IdentityLine::value_type, double>::value,
              "The identity representation remains scalar.");
static_assert(
  std::is_same<typename VectorField::value_type, dealii::Tensor<1, 2>>::value,
  "The vector representation must expose a typed physical value.");


TEST(RepresentationContract, IdentityAndTensorProductDimensions) // NOLINT
{
  EXPECT_EQ(IdentityLine::support_dimension, 1u);
  EXPECT_EQ(IdentityLine::representative_dimension, 1u);
  EXPECT_EQ(IdentityLine::ambient_dimension, 2u);

  EXPECT_EQ(CylindricalSurface::support_dimension, 2u);
  EXPECT_EQ(CylindricalSurface::representative_dimension, 1u);
  EXPECT_EQ(CylindricalSurface::ambient_dimension, 3u);

  EXPECT_EQ(VectorField::support_dimension, 2u);
  EXPECT_EQ(VectorField::representative_dimension, 2u);
  EXPECT_EQ(VectorField::ambient_dimension, 2u);
}

TEST(Representation, IdentityDomain) // NOLINT
{
  using Vector = dealii::Vector<double>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor descriptor;
  descriptor.name        = "temperature";
  const auto temperature = layout.add_field(descriptor);

  Vector values(3);
  values[0] = 1.;
  values[1] = 2.;
  values[2] = 3.;
  ImmersX::StateView<Vector> state_view(layout, 0.);
  state_view.bind(temperature, values);
  const ImmersX::EvaluationContext<Vector> context(0., state_view, nullptr);

  ImmersX::Representation<Vector> representation(temperature);
  EXPECT_EQ(representation.source(), temperature);
  EXPECT_EQ(representation.domain(),
            ImmersX::RepresentationDomain::algebraic());
  EXPECT_EQ(representation.quantity_space().domain(), representation.domain());
  EXPECT_EQ(representation.quantity_space().dimension(), 0u);
  EXPECT_EQ(&representation.evaluate(context), &values);

  const auto identity = representation.linearize(context);
  Vector     action(3);
  identity.vmult(action, values);
  EXPECT_EQ(action, values);
}

TEST(Representation, ScaledDomain) // NOLINT
{
  using Vector = dealii::Vector<double>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor descriptor;
  descriptor.name      = "potential";
  const auto potential = layout.add_field(descriptor);

  Vector values(3);
  values[0] = 1.;
  values[1] = 2.;
  values[2] = 3.;
  ImmersX::StateView<Vector> state_view(layout, 0.);
  state_view.bind(potential, values);
  const ImmersX::EvaluationContext<Vector> context(0., state_view, nullptr);

  const ImmersX::RepresentationDomain line_domain(1, 3, "centerline");
  const auto                          pressure =
    ImmersX::Representation<Vector>(potential, line_domain).scaled(2.);
  EXPECT_EQ(pressure.source(), potential);
  EXPECT_EQ(pressure.domain(), line_domain);
  EXPECT_EQ(pressure.domain().dimension, 1u);
  EXPECT_EQ(pressure.domain().spacedim, 3u);
  EXPECT_EQ(pressure.quantity_space().domain(), line_domain);
  EXPECT_EQ(pressure.quantity_space().dimension(), 1u);

  const auto evaluated = pressure.evaluate(context);
  EXPECT_EQ(evaluated[0], 2.);
  EXPECT_EQ(evaluated[1], 4.);
  EXPECT_EQ(evaluated[2], 6.);

  const auto jacobian = pressure.linearize(context);
  Vector     action(3);
  jacobian.vmult(action, values);
  EXPECT_EQ(action[0], 2.);
  EXPECT_EQ(action[1], 4.);
  EXPECT_EQ(action[2], 6.);
}

TEST(Representation, Lifting) // NOLINT
{
  using Vector = dealii::Vector<double>;
  using Point  = dealii::Point<3>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor descriptor;
  descriptor.name        = "line_temperature";
  const auto temperature = layout.add_field(descriptor);

  Vector values(2);
  values[0] = 0.;
  values[1] = 1.;
  ImmersX::StateView<Vector> state_view(layout, 0.);
  state_view.bind(temperature, values);
  const ImmersX::EvaluationContext<Vector> context(0., state_view, nullptr);

  const std::vector<Point> line_points = {Point(0., 0., 0.), Point(1., 0., 0.)};
  const ImmersX::RepresentationDomain   line_domain(1, 3, "centerline");
  const ImmersX::Representation<Vector> line_quantity(temperature, line_domain);

  const std::vector<Point>            surface_points = {Point(0., 1., 0.),
                                                        Point(0., 0., 1.),
                                                        Point(.5, 0., 1.),
                                                        Point(1., -1., 0.)};
  const ImmersX::RepresentationDomain surface_domain(2,
                                                     3,
                                                     "cylindrical-surface");
  const ImmersX::EvaluationRequest    surface_request(
    surface_points, std::string("surface-points"));
  Vector                               target_prototype(4);
  const ImmersX::ParametricGeometryMap cylinder_map({0., 1.}, surface_domain);
  const auto surface_quantity = ImmersX::lift(line_quantity,
                                              cylinder_map,
                                              target_prototype,
                                              surface_request);

  EXPECT_EQ(layout.n_fields(), 1u);
  EXPECT_EQ(surface_quantity.source(), temperature);
  EXPECT_EQ(surface_quantity.domain(), surface_domain);
  EXPECT_EQ(surface_quantity.quantity_space().domain(), surface_domain);
  EXPECT_EQ(surface_quantity.quantity_space().dimension(), 2u);

  const auto lifted_values =
    surface_quantity.evaluate(context, surface_request);
  EXPECT_DOUBLE_EQ(lifted_values[0], 0.);
  EXPECT_DOUBLE_EQ(lifted_values[1], 0.);
  EXPECT_DOUBLE_EQ(lifted_values[2], .5);
  EXPECT_DOUBLE_EQ(lifted_values[3], 1.);

  dealii::FullMatrix<double> weak_form(1, 4);
  weak_form(0, 2) = 1.;
  const auto coupling_matrix =
    ImmersX::payload_free(dealii::linear_operator<Vector, Vector>(weak_form));
  Vector                         target_residual_prototype(1);
  ImmersX::CouplingSpace<Vector> target_space(target_residual_prototype);
  ImmersX::CouplingOperator<Vector, Vector> coupling(coupling_matrix,
                                                     target_space);
  const auto residual = coupling.apply(lifted_values);
  EXPECT_DOUBLE_EQ(residual[0], .5);
}

TEST(Representation, LiftingLinearization) // NOLINT
{
  using Vector = dealii::Vector<double>;
  using Point  = dealii::Point<3>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor descriptor;
  descriptor.name        = "line_temperature";
  const auto temperature = layout.add_field(descriptor);

  Vector values(2);
  values[0] = 0.;
  values[1] = 1.;
  ImmersX::StateView<Vector> state_view(layout, 0.);
  state_view.bind(temperature, values);
  const ImmersX::EvaluationContext<Vector> context(0., state_view, nullptr);

  const std::vector<Point>              surface_points = {Point(0., 1., 0.),
                                                          Point(0., 0., 1.),
                                                          Point(.5, 0., 1.),
                                                          Point(1., -1., 0.)};
  const ImmersX::Representation<Vector> line_quantity(
    temperature, ImmersX::RepresentationDomain(1, 3, "centerline"));
  Vector                              target_prototype(4);
  const ImmersX::RepresentationDomain surface_domain(2,
                                                     3,
                                                     "cylindrical-surface");
  const ImmersX::EvaluationRequest    surface_request(
    surface_points, std::string("surface-points"));
  const ImmersX::ParametricGeometryMap cylinder_map({0., 1.}, surface_domain);
  const auto surface_quantity = ImmersX::lift(line_quantity,
                                              cylinder_map,
                                              target_prototype,
                                              surface_request);

  Vector direction(2);
  direction[0] = 2.;
  direction[1] = 4.;
  Vector action(4);
  surface_quantity.linearize(context, surface_request).vmult(action, direction);

  EXPECT_DOUBLE_EQ(action[0], 2.);
  EXPECT_DOUBLE_EQ(action[1], 2.);
  EXPECT_DOUBLE_EQ(action[2], 3.);
  EXPECT_DOUBLE_EQ(action[3], 4.);
  EXPECT_EQ(surface_quantity.source(), line_quantity.source());
  EXPECT_NE(surface_quantity.domain(), line_quantity.domain());
}

TEST(Representation, DomainIsIndependentOfEvaluationRequest) // NOLINT
{
  using Vector = dealii::Vector<double>;
  using Point  = dealii::Point<3>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor descriptor;
  descriptor.name        = "line_temperature";
  const auto temperature = layout.add_field(descriptor);

  Vector values(2);
  values[0] = 0.;
  values[1] = 1.;
  ImmersX::StateView<Vector> state_view(layout, 0.);
  state_view.bind(temperature, values);
  const ImmersX::EvaluationContext<Vector> context(0., state_view, nullptr);

  const ImmersX::Representation<Vector> line_quantity(
    temperature, ImmersX::RepresentationDomain(1, 3, "centerline"));
  const ImmersX::RepresentationDomain  surface_domain(2,
                                                     3,
                                                     "cylindrical-surface");
  const ImmersX::ParametricGeometryMap geometry({0., 1.}, surface_domain);
  const Vector                         target_prototype(3);
  const ImmersX::EvaluationRequest     request_a({Point(0., 1., 0.),
                                                  Point(.25, 0., 1.),
                                                  Point(1., -1., 0.)},
                                             std::string("surface-points-a"));
  const ImmersX::EvaluationRequest     request_b({Point(.5, 1., 0.),
                                                  Point(.75, 0., 1.),
                                                  Point(1., -1., 0.)},
                                             std::string("surface-points-b"));
  const auto                           surface_quantity =
    ImmersX::lift(line_quantity, geometry, target_prototype, request_a);

  const auto domain_before = surface_quantity.domain();
  const auto values_a      = surface_quantity.evaluate(context, request_a);
  const auto values_b      = surface_quantity.evaluate(context, request_b);

  EXPECT_EQ(surface_quantity.domain(), domain_before);
  EXPECT_EQ(surface_quantity.domain(), surface_domain);
  EXPECT_DOUBLE_EQ(values_a[0], 0.);
  EXPECT_DOUBLE_EQ(values_a[1], .25);
  EXPECT_DOUBLE_EQ(values_a[2], 1.);
  EXPECT_DOUBLE_EQ(values_b[0], .5);
  EXPECT_DOUBLE_EQ(values_b[1], .75);
  EXPECT_DOUBLE_EQ(values_b[2], 1.);
}

TEST(Representation, ScaledLiftingDerivativeComposition) // NOLINT
{
  using StateVector    = dealii::Vector<double>;
  using QuantityVector = dealii::Vector<float>;
  using Point          = dealii::Point<3>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor descriptor;
  descriptor.name        = "line_temperature";
  const auto temperature = layout.add_field(descriptor);

  StateVector values(2);
  values[0] = 0.;
  values[1] = 1.;
  ImmersX::StateView<StateVector> state_view(layout, 0.);
  state_view.bind(temperature, values);
  const ImmersX::EvaluationContext<StateVector> context(0.,
                                                        state_view,
                                                        nullptr);

  const ImmersX::Representation<StateVector> line_quantity(
    temperature, ImmersX::RepresentationDomain(1, 3, "centerline"));
  const auto scaled_quantity = line_quantity.scaled(2.);
  const ImmersX::RepresentationDomain surface_domain(2,
                                                     3,
                                                     "cylindrical-surface");
  const ImmersX::EvaluationRequest    request(
    {Point(0., 1., 0.), Point(.5, 0., 1.), Point(1., -1., 0.)});
  const ImmersX::ParametricGeometryMap geometry({0., 1.}, surface_domain);
  const QuantityVector                 target_prototype(3);
  const auto                           lifted_quantity =
    ImmersX::lift(scaled_quantity, geometry, target_prototype, request);

  const auto source_derivative = scaled_quantity.linearize(context, request);
  const auto transfer_derivative =
    lifted_quantity.value_transfer().linearize(request);
  const auto composed_derivative = transfer_derivative * source_derivative;
  const auto lifted_derivative   = lifted_quantity.linearize(context, request);

  StateVector direction(2);
  direction[0] = 2.;
  direction[1] = 4.;
  QuantityVector composed_action(3);
  QuantityVector lifted_action(3);
  composed_derivative.vmult(composed_action, direction);
  lifted_derivative.vmult(lifted_action, direction);

  EXPECT_EQ(lifted_quantity.source(), line_quantity.source());
  EXPECT_EQ(lifted_quantity.quantity_space().domain(), surface_domain);
  EXPECT_EQ(composed_action, lifted_action);
  EXPECT_FLOAT_EQ(lifted_action[0], 4.f);
  EXPECT_FLOAT_EQ(lifted_action[1], 6.f);
  EXPECT_FLOAT_EQ(lifted_action[2], 8.f);
}

TEST(Representation, GeometryMapIsIndependentOfValueTransfer) // NOLINT
{
  using Vector = dealii::Vector<double>;
  using Point  = dealii::Point<3>;

  const ImmersX::RepresentationDomain  domain(2, 3, "surface");
  const ImmersX::ParametricGeometryMap geometry({0., 1.}, domain);
  const ImmersX::EvaluationRequest     request_a(
    {Point(0., 0., 0.), Point(.5, 0., 0.)});
  const ImmersX::EvaluationRequest request_b(
    {Point(.25, 0., 0.), Point(1., 0., 0.)});
  const Vector                                 target_prototype(2);
  const ImmersX::ValueTransfer<Vector, Vector> transfer_a(geometry,
                                                          target_prototype,
                                                          request_a);
  const ImmersX::ValueTransfer<Vector, Vector> transfer_b(geometry,
                                                          target_prototype,
                                                          request_b);
  Vector                                       source(2);
  source[0] = 0.;
  source[1] = 1.;

  const auto values_a = transfer_a.apply(source);
  const auto values_b = transfer_b.apply(source);

  EXPECT_EQ(geometry.domain(), domain);
  EXPECT_DOUBLE_EQ(geometry.source_parameter(Point(.25, 0., 0.)), .25);
  EXPECT_DOUBLE_EQ(values_a[0], 0.);
  EXPECT_DOUBLE_EQ(values_a[1], .5);
  EXPECT_DOUBLE_EQ(values_b[0], .25);
  EXPECT_DOUBLE_EQ(values_b[1], 1.);
}

TEST(MixedField, ComponentViews) // NOLINT
{
  using Vector = dealii::Vector<double>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor descriptor;
  descriptor.name          = "mixed_state";
  descriptor.locally_owned = dealii::IndexSet(4);
  descriptor.locally_owned.add_range(0, 4);
  descriptor.locally_owned.compress();
  descriptor.differential_components = dealii::IndexSet(4);
  descriptor.differential_components.add_range(0, 2);
  descriptor.differential_components.compress();
  const auto mixed_state = layout.add_field(descriptor);

  Vector values(4);
  values[0] = 10.;
  values[1] = 20.;
  values[2] = 30.;
  values[3] = 40.;
  ImmersX::StateView<Vector> state_view(layout, 0.);
  state_view.bind(mixed_state, values);
  const ImmersX::EvaluationContext<Vector> context(0., state_view, nullptr);

  dealii::IndexSet area_components(4);
  area_components.add_range(0, 2);
  area_components.compress();
  dealii::IndexSet velocity_components(4);
  velocity_components.add_range(2, 4);
  velocity_components.compress();

  const ImmersX::FieldComponentView area_view(mixed_state, area_components);
  const ImmersX::FieldComponentView velocity_view(mixed_state,
                                                  velocity_components);
  const ImmersX::ComponentRepresentation<Vector> area(area_view);
  const ImmersX::ComponentRepresentation<Vector> velocity(velocity_view);

  const auto area_values     = area.evaluate(context);
  const auto velocity_values = velocity.evaluate(context);
  EXPECT_EQ(area_values.size(), 2u);
  EXPECT_EQ(area_values[0], 10.);
  EXPECT_EQ(area_values[1], 20.);
  EXPECT_EQ(velocity_values[2], 30.);
  EXPECT_EQ(velocity_values[3], 40.);
  EXPECT_EQ(&area_values.vector(), &values);
  EXPECT_EQ(&velocity_values.vector(), &values);

  const auto area_jacobian = area.linearize(context);
  Vector     action(4);
  area_jacobian.vmult(action, values);
  EXPECT_EQ(action[0], 10.);
  EXPECT_EQ(action[1], 20.);
  EXPECT_EQ(action[2], 0.);
  EXPECT_EQ(action[3], 0.);
}
