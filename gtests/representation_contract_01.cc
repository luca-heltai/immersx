// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <gtest/gtest.h>
#include <immersx/core/representation.h>
#include <immersx/core/state.h>

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

TEST(Representation, Identity) // NOLINT
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
  EXPECT_EQ(&representation.evaluate(context), &values);

  const auto identity = representation.linearize(context);
  Vector     action(3);
  identity.vmult(action, values);
  EXPECT_EQ(action, values);
}
