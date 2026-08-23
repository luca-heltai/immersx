// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/lac/full_matrix.h>

#include <gtest/gtest.h>
#include <immersx/core/contributor.h>
#include <immersx/core/representation.h>
#include <immersx/coupling/coupling_operator.h>

TEST(CouplingOperator, ComposesRepresentationDerivative)
{
  using Vector = dealii::Vector<double>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor descriptor;
  descriptor.name   = "source";
  const auto source = layout.add_field(descriptor);

  Vector values(2);
  values[0] = 1.;
  values[1] = 2.;
  ImmersX::StateView<Vector> state_view(layout, 0.);
  state_view.bind(source, values);
  const ImmersX::EvaluationContext<Vector> context(0., state_view, nullptr);
  const auto quantity = ImmersX::Representation<Vector>(source).scaled(2.);

  dealii::FullMatrix<double> weak_form(3, 2);
  weak_form(0, 0) = 1.;
  weak_form(0, 1) = 2.;
  weak_form(1, 0) = 3.;
  weak_form(1, 1) = 4.;
  weak_form(2, 0) = 5.;
  weak_form(2, 1) = 6.;
  const auto coupling_matrix =
    ImmersX::payload_free(dealii::linear_operator<Vector, Vector>(weak_form));

  Vector                                    target_prototype(3);
  ImmersX::CouplingSpace<Vector>            target_space(target_prototype);
  ImmersX::CouplingOperator<Vector, Vector> coupling(coupling_matrix,
                                                     target_space);

  const auto residual = coupling.apply(quantity.evaluate(context));
  EXPECT_DOUBLE_EQ(residual[0], 10.);
  EXPECT_DOUBLE_EQ(residual[1], 22.);
  EXPECT_DOUBLE_EQ(residual[2], 34.);

  Vector direction(2);
  direction[0] = 0.5;
  direction[1] = -1.;
  Vector jacobian_action(3);
  (coupling.linearize() * quantity.linearize(context))
    .vmult(jacobian_action, direction);

  const double epsilon = 1.e-5;
  Vector       plus    = values;
  Vector       minus   = values;
  plus.add(epsilon, direction);
  minus.add(-epsilon, direction);
  plus *= 2.;
  minus *= 2.;
  const auto plus_residual  = coupling.apply(plus);
  const auto minus_residual = coupling.apply(minus);
  for (unsigned int i = 0; i < 3; ++i)
    EXPECT_NEAR(jacobian_action[i],
                (plus_residual[i] - minus_residual[i]) / (2. * epsilon),
                1.e-9);
}
