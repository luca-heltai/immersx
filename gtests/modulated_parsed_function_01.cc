// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/physics/modulated_parsed_function.h>

namespace
{
  template <int spacedim>
  void
  expect_default_zero(const unsigned int             n_components,
                      const dealii::Point<spacedim> &point)
  {
    ImmersX::ModulatedParsedFunction<spacedim> function(
      "/Modulated function test/", n_components);

    dealii::Vector<double> values(n_components);
    EXPECT_NO_THROW(function.vector_value(point, values));
    for (unsigned int component = 0; component < n_components; ++component)
      EXPECT_DOUBLE_EQ(values[component], 0.);
  }
} // namespace

TEST(ModulatedParsedFunction, DefaultExpressionUsesComponentCount)
{
  dealii::ParameterAcceptor::clear();
  expect_default_zero<3>(1, dealii::Point<3>());

  dealii::ParameterAcceptor::clear();
  expect_default_zero<3>(2, dealii::Point<3>());

  dealii::ParameterAcceptor::clear();
  expect_default_zero<3>(3, dealii::Point<3>());

  dealii::ParameterAcceptor::clear();
  expect_default_zero<1>(3, dealii::Point<1>());
}
