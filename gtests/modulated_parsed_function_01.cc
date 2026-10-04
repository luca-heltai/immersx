// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/physics/modulated_parsed_function.h>

#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>

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

TEST(ModulatedParsedFunction, ParameterAcceptorProducesFunctionHandle)
{
  dealii::ParameterAcceptor::clear();
  static_assert(std::is_base_of_v<dealii::ParameterAcceptor,
                                  ImmersX::ModulatedParsedFunction<2>>);

  auto function = std::make_shared<ImmersX::ModulatedParsedFunction<2>>(
    "/Coral parameterized function/");
  std::ostringstream constants;
  constants << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "E=" << dealii::numbers::E << ",PI=" << dealii::numbers::PI;
  std::istringstream parameters("subsection Coral parameterized function\n"
                                "  set Function constants = " +
                                constants.str() +
                                "\n"
                                "  set Function expression = x + E + PI\n"
                                "  set Variable names = x,t\n"
                                "  set Modulation frequency = 0\n"
                                "  set Phase shift = 0\n"
                                "end\n");

  EXPECT_NO_THROW(dealii::ParameterAcceptor::initialize(parameters));
  const auto handle = function->function_handle();
  ASSERT_NE(handle, nullptr);
  EXPECT_DOUBLE_EQ(handle->value(dealii::Point<2>(0.25, 0.)),
                   0.25 + dealii::numbers::E + dealii::numbers::PI);

  dealii::ParameterAcceptor::clear();
}

TEST(ModulatedParsedFunction,
     LightweightExpressionUsesCoordinatesTimeAndConstants)
{
  std::ostringstream constants;
  constants << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "E=" << dealii::numbers::E << ",PI=" << dealii::numbers::PI;
  dealii::FunctionParser<2> function(
    "x + 2*y + 3*t + E + PI",
    constants.str(),
    dealii::FunctionParser<2>::default_variable_names() + ",t");

  const dealii::Point<2> point(0.25, -0.5);
  function.set_time(0.75);
  EXPECT_DOUBLE_EQ(function.value(point),
                   0.25 - 1. + 3. * 0.75 + dealii::numbers::E +
                     dealii::numbers::PI);
}
