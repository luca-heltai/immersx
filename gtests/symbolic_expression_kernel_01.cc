#include <gtest/gtest.h>
#include <immersx/core/symbolic_expression_kernel.h>

using namespace ImmersX;

TEST(SymbolicExpressionKernel, OneIndependentSymbol)
{
  SymbolicExpressionKernel kernel;
#ifdef DEAL_II_WITH_SYMENGINE
  kernel.initialize("A*A + beta*A", {"A"}, {{"beta", 3.0}});

  const auto result = kernel.evaluate(dealii::Point<2>(1.0, -2.0), 0.5, {2.0});
  EXPECT_DOUBLE_EQ(result.value, 10.0);
  ASSERT_EQ(result.derivatives.size(), 1u);
  EXPECT_DOUBLE_EQ(result.derivatives[0], 7.0);
#else
  EXPECT_FALSE(SymbolicExpressionKernel::available());
  EXPECT_THROW(kernel.initialize("A*A + beta*A", {"A"}, {{"beta", 3.0}}),
               std::runtime_error);
#endif
}

TEST(SymbolicExpressionKernel, PreservesIndependentSymbolOrder)
{
  SymbolicExpressionKernel kernel;
#ifdef DEAL_II_WITH_SYMENGINE
  kernel.initialize("A*U + A*A", {"A", "U"});
  const auto result = kernel.evaluate(std::vector<double>{}, 0.0, {2.0, 5.0});
  EXPECT_DOUBLE_EQ(result.value, 14.0);
  ASSERT_EQ(result.derivatives.size(), 2u);
  EXPECT_DOUBLE_EQ(result.derivatives[0], 9.0);
  EXPECT_DOUBLE_EQ(result.derivatives[1], 2.0);
#else
  EXPECT_THROW(kernel.initialize("A*U + A*A", {"A", "U"}), std::runtime_error);
#endif
}

TEST(SymbolicExpressionKernel, PreservesDuplicateDerivativeExpressions)
{
  SymbolicExpressionKernel kernel;
#ifdef DEAL_II_WITH_SYMENGINE
  kernel.initialize("a + b", {"a", "b"});

  const auto result = kernel.evaluate(std::vector<double>{}, 0.0, {2.0, 5.0});
  EXPECT_DOUBLE_EQ(result.value, 7.0);
  ASSERT_EQ(result.derivatives.size(), 2u);
  EXPECT_DOUBLE_EQ(result.derivatives[0], 1.0);
  EXPECT_DOUBLE_EQ(result.derivatives[1], 1.0);
#else
  EXPECT_FALSE(SymbolicExpressionKernel::available());
  EXPECT_THROW(kernel.initialize("a + b", {"a", "b"}), std::runtime_error);
#endif
}

TEST(SymbolicExpressionKernel, CoordinatesTimeAndConstants)
{
  SymbolicExpressionKernel kernel;
#ifdef DEAL_II_WITH_SYMENGINE
  kernel.initialize("x + 2*y + 3*z + 4*t + offset", {}, {{"offset", 7.0}});
  const auto result =
    kernel.evaluate(std::vector<double>{1.0, 2.0, 3.0}, 0.5, {});
  EXPECT_DOUBLE_EQ(result.value, 23.0);
  EXPECT_TRUE(result.derivatives.empty());
#else
  EXPECT_THROW(kernel.initialize("x + t + offset", {}, {{"offset", 1.0}}),
               std::runtime_error);
#endif
}

TEST(SymbolicExpressionKernel, ValidatesSymbolicInputs)
{
  SymbolicExpressionKernel kernel;
  EXPECT_THROW(kernel.initialize("A + unknown", {"A"}), std::runtime_error);
  EXPECT_THROW(kernel.initialize("A", {"bad-name"}), std::runtime_error);
  EXPECT_THROW(kernel.initialize("A", {"A"}, {{"A", 1.0}}), std::runtime_error);

#ifdef DEAL_II_WITH_SYMENGINE
  kernel.initialize("1 / A", {"A"});
  EXPECT_THROW(kernel.evaluate(std::vector<double>{}, 0.0, {0.0}),
               std::runtime_error);
#else
  EXPECT_FALSE(SymbolicExpressionKernel::available());
#endif
}
