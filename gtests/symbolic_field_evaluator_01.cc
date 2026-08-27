#include <gtest/gtest.h>
#include <immersx/core/symbolic_field_evaluator.h>

using namespace ImmersX;

TEST(SymbolicFieldEvaluator, ReportsAvailabilityAndEmptyExpressions)
{
  SymbolicFieldEvaluator evaluator;
  evaluator.initialize({}, {"radius"});
  EXPECT_EQ(evaluator.n_outputs(), 0u);
#ifdef DEAL_II_WITH_SYMENGINE
  EXPECT_TRUE(SymbolicFieldEvaluator::available());
  evaluator.initialize({"radius^2 + 5", "radius * hct"}, {"radius", "hct"}, {});
  const auto values = evaluator.evaluate(std::vector<double>{1.0, 2.0, 3.0},
                                         0.0,
                                         std::vector<double>{3.0, 0.4});
  ASSERT_EQ(values.size(), 2u);
  EXPECT_DOUBLE_EQ(values[0], 14.0);
  EXPECT_DOUBLE_EQ(values[1], 1.2);
  evaluator.initialize({"sin(x) + t + offset"}, {}, {{"offset", 2.0}});
  const auto sine =
    evaluator.evaluate(std::vector<double>{0.0, 0.0, 0.0}, 1.0, {});
  ASSERT_EQ(sine.size(), 1u);
  EXPECT_DOUBLE_EQ(sine[0], 3.0);
  evaluator.initialize({"1e-3 + radius"}, {"radius"});
  const auto scientific =
    evaluator.evaluate(std::vector<double>{0.0}, 0.0, {2.0});
  ASSERT_EQ(scientific.size(), 1u);
  EXPECT_DOUBLE_EQ(scientific[0], 2.001);
  EXPECT_THROW(evaluator.initialize({"unknown + 1"}, {"radius"}),
               std::runtime_error);
  evaluator.initialize({"1 / radius"}, {"radius"});
  EXPECT_THROW(evaluator.evaluate(std::vector<double>{0.0}, 0.0, {0.0}),
               std::runtime_error);
#else
  EXPECT_FALSE(SymbolicFieldEvaluator::available());
  EXPECT_THROW(evaluator.initialize({"radius^2"}, {"radius"}),
               std::runtime_error);
#endif
}

TEST(SymbolicFieldEvaluator, PreservesDuplicateOutputExpressions)
{
  SymbolicFieldEvaluator evaluator;
#ifdef DEAL_II_WITH_SYMENGINE
  evaluator.initialize({"0.1", "0.1"}, {});

  EXPECT_EQ(evaluator.n_outputs(), 2u);
  const auto values =
    evaluator.evaluate(std::vector<double>{1.0, 2.0}, 3.0, {});
  ASSERT_EQ(values.size(), 2u);
  EXPECT_DOUBLE_EQ(values[0], 0.1);
  EXPECT_DOUBLE_EQ(values[1], 0.1);
#else
  EXPECT_FALSE(SymbolicFieldEvaluator::available());
  EXPECT_THROW(evaluator.initialize({"0.1", "0.1"}, {}), std::runtime_error);
#endif
}
