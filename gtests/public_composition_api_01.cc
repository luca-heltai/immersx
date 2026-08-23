// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/contributor.h>
#include <immersx/core/linear_adapter.h>

namespace PublicCompositionTest
{
  using Vector       = ImmersX::ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;

  struct TestProblem
  {
    dealii::IndexSet owned;
  };

  struct TestFields
  {
    ImmersX::FieldId value;
  };

  template <typename Builder>
  TestFields
  contribute(Builder &builder, const TestProblem &problem)
  {
    return {builder.algebraic_field("value", problem.owned)};
  }

  struct TestQuantity
  {
    ImmersX::FieldId source;

    ImmersX::FieldId
    source_field() const
    {
      return source;
    }

    template <typename Geometry>
    decltype(auto)
    lift(const Geometry &geometry) const
    {
      return ImmersX::detail::invoke_lift(*this, geometry, 0);
    }
  };

  struct TestObservable
  {};

  template <typename ProblemHandle>
  TestQuantity
  make_representation(const ProblemHandle &problem, const TestObservable &)
  {
    return {problem.fields().value};
  }

  struct TestGeometry
  {};

  struct TestLiftedQuantity
  {
    ImmersX::FieldId source;
  };

  TestLiftedQuantity
  make_lift(const TestQuantity &quantity, const TestGeometry &)
  {
    return {quantity.source_field()};
  }

  struct TestCoupling
  {
    double factor;
  };

  struct TestInteraction
  {
    ImmersX::FieldId source;
    ImmersX::FieldId target;
    double           factor;
  };

  struct TestInteractionFields
  {};

  template <typename Quantity, typename ProblemHandle>
  TestInteraction
  make_interaction(const Quantity      &quantity,
                   const ProblemHandle &target,
                   const TestCoupling  &coupling)
  {
    return {quantity.source, target.fields().value, coupling.factor};
  }

  template <typename Builder>
  TestInteractionFields
  contribute(Builder &builder, const TestInteraction &interaction)
  {
    builder.term(interaction.target, "test-coupling")
      .residual([interaction](const auto &context) {
        const auto &source = context.state(interaction.source);
        dealii::PackagedOperation<Vector> result;
        result.reinit_vector = [source](Vector &vector, const bool omit) {
          vector.reinit(source, omit);
        };
        result.apply = [source, interaction](Vector &vector) {
          vector = source;
          vector *= interaction.factor;
        };
        result.apply_add = [source, interaction](Vector &vector) {
          Vector contribution;
          contribution.reinit(source);
          contribution = source;
          contribution *= interaction.factor;
          vector += contribution;
        };
        return result;
      });

    return {};
  }
} // namespace PublicCompositionTest

TEST(PublicComposition, AddObserveLiftAndCouple)
{
  using namespace PublicCompositionTest;
  using Adapter = ImmersX::LinearAdapter<PublicCompositionTest::Vector,
                                         PublicCompositionTest::GlobalVector>;

  dealii::IndexSet owned(2);
  owned.add_range(0, 2);
  owned.compress();

  Adapter adapter(MPI_COMM_WORLD,
                  [](const auto &, const auto &, auto &solution) {
                    solution = 0.;
                  });

  TestProblem problem{owned};
  auto        source = adapter.add(problem, "source");
  auto        target = adapter.add(problem, "target");

  auto quantity = source.observe(TestObservable{});
  auto lifted   = quantity.lift(TestGeometry{});
  adapter.couple(lifted, target, TestCoupling{2.});

  auto state                                  = adapter.make_state();
  adapter.field(state, source.fields().value) = 3.;

  GlobalVector residual;
  adapter.evaluate_residual(state, residual);
  EXPECT_DOUBLE_EQ(adapter.field(residual, target.fields().value)[0], 6.);
  EXPECT_DOUBLE_EQ(adapter.field(residual, target.fields().value)[1], 6.);
}
