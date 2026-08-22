#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/algebra/vector_lagrange_multiplier_interaction.h>
#include <immersx/core/contributor.h>

namespace
{
  struct FakeInteraction
  {
    using VectorType = dealii::Vector<double>;
    struct Fields
    {
      ImmersX::FieldId first;
      ImmersX::FieldId second;
      ImmersX::FieldId multiplier;
      std::string      prefix;

      std::string
      term(const char *name) const
      {
        return prefix + "." + name;
      }
    };

    bool
    assembly_is_current() const
    {
      return true;
    }

    const dealii::FullMatrix<double> &
    coupling_matrix() const
    {
      return C;
    }

    const dealii::FullMatrix<double> &
    pairing_matrix() const
    {
      return Q;
    }

    const dealii::IndexSet &
    multiplier_locally_owned_dofs() const
    {
      return owned;
    }

    const dealii::IndexSet &
    multiplier_locally_relevant_dofs() const
    {
      return owned;
    }

    dealii::FullMatrix<double> C;
    dealii::FullMatrix<double> Q;
    dealii::IndexSet           owned;
  };
} // namespace

TEST(ContributorCore, PackagedResidualAndSeparateOperators)
{
  using Vector = dealii::Vector<double>;
  using Model  = ImmersX::SemiDiscreteModel<Vector>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor descriptor;
  descriptor.name        = "heat.temperature";
  const auto temperature = layout.add_field(descriptor);
  Model      model;
  ImmersX::SemidiscreteBuilder<Vector> builder(layout, model);

  dealii::FullMatrix<double> matrix(2, 2);
  matrix(0, 0) = 2.;
  matrix(1, 1) = 3.;
  const auto K =
    ImmersX::payload_free(dealii::linear_operator<Vector, Vector>(matrix));
  builder.add_residual(temperature, "heat", [temperature, K](const auto &ctx) {
    return K * ctx.state_derivative()->field(temperature, ctx.time()) +
           K * ctx.state().field(temperature, ctx.time());
  });
  builder.add_state_operator(temperature, temperature, "diffusion", K);
  builder.add_derivative_operator(temperature, temperature, "mass", K);

  Vector state(2), state_dot(2), residual(2);
  state[0]     = 1.;
  state[1]     = 2.;
  state_dot[0] = 3.;
  state_dot[1] = 4.;
  ImmersX::StateView<Vector> state_view(layout, 0.);
  ImmersX::StateView<Vector> derivative_view(layout, 0.);
  state_view.bind(temperature, state);
  derivative_view.bind(temperature, state_dot);
  const ImmersX::EvaluationContext<Vector> context(0.,
                                                   state_view,
                                                   &derivative_view);
  model.evaluate_row(temperature, context, residual);
  EXPECT_DOUBLE_EQ(residual[0], 8.);
  EXPECT_DOUBLE_EQ(residual[1], 18.);

  Vector action(2);
  model.state_operator(temperature, temperature, context).vmult(action, state);
  EXPECT_DOUBLE_EQ(action[0], 2.);
  EXPECT_DOUBLE_EQ(action[1], 6.);
}

TEST(ContributorCore, InteractionUsesLinearOperators)
{
  using Vector = dealii::Vector<double>;
  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor descriptor;
  descriptor.name                             = "first.velocity";
  const auto first                            = layout.add_field(descriptor);
  descriptor.name                             = "second.velocity";
  const auto                           second = layout.add_field(descriptor);
  ImmersX::SemiDiscreteModel<Vector>   model;
  ImmersX::SemidiscreteBuilder<Vector> builder(layout, model);

  FakeInteraction interaction;
  interaction.C.reinit(2, 2);
  interaction.Q.reinit(2, 2);
  interaction.C(0, 0) = 1.;
  interaction.C(1, 1) = 2.;
  interaction.Q(0, 0) = 3.;
  interaction.Q(1, 1) = 4.;
  interaction.owned   = dealii::IndexSet(2);
  interaction.owned.add_range(0, 2);
  interaction.owned.compress();
  const auto fields =
    ImmersX::vector_lagrange_multiplier(interaction, first, second)(builder,
                                                                    "coupling");
  EXPECT_TRUE(fields.multiplier.is_valid());
  EXPECT_EQ(layout.field(fields.multiplier).time_role,
            ImmersX::TimeRole::algebraic);
}
