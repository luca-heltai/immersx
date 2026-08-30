#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/contributor.h>
#include <immersx/core/semidiscrete_pde_models.h>

namespace
{
  struct NonsymmetricPreconditioner
  {
    struct AdditionalData
    {};

    template <typename MatrixType>
    void
    initialize(const MatrixType &, const AdditionalData &)
    {}

    void
    vmult(dealii::Vector<double>       &destination,
          const dealii::Vector<double> &source) const
    {
      destination[0] = 2. * source[0];
      destination[1] = source[1];
    }

    void
    Tvmult(dealii::Vector<double>       &destination,
           const dealii::Vector<double> &source) const
    {
      destination[0] = 3. * source[0];
      destination[1] = source[1];
    }
  };
} // namespace

TEST(ContributorCore, PackagedResidualAndSeparateOperators)
{
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::SparseMatrix<double>;
  using Model  = ImmersX::SemiDiscreteModel<Vector, Matrix>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor descriptor;
  descriptor.name        = "heat.temperature";
  const auto temperature = layout.add_field(descriptor);
  Model      model;
  ImmersX::SemidiscreteBuilder<Vector, Matrix> builder(layout, model);

  dealii::SparsityPattern sparsity(2, 2);
  sparsity.add(0, 0);
  sparsity.add(1, 1);
  sparsity.compress();
  Matrix matrix;
  matrix.reinit(sparsity);
  matrix(0, 0) = 2.;
  matrix(1, 1) = 3.;
  const auto K = builder.matrix_operator(matrix);
  builder.term(temperature, "heat")
    .residual([temperature, K](const auto &ctx) {
      return K.view * ctx.derivative(temperature) +
             K.view * ctx.state(temperature);
    })
    .state(temperature, K)
    .derivative(temperature, K);

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

  const auto materialized =
    model.state_matrix_operator(temperature, temperature, context);
  ASSERT_TRUE(materialized.has_value());
  const auto materialized_matrix = materialized->matrix();
  Vector     materialized_action(2);
  materialized_matrix->vmult(materialized_action, state);
  EXPECT_DOUBLE_EQ(materialized_action[0], action[0]);
  EXPECT_DOUBLE_EQ(materialized_action[1], action[1]);
}

TEST(ContributorCore, MatrixOperatorObservesAndMaterializesLazily)
{
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::SparseMatrix<double>;

  dealii::SparsityPattern sparsity(2, 2);
  sparsity.add(0, 0);
  sparsity.add(1, 1);
  sparsity.compress();
  Matrix matrix;
  matrix.reinit(sparsity);
  matrix.set(0, 0, 2.);
  matrix.set(1, 1, 3.);

  const auto operator_view = ImmersX::matrix_operator<Vector>(matrix);
  matrix.set(0, 0, 5.);
  matrix.set(1, 1, 7.);

  Vector state(2), action(2), materialized_action(2), direct(2);
  state[0] = 11.;
  state[1] = 13.;
  operator_view.view.vmult(action, state);
  matrix.vmult(direct, state);
  EXPECT_EQ(action, direct);

  const auto materialized = operator_view.matrix();
  materialized->vmult(materialized_action, state);
  EXPECT_EQ(materialized_action, direct);

  Matrix destination;
  operator_view.materialize_into_matrix(destination);
  destination.vmult(materialized_action, state);
  EXPECT_EQ(materialized_action, direct);
}

#ifndef NDEBUG
TEST(ContributorCore, MatrixOperatorDetectsDestroyedSource)
{
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::SparseMatrix<double>;

  dealii::SparsityPattern sparsity(1, 1);
  sparsity.add(0, 0);
  sparsity.compress();

  EXPECT_DEATH(
    {
      auto *matrix = new Matrix;
      matrix->reinit(sparsity);
      const auto operator_view = ImmersX::matrix_operator<Vector>(*matrix);
      delete matrix;
      (void)operator_view.matrix();
    },
    ".*");
}
#endif

TEST(ContributorCore, LocalPreconditionerUsesItsTransposeAction)
{
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::FullMatrix<double>;

  Matrix     matrix(2, 2);
  Vector     prototype(2);
  const auto reinit_vector = [prototype](Vector &vector, const bool omit) {
    vector.reinit(prototype, omit);
  };
  const auto preconditioner = ImmersX::
    make_local_preconditioner<Vector, Matrix, NonsymmetricPreconditioner>(
      matrix, reinit_vector);

  Vector source(2), action(2), transpose_action(2);
  source[0] = 5.;
  source[1] = 7.;
  preconditioner.vmult(action, source);
  preconditioner.Tvmult(transpose_action, source);
  EXPECT_DOUBLE_EQ(action[0], 10.);
  EXPECT_DOUBLE_EQ(transpose_action[0], 15.);
  EXPECT_NE(action[0], transpose_action[0]);
}

TEST(ContributorCore, ConstrainedOperatorHasMatchingTranspose)
{
  using Vector = dealii::Vector<double>;
  dealii::FullMatrix<double> matrix(2, 2);
  matrix(0, 0) = 1.;
  matrix(0, 1) = 2.;
  matrix(1, 0) = 3.;
  matrix(1, 1) = 4.;

  dealii::AffineConstraints<double> constraints;
  constraints.add_line(1);
  constraints.close();
  const auto view = dealii::linear_operator<Vector, Vector>(matrix);
  const auto constrained =
    ImmersX::semidiscrete_detail::constrained_operator(view, constraints);

  Vector source(2), forward(2), transpose(2), dual(2);
  source[0] = 5.;
  source[1] = 7.;
  dual[0]   = 11.;
  dual[1]   = 13.;
  constrained.vmult(forward, source);
  constrained.Tvmult(transpose, dual);

  EXPECT_DOUBLE_EQ(forward[0], 19.);
  EXPECT_DOUBLE_EQ(forward[1], 0.);
  EXPECT_DOUBLE_EQ(transpose[0], 11.);
  EXPECT_DOUBLE_EQ(transpose[1], 22.);
  EXPECT_DOUBLE_EQ(forward * dual, source * transpose);
}

TEST(ContributorCore, MatrixOperatorPreservesCompositions)
{
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::SparseMatrix<double>;
  using Model  = ImmersX::SemiDiscreteModel<Vector, Matrix>;

  ImmersX::StateLayout layout;
  dealii::IndexSet     owned(2);
  owned.add_range(0, 2);
  owned.compress();
  const auto value = layout.add_field({"value", {}, owned, {}, {}});
  Model      model;
  ImmersX::SemidiscreteBuilder<Vector, Matrix> builder(layout, model);

  dealii::SparsityPattern sparsity(2, 2);
  for (unsigned int i = 0; i < 2; ++i)
    for (unsigned int j = 0; j < 2; ++j)
      sparsity.add(i, j);
  sparsity.compress();
  Matrix matrix;
  matrix.reinit(sparsity);
  matrix(0, 0)    = 1.;
  matrix(0, 1)    = 2.;
  matrix(1, 0)    = 3.;
  matrix(1, 1)    = 4.;
  const auto base = builder.matrix_operator(matrix);
  builder.term(value, "composed")
    .state(value, base)
    .state(value, 2. * ImmersX::transpose_operator(base));

  Vector state(2), action(2), materialized_action(2);
  state[0] = 5.;
  state[1] = 7.;
  ImmersX::StateView<Vector> state_view(layout, 0.);
  state_view.bind(value, state);
  const ImmersX::EvaluationContext<Vector> context(0., state_view);

  const auto view = model.state_operator(value, value, context);
  view.vmult(action, state);
  const auto materialized = model.state_matrix_operator(value, value, context);
  ASSERT_TRUE(materialized.has_value());
  materialized->matrix()->vmult(materialized_action, state);
  EXPECT_DOUBLE_EQ(materialized_action[0], action[0]);
  EXPECT_DOUBLE_EQ(materialized_action[1], action[1]);
}

TEST(ContributorCore, TermsAreScopedByContributorPrefix)
{
  using Vector = dealii::Vector<double>;
  using Model  = ImmersX::SemiDiscreteModel<Vector>;

  ImmersX::StateLayout layout;
  Model                model;
  dealii::IndexSet     owned(1);
  owned.add_index(0);
  owned.compress();
  ImmersX::SemidiscreteBuilder<Vector> a_builder(layout, model, "a");
  ImmersX::SemidiscreteBuilder<Vector> b_builder(layout, model, "b");
  const auto a = a_builder.algebraic_field("value", owned);
  const auto b = b_builder.algebraic_field("value", owned);

  a_builder.term(a, "physics").residual([](const auto &context) {
    (void)context;
    dealii::PackagedOperation<Vector> result;
    result.reinit_vector = [](Vector &vector, const bool) { vector.reinit(1); };
    result.apply         = [](Vector &vector) { vector = 1.; };
    result.apply_add     = [](Vector &vector) { vector[0] += 1.; };
    return result;
  });
  b_builder.term(b, "physics").residual([](const auto &context) {
    (void)context;
    dealii::PackagedOperation<Vector> result;
    result.reinit_vector = [](Vector &vector, const bool) { vector.reinit(1); };
    result.apply         = [](Vector &vector) { vector = 2.; };
    result.apply_add     = [](Vector &vector) { vector[0] += 2.; };
    return result;
  });

  Vector a_state(1), b_state(1), a_residual(1), b_residual(1);
  ImmersX::StateView<Vector> state_view(layout, 0.);
  state_view.bind(a, a_state);
  state_view.bind(b, b_state);
  const ImmersX::EvaluationContext<Vector> a_context(
    0., state_view, nullptr, ImmersX::TermSelection::only("a.physics"));
  const ImmersX::EvaluationContext<Vector> b_context(
    0., state_view, nullptr, ImmersX::TermSelection::only("b.physics"));
  model.evaluate_row(a, a_context, a_residual);
  model.evaluate_row(b, a_context, b_residual);
  EXPECT_DOUBLE_EQ(a_residual[0], 1.);
  EXPECT_DOUBLE_EQ(b_residual[0], 0.);
  a_residual = 0.;
  b_residual = 0.;
  model.evaluate_row(a, b_context, a_residual);
  model.evaluate_row(b, b_context, b_residual);
  EXPECT_DOUBLE_EQ(a_residual[0], 0.);
  EXPECT_DOUBLE_EQ(b_residual[0], 2.);
}

TEST(ContributorCore, RegistersSemanticSaddlePointMetadata)
{
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::SparseMatrix<double>;
  using Model  = ImmersX::SemiDiscreteModel<Vector, Matrix>;

  ImmersX::StateLayout layout;
  dealii::IndexSet     owned(1);
  owned.add_index(0);
  owned.compress();
  Model                                        model;
  ImmersX::SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  const auto velocity = builder.differential_field("velocity", owned);
  const auto pressure = builder.algebraic_field("pressure", owned);
  dealii::SparsityPattern sparsity(1, 1);
  sparsity.add(0, 0);
  sparsity.compress();
  Matrix metric;
  metric.reinit(sparsity);
  metric.set(0, 0, 2.);

  builder.saddle_point(pressure, {velocity}, builder.matrix_operator(metric));
  ASSERT_EQ(model.saddle_points().size(), 1u);
  EXPECT_EQ(model.saddle_points().front().multiplier, pressure);
  ASSERT_EQ(model.saddle_points().front().participants.size(), 1u);
  EXPECT_EQ(model.saddle_points().front().participants.front(), velocity);
  ASSERT_TRUE(
    model.has_multiplier_metric(model.saddle_points().front().multiplier));
  ImmersX::StateView<Vector>               state_view(layout, 0.);
  const ImmersX::EvaluationContext<Vector> context(0., state_view);
  const auto registered_metric = model.multiplier_metric(pressure, context);
  ASSERT_TRUE(registered_metric.has_value());
  EXPECT_EQ(registered_metric->source_matrix(), &metric);
}
