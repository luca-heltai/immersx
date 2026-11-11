#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/contributor.h>
#include <immersx/core/semidiscrete_pde_models.h>

#include <memory>

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

TEST(ContributorCore, MPI_LocalPreconditionerUsesCanonicalFieldSpace) // NOLINT
{
  using MatrixType = ImmersX::ImmersXLA::MPI::SparseMatrix;
  using VectorType = ImmersX::ImmersXLA::MPI::Vector;

  const unsigned int n_global = 8;
  const unsigned int rank =
    dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int n_ranks =
    dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  ASSERT_EQ(n_ranks, 2u);

  dealii::IndexSet   owned(n_global);
  const unsigned int first = rank * (n_global / n_ranks);
  owned.add_range(first, first + n_global / n_ranks);
  owned.compress();

  dealii::DynamicSparsityPattern sparsity(n_global, n_global, owned);
  for (const auto index : owned)
    sparsity.add(index, index);
  sparsity.compress();

  MatrixType matrix;
  matrix.reinit(owned, owned, sparsity, MPI_COMM_WORLD);
  for (const auto index : owned)
    matrix.set(index, index, 1.);
  matrix.compress(dealii::VectorOperation::insert);

  const std::function<void(VectorType &, bool)> fallback =
    [owned](VectorType &vector, const bool omit_zeroing) {
      vector.reinit(owned, MPI_COMM_WORLD, omit_zeroing);
    };
  const auto local = ImmersX::make_amg_preconditioner(matrix, fallback);

  VectorType source;
  VectorType destination;
  VectorType expected_source;
  VectorType expected_destination;
  local.reinit_domain_vector(source, false);
  local.reinit_range_vector(destination, false);
  expected_source.reinit(owned, MPI_COMM_WORLD);
  expected_destination.reinit(owned, MPI_COMM_WORLD);

  ImmersX::ImmersXLA::MPI::PreconditionAMG expected;
  expected.initialize(matrix);
  EXPECT_EQ(source.locally_owned_elements(), owned);
  EXPECT_EQ(destination.locally_owned_elements(), owned);
  for (const auto index : source.locally_owned_elements())
    source[index] = static_cast<double>(index + 1);
  for (const auto index : expected_source.locally_owned_elements())
    expected_source[index] = static_cast<double>(index + 1);
  source.compress(dealii::VectorOperation::insert);
  expected_source.compress(dealii::VectorOperation::insert);

  local.vmult(destination, source);
  expected.vmult(expected_destination, expected_source);
  expected_destination -= destination;
  EXPECT_GT(destination.l2_norm(), 0.);
  EXPECT_LT(expected_destination.l2_norm(), 1.e-12);
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

TEST(ContributorCore, OperatorSumsRetainEachContributionOnce)
{
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::SparseMatrix<double>;
  using Model  = ImmersX::SemiDiscreteModel<Vector, Matrix>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor row_descriptor, column_descriptor;
  row_descriptor.name    = "row";
  column_descriptor.name = "column";
  const auto row         = layout.add_field(row_descriptor);
  const auto column      = layout.add_field(column_descriptor);
  Vector     range(2), domain(3);
  domain[0] = 1.;
  domain[1] = 2.;
  domain[2] = 3.;
  ImmersX::StateView<Vector> state(layout, 0.);
  state.bind(row, range);
  state.bind(column, domain);
  const ImmersX::EvaluationContext<Vector> context(0., state);

  dealii::SparsityPattern sparsity(2, 3, 3);
  sparsity.add(0, 0);
  sparsity.add(0, 1);
  sparsity.add(1, 1);
  sparsity.add(1, 2);
  sparsity.compress();
  Matrix matrix;
  matrix.reinit(sparsity);
  matrix(0, 0) = 1.;
  matrix(0, 1) = 2.;
  matrix(1, 1) = 3.;
  matrix(1, 2) = 4.;
  auto leaf    = ImmersX::matrix_operator<Vector>(matrix);
  // A reference per callback copy exposes growing operator trees without
  // depending on allocator statistics or allocating a large physical system.
  const auto lifetime = std::make_shared<int>(0);
  leaf.view.vmult     = [lifetime, action = leaf.view.vmult](auto       &dst,
                                                         const auto &src) {
    action(dst, src);
  };
  Model                                        model;
  ImmersX::SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  constexpr unsigned int                       n_terms = 8;
  for (unsigned int i = 0; i < n_terms; ++i)
    builder.term(row, "sum").state(column, leaf).derivative(column, leaf);

  const auto check = [&](const auto &op) {
    Vector forward, transpose, dual(2);
    op.reinit_range_vector(forward, false);
    op.reinit_domain_vector(transpose, false);
    EXPECT_EQ(forward.size(), 2u);
    EXPECT_EQ(transpose.size(), 3u);
    op.vmult(forward, domain);
    EXPECT_DOUBLE_EQ(forward[0], n_terms * 5.);
    EXPECT_DOUBLE_EQ(forward[1], n_terms * 18.);
    op.vmult_add(forward, domain);
    EXPECT_DOUBLE_EQ(forward[0], n_terms * 10.);
    EXPECT_DOUBLE_EQ(forward[1], n_terms * 36.);
    dual[0] = 2.;
    dual[1] = 3.;
    op.Tvmult(transpose, dual);
    EXPECT_DOUBLE_EQ(transpose[0], n_terms * 2.);
    EXPECT_DOUBLE_EQ(transpose[1], n_terms * 13.);
    EXPECT_DOUBLE_EQ(transpose[2], n_terms * 12.);
    op.Tvmult_add(transpose, dual);
    EXPECT_DOUBLE_EQ(transpose[0], n_terms * 4.);
    EXPECT_DOUBLE_EQ(transpose[1], n_terms * 26.);
    EXPECT_DOUBLE_EQ(transpose[2], n_terms * 24.);
  };
  const auto before = lifetime.use_count();
  {
    const auto op = model.state_operator(row, column, context);
    EXPECT_LE(lifetime.use_count() - before, 8 * n_terms);
    check(op);
    const auto copy = op;
    EXPECT_LE(lifetime.use_count() - before, 8 * n_terms);
    check(copy);
  }
  {
    const auto op = model.derivative_operator(row, column, context);
    EXPECT_LE(lifetime.use_count() - before, 8 * n_terms);
    check(op);
  }
  {
    const auto op = model.state_matrix_operator(row, column, context);
    ASSERT_TRUE(op.has_value());
    EXPECT_LE(lifetime.use_count() - before, 8 * n_terms);
    check(op->view);
    Vector materialized(2);
    op->matrix()->vmult(materialized, domain);
    EXPECT_DOUBLE_EQ(materialized[0], n_terms * 5.);
    EXPECT_DOUBLE_EQ(materialized[1], n_terms * 18.);
  }
  {
    const auto op = model.derivative_matrix_operator(row, column, context);
    ASSERT_TRUE(op.has_value());
    EXPECT_LE(lifetime.use_count() - before, 8 * n_terms);
    check(op->view);
  }
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

TEST(ContributorCore, AbsentRectangularBlocksSkipExpensiveProducts)
{
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::SparseMatrix<double>;
  using Model  = ImmersX::SemiDiscreteModel<Vector, Matrix>;
  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor row_descriptor, column_descriptor;
  row_descriptor.name    = "row";
  column_descriptor.name = "column";
  const auto row         = layout.add_field(row_descriptor);
  const auto column      = layout.add_field(column_descriptor);
  Vector     range(2), domain(3), destination(2);
  domain = 1.;
  ImmersX::StateView<Vector> state(layout, 0.);
  state.bind(row, range);
  state.bind(column, domain);
  ImmersX::EvaluationContext<Vector> context(0., state);
  Model                              model;
  const auto zero = model.state_operator(row, column, context);
  EXPECT_TRUE(zero.is_null_operator);
  EXPECT_TRUE(model.derivative_operator(row, column, context).is_null_operator);
  EXPECT_TRUE(ImmersX::payload_free(zero).is_null_operator);

  unsigned int                   applications = 0;
  dealii::LinearOperator<Vector> expensive;
  expensive.reinit_domain_vector = [](Vector &v, bool) { v.reinit(3); };
  expensive.reinit_range_vector  = expensive.reinit_domain_vector;
  expensive.vmult = [&applications](Vector &dst, const Vector &src) {
    ++applications;
    dst = src;
  };
  expensive.Tvmult   = expensive.vmult;
  const auto product = zero * expensive;
  product.vmult(destination, domain);
  EXPECT_EQ(applications, 0u);
  EXPECT_EQ(destination.l2_norm(), 0.);
  Vector transpose_result;
  product.reinit_domain_vector(transpose_result, false);
  EXPECT_EQ(transpose_result.size(), 3u);
  product.Tvmult(transpose_result, range);
  EXPECT_EQ(applications, 0u);
  EXPECT_EQ(transpose_result.l2_norm(), 0.);
}

TEST(ContributorCore, SumReplaysOperandsForGraphAndValues)
{
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::SparseMatrix<double>;
  dealii::SparsityPattern diagonal(2, 2), off_diagonal(2, 2);
  diagonal.add(0, 0);
  diagonal.add(1, 1);
  off_diagonal.add(0, 1);
  off_diagonal.add(1, 0);
  diagonal.compress();
  off_diagonal.compress();
  Matrix a(diagonal), b(off_diagonal);
  a.set(0, 0, 2.);
  a.set(1, 1, 3.);
  b.set(0, 1, 5.);
  b.set(1, 0, -7.);
  auto         first       = ImmersX::matrix_operator<Vector>(a);
  auto         second      = ImmersX::matrix_operator<Vector>(b);
  unsigned int first_calls = 0, second_calls = 0;
  first.materialize = [factory = first.materialize, &first_calls]() {
    ++first_calls;
    return factory();
  };
  second.materialize = [factory = second.materialize, &second_calls]() {
    ++second_calls;
    return factory();
  };
  const auto sum = ImmersX::detail::sum_matrices(
    std::vector<ImmersX::MaterializedOperator<Vector, Matrix>>{first, second});
  EXPECT_EQ(first_calls, 2u);
  EXPECT_EQ(second_calls, 2u);
  a.set(0, 0, 99.);
  Vector source(2), result(2);
  source[0] = 11.;
  source[1] = 13.;
  sum->vmult(result, source);
  EXPECT_DOUBLE_EQ(result[0], 87.);
  EXPECT_DOUBLE_EQ(result[1], -38.);
}

TEST(ContributorCore, BOTH_DistributedMatrixCloneOwnsRectangularSnapshot)
{
  using Matrix     = ImmersX::ImmersXLA::MPI::SparseMatrix;
  using Vector     = ImmersX::ImmersXLA::MPI::Vector;
  const auto rank  = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const unsigned int n_rows = 2 * ranks;
  dealii::IndexSet   rows(n_rows), columns(n_rows + 1);
  rows.add_range(2 * rank, 2 * rank + 2);
  columns.add_range(2 * rank, 2 * rank + 2);
  if (rank + 1 == ranks)
    columns.add_index(n_rows);
  rows.compress();
  columns.compress();
  std::shared_ptr<Matrix> snapshot;
  {
    dealii::DynamicSparsityPattern sparsity(n_rows, n_rows + 1, rows);
    for (const auto i : rows)
      {
        sparsity.add(i, i);
        sparsity.add(i, i + 1);
      }
    Matrix matrix;
    matrix.reinit(rows, columns, sparsity, MPI_COMM_WORLD);
    for (const auto i : rows)
      {
        matrix.set(i, i, 2. + i);
        matrix.set(i, i + 1, -.5);
      }
    matrix.compress(dealii::VectorOperation::insert);
    snapshot = ImmersX::matrix_operator<Vector>(matrix).matrix();
    for (const auto i : rows)
      matrix.set(i, i, 99.);
    matrix.compress(dealii::VectorOperation::insert);
  }
  EXPECT_EQ(snapshot->m(), n_rows);
  EXPECT_EQ(snapshot->n(), n_rows + 1);
  EXPECT_EQ(snapshot->locally_owned_range_indices(), rows);
  EXPECT_EQ(snapshot->locally_owned_domain_indices(), columns);
  Vector source, result;
  source.reinit(columns, MPI_COMM_WORLD);
  result.reinit(rows, MPI_COMM_WORLD);
  for (const auto i : columns)
    source[i] = i + 1.;
  source.compress(dealii::VectorOperation::insert);
  snapshot->vmult(result, source);
  for (const auto i : rows)
    EXPECT_DOUBLE_EQ(result[i], (2. + i) * (i + 1.) - .5 * (i + 2.));
}

TEST(ContributorCore, BOTH_DistributedTransposeWithLongEmptyRowRanges)
{
  using Matrix     = ImmersX::ImmersXLA::MPI::SparseMatrix;
  using Vector     = ImmersX::ImmersXLA::MPI::Vector;
  const auto rank  = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  dealii::IndexSet rows(2048 * ranks), columns(3 * ranks);
  rows.add_range(2048 * rank, 2048 * (rank + 1));
  columns.add_range(3 * rank, 3 * (rank + 1));
  rows.compress();
  columns.compress();
  dealii::DynamicSparsityPattern sparsity(rows.size(), columns.size(), rows);
  if (rows.is_element(0))
    sparsity.add(0, columns.size() - 1);
  auto matrix = std::make_shared<Matrix>();
  matrix->reinit(rows, columns, sparsity, MPI_COMM_WORLD);
  if (rows.is_element(0))
    matrix->set(0, columns.size() - 1, 2.);
  matrix->compress(dealii::VectorOperation::insert);

  const auto transposed = ImmersX::detail::transpose_matrix(matrix);
  EXPECT_EQ(transposed->m(), columns.size());
  EXPECT_EQ(transposed->n(), rows.size());
  EXPECT_EQ(transposed->locally_owned_range_indices(), columns);
  EXPECT_EQ(transposed->locally_owned_domain_indices(), rows);
  Vector source, result;
  source.reinit(rows, MPI_COMM_WORLD);
  result.reinit(columns, MPI_COMM_WORLD);
  source = 1.;
  transposed->vmult(result, source);
  for (const auto column : columns)
    EXPECT_DOUBLE_EQ(result[column], column + 1 == columns.size() ? 2. : 0.);
}

TEST(ContributorCore, BOTH_ConstrainedMatrixWithLongEmptyRowRanges)
{
  using Matrix     = ImmersX::ImmersXLA::MPI::SparseMatrix;
  using Vector     = ImmersX::ImmersXLA::MPI::Vector;
  const auto rank  = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  dealii::IndexSet rows(2048 * ranks);
  rows.add_range(2048 * rank, 2048 * (rank + 1));
  rows.compress();
  dealii::DynamicSparsityPattern sparsity(rows.size(), rows.size(), rows);
  if (rows.is_element(0))
    sparsity.add(0, 0);
  if (rows.is_element(rows.size() - 1))
    sparsity.add(rows.size() - 1, rows.size() - 1);
  Matrix matrix;
  matrix.reinit(rows, rows, sparsity, MPI_COMM_WORLD);
  if (rows.is_element(0))
    matrix.set(0, 0, 2.);
  if (rows.is_element(rows.size() - 1))
    matrix.set(rows.size() - 1, rows.size() - 1, 3.);
  matrix.compress(dealii::VectorOperation::insert);
  dealii::AffineConstraints<double> constraints;
  constraints.reinit(rows, rows);
  for (const auto row : rows)
    if (row + 1 != rows.size())
      constraints.add_line(row);
  constraints.close();

  const auto projected =
    ImmersX::semidiscrete_detail::constrained_matrix_operator(
      ImmersX::matrix_operator<Vector>(matrix), constraints)
      .matrix();
  Vector source, result;
  source.reinit(rows, MPI_COMM_WORLD);
  result.reinit(rows, MPI_COMM_WORLD);
  source = 1.;
  projected->vmult(result, source);
  for (const auto row : rows)
    EXPECT_DOUBLE_EQ(result[row], row + 1 == rows.size() ? 3. : 0.);
}

TEST(ContributorCore, BOTH_DistributedSumReplaysOperandsForGraphAndValues)
{
  using Matrix     = ImmersX::ImmersXLA::MPI::SparseMatrix;
  using Vector     = ImmersX::ImmersXLA::MPI::Vector;
  const auto rank  = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const unsigned int size = 2 * ranks;
  dealii::IndexSet   rows(size);
  rows.add_range(2 * rank, 2 * rank + 2);
  rows.compress();
  dealii::DynamicSparsityPattern diagonal(size, size, rows);
  dealii::DynamicSparsityPattern shifted(size, size, rows);
  for (const auto i : rows)
    {
      diagonal.add(i, i);
      shifted.add(i, (i + 1) % size);
    }
  Matrix a, b;
  a.reinit(rows, rows, diagonal, MPI_COMM_WORLD);
  b.reinit(rows, rows, shifted, MPI_COMM_WORLD);
  for (const auto i : rows)
    {
      a.set(i, i, 2. + i);
      b.set(i, (i + 1) % size, -.5);
    }
  a.compress(dealii::VectorOperation::insert);
  b.compress(dealii::VectorOperation::insert);
  auto         first       = ImmersX::matrix_operator<Vector>(a);
  auto         second      = ImmersX::matrix_operator<Vector>(b);
  unsigned int first_calls = 0, second_calls = 0;
  first.materialize = [factory = first.materialize, &first_calls]() {
    ++first_calls;
    return factory();
  };
  second.materialize = [factory = second.materialize, &second_calls]() {
    ++second_calls;
    return factory();
  };
  const auto sum = ImmersX::detail::sum_matrices(
    std::vector<ImmersX::MaterializedOperator<Vector, Matrix>>{first, second});
  EXPECT_EQ(first_calls, 2u);
  EXPECT_EQ(second_calls, 2u);
  Vector source, result;
  source.reinit(rows, MPI_COMM_WORLD);
  result.reinit(rows, MPI_COMM_WORLD);
  for (const auto i : rows)
    source[i] = i + 1.;
  source.compress(dealii::VectorOperation::insert);
  sum->vmult(result, source);
  for (const auto i : rows)
    EXPECT_DOUBLE_EQ(result[i],
                     (2. + i) * (i + 1.) - .5 * ((i + 1) % size + 1.));
}

TEST(ContributorCore, BOTH_DistributedSumWithLongEmptyRowRanges)
{
  using Matrix     = ImmersX::ImmersXLA::MPI::SparseMatrix;
  using Vector     = ImmersX::ImmersXLA::MPI::Vector;
  const auto rank  = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const unsigned int size = 2048 * ranks;
  dealii::IndexSet   rows(size);
  rows.add_range(2048 * rank, 2048 * (rank + 1));
  rows.compress();
  dealii::DynamicSparsityPattern diagonal(size, size, rows),
    sparse(size, size, rows);
  for (const auto row : rows)
    diagonal.add(row, row);
  if (rows.is_element(size - 1))
    sparse.add(size - 1, 0);
  Matrix a, b;
  a.reinit(rows, rows, diagonal, MPI_COMM_WORLD);
  b.reinit(rows, rows, sparse, MPI_COMM_WORLD);
  for (const auto row : rows)
    a.set(row, row, 2.);
  if (rows.is_element(size - 1))
    b.set(size - 1, 0, -.5);
  a.compress(dealii::VectorOperation::insert);
  b.compress(dealii::VectorOperation::insert);
  const auto sum = ImmersX::detail::sum_matrices(
    std::vector<ImmersX::MaterializedOperator<Vector, Matrix>>{
      ImmersX::matrix_operator<Vector>(a),
      ImmersX::matrix_operator<Vector>(b)});
  Vector source, result;
  source.reinit(rows, MPI_COMM_WORLD);
  result.reinit(rows, MPI_COMM_WORLD);
  source = 1.;
  sum->vmult(result, source);
  for (const auto row : rows)
    EXPECT_DOUBLE_EQ(result[row], row == size - 1 ? 1.5 : 2.);
}

TEST(ContributorCore, ReplayingScaledSharedMatrixLeavesSourceUnchanged)
{
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::SparseMatrix<double>;
  dealii::SparsityPattern sparsity(2, 2);
  sparsity.add(0, 0);
  sparsity.add(1, 1);
  sparsity.compress();
  const auto matrix = std::make_shared<Matrix>(sparsity);
  matrix->set(0, 0, 2.);
  matrix->set(1, 1, 5.);
  auto source        = ImmersX::matrix_operator<Vector>(*matrix);
  source.materialize = [matrix] { return matrix; };
  const auto scaled  = 3. * source;
  for (unsigned int build = 0; build < 2; ++build)
    {
      const auto sum = ImmersX::detail::sum_matrices(
        std::vector<ImmersX::MaterializedOperator<Vector, Matrix>>{scaled,
                                                                   scaled});
      EXPECT_DOUBLE_EQ(sum->el(0, 0), 12.);
      EXPECT_DOUBLE_EQ(sum->el(1, 1), 30.);
      EXPECT_DOUBLE_EQ(matrix->el(0, 0), 2.);
      EXPECT_DOUBLE_EQ(matrix->el(1, 1), 5.);
    }
}

TEST(ContributorCore, PayloadFreeCopiesShareCallbackState)
{
  using Vector = dealii::Vector<double>;
  struct Action
  {
    explicit Action(std::shared_ptr<unsigned int> copies)
      : copies(std::move(copies))
    {}
    Action(const Action &other)
      : copies(other.copies)
    {
      ++*copies;
    }
    void
    operator()(Vector &dst, const Vector &src) const
    {
      dst = src;
      dst *= 2.;
    }
    std::shared_ptr<unsigned int> copies;
  };
  const auto copies    = std::make_shared<unsigned int>(0);
  const auto make_view = [&]() {
    dealii::LinearOperator<Vector> original;
    original.vmult                = Action(copies);
    original.Tvmult               = original.vmult;
    original.reinit_range_vector  = [](Vector &v, const bool) { v.reinit(2); };
    original.reinit_domain_vector = original.reinit_range_vector;
    return ImmersX::payload_free(original);
  };
  const auto view        = make_view();
  const auto before      = *copies;
  const auto copy        = view;
  const auto composition = copy * view;
  EXPECT_EQ(*copies, before);
  Vector src(2), dst(2);
  src = 3.;
  composition.vmult(dst, src);
  EXPECT_DOUBLE_EQ(dst[0], 12.);
  EXPECT_DOUBLE_EQ(dst[1], 12.);
  copy.Tvmult(dst, src);
  EXPECT_DOUBLE_EQ(dst[0], 6.);
}
