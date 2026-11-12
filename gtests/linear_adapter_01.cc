#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>

#include <array>
#include <tuple>

namespace ImmersX
{
  struct FakeProblem
  {
    ImmersXLA::MPI::SparseMatrix matrix;
    ImmersXLA::MPI::Vector       rhs;
    dealii::IndexSet             owned;
  };

  struct FakeFields
  {
    ImmersX::FieldId solution;
  };

  struct TriangularProblem
  {
    ImmersXLA::MPI::SparseMatrix diagonal_u;
    ImmersXLA::MPI::SparseMatrix diagonal_v;
    ImmersXLA::MPI::SparseMatrix lower;
    dealii::IndexSet             owned;
  };

  struct TriangularFields
  {
    ImmersX::FieldId u;
    ImmersX::FieldId v;
  };

  struct SchurProblem
  {
    dealii::IndexSet             primal_owned;
    dealii::IndexSet             auxiliary_owned;
    dealii::IndexSet             multiplier_owned;
    ImmersXLA::MPI::SparseMatrix primal_diagonal;
    ImmersXLA::MPI::SparseMatrix auxiliary_diagonal;
    ImmersXLA::MPI::SparseMatrix multiplier_diagonal;
  };

  struct SchurFields
  {
    ImmersX::FieldId primal;
    ImmersX::FieldId auxiliary;
    ImmersX::FieldId multiplier;
  };
} // namespace ImmersX

namespace ImmersX
{
  template <typename Builder>
  FakeFields
  contribute(Builder &builder, const FakeProblem &problem)
  {
    const auto solution = builder.algebraic_field("solution", problem.owned);
    using VectorType    = ImmersXLA::MPI::Vector;
    const auto matrix   = builder.matrix_operator(problem.matrix);
    builder.preconditioner(
      solution, [](const auto &linearized_matrix, const auto &reinit_vector) {
        return make_amg_preconditioner(linearized_matrix, reinit_vector);
      });
    builder.saddle_point(solution, {solution});
    builder.term(solution, "fake")
      .residual([solution, &problem](const auto &context) {
        const auto &state = context.state(solution);
        dealii::PackagedOperation<ImmersXLA::MPI::Vector> operation;
        operation.reinit_vector = [state](auto &vector, const bool omit) {
          vector.reinit(state, omit);
        };
        operation.apply = [&problem, &state](auto &vector) {
          problem.matrix.vmult(vector, state);
          vector -= problem.rhs;
        };
        operation.apply_add = [&problem, &state](auto &vector) {
          ImmersXLA::MPI::Vector contribution;
          contribution.reinit(state);
          problem.matrix.vmult(contribution, state);
          contribution -= problem.rhs;
          vector += contribution;
        };
        return operation;
      })
      .state(solution, matrix);
    return {solution};
  }
} // namespace ImmersX

namespace ImmersX
{
  template <typename Builder>
  TriangularFields
  contribute(Builder &builder, const TriangularProblem &problem)
  {
    using VectorType      = ImmersXLA::MPI::Vector;
    const auto u          = builder.algebraic_field("u", problem.owned);
    const auto v          = builder.algebraic_field("v", problem.owned);
    const auto diagonal_u = builder.matrix_operator(problem.diagonal_u);
    const auto diagonal_v = builder.matrix_operator(problem.diagonal_v);
    const auto lower      = builder.matrix_operator(problem.lower);

    const auto inverse = [](const double factor) {
      return [factor](const auto &, const auto &reinit_vector) {
        dealii::LinearOperator<VectorType> result;
        result.reinit_range_vector  = reinit_vector;
        result.reinit_domain_vector = reinit_vector;
        result.vmult                = [factor](VectorType       &destination,
                                const VectorType &source) {
          destination = source;
          destination *= factor;
        };
        result.vmult_add = [factor](VectorType       &destination,
                                    const VectorType &source) {
          VectorType contribution;
          contribution.reinit(destination);
          contribution = source;
          contribution *= factor;
          destination += contribution;
        };
        result.Tvmult     = result.vmult;
        result.Tvmult_add = result.vmult_add;
        return result;
      };
    };
    builder.preconditioner(u, inverse(0.5));
    builder.preconditioner(v, inverse(1. / 3.));

    builder.term(u, "diagonal").state(u, diagonal_u);
    builder.term(v, "diagonal").state(v, diagonal_v);
    builder.term(v, "lower").state(u, lower);
    return {u, v};
  }
} // namespace ImmersX

namespace
{
  using VectorType = ImmersX::ImmersXLA::MPI::Vector;
  using Operator   = dealii::LinearOperator<VectorType>;

  Operator
  make_rectangular_operator(const dealii::IndexSet    &range,
                            const dealii::IndexSet    &domain,
                            const std::vector<double> &values)
  {
    const auto range_size  = range.size();
    const auto domain_size = domain.size();
    AssertThrow(values.size() == range_size * domain_size,
                dealii::ExcMessage("Rectangular operator has wrong size."));

    Operator result;
    result.reinit_range_vector = [range](VectorType &vector, const bool) {
      vector.reinit(range, MPI_COMM_WORLD);
    };
    result.reinit_domain_vector = [domain](VectorType &vector, const bool) {
      vector.reinit(domain, MPI_COMM_WORLD);
    };
    result.vmult = [range_size, domain_size, values](VectorType       &dst,
                                                     const VectorType &src) {
      dst = 0.;
      for (std::size_t row = 0; row < range_size; ++row)
        for (std::size_t column = 0; column < domain_size; ++column)
          dst[row] += values[row * domain_size + column] * src[column];
    };
    result.vmult_add = [range_size,
                        domain_size,
                        values](VectorType &dst, const VectorType &src) {
      for (std::size_t row = 0; row < range_size; ++row)
        for (std::size_t column = 0; column < domain_size; ++column)
          dst[row] += values[row * domain_size + column] * src[column];
    };
    result.Tvmult = [range_size, domain_size, values](VectorType       &dst,
                                                      const VectorType &src) {
      dst = 0.;
      for (std::size_t row = 0; row < range_size; ++row)
        for (std::size_t column = 0; column < domain_size; ++column)
          dst[column] += values[row * domain_size + column] * src[row];
    };
    result.Tvmult_add = [range_size,
                         domain_size,
                         values](VectorType &dst, const VectorType &src) {
      for (std::size_t row = 0; row < range_size; ++row)
        for (std::size_t column = 0; column < domain_size; ++column)
          dst[column] += values[row * domain_size + column] * src[row];
    };
    return result;
  }
} // namespace

namespace ImmersX
{
  template <typename Builder>
  SchurFields
  contribute(Builder &builder, const SchurProblem &problem)
  {
    const auto primal = builder.algebraic_field("primal", problem.primal_owned);
    const auto auxiliary =
      builder.algebraic_field("auxiliary", problem.auxiliary_owned);
    const auto multiplier =
      builder.algebraic_field("multiplier", problem.multiplier_owned);

    using LocalOperator = dealii::LinearOperator<ImmersXLA::MPI::Vector>;
    const auto inverse  = [](const double vmult_factor,
                            const double Tvmult_factor) {
      return [vmult_factor, Tvmult_factor](const auto &, const auto &reinit) {
        LocalOperator result;
        result.reinit_range_vector  = reinit;
        result.reinit_domain_vector = reinit;
        result.vmult = [vmult_factor](ImmersXLA::MPI::Vector       &dst,
                                      const ImmersXLA::MPI::Vector &src) {
          dst = src;
          dst *= vmult_factor;
        };
        result.vmult_add = [vmult_factor](ImmersXLA::MPI::Vector       &dst,
                                          const ImmersXLA::MPI::Vector &src) {
          ImmersXLA::MPI::Vector contribution;
          contribution.reinit(dst);
          contribution = src;
          contribution *= vmult_factor;
          dst += contribution;
        };
        result.Tvmult = [Tvmult_factor](ImmersXLA::MPI::Vector       &dst,
                                        const ImmersXLA::MPI::Vector &src) {
          dst = src;
          dst *= Tvmult_factor;
        };
        result.Tvmult_add = [Tvmult_factor](ImmersXLA::MPI::Vector       &dst,
                                            const ImmersXLA::MPI::Vector &src) {
          ImmersXLA::MPI::Vector contribution;
          contribution.reinit(dst);
          contribution = src;
          contribution *= Tvmult_factor;
          dst += contribution;
        };
        return result;
      };
    };
    builder.preconditioner(primal, inverse(2., 3.));
    builder.preconditioner(auxiliary, inverse(2., 3.));
    builder.preconditioner(multiplier, inverse(5., 7.));

    builder.saddle_point(multiplier, {primal, auxiliary});
    builder.term(primal, "diagonal")
      .state(primal, builder.matrix_operator(problem.primal_diagonal));
    builder.term(auxiliary, "diagonal")
      .state(auxiliary, builder.matrix_operator(problem.auxiliary_diagonal));
    builder.term(multiplier, "diagonal")
      .state(multiplier, builder.matrix_operator(problem.multiplier_diagonal));
    builder.term(multiplier, "primal")
      .state(primal,
             make_rectangular_operator(problem.multiplier_owned,
                                       problem.primal_owned,
                                       {1., 2.}));
    builder.term(primal, "multiplier")
      .state(multiplier,
             make_rectangular_operator(problem.primal_owned,
                                       problem.multiplier_owned,
                                       {7., 11.}));
    builder.term(multiplier, "auxiliary")
      .state(auxiliary,
             make_rectangular_operator(problem.multiplier_owned,
                                       problem.auxiliary_owned,
                                       {3., 4., 5.}));
    builder.term(auxiliary, "multiplier")
      .state(multiplier,
             make_rectangular_operator(problem.auxiliary_owned,
                                       problem.multiplier_owned,
                                       {13., 17., 19.}));
    return {primal, auxiliary, multiplier};
  }
} // namespace ImmersX

namespace
{
  void
  initialize_scalar_matrix(ImmersX::ImmersXLA::MPI::SparseMatrix &matrix,
                           const dealii::IndexSet                &owned,
                           const double                           value)
  {
    dealii::DynamicSparsityPattern sparsity(owned.size(), owned.size());
    for (const auto index : owned)
      sparsity.add(index, index);
    matrix.reinit(owned, owned, sparsity, MPI_COMM_WORLD);
    for (const auto index : owned)
      matrix.set(index, index, value);
    matrix.compress(dealii::VectorOperation::insert);
  }
} // namespace

TEST(LinearAdapter, DirectContributorAndSemanticFieldAccess)
{
  ImmersX::FakeProblem problem;
  problem.owned = dealii::IndexSet(2);
  problem.owned.add_range(0, 2);
  problem.owned.compress();
  dealii::DynamicSparsityPattern sparsity(2, 2);
  sparsity.add(0, 0);
  sparsity.add(1, 1);
  problem.matrix.reinit(problem.owned, problem.owned, sparsity, MPI_COMM_WORLD);
  problem.matrix.set(0, 0, 2.);
  problem.matrix.set(1, 1, 4.);
  problem.matrix.compress(dealii::VectorOperation::insert);
  problem.rhs.reinit(problem.owned, MPI_COMM_WORLD);
  problem.rhs[0] = 2.;
  problem.rhs[1] = 8.;
  using Adapter  = ImmersX::LinearAdapter<ImmersX::ImmersXLA::MPI::Vector,
                                         ImmersX::ImmersXLA::MPI::BlockVector>;
  ImmersX::LinearSolverParameters adapter_parameters;
  Adapter                         adapter(adapter_parameters,
                  MPI_COMM_WORLD,
                  [](const auto &operator_view,
                     const auto &rhs,
                     auto       &solution) {
                    (void)operator_view;
                    solution = rhs;
                    solution.block(0)[0] /= 2.;
                    solution.block(0)[1] /= 4.;
                  });
  const auto                      fields = adapter.add(problem, "fake");
  auto                            state  = adapter.make_state();
  EXPECT_EQ(state.n_blocks(), 1u);
  adapter.solve(state);

  EXPECT_NEAR(adapter.field(state, fields.fields().solution)[0], 1., 1.e-12);
  EXPECT_NEAR(adapter.field(state, fields.fields().solution)[1], 2., 1.e-12);

  EXPECT_TRUE(adapter.has_local_preconditioner(fields.fields().solution));
  const auto local =
    adapter.local_preconditioner(fields.fields().solution, state);
  ASSERT_TRUE(local.has_value());
  const auto input          = adapter.field(state, fields.fields().solution);
  auto       preconditioned = input;
  (*local).vmult(preconditioned, input);
  EXPECT_GT(preconditioned.l2_norm(), 0.);
  const auto block_diagonal = adapter.block_diagonal_preconditioner(state);
  auto       block_diagonal_action = adapter.make_state();
  block_diagonal.vmult(block_diagonal_action, state);
  EXPECT_GT(block_diagonal_action.l2_norm(), 0.);
  const auto block_lower = adapter.block_triangular_preconditioner(state);
  auto       block_lower_action = adapter.make_state();
  block_lower.vmult(block_lower_action, state);
  EXPECT_NEAR(adapter.field(block_lower_action, fields.fields().solution)[0],
              preconditioned[0],
              1.e-12);
  EXPECT_NEAR(adapter.field(block_lower_action, fields.fields().solution)[1],
              preconditioned[1],
              1.e-12);
  ASSERT_EQ(adapter.saddle_points().size(), 1u);
  const auto schur = adapter.schur_operator(fields.fields().solution, state);
  auto       schur_action = adapter.field(state, fields.fields().solution);
  schur.vmult(schur_action, input);
  EXPECT_GT(schur_action.l2_norm(), 0.);
  const auto augmented = adapter.augmented_lagrangian_operator(state, 2.);
  auto       augmented_action = adapter.make_state();
  augmented.vmult(augmented_action, state);
  EXPECT_GT(augmented_action.l2_norm(), 0.);

  EXPECT_TRUE(adapter.can_materialize_matrix(state));
  const auto block_matrix = adapter.block_matrix(state);
  EXPECT_EQ(block_matrix.n_block_rows(), 1u);
  const auto monolithic = adapter.monolithic_matrix(state);
  EXPECT_EQ(monolithic.m(), 2u);
  auto block_action = adapter.make_state();
  block_matrix.vmult(block_action, state);
  auto operator_action = adapter.make_state();
  adapter.jacobian(state).vmult(operator_action, state);
  EXPECT_NEAR(adapter.field(block_action, fields.fields().solution)[0],
              adapter.field(operator_action, fields.fields().solution)[0],
              1.e-12);
  EXPECT_NEAR(adapter.field(block_action, fields.fields().solution)[1],
              adapter.field(operator_action, fields.fields().solution)[1],
              1.e-12);
  const auto                      flat = adapter.pack(state);
  ImmersX::ImmersXLA::MPI::Vector monolithic_action;
  monolithic_action.reinit(flat);
  monolithic.vmult(monolithic_action, flat);
  EXPECT_NEAR(monolithic_action[0],
              adapter.field(operator_action, fields.fields().solution)[0],
              1.e-12);
  EXPECT_NEAR(monolithic_action[1],
              adapter.field(operator_action, fields.fields().solution)[1],
              1.e-12);
  auto unpacked = adapter.make_state();
  adapter.unpack(flat, unpacked);
  EXPECT_NEAR(adapter.field(unpacked, fields.fields().solution)[0], 1., 1.e-12);
  EXPECT_NEAR(adapter.field(unpacked, fields.fields().solution)[1], 2., 1.e-12);

  adapter.solve_direct(state);
  EXPECT_NEAR(adapter.field(state, fields.fields().solution)[0], 1., 1.e-12);
  EXPECT_NEAR(adapter.field(state, fields.fields().solution)[1], 2., 1.e-12);

  ImmersX::LinearSolverParameters default_options;
  default_options.preconditioner =
    ImmersX::LinearPreconditioner::block_diagonal;
  Adapter    default_adapter(default_options, MPI_COMM_WORLD);
  const auto default_fields = default_adapter.add(problem, "default");
  auto       default_state  = default_adapter.make_state();
  default_adapter.solve(default_state);
  EXPECT_NEAR(default_adapter.field(default_state,
                                    default_fields.fields().solution)[0],
              1.,
              1.e-10);
  EXPECT_NEAR(default_adapter.field(default_state,
                                    default_fields.fields().solution)[1],
              2.,
              1.e-10);

  ImmersX::LinearSolverParameters direct_options;
  direct_options.solver = ImmersX::LinearSolver::direct;
  Adapter    direct_adapter(direct_options, MPI_COMM_WORLD);
  const auto direct_fields = direct_adapter.add(problem, "direct");
  auto       direct_state  = direct_adapter.make_state();
  direct_adapter.solve(direct_state);
  EXPECT_NEAR(direct_adapter.field(direct_state,
                                   direct_fields.fields().solution)[0],
              1.,
              1.e-12);
  EXPECT_NEAR(direct_adapter.field(direct_state,
                                   direct_fields.fields().solution)[1],
              2.,
              1.e-12);

  direct_adapter.setup_direct(direct_state);
  problem.matrix.set(0, 0, 4.);
  problem.matrix.compress(dealii::VectorOperation::insert);
  direct_adapter.solve_with_current_direct(direct_state);
  EXPECT_NEAR(direct_adapter.field(direct_state,
                                   direct_fields.fields().solution)[0],
              1.,
              1.e-12);
  direct_adapter.solve(direct_state);
  EXPECT_NEAR(direct_adapter.field(direct_state,
                                   direct_fields.fields().solution)[0],
              .5,
              1.e-12);
  EXPECT_NEAR(direct_adapter.field(direct_state,
                                   direct_fields.fields().solution)[1],
              2.,
              1.e-12);

  problem.matrix.set(0, 0, 2.);
  problem.matrix.compress(dealii::VectorOperation::insert);

  ImmersX::ImmersXLA::MPI::BlockVector residual;
  adapter.evaluate_residual(state, residual);
  EXPECT_NEAR(residual.l2_norm(), 0., 1.e-12);

  // Changing the assembled matrix must invalidate the cached factorization.
  problem.matrix.set(0, 0, 1.);
  problem.matrix.set(1, 1, 2.);
  problem.matrix.compress(dealii::VectorOperation::insert);
  problem.rhs[0]            = 1.;
  problem.rhs[1]            = 6.;
  auto changed_direct_state = direct_adapter.make_state();
  direct_adapter.solve(changed_direct_state);
  EXPECT_NEAR(direct_adapter.field(changed_direct_state,
                                   direct_fields.fields().solution)[0],
              1.,
              1.e-12);
  EXPECT_NEAR(direct_adapter.field(changed_direct_state,
                                   direct_fields.fields().solution)[1],
              3.,
              1.e-12);
}

TEST(LinearAdapter, BOTH_SparseAugmentedMatrixPreservesRemoteCoupling)
{
  using Matrix = ImmersX::ImmersXLA::MPI::SparseMatrix;
  using Vector = ImmersX::ImmersXLA::MPI::Vector;
  using Adapter =
    ImmersX::LinearAdapter<Vector, ImmersX::ImmersXLA::MPI::BlockVector>;
  const auto rank  = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  dealii::IndexSet primal(2048 * ranks), multiplier(2 * ranks);
  primal.add_range(2048 * rank, 2048 * (rank + 1));
  multiplier.add_range(2 * rank, 2 * (rank + 1));
  primal.compress();
  multiplier.compress();
  dealii::DynamicSparsityPattern a_graph(primal.size(), primal.size(), primal);
  dealii::DynamicSparsityPattern b_graph(multiplier.size(),
                                         primal.size(),
                                         multiplier);
  dealii::DynamicSparsityPattern metric_graph(multiplier.size(),
                                              multiplier.size(),
                                              multiplier);
  for (const auto row : primal)
    a_graph.add(row, row);
  for (const auto row : multiplier)
    metric_graph.add(row, row);
  if (multiplier.is_element(multiplier.size() - 1))
    b_graph.add(multiplier.size() - 1, 0);
  Matrix a, b, metric;
  a.reinit(primal, primal, a_graph, MPI_COMM_WORLD);
  b.reinit(multiplier, primal, b_graph, MPI_COMM_WORLD);
  metric.reinit(multiplier, multiplier, metric_graph, MPI_COMM_WORLD);
  for (const auto row : primal)
    a.set(row, row, 2.);
  for (const auto row : multiplier)
    metric.set(row, row, 2.);
  if (multiplier.is_element(multiplier.size() - 1))
    b.set(multiplier.size() - 1, 0, 3.);
  a.compress(dealii::VectorOperation::insert);
  b.compress(dealii::VectorOperation::insert);
  metric.compress(dealii::VectorOperation::insert);

  ImmersX::LinearAdapterParameters parameters;
  Adapter                          adapter(parameters, MPI_COMM_WORLD);
  const auto                       fields =
    adapter
      .add(
        [&](auto &builder) {
          const auto u      = builder.algebraic_field("primal", primal);
          const auto lambda = builder.algebraic_field("multiplier", multiplier);
          const auto coupling = builder.matrix_operator(b);
          builder.term(u, "diagonal").state(u, builder.matrix_operator(a));
          builder.term(lambda, "coupling").state(u, coupling);
          builder.term(u, "reaction")
            .state(lambda, ImmersX::transpose_operator(coupling));
          builder.saddle_point(lambda, {u}, builder.matrix_operator(metric));
          return std::make_pair(u, lambda);
        },
        "sparse")
      .fields();
  auto state                          = adapter.make_state();
  adapter.field(state, fields.first)  = 1.;
  adapter.field(state, fields.second) = 0.;
  const auto augmented = adapter.augmented_lagrangian_matrix(state, 4.);
  auto       action    = adapter.make_state();
  augmented.vmult(action, state);
  for (const auto row : primal)
    EXPECT_DOUBLE_EQ(adapter.field(action, fields.first)[row],
                     row == 0 ? 20. : 2.);
  for (const auto row : multiplier)
    EXPECT_DOUBLE_EQ(adapter.field(action, fields.second)[row],
                     row + 1 == multiplier.size() ? 3. : 0.);
}

TEST(LinearAdapter, ParameterAcceptorControlsSolverOptions)
{
  dealii::ParameterAcceptor::clear();

  using Adapter = ImmersX::LinearAdapter<ImmersX::ImmersXLA::MPI::Vector,
                                         ImmersX::ImmersXLA::MPI::BlockVector>;
  ImmersX::LinearSolverParameters adapter_parameters(
    "Linear adapter parameters");
  Adapter adapter(adapter_parameters, MPI_COMM_WORLD, Adapter::SolveFunction{});

  ImmersX::initialize_parameters_from_string(R"(
    subsection Linear adapter parameters
      set Solver = direct
      set Preconditioner = augmented_lagrangian
      set Maximum iterations = 37
      set Tolerance = 1.e-9
      set Augmented Lagrangian parameter = 4.5
    end
  )");

  const auto &options = adapter.solver_options();
  EXPECT_EQ(options.solver, ImmersX::LinearSolver::direct);
  EXPECT_EQ(options.preconditioner,
            ImmersX::LinearPreconditioner::augmented_lagrangian);
  EXPECT_EQ(options.maximum_iterations, 37u);
  EXPECT_DOUBLE_EQ(options.tolerance, 1.e-9);
  EXPECT_DOUBLE_EQ(options.augmented_lagrangian_parameter, 4.5);
}

TEST(LinearAdapter, TwoFieldTriangularTransposeIsDistinct)
{
  ImmersX::TriangularProblem problem;
  problem.owned = dealii::IndexSet(1);
  problem.owned.add_index(0);
  problem.owned.compress();
  initialize_scalar_matrix(problem.diagonal_u, problem.owned, 2.);
  initialize_scalar_matrix(problem.diagonal_v, problem.owned, 3.);
  initialize_scalar_matrix(problem.lower, problem.owned, 4.);

  using Adapter = ImmersX::LinearAdapter<ImmersX::ImmersXLA::MPI::Vector,
                                         ImmersX::ImmersXLA::MPI::BlockVector>;
  ImmersX::LinearSolverParameters adapter_parameters;
  Adapter                         adapter(adapter_parameters,
                  MPI_COMM_WORLD,
                  [](const auto &, const auto &, auto &) {});
  const auto                      fields = adapter.add(problem, "triangular");
  auto                            state  = adapter.make_state();
  adapter.field(state, fields.fields().v)[0] = 9.;
  adapter.field(state, fields.fields().u)[0] = 6.;

  const auto preconditioner =
    adapter.block_triangular_preconditioner(state, true);
  auto forward   = adapter.make_state();
  auto transpose = adapter.make_state();
  preconditioner.vmult(forward, state);
  preconditioner.Tvmult(transpose, state);

  EXPECT_NEAR(adapter.field(forward, fields.fields().u)[0], 3., 1.e-12);
  EXPECT_NEAR(adapter.field(forward, fields.fields().v)[0], -1., 1.e-12);
  EXPECT_NEAR(adapter.field(transpose, fields.fields().u)[0], -3., 1.e-12);
  EXPECT_NEAR(adapter.field(transpose, fields.fields().v)[0], 3., 1.e-12);
  EXPECT_NE(adapter.field(forward, fields.fields().u)[0],
            adapter.field(transpose, fields.fields().u)[0]);
}

TEST(LinearAdapter, BOTH_BlockPreconditionersUseMultiplierMetric)
{
  using Vector = ImmersX::ImmersXLA::MPI::Vector;
  using Adapter =
    ImmersX::LinearAdapter<Vector, ImmersX::ImmersXLA::MPI::BlockVector>;

  const auto primal_owned =
    dealii::Utilities::MPI::create_evenly_distributed_partitioning(
      MPI_COMM_WORLD, 2);
  const auto auxiliary_owned  = primal_owned;
  const auto multiplier_owned = primal_owned;

  ImmersX::ImmersXLA::MPI::SparseMatrix primal_matrix;
  ImmersX::ImmersXLA::MPI::SparseMatrix auxiliary_matrix;
  ImmersX::ImmersXLA::MPI::SparseMatrix metric_matrix;
  initialize_scalar_matrix(primal_matrix, primal_owned, 1.);
  initialize_scalar_matrix(auxiliary_matrix, auxiliary_owned, 1.);
  initialize_scalar_matrix(metric_matrix, multiplier_owned, 2.);

  ImmersX::LinearSolverParameters parameters;
  Adapter                         adapter(parameters,
                  MPI_COMM_WORLD,
                  [](const auto &, const auto &, auto &) {});
  const auto                      fields = adapter.add(
    [&](auto &builder) {
      const auto primal = builder.algebraic_field("primal", primal_owned);
      const auto auxiliary =
        builder.algebraic_field("auxiliary", auxiliary_owned);
      const auto multiplier =
        builder.algebraic_field("multiplier", multiplier_owned);
      builder.preconditioner(primal, [](const auto &, const auto &reinit) {
        return dealii::identity_operator<Vector>(reinit);
      });
      builder.preconditioner(auxiliary, [](const auto &, const auto &reinit) {
        return dealii::identity_operator<Vector>(reinit);
      });
      const auto primal_op = builder.matrix_operator(primal_matrix);
      const auto auxiliary_op = builder.matrix_operator(auxiliary_matrix);
      const auto metric_op = builder.matrix_operator(metric_matrix);
      builder.term(primal, "diagonal").state(primal, primal_op);
      builder.term(auxiliary, "diagonal").state(auxiliary, auxiliary_op);
      builder.term(multiplier, "metric").state(multiplier, metric_op);
      builder.saddle_point(multiplier, {primal, auxiliary}, metric_op);
      return std::make_tuple(primal, auxiliary, multiplier);
    },
    "metric-preconditioner");
  auto state                                         = adapter.make_state();
  adapter.field(state, std::get<0>(fields.fields())) = 1.;
  adapter.field(state, std::get<1>(fields.fields())) = 1.;
  adapter.field(state, std::get<2>(fields.fields())) = 4.;

  const auto diagonal        = adapter.block_diagonal_preconditioner(state);
  auto       diagonal_action = adapter.make_state();
  diagonal.vmult(diagonal_action, state);
  const auto &diagonal_multiplier =
    adapter.field(diagonal_action, std::get<2>(fields.fields()));
  for (const auto index : multiplier_owned)
    EXPECT_DOUBLE_EQ(diagonal_multiplier[index], 2.);

  const auto triangular        = adapter.block_triangular_preconditioner(state);
  auto       triangular_action = adapter.make_state();
  triangular.vmult(triangular_action, state);
  const auto &triangular_multiplier =
    adapter.field(triangular_action, std::get<2>(fields.fields()));
  for (const auto index : multiplier_owned)
    EXPECT_DOUBLE_EQ(triangular_multiplier[index], 2.);
}

TEST(LinearAdapter, SchurUsesDistinctParticipantSpacesAndTranspose)
{
  ImmersX::SchurProblem problem;
  problem.primal_owned = dealii::IndexSet(2);
  problem.primal_owned.add_range(0, 2);
  problem.primal_owned.compress();
  problem.auxiliary_owned = dealii::IndexSet(3);
  problem.auxiliary_owned.add_range(0, 3);
  problem.auxiliary_owned.compress();
  problem.multiplier_owned = dealii::IndexSet(1);
  problem.multiplier_owned.add_index(0);
  problem.multiplier_owned.compress();
  initialize_scalar_matrix(problem.primal_diagonal, problem.primal_owned, 1.);
  initialize_scalar_matrix(problem.auxiliary_diagonal,
                           problem.auxiliary_owned,
                           1.);
  initialize_scalar_matrix(problem.multiplier_diagonal,
                           problem.multiplier_owned,
                           1.);

  using Adapter = ImmersX::LinearAdapter<ImmersX::ImmersXLA::MPI::Vector,
                                         ImmersX::ImmersXLA::MPI::BlockVector>;
  ImmersX::LinearSolverParameters adapter_parameters;
  Adapter                         adapter(adapter_parameters,
                  MPI_COMM_WORLD,
                  [](const auto &, const auto &, auto &) {});
  const auto                      fields       = adapter.add(problem, "schur");
  auto                            state        = adapter.make_state();
  adapter.field(state, fields.fields().primal) = 1.;
  adapter.field(state, fields.fields().auxiliary)  = 1.;
  adapter.field(state, fields.fields().multiplier) = 0.;
  const auto schur = adapter.schur_operator(fields.fields().multiplier, state);

  ImmersX::ImmersXLA::MPI::Vector input;
  input.reinit(problem.multiplier_owned, MPI_COMM_WORLD);
  input       = 1.;
  auto normal = input;
  schur.vmult(normal, input);
  auto transpose = input;
  schur.Tvmult(transpose, input);

  EXPECT_NEAR(normal[0], 461., 1.e-12);
  EXPECT_NEAR(transpose[0], 692., 1.e-12);

  const auto augmented = adapter.augmented_lagrangian_operator(state, 2.);
  auto       augmented_normal    = adapter.make_state();
  auto       augmented_transpose = adapter.make_state();
  augmented.vmult(augmented_normal, state);
  augmented.Tvmult(augmented_transpose, state);

  EXPECT_NEAR(adapter.field(augmented_normal, fields.fields().primal)[0],
              1051.,
              1.e-12);
  EXPECT_NEAR(adapter.field(augmented_normal, fields.fields().primal)[1],
              1651.,
              1.e-12);
  EXPECT_NEAR(adapter.field(augmented_transpose, fields.fields().primal)[0],
              939.,
              1.e-12);
  EXPECT_NEAR(adapter.field(augmented_transpose, fields.fields().primal)[1],
              1877.,
              1.e-12);

  const auto schur_prec =
    adapter.schur_preconditioner(fields.fields().multiplier, state);
  auto preconditioned            = adapter.make_state();
  auto transposed_preconditioned = adapter.make_state();
  schur_prec.vmult(preconditioned, state);
  schur_prec.Tvmult(transposed_preconditioned, state);

  const double normal_multiplier = 30. / 461.;
  EXPECT_NEAR(adapter.field(preconditioned, fields.fields().multiplier)[0],
              normal_multiplier,
              1.e-10);
  EXPECT_NEAR(adapter.field(preconditioned, fields.fields().primal)[0],
              2. - 14. * normal_multiplier,
              1.e-10);
  EXPECT_NEAR(adapter.field(preconditioned, fields.fields().primal)[1],
              2. - 22. * normal_multiplier,
              1.e-10);

  const double transpose_multiplier = 201. / 692.;
  EXPECT_NEAR(adapter.field(transposed_preconditioned,
                            fields.fields().multiplier)[0],
              transpose_multiplier,
              1.e-10);
  EXPECT_NEAR(adapter.field(transposed_preconditioned,
                            fields.fields().primal)[0],
              3. * (1. - transpose_multiplier),
              1.e-10);
  EXPECT_NEAR(adapter.field(transposed_preconditioned,
                            fields.fields().primal)[1],
              3. * (1. - 2. * transpose_multiplier),
              1.e-10);

  const auto copied_preconditioner = [&]() {
    const auto temporary =
      adapter.schur_preconditioner(fields.fields().multiplier, state);
    auto copy = temporary;
    return copy;
  }();
  auto accumulated = state;
  copied_preconditioner.vmult_add(accumulated, state);
  accumulated -= state;
  accumulated -= preconditioned;
  EXPECT_LT(accumulated.l2_norm(), 1.e-10);
  accumulated = state;
  copied_preconditioner.Tvmult_add(accumulated, state);
  accumulated -= state;
  accumulated -= transposed_preconditioned;
  EXPECT_LT(accumulated.l2_norm(), 1.e-10);
}


namespace ImmersX
{
  struct RetainedSolveProblem
  {
    std::array<dealii::IndexSet, 2>             owned;
    std::array<ImmersXLA::MPI::SparseMatrix, 4> matrices;
    std::array<ImmersXLA::MPI::Vector, 2>       rhs;
    mutable std::array<unsigned int, 4>         recipe_actions{};
    mutable std::array<unsigned int, 4>         materializations{};
  };

  struct RetainedSolveFields
  {
    FieldId primal;
    FieldId multiplier;
  };

  template <typename Builder>
  RetainedSolveFields
  contribute(Builder &builder, const RetainedSolveProblem &problem)
  {
    using VectorType    = ImmersXLA::MPI::Vector;
    using MatrixType    = ImmersXLA::MPI::SparseMatrix;
    using MatrixFactory = typename Builder::Model::MatrixOperatorFactory;
    const std::array<FieldId, 2> fields = {
      {builder.algebraic_field("primal", problem.owned[0]),
       builder.algebraic_field("multiplier", problem.owned[1])}};
    builder.saddle_point(fields[1], {fields[0]});
    builder.preconditioner(fields[0], [](const auto &, const auto &reinit) {
      return dealii::identity_operator<VectorType>(reinit);
    });
    for (unsigned int row = 0; row < 2; ++row)
      {
        builder.term(fields[row], "physical residual")
          .residual([fields, row, &problem](const auto &context) {
            const auto *primal     = &context.state(fields[0]);
            const auto *multiplier = &context.state(fields[1]);
            dealii::PackagedOperation<VectorType> result;
            result.reinit_vector = [owned =
                                      problem.owned[row]](VectorType &vector,
                                                          const bool  omit) {
              vector.reinit(owned, MPI_COMM_WORLD, omit);
            };
            result.apply =
              [primal, multiplier, row, &problem](VectorType &dst) {
                problem.matrices[2 * row].vmult(dst, *primal);
                problem.matrices[2 * row + 1].vmult_add(dst, *multiplier);
                dst -= problem.rhs[row];
              };
            result.apply_add = [apply = result.apply](VectorType &dst) {
              VectorType contribution;
              contribution.reinit(dst);
              apply(contribution);
              dst += contribution;
            };
            return result;
          });
        for (unsigned int column = 0; column < 2; ++column)
          {
            const auto slot = 2 * row + column;
            builder.term(fields[row], "matrix recipe")
              .state(fields[column],
                     MatrixFactory([slot, &problem](const auto &context) {
                       EXPECT_FALSE(context.has_state_derivative());
                       auto result =
                         matrix_operator<VectorType>(problem.matrices[slot]);
                       const auto count_action =
                         [slot, &problem](const auto &action) {
                           return
                             [slot, &problem, action](VectorType       &dst,
                                                      const VectorType &src) {
                               ++problem.recipe_actions[slot];
                               action(dst, src);
                             };
                         };
                       result.view.vmult = count_action(result.view.vmult);
                       result.view.vmult_add =
                         count_action(result.view.vmult_add);
                       result.view.Tvmult = count_action(result.view.Tvmult);
                       result.view.Tvmult_add =
                         count_action(result.view.Tvmult_add);
                       result.direct_matrix   = false;
                       result.observes_matrix = false;
                       result.observed_matrix = nullptr;
                       result.materialize     = [slot, &problem] {
                         ++problem.materializations[slot];
                         auto matrix = std::make_shared<MatrixType>();
                         matrix->reinit(problem.matrices[slot]);
                         matrix->copy_from(problem.matrices[slot]);
                         return matrix;
                       };
                       return result;
                     }));
          }
      }
    return {fields[0], fields[1]};
  }
} // namespace ImmersX

TEST(LinearAdapter, BOTH_SchurSolveRetainsRecipesAndRefreshesMatrices)
{
  using VectorType = ImmersX::ImmersXLA::MPI::Vector;
  using Adapter =
    ImmersX::LinearAdapter<VectorType, ImmersX::ImmersXLA::MPI::BlockVector>;
  dealii::ParameterAcceptor::clear();
  ImmersX::RetainedSolveProblem problem;
  problem.owned[0] =
    dealii::Utilities::MPI::create_evenly_distributed_partitioning(
      MPI_COMM_WORLD, 2);
  problem.owned[1] =
    dealii::Utilities::MPI::create_evenly_distributed_partitioning(
      MPI_COMM_WORLD, 1);
  EXPECT_EQ(problem.owned[1].n_elements(),
            dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 ? 1u :
                                                                            0u);
  for (unsigned int row = 0; row < 2; ++row)
    {
      problem.rhs[row].reinit(problem.owned[row], MPI_COMM_WORLD);
      for (const auto i : problem.owned[row])
        problem.rhs[row][i] = row == 0 ? (i == 0 ? 5. : 7.) : 2.;
      problem.rhs[row].compress(dealii::VectorOperation::insert);
      for (unsigned int column = 0; column < 2; ++column)
        {
          dealii::DynamicSparsityPattern graph(problem.owned[row].size(),
                                               problem.owned[column].size(),
                                               problem.owned[row]);
          for (const auto i : problem.owned[row])
            for (unsigned int j = 0; j < problem.owned[column].size(); ++j)
              graph.add(i, j);
          problem.matrices[2 * row + column].reinit(problem.owned[row],
                                                    problem.owned[column],
                                                    graph,
                                                    MPI_COMM_WORLD);
        }
    }
  ImmersX::LinearAdapterParameters parameters;
  parameters.solver             = ImmersX::LinearSolver::iterative;
  parameters.preconditioner     = ImmersX::LinearPreconditioner::schur;
  parameters.tolerance          = 1.e-11;
  parameters.maximum_iterations = 30;
  Adapter    adapter(parameters, MPI_COMM_WORLD);
  const auto fields = adapter.add(problem, "retained solve").fields();
  auto       state  = adapter.make_state();
  const std::array<std::array<double, 3>, 2> expected = {{
    {{19. / 11., 20. / 11., 17. / 11.}},
    {{66. / 61., 69. / 61., 41. / 61.}},
  }};
  std::array<unsigned int, 4>                previous_materializations{};
  for (unsigned int solve = 0; solve < 2; ++solve)
    {
      // [A D; B -m], f=(5,7), g=2. The second solve changes all four blocks.
      const std::array<std::vector<double>, 4> values = {
        {std::vector<double>{
           solve == 0 ? 2. : 4., 0., 0., solve == 0 ? 3. : 5.},
         std::vector<double>{1., solve == 0 ? 1. : 2.},
         std::vector<double>{1., solve == 0 ? 1. : 2.},
         std::vector<double>{solve == 0 ? -1. : -2.}}};
      for (unsigned int slot = 0; slot < 4; ++slot)
        {
          auto &matrix = problem.matrices[slot];
          for (const auto i : problem.owned[slot / 2])
            for (unsigned int j = 0; j < matrix.n(); ++j)
              matrix.set(i, j, values[slot][i * matrix.n() + j]);
          matrix.compress(dealii::VectorOperation::insert);
        }
      adapter.solve(state);
      const auto &primal     = adapter.field(state, fields.primal);
      const auto &multiplier = adapter.field(state, fields.multiplier);
      for (const auto i : problem.owned[0])
        EXPECT_NEAR(primal[i], expected[solve][i], 1.e-10);
      for (const auto i : problem.owned[1])
        EXPECT_NEAR(multiplier[i], expected[solve][2], 1.e-10);
      for (unsigned int slot = 0; slot < 4; ++slot)
        {
          EXPECT_EQ(problem.recipe_actions[slot], 0u);
          EXPECT_GT(problem.materializations[slot],
                    previous_materializations[slot]);
          previous_materializations[slot] = problem.materializations[slot];
        }
    }
}
