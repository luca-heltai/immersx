#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>

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
  Adapter    adapter(MPI_COMM_WORLD,
                  [](const auto &operator_view,
                     const auto &rhs,
                     auto       &solution) {
                    (void)operator_view;
                    solution = rhs;
                    solution.block(0)[0] /= 2.;
                    solution.block(0)[1] /= 4.;
                  });
  const auto fields   = adapter.add(problem, "fake");
  const auto observed = fields.observe(fields.fields().solution);
  auto       state    = adapter.make_state();
  EXPECT_EQ(observed.source(), fields.fields().solution);
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

  ImmersX::LinearSolverOptions default_options;
  default_options.preconditioner =
    ImmersX::LinearPreconditioner::block_diagonal;
  Adapter    default_adapter(MPI_COMM_WORLD, default_options);
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

  ImmersX::LinearSolverOptions direct_options;
  direct_options.solver = ImmersX::LinearSolver::direct;
  Adapter    direct_adapter(MPI_COMM_WORLD, direct_options);
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

TEST(LinearAdapter, ParameterAcceptorControlsSolverOptions)
{
  dealii::ParameterAcceptor::clear();

  using Adapter = ImmersX::LinearAdapter<ImmersX::ImmersXLA::MPI::Vector,
                                         ImmersX::ImmersXLA::MPI::BlockVector>;
  Adapter adapter(MPI_COMM_WORLD,
                  ImmersX::LinearSolverOptions{},
                  Adapter::SolveFunction{},
                  "Linear adapter parameters");

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
  Adapter    adapter(MPI_COMM_WORLD, [](const auto &, const auto &, auto &) {});
  const auto fields = adapter.add(problem, "triangular");
  auto       state  = adapter.make_state();
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
  Adapter    adapter(MPI_COMM_WORLD, [](const auto &, const auto &, auto &) {});
  const auto fields                            = adapter.add(problem, "schur");
  auto       state                             = adapter.make_state();
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

  EXPECT_NEAR(normal[0], 462., 1.e-12);
  EXPECT_NEAR(transpose[0], 693., 1.e-12);

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

  const double normal_multiplier = 30. / 462.;
  EXPECT_NEAR(adapter.field(preconditioned, fields.fields().multiplier)[0],
              normal_multiplier,
              1.e-10);
  EXPECT_NEAR(adapter.field(preconditioned, fields.fields().primal)[0],
              2. - 14. * normal_multiplier,
              1.e-10);
  EXPECT_NEAR(adapter.field(preconditioned, fields.fields().primal)[1],
              2. - 22. * normal_multiplier,
              1.e-10);

  const double transpose_multiplier = 201. / 693.;
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
}
