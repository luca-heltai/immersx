#include <deal.II/base/mpi.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>

#include <gtest/gtest.h>
#include <immersx/algebra/lagrange_multiplier_constraint_solver.h>
#include <immersx/algebra/lagrange_multiplier_schur_solver.h>
#include <immersx/core/detail/execution_composition.h>

#include <array>

namespace ImmersX
{
  struct OutsideMassProblem
  {
    using Matrix = ImmersXLA::MPI::SparseMatrix;
    dealii::IndexSet owned;
    Matrix           bulk, embedded, coupling, mass, negative_mass;
    Matrix           outside, negative_outside, metric;
  };

  struct OutsideMassFields
  {
    FieldId bulk, embedded, multiplier;
  };

  template <typename Builder>
  OutsideMassFields
  contribute(Builder &builder, const OutsideMassProblem &problem)
  {
    const auto bulk     = builder.algebraic_field("bulk", problem.owned);
    const auto embedded = builder.algebraic_field("embedded", problem.owned);
    const auto multiplier =
      builder.algebraic_field("multiplier", problem.owned);
    const auto inverse = [](const auto &matrix, const auto &reinit) {
      return make_local_preconditioner<ImmersXLA::MPI::Vector,
                                       ImmersXLA::MPI::SparseMatrix,
                                       ImmersXLA::MPI::PreconditionJacobi>(
        matrix, reinit);
    };
    builder.preconditioner(bulk, inverse);
    builder.preconditioner(embedded, inverse);
    builder.saddle_point(multiplier,
                         {bulk, embedded},
                         builder.matrix_operator(problem.metric));
    builder.term(bulk, "diagonal")
      .state(bulk, builder.matrix_operator(problem.bulk));
    builder.term(embedded, "diagonal")
      .state(embedded, builder.matrix_operator(problem.embedded));
    builder.term(bulk, "coupling")
      .state(multiplier, builder.matrix_operator(problem.coupling));
    builder.term(multiplier, "bulk")
      .state(bulk, builder.matrix_operator(problem.coupling));
    builder.term(embedded, "coupling")
      .state(multiplier, builder.matrix_operator(problem.negative_mass));
    builder.term(multiplier, "embedded")
      .state(embedded, builder.matrix_operator(problem.negative_mass));
    builder.term(multiplier, "outside")
      .state(multiplier, builder.matrix_operator(problem.negative_outside))
      .derivative(multiplier,
                  builder.matrix_operator(problem.negative_outside));
    return {bulk, embedded, multiplier};
  }
} // namespace ImmersX

namespace
{
  using Matrix         = ImmersX::ImmersXLA::MPI::SparseMatrix;
  using Vector         = ImmersX::ImmersXLA::MPI::Vector;
  using BlockVector    = ImmersX::ImmersXLA::MPI::BlockVector;
  using Preconditioner = ImmersX::ImmersXLA::MPI::PreconditionJacobi;
  using Composition =
    ImmersX::detail::ExecutionComposition<Vector, BlockVector>;

  void
  initialize(Matrix                      &matrix,
             const dealii::IndexSet      &owned,
             const std::array<double, 4> &entries)
  {
    dealii::DynamicSparsityPattern pattern(2, 2);
    for (const auto row : owned)
      for (unsigned int column = 0; column < 2; ++column)
        pattern.add(row, column);
    matrix.reinit(owned, owned, pattern, MPI_COMM_WORLD);
    for (const auto row : owned)
      for (unsigned int column = 0; column < 2; ++column)
        matrix.set(row, column, entries[2 * row + column]);
    matrix.compress(dealii::VectorOperation::insert);
  }

  void
  initialize(ImmersX::OutsideMassProblem &problem, const bool all_outside)
  {
    problem.owned =
      dealii::Utilities::MPI::create_evenly_distributed_partitioning(
        MPI_COMM_WORLD, 2);
    initialize(problem.bulk, problem.owned, {{2., 0., 0., 3.}});
    initialize(problem.embedded, problem.owned, {{4., 0., 0., 5.}});
    initialize(problem.coupling,
               problem.owned,
               {{all_outside ? 0. : 1., 0., 0., 0.}});
    initialize(problem.mass,
               problem.owned,
               {{all_outside ? 0. : 0.5, 0., 0., 0.}});
    initialize(problem.negative_mass,
               problem.owned,
               {{all_outside ? 0. : -0.5, 0., 0., 0.}});
    initialize(problem.outside, problem.owned, {{0.5, 0.125, 0.125, 2.}});
    initialize(problem.negative_outside,
               problem.owned,
               {{-0.5, -0.125, -0.125, -2.}});
    initialize(problem.metric, problem.owned, {{2., 0., 0., 3.}});
  }

  // Independent dense physical block, with no Schur or row-transformation code.
  dealii::FullMatrix<double>
  physical(const bool all_outside, const double alpha = 0.)
  {
    dealii::FullMatrix<double> result(6, 6);
    result(0, 0) = 2.;
    result(1, 1) = 3.;
    result(2, 2) = 4.;
    result(3, 3) = 5.;
    result(0, 4) = result(4, 0) = all_outside ? 0. : 1.;
    result(2, 4) = result(4, 2) = all_outside ? 0. : -0.5;
    result(4, 4)                = -0.5 * (1. + alpha);
    result(4, 5) = result(5, 4) = -0.125 * (1. + alpha);
    result(5, 5)                = -2. * (1. + alpha);
    return result;
  }

  dealii::Vector<double>
  direct_solution(dealii::FullMatrix<double>    matrix,
                  const dealii::Vector<double> &rhs)
  {
    matrix.gauss_jordan();
    dealii::Vector<double> result(rhs.size());
    matrix.vmult(result, rhs);
    return result;
  }

  void
  compare(const BlockVector &actual, const dealii::Vector<double> &expected)
  {
    for (unsigned int block = 0; block < actual.n_blocks(); ++block)
      for (const auto index : actual.block(block).locally_owned_elements())
        EXPECT_NEAR(actual.block(block)[index],
                    expected[2 * block + index],
                    1.e-10);
  }

  void
  assign(BlockVector &vector, const dealii::Vector<double> &values)
  {
    for (unsigned int block = 0; block < vector.n_blocks(); ++block)
      for (const auto index : vector.block(block).locally_owned_elements())
        vector.block(block)[index] = values[2 * block + index];
    vector.compress(dealii::VectorOperation::insert);
  }
} // namespace

TEST(LinearAdapter, BOTH_OutsideMassStandaloneSolversMatchDirectBlock)
{
  for (const bool all_outside : {false, true})
    {
      SCOPED_TRACE(all_outside);
      ImmersX::OutsideMassProblem problem;
      initialize(problem, all_outside);
      Vector bulk_rhs, embedded_rhs, constraint_rhs;
      bulk_rhs.reinit(problem.owned, MPI_COMM_WORLD);
      embedded_rhs.reinit(bulk_rhs);
      constraint_rhs.reinit(bulk_rhs);
      for (const auto i : problem.owned)
        {
          bulk_rhs[i]       = 1. + i;
          embedded_rhs[i]   = 3. + i;
          constraint_rhs[i] = 0.75 + i;
        }
      bulk_rhs.compress(dealii::VectorOperation::insert);
      embedded_rhs.compress(dealii::VectorOperation::insert);
      constraint_rhs.compress(dealii::VectorOperation::insert);
      Vector bulk, embedded, multiplier;
      ImmersX::LagrangeMultiplierSchurSolver<Matrix, Vector, Preconditioner>
        solver(problem.bulk,
               problem.embedded,
               problem.coupling,
               problem.mass,
               problem.owned,
               problem.owned,
               problem.owned,
               MPI_COMM_WORLD,
               100,
               1.e-12,
               1.e-14,
               &problem.outside);
      solver.solve(bulk, embedded, multiplier, bulk_rhs, embedded_rhs);
      dealii::Vector<double> rhs(6);
      rhs[0]              = 1.;
      rhs[1]              = 2.;
      rhs[2]              = 3.;
      rhs[3]              = 4.;
      const auto expected = direct_solution(physical(all_outside), rhs);
      for (const auto i : problem.owned)
        {
          EXPECT_NEAR(bulk[i], expected[i], 1.e-10);
          EXPECT_NEAR(embedded[i], expected[2 + i], 1.e-10);
          EXPECT_NEAR(multiplier[i], expected[4 + i], 1.e-10);
        }
      ImmersX::
        LagrangeMultiplierConstraintSolver<Matrix, Vector, Preconditioner>
          prescribed(problem.bulk,
                     problem.coupling,
                     problem.owned,
                     problem.owned,
                     MPI_COMM_WORLD,
                     100,
                     1.e-12,
                     1.e-14,
                     &problem.outside);
      prescribed.solve(bulk, multiplier, bulk_rhs, constraint_rhs);
      const auto                        full    = physical(all_outside);
      const std::array<unsigned int, 4> indices = {{0, 1, 4, 5}};
      dealii::FullMatrix<double>        constrained(4, 4);
      for (unsigned int i = 0; i < 4; ++i)
        for (unsigned int j = 0; j < 4; ++j)
          constrained(i, j) = full(indices[i], indices[j]);
      dealii::Vector<double> constrained_rhs(4);
      constrained_rhs[0] = 1.;
      constrained_rhs[1] = 2.;
      constrained_rhs[2] = 0.75;
      constrained_rhs[3] = 1.75;
      const auto constrained_expected =
        direct_solution(constrained, constrained_rhs);
      for (const auto i : problem.owned)
        {
          EXPECT_NEAR(bulk[i], constrained_expected[i], 1.e-10);
          EXPECT_NEAR(multiplier[i], constrained_expected[2 + i], 1.e-10);
        }
    }
}

TEST(LinearAdapter, BOTH_OutsideMassSchurIncludesStateAndDerivativeDiagonal)
{
  for (const bool all_outside : {false, true})
    for (const double alpha : {0., 0.25})
      {
        SCOPED_TRACE(all_outside);
        SCOPED_TRACE(alpha);
        ImmersX::OutsideMassProblem problem;
        initialize(problem, all_outside);
        Composition composition(MPI_COMM_WORLD);
        const auto  fields = composition.add(problem, "outside");
        const auto  state  = composition.make_state();
        Vector      input, actual;
        input.reinit(problem.owned, MPI_COMM_WORLD);
        actual.reinit(input);
        for (const auto i : problem.owned)
          input[i] = i == 0 ? 2. : -1.;
        input.compress(dealii::VectorOperation::insert);
        const auto schur =
          composition.schur_operator(fields.multiplier, state, nullptr, alpha);
        schur.vmult(actual, input);
        const double first  = (1. + alpha) * 0.875 + (all_outside ? 0. : 1.125);
        const double second = -(1. + alpha) * 1.75;
        for (const auto i : problem.owned)
          EXPECT_NEAR(actual[i], i == 0 ? first : second, 1.e-12);
        auto transpose = actual;
        schur.Tvmult(transpose, input);
        transpose -= actual;
        EXPECT_NEAR(transpose.l2_norm(), 0., 1.e-12);

        dealii::Vector<double> rhs(6);
        for (unsigned int i = 0; i < 6; ++i)
          rhs[i] = 1. + i;
        auto source   = composition.make_state();
        auto solution = composition.make_state();
        assign(source, rhs);
        const auto preconditioner = composition.schur_preconditioner(
          fields.multiplier, state, nullptr, alpha);
        preconditioner.vmult(solution, source);
        compare(solution, direct_solution(physical(all_outside, alpha), rhs));
        preconditioner.Tvmult(solution, source);
        compare(solution, direct_solution(physical(all_outside, alpha), rhs));
      }
}

TEST(LinearAdapter, BOTH_OutsideMassAugmentationMatchesFullRowTransformation)
{
  const double gamma = 3.;
  for (const bool all_outside : {false, true})
    for (const double alpha : {0., 0.25})
      {
        SCOPED_TRACE(all_outside);
        SCOPED_TRACE(alpha);
        ImmersX::OutsideMassProblem problem;
        initialize(problem, all_outside);
        Composition composition(MPI_COMM_WORLD);
        composition.add(problem, "outside");
        const auto state           = composition.make_state();
        const auto physical_matrix = physical(all_outside, alpha);
        dealii::FullMatrix<double> transform(6, 6);
        for (unsigned int i = 0; i < 6; ++i)
          transform(i, i) = 1.;
        // W = diag(2,3); the two primal rows observe only multiplier 0.
        transform(0, 4) = all_outside ? 0. : gamma / 2.;
        transform(2, 4) = all_outside ? 0. : -gamma / 4.;
        dealii::FullMatrix<double> expected_matrix(6, 6);
        transform.mmult(expected_matrix, physical_matrix);
        dealii::Vector<double> source_values(6), expected(6);
        for (unsigned int i = 0; i < 6; ++i)
          source_values[i] = 1. + 0.5 * i;
        auto source = composition.make_state();
        auto actual = composition.make_state();
        assign(source, source_values);
        const auto op = composition.augmented_lagrangian_operator(state,
                                                                  gamma,
                                                                  nullptr,
                                                                  alpha);
        const auto matrix =
          composition.augmented_lagrangian_matrix(state, gamma, nullptr, alpha);
        expected_matrix.vmult(expected, source_values);
        op.vmult(actual, source);
        compare(actual, expected);
        op.vmult_add(actual, source);
        expected *= 2.;
        compare(actual, expected);
        matrix.vmult(actual, source);
        expected *= 0.5;
        compare(actual, expected);
        expected_matrix.Tvmult(expected, source_values);
        op.Tvmult(actual, source);
        compare(actual, expected);
        op.Tvmult_add(actual, source);
        expected *= 2.;
        compare(actual, expected);
        matrix.Tvmult(actual, source);
        expected *= 0.5;
        compare(actual, expected);
        // The same row transformation of the RHS preserves the physical solve.
        dealii::Vector<double> transformed_rhs(6);
        transform.vmult(transformed_rhs, source_values);
        const auto augmented_solution =
          direct_solution(expected_matrix, transformed_rhs);
        const auto physical_solution =
          direct_solution(physical_matrix, source_values);
        for (unsigned int i = 0; i < 6; ++i)
          EXPECT_NEAR(augmented_solution[i], physical_solution[i], 1.e-12);
      }
}

TEST(LinearAdapter, BOTH_OutsideMassALPreconditionsPhysicalSystem)
{
  const double gamma = 3.;
  for (const bool all_outside : {false, true})
    {
      SCOPED_TRACE(all_outside);
      ImmersX::OutsideMassProblem problem;
      initialize(problem, all_outside);
      Composition            composition(MPI_COMM_WORLD);
      const auto             fields   = composition.add(problem, "outside");
      const auto             state    = composition.make_state();
      auto                   rhs      = composition.make_state();
      auto                   solution = composition.make_state();
      dealii::Vector<double> values(6);
      for (unsigned int i = 0; i < 6; ++i)
        values[i] = 1. + i;
      assign(rhs, values);
      const auto preconditioner =
        composition.augmented_lagrangian_preconditioner(state, gamma);
      preconditioner.vmult(solution, rhs);
      const auto &lambda = composition.field(solution, fields.multiplier);
      for (const auto i : problem.owned)
        EXPECT_NEAR(lambda[i], -gamma * values[4 + i] / (2. + i), 1.e-12);
      EXPECT_GT(lambda.l2_norm(), 0.);
      solution = 0.;
      dealii::SolverControl            control(100, 1.e-11);
      dealii::SolverGMRES<BlockVector> solver(control);
      solver.solve(composition.jacobian(0., state, nullptr, 0.),
                   solution,
                   rhs,
                   preconditioner);
      compare(solution, direct_solution(physical(all_outside), values));
    }
}
