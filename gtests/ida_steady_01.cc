// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/config.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <array>
#include <limits>

#if defined(DEAL_II_WITH_SUNDIALS) && defined(DEAL_II_WITH_MUMPS) && \
  !defined(IMMERSX_USE_PETSC_LA)
namespace
{
  using namespace ImmersX;
  using Vector  = ImmersXLA::MPI::Vector;
  using Global  = ImmersXLA::MPI::BlockVector;
  using Matrix  = ImmersXLA::MPI::SparseMatrix;
  using Adapter = IDAAdapter<Vector, Global>;

  auto
  add_pair(Adapter   &adapter,
           const bool zero_row      = false,
           const bool nonfinite_row = false,
           const bool empty_rank    = false)
  {
    return adapter.add([=](auto &builder) {
      using Model = typename std::decay_t<decltype(builder)>::Model;
      dealii::IndexSet owned(2);
      const auto       rank =
        dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      const auto size = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      for (unsigned int i = 0; i < 2; ++i)
        if (empty_rank ? rank == 0 : i % size == rank)
          owned.add_index(i);
      owned.compress();
      const auto field  = builder.algebraic_field("pair", owned);
      const auto values = [field](const auto &context) {
        std::array<double, 2> local{};
        const auto           &state = context.state(field);
        for (const auto index : state.locally_owned_elements())
          local[index] = state[index];
        std::array<double, 2> result{};
        for (unsigned int i = 0; i < 2; ++i)
          result[i] = dealii::Utilities::MPI::sum(local[i], MPI_COMM_WORLD);
        return result;
      };
      typename Model::ResidualFactory residual = [=](const auto &context) {
        const auto                y = values(context);
        typename Model::Operation op;
        op.reinit_vector = [=](Vector &dst, const bool omit) {
          dst.reinit(owned, MPI_COMM_WORLD, omit);
        };
        op.apply = [=](Vector &dst) {
          dst = 0.;
          for (const auto index : owned)
            dst[index] = index == 0 ? y[0] * y[0] - y[1] : y[0] + y[1];
          dst.compress(dealii::VectorOperation::insert);
        };
        op.apply_add = [apply = op.apply](Vector &dst) {
          Vector increment;
          increment.reinit(dst);
          apply(increment);
          dst += increment;
        };
        return op;
      };
      typename Model::MatrixOperatorFactory matrix = [=](const auto &context) {
        const auto                     y = values(context);
        dealii::DynamicSparsityPattern sparsity(2, 2);
        for (const auto row : owned)
          for (unsigned int column = 0; column < 2; ++column)
            sparsity.add(row, column);
        auto entries = std::make_shared<Matrix>();
        entries->reinit(owned, owned, sparsity, MPI_COMM_WORLD);
        for (const auto row : owned)
          {
            entries->set(row,
                         0,
                         row == 0 ? (nonfinite_row ?
                                       std::numeric_limits<double>::max() / 2. :
                                       2. * y[0]) :
                                    (zero_row ? 0. : 1.));
            entries->set(row, 1, row == 0 ? -1. : (zero_row ? 0. : 1.));
          }
        entries->compress(dealii::VectorOperation::insert);
        return matrix_operator<Vector>(entries);
      };
      builder.term(field, "nonlinear pair")
        .residual(residual)
        .state(field, matrix);
      return field;
    });
  }

  template <class Check>
  void
  with_pair(Check check,
            bool  zero      = false,
            bool  nonfinite = false,
            bool  empty     = false)
  {
    dealii::ParameterAcceptor::clear();
    TimeIntervalParameters time("/", false);
    IDAParameters          parameters("/", &time);
    Adapter                adapter(time, parameters, MPI_COMM_WORLD);
    adapter.set_linear_solver(Adapter::LinearSolver::mumps);
    const auto field    = add_pair(adapter, zero, nonfinite, empty);
    auto       state    = adapter.make_state();
    state               = .5;
    auto dot            = adapter.make_state();
    dot                 = 0.;
    auto characteristic = adapter.make_state();
    characteristic      = 1.;
    auto scaling        = adapter.make_state();
    check(adapter, field.fields(), state, dot, characteristic, scaling);
    dealii::ParameterAcceptor::clear();
  }
} // namespace

TEST(DistributedIDA, BOTH_SteadyAbsoluteRowScalingAndFreshSetup)
{
  with_pair([](auto       &adapter,
               const auto &field,
               auto       &state,
               auto       &dot,
               auto       &characteristic,
               auto       &scaling) {
    adapter.setup_steady_jacobian(0., state, dot, &characteristic, &scaling);
    for (const auto index : scaling.locally_owned_elements())
      EXPECT_DOUBLE_EQ(scaling[index], .5);
    // J*[1,1] has a zero first row; sum_j |Jij|*cj is two, not zero.
    auto source = characteristic;
    auto action = adapter.make_state();
    adapter.current_jacobian().vmult(action, source);
    for (const auto index :
         adapter.field(action, field).locally_owned_elements())
      EXPECT_DOUBLE_EQ(adapter.field(action, field)[index],
                       index == 0 ? 0. : 2.);
    adapter.field(state, field) = 1.5;
    auto residual               = adapter.make_state();
    adapter.solver().residual(0., state, dot, residual);
    adapter.setup_steady_jacobian(0., state, dot);
    adapter.current_jacobian().vmult(action, source);
    for (const auto index : action.locally_owned_elements())
      EXPECT_DOUBLE_EQ(action[index], 2.);
    for (const auto index : scaling.locally_owned_elements())
      EXPECT_DOUBLE_EQ(scaling[index], .5);
  });
}

TEST(DistributedIDA, BOTH_SteadyRowViewsUseImportedColumnCharacteristics)
{
  with_pair([](auto       &adapter,
               const auto &field,
               auto       &state,
               auto       &dot,
               auto       &characteristic,
               auto       &scaling) {
    adapter.field(state, field) = .75;
    auto &values                = adapter.field(characteristic, field);
    for (const auto index : values.locally_owned_elements())
      values[index] = index == 0 ? 2. : 5.;
    characteristic.compress(dealii::VectorOperation::insert);
    adapter.setup_steady_jacobian(0., state, dot, &characteristic, &scaling);
    // Independent dense oracle: |[[1.5,-1],[1,1]]| * [2,5] = [8,7].
    // MPI2 assigns the two columns to different ranks, requiring both ghosts.
    for (const auto index :
         adapter.field(scaling, field).locally_owned_elements())
      EXPECT_DOUBLE_EQ(adapter.field(scaling, field)[index],
                       index == 0 ? 1. / 8. : 1. / 7.);
  });
}

TEST(DistributedIDA, MPI_SemanticFieldsUseDisjointOwnedRows)
{
  using namespace dealii;
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ASSERT_GE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  IndexSet owned(2);
  if (rank < 2)
    owned.add_index(rank);
  owned.compress();
  for (const bool diagonal : {false, true})
    {
      ParameterAcceptor::clear();
      const std::array<std::array<double, 4>, 4> coefficients =
        diagonal ? std::array<std::array<double, 4>, 4>{{{{2., 0., 0., 0.}},
                                                         {{0., -3., 0., 0.}},
                                                         {{0., 0., 4., 0.}},
                                                         {{0., 0., 0., -5.}}}} :
                   std::array<std::array<double, 4>, 4>{{{{2., -1., 1., 0.}},
                                                         {{-1., 3., 0., -1.}},
                                                         {{1., 0., 4., -2.}},
                                                         {{0., 1., -2., 5.}}}};
      TimeIntervalParameters time("/", false);
      IDAParameters          parameters("/", &time);
      Adapter                adapter(time, parameters, MPI_COMM_WORLD);
      adapter.set_linear_solver(Adapter::LinearSolver::mumps);
      const auto fields = adapter.add([&](auto &builder) {
        using Model = typename std::decay_t<decltype(builder)>::Model;
        const std::array<ImmersX::FieldId, 2> ids{
          {builder.algebraic_field("first", owned),
           builder.algebraic_field("second", owned)}};
        for (unsigned int row = 0; row < 2; ++row)
          for (unsigned int column = 0; column < 2; ++column)
            {
              typename Model::MatrixOperatorFactory factory =
                [owned, coefficients, row, column](const auto &) {
                  DynamicSparsityPattern sparsity(2, 2);
                  for (const auto i : owned)
                    for (unsigned int j = 0; j < 2; ++j)
                      sparsity.add(i, j);
                  auto matrix = std::make_shared<Matrix>();
                  matrix->reinit(owned, owned, sparsity, MPI_COMM_WORLD);
                  for (const auto i : owned)
                    for (unsigned int j = 0; j < 2; ++j)
                      matrix->set(i,
                                  j,
                                  coefficients[2 * row + i][2 * column + j]);
                  matrix->compress(VectorOperation::insert);
                  return ImmersX::matrix_operator<
                    ImmersX::ImmersXLA::MPI::Vector>(matrix);
                };
              builder
                .term(ids[row],
                      "block-" + std::to_string(row) + "-" +
                        std::to_string(column))
                .state(ids[column], factory);
            }
        return ids;
      });
      auto       state = adapter.make_state(), dot = adapter.make_state();
      auto       characteristic = adapter.make_state(),
           scaling              = adapter.make_state();
      state                     = 0.;
      dot                       = 0.;
      const std::array<double, 4> weights{{2., 3., 5., 7.}};
      const std::array<double, 4> sums =
        diagonal ? std::array<double, 4>{{4., 9., 20., 35.}} :
                   std::array<double, 4>{{12., 18., 36., 48.}};
      for (unsigned int block = 0; block < 2; ++block)
        for (const auto i : owned)
          adapter.field(characteristic, fields.fields()[block])[i] =
            weights[2 * block + i];
      characteristic.compress(VectorOperation::insert);
      IndexSet expected_owned(4);
      if (rank < 2)
        {
          expected_owned.add_index(rank);
          expected_owned.add_index(rank + 2);
        }
      expected_owned.compress();
      EXPECT_EQ(characteristic.locally_owned_elements(), expected_owned);
      adapter.setup_steady_jacobian(0., state, dot, &characteristic, &scaling);
      // Independent dense absolute-row oracle; each rank needs ghost weights
      // from both fields. Stored off-diagonal zeros also test diagonal
      // detection.
      for (unsigned int block = 0; block < 2; ++block)
        for (const auto i : owned)
          EXPECT_DOUBLE_EQ(adapter.field(scaling, fields.fields()[block])[i],
                           1. / sums[2 * block + i]);
      const std::array<double, 4> exact{{1., -2., 3., -4.}};
      const std::array<double, 4> forcing =
        diagonal ? std::array<double, 4>{{2., 6., 12., 20.}} :
                   std::array<double, 4>{{7., -3., 21., -28.}};
      auto rhs = adapter.make_state(), solution = adapter.make_state();
      for (unsigned int block = 0; block < 2; ++block)
        for (const auto i : owned)
          adapter.field(rhs, fields.fields()[block])[i] =
            forcing[2 * block + i];
      rhs.compress(VectorOperation::insert);
      adapter.solver().solve_with_jacobian(rhs, solution, 1.e-12);
      for (unsigned int block = 0; block < 2; ++block)
        for (const auto i : owned)
          EXPECT_NEAR(adapter.field(solution, fields.fields()[block])[i],
                      exact[2 * block + i],
                      1.e-12);
      auto defect = adapter.make_state();
      adapter.current_jacobian().vmult(defect, solution);
      defect -= rhs;
      EXPECT_LE(defect.l2_norm(), 1.e-12 * rhs.l2_norm());
      ParameterAcceptor::clear();
    }
}

TEST(DistributedIDA, BOTH_SteadyScalingRejectsZeroRows)
{
  with_pair(
    [](auto &adapter,
       const auto &,
       auto &state,
       auto &dot,
       auto &characteristic,
       auto &scaling) {
      EXPECT_THROW(adapter.setup_steady_jacobian(
                     0., state, dot, &characteristic, &scaling),
                   dealii::ExceptionBase);
    },
    true);
}

TEST(DistributedIDA, BOTH_SteadyScalingRejectsOverflowingRowScale)
{
  with_pair(
    [](auto &adapter,
       const auto &,
       auto &state,
       auto &dot,
       auto &characteristic,
       auto &scaling) {
      characteristic = 4.;
      EXPECT_THROW(adapter.setup_steady_jacobian(
                     0., state, dot, &characteristic, &scaling),
                   dealii::ExceptionBase);
    },
    false,
    true);
}

TEST(DistributedIDA, BOTH_SteadyScalingRejectsNonpositiveCharacteristics)
{
  with_pair([](auto &adapter,
               const auto &,
               auto &state,
               auto &dot,
               auto &characteristic,
               auto &scaling) {
    characteristic = 0.;
    EXPECT_THROW(
      adapter.setup_steady_jacobian(0., state, dot, &characteristic, &scaling),
      dealii::ExceptionBase);
    characteristic = std::numeric_limits<double>::max();
    characteristic *= 2.;
    EXPECT_THROW(
      adapter.setup_steady_jacobian(0., state, dot, &characteristic, &scaling),
      dealii::ExceptionBase);
  });
}

TEST(DistributedIDA, BOTH_SteadyScalingHandlesAnEmptyRank)
{
  with_pair(
    [](auto &adapter,
       const auto &,
       auto &state,
       auto &dot,
       auto &characteristic,
       auto &scaling) {
      adapter.setup_steady_jacobian(0., state, dot, &characteristic, &scaling);
      for (const auto index : scaling.locally_owned_elements())
        EXPECT_DOUBLE_EQ(scaling[index], .5);
    },
    false,
    false,
    true);
}
#endif
