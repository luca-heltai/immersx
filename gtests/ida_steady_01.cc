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
