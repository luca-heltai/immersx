// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <string>

using namespace ImmersX;

#ifdef DEAL_II_WITH_SUNDIALS

TEST(DistributedIDA, MPI_StateDependentJacobianOwnsEvaluationState) // NOLINT
{
  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = IDAAdapter<FieldVector, GlobalVector>;

  Adapter::AdditionalData data;
  data.initial_time = 0.;
  data.final_time   = 0.01;
  Adapter adapter(data,
                  MPI_COMM_WORLD,
                  [](const dealii::LinearOperator<GlobalVector> &,
                     const GlobalVector &,
                     GlobalVector &,
                     const double) {});

  const auto field = adapter.add(
    [](auto &builder) {
      dealii::IndexSet owned(2);
      owned.add_index(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
      owned.compress();
      const auto value = builder.algebraic_field("value", owned);
      using Model      = typename std::decay_t<decltype(builder)>::Model;
      typename Model::OperatorFactory factory = [value](const auto &context) {
        const auto              &state = context.state(value);
        typename Model::Operator result;
        result.reinit_range_vector = [state](FieldVector &vector, bool) {
          vector.reinit(state);
        };
        result.reinit_domain_vector = result.reinit_range_vector;
        result.vmult                = [state](FieldVector       &destination,
                               const FieldVector &source) {
          destination = source;
          destination *= state(*state.locally_owned_elements().begin());
        };
        result.vmult_add = [state](FieldVector       &destination,
                                   const FieldVector &source) {
          FieldVector contribution;
          contribution.reinit(source);
          contribution = source;
          contribution *= state(*state.locally_owned_elements().begin());
          destination += contribution;
        };
        result.Tvmult     = result.vmult;
        result.Tvmult_add = result.vmult_add;
        return result;
      };
      builder.term(value, "state-dependent").state(value, factory);
      return value;
    },
    "state");

  auto state                           = adapter.make_state();
  auto state_dot                       = adapter.make_state();
  adapter.field(state, field.fields()) = 3.;
  adapter.solver().setup_jacobian(0., state, state_dot, 0.);

  auto source = adapter.make_state();
  auto result = adapter.make_state();
  source      = 1.;
  adapter.current_jacobian().vmult(result, source);
  const auto rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  EXPECT_EQ(adapter.field(state, field.fields())[rank], 3.);
  EXPECT_EQ(result.block(0)[rank], 3.);
}

TEST(DistributedIDA, MPI_MixedFieldDifferentialComponents) // NOLINT
{
  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = IDAAdapter<FieldVector, GlobalVector>;

  ASSERT_EQ(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);

  Adapter::AdditionalData data;
  Adapter                 adapter(data,
                  MPI_COMM_WORLD,
                  [](const dealii::LinearOperator<GlobalVector> &,
                     const GlobalVector &,
                     GlobalVector &,
                     const double) {});

  const auto field = adapter.add(
    [](auto &builder) {
      using Model = typename std::decay_t<decltype(builder)>::Model;

      const auto rank =
        dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      dealii::IndexSet owned(4);
      owned.add_range(2 * rank, 2 * (rank + 1));
      owned.compress();

      dealii::IndexSet differential(4);
      if (rank == 0)
        differential.add_range(0, 2);
      differential.compress();

      const auto mixed = builder.field("mixed_state", owned, {}, differential);

      auto scaled_identity = [](const FieldVector &reference,
                                const double       scale) {
        typename Model::Operator result;
        result.reinit_range_vector = [reference](FieldVector &vector, bool) {
          vector.reinit(reference);
        };
        result.reinit_domain_vector = result.reinit_range_vector;
        result.vmult                = [scale](FieldVector       &destination,
                               const FieldVector &source) {
          destination = source;
          destination *= scale;
        };
        result.vmult_add = [scale](FieldVector       &destination,
                                   const FieldVector &source) {
          FieldVector contribution;
          contribution.reinit(source);
          contribution = source;
          contribution *= scale;
          destination += contribution;
        };
        result.Tvmult     = result.vmult;
        result.Tvmult_add = result.vmult_add;
        return result;
      };

      typename Model::OperatorFactory state_factory =
        [mixed, scaled_identity](const auto &context) {
          return scaled_identity(context.state(mixed), 1.);
        };
      typename Model::OperatorFactory derivative_factory =
        [mixed, scaled_identity](const auto &context) {
          return scaled_identity(context.state(mixed), 1.);
        };

      builder.term(mixed, "mixed-dae")
        .residual([mixed](const auto &context) {
          const auto               &state     = context.state(mixed);
          const auto               &state_dot = context.derivative(mixed);
          typename Model::Operation result;
          result.reinit_vector = [state](FieldVector &vector, bool) {
            vector.reinit(state);
          };
          result.apply = [state, state_dot](FieldVector &vector) {
            vector = state;
            vector += state_dot;
          };
          result.apply_add = [state, state_dot](FieldVector &vector) {
            FieldVector contribution;
            contribution.reinit(state);
            contribution = state;
            contribution += state_dot;
            vector += contribution;
          };
          return result;
        })
        .state(mixed, state_factory)
        .derivative(mixed, derivative_factory);

      return mixed;
    },
    "mixed");

  const auto mask = adapter.differential_components();
  EXPECT_EQ(mask.size(), 4u);
  const auto rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  EXPECT_EQ(mask.is_element(0), rank == 0);
  EXPECT_EQ(mask.is_element(1), rank == 0);
  EXPECT_FALSE(mask.is_element(2));
  EXPECT_FALSE(mask.is_element(3));

  auto state     = adapter.make_state();
  auto state_dot = adapter.make_state();
  auto residual  = adapter.make_state();
  state          = 1.;
  state_dot      = 2.;
  adapter.solver().residual(0., state, state_dot, residual);
  const auto local_index = 2 * rank;
  EXPECT_EQ(adapter.field(residual, field.fields())[local_index], 3.);

  adapter.solver().setup_jacobian(0., state, state_dot, 3.);
  auto increment = adapter.make_state();
  auto action    = adapter.make_state();
  increment      = 1.;
  adapter.current_jacobian().vmult(action, increment);
  EXPECT_EQ(adapter.field(action, field.fields())[local_index], 4.);

  adapter.solver().setup_jacobian(0., state, state_dot, 4.);
  adapter.current_jacobian().vmult(action, increment);
  EXPECT_EQ(adapter.field(action, field.fields())[local_index], 5.);
}

#else

TEST(DistributedIDA, AdapterIsConditionallyUnavailable) // NOLINT
{
  GTEST_SKIP() << "deal.II was configured without SUNDIALS support.";
}

#endif
