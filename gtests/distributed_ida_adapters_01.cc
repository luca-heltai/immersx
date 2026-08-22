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
      const auto value = builder.field("value", TimeRole::algebraic, owned);
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

  auto state                  = adapter.make_state();
  auto state_dot              = adapter.make_state();
  adapter.field(state, field) = 3.;
  adapter.solver().setup_jacobian(0., state, state_dot, 0.);

  auto source = adapter.make_state();
  auto result = adapter.make_state();
  source      = 1.;
  adapter.current_jacobian().vmult(result, source);
  const auto rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  EXPECT_EQ(adapter.field(state, field)[rank], 3.);
  EXPECT_EQ(result.block(0)[rank], 3.);
}

#else

TEST(DistributedIDA, AdapterIsConditionallyUnavailable) // NOLINT
{
  GTEST_SKIP() << "deal.II was configured without SUNDIALS support.";
}

#endif
