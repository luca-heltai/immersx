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
#include <immersx/core/sundials_kinsol_adapter.h>

#include <cmath>
#include <type_traits>

#ifdef DEAL_II_WITH_SUNDIALS

namespace
{
  using FieldVector  = ImmersX::ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter      = ImmersX::KINSOLAdapter<FieldVector, GlobalVector>;

  auto
  add_quadratic_field(Adapter &adapter)
  {
    return adapter.add(
      [](auto &builder) {
        using Model = typename std::decay_t<decltype(builder)>::Model;

        const auto rank =
          dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        const auto n_ranks =
          dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
        dealii::IndexSet owned(n_ranks == 1 ? 2 : n_ranks);
        if (n_ranks == 1)
          owned.add_range(0, 2);
        else
          owned.add_index(rank);
        owned.compress();

        const auto field = builder.algebraic_field("value", owned);

        typename Model::ResidualFactory residual = [field](
                                                     const auto &context) {
          const auto                state = context.state(field);
          typename Model::Operation result;
          result.reinit_vector = [state](FieldVector &vector, const bool omit) {
            vector.reinit(state, omit);
          };
          result.apply = [state](FieldVector &destination) {
            destination = 0.;
            for (const auto index : state.locally_owned_elements())
              destination(index) = state(index) * state(index) - 2.;
            destination.compress(dealii::VectorOperation::insert);
          };
          result.apply_add = [state](FieldVector &destination) {
            FieldVector contribution;
            contribution.reinit(state);
            for (const auto index : state.locally_owned_elements())
              contribution(index) = state(index) * state(index) - 2.;
            contribution.compress(dealii::VectorOperation::insert);
            destination += contribution;
          };
          return result;
        };

        typename Model::OperatorFactory jacobian =
          [field](const auto &context) {
            const auto               state = context.state(field);
            typename Model::Operator result;
            result.reinit_range_vector = [state](FieldVector &vector,
                                                 const bool   omit) {
              vector.reinit(state, omit);
            };
            result.reinit_domain_vector = result.reinit_range_vector;
            result.vmult = [state](FieldVector       &destination,
                                   const FieldVector &source) {
              destination = source;
              for (const auto index : state.locally_owned_elements())
                destination(index) *= 2. * state(index);
              destination.compress(dealii::VectorOperation::insert);
            };
            result.vmult_add = [state](FieldVector       &destination,
                                       const FieldVector &source) {
              FieldVector contribution;
              contribution.reinit(source);
              contribution = source;
              for (const auto index : state.locally_owned_elements())
                contribution(index) *= 2. * state(index);
              contribution.compress(dealii::VectorOperation::insert);
              destination += contribution;
            };
            result.Tvmult     = result.vmult;
            result.Tvmult_add = result.vmult_add;
            return result;
          };

        builder.term(field, "quadratic")
          .residual(residual)
          .state(field, jacobian);
        return field;
      },
      "quadratic");
  }
} // namespace

TEST(KINSOLAdapter, MPI_NonlinearScalarSolve) // NOLINT
{
  ImmersX::KINSOLAdapterParameters parameters;
  parameters.strategy                      = "newton";
  parameters.maximum_non_linear_iterations = 20;
  parameters.function_tolerance            = 1.e-12;
  parameters.step_tolerance                = 1.e-12;

  Adapter    adapter(parameters, MPI_COMM_WORLD);
  const auto field = add_quadratic_field(adapter);

  auto state                           = adapter.make_state();
  adapter.field(state, field.fields()) = 1.;

  auto candidate                        = adapter.make_state();
  auto residual                         = adapter.make_state();
  auto source                           = adapter.make_state();
  auto action                           = adapter.make_state();
  adapter.field(source, field.fields()) = 1.;

  adapter.field(candidate, field.fields()) = 1.;
  adapter.solver().setup_jacobian(candidate, residual);
  adapter.current_jacobian().vmult(action, source);
  for (const auto index :
       adapter.field(action, field.fields()).locally_owned_elements())
    EXPECT_DOUBLE_EQ(adapter.field(action, field.fields())(index), 2.);

  adapter.field(candidate, field.fields()) = 3.;
  adapter.solver().setup_jacobian(candidate, residual);
  adapter.current_jacobian().vmult(action, source);
  for (const auto index :
       adapter.field(action, field.fields()).locally_owned_elements())
    EXPECT_DOUBLE_EQ(adapter.field(action, field.fields())(index), 6.);

  EXPECT_GT(adapter.solve(state), 0u);
  for (const auto index :
       adapter.field(state, field.fields()).locally_owned_elements())
    EXPECT_NEAR(adapter.field(state, field.fields())(index),
                std::sqrt(2.),
                1.e-10);
}

#endif
