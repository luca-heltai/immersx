// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/physics/metric_flow_x.h>

#include <cmath>

#include "test_paths.h"

#if defined(IMMERSX_WITH_METRIC_FLOW_X) && defined(DEAL_II_WITH_SUNDIALS)

namespace
{
  using Problem      = MetricFlowX::BloodFlowSystem<1, 3>;
  using FieldVector  = MetricFlowX::VectorType;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter      = ImmersX::IDAAdapter<FieldVector, GlobalVector>;

  void
  initialize_problem(Problem &problem)
  {
    problem.initialize_params(ImmersX::TestPaths::parameter_path(
      "gtests/parameters/metric_flow_x.prm"));
    problem.setup();
  }

  Adapter::AdditionalData
  adapter_data()
  {
    Adapter::AdditionalData data;
    data.initial_time                  = 0.;
    data.final_time                    = 1.e-4;
    data.initial_step_size             = 1.e-5;
    data.maximum_order                 = 1;
    data.maximum_non_linear_iterations = 10;
    data.absolute_tolerance            = 1.e-6;
    data.relative_tolerance            = 1.e-5;
    data.ic_type                       = Adapter::AdditionalData::use_y_diff;
    data.reset_type                    = Adapter::AdditionalData::none;
    return data;
  }

  template <typename Fields>
  void
  initialize_adapter_state(Problem      &problem,
                           Adapter      &adapter,
                           const Fields &fields,
                           GlobalVector &state,
                           GlobalVector &state_dot)
  {
    problem.initialize_state(adapter.field(state, fields.fields().state), 0.);
    problem.initialize_state_derivative(adapter.field(state_dot,
                                                      fields.fields().state),
                                        0.);
  }

  void
  add_synthetic_residual(Adapter &adapter, const ImmersX::FieldId state)
  {
    adapter.add(
      [state](auto &builder) {
        auto term = builder.term(state, "synthetic");
        term.residual([state](const auto &context) {
          const auto *reference = &context.state(state);
          dealii::PackagedOperation<FieldVector> result;
          result.reinit_vector = [reference](FieldVector &vector, const bool) {
            vector.reinit(*reference);
          };
          result.apply     = [](FieldVector &vector) { vector = 2.5; };
          result.apply_add = [](FieldVector &vector) {
            FieldVector contribution;
            contribution.reinit(vector);
            contribution = 2.5;
            vector += contribution;
          };
          return result;
        });
        return state;
      },
      "synthetic");
  }
} // namespace

TEST(MetricFlowX, FeatureMacroIsEnabled)
{
  SUCCEED();
}

TEST(MetricFlowX, MPI_RegistersOneMixedStateField) // NOLINT
{
  dealii::ParameterAcceptor::clear();
  Problem problem(MPI_COMM_WORLD);
  initialize_problem(problem);

  Adapter    adapter(adapter_data(), MPI_COMM_WORLD);
  const auto fields =
    adapter.add(ImmersX::metric_flow_x(problem), "blood-flow");

  EXPECT_TRUE(fields.fields().state.is_valid());
  EXPECT_EQ(adapter.differential_components(), problem.differential_dofs());
  EXPECT_EQ(fields.fields().area.source(), fields.fields().state);
  EXPECT_EQ(fields.fields().velocity.source(), fields.fields().state);
  EXPECT_EQ(fields.fields().area.components(),
            problem.component_dofs(Problem::Component::area));
  EXPECT_EQ(fields.fields().velocity.components(),
            problem.component_dofs(Problem::Component::velocity));
  auto state = adapter.make_state();
  EXPECT_EQ(state.n_blocks(), 1u);
}

TEST(MetricFlowX, MPI_ResidualMatchesNativeAndPreservesAdditivity) // NOLINT
{
  dealii::ParameterAcceptor::clear();
  Problem problem(MPI_COMM_WORLD);
  initialize_problem(problem);

  Adapter    adapter(adapter_data(), MPI_COMM_WORLD);
  const auto fields =
    adapter.add(ImmersX::metric_flow_x(problem), "blood-flow");
  add_synthetic_residual(adapter, fields.fields().state);
  auto state     = adapter.make_state();
  auto state_dot = adapter.make_state();
  initialize_adapter_state(problem, adapter, fields, state, state_dot);

  auto native_residual = problem.make_state();
  problem.assemble_residual(0.,
                            adapter.field(state, fields.fields().state),
                            adapter.field(state_dot, fields.fields().state),
                            native_residual);

  GlobalVector residual;
  adapter.solver().reinit_vector(residual);
  adapter.solver().residual(0., state, state_dot, residual);
  auto difference = adapter.field(residual, fields.fields().state);
  difference -= native_residual;
  for (const auto index : problem.locally_owned_dofs())
    EXPECT_NEAR(difference[index], 2.5, 1.e-12);
}

TEST(MetricFlowX, MPI_JacobianActionsMatchNativeMatrices) // NOLINT
{
  dealii::ParameterAcceptor::clear();
  Problem problem(MPI_COMM_WORLD);
  initialize_problem(problem);

  Adapter    adapter(adapter_data(), MPI_COMM_WORLD);
  const auto fields =
    adapter.add(ImmersX::metric_flow_x(problem), "blood-flow");
  auto state     = adapter.make_state();
  auto state_dot = adapter.make_state();
  initialize_adapter_state(problem, adapter, fields, state, state_dot);

  auto direction      = adapter.make_state();
  direction           = 0.125;
  auto expected_state = problem.make_state();
  auto expected_dot   = problem.make_state();
  problem.assemble_state_jacobian(0.,
                                  adapter.field(state, fields.fields().state),
                                  adapter.field(state_dot,
                                                fields.fields().state));
  problem.state_jacobian_matrix().vmult(expected_state,
                                        adapter.field(direction,
                                                      fields.fields().state));
  problem.assemble_derivative_jacobian(
    0.,
    adapter.field(state, fields.fields().state),
    adapter.field(state_dot, fields.fields().state));
  problem.derivative_jacobian_matrix().vmult(
    expected_dot, adapter.field(direction, fields.fields().state));

  auto action = adapter.make_state();
  adapter.solver().setup_jacobian(0., state, state_dot, 0.);
  adapter.current_jacobian().vmult(action, direction);
  auto difference = adapter.field(action, fields.fields().state);
  difference -= expected_state;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-10);

  adapter.solver().setup_jacobian(0., state, state_dot, 1.);
  adapter.current_jacobian().vmult(action, direction);
  difference = adapter.field(action, fields.fields().state);
  difference -= expected_state;
  difference -= expected_dot;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-10);
}

TEST(MetricFlowX, MPI_IDAVerticalSmoke) // NOLINT
{
  dealii::ParameterAcceptor::clear();
  Problem problem(MPI_COMM_WORLD);
  initialize_problem(problem);

  Adapter    adapter(adapter_data(), MPI_COMM_WORLD);
  const auto fields =
    adapter.add(ImmersX::metric_flow_x(problem), "blood-flow");
  auto state     = adapter.make_state();
  auto state_dot = adapter.make_state();
  initialize_adapter_state(problem, adapter, fields, state, state_dot);
  adapter.set_compute_consistent_initial_conditions(
    [&problem, &adapter, fields](const double  time,
                                 GlobalVector &state,
                                 GlobalVector &state_dot) {
      problem.initialize_state_derivative(adapter.field(state_dot,
                                                        fields.fields().state),
                                          time);
      (void)state;
    });

  EXPECT_GT(adapter.solve(state, state_dot), 0u);
}

#else

TEST(MetricFlowX, FeatureMacroIsEnabled)
{
  GTEST_SKIP() << "MetricFlowX or deal.II SUNDIALS support is unavailable.";
}

#endif
