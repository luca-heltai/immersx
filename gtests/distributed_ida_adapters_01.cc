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
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

using namespace ImmersX;

#ifdef DEAL_II_WITH_SUNDIALS

namespace
{
  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = IDAAdapter<FieldVector, GlobalVector>;

  template <typename AdapterType>
  auto
  add_identity_field(AdapterType       &adapter,
                     const unsigned int size,
                     bool              *initial_output = nullptr,
                     bool              *early_residual = nullptr)
  {
    return adapter.add(
      [=](auto &builder) {
        using Model = typename std::decay_t<decltype(builder)>::Model;

        dealii::IndexSet owned(size);
        owned.add_range(0, size);
        owned.compress();
        const auto field = builder.differential_field("value", owned);

        typename Model::OperatorFactory identity_factory =
          [field](const auto &context) {
            const auto              &reference = context.state(field);
            typename Model::Operator result;
            result.reinit_range_vector = [reference](FieldVector &vector,
                                                     bool) {
              vector.reinit(reference);
            };
            result.reinit_domain_vector = result.reinit_range_vector;
            result.vmult                = [](FieldVector       &destination,
                              const FieldVector &source) {
              destination = source;
            };
            result.vmult_add = [](FieldVector       &destination,
                                  const FieldVector &source) {
              destination += source;
            };
            result.Tvmult     = result.vmult;
            result.Tvmult_add = result.vmult_add;
            return result;
          };

        builder.term(field, "identity")
          .residual(
            [field, initial_output, early_residual](const auto &context) {
              const auto &state     = context.state(field);
              const auto &state_dot = context.derivative(field);
              if (early_residual != nullptr && initial_output != nullptr &&
                  *initial_output == false)
                *early_residual = true;

              typename Model::Operation result;
              result.reinit_vector = [state](FieldVector &vector, bool) {
                vector.reinit(state);
              };
              result.apply = [state, state_dot](FieldVector &vector) {
                vector = state_dot;
                vector += state;
              };
              result.apply_add = [state, state_dot](FieldVector &vector) {
                FieldVector contribution;
                contribution.reinit(state);
                contribution = state_dot;
                contribution += state;
                vector += contribution;
              };
              return result;
            })
          .state(field, identity_factory)
          .derivative(field, identity_factory);
        return field;
      },
      "identity");
  }
} // namespace

TEST(DistributedIDA, ApplicationConsistentInitialConditions) // NOLINT
{
  TimeIntervalParameters time_parameters;
  FixedStepParameters    fixed_step_parameters;
  IDAParameters          ida_parameters;
  time_parameters.initial_time                     = 0.;
  time_parameters.final_time                       = 0.01;
  fixed_step_parameters.time_step                  = 0.01;
  time_parameters.output_time_interval             = 0.01;
  ida_parameters.initial_step_size                 = 0.01;
  ida_parameters.maximum_order                     = 1;
  ida_parameters.minimum_step_size                 = 1.e-6;
  ida_parameters.maximum_non_linear_iterations     = 10;
  ida_parameters.absolute_tolerance                = 1.e-6;
  ida_parameters.relative_tolerance                = 1.e-5;
  ida_parameters.maximum_non_linear_iterations_ic  = 5;
  ida_parameters.ls_norm_factor                    = 0.;
  ida_parameters.correction_type_at_initial_time   = "use_y_diff";
  ida_parameters.correction_type_after_restart     = "none";
  ida_parameters.ignore_algebraic_terms_for_errors = false;

  Adapter      adapter(time_parameters, ida_parameters, MPI_COMM_SELF);
  bool         initial_output     = false;
  bool         early_residual     = false;
  unsigned int callback_count     = 0;
  double       initial_state_norm = 0.;
  double       initial_dot_norm   = 0.;
  const auto   field =
    add_identity_field(adapter, 2, &initial_output, &early_residual);
  auto state = adapter.make_state();
  auto dot   = adapter.make_state();

  adapter.set_compute_consistent_initial_conditions(
    [&](const double time, GlobalVector &y, GlobalVector &y_dot) {
      EXPECT_DOUBLE_EQ(time, 0.);
      ++callback_count;
      adapter.field(y, field.fields())     = 2.;
      adapter.field(y_dot, field.fields()) = -2.;
    });
  adapter.set_output_step([&](const double,
                              const GlobalVector &y,
                              const GlobalVector &y_dot,
                              const unsigned int  step) {
    if (step == 0)
      {
        initial_output     = true;
        initial_state_norm = adapter.field(y, field.fields()).l2_norm();
        initial_dot_norm   = adapter.field(y_dot, field.fields()).l2_norm();
      }
  });

  EXPECT_GT(adapter.solve(state, dot), 0u);
  EXPECT_EQ(callback_count, 1u);
  EXPECT_TRUE(initial_output);
  EXPECT_FALSE(early_residual);
  EXPECT_DOUBLE_EQ(initial_state_norm, std::sqrt(8.));
  EXPECT_DOUBLE_EQ(initial_dot_norm, std::sqrt(8.));
}

TEST(DistributedIDA, CompleteParameterControls)
{
  TimeIntervalParameters   time;
  IDAParameters            parameters;
  dealii::ParameterHandler prm;
  parameters.declare_parameters(prm);
  prm.parse_input_from_string(R"(
subsection Running parameters
  set Maximum step size = 0.002
  set Maximum number of steps = 123
  set Maximum error test failures = 7
  set Maximum nonlinear convergence failures = 8
end
subsection Nonlinear solver
  set Convergence coefficient = 0.2
end
subsection Initial condition correction parameters
  set Maximum number of steps = 9
  set Maximum number of Jacobians = 6
  set Maximum number of backtracks = 12
  set Line search enabled = false
  set Correction type at initial time = use_y_diff
end
subsection Linear solver
  set Linear tolerance factor = 0.04
  set Difference quotient increment factor = 0.9
  set Linear solution scaling = false
end
)");
  const auto data = parameters.additional_data<GlobalVector>(time);
  EXPECT_DOUBLE_EQ(data.maximum_step_size, 0.002);
  EXPECT_EQ(data.maximum_number_of_steps, 123);
  EXPECT_EQ(data.maximum_error_test_failures, 7u);
  EXPECT_EQ(data.maximum_convergence_failures, 8u);
  EXPECT_DOUBLE_EQ(data.nonlinear_convergence_coefficient, 0.2);
  EXPECT_EQ(data.maximum_number_of_steps_ic, 9u);
  EXPECT_EQ(data.maximum_number_of_jacobians_ic, 6u);
  EXPECT_EQ(data.maximum_number_of_backtracks_ic, 12u);
  EXPECT_FALSE(data.line_search_enabled_ic);
  EXPECT_EQ(data.ic_type, Adapter::AdditionalData::use_y_diff);
  EXPECT_DOUBLE_EQ(data.linear_tolerance_factor, 0.04);
  EXPECT_DOUBLE_EQ(data.increment_factor, 0.9);
  EXPECT_FALSE(data.linear_solution_scaling);
  EXPECT_EQ(data.maximum_order, 1u);
  EXPECT_DOUBLE_EQ(data.absolute_tolerance, 1.e-4);
}

TEST(DistributedIDA, NativeSetupFailureThrows)
{
  TimeIntervalParameters time;
  IDAParameters          parameters;
  parameters.maximum_order = 6;
  Adapter adapter(time, parameters, MPI_COMM_SELF);
  add_identity_field(adapter, 1);
  auto y = adapter.make_state(), yp = adapter.make_state();
  y  = 1.;
  yp = -1.;
  EXPECT_THROW(adapter.solve(y, yp), dealii::ExceptionBase);
}

TEST(DistributedIDA, NativeWorkLimitThrows)
{
  TimeIntervalParameters time;
  IDAParameters          parameters;
  time.initial_time                  = 0.25;
  time.final_time                    = 0.35;
  time.output_time_interval          = 0.1;
  parameters.initial_step_size       = 1.e-4;
  parameters.maximum_step_size       = 1.e-4;
  parameters.maximum_number_of_steps = 2;
  Adapter adapter(time, parameters, MPI_COMM_SELF);
  add_identity_field(adapter, 1);
  auto y = adapter.make_state(), yp = adapter.make_state();
  y  = 1.;
  yp = -1.;
  adapter.set_output_step(
    [&](double t, const auto &, const auto &, unsigned int step) {
      if (step == 0)
        {
          EXPECT_DOUBLE_EQ(t, time.initial_time);
        }
    });
  EXPECT_THROW(adapter.solve(y, yp), dealii::ExceptionBase);
}

TEST(DistributedIDA, BackportedControlsAndConstraints) // NOLINT
{
  TimeIntervalParameters time;
  IDAParameters          parameters;
  time.final_time                              = 0.01;
  time.output_time_interval                    = 0.01;
  parameters.ignore_algebraic_terms_for_errors = false;
  Adapter    adapter(time, parameters, MPI_COMM_SELF);
  const auto field        = add_identity_field(adapter, 1);
  auto       state        = adapter.make_state();
  auto       dot          = adapter.make_state();
  auto       constraints  = adapter.make_state();
  state                   = 2.;
  dot                     = -2.;
  constraints             = 2.;
  auto data               = adapter.additional_data();
  data.maximum_step_size  = 0.001;
  data.initial_step_size  = 0.001;
  data.absolute_tolerance = 1.e-8;
  data.relative_tolerance = 1.e-8;
  adapter.set_additional_data(data);
  unsigned int setup_count      = 0;
  adapter.solver().custom_setup = [&](void *memory) {
    ++setup_count;
    EXPECT_EQ(memory, adapter.solver().get_ida_memory());
  };
  adapter.solver().get_constraint_vector = [&]() -> GlobalVector & {
    return constraints;
  };
  EXPECT_THROW(adapter.set_additional_data(data), dealii::ExceptionBase);
  EXPECT_GE(adapter.solve(state, dot), 10u);
  EXPECT_EQ(setup_count, 1u);
  EXPECT_NEAR(adapter.field(state, field.fields())[0],
              2. * std::exp(-time.final_time),
              1.e-6);
}

TEST(DistributedIDA, IDACalcICFallbackRemainsAvailable) // NOLINT
{
  TimeIntervalParameters time_parameters;
  FixedStepParameters    fixed_step_parameters;
  IDAParameters          ida_parameters;
  time_parameters.initial_time                     = 0.;
  time_parameters.final_time                       = 0.01;
  fixed_step_parameters.time_step                  = 0.01;
  time_parameters.output_time_interval             = 0.01;
  ida_parameters.initial_step_size                 = 0.01;
  ida_parameters.maximum_order                     = 1;
  ida_parameters.minimum_step_size                 = 1.e-6;
  ida_parameters.maximum_non_linear_iterations     = 10;
  ida_parameters.absolute_tolerance                = 1.e-6;
  ida_parameters.relative_tolerance                = 1.e-5;
  ida_parameters.maximum_non_linear_iterations_ic  = 5;
  ida_parameters.ls_norm_factor                    = 0.;
  ida_parameters.correction_type_at_initial_time   = "use_y_diff";
  ida_parameters.correction_type_after_restart     = "none";
  ida_parameters.ignore_algebraic_terms_for_errors = false;

  Adapter    adapter(time_parameters, ida_parameters, MPI_COMM_SELF);
  const auto field                     = add_identity_field(adapter, 2);
  auto       state                     = adapter.make_state();
  auto       dot                       = adapter.make_state();
  adapter.field(state, field.fields()) = 1.;
  double initial_dot_norm              = 0.;
  adapter.set_output_step([&](const double,
                              const GlobalVector &,
                              const GlobalVector &y_dot,
                              const unsigned int  step) {
    if (step == 0)
      initial_dot_norm = adapter.field(y_dot, field.fields()).l2_norm();
  });

  EXPECT_GT(adapter.solve(state, dot), 0u);
  EXPECT_NEAR(initial_dot_norm, std::sqrt(2.), 1.e-8);
}

TEST(DistributedIDA,
     ApplicationConsistentInitialConditionFailurePropagates) // NOLINT
{
  TimeIntervalParameters time_parameters;
  FixedStepParameters    fixed_step_parameters;
  IDAParameters          ida_parameters;
  time_parameters.final_time                      = 0.01;
  fixed_step_parameters.time_step                 = 0.01;
  time_parameters.output_time_interval            = 0.01;
  ida_parameters.initial_step_size                = 1.e-2;
  ida_parameters.maximum_order                    = 5;
  ida_parameters.minimum_step_size                = 1.e-6;
  ida_parameters.maximum_non_linear_iterations    = 10;
  ida_parameters.absolute_tolerance               = 1.e-6;
  ida_parameters.relative_tolerance               = 1.e-5;
  ida_parameters.maximum_non_linear_iterations_ic = 5;
  ida_parameters.ls_norm_factor                   = 0.;
  ida_parameters.correction_type_at_initial_time  = "use_y_diff";

  Adapter adapter(time_parameters, ida_parameters, MPI_COMM_SELF);
  auto    field = add_identity_field(adapter, 2);
  auto    state = adapter.make_state();
  auto    dot   = adapter.make_state();
  (void)field;
  adapter.set_compute_consistent_initial_conditions(
    [](const double, GlobalVector &, GlobalVector &) {
      throw std::runtime_error("consistent initial condition failure");
    });

  EXPECT_THROW(adapter.solve(state, dot), std::runtime_error);
}

TEST(DistributedIDA, NoneLeavesInitialConditionsUntouched) // NOLINT
{
  TimeIntervalParameters time_parameters;
  FixedStepParameters    fixed_step_parameters;
  IDAParameters          ida_parameters;
  time_parameters.initial_time                     = 0.;
  time_parameters.final_time                       = 0.01;
  fixed_step_parameters.time_step                  = 0.01;
  time_parameters.output_time_interval             = 0.01;
  ida_parameters.initial_step_size                 = 0.01;
  ida_parameters.maximum_order                     = 1;
  ida_parameters.minimum_step_size                 = 1.e-6;
  ida_parameters.maximum_non_linear_iterations     = 10;
  ida_parameters.absolute_tolerance                = 1.e-6;
  ida_parameters.relative_tolerance                = 1.e-5;
  ida_parameters.maximum_non_linear_iterations_ic  = 5;
  ida_parameters.ls_norm_factor                    = 0.;
  ida_parameters.correction_type_at_initial_time   = "none";
  ida_parameters.correction_type_after_restart     = "none";
  ida_parameters.ignore_algebraic_terms_for_errors = false;

  Adapter    adapter(time_parameters, ida_parameters, MPI_COMM_SELF);
  const auto field                     = add_identity_field(adapter, 2);
  auto       state                     = adapter.make_state();
  auto       dot                       = adapter.make_state();
  adapter.field(state, field.fields()) = 1.;
  bool   initial_output                = false;
  double initial_dot_norm              = 0.;
  adapter.set_output_step([&](const double,
                              const GlobalVector &,
                              const GlobalVector &y_dot,
                              const unsigned int  step) {
    if (step == 0)
      {
        initial_output   = true;
        initial_dot_norm = adapter.field(y_dot, field.fields()).l2_norm();
      }
  });

  EXPECT_GT(adapter.solve(state, dot), 0u);
  EXPECT_TRUE(initial_output);
  EXPECT_DOUBLE_EQ(initial_dot_norm, 0.);
}

TEST(DistributedIDA, RestartCallbackPrecedesResetAndCanResizeState) // NOLINT
{
  TimeIntervalParameters time_parameters;
  FixedStepParameters    fixed_step_parameters;
  IDAParameters          ida_parameters;
  time_parameters.initial_time                     = 0.;
  time_parameters.final_time                       = 0.02;
  fixed_step_parameters.time_step                  = 0.01;
  time_parameters.output_time_interval             = 0.01;
  ida_parameters.initial_step_size                 = 0.01;
  ida_parameters.maximum_order                     = 1;
  ida_parameters.minimum_step_size                 = 1.e-6;
  ida_parameters.maximum_non_linear_iterations     = 10;
  ida_parameters.absolute_tolerance                = 1.e-6;
  ida_parameters.relative_tolerance                = 1.e-5;
  ida_parameters.maximum_non_linear_iterations_ic  = 5;
  ida_parameters.ls_norm_factor                    = 0.;
  ida_parameters.correction_type_at_initial_time   = "use_y_diff";
  ida_parameters.correction_type_after_restart     = "use_y_diff";
  ida_parameters.ignore_algebraic_terms_for_errors = false;

  Adapter      adapter(time_parameters, ida_parameters, MPI_COMM_SELF);
  const auto   field            = add_identity_field(adapter, 2);
  auto         state            = adapter.make_state();
  auto         dot              = adapter.make_state();
  unsigned int restart_count    = 0;
  unsigned int consistent_count = 0;
  unsigned int tolerance_builds = 0;
  adapter.set_local_tolerances(field.fields(), [&](FieldVector &values) {
    ++tolerance_builds;
    EXPECT_EQ(values.size(), tolerance_builds == 1 ? 2u : 3u);
    values = 1.e-6;
  });
  adapter.set_compute_consistent_initial_conditions(
    [&](const double, GlobalVector &y, GlobalVector &y_dot) {
      ++consistent_count;
      adapter.field(y, field.fields())     = 3.;
      adapter.field(y_dot, field.fields()) = -3.;
    });
  adapter.set_solver_should_restart(
    [&](const double, GlobalVector &y, GlobalVector &y_dot) {
      if (restart_count++ != 0)
        return false;
      dealii::IndexSet owned(3);
      owned.add_range(0, 3);
      owned.compress();
      y.reinit(std::vector<dealii::IndexSet>{owned}, MPI_COMM_SELF);
      y_dot.reinit(std::vector<dealii::IndexSet>{owned}, MPI_COMM_SELF);
      return true;
    });

  EXPECT_GT(adapter.solve(state, dot), 0u);
  EXPECT_GE(restart_count, 2u);
  EXPECT_EQ(consistent_count, 2u);
  EXPECT_EQ(tolerance_builds, 2u);
  EXPECT_EQ(state.block(0).size(), 3u);
  EXPECT_EQ(dot.block(0).size(), 3u);
}

TEST(DistributedIDA, MPI_StateDependentJacobianOwnsEvaluationState) // NOLINT
{
  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = IDAAdapter<FieldVector, GlobalVector>;

  TimeIntervalParameters time_parameters;
  FixedStepParameters    fixed_step_parameters;
  IDAParameters          ida_parameters;
  time_parameters.initial_time = 0.;
  time_parameters.final_time   = 0.01;
  Adapter adapter(time_parameters,
                  ida_parameters,
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

  TimeIntervalParameters time_parameters;
  FixedStepParameters    fixed_step_parameters;
  IDAParameters          ida_parameters;
  Adapter                adapter(time_parameters,
                  ida_parameters,
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

TEST(DistributedIDA, BOTH_SchurRetainsDisplacementVelocityPath)
{
  using namespace dealii;
  using namespace ImmersX;
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  IndexSet   owned(size);
  owned.add_index(rank);
  DynamicSparsityPattern sparsity(size, size, owned);
  sparsity.add(rank, rank);
  auto mass = std::make_shared<ImmersXLA::MPI::SparseMatrix>();
  mass->reinit(owned, owned, sparsity, MPI_COMM_WORLD);
  mass->set(rank, rank, 1.);
  mass->compress(VectorOperation::insert);
  Adapter::Composition composition(MPI_COMM_WORLD);
  const auto           fields = composition.add(
    [owned, mass](auto &builder) {
      const auto u = builder.field("u", owned, owned);
      const auto v = builder.field("v", owned, owned);
      const auto lambda = builder.algebraic_field("lambda", owned, owned);
      auto       identity = builder.matrix_operator(*mass);
      builder.term(u, "kinematic")
        .derivative(u, identity)
        .state(v, -1. * identity);
      builder.term(v, "momentum")
        .derivative(v, identity)
        .state(lambda, identity);
      builder.term(lambda, "position").state(u, identity);
      const auto inverse = [](const auto &matrix, const auto &prototype) {
        return make_local_preconditioner<FieldVector,
                                         ImmersXLA::MPI::SparseMatrix,
                                         ImmersXLA::MPI::PreconditionJacobi>(
          matrix, prototype);
      };
      builder.preconditioner(u, inverse);
      builder.preconditioner(v, inverse);
      builder.saddle_point(lambda, {u, v}, identity);
      return std::array<FieldId, 3>{u, v, lambda};
    },
    "solid");
  auto state        = composition.make_state();
  auto dot          = composition.make_state();
  auto input        = composition.field(state, fields[2]);
  input             = 1.;
  auto       action = input;
  const auto schur  = composition.schur_operator(fields[2], state, &dot, 2.);
  schur.vmult(action, input);
  EXPECT_NEAR(action[rank], 0., 1.e-14);
  const auto coupled =
    composition.schur_operator(fields[2], state, &dot, 2., true);
  coupled.vmult(action, input);
  // For 2u-v=f_u, 2v+lambda=f_v, u=0, elimination gives S=1/4.
  EXPECT_NEAR(action[rank], 0.25, 1.e-14);
  coupled.Tvmult(action, input);
  EXPECT_NEAR(action[rank], 0.25, 1.e-14);

  auto rhs                          = composition.make_state();
  composition.field(rhs, fields[0]) = 1.;
  composition.field(rhs, fields[1]) = 2.;
  composition.field(rhs, fields[2]) = 3.;
  auto       solution               = composition.make_state();
  const auto inverse                = composition.schur_preconditioner(
    fields[2], state, &dot, 2., 100, 1.e-12, true);
  inverse.vmult(solution, rhs);
  EXPECT_NEAR(composition.field(solution, fields[0])[rank], 3., 1.e-12);
  EXPECT_NEAR(composition.field(solution, fields[1])[rank], 5., 1.e-12);
  EXPECT_NEAR(composition.field(solution, fields[2])[rank], -8., 1.e-12);
  inverse.Tvmult(solution, rhs);
  EXPECT_NEAR(composition.field(solution, fields[0])[rank], 4., 1.e-12);
  EXPECT_NEAR(composition.field(solution, fields[1])[rank], 3., 1.e-12);
  EXPECT_NEAR(composition.field(solution, fields[2])[rank], -7., 1.e-12);
}

#  ifdef DEAL_II_WITH_UMFPACK
TEST(DistributedIDA, SerialUMFPACKSolvesCurrentDAEJacobian)
{
  using namespace dealii;
  TimeIntervalParameters time_parameters;
  IDAParameters          ida_parameters;
  Adapter adapter(time_parameters, ida_parameters, MPI_COMM_SELF);
  adapter.set_serial_direct_solver();
  IndexSet owned(1);
  owned.add_index(0);
  DynamicSparsityPattern sparsity(1, 1);
  sparsity.add(0, 0);
  auto mass = std::make_shared<ImmersXLA::MPI::SparseMatrix>();
  mass->reinit(owned, owned, sparsity, MPI_COMM_SELF);
  mass->set(0, 0, 1.);
  mass->compress(VectorOperation::insert);
  const auto fields =
    adapter
      .add([owned, mass](auto &builder) {
        using Model         = typename std::decay_t<decltype(builder)>::Model;
        const auto u        = builder.field("u", owned, owned);
        const auto v        = builder.field("v", owned, owned);
        const auto lambda   = builder.algebraic_field("lambda", owned, owned);
        auto       identity = builder.matrix_operator(*mass);
        typename Model::MatrixOperatorFactory time_coupling =
          [identity](const auto &context) {
            return -context.time() * identity;
          };
        builder.term(u, "kinematic")
          .derivative(u, identity)
          .state(v, time_coupling);
        builder.term(v, "momentum")
          .derivative(v, identity)
          .state(lambda, identity);
        builder.term(lambda, "position").state(u, identity);
        return std::array<FieldId, 3>{u, v, lambda};
      })
      .fields();
  auto state                    = adapter.make_state();
  auto dot                      = adapter.make_state();
  auto rhs                      = adapter.make_state();
  auto solution                 = adapter.make_state();
  auto action                   = adapter.make_state();
  adapter.field(rhs, fields[0]) = 1.;
  adapter.field(rhs, fields[1]) = 2.;
  adapter.field(rhs, fields[2]) = 3.;
  adapter.solver().setup_jacobian(1., state, dot, 2.);
  EXPECT_FALSE(adapter.has_current_preconditioner());
  adapter.solver().solve_with_jacobian(rhs, solution, 1.e-12);
  EXPECT_NEAR(adapter.field(solution, fields[0])[0], 3., 1.e-12);
  EXPECT_NEAR(adapter.field(solution, fields[1])[0], 5., 1.e-12);
  EXPECT_NEAR(adapter.field(solution, fields[2])[0], -8., 1.e-12);
  // A second right-hand side reuses this factorization.
  rhs *= 2.;
  adapter.solver().solve_with_jacobian(rhs, solution, 1.e-12);
  adapter.current_jacobian().vmult(action, solution);
  action -= rhs;
  EXPECT_LT(action.l2_norm(), 1.e-12);
  rhs *= .5;
  // Both physical time and IDA's derivative scaling change the matrix.
  adapter.solver().setup_jacobian(2., state, dot, 3.);
  adapter.solver().solve_with_jacobian(rhs, solution, 1.e-12);
  EXPECT_NEAR(adapter.field(solution, fields[0])[0], 3., 1.e-12);
  EXPECT_NEAR(adapter.field(solution, fields[1])[0], 4., 1.e-12);
  EXPECT_NEAR(adapter.field(solution, fields[2])[0], -10., 1.e-12);
  adapter.current_jacobian().vmult(action, solution);
  action -= rhs;
  EXPECT_LT(action.l2_norm(), 1.e-12);
  EXPECT_THROW(adapter.set_serial_direct_solver(false), ExceptionBase);
}
#  endif

#  ifdef DEAL_II_WITH_MUMPS
namespace
{
  void
  check_mumps_jacobian(const bool empty_last_rank)
  {
    using namespace dealii;
    const auto         rank  = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto         ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const unsigned int size  = 2 * (empty_last_rank ? ranks - 1 : ranks);
    IndexSet           owned(size);
    if (!empty_last_rank || rank + 1 < ranks)
      owned.add_range(2 * rank, 2 * rank + 2);
    owned.compress();
    DynamicSparsityPattern sparsity(size, size, owned);
    for (const auto i : owned)
      {
        sparsity.add(i, i);
        sparsity.add(i, (i + 1) % size);
      }
    auto mass = std::make_shared<ImmersXLA::MPI::SparseMatrix>();
    mass->reinit(owned, owned, sparsity, MPI_COMM_WORLD);
    for (const auto i : owned)
      {
        mass->set(i, i, 2.);
        mass->set(i, (i + 1) % size, .25);
      }
    mass->compress(VectorOperation::insert);
    TimeIntervalParameters time_parameters;
    IDAParameters          ida_parameters;
    Adapter adapter(time_parameters, ida_parameters, MPI_COMM_WORLD);
    adapter.set_linear_solver(Adapter::LinearSolver::mumps);
    const auto fields =
      adapter
        .add([owned, mass](auto &builder) {
          using Model         = typename std::decay_t<decltype(builder)>::Model;
          const auto u        = builder.field("u", owned, owned);
          const auto v        = builder.field("v", owned, owned);
          const auto lambda   = builder.algebraic_field("lambda", owned, owned);
          auto       identity = builder.matrix_operator(*mass);
          typename Model::MatrixOperatorFactory time_coupling =
            [identity](const auto &context) {
              return -context.time() * identity;
            };
          builder.term(u, "kinematic")
            .derivative(u, identity)
            .state(v, time_coupling);
          builder.term(v, "momentum")
            .derivative(v, identity)
            .state(lambda, identity);
          builder.term(lambda, "position").state(u, identity);
          return std::array<FieldId, 3>{{u, v, lambda}};
        })
        .fields();
    auto       state    = adapter.make_state();
    auto       dot      = adapter.make_state();
    auto       rhs      = adapter.make_state();
    auto       solution = adapter.make_state();
    auto       action   = adapter.make_state();
    const auto exact    = [](const unsigned int field, const unsigned int i) {
      if (field == 0)
        return 2. + i;
      else if (field == 1)
        return -1. + 2. * i;
      else if (field == 2)
        return 4. - i;
      return 0.;
    };
    for (unsigned int setup = 0; setup < 2; ++setup)
      {
        const double time = 1. + setup, alpha = 2. + setup;
        for (const auto i : owned)
          {
            const auto m = [&](const unsigned int field) {
              return 2. * exact(field, i) + .25 * exact(field, (i + 1) % size);
            };
            adapter.field(rhs, fields[0])[i] = alpha * m(0) - time * m(1);
            adapter.field(rhs, fields[1])[i] = alpha * m(1) + m(2);
            adapter.field(rhs, fields[2])[i] = m(0);
          }
        rhs.compress(VectorOperation::insert);
        adapter.solver().setup_jacobian(time, state, dot, alpha);
        EXPECT_FALSE(adapter.has_current_preconditioner());
        for (const double scale : {1., -.5})
          {
            auto scaled_rhs = rhs;
            scaled_rhs *= scale;
            adapter.solver().solve_with_jacobian(scaled_rhs, solution, 1.e-12);
            for (unsigned int field = 0; field < fields.size(); ++field)
              for (const auto i : owned)
                EXPECT_NEAR(adapter.field(solution, fields[field])[i],
                            scale * exact(field, i),
                            1.e-11);
            adapter.current_jacobian().vmult(action, solution);
            action -= scaled_rhs;
            EXPECT_LT(action.l2_norm(), 1.e-10);
          }
      }
    EXPECT_THROW(adapter.set_linear_solver(Adapter::LinearSolver::iterative),
                 ExceptionBase);
  }
} // namespace

TEST(DistributedIDA, BOTH_MUMPSSolvesCurrentDAEJacobian)
{
  check_mumps_jacobian(false);
}

TEST(DistributedIDA, MPI_MUMPSHandlesEmptyExecutionRank)
{
  ASSERT_GT(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);
  check_mumps_jacobian(true);
}
#  else
TEST(DistributedIDA, MUMPSRejectsUnavailableBackend)
{
  TimeIntervalParameters time_parameters;
  IDAParameters          ida_parameters;
  Adapter adapter(time_parameters, ida_parameters, MPI_COMM_SELF);
  EXPECT_THROW(adapter.set_linear_solver(Adapter::LinearSolver::mumps),
               dealii::ExceptionBase);
}
#  endif

TEST(DistributedIDA, MPI_UMFPACKRejectsMultipleRanks)
{
  TimeIntervalParameters time_parameters;
  IDAParameters          ida_parameters;
  ASSERT_GT(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);
  Adapter adapter(time_parameters, ida_parameters, MPI_COMM_WORLD);
  EXPECT_THROW(adapter.set_serial_direct_solver(), dealii::ExceptionBase);
}

TEST(DistributedIDA, BOTH_PreservesFatalCallbackException)
{
  unsigned int calls    = 0;
  const auto   callback = [&calls] {
    ++calls;
    throw std::runtime_error("original callback failure");
  };
  std::exception_ptr pending;
  using Compatibility::SUNDIALS::Utilities::call_and_possibly_capture_exception;
  EXPECT_EQ(call_and_possibly_capture_exception(callback, pending), -1);
  EXPECT_EQ(call_and_possibly_capture_exception(callback, pending), -1);
  EXPECT_EQ(calls, 1u);
  ASSERT_TRUE(pending);
  try
    {
      std::rethrow_exception(pending);
    }
  catch (const std::runtime_error &exception)
    {
      EXPECT_STREQ(exception.what(), "original callback failure");
    }
}

#else

TEST(DistributedIDA, AdapterIsConditionallyUnavailable) // NOLINT
{
  GTEST_SKIP() << "deal.II was configured without SUNDIALS support.";
}

#endif
