// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/lac/solver_gmres.h>

#include <gtest/gtest.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>
#include <immersx/physics/fiber_reinforced_elastodynamics.h>

#include <algorithm>
#include <cmath>

#include "test_paths.h"

using namespace ImmersX;
using namespace dealii;

namespace
{
  void
  configure_problem(FiberReinforcedElastodynamicsParameters<2> &parameters,
                    const bool                                  forcing,
                    const bool                                  ramp = false)
  {
    parameters.output_directory = TestPaths::output_directory(
      forcing ? "fiber-reinforced-forced" : "fiber-reinforced-zero");
    parameters.output_frequency = 0;

    initialize_parameters_from_string(
      std::string(R"(
        set dimension       = 2
        set space dimension = 2
        subsection Fiber Reinforced Elastodynamics
          subsection Time integration
            set Final time   = 0.02
            set Time step    = 0.01
            set Number of time steps = 2
          end
          subsection Matrix Elastodynamics
            set Initial refinement = 1
            set Dirichlet boundary ids = 0,1,2,3
            subsection Grid generation
              set Grid generator           = hyper_cube
              set Grid generator arguments = -1: 1: false
            end
            subsection Functions
              subsection Body force
                set Function expression = )") +
      (forcing ? (ramp ? "t; t" : "0; 1") : "0; 0") +
      R"(
                set Variable names      = x,y,t
              end
              subsection Displacement boundary
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Velocity boundary
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Initial displacement
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Initial velocity
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
            end
          end
          subsection Fiber Elastodynamics
            set Initial refinement = 1
            set Dirichlet boundary ids =
            subsection Grid generation
              set Grid generator           = subdivided_hyper_rectangle
              set Grid generator arguments = 3, 1: -0.6, -0.1: 0.6, 0.1: true
            end
            subsection Material
              set Density     = 2.0
              set Lame mu     = 2.0
              set Lame lambda = 2.0
            end
            subsection Functions
              subsection Body force
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Displacement boundary
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Velocity boundary
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Initial displacement
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Initial velocity
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
            end
          end
        end
      )");
  }

  template <int dim>
  void
  zero_constrained_entries(
    const AffineConstraints<double>                &constraints,
    typename ElastodynamicsSolver<dim>::VectorType &vector)
  {
    for (const auto index : vector.locally_owned_elements())
      if (constraints.is_constrained(index))
        vector(index) = 0.;
  }

#ifdef DEAL_II_WITH_SUNDIALS
  using FieldVector  = LA::MPI::Vector;
  using GlobalVector = LA::MPI::BlockVector;

  void
  solve_global_operator(
    const dealii::LinearOperator<GlobalVector> &operator_view,
    const GlobalVector                         &rhs,
    GlobalVector                               &dst,
    const double                                tolerance)
  {
    SolverControl              control(5000, std::max(1.e-6, tolerance), false);
    SolverFGMRES<GlobalVector> solver(control);
    dst = 0.;
    solver.solve(operator_view, dst, rhs, PreconditionIdentity());
  }
#endif
} // namespace

TEST(FiberReinforcedElastodynamics, ZeroPreservation)
{
  ParameterAcceptor::clear();
  FiberReinforcedElastodynamicsParameters<2> parameters;
  configure_problem(parameters, false);
  FiberReinforcedElastodynamics<2> driver(parameters);
  driver.run();

  EXPECT_EQ(driver.time_step_number(), 2u);
  EXPECT_NEAR(driver.current_time(), 0.02, 1.e-14);
  EXPECT_TRUE(driver.matrix_problem().state_is_finite());
  EXPECT_TRUE(driver.fiber_problem().state_is_finite());
  EXPECT_NEAR(driver.matrix_problem().displacement().l2_norm(), 0., 1.e-12);
  EXPECT_NEAR(driver.fiber_problem().displacement().l2_norm(), 0., 1.e-12);
  EXPECT_LT(driver.residuals().velocity_constraint, 1.e-10);
}

TEST(FiberReinforcedElastodynamics, CoupledResidualAndExcessResponse)
{
  ParameterAcceptor::clear();
  FiberReinforcedElastodynamicsParameters<2> parameters;
  configure_problem(parameters, true);
  FiberReinforcedElastodynamics<2> driver(parameters);
  driver.setup();
  driver.set_initial_conditions();
  driver.advance_one_timestep();
  driver.advance_one_timestep();

  EXPECT_TRUE(driver.matrix_problem().state_is_finite());
  EXPECT_TRUE(driver.fiber_problem().state_is_finite());
  EXPECT_LT(driver.residuals().velocity_constraint, 1.e-8);
  EXPECT_TRUE(std::isfinite(driver.fiber_excess_elastic_energy()));
  EXPECT_GT(driver.matrix_only_displacement_difference(), 1.e-12);
}

#ifdef DEAL_II_WITH_SUNDIALS
TEST(FiberReinforcedElastodynamics, Serial_IDAInitialDerivativeLifetime)
{
  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) != 1)
    GTEST_SKIP() << "Serial lifetime regression – skipped on multi-rank";

  ParameterAcceptor::clear();
  FiberReinforcedElastodynamicsParameters<2> parameters;
  configure_problem(parameters, true);
  FiberReinforcedElastodynamics<2> driver(parameters);

  // run() exercises setup(), initial conditions, setup_ida(), and the initial
  // derivative calculation, then continues into IDA after that scope ends.
  driver.run();

  EXPECT_EQ(driver.time_step_number(), 2u);
  EXPECT_TRUE(driver.matrix_problem().state_is_finite());
  EXPECT_TRUE(driver.fiber_problem().state_is_finite());
  EXPECT_LT(driver.residuals().velocity_constraint, 1.e-8);
}

TEST(FiberReinforcedElastodynamics, MPI_FiveFieldFiberIDA)
{
  ASSERT_GE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  ParameterAcceptor::clear();
  FiberReinforcedElastodynamicsParameters<2> parameters;
  configure_problem(parameters, true, true);
  parameters.final_time      = 0.001;
  parameters.number_of_steps = 1;
  FiberReinforcedElastodynamics<2> driver(parameters);
  driver.setup();
  driver.set_initial_conditions();
  const auto &interaction = driver.interaction();

  using Adapter = IDAAdapter<FieldVector, GlobalVector>;
  Adapter::AdditionalData data;
  data.initial_time                  = 0.;
  data.final_time                    = 0.001;
  data.initial_step_size             = 0.0005;
  data.output_period                 = 0.001;
  data.absolute_tolerance            = 1.e-7;
  data.relative_tolerance            = 1.e-7;
  data.maximum_order                 = 1;
  data.maximum_non_linear_iterations = 20;
  data.ic_type                       = Adapter::AdditionalData::none;
  data.reset_type                    = Adapter::AdditionalData::none;
  Adapter    ida(data, MPI_COMM_WORLD, solve_global_operator);
  const auto matrix   = ida.add(driver.matrix_problem(), "matrix");
  const auto fiber    = ida.add(driver.fiber_problem(), "fiber");
  const auto coupling = ida.add(driver.interaction(),
                                "fiber_coupling",
                                matrix.fields().velocity,
                                fiber.fields().velocity);

  auto state     = ida.make_state();
  auto state_dot = ida.make_state();
  auto residual  = ida.make_state();

  ida.field(state, matrix.fields().displacement)     = 0.2;
  ida.field(state, matrix.fields().velocity)         = -0.4;
  ida.field(state, fiber.fields().displacement)      = 0.3;
  ida.field(state, fiber.fields().velocity)          = 0.5;
  ida.field(state, coupling.fields().multiplier)     = -0.7;
  ida.field(state_dot, matrix.fields().displacement) = 0.6;
  ida.field(state_dot, matrix.fields().velocity)     = 0.8;
  ida.field(state_dot, fiber.fields().displacement)  = -0.9;
  ida.field(state_dot, fiber.fields().velocity)      = 1.1;
  ida.field(state_dot, coupling.fields().multiplier) = 0.;

  ida.solver().residual(0., state, state_dot, residual);
  const auto actual_matrix_d =
    ida.field(residual, matrix.fields().displacement);
  const auto  actual_matrix_v = ida.field(residual, matrix.fields().velocity);
  const auto  actual_fiber_d = ida.field(residual, fiber.fields().displacement);
  const auto  actual_fiber_v = ida.field(residual, fiber.fields().velocity);
  const auto  actual_lambda = ida.field(residual, coupling.fields().multiplier);
  FieldVector matrix_d_residual;
  FieldVector matrix_v_residual;
  FieldVector fiber_d_residual;
  FieldVector fiber_v_residual;
  FieldVector lambda_residual;
  matrix_d_residual.reinit(actual_matrix_d);
  matrix_v_residual.reinit(actual_matrix_v);
  fiber_d_residual.reinit(actual_fiber_d);
  fiber_v_residual.reinit(actual_fiber_v);
  lambda_residual.reinit(actual_lambda);
  FieldVector work;
  FieldVector work_f;
  FieldVector work_lambda;
  work.reinit(matrix_d_residual);
  work_f.reinit(fiber_d_residual);
  work_lambda.reinit(lambda_residual);

  driver.matrix_problem().mass_matrix().vmult(
    matrix_d_residual, ida.field(state_dot, matrix.fields().displacement));
  driver.matrix_problem().mass_matrix().vmult(
    work, ida.field(state, matrix.fields().velocity));
  matrix_d_residual -= work;
  zero_constrained_entries<2>(driver.matrix_problem().constraints(),
                              matrix_d_residual);

  driver.matrix_problem().mass_matrix().vmult(
    matrix_v_residual, ida.field(state_dot, matrix.fields().velocity));
  driver.matrix_problem().stiffness_matrix().vmult(
    work, ida.field(state, matrix.fields().displacement));
  matrix_v_residual += work;
  driver.matrix_problem().damping_matrix().vmult(
    work, ida.field(state, matrix.fields().velocity));
  matrix_v_residual += work;
  FieldVector force;
  driver.matrix_problem().body_force_at_time(0., force);
  matrix_v_residual -= force;
  interaction.coupling_matrix().vmult(work,
                                      ida.field(state,
                                                coupling.fields().multiplier));
  matrix_v_residual += work;
  zero_constrained_entries<2>(driver.matrix_problem().velocity_constraints(),
                              matrix_v_residual);

  driver.fiber_problem().mass_matrix().vmult(
    fiber_d_residual, ida.field(state_dot, fiber.fields().displacement));
  driver.fiber_problem().mass_matrix().vmult(
    work_f, ida.field(state, fiber.fields().velocity));
  fiber_d_residual -= work_f;
  zero_constrained_entries<2>(driver.fiber_problem().constraints(),
                              fiber_d_residual);

  driver.fiber_problem().mass_matrix().vmult(
    fiber_v_residual, ida.field(state_dot, fiber.fields().velocity));
  driver.fiber_problem().stiffness_matrix().vmult(
    work_f, ida.field(state, fiber.fields().displacement));
  fiber_v_residual += work_f;
  driver.fiber_problem().damping_matrix().vmult(
    work_f, ida.field(state, fiber.fields().velocity));
  fiber_v_residual += work_f;
  driver.fiber_problem().body_force_at_time(0., force);
  fiber_v_residual -= force;
  interaction.pairing_matrix().Tvmult(work_f,
                                      ida.field(state,
                                                coupling.fields().multiplier));
  fiber_v_residual -= work_f;
  zero_constrained_entries<2>(driver.fiber_problem().velocity_constraints(),
                              fiber_v_residual);

  interaction.coupling_matrix().Tvmult(lambda_residual,
                                       ida.field(state,
                                                 matrix.fields().velocity));
  interaction.pairing_matrix().vmult(work_lambda,
                                     ida.field(state, fiber.fields().velocity));
  lambda_residual -= work_lambda;
  auto check = actual_matrix_d;
  check -= matrix_d_residual;
  EXPECT_LT(check.l2_norm(), 1.e-11);
  check = actual_matrix_v;
  check -= matrix_v_residual;
  EXPECT_LT(check.l2_norm(), 1.e-11);
  check = actual_fiber_d;
  check -= fiber_d_residual;
  EXPECT_LT(check.l2_norm(), 1.e-11);
  check = actual_fiber_v;
  check -= fiber_v_residual;
  EXPECT_LT(check.l2_norm(), 1.e-11);
  check = actual_lambda;
  check -= lambda_residual;
  EXPECT_LT(check.l2_norm(), 1.e-11);

  auto increment                                     = ida.make_state();
  ida.field(increment, matrix.fields().displacement) = -0.6;
  ida.field(increment, matrix.fields().velocity)     = 0.4;
  ida.field(increment, fiber.fields().displacement)  = 0.7;
  ida.field(increment, fiber.fields().velocity)      = -0.3;
  ida.field(increment, coupling.fields().multiplier) = 0.8;
  auto action                                        = ida.make_state();
  ida.solver().setup_jacobian(0., state, state_dot, 1.5);
  ida.current_jacobian().vmult(action, increment);

  auto expected_matrix_d_action =
    ida.field(action, matrix.fields().displacement);
  auto expected_matrix_v_action = ida.field(action, matrix.fields().velocity);
  auto expected_fiber_d_action = ida.field(action, fiber.fields().displacement);
  auto expected_fiber_v_action = ida.field(action, fiber.fields().velocity);
  auto expected_lambda_action = ida.field(action, coupling.fields().multiplier);
  work.reinit(expected_matrix_d_action);
  driver.matrix_problem().mass_matrix().vmult(
    expected_matrix_d_action,
    ida.field(increment, matrix.fields().displacement));
  expected_matrix_d_action *= 1.5;
  driver.matrix_problem().mass_matrix().vmult(
    work, ida.field(increment, matrix.fields().velocity));
  expected_matrix_d_action -= work;
  zero_constrained_entries<2>(driver.matrix_problem().constraints(),
                              expected_matrix_d_action);
  driver.matrix_problem().stiffness_matrix().vmult(
    expected_matrix_v_action,
    ida.field(increment, matrix.fields().displacement));
  driver.matrix_problem().damping_matrix().vmult(
    work, ida.field(increment, matrix.fields().velocity));
  expected_matrix_v_action += work;
  driver.matrix_problem().mass_matrix().vmult(
    work, ida.field(increment, matrix.fields().velocity));
  work *= 1.5;
  expected_matrix_v_action += work;
  interaction.coupling_matrix().vmult(work,
                                      ida.field(increment,
                                                coupling.fields().multiplier));
  expected_matrix_v_action += work;
  zero_constrained_entries<2>(driver.matrix_problem().velocity_constraints(),
                              expected_matrix_v_action);

  work_f.reinit(expected_fiber_d_action);
  driver.fiber_problem().mass_matrix().vmult(
    expected_fiber_d_action, ida.field(increment, fiber.fields().displacement));
  expected_fiber_d_action *= 1.5;
  driver.fiber_problem().mass_matrix().vmult(
    work_f, ida.field(increment, fiber.fields().velocity));
  expected_fiber_d_action -= work_f;
  zero_constrained_entries<2>(driver.fiber_problem().constraints(),
                              expected_fiber_d_action);
  driver.fiber_problem().stiffness_matrix().vmult(
    expected_fiber_v_action, ida.field(increment, fiber.fields().displacement));
  driver.fiber_problem().damping_matrix().vmult(
    work_f, ida.field(increment, fiber.fields().velocity));
  expected_fiber_v_action += work_f;
  driver.fiber_problem().mass_matrix().vmult(
    work_f, ida.field(increment, fiber.fields().velocity));
  work_f *= 1.5;
  expected_fiber_v_action += work_f;
  interaction.pairing_matrix().Tvmult(work_f,
                                      ida.field(increment,
                                                coupling.fields().multiplier));
  expected_fiber_v_action -= work_f;
  zero_constrained_entries<2>(driver.fiber_problem().velocity_constraints(),
                              expected_fiber_v_action);

  interaction.coupling_matrix().Tvmult(expected_lambda_action,
                                       ida.field(increment,
                                                 matrix.fields().velocity));
  interaction.pairing_matrix().vmult(work_lambda,
                                     ida.field(increment,
                                               fiber.fields().velocity));
  expected_lambda_action -= work_lambda;
  check = ida.field(action, matrix.fields().displacement);
  check -= expected_matrix_d_action;
  EXPECT_LT(check.l2_norm(), 1.e-11);
  check = ida.field(action, matrix.fields().velocity);
  check -= expected_matrix_v_action;
  EXPECT_LT(check.l2_norm(), 1.e-11);
  check = ida.field(action, fiber.fields().displacement);
  check -= expected_fiber_d_action;
  EXPECT_LT(check.l2_norm(), 1.e-11);
  check = ida.field(action, fiber.fields().velocity);
  check -= expected_fiber_v_action;
  EXPECT_LT(check.l2_norm(), 1.e-11);
  check = ida.field(action, coupling.fields().multiplier);
  check -= expected_lambda_action;
  EXPECT_LT(check.l2_norm(), 1.e-11);

  // Use the problem's benign initial state for the nonlinear solve after
  // exercising the residual and Jacobian at arbitrary states above.
  state            = 0.;
  state_dot        = 0.;
  const auto steps = ida.solve(state, state_dot);
  EXPECT_GT(steps, 0u);
  EXPECT_TRUE(std::isfinite(state.l2_norm()));

  ida.solver().residual(data.final_time, state, state_dot, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-5);
  EXPECT_LT(ida.field(residual, coupling.fields().multiplier).l2_norm(), 1.e-5);
}
#endif

TEST(FiberReinforcedElastodynamics, MPI_CoupledTransient)
{
  ASSERT_GE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  ParameterAcceptor::clear();
  FiberReinforcedElastodynamicsParameters<2> parameters;
  configure_problem(parameters, true);
  FiberReinforcedElastodynamics<2> driver(parameters);
  driver.run();

  EXPECT_TRUE(driver.matrix_problem().state_is_finite());
  EXPECT_TRUE(driver.fiber_problem().state_is_finite());
  EXPECT_LT(driver.residuals().velocity_constraint, 1.e-8);
}

TEST(FiberReinforcedElastodynamics, ThreeDimensionalSmoke)
{
  ParameterAcceptor::clear();
  FiberReinforcedElastodynamicsParameters<3> parameters;
  parameters.output_directory =
    TestPaths::output_directory("fiber-reinforced-3d");
  parameters.output_frequency                     = 0;
  parameters.matrix_parameters.initial_refinement = 0;
  parameters.matrix_parameters.dirichlet_ids      = {0, 1, 2, 3, 4, 5};
  parameters.fiber_parameters.initial_refinement  = 0;
  parameters.fiber_parameters.dirichlet_ids.clear();

  initialize_parameters_from_string(R"(
    set dimension       = 3
    set space dimension = 3
    subsection Fiber Reinforced Elastodynamics
      subsection Time integration
        set Time step          = 0.01
        set Number of time steps = 1
      end
      subsection Matrix Elastodynamics
        subsection Grid generation
          set Grid generator           = hyper_cube
          set Grid generator arguments = -1: 1: false
        end
      end
      subsection Fiber Elastodynamics
        subsection Grid generation
          set Grid generator           = subdivided_hyper_rectangle
          set Grid generator arguments = 2, 1, 1: -0.6, -0.1, -0.1: 0.6, 0.1, 0.1: true
        end
      end
    end
  )");

  FiberReinforcedElastodynamics<3> driver(parameters);
  driver.run();
  EXPECT_TRUE(driver.matrix_problem().state_is_finite());
  EXPECT_TRUE(driver.fiber_problem().state_is_finite());
  EXPECT_EQ(driver.time_step_number(), 1u);
}
