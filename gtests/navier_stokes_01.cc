// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/lac/solver_gmres.h>

#include <gtest/gtest.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/physics/navier_stokes.h>
#include <immersx/physics/navier_stokes_semidiscrete.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

using namespace ImmersX;
#include <immersx/io/utils.h>


using namespace dealii;


namespace
{
  const std::string two_dimensional_parameters = R"(
    subsection Navier-Stokes
      set Output frequency   = 0
      set Initial refinement  = 1
      set Dirichlet boundary ids = 0
      subsection Finite element spaces
        set Velocity degree = 2
        set Pressure degree = 1
      end
      subsection Grid generation
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Physical properties
        set Density                 = 1
        set Viscosity               = 1
        set Include convective term = false
      end
      subsection Time stepping
        set Policy             = number_of_steps
        set Initial time       = 0
        set Final time         = 0.05
        set Number of time steps = 1
      end
      subsection Right hand side
        set Function expression = 1; 0; 0
        set Variable names      = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0; 0; 0
        set Variable names      = x,y,t
      end
      subsection Initial condition
        set Function expression = 0; 0; 0
        set Variable names      = x,y,t
      end
      subsection Solver
        subsection Control
          set Max steps  = 100
          set Reduction  = 1.e-10
          set Tolerance  = 1.e-12
          set Log result = false
        end
        set Inner maximum steps = 100
        set Inner tolerance     = 1.e-10
        set Log iterations      = false
      end
    end
  )";


  const std::string convective_parameters = R"(
    subsection Navier-Stokes
      set Output frequency   = 0
      set Initial refinement = 1
      set Dirichlet boundary ids = 0
      subsection Finite element spaces
        set Velocity degree = 2
        set Pressure degree = 1
      end
      subsection Grid generation
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Physical properties
        set Density                 = 1
        set Viscosity               = 1
        set Include convective term = true
      end
      subsection Time stepping
        set Policy               = number_of_steps
        set Initial time         = 0
        set Final time           = 0.05
        set Number of time steps = 1
      end
      subsection Right hand side
        set Function expression = 0; 0; 0
        set Variable names      = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0; 0; 0
        set Variable names      = x,y,t
      end
      subsection Initial condition
        set Function expression = 0.1*(1-y^2); 0; 0
        set Variable names      = x,y,t
      end
      subsection Solver
        subsection Control
          set Max steps  = 100
          set Reduction  = 1.e-10
          set Tolerance  = 1.e-12
          set Log result = false
        end
        set Inner maximum steps = 100
        set Inner tolerance     = 1.e-10
        set Log iterations      = false
      end
    end
  )";


  template <typename Parameters>
  void
  configure_output(Parameters &parameters, const std::string &name)
  {
    parameters.output_directory =
      (std::filesystem::temp_directory_path() / name).string();
  }
} // namespace


TEST(NavierStokes, Construction)
{
  ParameterAcceptor::clear();
  NavierStokesParameters<2> parameters;
  NavierStokesSolver<2>     problem(parameters);

  EXPECT_EQ(problem.n_dofs(), 0u);
  EXPECT_FALSE(problem.solution_is_finite());
}


TEST(NavierStokes, ParameterParsing)
{
  ParameterAcceptor::clear();
  NavierStokesParameters<2> parameters;
  initialize_parameters_from_string(two_dimensional_parameters);

  EXPECT_EQ(parameters.velocity_degree, 2u);
  EXPECT_EQ(parameters.pressure_degree, 1u);
  EXPECT_FALSE(parameters.include_convective_term);
  EXPECT_EQ(parameters.number_of_time_steps, 1u);
  EXPECT_DOUBLE_EQ(parameters.final_time, 0.05);
}


TEST(NavierStokes, VelocityPressureComponentMask)
{
  ParameterAcceptor::clear();
  NavierStokesParameters<2> parameters;
  initialize_parameters_from_string(two_dimensional_parameters);
  configure_output(parameters, "immersx_navier_stokes_mask");

  NavierStokesSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();

  const auto velocity_mask = problem.velocity_component_mask();
  const auto pressure_mask =
    problem.finite_element().component_mask(problem.pressure_extractor());
  EXPECT_EQ(velocity_mask.n_selected_components(), 2u);
  EXPECT_EQ(pressure_mask.n_selected_components(), 1u);
  EXPECT_TRUE(velocity_mask[0]);
  EXPECT_TRUE(velocity_mask[1]);
  EXPECT_FALSE(velocity_mask[2]);
  EXPECT_FALSE(pressure_mask[0]);
  EXPECT_FALSE(pressure_mask[1]);
  EXPECT_TRUE(pressure_mask[2]);
}


TEST(NavierStokes, MPI_TransientStokes)
{
  ParameterAcceptor::clear();
  NavierStokesParameters<2> parameters;
  initialize_parameters_from_string(two_dimensional_parameters);
  configure_output(parameters, "immersx_navier_stokes_stokes");

  NavierStokesSolver<2> problem(parameters);
  problem.run();

  ASSERT_EQ(problem.n_time_steps(), 1u);
  EXPECT_EQ(problem.timestep_number(), 1u);
  EXPECT_GT(problem.n_dofs(), 0u);
  ASSERT_EQ(problem.system_matrix().n_block_rows(), 2u);
  EXPECT_GT(problem.system_matrix().block(0, 0).m(), 0u);
  EXPECT_GT(problem.system_matrix().block(1, 1).m(), 0u);
  EXPECT_EQ(problem.system_matrix().block(0, 0).m(),
            problem.system_matrix().block(0, 0).n());
  EXPECT_TRUE(problem.solution_is_finite());
  EXPECT_TRUE(std::isfinite(problem.solution().block(0).l2_norm()));
  EXPECT_TRUE(std::isfinite(problem.solution().block(1).l2_norm()));
  EXPECT_LT(problem.system_residual_l2_norm(), 1.e-7);
  EXPECT_LT(problem.divergence_l2_norm(), 1.e-8);
  EXPECT_NEAR(problem.current_time(), 0.05, 1.e-14);
}

#ifdef DEAL_II_WITH_SUNDIALS
TEST(NavierStokes, MPI_IDAResidualJacobianAndSolve)
{
  ParameterAcceptor::clear();
  NavierStokesParameters<2> parameters;
  initialize_parameters_from_string(two_dimensional_parameters);
  configure_output(parameters, "immersx_navier_stokes_ida");

  NavierStokesSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_system();

  using FieldVector  = LA::MPI::Vector;
  using GlobalVector = LA::MPI::BlockVector;
  using Adapter      = IDAAdapter<FieldVector, GlobalVector>;
  Adapter::AdditionalData data;
  data.initial_time      = 0.;
  data.final_time        = 0.05;
  data.initial_step_size = 0.025;
  data.output_period     = 0.05;
  data.maximum_order     = 1;
  data.ic_type           = Adapter::AdditionalData::none;
  Adapter    ida(data,
              MPI_COMM_WORLD,
              [](const dealii::LinearOperator<GlobalVector> &operator_view,
                 const GlobalVector                         &rhs,
                 GlobalVector                               &dst,
                 const double                                tolerance) {
                SolverControl control(5000, std::max(1.e-8, tolerance));
                SolverFGMRES<GlobalVector> solver(control);
                dst = 0.;
                solver.solve(operator_view, dst, rhs, PreconditionIdentity());
              });
  const auto fields = ida.add(problem, "fluid");

  auto state                                     = ida.make_state();
  auto state_dot                                 = ida.make_state();
  auto residual                                  = ida.make_state();
  ida.field(state, fields.fields().velocity)     = 0.2;
  ida.field(state, fields.fields().pressure)     = -0.1;
  ida.field(state_dot, fields.fields().velocity) = 0.4;
  ida.field(state_dot, fields.fields().pressure) = 0.;
  ida.solver().residual(0., state, state_dot, residual);

  auto expected_velocity = ida.field(residual, fields.fields().velocity);
  auto work              = expected_velocity;
  problem.velocity_mass_matrix().vmult(expected_velocity,
                                       ida.field(state_dot,
                                                 fields.fields().velocity));
  problem.continuous_operator().block(0, 0).vmult(
    work, ida.field(state, fields.fields().velocity));
  expected_velocity += work;
  problem.continuous_operator().block(0, 1).vmult(
    work, ida.field(state, fields.fields().pressure));
  expected_velocity += work;
  LA::MPI::Vector force;
  problem.velocity_forcing_at_time(0., force);
  force *= problem.density();
  expected_velocity -= force;
  for (const auto index : expected_velocity.locally_owned_elements())
    if (problem.constraints().is_constrained(index))
      expected_velocity(index) = 0.;
  auto velocity_difference = ida.field(residual, fields.fields().velocity);
  velocity_difference -= expected_velocity;
  EXPECT_NEAR(velocity_difference.l2_norm(), 0., 1.e-10);

  auto expected_pressure = ida.field(residual, fields.fields().pressure);
  problem.continuous_operator().block(1, 0).vmult(
    expected_pressure, ida.field(state, fields.fields().velocity));
  for (const auto index : expected_pressure.locally_owned_elements())
    if (problem.constraints().is_constrained(problem.velocity_block_size() +
                                             index))
      expected_pressure(index) = 0.;
  auto pressure_difference = ida.field(residual, fields.fields().pressure);
  pressure_difference -= expected_pressure;
  EXPECT_NEAR(pressure_difference.l2_norm(), 0., 1.e-10);

  auto increment                                 = ida.make_state();
  ida.field(increment, fields.fields().velocity) = -0.3;
  ida.field(increment, fields.fields().pressure) = 0.5;
  auto action                                    = ida.make_state();
  ida.solver().setup_jacobian(0., state, state_dot, 2.);
  ida.current_jacobian().vmult(action, increment);

  auto expected_velocity_action = ida.field(action, fields.fields().velocity);
  problem.velocity_mass_matrix().vmult(expected_velocity_action,
                                       ida.field(increment,
                                                 fields.fields().velocity));
  expected_velocity_action *= 2.;
  problem.continuous_operator().block(0, 0).vmult(
    work, ida.field(increment, fields.fields().velocity));
  expected_velocity_action += work;
  problem.continuous_operator().block(0, 1).vmult(
    work, ida.field(increment, fields.fields().pressure));
  expected_velocity_action += work;
  for (const auto index : expected_velocity_action.locally_owned_elements())
    if (problem.constraints().is_constrained(index))
      expected_velocity_action(index) = 0.;
  velocity_difference = ida.field(action, fields.fields().velocity);
  velocity_difference -= expected_velocity_action;
  EXPECT_NEAR(velocity_difference.l2_norm(), 0., 1.e-10);

  auto expected_pressure_action = ida.field(action, fields.fields().pressure);
  problem.continuous_operator().block(1, 0).vmult(
    expected_pressure_action, ida.field(increment, fields.fields().velocity));
  for (const auto index : expected_pressure_action.locally_owned_elements())
    if (problem.constraints().is_constrained(problem.velocity_block_size() +
                                             index))
      expected_pressure_action(index) = 0.;
  pressure_difference = ida.field(action, fields.fields().pressure);
  pressure_difference -= expected_pressure_action;
  EXPECT_NEAR(pressure_difference.l2_norm(), 0., 1.e-10);

  auto state_for_solve     = ida.make_state();
  auto state_dot_for_solve = ida.make_state();
  EXPECT_GT(ida.solve(state_for_solve, state_dot_for_solve), 0u);
  ida.solver().residual(data.final_time,
                        state_for_solve,
                        state_dot_for_solve,
                        residual);
  EXPECT_TRUE(std::isfinite(state_for_solve.l2_norm()));
  EXPECT_LT(residual.l2_norm(), 1.e-6);
}
#endif


TEST(NavierStokes, MPI_ExplicitConvectionPath)
{
  double          stokes_rhs_norm     = 0.;
  double          convective_rhs_norm = 0.;
  double          rhs_difference      = 0.;
  LA::MPI::Vector stokes_rhs;

  {
    ParameterAcceptor::clear();
    NavierStokesParameters<2> parameters;
    initialize_parameters_from_string(convective_parameters);
    parameters.include_convective_term = false;
    configure_output(parameters, "immersx_navier_stokes_no_convection");

    NavierStokesSolver<2> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_system();
    stokes_rhs_norm = problem.system_rhs().block(0).l2_norm();
    stokes_rhs.reinit(problem.system_rhs().block(0));
    stokes_rhs = problem.system_rhs().block(0);
    problem.advance_one_timestep();
    EXPECT_TRUE(problem.solution_is_finite());
  }

  {
    ParameterAcceptor::clear();
    NavierStokesParameters<2> parameters;
    initialize_parameters_from_string(convective_parameters);
    configure_output(parameters, "immersx_navier_stokes_convection");

    NavierStokesSolver<2> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_system();
    convective_rhs_norm = problem.system_rhs().block(0).l2_norm();
    auto rhs_delta      = problem.system_rhs().block(0);
    rhs_delta -= stokes_rhs;
    rhs_difference = rhs_delta.l2_norm();
    problem.advance_one_timestep();
    EXPECT_TRUE(problem.solution_is_finite());
    EXPECT_EQ(problem.timestep_number(), 1u);
  }

  EXPECT_TRUE(std::isfinite(stokes_rhs_norm));
  EXPECT_TRUE(std::isfinite(convective_rhs_norm));
  EXPECT_GT(rhs_difference, 1.e-10);
}


TEST(NavierStokes, MPI_ThreeDimensionalSmoke)
{
  ParameterAcceptor::clear();
  NavierStokesParameters<3> parameters;
  initialize_parameters_from_string(R"(
    subsection Navier-Stokes
      set Output frequency   = 0
      set Initial refinement = 0
      set Dirichlet boundary ids = 0
      subsection Time stepping
        set Final time           = 0.02
        set Number of time steps = 1
      end
      subsection Right hand side
        set Function expression = 1; 0; 0; 0
        set Variable names      = x,y,z,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0; 0; 0; 0
        set Variable names      = x,y,z,t
      end
      subsection Initial condition
        set Function expression = 0; 0; 0; 0
        set Variable names      = x,y,z,t
      end
      subsection Solver
        subsection Control
          set Max steps  = 100
          set Reduction  = 1.e-9
          set Tolerance  = 1.e-11
          set Log result = false
        end
      end
    end
  )");
  configure_output(parameters, "immersx_navier_stokes_3d");

  NavierStokesSolver<3> problem(parameters);
  problem.run();

  EXPECT_TRUE(problem.solution_is_finite());
  EXPECT_LT(problem.system_residual_l2_norm(), 1.e-6);
}
