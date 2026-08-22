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
                                matrix.velocity,
                                fiber.velocity);

  auto state     = ida.make_state();
  auto state_dot = ida.make_state();
  auto residual  = ida.make_state();
  ida.solver().residual(0., state, state_dot, residual);
  ida.solver().setup_jacobian(0., state, state_dot, 1.);
  auto action = ida.make_state();
  ida.current_jacobian().vmult(action, state);
  const auto steps = ida.solve(state, state_dot);
  EXPECT_GT(steps, 0u);
  EXPECT_TRUE(std::isfinite(state.l2_norm()));

  ida.solver().residual(data.final_time, state, state_dot, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-5);
  EXPECT_LT(ida.field(state, coupling.multiplier).l2_norm(), 1.e-5);
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
