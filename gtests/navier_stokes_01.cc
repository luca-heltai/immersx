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

#include <gtest/gtest.h>
#include <immersx/physics/navier_stokes.h>

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
        set Final time         = 0.15
        set Number of time steps = 3
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
          set Max steps  = 400
          set Reduction  = 1.e-10
          set Tolerance  = 1.e-12
          set Log result = false
        end
        set Inner maximum steps = 400
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
          set Max steps  = 400
          set Reduction  = 1.e-10
          set Tolerance  = 1.e-12
          set Log result = false
        end
        set Inner maximum steps = 400
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
  EXPECT_EQ(parameters.number_of_time_steps, 3u);
  EXPECT_DOUBLE_EQ(parameters.final_time, 0.15);
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

  ASSERT_EQ(problem.n_time_steps(), 3u);
  EXPECT_EQ(problem.timestep_number(), 3u);
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
  EXPECT_NEAR(problem.current_time(), 0.15, 1.e-14);
}


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
      set Initial refinement = 1
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
          set Max steps  = 500
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
