// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/navier_stokes.h>
#include <immersx/physics/navier_stokes_semidiscrete.h>

#include <sstream>

#include "test_paths.h"

using namespace ImmersX;
using namespace dealii;

namespace
{
  template <int dim>
  std::string
  robin_parameters(const bool zero_beta = false, const bool mixed = false)
  {
    const std::string variables = dim == 2 ? "x,y,t" : "x,y,z,t";
    const std::string velocity =
      dim == 2 ? (mixed ? "1;2" : "1+t;2-t") : "1+t;2-t;.5+2*t";
    const std::string force =
      dim == 2 ? (mixed ? "0;0;0" : "1;-1;0") : "1;-1;2;0";
    const std::string beta     = zero_beta ? "0" : "2+x+t";
    const std::string nx       = "(if(x<1.e-8,-1,0)+if(x>1-1.e-8,1,0))";
    const std::string ny       = "(if(y<1.e-8,-1,0)+if(y>1-1.e-8,1,0))";
    std::string       traction = "(" + beta + ")*(" + (mixed ? "1" : "1+t") +
                           ")-(3+t)*" + nx + ";(" + beta + ")*(" +
                           (mixed ? "2" : "2-t") + ")-(3+t)*" + ny;
    if constexpr (dim == 3)
      traction +=
        ";(" + beta + ")*(.5+2*t)-(3+t)*(if(z<1.e-8,-1,0)+if(z>1-1.e-8,1,0))";
    std::ostringstream prm;
    prm
      << "subsection Navier-Stokes\n"
      << " set Dirichlet boundary ids = " << (mixed ? "0" : "")
      << "\n set Robin boundary ids = " << (mixed ? "1,2,3" : "0") << "\n"
      << " subsection Domain\n set Initial refinement = 0\n"
      << " set Grid generator = hyper_cube\n set Grid generator arguments = 0:1:"
      << (mixed ? "true" : "false") << "\n end\n"
      << " subsection Physical properties\n set Density = 3\n set Viscosity = .7\n set Include convective term = false\n end\n"
      << " subsection Time interval\n set Initial time = 0\n set Final time = .02\n set Output time interval = .02\n end\n"
      << " subsection Fixed step\n set Policy = number_of_steps\n set Number of time steps = 2\n end\n"
      << " subsection Right hand side\n set Variable names = " << variables
      << "\n set Function expression = " << force << "\n end\n"
      << " subsection Initial condition\n set Variable names = " << variables
      << "\n set Function expression = " << velocity << ";3+t\n end\n"
      << " subsection Dirichlet boundary conditions\n set Variable names = "
      << variables << "\n set Function expression = " << velocity
      << ";0\n end\n"
      << " subsection Robin coefficient\n set Variable names = " << variables
      << "\n set Function expression = " << beta << "\n end\n"
      << " subsection Robin boundary conditions\n set Variable names = "
      << variables << "\n set Function expression = " << traction << "\n end\n"
      << " subsection Solver\n subsection Control\n set Max steps = 1000\n set Reduction = 1.e-12\n set Tolerance = 1.e-13\n end\n"
      << " set Inner maximum steps = 1000\n set Inner tolerance = 1.e-13\n end\n end\n";
    return prm.str();
  }

  template <int dim>
  void
  check_robin_solution(const NavierStokesSolver<dim> &problem,
                       const double                   time,
                       const bool                     mixed = false)
  {
    FunctionParser<dim> exact(dim + 1);
    exact.initialize(dim == 2 ? "x,y,t" : "x,y,z,t",
                     mixed ? "1;2;3+t" :
                             (dim == 2 ? "1+t;2-t;3+t" : "1+t;2-t;.5+2*t;3+t"),
                     typename FunctionParser<dim>::ConstMap(),
                     true);
    exact.set_time(time);
    Vector<float> cell_error(
      problem.dof_handler().get_triangulation().n_active_cells());
    VectorTools::integrate_difference(problem.mapping(),
                                      problem.dof_handler(),
                                      problem.locally_relevant_solution(),
                                      exact,
                                      cell_error,
                                      QGauss<dim>(4),
                                      VectorTools::L2_norm);
    EXPECT_LT(VectorTools::compute_global_error(
                problem.dof_handler().get_triangulation(),
                cell_error,
                VectorTools::L2_norm),
              2.e-8);
    const auto pressure_dof = problem.velocity_block_size();
    if (problem.locally_relevant_dofs().is_element(pressure_dof))
      {
        EXPECT_FALSE(problem.constraints().is_constrained(pressure_dof));
      }
  }

  template <int dim>
  void
  check_native_robin(const bool zero_beta = false, const bool mixed = false)
  {
    ParameterAcceptor::clear();
    NavierStokesParameters<dim> parameters;
    parameters.output_directory =
      TestPaths::output_directory("navier-stokes-robin-native");
    initialize_parameters_from_string(robin_parameters<dim>(zero_beta, mixed));
    NavierStokesSolver<dim> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    ASSERT_TRUE(problem.has_robin_boundary());
    for (unsigned int step = 0; step < 2; ++step)
      {
        problem.advance_one_timestep();
        check_robin_solution(problem, problem.current_time(), mixed);
        EXPECT_LT(problem.system_residual_l2_norm(), 1.e-8);
      }
  }
} // namespace

TEST(NavierStokes, BOTH_RobinNative2D)
{
  check_native_robin<2>();
}

TEST(NavierStokes, BOTH_RobinNative3D)
{
  check_native_robin<3>();
}

TEST(NavierStokes, BOTH_RobinZeroCoefficientTractionFixesPressure)
{
  check_native_robin<2>(true);
}

#ifdef DEAL_II_WITH_SUNDIALS
namespace
{
  void
  check_ida_robin(const bool mixed)
  {
    ParameterAcceptor::clear();
    NavierStokesParameters<2> parameters;
    parameters.output_directory =
      TestPaths::output_directory("navier-stokes-robin-ida");
    initialize_parameters_from_string(robin_parameters<2>(false, mixed));
    parameters.ida_parameters.initial_step_size               = .005;
    parameters.ida_parameters.absolute_tolerance              = 1.e-10;
    parameters.ida_parameters.relative_tolerance              = 1.e-10;
    parameters.ida_parameters.correction_type_at_initial_time = "none";
    NavierStokesSolver<2> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_system();

    using FieldVector  = NavierStokesSolver<2>::VectorType;
    using GlobalVector = NavierStokesSolver<2>::BlockVectorType;
    IDAAdapter<FieldVector, GlobalVector> adapter(parameters.time_parameters,
                                                  parameters.ida_parameters,
                                                  MPI_COMM_WORLD);
    const auto fields       = adapter.add(problem, "fluid");
    auto       state        = adapter.make_state();
    auto       state_dot    = adapter.make_state();
    const auto assign_exact = [&](const double time) {
      FunctionParser<2> exact(3), derivative(3);
      exact.initialize("x,y,t",
                       mixed ? "1;2;3+t" : "1+t;2-t;3+t",
                       FunctionParser<2>::ConstMap(),
                       true);
      derivative.initialize("x,y",
                            mixed ? "0;0;1" : "1;-1;1",
                            FunctionParser<2>::ConstMap());
      exact.set_time(time);
      auto native_state = problem.solution();
      auto native_dot   = problem.solution();
      VectorTools::interpolate(problem.mapping(),
                               problem.dof_handler(),
                               exact,
                               native_state);
      VectorTools::interpolate(problem.mapping(),
                               problem.dof_handler(),
                               derivative,
                               native_dot);
      adapter.field(state, fields.fields().velocity) = native_state.block(0);
      adapter.field(state, fields.fields().pressure) = native_state.block(1);
      adapter.field(state_dot, fields.fields().velocity) = native_dot.block(0);
      adapter.field(state_dot, fields.fields().pressure) = native_dot.block(1);
    };

    // The native clock is still zero. These checks require the adapter to use
    // its own evaluation time for both Robin coefficient and traction data.
    const double check_time = .007;
    assign_exact(check_time);
    auto residual = adapter.make_state();
    adapter.solver().residual(check_time, state, state_dot, residual);
    EXPECT_LT(residual.l2_norm(), 1.e-9);

    auto increment = adapter.make_state();
    for (unsigned int block = 0; block < increment.n_blocks(); ++block)
      for (const auto index : increment.block(block).locally_owned_elements())
        increment.block(block)[index] =
          block == 0 && problem.constraints().is_constrained(index) ?
            0. :
            .01 * (1. + index) / increment.block(block).size();
    const double sigma = 1.4;
    adapter.solver().setup_jacobian(check_time, state, state_dot, sigma);
    auto action = adapter.make_state();
    adapter.current_jacobian().vmult(action, increment);
    const double epsilon         = 1.e-5;
    auto         perturbed_state = state;
    auto         perturbed_dot   = state_dot;
    perturbed_state.add(epsilon, increment);
    perturbed_dot.add(epsilon * sigma, increment);
    auto perturbed_residual = adapter.make_state();
    adapter.solver().residual(check_time,
                              perturbed_state,
                              perturbed_dot,
                              perturbed_residual);
    perturbed_residual -= residual;
    perturbed_residual /= epsilon;
    perturbed_residual -= action;
    EXPECT_LT(perturbed_residual.l2_norm(), 1.e-7);

    assign_exact(0.);
    adapter.solve(state, state_dot);
    problem.accept_state(adapter.field(state, fields.fields().velocity),
                         adapter.field(state, fields.fields().pressure),
                         parameters.time_parameters.final_time,
                         1);
    check_robin_solution(problem, parameters.time_parameters.final_time, mixed);
  }
} // namespace
TEST(NavierStokes, BOTH_RobinIDAResidualJacobianAndTime)
{
  check_ida_robin(false);
}
TEST(NavierStokes, BOTH_RobinIDAMixedNonzeroDirichlet)
{
  check_ida_robin(true);
}
#endif

TEST(NavierStokes, BOTH_RobinRejectsConflictingAndInvalidDataCollectively)
{
  {
    ParameterAcceptor::clear();
    NavierStokesParameters<2> parameters;
    parameters.output_directory =
      TestPaths::output_directory("navier-stokes-robin-invalid");
    initialize_parameters_from_string(robin_parameters<2>());
    parameters.dirichlet_ids = {0};
    NavierStokesSolver<2> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    EXPECT_THROW(problem.setup_system(), ExceptionBase);
  }
  {
    ParameterAcceptor::clear();
    NavierStokesParameters<2> parameters;
    parameters.output_directory =
      TestPaths::output_directory("navier-stokes-robin-invalid");
    initialize_parameters_from_string(robin_parameters<2>() + R"(
      subsection Navier-Stokes
        subsection Robin coefficient
          set Function expression = -1
        end
      end
    )");
    NavierStokesSolver<2> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    // With one cell and two ranks, the empty rank must also throw before
    // matrix compression rather than waiting in a different collective.
    EXPECT_THROW(problem.assemble_system(), ExceptionBase);
  }
}

TEST(NavierStokes, BOTH_RobinNativeMixedNonzeroDirichlet)
{
  check_native_robin<2>(false, true);
}

TEST(NavierStokes, BOTH_RobinAbsentBoundaryPreservesPressurePin)
{
  ParameterAcceptor::clear();
  NavierStokesParameters<2> parameters;
  parameters.output_directory =
    TestPaths::output_directory("navier-stokes-robin-absent");
  initialize_parameters_from_string(robin_parameters<2>());
  parameters.dirichlet_ids = {0};
  parameters.robin_ids     = {999};
  NavierStokesSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  EXPECT_FALSE(problem.has_robin_boundary());
  const auto pressure_dof = problem.velocity_block_size();
  if (problem.locally_relevant_dofs().is_element(pressure_dof))
    {
      EXPECT_TRUE(problem.constraints().is_constrained(pressure_dof));
    }
}
