// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_parser.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/navier_stokes.h>
#ifdef DEAL_II_WITH_SUNDIALS
#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/physics/navier_stokes_semidiscrete.h>
#endif
#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;
namespace
{
  template <int dim>
  void
  check_poiseuille(const bool simplex, const bool use_ida = false)
  {
    ParameterAcceptor::clear();
    NavierStokesParameters<dim> parameters;
    const std::string           variables = dim == 2 ? "x,y,t" : "x,y,z,t";
    const std::string exact = dim == 2 ? "y*(1-y);0;0" : "y*(1-y);0;0;0";
    const std::string force = dim == 2 ? "2;0;0" : "2;0;0;0";
    initialize_parameters_from_string(
      "subsection Navier-Stokes\n"
      " set Output directory = " +
      TestPaths::output_directory("navier-simplex-" + std::to_string(dim) +
                                  (simplex ? "p" : "q") +
                                  (use_ida ? "ida" : "native")) +
      "\n"
      " set Dirichlet boundary ids = 0\n"
      " subsection Finite element spaces\n set Velocity degree = 2\n set Pressure degree = 1\nend\n"
      " subsection Domain\n set Triangulation type = fullydistributed\n set Initial refinement = 0\n"
      " set Grid generator = " +
      (simplex ?
         TestPaths::data_filename(dim == 2 ? "tests/simplex_square.msh" :
                                             "tests/simplex_cube.msh") :
         "subdivided_hyper_cube") +
      "\n"
      " set Grid generator arguments = 2:0:1:false\n end\n"
      " subsection Physical properties\n set Density = 1\n set Viscosity = 1\n set Include convective term = true\nend\n"
      " subsection Time interval\n set Initial time = 0\n set Final time = 0.02\n set Output time interval = 0.02\nend\n"
      " subsection Fixed step\n set Policy = number_of_steps\n set Number of time steps = 2\nend\n"
      " subsection Right hand side\n set Variable names = " +
      variables + "\n set Function expression = " + force +
      "\nend\n"
      " subsection Dirichlet boundary conditions\n set Variable names = " +
      variables + "\n set Function expression = " + exact +
      "\nend\n"
      " subsection Initial condition\n set Variable names = " +
      variables + "\n set Function expression = " + exact +
      "\nend\n"
      " subsection Solver\n subsection Control\n set Max steps = 1000\n set Tolerance = 1.e-13\n set Reduction = 1.e-13\nend\n set Inner maximum steps = 1000\n set Inner tolerance = 1.e-13\nend\nend\n");
    NavierStokesSolver<dim> problem(parameters);
    if (use_ida)
      {
#ifdef DEAL_II_WITH_SUNDIALS
        problem.make_grid();
        problem.setup_fe();
        problem.setup_system();
        problem.assemble_system();
        parameters.ida_parameters.initial_step_size               = .005;
        parameters.ida_parameters.absolute_tolerance              = 1.e-10;
        parameters.ida_parameters.relative_tolerance              = 1.e-10;
        parameters.ida_parameters.correction_type_at_initial_time = "none";
        using FieldVector  = typename NavierStokesSolver<dim>::VectorType;
        using GlobalVector = ImmersXLA::MPI::BlockVector;
        IDAAdapter<FieldVector, GlobalVector> adapter(
          parameters.time_parameters,
          parameters.ida_parameters,
          MPI_COMM_WORLD);
        const auto fields    = adapter.add(problem, "fluid");
        auto       state     = adapter.make_state();
        auto       state_dot = adapter.make_state();
        adapter.field(state, fields.fields().velocity) =
          problem.solution().block(0);
        adapter.field(state, fields.fields().pressure) =
          problem.solution().block(1);
        adapter.field(state_dot, fields.fields().velocity) = 0.;
        adapter.field(state_dot, fields.fields().pressure) = 0.;
        auto residual = adapter.make_state();
        adapter.solver().residual(0., state, state_dot, residual);
        EXPECT_LT(residual.l2_norm(), 1.e-11)
          << "The manufactured initial state must satisfy the DAE.";
        adapter.solve(state, state_dot);
        problem.accept_state(adapter.field(state, fields.fields().velocity),
                             adapter.field(state, fields.fields().pressure),
                             parameters.time_parameters.final_time,
                             1);
#endif
      }
    else
      problem.run();
    EXPECT_EQ(problem.finite_element().reference_cell().is_simplex(), simplex);
    EXPECT_NEAR(problem.current_time(),
                parameters.time_parameters.final_time,
                1.e-14);
    EXPECT_LT(problem.divergence_l2_norm(), 1.e-10);
    FunctionParser<dim> function(dim + 1);
    function.initialize(variables, exact, {}, true);
    auto expected = problem.solution();
    VectorTools::interpolate(problem.mapping(),
                             problem.dof_handler(),
                             function,
                             expected);
    expected -= problem.solution();
    EXPECT_LT(expected.linfty_norm(), 1.e-9);
  }
} // namespace
TEST(NavierStokes, BOTH_SimplexPoiseuille)
{
  check_poiseuille<2>(true);
  check_poiseuille<3>(true);
}
TEST(NavierStokes, BOTH_HypercubePoiseuille)
{
  check_poiseuille<2>(false);
  check_poiseuille<3>(false);
}
#ifdef DEAL_II_WITH_SUNDIALS
TEST(NavierStokes, BOTH_HypercubePoiseuilleIDA)
{
  check_poiseuille<2>(false, true);
  check_poiseuille<3>(false, true);
}
TEST(NavierStokes, BOTH_SimplexPoiseuilleIDA)
{
  check_poiseuille<2>(true, true);
  check_poiseuille<3>(true, true);
}
#endif
