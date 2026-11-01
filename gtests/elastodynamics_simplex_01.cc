// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

// ElastodynamicsSolver on a tetrahedral Gmsh mesh and on a hexahedral mesh.
//
// The exact solution u = (1 + t + t^2) (A x + b) is linear in space and
// quadratic in time. Its stress is constant, so the body force is rho u'' =
// 2 rho (A x + b), and the time-dependent Dirichlet data is u itself. P1
// elements represent u exactly and the trapezoidal rule integrates its time
// dependence exactly, so the discrete displacement and velocity must match
// the interpolated exact fields up to the linear solver tolerance.

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics.h>

#include <algorithm>
#include <cmath>
#include <string>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  const std::string linear_field =
    "0.1*x+0.2*y+0.01; 0.3*y-0.1*z-0.02; 0.2*x+0.1*z+0.03";

  std::string
  scaled(const std::string &factor)
  {
    return "(" + factor + ")*(0.1*x+0.2*y+0.01); (" + factor +
           ")*(0.3*y-0.1*z-0.02); (" + factor + ")*(0.2*x+0.1*z+0.03)";
  }

  double
  max_difference_to(const ElastodynamicsSolver<3>             &problem,
                    const ElastodynamicsSolver<3>::VectorType &computed,
                    const std::string                         &expression)
  {
    FunctionParser<3> exact(3);
    exact.initialize("x,y,z,t", expression, {}, true);
    exact.set_time(problem.current_time());

    ElastodynamicsSolver<3>::VectorType expected;
    expected.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    VectorTools::interpolate(problem.mapping(),
                             problem.dof_handler(),
                             exact,
                             expected);

    double difference = 0.;
    for (const auto index : problem.locally_owned_dofs())
      difference =
        std::max(difference, std::abs(computed(index) - expected(index)));
    return Utilities::MPI::max(difference, MPI_COMM_WORLD);
  }

  void
  run_linear_field(ElastodynamicsParameters<3> &parameters,
                   const std::string           &grid_generator,
                   const std::string           &grid_arguments,
                   const std::string           &triangulation_type,
                   const std::string           &output_name)
  {
    initialize_parameters_from_string(
      R"(
      subsection Elastodynamics
        set FE degree = 1
        set Number of refinement cycles = 1
        set Dirichlet boundary ids = 0
        set Output name = )" +
      output_name + R"(
        set Output directory = )" +
      TestPaths::output_directory("elastodynamics_simplex_01/" + output_name) +
      R"(
        subsection Domain
          set Initial refinement = 0
          set Grid generator = )" +
      grid_generator +
      R"(
          set Grid generator arguments = )" +
      grid_arguments + R"(
          set Triangulation type = )" +
      triangulation_type + R"(
        end
        subsection Material properties
          set Material tags by material id =
          subsection default
            set Density = 1
            set Lame mu = 1
            set Lame lambda = 2
            set Rayleigh alpha = 0
            set Rayleigh beta  = 0
            set Viscosity eta  = 0
          end
        end
        subsection Time interval
          set Initial time = 0
          set Final time = 0.02
          set Output time interval = 0.02
        end
        subsection Fixed step
          set Time step = 0.01
          set Number of time steps = 2
          set Policy = number_of_steps
          set Strategy = trapezoidal
        end
        subsection Functions
          subsection Body force
            set Function expression = )" +
      scaled("2") + R"(
            set Variable names = x,y,z,t
          end
          subsection Displacement boundary
            set Function expression = )" +
      scaled("1+t+t^2") + R"(
            set Variable names = x,y,z,t
          end
          subsection Initial displacement
            set Function expression = )" +
      linear_field + R"(
            set Variable names = x,y,z,t
          end
          subsection Initial velocity
            set Function expression = )" +
      linear_field + R"(
            set Variable names = x,y,z,t
          end
          subsection Exact solution
            set Function expression = )" +
      scaled("1+t+t^2") + R"(
            set Variable names = x,y,z,t
          end
        end
        subsection Solver
          subsection Control
            set Max steps = 1000
            set Reduction = 1.e-14
            set Tolerance = 1.e-14
          end
        end
      end
    )");
  }

  void
  expect_exact_linear_field(ElastodynamicsSolver<3> &problem)
  {
    problem.run();
    EXPECT_NEAR(problem.current_time(), 0.02, 1.e-14);
    EXPECT_LT(max_difference_to(problem,
                                problem.displacement(),
                                scaled("1+t+t^2")),
              1.e-10);
    EXPECT_LT(max_difference_to(problem, problem.velocity(), scaled("1+2*t")),
              1.e-10);
  }
} // namespace


TEST(Elastodynamics, BOTH_LinearFieldIsExactOnSimplexMesh)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<3> parameters;
  run_linear_field(parameters,
                   TestPaths::data_filename("tests/simplex_cube.msh"),
                   "",
                   "fullydistributed",
                   "simplex");

  ElastodynamicsSolver<3> problem(parameters);
  expect_exact_linear_field(problem);
  EXPECT_TRUE(problem.triangulation().all_reference_cells_are_simplex());
  EXPECT_EQ(problem.triangulation().n_global_active_cells(), 48u);

  // ParsedConvergenceTable cannot integrate errors on simplex meshes.
  EXPECT_ANY_THROW(problem.compute_error());
}


TEST(Elastodynamics, BOTH_LinearFieldIsExactOnHypercubeMesh)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<3> parameters;
  run_linear_field(parameters,
                   "subdivided_hyper_cube",
                   "2: 0: 1: false",
                   "distributed",
                   "hypercube");

  ElastodynamicsSolver<3> problem(parameters);
  expect_exact_linear_field(problem);
  EXPECT_TRUE(problem.triangulation().all_reference_cells_are_hyper_cube());
  EXPECT_NO_THROW(problem.compute_error());
}
