// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics.h>

#include <map>
#include <string>

using namespace dealii;
using namespace ImmersX;

namespace
{
  using Problem    = ElastodynamicsSolver<2>;
  using VectorType = Problem::VectorType;

  void
  configure_reuse_problem(const std::string &additional_parameters)
  {
    initialize_parameters_from_string(R"(
      subsection Elastodynamics
        set Dirichlet boundary ids =
        subsection Domain
          set Grid generator = hyper_cube
          set Grid generator arguments = 0: 1: true
          set Initial refinement = 1
          set Triangulation type = distributed
        end
        subsection Material properties
          subsection default
            set Density = 1
            set Lame mu = 1
            set Lame lambda = 1
            set Rayleigh alpha = 0
            set Rayleigh beta = 0
            set Viscosity eta = 0
          end
        end
        subsection Solver
          subsection Control
            set Max steps = 1000
            set Reduction = 1.e-12
            set Tolerance = 1.e-13
          end
        end
      end
    )" + additional_parameters);
  }

  VectorType
  component_direction(const Problem &problem, const unsigned int component)
  {
    VectorType direction;
    direction.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    VectorTools::interpolate(problem.mapping(),
                             problem.dof_handler(),
                             ComponentSelectFunction<2>(component, 2),
                             direction);
    return direction;
  }

  double
  resultant(const Problem     &problem,
            const VectorType  &force,
            const unsigned int component)
  {
    // A constant test function integrates the volume and boundary loads.
    return component_direction(problem, component) * force;
  }

  std::map<types::global_dof_index, Point<2>>
  support_points(const Problem &problem)
  {
    return DoFTools::map_dofs_to_support_points(problem.mapping(),
                                                problem.dof_handler());
  }

  void
  refine_lower_left(Problem &problem)
  {
    auto &tria =
      const_cast<parallel::TriangulationBase<2> &>(problem.triangulation());
    // Keep the isolated refinement patch: Domain's island smoothing would
    // otherwise remove its refine flag and leave a uniform mesh.
    tria.set_mesh_smoothing(
      Triangulation<2>::limit_level_difference_at_vertices);
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned() && cell->center()[0] < 0.5 &&
          cell->center()[1] < 0.5)
        cell->set_refine_flag();
    tria.execute_coarsening_and_refinement();
  }
} // namespace


TEST(Elastodynamics, BOTH_StaticLoadsSurviveRefinementAndRepeatedAssembly)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_reuse_problem(R"(
    subsection Elastodynamics
      set Dirichlet boundary ids = 0
      set Neumann boundary ids = 1
      subsection Functions
        subsection Body force
          set Function constants = theta=3
          set Function expression = theta*x*x; y*y
        end
        subsection Neumann boundary
          set Function expression = 3; -2
        end
        subsection Displacement boundary
          set Function expression = 1+x; 0
        end
        subsection Velocity boundary
          set Function expression = 2; 3
        end
      end
    end
  )");

  Problem problem(parameters);
  problem.make_grid();
  problem.setup_fe();

  types::global_dof_index previous_dofs = 0;
  for (unsigned int cycle = 0; cycle < 2; ++cycle)
    {
      problem.setup_system();
      problem.assemble_operators();
      EXPECT_EQ(problem.assembly_statistics().spatial_operators, cycle + 1u);
      EXPECT_GT(problem.n_dofs(), previous_dofs);
      previous_dofs = problem.n_dofs();

      // u=(1,0) has unit squared norm and zero strain on the unit square.
      const auto direction = component_direction(problem, 0);
      VectorType product;
      product.reinit(direction);
      for (unsigned int repeat = 0; repeat < 2; ++repeat)
        {
          problem.assemble_operators();
          problem.mass_matrix().vmult(product, direction);
          EXPECT_NEAR(direction * product, 1., 1.e-12);
          problem.stiffness_matrix().vmult(product, direction);
          EXPECT_NEAR(product.l2_norm(), 0., 1.e-12);
          EXPECT_EQ(problem.assembly_statistics().spatial_operators,
                    cycle + 1u);
        }

      VectorType force;
      const auto constraint_updates = problem.assembly_statistics().constraints;
      for (const double time : {0., 0.5, 0.})
        {
          problem.update_constraints(time);
          EXPECT_EQ(problem.assembly_statistics().constraints,
                    constraint_updates);
          problem.body_force_at_time(time, force);
          EXPECT_NEAR(resultant(problem, force, 0), 4., 1.e-12);
          EXPECT_NEAR(resultant(problem, force, 1), -5. / 3., 1.e-12);
          EXPECT_EQ(problem.assembly_statistics().body_force, cycle + 1u);
        }
      if (cycle == 0)
        problem.refine_global();
    }
}


TEST(Elastodynamics, BOTH_TimeDependentLoadsUseConfiguredTimeVariable)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_reuse_problem(R"(
    subsection Elastodynamics
      set Neumann boundary ids = 1
      subsection Functions
        subsection Body force
          set Variable names = x,y,tau
          set Function expression = x+tau; 2*y-3*tau
        end
        subsection Neumann boundary
          set Variable names = x,y,tau
          set Function expression = 2*tau; 3*tau
        end
      end
    end
  )");

  Problem problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();

  VectorType force;
  // Residual evaluations can revisit earlier candidate times after rejection.
  for (const double time : {0., 0.25, 0.25, 0.75, 0.25})
    {
      problem.body_force_at_time(time, force);
      EXPECT_NEAR(resultant(problem, force, 0), 0.5 + 3. * time, 1.e-12);
      EXPECT_NEAR(resultant(problem, force, 1), 1., 1.e-12);
    }
  EXPECT_EQ(problem.assembly_statistics().body_force, 4u);
}


TEST(Elastodynamics, BOTH_TimeDependentConstraintsRefreshAfterInitialState)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_reuse_problem(R"(
    subsection Elastodynamics
      set Dirichlet boundary ids = 0,1,2,3
      subsection Functions
        subsection Displacement boundary
          set Variable names = x,y,tau
          set Function expression = (1+x)*tau; 2*tau
        end
        subsection Velocity boundary
          set Variable names = x,y,tau
          set Function expression = (1+x)*(1+tau); 2+tau
        end
        subsection Initial velocity
          set Function expression = 7; 8
        end
      end
    end
  )");

  Problem problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();
  const auto points      = support_points(problem);
  const auto x_component = component_direction(problem, 0);

  double previous_time = -1.;
  for (const double time : {0., 0.25, 0.25, 0.75, 0.25})
    {
      const auto previous_updates = problem.assembly_statistics().constraints;
      problem.update_constraints(time);
      EXPECT_EQ(problem.assembly_statistics().constraints,
                previous_updates + (time != previous_time ? 1u : 0u));
      previous_time = time;
      for (const auto index : problem.locally_owned_dofs())
        if (problem.constraints().is_constrained(index))
          {
            const bool   is_x  = x_component(index) > 0.5;
            const double scale = 1. + points.at(index)[0];
            EXPECT_NEAR(problem.constraints().get_inhomogeneity(index),
                        is_x ? scale * time : 2. * time,
                        1.e-13);
            EXPECT_NEAR(problem.velocity_constraints().get_inhomogeneity(index),
                        is_x ? scale * (1. + time) : 2. + time,
                        1.e-13);
          }
    }
}


TEST(Elastodynamics, BOTH_TrapezoidalMovingBoundaryWithHangingNodes)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_reuse_problem(R"(
    subsection Elastodynamics
      set Dirichlet boundary ids = 0,1,2,3
      subsection Fixed step
        set Policy = fixed
        set Strategy = trapezoidal
        set Time step = 0.025
      end
      subsection Functions
        subsection Body force
          set Function expression = 2*(1+x); 0
        end
        subsection Displacement boundary
          set Function expression = t*t*(1+x); 0
        end
      end
    end
  )");

  Problem problem(parameters);
  problem.make_grid();
  refine_lower_left(problem);
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();

  AffineConstraints<double> hanging;
  hanging.reinit(problem.locally_owned_dofs(), problem.locally_relevant_dofs());
  DoFTools::make_hanging_node_constraints(problem.dof_handler(), hanging);
  hanging.close();
  EXPECT_GT(Utilities::MPI::sum(hanging.n_constraints(), MPI_COMM_WORLD), 0u);

  const auto points      = support_points(problem);
  const auto x_component = component_direction(problem, 0);
  // Affine space dependence and quadratic time dependence are represented
  // exactly by Q1 and the trapezoidal rule, including hanging-node traces.
  for (unsigned int step = 0; step < 3; ++step)
    {
      if (step == 1)
        parameters.fixed_step_parameters.time_step = 0.01;
      problem.advance_one_timestep();
      const double time = problem.current_time();
      for (const auto index : problem.locally_owned_dofs())
        {
          const double scale = x_component(index) * (1. + points.at(index)[0]);
          EXPECT_NEAR(problem.displacement()(index),
                      time * time * scale,
                      1.e-10);
          EXPECT_NEAR(problem.velocity()(index), 2. * time * scale, 1.e-9);
        }
      EXPECT_EQ(problem.assembly_statistics().time_system_matrix,
                step == 0 ? 1u : 2u);
    }
  EXPECT_EQ(problem.assembly_statistics().spatial_operators, 1u);
  EXPECT_EQ(problem.assembly_statistics().body_force, 1u);
  EXPECT_EQ(problem.assembly_statistics().time_system_matrix, 2u);
}


TEST(Elastodynamics, BOTH_BackwardEulerMovingBoundaryWithHangingNodes)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_reuse_problem(R"(
    subsection Elastodynamics
      set Dirichlet boundary ids = 0,1,2,3
      subsection Material properties
        subsection default
          set Rayleigh alpha = 1
        end
      end
      subsection Fixed step
        set Policy = fixed
        set Strategy = backward_euler
        set Time step = 0.025
      end
      subsection Functions
        subsection Body force
          set Function expression = 1+x; 0
        end
        subsection Displacement boundary
          set Function expression = t*(1+x); 0
        end
        subsection Initial velocity
          set Function expression = 1+x; 0
        end
      end
    end
  )");

  Problem problem(parameters);
  problem.make_grid();
  refine_lower_left(problem);
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();
  AffineConstraints<double> hanging;
  hanging.reinit(problem.locally_owned_dofs(), problem.locally_relevant_dofs());
  DoFTools::make_hanging_node_constraints(problem.dof_handler(), hanging);
  hanging.close();
  EXPECT_GT(Utilities::MPI::sum(hanging.n_constraints(), MPI_COMM_WORLD), 0u);
  const auto points      = support_points(problem);
  const auto x_component = component_direction(problem, 0);

  // u=t*(1+x,0) has zero acceleration and interior elastic force. With
  // Rayleigh alpha=1 its nonzero damping force is exactly (1+x,0).
  for (unsigned int step = 0; step < 3; ++step)
    {
      problem.advance_one_timestep();
      for (const auto index : problem.locally_owned_dofs())
        {
          const double scale = x_component(index) * (1. + points.at(index)[0]);
          EXPECT_NEAR(problem.displacement()(index),
                      problem.current_time() * scale,
                      1.e-10);
          EXPECT_NEAR(problem.velocity()(index), scale, 1.e-9);
        }
    }
  EXPECT_EQ(problem.assembly_statistics().spatial_operators, 1u);
  EXPECT_EQ(problem.assembly_statistics().body_force, 1u);
  EXPECT_EQ(problem.assembly_statistics().time_system_matrix, 1u);
}
