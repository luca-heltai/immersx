// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/physics/elasticity.h>
#include <immersx/physics/elastodynamics.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

using namespace ImmersX;
#include <immersx/io/utils.h>


using namespace dealii;

namespace
{
  template <int dim>
  void
  configure_small_problem(ElastodynamicsParameters<dim> &par)
  {
    par.output_directory =
      (std::filesystem::temp_directory_path() / "immersx_elastodynamics")
        .string();
    par.time_parameters.output_time_interval  = 1.e-2;
    par.initial_refinement                    = 1;
    par.fixed_step_parameters.time_step       = 1.e-2;
    par.time_parameters.final_time            = 1.e-2;
    par.fixed_step_parameters.number_of_steps = 1;
    par.solver_control.set_reduction(1.e-11);
    par.solver_control.set_tolerance(1.e-12);
  }

  void
  initialize_configured_parameters()
  {
    initialize_parameters();
    ParameterAcceptor::parse_all_parameters();
  }

  template <int dim>
  typename ElastodynamicsSolver<dim>::VectorType
  constant_vector(const ElastodynamicsSolver<dim> &problem, const double value)
  {
    typename ElastodynamicsSolver<dim>::VectorType vector;
    vector.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    vector = value;
    return vector;
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

  template <int dim>
  double
  quadratic_form(const typename ElastodynamicsSolver<dim>::MatrixType &matrix,
                 const typename ElastodynamicsSolver<dim>::VectorType &vector)
  {
    typename ElastodynamicsSolver<dim>::VectorType product;
    product.reinit(vector);
    matrix.vmult(product, vector);
    return vector * product;
  }
} // namespace


TEST(Elastodynamics, ParameterParsing)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;

  initialize_parameters_from_string(R"(
    subsection Elastodynamics
      set FE degree             = 2
      set Initial refinement    = 1
      set Dirichlet boundary ids = 0
      subsection Material
        set Density       = 2.5
        set Lame mu       = 3.0
        set Lame lambda   = 4.0
        set Damping shear = 0.2
        set Damping bulk  = 0.1
      end
      subsection Time interval
        set Initial time       = 0.0
        set Final time         = 0.01
        set Output time interval = 0.01
      end
      subsection Fixed step
        set Time step          = 0.01
        set Number of time steps = 1
        set Policy               = number_of_steps
      end
      subsection Functions
        subsection Body force
          set Function expression = 1; 2
          set Variable names      = x,y,t
        end
      end
    end
  )");

  EXPECT_EQ(parameters.fe_degree, 2u);
  EXPECT_DOUBLE_EQ(parameters.density, 2.5);
  EXPECT_DOUBLE_EQ(parameters.lame_mu, 3.0);
  EXPECT_DOUBLE_EQ(parameters.lame_lambda, 4.0);
  EXPECT_DOUBLE_EQ(parameters.damping_shear, 0.2);
  EXPECT_EQ(parameters.fixed_step_parameters.number_of_steps, 1u);
  EXPECT_DOUBLE_EQ(parameters.body_force.value(Point<2>(), 0), 1.0);
  EXPECT_DOUBLE_EQ(parameters.body_force.value(Point<2>(), 1), 2.0);
}


TEST(Elastodynamics, SetupAndOperatorSymmetry)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_small_problem(parameters);
  initialize_configured_parameters();

  ElastodynamicsSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();

  EXPECT_GT(problem.n_dofs(), 0u);
  EXPECT_EQ(problem.mass_matrix().m(), problem.n_dofs());
  EXPECT_EQ(problem.mass_matrix().n(), problem.n_dofs());
  EXPECT_EQ(problem.stiffness_matrix().m(), problem.n_dofs());
  EXPECT_EQ(problem.stiffness_matrix().n(), problem.n_dofs());
  EXPECT_TRUE(problem.state_is_finite());

  auto x = constant_vector(problem, 1.0);
  typename ElastodynamicsSolver<2>::VectorType y;
  typename ElastodynamicsSolver<2>::VectorType z;
  y.reinit(x);
  z.reinit(x);
  problem.mass_matrix().vmult(y, x);
  problem.mass_matrix().Tvmult(z, x);
  y -= z;
  EXPECT_NEAR(y.l2_norm(), 0.0, 1.e-12);

  y = 0.;
  problem.stiffness_matrix().vmult(y, x);
  EXPECT_NEAR(y.l2_norm(), 0.0, 1.e-11);
}


TEST(ElastodynamicsValidation, ZeroSolutionPreservation)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_small_problem(parameters);
  initialize_configured_parameters();

  ElastodynamicsSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();

  for (unsigned int step = 0; step < 3; ++step)
    {
      problem.advance_one_timestep();
      EXPECT_NEAR(problem.displacement().l2_norm(), 0.0, 1.e-11);
      EXPECT_NEAR(problem.velocity().l2_norm(), 0.0, 1.e-11);
    }
}


TEST(ElastodynamicsValidation, BOTH_MovingBoundaryDerivesVelocityConstraint)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  initialize_parameters_from_string(R"(
    subsection Elastodynamics
      set Initial refinement = 1
      set Dirichlet boundary ids = 0
      subsection Fixed step
        set Time step = 0.01
        set Number of time steps = 1
        set Policy = number_of_steps
      end
      subsection Functions
        subsection Body force
          set Function expression = 0; 0
        end
        subsection Displacement boundary
          set Function expression = t; 0
        end
        subsection Initial displacement
          set Function expression = 0; 0
        end
        subsection Initial velocity
          set Function expression = 1; 0
        end
      end
    end
  )");

  ElastodynamicsSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();

  unsigned int boundary_lines = 0;
  unsigned int unit_lines     = 0;
  for (const auto &line : problem.velocity_constraints().get_lines())
    if (line.entries.empty())
      {
        ++boundary_lines;
        const auto value =
          problem.velocity_constraints().get_inhomogeneity(line.index);
        EXPECT_TRUE(value == 0. || value == 1.);
        unit_lines += value == 1.;
      }
  EXPECT_GT(boundary_lines, 0u);
  EXPECT_GT(unit_lines, 0u);

  problem.advance_one_timestep();

  for (const auto &line : problem.velocity_constraints().get_lines())
    if (line.entries.empty())
      {
        const auto value =
          problem.velocity_constraints().get_inhomogeneity(line.index);
        EXPECT_TRUE(value == 0. || value == 1.);
      }
}


TEST(ElastodynamicsValidation, BOTH_RefinementCyclesAdvanceSequentially)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_small_problem(parameters);
  parameters.initial_refinement  = 0;
  parameters.n_refinement_cycles = 4;
  parameters.output_directory    = (std::filesystem::temp_directory_path() /
                                 "immersx_elastodynamics_refinement_cycles")
                                  .string();
  parameters.output_name                           = "refinement_cycles";
  parameters.time_parameters.output_time_interval  = 1.e-2;
  parameters.fixed_step_parameters.number_of_steps = 1;
  initialize_configured_parameters();

  ElastodynamicsSolver<2> problem(parameters);
  problem.run();

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    for (unsigned int cycle = 0; cycle < parameters.n_refinement_cycles;
         ++cycle)
      EXPECT_TRUE(std::filesystem::exists(
        std::filesystem::path(parameters.output_directory) /
        (parameters.output_name + "_cycle_" + std::to_string(cycle) + ".pvd")));
}


TEST(ElastodynamicsValidation, BOTH_RefineTimeStepPerCycle)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_small_problem(parameters);
  parameters.initial_refinement                     = 0;
  parameters.n_refinement_cycles                    = 2;
  parameters.time_parameters.final_time             = 2.e-2;
  parameters.time_parameters.output_time_interval   = 2.e-2;
  parameters.fixed_step_parameters.time_step        = 1.e-2;
  parameters.fixed_step_parameters.number_of_steps  = 2;
  parameters.fixed_step_parameters.time_step_policy = "number_of_steps";
  parameters.fixed_step_parameters.refine_time_step = true;
  initialize_configured_parameters();

  ElastodynamicsSolver<2> problem(parameters);
  problem.run();

  EXPECT_EQ(problem.time_step_number(), 4u);
  EXPECT_NEAR(problem.time_step(), 5.e-3, 1.e-14);
  EXPECT_NEAR(problem.current_time(), 2.e-2, 1.e-14);
}


TEST(ElastodynamicsValidation, BOTH_TrapezoidalMatchesNewmarkMMS)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2>    trapezoidal_parameters;
  ElasticityProblemParameters<2> newmark_parameters;
  initialize_parameters_from_string(R"(
    subsection Elastodynamics
      set FE degree = 1
      set Initial refinement = 1
      set Number of refinement cycles = 1
      set Dirichlet boundary ids = 0,1,2,3
      subsection Grid generation
        set Grid generator = hyper_cube
        set Grid generator arguments = 0: 1: true
      end
      subsection Material
        set Density = 1
        set Lame mu = 1
        set Lame lambda = 2
        set Damping shear = 0
        set Damping bulk = 0
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
          set Function expression = 4*pi^2*sin(pi*x)*sin(pi*y)*cos(pi*t); -3*pi^2*cos(pi*x)*cos(pi*y)*cos(pi*t)
          set Variable names = x,y,t
        end
        subsection Displacement boundary
          set Function expression = 0; 0
          set Variable names = x,y,t
        end
        subsection Initial displacement
          set Function expression = sin(pi*x)*sin(pi*y); 0
          set Variable names = x,y,t
        end
        subsection Initial velocity
          set Function expression = 0; 0
          set Variable names = x,y,t
        end
        subsection Exact solution
          set Function expression = sin(pi*x)*sin(pi*y)*cos(pi*t); 0
          set Variable names = x,y,t
        end
      end
      subsection Solver
        subsection Control
          set Max steps = 1000
          set Reduction = 1.e-12
          set Tolerance = 1.e-12
        end
      end
    end

    subsection Functions
      subsection Dirichlet boundary conditions
        set Function expression = 0; 0
        set Variable names = x,y,t
      end
      subsection Neumann boundary conditions
        set Function expression = 0; 0
        set Variable names = x,y,t
      end
      subsection Initial displacement
        set Function expression = sin(pi*x)*sin(pi*y); 0
        set Variable names = x,y,t
      end
      subsection Initial velocity
        set Function expression = 0; 0
        set Variable names = x,y,t
      end
      subsection Exact solution
        set Function expression = sin(pi*x)*sin(pi*y)*cos(pi*t); 0
        set Variable names = x,y,t
      end
      subsection Right hand side
        set Function expression = 4*pi^2*sin(pi*x)*sin(pi*y)*cos(pi*t); -3*pi^2*cos(pi*x)*cos(pi*y)*cos(pi*t)
        set Variable names = x,y,t
      end
    end

    subsection Immersed Problem
      set FE degree = 1
      set Initial refinement = 1
      set Dirichlet boundary ids = 0,1,2,3
      set Output results also before solving = false
      subsection Grid generation
        set Domain type = generate
        set Grid generator = hyper_cube
        set Grid generator arguments = 0: 1: true
        set Triangulation type = distributed
      end
      subsection Refinement and remeshing
        set Strategy = global
        set Number of refinement cycles = 1
      end
      subsection Material properties
        subsection default
          set Density = 1
          set Lame lambda = 2
          set Lame mu = 1
          set Rayleigh alpha = 0
          set Rayleigh beta = 0
          set Viscosity eta = 0
        end
      end
      subsection Time parameters
        set Initial time = 0
        set Final time = 0.02
        set Time step = 0.01
        set Newmark beta = 0.25
        set Newmark gamma = 0.5
      end
    end
  )");

  const auto output_root = (std::filesystem::temp_directory_path() /
                            "immersx_trapezoidal_newmark_equivalence")
                             .string();
  trapezoidal_parameters.output_directory = output_root + "/trapezoidal";
  trapezoidal_parameters.output_name      = "trapezoidal";
  newmark_parameters.output_directory     = output_root + "/newmark";
  newmark_parameters.output_name          = "newmark";
  std::filesystem::create_directories(trapezoidal_parameters.output_directory);
  std::filesystem::create_directories(newmark_parameters.output_directory);

  ElastodynamicsSolver<2> trapezoidal_solver(trapezoidal_parameters);
  ElasticityProblem<2>    newmark_solver(newmark_parameters);
  trapezoidal_solver.run();
  newmark_solver.run();

  double displacement_difference = 0.;
  double velocity_difference     = 0.;
  for (const auto index : trapezoidal_solver.locally_owned_dofs())
    {
      displacement_difference =
        std::max(displacement_difference,
                 std::abs(
                   trapezoidal_solver.displacement()(index) -
                   newmark_solver.locally_relevant_solution.block(0)(index)));
      velocity_difference =
        std::max(velocity_difference,
                 std::abs(trapezoidal_solver.velocity()(index) -
                          newmark_solver.velocity.block(0)(index)));
    }

  displacement_difference =
    Utilities::MPI::max(displacement_difference, MPI_COMM_WORLD);
  velocity_difference =
    Utilities::MPI::max(velocity_difference, MPI_COMM_WORLD);

  EXPECT_NEAR(displacement_difference, 0., 1.e-10);
  EXPECT_NEAR(velocity_difference, 0., 1.e-10);
}


TEST(Elastodynamics, FirstOrderResiduals)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_small_problem(parameters);
  initialize_configured_parameters();

  ElastodynamicsSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();

  auto previous_displacement = problem.displacement();
  auto previous_velocity     = problem.velocity();
  problem.advance_one_timestep();

  typename ElastodynamicsSolver<2>::VectorType kinematic_residual;
  kinematic_residual.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  kinematic_residual = problem.displacement();
  kinematic_residual -= previous_displacement;
  kinematic_residual *= 1. / problem.time_step();
  kinematic_residual -= problem.velocity();
  zero_constrained_entries<2>(problem.constraints(), kinematic_residual);
  EXPECT_NEAR(kinematic_residual.l2_norm(), 0.0, 1.e-10);

  typename ElastodynamicsSolver<2>::VectorType dynamic_residual;
  typename ElastodynamicsSolver<2>::VectorType work;
  dynamic_residual.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  work.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  work = problem.velocity();
  work -= previous_velocity;
  work *= 1. / problem.time_step();
  problem.mass_matrix().vmult(dynamic_residual, work);
  problem.stiffness_matrix().vmult(work, problem.displacement());
  dynamic_residual += work;
  problem.damping_matrix().vmult(work, problem.velocity());
  dynamic_residual += work;
  dynamic_residual -= problem.body_force_vector();
  zero_constrained_entries<2>(problem.constraints(), dynamic_residual);
  EXPECT_NEAR(dynamic_residual.l2_norm(), 0.0, 1.e-10);
}


#ifdef DEAL_II_WITH_SUNDIALS
TEST(Elastodynamics, IDAResidualAndJacobianOracle)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_small_problem(parameters);
  initialize_configured_parameters();

  ElastodynamicsSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();

  using FieldVector  = ElastodynamicsSolver<2>::VectorType;
  using GlobalVector = LA::MPI::BlockVector;
  using Adapter      = IDAAdapter<FieldVector, GlobalVector>;
  TimeIntervalParameters time_parameters;
  FixedStepParameters    fixed_step_parameters;
  IDAParameters          ida_parameters;
  time_parameters.initial_time = 0.;
  time_parameters.final_time   = 0.01;
  Adapter    ida(time_parameters,
              ida_parameters,
              MPI_COMM_WORLD,
              [](const dealii::LinearOperator<GlobalVector> &,
                 const GlobalVector &,
                 GlobalVector &,
                 const double) {});
  const auto fields = ida.add(problem, "solid");

  auto state                                         = ida.make_state();
  auto state_dot                                     = ida.make_state();
  auto residual                                      = ida.make_state();
  ida.field(state, fields.fields().displacement)     = 0.25;
  ida.field(state, fields.fields().velocity)         = -0.5;
  ida.field(state_dot, fields.fields().displacement) = 0.75;
  ida.field(state_dot, fields.fields().velocity)     = 1.25;

  ida.solver().residual(0., state, state_dot, residual);
  FieldVector expected_displacement;
  FieldVector expected_velocity;
  FieldVector work;
  expected_displacement.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  expected_velocity.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  work.reinit(expected_displacement);
  problem.mass_matrix().vmult(expected_displacement,
                              ida.field(state_dot,
                                        fields.fields().displacement));
  problem.mass_matrix().vmult(work, ida.field(state, fields.fields().velocity));
  expected_displacement -= work;
  for (const auto index : expected_displacement.locally_owned_elements())
    if (problem.constraints().is_constrained(index))
      expected_displacement(index) =
        ida.field(state, fields.fields().displacement)(index) -
        problem.constraints().get_inhomogeneity(index);

  problem.mass_matrix().vmult(expected_velocity,
                              ida.field(state_dot, fields.fields().velocity));
  problem.stiffness_matrix().vmult(work,
                                   ida.field(state,
                                             fields.fields().displacement));
  expected_velocity += work;
  problem.damping_matrix().vmult(work,
                                 ida.field(state, fields.fields().velocity));
  expected_velocity += work;
  FieldVector force;
  problem.body_force_at_time(0., force);
  expected_velocity -= force;
  for (const auto index : expected_velocity.locally_owned_elements())
    if (problem.velocity_constraints().is_constrained(index))
      expected_velocity(index) =
        ida.field(state, fields.fields().velocity)(index) -
        problem.velocity_constraints().get_inhomogeneity(index);

  auto difference = ida.field(residual, fields.fields().displacement);
  difference -= expected_displacement;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-11);
  difference = ida.field(residual, fields.fields().velocity);
  difference -= expected_velocity;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-11);

  auto increment                                     = ida.make_state();
  ida.field(increment, fields.fields().displacement) = -0.8;
  ida.field(increment, fields.fields().velocity)     = 0.6;
  auto action                                        = ida.make_state();
  ida.solver().setup_jacobian(0., state, state_dot, 2.);
  ida.current_jacobian().vmult(action, increment);

  FieldVector expected_displacement_action;
  FieldVector expected_velocity_action;
  expected_displacement_action.reinit(problem.locally_owned_dofs(),
                                      MPI_COMM_WORLD);
  expected_velocity_action.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  problem.mass_matrix().vmult(expected_displacement_action,
                              ida.field(increment,
                                        fields.fields().displacement));
  expected_displacement_action *= 2.;
  problem.mass_matrix().vmult(work,
                              ida.field(increment, fields.fields().velocity));
  expected_displacement_action -= work;
  for (const auto index : expected_displacement_action.locally_owned_elements())
    if (problem.constraints().is_constrained(index))
      expected_displacement_action(index) =
        ida.field(increment, fields.fields().displacement)(index);

  problem.stiffness_matrix().vmult(expected_velocity_action,
                                   ida.field(increment,
                                             fields.fields().displacement));
  problem.damping_matrix().vmult(work,
                                 ida.field(increment,
                                           fields.fields().velocity));
  expected_velocity_action += work;
  problem.mass_matrix().vmult(work,
                              ida.field(increment, fields.fields().velocity));
  work *= 2.;
  expected_velocity_action += work;
  for (const auto index : expected_velocity_action.locally_owned_elements())
    if (problem.velocity_constraints().is_constrained(index))
      expected_velocity_action(index) =
        ida.field(increment, fields.fields().velocity)(index);

  difference = ida.field(action, fields.fields().displacement);
  difference -= expected_displacement_action;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-11);
  difference = ida.field(action, fields.fields().velocity);
  difference -= expected_velocity_action;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-11);
}
#endif

TEST(ElastodynamicsValidation, NontrivialTransient)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_small_problem(parameters);
  parameters.fixed_step_parameters.number_of_steps = 1;
  initialize_configured_parameters();

  ElastodynamicsSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();

  problem.set_velocity(constant_vector(problem, 1.0));
  const double initial_energy =
    0.5 * quadratic_form<2>(problem.mass_matrix(), problem.velocity());
  for (unsigned int step = 0; step < 1; ++step)
    problem.advance_one_timestep();

  const double final_energy =
    0.5 * quadratic_form<2>(problem.mass_matrix(), problem.velocity()) +
    0.5 * quadratic_form<2>(problem.stiffness_matrix(), problem.displacement());
  EXPECT_TRUE(problem.state_is_finite());
  EXPECT_TRUE(std::isfinite(final_energy));
  EXPECT_LE(final_energy, 1.1 * initial_energy + 1.e-10);
}


TEST(ElastodynamicsValidation, ThreeDimensionalSmoke)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<3> parameters;
  configure_small_problem(parameters);
  parameters.initial_refinement                    = 0;
  parameters.fixed_step_parameters.number_of_steps = 1;
  initialize_configured_parameters();

  ElastodynamicsSolver<3> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();
  problem.advance_one_timestep();

  EXPECT_GT(problem.n_dofs(), 0u);
  EXPECT_TRUE(problem.state_is_finite());
}


TEST(ElastodynamicsValidation, MPI_Transient)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_small_problem(parameters);
  parameters.initial_refinement                    = 1;
  parameters.fixed_step_parameters.number_of_steps = 1;
  initialize_configured_parameters();

  ElastodynamicsSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();
  problem.set_velocity(constant_vector(problem, 0.25));
  problem.advance_one_timestep();

  EXPECT_TRUE(problem.state_is_finite());
  EXPECT_EQ(problem.time_step_number(), 1u);
}
