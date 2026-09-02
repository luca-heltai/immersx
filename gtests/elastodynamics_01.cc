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
#include <immersx/physics/elastodynamics.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>

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
    par.time_parameters.output_frequency = 0;
    par.initial_refinement               = 1;
    par.time_parameters.time_step        = 1.e-2;
    par.time_parameters.final_time       = 1.e-2;
    par.time_parameters.number_of_steps  = 1;
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
      subsection Time parameters
        set Initial time       = 0.0
        set Final time         = 0.01
        set Time step          = 0.01
        set Number of time steps = 1
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
  EXPECT_EQ(parameters.time_parameters.number_of_steps, 1u);
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


TEST(Elastodynamics, ZeroSolutionPreservation)
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
  TimeParameters time_parameters;
  time_parameters.initial_time = 0.;
  time_parameters.final_time   = 0.01;
  Adapter    ida(time_parameters,
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
  zero_constrained_entries<2>(problem.constraints(), expected_displacement);

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
  zero_constrained_entries<2>(problem.velocity_constraints(),
                              expected_velocity);

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
  zero_constrained_entries<2>(problem.constraints(),
                              expected_displacement_action);

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
  zero_constrained_entries<2>(problem.velocity_constraints(),
                              expected_velocity_action);

  difference = ida.field(action, fields.fields().displacement);
  difference -= expected_displacement_action;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-11);
  difference = ida.field(action, fields.fields().velocity);
  difference -= expected_velocity_action;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-11);
}
#endif

TEST(Elastodynamics, NontrivialTransient)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_small_problem(parameters);
  parameters.time_parameters.number_of_steps = 1;
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


TEST(Elastodynamics, ThreeDimensionalSmoke)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<3> parameters;
  configure_small_problem(parameters);
  parameters.initial_refinement              = 0;
  parameters.time_parameters.number_of_steps = 1;
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


TEST(Elastodynamics, MPI_Transient)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_small_problem(parameters);
  parameters.initial_refinement              = 1;
  parameters.time_parameters.number_of_steps = 1;
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
