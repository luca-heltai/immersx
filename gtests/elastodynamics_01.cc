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
#include <immersx/physics/elastodynamics.h>

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
    par.output_frequency   = 0;
    par.initial_refinement = 1;
    par.time_step          = 1.e-2;
    par.final_time         = 1.e-2;
    par.number_of_steps    = 1;
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
      subsection Time integration
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
  EXPECT_EQ(parameters.number_of_steps, 1u);
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


TEST(Elastodynamics, NontrivialTransient)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_small_problem(parameters);
  parameters.number_of_steps = 1;
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
  parameters.initial_refinement = 0;
  parameters.number_of_steps    = 1;
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
  parameters.initial_refinement = 1;
  parameters.number_of_steps    = 1;
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
