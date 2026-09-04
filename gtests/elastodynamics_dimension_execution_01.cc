// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics.h>

#ifdef DEAL_II_WITH_SUNDIALS
#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/physics/elastodynamics_semidiscrete.h>
#endif

#include <cmath>
#include <filesystem>

#include "test_paths.h"

namespace
{
  template <int dim, int spacedim>
  void
  configure_parameters(
    ImmersX::ElastodynamicsParameters<dim, spacedim> &parameters,
    const unsigned int                                refinement = 0)
  {
    parameters.output_directory =
      ImmersX::TestPaths::output_directory("elastodynamics-dimension");
    parameters.output_name         = "elastodynamics_dimension_test";
    parameters.initial_refinement  = refinement;
    parameters.n_refinement_cycles = 1;
    parameters.dirichlet_ids.clear();
    for (unsigned int boundary = 0; boundary < 2 * dim; ++boundary)
      parameters.dirichlet_ids.insert(boundary);

    parameters.time_parameters.output_frequency = 0;
    parameters.time_parameters.initial_time     = 0.;
    parameters.time_parameters.final_time       = 1.e-3;
    parameters.time_parameters.time_step        = 1.e-3;
    parameters.time_parameters.number_of_steps  = 1;
    parameters.solver_control.set_max_steps(500);
    parameters.solver_control.set_reduction(1.e-10);
    parameters.solver_control.set_tolerance(1.e-11);
#ifdef DEAL_II_WITH_SUNDIALS
    parameters.time_parameters.initial_step_size  = 1.e-4;
    parameters.time_parameters.absolute_tolerance = 1.e-8;
    parameters.time_parameters.relative_tolerance = 1.e-8;
    parameters.time_parameters.maximum_order      = 1;
#endif
  }

  template <int dim, int spacedim>
  void
  initialize_problem(ImmersX::ElastodynamicsSolver<dim, spacedim> &problem)
  {
    ImmersX::initialize_parameters();
    dealii::ParameterAcceptor::parse_all_parameters();
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_operators();
    problem.set_initial_conditions();
  }

  template <int dim, int spacedim>
  void
  check_native_case(const unsigned int refinement = 0)
  {
    dealii::ParameterAcceptor::clear();
    ImmersX::ElastodynamicsParameters<dim, spacedim> parameters;
    configure_parameters(parameters, refinement);
    ImmersX::ElastodynamicsSolver<dim, spacedim> problem(parameters);
    initialize_problem(problem);

    EXPECT_GT(problem.n_dofs(), 0u);
    EXPECT_EQ(problem.mass_matrix().m(), problem.n_dofs());
    EXPECT_EQ(problem.mass_matrix().n(), problem.n_dofs());

    typename ImmersX::ElastodynamicsSolver<dim, spacedim>::VectorType vector;
    typename ImmersX::ElastodynamicsSolver<dim, spacedim>::VectorType action;
    vector.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    action.reinit(vector);
    for (const auto index : vector.locally_owned_elements())
      vector(index) = static_cast<double>(index + 1);
    problem.mass_matrix().vmult(action, vector);
    EXPECT_GT(action.l2_norm(), 0.);
    problem.stiffness_matrix().vmult(action, vector);
    EXPECT_GT(action.l2_norm(), 0.);

    typename ImmersX::ElastodynamicsSolver<dim, spacedim>::VectorType
      acceleration;
    problem.initial_acceleration(acceleration);
    EXPECT_TRUE(std::isfinite(acceleration.l2_norm()));
    EXPECT_TRUE(problem.state_is_finite());

    problem.advance_one_timestep();
    EXPECT_EQ(problem.time_step_number(), 1u);
    EXPECT_DOUBLE_EQ(problem.current_time(),
                     parameters.time_parameters.final_time);
    EXPECT_TRUE(problem.state_is_finite());
    EXPECT_NEAR(problem.displacement().l2_norm(), 0., 1.e-12);
    EXPECT_NEAR(problem.velocity().l2_norm(), 0., 1.e-12);
    EXPECT_TRUE(std::isfinite(problem.displacement().l2_norm()));
    EXPECT_TRUE(std::isfinite(problem.velocity().l2_norm()));
  }
} // namespace

TEST(ElastodynamicsExecution, NativeRunStandalone)
{
  dealii::ParameterAcceptor::clear();
  ImmersX::ElastodynamicsParameters<1, 3> parameters;
  configure_parameters(parameters);
  parameters.output_name = "elastodynamics_native_run";
  ImmersX::ElastodynamicsSolver<1, 3> problem(parameters);
  ImmersX::initialize_parameters();
  dealii::ParameterAcceptor::parse_all_parameters();

  problem.run();

  EXPECT_DOUBLE_EQ(problem.current_time(),
                   parameters.time_parameters.final_time);
  EXPECT_EQ(problem.time_step_number(), 1u);
  EXPECT_TRUE(problem.state_is_finite());
  EXPECT_TRUE(std::filesystem::exists(ImmersX::TestPaths::output_path(
    "elastodynamics-dimension/elastodynamics_native_run.pvd")));
}

TEST(ElastodynamicsDimension, OneOneNative)
{
  check_native_case<1, 1>();
}

TEST(ElastodynamicsDimension, OneTwoNative)
{
  check_native_case<1, 2>();
}

TEST(ElastodynamicsDimension, OneThreeNative)
{
  check_native_case<1, 3>();
}

TEST(ElastodynamicsDimension, TwoTwoNative)
{
  check_native_case<2, 2>();
}

TEST(ElastodynamicsDimension, TwoThreeNative)
{
  check_native_case<2, 3>();
}

TEST(ElastodynamicsDimension, ThreeThreeNative)
{
  check_native_case<3, 3>();
}

TEST(ElastodynamicsDimension, MPI_EmbeddedLineNative)
{
  check_native_case<1, 3>(2);
}

#ifdef DEAL_II_WITH_SUNDIALS
namespace
{
  template <int dim, int spacedim>
  void
  check_ida_case()
  {
    dealii::ParameterAcceptor::clear();
    ImmersX::ElastodynamicsParameters<dim, spacedim> parameters;
    configure_parameters(parameters);
    ImmersX::ElastodynamicsSolver<dim, spacedim> problem(parameters);
    initialize_problem(problem);

    using FieldVector =
      typename ImmersX::ElastodynamicsSolver<dim, spacedim>::VectorType;
    using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
    using Adapter      = ImmersX::IDAAdapter<FieldVector, GlobalVector>;

    Adapter    adapter(parameters.time_parameters, MPI_COMM_WORLD);
    const auto fields = adapter.add(problem, "elastodynamics");
    EXPECT_TRUE(fields.fields().displacement.is_valid());
    EXPECT_TRUE(fields.fields().velocity.is_valid());

    auto state                                         = adapter.make_state();
    auto state_dot                                     = adapter.make_state();
    adapter.field(state, fields.fields().displacement) = problem.displacement();
    adapter.field(state, fields.fields().velocity)     = problem.velocity();
    adapter.field(state_dot, fields.fields().displacement) = problem.velocity();

    FieldVector acceleration;
    problem.initial_acceleration(acceleration);
    adapter.field(state_dot, fields.fields().velocity) = acceleration;

    bool accepted_output = false;
    adapter.set_output_step([&problem, &adapter, fields, &accepted_output](
                              const double        time,
                              const GlobalVector &accepted_state,
                              const GlobalVector &,
                              const unsigned int step) {
      accepted_output = true;
      problem.accept_state(
        adapter.field(accepted_state, fields.fields().displacement),
        adapter.field(accepted_state, fields.fields().velocity),
        time,
        step);
    });

    adapter.solve(state, state_dot);
    EXPECT_TRUE(accepted_output);
    EXPECT_NEAR(problem.current_time(),
                parameters.time_parameters.final_time,
                1.e-10);
    EXPECT_TRUE(problem.state_is_finite());
    EXPECT_NEAR(problem.displacement().l2_norm(), 0., 1.e-7);
    EXPECT_NEAR(problem.velocity().l2_norm(), 0., 1.e-7);

    GlobalVector residual;
    adapter.solver().reinit_vector(residual);
    adapter.solver().residual(problem.current_time(),
                              state,
                              state_dot,
                              residual);
    EXPECT_LT(residual.l2_norm(), 1.e-7);

    adapter.solver().setup_jacobian(problem.current_time(),
                                    state,
                                    state_dot,
                                    1.);
    GlobalVector action;
    adapter.solver().reinit_vector(action);
    adapter.current_jacobian().vmult(action, state);
    EXPECT_TRUE(std::isfinite(action.l2_norm()));
  }
} // namespace

TEST(ElastodynamicsExecution, IDA_OneOne)
{
  check_ida_case<1, 1>();
}

TEST(ElastodynamicsExecution, IDA_OneTwo)
{
  check_ida_case<1, 2>();
}

TEST(ElastodynamicsExecution, IDA_OneThree)
{
  check_ida_case<1, 3>();
}

TEST(ElastodynamicsExecution, IDA_TwoTwo)
{
  check_ida_case<2, 2>();
}

TEST(ElastodynamicsExecution, IDA_TwoThree)
{
  check_ida_case<2, 3>();
}

TEST(ElastodynamicsExecution, IDA_ThreeThree)
{
  check_ida_case<3, 3>();
}

TEST(ElastodynamicsExecution, MPI_IDA_EmbeddedLine)
{
  check_ida_case<1, 3>();
}
#else
TEST(ElastodynamicsExecution, IDA_DimensionsUnavailable)
{
  GTEST_SKIP() << "deal.II was configured without SUNDIALS support.";
}
#endif
