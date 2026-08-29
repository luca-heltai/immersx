#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>
#include <immersx/physics/navier_stokes.h>
#include <immersx/physics/navier_stokes_semidiscrete.h>

#include <cmath>

#include "test_paths.h"

#ifdef DEAL_II_WITH_SUNDIALS

TEST(ContributorPhysics, ElastodynamicsCanPopulateIDAAdapter)
{
  dealii::ParameterAcceptor::clear();
  ImmersX::ElastodynamicsParameters<2> parameters;
  parameters.output_directory =
    ImmersX::TestPaths::output_directory("contributor-physics");
  parameters.output_frequency   = 0;
  parameters.initial_refinement = 0;
  parameters.number_of_steps    = 1;
  parameters.time_step          = 0.01;
  parameters.final_time         = 0.01;
  ImmersX::initialize_parameters();
  dealii::ParameterAcceptor::parse_all_parameters();

  ImmersX::ElastodynamicsSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();

  using FieldVector  = ImmersX::ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter      = ImmersX::IDAAdapter<FieldVector, GlobalVector>;
  Adapter::AdditionalData data;
  data.initial_time = 0.;
  data.final_time   = 0.01;
  data.ic_type      = Adapter::AdditionalData::none;
  Adapter    adapter(data,
                  MPI_COMM_WORLD,
                  [](const dealii::LinearOperator<GlobalVector> &,
                     const GlobalVector &,
                     GlobalVector &,
                     double) {
                    FAIL() << "The test only checks composition.";
                  });
  const auto fields = adapter.add(problem, "solid");
  EXPECT_TRUE(fields.fields().displacement.is_valid());
  EXPECT_TRUE(fields.fields().velocity.is_valid());

  GlobalVector state;
  GlobalVector state_dot;
  adapter.reinit(state);
  adapter.reinit(state_dot);
  GlobalVector residual;
  adapter.solver().reinit_vector(residual);
  EXPECT_TRUE(adapter.has_local_preconditioner(fields.fields().displacement));
  EXPECT_TRUE(adapter.has_local_preconditioner(fields.fields().velocity));
  EXPECT_TRUE(adapter.can_materialize_matrix(state, state_dot, 1.));
  const auto materialized = adapter.monolithic_matrix(state, state_dot, 1.);
  EXPECT_EQ(materialized.m(), state.size());
  EXPECT_EQ(materialized.n(), state.size());
  adapter.solver().residual(0., state, state_dot, residual);
  EXPECT_TRUE(std::isfinite(residual.l2_norm()));
  adapter.solver().setup_jacobian(0., state, state_dot, 1.);
  GlobalVector action;
  adapter.solver().reinit_vector(action);
  adapter.current_jacobian().vmult(action, state);
  EXPECT_TRUE(std::isfinite(action.l2_norm()));
}

TEST(ContributorPhysics, StokesCanPopulateIDAAdapter)
{
  dealii::ParameterAcceptor::clear();
  ImmersX::NavierStokesParameters<2> parameters;
  parameters.output_directory =
    ImmersX::TestPaths::output_directory("contributor-stokes");
  parameters.output_frequency        = 0;
  parameters.initial_refinement      = 0;
  parameters.include_convective_term = false;
  ImmersX::initialize_parameters();
  dealii::ParameterAcceptor::parse_all_parameters();

  ImmersX::NavierStokesSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_system();

  using FieldVector  = ImmersX::ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter      = ImmersX::IDAAdapter<FieldVector, GlobalVector>;
  Adapter::AdditionalData data;
  data.initial_time = 0.;
  data.final_time   = 0.01;
  Adapter    adapter(data,
                  MPI_COMM_WORLD,
                  [](const dealii::LinearOperator<GlobalVector> &,
                     const GlobalVector &,
                     GlobalVector &,
                     double) {});
  const auto fields = adapter.add(problem, "fluid");
  EXPECT_TRUE(fields.fields().velocity.is_valid());
  EXPECT_TRUE(fields.fields().pressure.is_valid());
  EXPECT_TRUE(adapter.has_local_preconditioner(fields.fields().velocity));
  EXPECT_TRUE(adapter.has_local_preconditioner(fields.fields().pressure));
  auto state     = adapter.make_state();
  auto state_dot = adapter.make_state();
  EXPECT_TRUE(adapter.can_materialize_matrix(state, state_dot, 1.));
  const auto materialized = adapter.monolithic_matrix(state, state_dot, 1.);
  EXPECT_EQ(materialized.m(), state.size());
  EXPECT_EQ(materialized.n(), state.size());
}

#else

TEST(ContributorPhysics, ElastodynamicsCanPopulateIDAAdapter)
{
  GTEST_SKIP() << "deal.II was configured without SUNDIALS support.";
}

TEST(ContributorPhysics, StokesCanPopulateIDAAdapter)
{
  GTEST_SKIP() << "deal.II was configured without SUNDIALS support.";
}

#endif
