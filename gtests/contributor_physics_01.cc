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
  Adapter::Parameters adapter_parameters;
  auto               &data = adapter_parameters.data;
  data.initial_time        = 0.;
  data.final_time          = 0.01;
  data.ic_type             = Adapter::AdditionalData::none;
  Adapter    adapter(adapter_parameters,
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
  Adapter::Parameters adapter_parameters;
  auto               &data = adapter_parameters.data;
  data.initial_time        = 0.;
  data.final_time          = 0.01;
  Adapter    adapter(adapter_parameters,
                  MPI_COMM_WORLD,
                  [](const dealii::LinearOperator<GlobalVector> &,
                     const GlobalVector &,
                     GlobalVector &,
                     double) {});
  const auto fields = adapter.add(problem, "fluid");
  EXPECT_TRUE(fields.fields().velocity.is_valid());
  EXPECT_TRUE(fields.fields().pressure.is_valid());
}

TEST(ContributorPhysics, IDAAdapterUsesAdditionalDataParameters)
{
  dealii::ParameterAcceptor::clear();

  using FieldVector  = ImmersX::ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter      = ImmersX::IDAAdapter<FieldVector, GlobalVector>;

  Adapter::Parameters adapter_parameters("IDA adapter parameters");
  Adapter             adapter(adapter_parameters,
                  MPI_COMM_WORLD,
                  Adapter::LinearSolveFunction{});

  ImmersX::initialize_parameters_from_string(R"(
    subsection IDA adapter parameters
      set Initial time = 0.25
      set Final time = 2.5
      set Time interval between each output = 0.125
      subsection Running parameters
        set Initial step size = 0.01
        set Minimum step size = 1.e-8
        set Maximum order of BDF = 2
        set Maximum number of nonlinear iterations = 17
      end
      subsection Error control
        set Absolute error tolerance = 1.e-7
        set Relative error tolerance = 2.e-6
        set Ignore algebraic terms for error computations = false
      end
      subsection Initial condition correction parameters
        set Correction type at initial time = use_y_dot
        set Correction type after restart = none
        set Maximum number of nonlinear iterations = 13
        set Factor to use when converting from the integrator tolerance to the linear solver tolerance = 3.5
      end
    end
  )");

  const auto &data = adapter.additional_data();
  EXPECT_DOUBLE_EQ(data.initial_time, 0.25);
  EXPECT_DOUBLE_EQ(data.final_time, 2.5);
  EXPECT_DOUBLE_EQ(data.output_period, 0.125);
  EXPECT_DOUBLE_EQ(data.initial_step_size, 0.01);
  EXPECT_DOUBLE_EQ(data.minimum_step_size, 1.e-8);
  EXPECT_EQ(data.maximum_order, 2u);
  EXPECT_EQ(data.maximum_non_linear_iterations, 17u);
  EXPECT_DOUBLE_EQ(data.absolute_tolerance, 1.e-7);
  EXPECT_DOUBLE_EQ(data.relative_tolerance, 2.e-6);
  EXPECT_FALSE(data.ignore_algebraic_terms_for_errors);
  EXPECT_EQ(data.ic_type, Adapter::AdditionalData::use_y_dot);
  EXPECT_EQ(data.reset_type, Adapter::AdditionalData::none);
  EXPECT_EQ(data.maximum_non_linear_iterations_ic, 13u);
  EXPECT_DOUBLE_EQ(data.ls_norm_factor, 3.5);
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
