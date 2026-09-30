// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/core/state.h>
#include <immersx/io/output_handler.h>
#include <immersx/physics/navier_stokes.h>
#include <immersx/physics/navier_stokes_semidiscrete.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "test_paths.h"

namespace
{
  using namespace dealii;
  using FieldVector = ImmersX::ImmersXLA::MPI::Vector;

  std::string
  output_text(const std::string &directory)
  {
    std::string result;
    for (const auto &entry : std::filesystem::directory_iterator(directory))
      if (entry.path().extension() == ".vtu" ||
          entry.path().extension() == ".pvtu")
        {
          std::ifstream input(entry.path());
          result.append(std::istreambuf_iterator<char>(input), {});
        }
    return result;
  }

  void
  configure_navier_stokes_parameters(
    ImmersX::NavierStokesParameters<2> &parameters)
  {
    const auto variables =
      dealii::FunctionParser<2>::default_variable_names() + ",t";
    parameters.rhs.initialize(variables, "0; 0; 0");
    parameters.bc.initialize(variables, "0; 0; 0");
    parameters.initial_condition.initialize(variables, "0; 0; 0");
    parameters.domain_parameters.initial_refinement = 0;
    parameters.include_convective_term              = false;
    parameters.dirichlet_ids                        = {0};
  }
} // namespace

TEST(OutputHandler, BOTH_ScalarField)
{
  ImmersX::OutputHandlerParameters parameters("/Output scalar/");
  parameters.output_directory =
    ImmersX::TestPaths::output_directory("output-handler-scalar");

  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  FE_Q<2>       finite_element(1);
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(finite_element);
  AffineConstraints<double> constraints;
  constraints.close();
  const auto space =
    ImmersX::fe_space(dof_handler,
                      StaticMappingQ1<2>::mapping,
                      constraints,
                      DoFTools::extract_locally_relevant_dofs(dof_handler));

  ImmersX::StateLayout layout;
  const auto           temperature = space.field(layout, "temperature");
  FieldVector          values(space.locally_owned_dofs(), MPI_COMM_WORLD);
  values = 3.;
  ImmersX::StateView<FieldVector> state(layout, 0.);
  state.bind(temperature.id(), values);

  ImmersX::OutputHandler<2, 2, FieldVector> output(space, parameters, "scalar");
  output.add_field(temperature);
  ASSERT_NO_THROW(output.write(state, 0.));
  MPI_Barrier(MPI_COMM_WORLD);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      const auto directory = std::filesystem::path(parameters.output_directory);
      EXPECT_TRUE(std::filesystem::exists(directory / "scalar.pvd"));
      bool has_vtu = false;
      for (const auto &entry : std::filesystem::directory_iterator(directory))
        has_vtu = has_vtu || entry.path().extension() == ".vtu";
      EXPECT_TRUE(has_vtu);
      EXPECT_NE(output_text(parameters.output_directory).find("temperature"),
                std::string::npos);
    }
}

TEST(OutputHandler, BOTH_ReindexedNavierStokesFields)
{
  dealii::ParameterAcceptor::clear();
  ImmersX::NavierStokesParameters<2> parameters;
  configure_navier_stokes_parameters(parameters);
  ImmersX::NavierStokesSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();

  const auto space = std::make_shared<ImmersX::FiniteElementSpaceView<2, 2>>(
    problem.dof_handler(),
    problem.mapping(),
    problem.constraints(),
    &problem.locally_relevant_dofs());

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor velocity_descriptor;
  velocity_descriptor.name          = "velocity";
  velocity_descriptor.locally_owned = problem.locally_owned_dofs_by_block()[0];
  velocity_descriptor.locally_relevant =
    problem.locally_relevant_dofs_by_block()[0];
  const auto velocity_id = layout.add_field(std::move(velocity_descriptor));

  ImmersX::FieldDescriptor pressure_descriptor;
  pressure_descriptor.name          = "pressure";
  pressure_descriptor.locally_owned = problem.locally_owned_dofs_by_block()[1];
  pressure_descriptor.locally_relevant =
    problem.locally_relevant_dofs_by_block()[1];
  const auto pressure_id = layout.add_field(std::move(pressure_descriptor));

  const auto velocity = ImmersX::navier_stokes_detail::make_block_field(
    *space,
    velocity_id,
    "velocity",
    problem.velocity_extractor(),
    0,
    problem.velocity_block_size(),
    problem.locally_owned_dofs_by_block()[0],
    problem.locally_relevant_dofs_by_block()[0],
    problem.constraints());
  const auto pressure = ImmersX::navier_stokes_detail::make_block_field(
    *space,
    pressure_id,
    "pressure",
    problem.pressure_extractor(),
    problem.velocity_block_size(),
    problem.locally_owned_dofs_by_block()[1].size(),
    problem.locally_owned_dofs_by_block()[1],
    problem.locally_relevant_dofs_by_block()[1],
    problem.constraints());

  ASSERT_TRUE(velocity.is_reindexed());
  ASSERT_TRUE(pressure.is_reindexed());
  EXPECT_LT(velocity.locally_owned_dofs().size(),
            problem.dof_handler().n_dofs());
  EXPECT_LT(pressure.locally_owned_dofs().size(),
            problem.dof_handler().n_dofs());

  FieldVector velocity_values(velocity.locally_owned_dofs(), MPI_COMM_WORLD);
  FieldVector pressure_values(pressure.locally_owned_dofs(), MPI_COMM_WORLD);
  velocity_values = 2.;
  pressure_values = -5.;
  ImmersX::StateView<FieldVector> state(layout, 0.);
  state.bind(velocity.id(), velocity_values);
  state.bind(pressure.id(), pressure_values);

  ImmersX::OutputHandlerParameters output_parameters("/Output Stokes/");
  output_parameters.output_directory =
    ImmersX::TestPaths::output_directory("output-handler-stokes");
  ImmersX::OutputHandler<2, 2, FieldVector> output(*space,
                                                   output_parameters,
                                                   "stokes");
  output.add_field(velocity);
  output.add_field(pressure);
  ASSERT_NO_THROW(output.write(state, 0.));
  MPI_Barrier(MPI_COMM_WORLD);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      const auto directory =
        std::filesystem::path(output_parameters.output_directory);
      EXPECT_TRUE(std::filesystem::exists(directory / "stokes.pvd"));
      EXPECT_NE(
        output_text(output_parameters.output_directory).find("velocity"),
        std::string::npos);
      EXPECT_NE(
        output_text(output_parameters.output_directory).find("pressure"),
        std::string::npos);
    }
}
