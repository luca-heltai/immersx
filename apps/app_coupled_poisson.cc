// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <immersx/algebra/lagrange_multiplier_interaction.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/core/representation.h>
#include <immersx/coupling/particle_coupling.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

namespace
{
  class CoupledPoissonApplicationParameters : public dealii::ParameterAcceptor
  {
  public:
    CoupledPoissonApplicationParameters()
      : ParameterAcceptor("/Coupled Poisson/")
    {
      add_parameter("Output directory", output_directory);
      add_parameter("Multiplier output name", multiplier_output_name);
    }

    std::string output_directory       = ".";
    std::string multiplier_output_name = "multiplier";
  };

  void
  run_coupled_poisson(const std::string &parameter_file)
  {
    using namespace ImmersX;

    CoupledPoissonApplicationParameters application_parameters;
    PoissonParameters<2>                bulk_parameters;
    PoissonParameters<1, 2>       embedded_parameters("/Embedded Poisson/");
    ParticleCouplingParameters<2> search_parameters("/Particle coupling/");
    LinearSolverOptions           adapter_parameters;
    initialize_parameters(parameter_file);

    bulk_parameters.output_directory = application_parameters.output_directory;
    embedded_parameters.output_directory =
      application_parameters.output_directory;

    PoissonSolver<2>    bulk_problem(bulk_parameters);
    PoissonSolver<1, 2> embedded_problem(embedded_parameters);

    const auto initialize_problem = [](auto &problem) {
      problem.make_grid();
      problem.setup_fe();
      problem.setup_system();
      problem.assemble_system();
    };
    initialize_problem(bulk_problem);
    initialize_problem(embedded_problem);

    IdentityRepresentation<2, 2> bulk_representation(
      bulk_problem.triangulation(),
      bulk_problem.dof_handler(),
      bulk_problem.locally_owned_dofs(),
      bulk_problem.locally_relevant_dofs(),
      bulk_problem.constraints());
    IdentityRepresentation<1, 2> embedded_representation(
      embedded_problem.triangulation(),
      embedded_problem.dof_handler(),
      embedded_problem.locally_owned_dofs(),
      embedded_problem.locally_relevant_dofs(),
      embedded_problem.constraints());

    LagrangeMultiplierInteraction<IdentityRepresentation<2, 2>,
                                  IdentityRepresentation<1, 2>>
      interaction(bulk_representation,
                  embedded_representation,
                  search_parameters);
    interaction.assemble();

    using FieldVector  = ImmersXLA::MPI::Vector;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

    Adapter    adapter(adapter_parameters, MPI_COMM_WORLD);
    const auto bulk     = adapter.add(bulk_problem, "bulk");
    const auto embedded = adapter.add(embedded_problem, "embedded");
    const auto coupling = adapter.add(interaction,
                                      "continuity",
                                      bulk.fields().solution,
                                      embedded.fields().solution);

    auto state = adapter.make_state();
    adapter.solve(state);

    bulk_problem.set_solution(adapter.field(state, bulk.fields().solution));
    embedded_problem.set_solution(
      adapter.field(state, embedded.fields().solution));
    interaction.set_multiplier(
      adapter.field(state, coupling.fields().multiplier));

    bulk_problem.output_results();
    embedded_problem.output_results();
    interaction.output_results(application_parameters.output_directory,
                               application_parameters.multiplier_output_name,
                               0);

    auto residual = adapter.make_state();
    adapter.evaluate_residual(state, residual);
    const double residual_norm = residual.l2_norm();
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "coupled_residual = " << residual_norm << '\n';

    AssertThrow(std::isfinite(residual_norm) && residual_norm < 1.e-7,
                dealii::ExcMessage(
                  "The coupled Poisson residual is too large."));
  }
} // namespace

int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                                  argv,
                                                                  1);
      const std::string parameter_file = argc > 1 ? argv[1] : "parameters.prm";
      const auto dimensions = ImmersX::get_dimension_parameters(parameter_file);

      if (dimensions.dimension != 2 || dimensions.space_dimension != 2 ||
          dimensions.reduced_dimension != 1)
        ImmersX::throw_unsupported_dimension_combination(dimensions);

      run_coupled_poisson(parameter_file);
    }
  catch (const std::exception &exception)
    {
      std::cerr << exception.what() << std::endl;
      return 1;
    }

  return 0;
}
