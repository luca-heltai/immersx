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

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/core/constraint.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>

#include <cmath>
#include <filesystem>
#include <fstream>
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
    PoissonParameters<1, 2> embedded_parameters("/Embedded Poisson/");
    LinearSolverParameters  adapter_parameters;
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

    using FieldVector  = ImmersXLA::MPI::Vector;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

    Adapter    adapter(adapter_parameters, MPI_COMM_WORLD);
    const auto bulk     = adapter.add(bulk_problem, "bulk");
    const auto embedded = adapter.add(embedded_problem, "embedded");

    const auto bulk_view = fe_space(bulk_problem.dof_handler(),
                                    StaticMappingQ1<2>::mapping,
                                    bulk_problem.constraints(),
                                    bulk_problem.locally_relevant_dofs());
    const auto embedded_view =
      fe_space(embedded_problem.dof_handler(),
               StaticMappingQ1<1, 2>::mapping,
               embedded_problem.constraints(),
               embedded_problem.locally_relevant_dofs());

    // The multiplier has its own FE/DoFHandler, while sharing the embedded
    // geometry with the line problem.  This deliberately exercises the
    // paired-DoFHandler weak-term path rather than particle search.
    FE_DGQ<1, 2>     multiplier_fe(0);
    DoFHandler<1, 2> multiplier_dh(embedded_problem.triangulation());
    multiplier_dh.distribute_dofs(multiplier_fe);
    const auto multiplier_owned = multiplier_dh.locally_owned_dofs();
    const auto multiplier_relevant =
      DoFTools::extract_locally_relevant_dofs(multiplier_dh);
    AffineConstraints<double> multiplier_constraints;
    multiplier_constraints.reinit(multiplier_owned, multiplier_relevant);
    multiplier_constraints.close();

    const auto multiplier_view = fe_space(multiplier_dh,
                                          StaticMappingQ1<1, 2>::mapping,
                                          multiplier_constraints,
                                          multiplier_relevant);
    const auto bulk_field =
      bulk_view.field(bulk.fields().solution, "bulk_solution");
    const auto embedded_field =
      embedded_view.field(embedded.fields().solution, "embedded_solution");
    const auto multiplier = multiplier_view.field("lambda");
    const auto constraint =
      make_constraint(weak_term(value(bulk_field), multiplier) -
                      weak_term(value(embedded_field), multiplier));
    const auto coupling = adapter.add(constraint, "continuity");

    auto state = adapter.make_state();
    adapter.solve(state);

    bulk_problem.set_solution(adapter.field(state, bulk.fields().solution));
    embedded_problem.set_solution(
      adapter.field(state, embedded.fields().solution));

    bulk_problem.output_results();
    embedded_problem.output_results();

    std::filesystem::create_directories(
      application_parameters.output_directory);
    dealii::DataOut<1, 2> multiplier_output;
    multiplier_output.attach_dof_handler(multiplier_dh);
    multiplier_output.add_data_vector(
      adapter.field(state, coupling.fields().multiplier),
      application_parameters.multiplier_output_name,
      dealii::DataOut<1, 2>::type_dof_data);
    multiplier_output.build_patches();
    const auto rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto multiplier_filename =
      std::filesystem::path(application_parameters.output_directory) /
      (application_parameters.multiplier_output_name + "-0." +
       std::to_string(rank) + ".vtu");
    std::ofstream multiplier_file(multiplier_filename);
    multiplier_output.write_vtu(multiplier_file);

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
