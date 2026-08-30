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

#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastic_static.h>
#include <immersx/physics/poisson.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <type_traits>

#include "coupled_poisson_elasticity.h"

using namespace ImmersX;

int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                                  argv,
                                                                  1);

      PoissonParameters<1, 3>                poisson_parameters;
      ElasticStaticParameters<3, 3>          elasticity_parameters;
      CoupledPoissonElasticity::PressureLift pressure_lift("/Pressure lift/");
      CoupledPoissonElasticity::Traction     traction("/Pressure traction/");
      const std::string prm_file = argc > 1 ? argv[1] : "parameters.prm";
      initialize_parameters(prm_file);

      PoissonSolver<1, 3> poisson_problem(poisson_parameters);
      poisson_problem.make_grid();
      poisson_problem.setup_fe();
      poisson_problem.setup_system();
      poisson_problem.assemble_system();

      ElasticStaticProblem<3, 3> elasticity_problem(elasticity_parameters);
      elasticity_problem.setup();
      traction.attach(elasticity_problem);

      using FieldVector  = ImmersXLA::MPI::Vector;
      using GlobalVector = ImmersXLA::MPI::BlockVector;
      using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

      Adapter adapter(MPI_COMM_WORLD);

      const auto poisson = adapter.add(poisson_problem);
      const auto elastic = adapter.add(elasticity_problem);
      const auto pressure =
        poisson.observe(CoupledPoissonElasticity::Pressure{})
          .lift(pressure_lift);
      adapter.couple(pressure, elastic, traction);

      auto state = adapter.make_state();
      adapter.solve(state);

      auto residual = adapter.make_state();
      adapter.evaluate_residual(state, residual);

      const auto &poisson_state =
        adapter.field(state, poisson.fields().solution);
      const auto &elastic_state =
        adapter.field(state, elastic.fields().displacement);
      const double pressure_error =
        std::abs(CoupledPoissonElasticity::Pressure{}.factor - 2.);
      const double traction_error = CoupledPoissonElasticity::traction_balance(
        pressure, traction, poisson_state, elastic_state);

      std::cout << "coupled_residual = " << residual.l2_norm() << '\n'
                << "pressure_scale_error = " << pressure_error << '\n'
                << "traction_balance_error = " << traction_error << '\n'
                << "traction_points = " << pressure.lifted_points().size()
                << '\n';

      std::filesystem::create_directories(poisson_parameters.output_directory);
      std::ofstream diagnostics(
        std::filesystem::path(poisson_parameters.output_directory) /
        "coupled_poisson_elasticity_diagnostics.txt");
      diagnostics << "coupled_residual = " << residual.l2_norm() << '\n'
                  << "pressure_scale_error = " << pressure_error << '\n'
                  << "traction_balance_error = " << traction_error << '\n';

      AssertThrow(residual.l2_norm() < 1.e-9,
                  dealii::ExcMessage("Coupled solve residual is too large."));
      AssertThrow(pressure_error < 1.e-14,
                  dealii::ExcMessage("Pressure scaling check failed."));
      AssertThrow(traction_error < 1.e-9,
                  dealii::ExcMessage("Traction balance check failed."));
    }
  catch (const std::exception &exception)
    {
      std::cerr << exception.what() << std::endl;
      return 1;
    }

  return 0;
}
