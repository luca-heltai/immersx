// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <immersx/physics/elastodynamics.h>

#include <iostream>
#include <string>

using namespace ImmersX;
#include <immersx/io/utils.h>

namespace
{
  template <int dim, int spacedim = dim>
  void
  run_elastodynamics(const std::string &parameter_file)
  {
    ElastodynamicsParameters<dim, spacedim> parameters;
    initialize_parameters(parameter_file);

    ElastodynamicsSolver<dim, spacedim> problem(parameters);
    problem.run();
  }
} // namespace

int
main(int argc, char *argv[])
{
  using namespace dealii;

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      const std::string prm_file   = argc > 1 ? argv[1] : "parameters.prm";
      const auto        dimensions = get_dimension_parameters(prm_file);

      if (dimensions.dimension == 1 && dimensions.space_dimension == 1)
        run_elastodynamics<1, 1>(prm_file);
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 2)
        run_elastodynamics<1, 2>(prm_file);
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 3)
        run_elastodynamics<1, 3>(prm_file);
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 2)
        run_elastodynamics<2, 2>(prm_file);
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 3)
        run_elastodynamics<2, 3>(prm_file);
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        run_elastodynamics<3, 3>(prm_file);
      else
        throw_unsupported_dimension_combination(dimensions);
    }
  catch (const std::exception &exc)
    {
      std::cerr << "Exception on processing: " << exc.what() << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << "Unknown exception!" << std::endl;
      return 1;
    }

  return 0;
}
