// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <iostream>
#include <string>

#include "navier_stokes.h"
#include "utils.h"


int
main(int argc, char *argv[])
{
  using namespace dealii;

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      const std::string prm_file   = argc > 1 ? argv[1] : "parameters.prm";
      const auto        dimensions = get_dimension_parameters(prm_file);

      if (dimensions.dimension == 2 && dimensions.space_dimension == 2)
        {
          NavierStokesParameters<2> par;
          NavierStokesSolver<2>     problem(par);
          initialize_parameters(prm_file);
          problem.run();
        }
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        {
          NavierStokesParameters<3> par;
          NavierStokesSolver<3>     problem(par);
          initialize_parameters(prm_file);
          problem.run();
        }
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
