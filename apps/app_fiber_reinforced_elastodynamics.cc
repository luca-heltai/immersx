// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <immersx/io/utils.h>
#include <immersx/physics/fiber_reinforced_elastodynamics.h>

#include <iostream>
#include <string>

using namespace ImmersX;

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
          FiberReinforcedElastodynamicsParameters<2> parameters;
          FiberReinforcedElastodynamics<2>           driver(parameters);
          initialize_parameters(prm_file);
          driver.run();
        }
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        {
          FiberReinforcedElastodynamicsParameters<3> parameters;
          FiberReinforcedElastodynamics<3>           driver(parameters);
          initialize_parameters(prm_file);
          driver.run();
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
