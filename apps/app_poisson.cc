// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the LICENSE.md file at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <iostream>
#include <string>

#include "poisson.h"
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

      if (dimensions.dimension == 1 && dimensions.space_dimension == 3)
        {
          PoissonParameters<1, 3> par;
          initialize_parameters(prm_file);
          PoissonSolver<1, 3> problem(par);
          problem.run();
        }
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 2)
        {
          PoissonParameters<1, 2> par;
          initialize_parameters(prm_file);
          PoissonSolver<1, 2> problem(par);
          problem.run();
        }
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 1)
        {
          PoissonParameters<1> par;
          initialize_parameters(prm_file);
          PoissonSolver<1> problem(par);
          problem.run();
        }
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 3)
        {
          PoissonParameters<2, 3> par;
          initialize_parameters(prm_file);
          PoissonSolver<2, 3> problem(par);
          problem.run();
        }
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        {
          PoissonParameters<3> par;
          initialize_parameters(prm_file);
          PoissonSolver<3> problem(par);
          problem.run();
        }
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 2)
        {
          PoissonParameters<2> par;
          initialize_parameters(prm_file);
          PoissonSolver<2> problem(par);
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
