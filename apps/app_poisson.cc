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

      const std::string prm_file = argc > 1 ? argv[1] : "parameters.prm";

      if (prm_file.find("13d") != std::string::npos)
        {
          PoissonParameters<1, 3> par;
          PoissonSolver<1, 3>     problem(par);
          initialize_parameters(prm_file);
          problem.run();
        }
      else if (prm_file.find("12d") != std::string::npos)
        {
          PoissonParameters<1, 2> par;
          PoissonSolver<1, 2>     problem(par);
          initialize_parameters(prm_file);
          problem.run();
        }
      else if (prm_file.find("1d") != std::string::npos)
        {
          PoissonParameters<1> par;
          PoissonSolver<1>     problem(par);
          initialize_parameters(prm_file);
          problem.run();
        }
      else if (prm_file.find("23d") != std::string::npos)
        {
          PoissonParameters<2, 3> par;
          PoissonSolver<2, 3>     problem(par);
          initialize_parameters(prm_file);
          problem.run();
        }
      else if (prm_file.find("3d") != std::string::npos)
        {
          PoissonParameters<3> par;
          PoissonSolver<3>     problem(par);
          initialize_parameters(prm_file);
          problem.run();
        }
      else
        {
          PoissonParameters<2> par;
          PoissonSolver<2>     problem(par);
          initialize_parameters(prm_file);
          problem.run();
        }
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
