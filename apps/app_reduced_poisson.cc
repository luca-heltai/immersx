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
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 * Modified by: Luca Heltai, 2020
 */

#include <deal.II/base/config.h>

#include <iostream>

#ifdef DEAL_II_WITH_VTK

#  include "../tests/tests.h"
#  include "reduced_poisson.h"
#  include "utils.h"
int
main(int argc, char *argv[])
{
  using namespace dealii;

  // deallog.depth_console(10);
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      mpi_initlog(true);
      const std::string prm_file   = argc > 1 ? argv[1] : "parameters.prm";
      const auto        dimensions = get_dimension_parameters(prm_file);

      if (dimensions.dimension == 2 && dimensions.space_dimension == 2 &&
          dimensions.reduced_dimension == 0)
        {
          ReducedPoissonParameters<2, 0> par;
          initialize_parameters(prm_file);
          ReducedPoisson<2, 2, 0> problem(par);
          problem.run();
        }
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 2 &&
               dimensions.reduced_dimension == 1)
        {
          ReducedPoissonParameters<2, 1> par;
          initialize_parameters(prm_file);
          ReducedPoisson<2, 2, 1> problem(par);
          problem.run();
        }
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3 &&
               dimensions.reduced_dimension == 1)
        {
          ReducedPoissonParameters<3, 1> par;
          initialize_parameters(prm_file);
          ReducedPoisson<3, 3, 1> problem(par);
          problem.run();
        }
      else
        throw_unsupported_dimension_combination(dimensions);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}

#else
int
main()
{
  std::cerr
    << "app_reduced_poisson requires deal.II to be configured with VTK support "
       "(DEAL_II_WITH_VTK)."
    << std::endl;
  return 0;
}
#endif // DEAL_II_WITH_VTK
