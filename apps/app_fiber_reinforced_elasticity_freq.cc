// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on the deal.II library.
// ---------------------------------------------------------------------
#include <deal.II/base/mpi.h>

#include <immersx/io/utils.h>
#include <immersx/physics/fiber_reinforced_elasticity_freq.h>

#include <iostream>
#include <string>

namespace
{
  template <int dim, int spacedim>
  void
  run(const std::string &filename)
  {
    ImmersX::FiberReinforcedElasticityFreqParameters<dim, spacedim> parameters;
    ImmersX::initialize_parameters(filename);
    ImmersX::FiberReinforcedElasticityFreq<dim, spacedim> driver(parameters);
    driver.run();
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
      const std::string filename = argc > 1 ? argv[1] : "parameters.prm";
      const auto dimensions      = ImmersX::get_dimension_parameters(filename);
      if (dimensions.dimension == 1 && dimensions.space_dimension == 1)
        run<1, 1>(filename);
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 2)
        run<1, 2>(filename);
      else if (dimensions.dimension == 1 && dimensions.space_dimension == 3)
        run<1, 3>(filename);
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 2)
        run<2, 2>(filename);
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 3)
        run<2, 3>(filename);
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        run<3, 3>(filename);
      else
        ImmersX::throw_unsupported_dimension_combination(dimensions);
    }
  catch (const std::exception &exception)
    {
      std::cerr << "Exception on processing: " << exception.what() << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << "Unknown exception!" << std::endl;
      return 1;
    }
  return 0;
}
