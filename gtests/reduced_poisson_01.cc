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

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include <filesystem>

#include "test_paths.h"

#ifdef DEAL_II_WITH_VTK

#  include <immersx/physics/reduced_poisson.h>

using namespace ImmersX;
#  include <immersx/io/utils.h>

using namespace dealii;

TEST(ReducedPoisson, MPI_OneCylinder) // NOLINT
{
  ParameterAcceptor::clear();
  ReducedPoissonParameters<3> par;
  initialize_parameters(ImmersX::TestPaths::data_filename(
                          "tests/reduced_poisson_01_one_cylinder.prm"),
                        ImmersX::TestPaths::output_directory(
                          "reduced-poisson/one-cylinder.prm"));

  par.reduced_coupling_parameters.tensor_product_space_parameters
    .reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
  par.output_directory =
    ImmersX::TestPaths::output_directory("reduced-poisson/one-cylinder");
  par.output_name = "reduced_poisson_01_one_cylinder";
  std::filesystem::create_directories(par.output_directory);

  ReducedPoisson<3> problem(par);
  problem.run();
}


TEST(ReducedPoisson, MPI_OneCylinderP1) // NOLINT
{
  ParameterAcceptor::clear();
  ReducedPoissonParameters<3> par;
  initialize_parameters(ImmersX::TestPaths::data_filename(
    "tests/reduced_poisson_01_one_cylinder_p1.prm"));

  par.reduced_coupling_parameters.tensor_product_space_parameters
    .reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
  par.output_directory =
    ImmersX::TestPaths::output_directory("reduced-poisson/one-cylinder-p1");
  par.output_name = "reduced_poisson_01_one_cylinder_p1";
  std::filesystem::create_directories(par.output_directory);
  par.reduced_coupling_parameters.tensor_product_space_parameters.section
    .inclusion_degree = 1;

  ReducedPoisson<3> problem(par);
  problem.run();
}

#endif // DEAL_II_WITH_VTK
