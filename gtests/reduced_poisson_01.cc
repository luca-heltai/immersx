// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "reduced_poisson.h"

using namespace dealii;

TEST(ReducedPoisson, MPI_OneCylinder) // NOLINT
{
  ParameterAcceptor::clear();
  ReducedPoissonParameters<3> par;
  ParameterAcceptor::initialize(
    SOURCE_DIR "/data/tests/reduced_poisson_01_one_cylinder.prm");

  par.reduced_coupling_parameters.reduced_grid_name =
    SOURCE_DIR "/data/tests/reduced_poisson_01_one_cylinder.vtk";
  par.output_directory = SOURCE_DIR "/data/tests/tests_results";
  par.output_name      = "reduced_poisson_01_one_cylinder";

  ReducedPoisson<3> problem(par);
  problem.run();
}