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

#include "inclusions.h"
#include "reference_cross_section.h" // Add include for ReferenceCrossSection


using namespace dealii;

TEST(Inclusion2, CheckComponents) // NOLINT
{
  parallel::distributed::Triangulation<2> tria(
    MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(2);

  Inclusions<2>       ref(2);
  ref.set_n_q_points(4);
  ref.set_n_coefficients(2);
  ref.set_fourier_coefficients({{2,5}});
  ref.inclusions.push_back({{0, 0, .5}});
  ref.initialize();
  ref.setup_inclusions_particles(tria);
  const auto N = ref.n_dofs();

  std::vector<unsigned int> exact_component({0,1});

  ASSERT_NEAR(exact_component.size(), N, 0.1);

  for (unsigned int i = 0; i < N; ++i)
  {
    ASSERT_NEAR(ref.get_component(i), exact_component[i], 0.1);
  }
}

TEST(Inclusion3, CheckComponents) // NOLINT
{
  parallel::distributed::Triangulation<3> tria(
    MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(2);

  // cx, cy, cz, dx, dy, dz, r
  std::vector<double> inc1({{0, 0, 0, 0, 0, 1, .5, 0}});
  std::vector<double> inc2({{0, 0, 0.1, 0, 0, 1, .5, 0}});
  Inclusions<3>       ref(3);
  ref.set_n_q_points(4);
  ref.set_n_coefficients(2);
  ref.set_fourier_coefficients({{3,7}});
  ref.inclusions.push_back(inc1);
  ref.inclusions.push_back(inc2);
  ref.initialize();
  ref.setup_inclusions_particles(tria);
  const auto N = ref.n_dofs();

  std::vector<unsigned int> exact_component({0,1, 0,1});

  ASSERT_NEAR(exact_component.size(), N, 0.1);

  for (unsigned int i = 0; i < N; ++i)
  {
    ASSERT_NEAR(ref.get_component(i), exact_component[i], 0.1);
  }
}