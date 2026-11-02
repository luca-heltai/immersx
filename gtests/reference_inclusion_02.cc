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
#include <immersx/coupling/inclusions.h>

using namespace ImmersX;
#include <immersx/coupling/reference_cross_section.h> // Add include for ReferenceCrossSection


using namespace dealii;

TEST(Inclusion2, CheckComponents) // NOLINT
{
  parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(2);

  Inclusions<2> ref(2);
  ref.set_n_q_points(4);
  ref.set_n_coefficients(2);
  ref.set_fourier_coefficients({{2, 5}});
  ref.inclusions.push_back({{0, 0, .5}});
  ref.initialize();
  ref.setup_inclusions_particles(tria);
  const auto N = ref.n_dofs();

  std::vector<unsigned int> exact_component({0, 1});

  ASSERT_NEAR(exact_component.size(), N, 0.1);

  for (unsigned int i = 0; i < N; ++i)
    ASSERT_NEAR(ref.get_component(i), exact_component[i], 0.1);
}

TEST(Inclusion3, CheckComponents) // NOLINT
{
  parallel::distributed::Triangulation<3> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(2);

  // cx, cy, cz, dx, dy, dz, r
  std::vector<double> inc1({{0, 0, 0, 0, 0, 1, .5, 0}});
  std::vector<double> inc2({{0, 0, 0.1, 0, 0, 1, .5, 0}});
  Inclusions<3>       ref(3);
  ref.set_n_q_points(4);
  ref.set_n_coefficients(2);
  ref.set_fourier_coefficients({{3, 7}});
  ref.inclusions.push_back(inc1);
  ref.inclusions.push_back(inc2);
  ref.initialize();
  ref.setup_inclusions_particles(tria);
  const auto N = ref.n_dofs();

  std::vector<unsigned int> exact_component({0, 1, 0, 1});

  ASSERT_NEAR(exact_component.size(), N, 0.1);

  for (unsigned int i = 0; i < N; ++i)
    ASSERT_NEAR(ref.get_component(i), exact_component[i], 0.1);
}
namespace
{
  // One inclusion along z, in the format cx, cy, cz, dx, dy, dz, r, vessel id.
  void
  setup_inclusions(Inclusions<3>                   &inclusions,
                   const unsigned int               n_coefficients,
                   const std::vector<unsigned int> &selection,
                   const bool                       with_inclusion = true)
  {
    inclusions.set_n_q_points(4);
    inclusions.set_n_coefficients(n_coefficients);
    inclusions.set_fourier_coefficients(selection);
    if (with_inclusion)
      inclusions.inclusions.push_back({0, 0, 0, 0, 0, 1, .5, 0});
  }
} // namespace

TEST(Inclusion3, SelectionSetsNumberOfCoefficients) // NOLINT
{
  Inclusions<3> inclusions(3);
  setup_inclusions(inclusions, 0, {3, 7});
  inclusions.initialize();
  EXPECT_EQ(inclusions.n_dofs_per_inclusion(), 2u);
  EXPECT_EQ(inclusions.get_n_coefficients(), 2u);
  EXPECT_EQ(inclusions.get_selected_coefficients(),
            (std::vector<unsigned int>{3, 7}));
}

TEST(Inclusion3, SelectionMatchesNumberOfCoefficients) // NOLINT
{
  Inclusions<3> inclusions(3);
  setup_inclusions(inclusions, 2, {3, 7});
  inclusions.initialize();
  EXPECT_EQ(inclusions.n_dofs_per_inclusion(), 2u);
  EXPECT_EQ(inclusions.get_n_coefficients(), 2u);
}

TEST(Inclusion3, SelectionMismatchThrows) // NOLINT
{
  Inclusions<3> inclusions(3);
  setup_inclusions(inclusions, 3, {3, 7});
  EXPECT_THROW(inclusions.initialize(), ExceptionBase);
}

TEST(Inclusion3, NoSelectionUsesAllComponentsOfFirstModes) // NOLINT
{
  Inclusions<3> inclusions(3);
  setup_inclusions(inclusions, 2, {});
  inclusions.initialize();
  EXPECT_EQ(inclusions.n_dofs_per_inclusion(), 6u);
  EXPECT_EQ(inclusions.get_n_coefficients(), 2u);
  EXPECT_EQ(inclusions.get_selected_coefficients(),
            (std::vector<unsigned int>{0, 1, 2, 3, 4, 5}));
}

TEST(Inclusion3, NoSelectionNoCoefficientsGivesNoDofs) // NOLINT
{
  Inclusions<3> inclusions(3);
  setup_inclusions(inclusions, 0, {});
  testing::internal::CaptureStdout();
  inclusions.initialize();
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(inclusions.n_dofs_per_inclusion(), 0u);
  EXPECT_EQ(inclusions.n_dofs(), 0u);
  EXPECT_NE(output.find("no Fourier coefficients are selected"),
            std::string::npos);
}

TEST(Inclusion3, NoInclusionsNoCoefficientsIsSilent) // NOLINT
{
  Inclusions<3> inclusions(3);
  setup_inclusions(inclusions, 0, {}, false);
  testing::internal::CaptureStdout();
  inclusions.initialize();
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(inclusions.n_dofs_per_inclusion(), 0u);
  EXPECT_TRUE(output.empty());
}
