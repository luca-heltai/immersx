// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/reduced_poisson.h>

#include <filesystem>

#include "test_paths.h"
#ifdef DEAL_II_WITH_VTK
#  include <vtkDataArray.h>
#  include <vtkNew.h>
#  include <vtkPointData.h>
#  include <vtkUnstructuredGrid.h>
#  include <vtkXMLUnstructuredGridReader.h>
using namespace dealii;
using namespace ImmersX;
namespace
{
  template <int spacedim, int reduced_dim = 1>
  vtkIdType
  check_reduced_quadratic(const bool simplex, const bool refine = false)
  {
    ParameterAcceptor::clear();
    constexpr int inclusion_dimension = reduced_dim == 0 ? spacedim : 2;
    ReducedPoissonParameters<spacedim, reduced_dim, inclusion_dimension>
                      parameters;
    const std::string kind = reduced_dim == 0 ? "point" : "line";
    initialize_parameters(TestPaths::parameter_path(
      "gtests/parameters/reduced_poisson_simplex_" + kind + "_" +
      std::to_string(spacedim) + "d.prm"));
    if (!simplex)
      {
        parameters.domain_parameters.triangulation_type = "distributed";
        parameters.name_of_grid       = "subdivided_hyper_cube";
        parameters.arguments_for_grid = "2:0:1:false";
      }
    if (refine)
      {
        auto &refinement =
          parameters.reduced_coupling_parameters.refinement_parameters;
        refinement.max_refinement_level = 1;
        if constexpr (reduced_dim > 0)
          {
            refinement.space_pre_refinement_cycles    = 1;
            refinement.embedded_pre_refinement_cycles = 1;
          }
      }
    parameters.output_directory = TestPaths::output_directory(
      "reduced-simplex-" + kind + std::to_string(spacedim) +
      (simplex ? "p" : "q") + (refine ? "refined" : "coarse"));
    std::filesystem::create_directories(parameters.output_directory);
    ReducedPoisson<spacedim, spacedim, reduced_dim, inclusion_dimension>
      problem(parameters);
    problem.run();
    if constexpr (reduced_dim == 0)
      {
        EXPECT_EQ(problem.n_reduced_dofs(), 2u);
      }
    else
      {
        EXPECT_EQ(problem.n_reduced_dofs(), refine ? 5u : 3u);
      }
    EXPECT_GT(problem.coupling_matrix_frobenius_norm(), 0.);
    EXPECT_LT(problem.multiplier_solution_l2_norm(), 1.e-8);
    const auto snapshot =
      parameters.output_directory + "/" + problem.output_solution();
    MPI_Barrier(MPI_COMM_WORLD);
    vtkIdType cells = 0;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      [&] {
        vtkNew<vtkXMLUnstructuredGridReader> reader;
        reader->SetFileName(snapshot.c_str());
        reader->Update();
        auto *grid = reader->GetOutput();
        cells      = grid->GetNumberOfCells();
        ASSERT_GT(cells, 0);
        auto *values = grid->GetPointData()->GetArray("solution");
        ASSERT_NE(values, nullptr);
        bool has_interior = false;
        for (vtkIdType i = 0; i < grid->GetNumberOfPoints(); ++i)
          {
            double point[3];
            grid->GetPoint(i, point);
            bool interior = true;
            for (unsigned int d = 0; d < spacedim; ++d)
              interior = interior && point[d] > 0. && point[d] < 1.;
            has_interior = has_interior || interior;
            // Compare the physical polynomial in the Float32 output format.
            const double expected =
              static_cast<float>(1. + point[0] * point[0]);
            EXPECT_NEAR(values->GetComponent(i, 0), expected, 1.e-11);
          }
        EXPECT_TRUE(has_interior);
      }();
    MPI_Barrier(MPI_COMM_WORLD);
    return cells;
  }
} // namespace
TEST(ReducedPoissonValidation, BOTH_SimplexLineQuadraticSolution)
{
  check_reduced_quadratic<2>(true);
  check_reduced_quadratic<3>(true);
}
TEST(ReducedPoissonValidation, BOTH_HypercubeLineQuadraticSolution)
{
  check_reduced_quadratic<2>(false);
  check_reduced_quadratic<3>(false);
}
TEST(ReducedPoissonValidation, BOTH_SimplexPointQuadraticSolution)
{
  check_reduced_quadratic<2, 0>(true);
  check_reduced_quadratic<3, 0>(true);
}
TEST(ReducedPoissonValidation, BOTH_HypercubePointQuadraticSolution)
{
  check_reduced_quadratic<2, 0>(false);
  check_reduced_quadratic<3, 0>(false);
}
TEST(ReducedPoissonValidation, BOTH_SimplexLinePreparation)
{
  const auto coarse2  = check_reduced_quadratic<2>(true);
  const auto refined2 = check_reduced_quadratic<2>(true, true);
  const auto coarse3  = check_reduced_quadratic<3>(true);
  const auto refined3 = check_reduced_quadratic<3>(true, true);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      EXPECT_GT(refined2, coarse2);
      EXPECT_GT(refined3, coarse3);
    }
}
TEST(ReducedPoissonValidation, BOTH_SimplexPointPreparation)
{
  const auto coarse2  = check_reduced_quadratic<2, 0>(true);
  const auto refined2 = check_reduced_quadratic<2, 0>(true, true);
  const auto coarse3  = check_reduced_quadratic<3, 0>(true);
  const auto refined3 = check_reduced_quadratic<3, 0>(true, true);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      EXPECT_GT(refined2, coarse2);
      EXPECT_GT(refined3, coarse3);
    }
}
#endif
