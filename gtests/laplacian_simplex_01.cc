// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/laplacian.h>

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
  template <int dim>
  void
  check_laplacian(const bool simplex)
  {
    ParameterAcceptor::clear();
    ProblemParameters<dim> parameters;
    // The executable constructs the Problem before reading its parameters.
    PoissonProblem<dim> problem(parameters);
    const auto          directory = TestPaths::output_directory(
      "laplacian-simplex-" + std::to_string(dim) + (simplex ? "p" : "q"));
    initialize_parameters_from_string(
      "subsection Immersed Problem\n"
      " set FE degree = 2\n set Dirichlet boundary ids = 0\n"
      " set Output directory = " +
      directory +
      "\n set Output name = quadratic\n"
      " subsection Domain\n set Triangulation type = fullydistributed\n set Initial refinement = 0\n"
      " set Grid generator = " +
      (simplex ?
         TestPaths::data_filename(dim == 2 ? "tests/simplex_square.msh" :
                                             "tests/simplex_cube.msh") :
         "subdivided_hyper_cube") +
      "\n"
      " set Grid generator arguments = 2:0:1:false\n end\n"
      " subsection Right hand side\n set Function expression = " +
      std::to_string(-2 * dim) +
      "\n end\n"
      " subsection Dirichlet boundary conditions\n set Variable names = " +
      std::string(dim == 2 ? "x,y,t" : "x,y,z,t") +
      "\n set Function expression = " +
      std::string(dim == 2 ? "1+x*x+y*y" : "1+x*x+y*y+z*z") +
      "\n end\n"
      " subsection Immersed inclusions\n set Inclusions =\n set Inclusions file =\n subsection Boundary data\n set Function expression = 0\n end\n end\n"
      " subsection Solver\n subsection Inner control\n set Max steps = 1000\n set Tolerance = 1.e-13\n set Reduction = 1.e-13\n end\n subsection Outer control\n set Max steps = 1000\n set Tolerance = 1.e-13\n set Reduction = 1.e-13\n end\nend\nend\n");
    std::filesystem::create_directories(directory);
    problem.run();
    const auto snapshot = directory + "/" + problem.output_solution();
    MPI_Barrier(MPI_COMM_WORLD);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      [](const std::string &snapshot) {
        vtkNew<vtkXMLUnstructuredGridReader> reader;
        reader->SetFileName(snapshot.c_str());
        reader->Update();
        auto *grid = reader->GetOutput();
        ASSERT_GT(grid->GetNumberOfPoints(), 0);
        auto *solution = grid->GetPointData()->GetArray("solution");
        ASSERT_NE(solution, nullptr);
        bool has_interior_point = false;
        for (vtkIdType i = 0; i < grid->GetNumberOfPoints(); ++i)
          {
            double point[3];
            grid->GetPoint(i, point);
            double expected = 1.;
            bool   interior = true;
            for (unsigned int d = 0; d < dim; ++d)
              {
                expected += point[d] * point[d];
                interior = interior && point[d] > 0. && point[d] < 1.;
              }
            has_interior_point = has_interior_point || interior;
            // DataOut stores solution values in Float32 arrays.
            EXPECT_NEAR(solution->GetComponent(i, 0),
                        static_cast<double>(static_cast<float>(expected)),
                        1.e-11);
          }
        EXPECT_TRUE(has_interior_point);
      }(snapshot);
    MPI_Barrier(MPI_COMM_WORLD);
  }
} // namespace
TEST(LaplacianValidation, BOTH_SimplexQuadraticSolution)
{
  check_laplacian<2>(true);
  check_laplacian<3>(true);
}
TEST(LaplacianValidation, BOTH_HypercubeQuadraticSolution)
{
  check_laplacian<2>(false);
  check_laplacian<3>(false);
}
#endif
