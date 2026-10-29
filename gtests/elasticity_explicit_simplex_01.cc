// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elasticity_explicit.h>

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
  check_explicit_translation(const bool simplex)
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<dim> parameters;
    ElasticityExplicit<dim>          problem(parameters);
    const std::string                variables = dim == 2 ? "x,y,t" : "x,y,z,t";
    const std::string                initial   = dim == 2 ? "1;0" : "1;0;0";
    const std::string                force     = dim == 2 ? "2;0" : "2;0;0";
    const auto                       directory = TestPaths::output_directory(
      "explicit-simplex-" + std::to_string(dim) + (simplex ? "p" : "q"));
    initialize_parameters_from_string(
      "subsection Immersed Problem\n"
      " set FE degree = 1\n set Initial refinement = 0\n set Dirichlet boundary ids =\n set Weak Dirichlet boundary ids =\n set Neumann boundary ids =\n"
      " set Output directory = " +
      directory +
      "\n set Output name = translation\n"
      " subsection Grid generation\n set Triangulation type = fullydistributed\n"
      " set Domain type = " +
      std::string(simplex ? "file" : "generate") +
      "\n"
      " set Grid generator = " +
      (simplex ?
         TestPaths::data_filename(dim == 2 ? "tests/simplex_square.msh" :
                                             "tests/simplex_cube.msh") :
         "hyper_cube") +
      "\n"
      " set Grid generator arguments = 0:1:false\n end\n"
      " subsection Material properties\n subsection default\n set Density = 1\n set Lame mu = 1\n set Lame lambda = 1\n set Rayleigh alpha = 0\n set Rayleigh beta = 0\n set Viscosity eta = 0\n end\nend\n"
      " subsection Time parameters\n set Initial time = 0\n set Final time = 0.02\n set Time step = 0.01\n end\nend\n"
      "subsection Functions\n"
      " subsection Right hand side\n set Variable names = " +
      variables + "\n set Function expression = " + force +
      "\n end\n"
      " subsection Initial displacement\n set Variable names = " +
      variables + "\n set Function expression = " + initial +
      "\n end\n"
      " subsection Initial velocity\n set Variable names = " +
      variables + "\n set Function expression = " + initial + "\n end\nend\n");
    problem.run();
    MPI_Barrier(MPI_COMM_WORLD);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      [](const std::string &directory) {
        vtkNew<vtkXMLUnstructuredGridReader> reader;
        reader->SetFileName((directory + "/translation_002.vtu").c_str());
        reader->Update();
        auto *grid = reader->GetOutput();
        ASSERT_GT(grid->GetNumberOfPoints(), 0);
        auto *displacement = grid->GetPointData()->GetArray("displacement");
        ASSERT_NE(displacement, nullptr);
        // Forward Euler advances displacement with the old velocity.
        // DataOut stores the displacement in Float32 arrays.
        const double expected = 1. + .02 + .02 * .02 - .02 * .01;
        for (vtkIdType i = 0; i < grid->GetNumberOfPoints(); ++i)
          for (unsigned int d = 0; d < dim; ++d)
            EXPECT_NEAR(displacement->GetComponent(i, d),
                        d == 0 ?
                          static_cast<double>(static_cast<float>(expected)) :
                          0.,
                        1.e-11);
      }(directory);
    MPI_Barrier(MPI_COMM_WORLD);
  }
} // namespace
TEST(ElasticityExplicitValidation, BOTH_SimplexForcedTranslation)
{
  check_explicit_translation<2>(true);
  check_explicit_translation<3>(true);
}
TEST(ElasticityExplicitValidation, BOTH_HypercubeForcedTranslation)
{
  check_explicit_translation<2>(false);
  check_explicit_translation<3>(false);
}
TEST(ElasticityExplicitValidation, BOTH_RejectsHigherSimplexDegree)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> parameters;
  initialize_parameters(TestPaths::parameter_path(
    "gtests/parameters/elasticity_explicit_simplex.prm"));
  parameters.fe_degree = 2;
  ElasticityExplicit<3> problem(parameters);
  try
    {
      problem.run();
      ADD_FAILURE() << "A higher-degree row-sum simplex mass must be rejected.";
    }
  catch (const ExceptionBase &error)
    {
      EXPECT_NE(std::string(error.what()).find("FE degree 1"),
                std::string::npos);
    }
}
#endif
