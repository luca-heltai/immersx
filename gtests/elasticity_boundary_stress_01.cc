// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
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

// Boundary tractions and normal displacement fluxes written by
// ElasticityProblem::compute_internal_and_boundary_stress(), checked against
// exact linear elasticity solutions imposed as Dirichlet data on the whole
// boundary of the unit square/cube.
//
// Each displacement field lies in the finite element space (affine with Q1,
// quadratic with Q2), so the discrete solution is exact up to the solver
// tolerance and the boundary postprocessing can be checked to near round-off.
// All fields have a nonzero skew-symmetric gradient, so a traction computed
// from grad(u) instead of the strain would fail.

#include <deal.II/base/mpi.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elasticity.h>

#include <array>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  template <int dim>
  struct ExpectedBoundaryData
  {
    types::boundary_id    id;
    std::array<double, 3> mean_traction;
    double                integrated_u_dot_n;
  };

  // Unit square/cube with colorized ids (0: x=0, 1: x=1, 2: y=0, 3: y=1,
  // 4: z=0, 5: z=1), mu = lambda = 1 and the exact displacement imposed as
  // strong Dirichlet data on every face.
  std::string
  parameters(const unsigned int dim,
             const unsigned int fe_degree,
             const std::string &displacement,
             const std::string &body_force)
  {
    const std::string ids = (dim == 2) ? "0, 1, 2, 3" : "0, 1, 2, 3, 4, 5";
    return R"(
      subsection Immersed Problem
        set FE degree              = )" +
           std::to_string(fe_degree) + R"(
        set Initial refinement     = )" +
           std::to_string(dim == 2 ? 3 : 2) + R"(
        set Dirichlet boundary ids = )" +
           ids + R"(
        subsection Grid generation
          set Domain type              = generate
          set Grid generator           = hyper_cube
          set Grid generator arguments = 0: 1: true
        end
        subsection Material properties
          subsection default
            set Lame mu     = 1
            set Lame lambda = 1
          end
        end
      end
      subsection Functions
        subsection Dirichlet boundary conditions
          set Function expression = )" +
           displacement + R"(
        end
        subsection Right hand side
          set Function expression = )" +
           body_force + R"(
        end
      end
      subsection Solvers
        subsection Displacement
          set Max steps = 1000
          set Reduction = 1.e-12
          set Tolerance = 1.e-13
        end
      end
    )";
  }

  // Run the static problem and return the last row of the forces file, keyed
  // by its header. Only rank 0 writes the file, so other ranks get an empty
  // map.
  template <int dim>
  std::map<std::string, double>
  run_and_read_forces(const std::string &name, const std::string &prm)
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<dim> par;
    ElasticityProblem<dim>           problem(par);
    // Serial and MPI runs of the same test may share the output directory.
    par.output_directory =
      TestPaths::output_directory("elasticity-boundary-stress-01");
    par.output_name =
      name + "_np" +
      std::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
    initialize_parameters_from_string(prm);

    problem.run();

    std::map<std::string, double> row;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
      return row;

    std::ifstream file(par.output_directory + "/" + par.output_name +
                       "_forces.txt");
    EXPECT_TRUE(file.good());
    std::string header, line, last;
    std::getline(file, header);
    while (std::getline(file, line))
      if (!line.empty())
        last = line;

    std::istringstream header_stream(header), value_stream(last);
    std::string        key;
    double             value;
    while (header_stream >> key)
      {
        EXPECT_TRUE(value_stream >> value) << "missing value for " << key;
        row[key] = value;
      }
    return row;
  }

  template <int dim>
  void
  check_boundary_data(const std::map<std::string, double>          &row,
                      const std::vector<ExpectedBoundaryData<dim>> &expected)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
      return;

    const double                     tol = 1e-8;
    const std::array<std::string, 3> xyz{{"X", "Y", "Z"}};
    const std::string prefix = (dim == 2) ? "boundaryStress" : "sigman";
    for (const auto &data : expected)
      {
        const std::string id = std::to_string(data.id);
        for (unsigned int d = 0; d < dim; ++d)
          {
            const std::string key = prefix + xyz[d] + "_" + id;
            ASSERT_EQ(row.count(key), 1u) << key;
            EXPECT_NEAR(row.at(key), data.mean_traction[d], tol) << key;
          }
        const std::string key = "uDotN_" + id;
        ASSERT_EQ(row.count(key), 1u) << key;
        EXPECT_NEAR(row.at(key), data.integrated_u_dot_n, tol) << key;
      }
  }
} // namespace



// u = (0.1 x + 0.2 y, -0.05 y), f = 0, Q1.
// sigma_xx = 0.25, sigma_yy = -0.05, sigma_xy = 0.2.
TEST(ElasticityProblem, BOTH_BoundaryStressUniformStrain2D)
{
  constexpr int dim = 2;
  const auto    row = run_and_read_forces<dim>(
    "uniform_strain_2d", parameters(dim, 1, "0.1*x + 0.2*y; -0.05*y", "0; 0"));

  check_boundary_data<dim>(row,
                           {{0, {{-0.25, -0.2, 0}}, -0.1},
                            {1, {{0.25, 0.2, 0}}, 0.2},
                            {2, {{-0.2, 0.05, 0}}, 0.0},
                            {3, {{0.2, -0.05, 0}}, -0.05}});
}



// u = (x^2, x y), f = -div(sigma) = (-8, 0), Q2.
// sigma_xx = 7 x, sigma_yy = 5 x, sigma_xy = y.
TEST(ElasticityProblem, BOTH_BoundaryStressQuadratic2D)
{
  constexpr int dim = 2;
  const auto    row =
    run_and_read_forces<dim>("quadratic_2d",
                             parameters(dim, 2, "x^2; x*y", "-8; 0"));

  check_boundary_data<dim>(row,
                           {{0, {{0, -0.5, 0}}, 0.0},
                            {1, {{7, 0.5, 0}}, 1.0},
                            {2, {{0, -2.5, 0}}, 0.0},
                            {3, {{1, 2.5, 0}}, 0.5}});
}



// u = (0.1 x + 0.2 y, -0.05 y + 0.1 z, 0.03 x - 0.02 z), f = 0, Q1.
// sigma_xx = 0.23, sigma_yy = -0.07, sigma_zz = -0.01,
// sigma_xy = 0.2, sigma_xz = 0.03, sigma_yz = 0.1.
TEST(ElasticityProblem, BOTH_BoundaryStressUniformStrain3D)
{
  constexpr int dim = 3;
  const auto    row = run_and_read_forces<dim>(
    "uniform_strain_3d",
    parameters(
      dim, 1, "0.1*x + 0.2*y; -0.05*y + 0.1*z; 0.03*x - 0.02*z", "0; 0; 0"));

  check_boundary_data<dim>(row,
                           {{0, {{-0.23, -0.2, -0.03}}, -0.1},
                            {1, {{0.23, 0.2, 0.03}}, 0.2},
                            {2, {{-0.2, 0.07, -0.1}}, -0.05},
                            {3, {{0.2, -0.07, 0.1}}, 0.0},
                            {4, {{-0.03, -0.1, 0.01}}, -0.015},
                            {5, {{0.03, 0.1, -0.01}}, -0.005}});
}
