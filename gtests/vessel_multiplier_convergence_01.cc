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

// Convergence of the multiplier of an immersed vessel to the exact traction
// jump, on a cube with the manufactured field of
// doc/tutorials/vessel-multiplier-convergence.md (static step).

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elasticity.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>

#include "test_paths.h"

#ifdef DEAL_II_WITH_VTK

using namespace dealii;
using namespace ImmersX;

namespace
{
  constexpr double radius   = 0.2;
  constexpr double dilation = 0.5;

  std::string
  output_dir()
  {
    const std::string dir =
      TestPaths::output_directory("vessel-multiplier-convergence");
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::filesystem::create_directories(dir);
    MPI_Barrier(MPI_COMM_WORLD);
    return dir;
  }

  // Straight vessel on the x axis of the cube [-1, 1]^3 with eight cells.
  std::string
  write_vessel_grid()
  {
    const std::string name = output_dir() + "/vessel_x.vtk";
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::ofstream out(name);
        out << "# vtk DataFile Version 3.0\nvessel\nASCII\n"
            << "DATASET UNSTRUCTURED_GRID\nPOINTS 9 double\n";
        for (unsigned int i = 0; i < 9; ++i)
          out << -1. + 0.25 * i << " 0 0\n";
        out << "CELLS 8 24\n";
        for (unsigned int i = 0; i < 8; ++i)
          out << "2 " << i << " " << i + 1 << "\n";
        out << "CELL_TYPES 8\n";
        for (unsigned int i = 0; i < 8; ++i)
          out << "3\n";
        out << "CELL_DATA 8\nSCALARS vessel_id int\nLOOKUP_TABLE default\n";
        for (unsigned int i = 0; i < 8; ++i)
          out << "0\n";
      }
    MPI_Barrier(MPI_COMM_WORLD);
    return name;
  }

  // Static manufactured field u = (h x, g W_y, g W_z) with
  // W = a^2 / max(r^2, a^2) (y, z), r^2 = y^2 + z^2: a uniform dilation g
  // inside the vessel of radius a, the field g a^2 / r e_r outside, and the
  // same axial strain h on both sides. With lambda = mu = 1 the jump of
  // sigma_rr across the wall is -(4 mu + 2 lambda) g = -6 g, independent of
  // h. The cube faces get the exact field.
  //
  // The mesh is refined `passes` times near the wall. Returns the mean
  // multiplier of the modes 3 and 7 over the support points with
  // |x| <= 0.5.
  double
  solve_cube(const unsigned int passes, const double axial_strain)
  {
    const std::string grid = write_vessel_grid();
    const std::string r2   = std::to_string(radius * radius);
    const std::string w    = r2 + "/max(y*y+z*z," + r2 + ")";
    const std::string g    = std::to_string(dilation);
    const std::string expr = std::to_string(axial_strain) + "*x; " + g + "*y*" +
                             w + "; " + g + "*z*" + w;
    const std::string data = std::to_string(dilation * radius);
    const std::string dir  = output_dir();

    ParameterAcceptor::clear();
    ElasticityProblemParameters<3> par;
    ElasticityProblem<3>           problem(par);
    const std::string              prm = output_dir() + "/cube.prm";
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::ofstream out(prm);
        out << R"(
subsection Functions
  subsection Dirichlet boundary conditions
    set Function expression = )"
            << expr << R"(
  end
  subsection Exact solution
    set Function expression = )"
            << expr << R"(
  end
end
subsection Immersed Problem
  set Coupling type                      = TensorProduct
  set Dirichlet boundary ids             = 0
  set Initial refinement                 = 3
  set Output directory                   = )"
            << dir << R"(
  set Output results also before solving = false
  subsection Grid generation
    set Domain type              = generate
    set Grid generator           = hyper_cube
    set Grid generator arguments = -1: 1: false
  end
end
subsection Tensor product coupling
  subsection Cross section
    set Geometry type        = hyper_ball
    set Maximum basis degree = 1
    set Refinement level     = 3
    set Selected modes       = 3, 7
  end
  subsection Representative domain
    set Finite element degree            = 1
    set Input file fields                = vessel_id
    set Number of quadrature points      = 3
    set Number of quadrature repetitions = 16
    set Reduced grid name                = )"
            << grid << R"(
    set Reduced right hand side          = )"
            << data << "; " << data << R"(
    set Thickness                        = )"
            << radius << R"(
  end
end
subsection Solvers
  subsection Augmented Lagrange
    set Max steps = 10000
    set Reduction = 1.e-10
    set Tolerance = 1.e-12
  end
  subsection Displacement
    set Max steps = 10000
    set Reduction = 1.e-8
    set Tolerance = 1.e-14
  end
end
)";
      }
    MPI_Barrier(MPI_COMM_WORLD);
    initialize_parameters(prm);

    problem.make_grid();
    for (unsigned int p = 0; p < passes; ++p)
      {
        // Refine the cells within one cell diameter of the wall r = a.
        for (const auto &cell : problem.tria->active_cell_iterators())
          if (cell->is_locally_owned())
            {
              double rmin = 1e10, rmax = 0.;
              for (const auto v : cell->vertex_indices())
                {
                  const auto  &q = cell->vertex(v);
                  const double r = std::sqrt(q[1] * q[1] + q[2] * q[2]);
                  rmin           = std::min(rmin, r);
                  rmax           = std::max(rmax, r);
                }
              const double d = cell->diameter();
              if (rmin <= radius + d && rmax >= radius - d)
                cell->set_refine_flag();
            }
        problem.tria->execute_coarsening_and_refinement();
      }
    problem.setup_fe();
    problem.setup_dofs();
    problem.cycle = 0;
    problem.assemble_elasticity_system();
    problem.assemble_coupling();
    problem.solve_static();

    const auto &lambda = problem.locally_relevant_solution.block(1);
    const auto &rdh    = problem.tensor_product_coupling->get_dof_handler();
    const auto &fe     = rdh.get_fe();
    std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
    std::set<types::global_dof_index>    seen;
    double                               sum = 0., count = 0.;
    for (const auto &cell : rdh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            if (lambda.locally_owned_elements().is_element(indices[i]) &&
                seen.insert(indices[i]).second)
              {
                const auto vertex = fe.system_to_component_index(i).second;
                if (std::abs(cell->vertex(vertex)[0]) <= 0.5 + 1e-12)
                  {
                    sum += lambda[indices[i]];
                    count += 1.;
                  }
              }
        }
    return Utilities::MPI::sum(sum, MPI_COMM_WORLD) /
           Utilities::MPI::sum(count, MPI_COMM_WORLD);
  }
} // namespace

// The multiplier converges to the traction jump -6 g = -3 under refinement
// near the wall. Release runs of the same setup give errors 0.92, 0.25,
// 0.15, 0.080, 0.030 for one to five passes (h / a = 0.625 to 0.039); the
// test checks the first two levels.
TEST(VesselMultiplierValidation, BOTH_StaticCubeJumpConverges)
{
  const double exact  = -6. * dilation;
  const double coarse = std::abs(solve_cube(1, 0.) - exact);
  const double fine   = std::abs(solve_cube(2, 0.) - exact);
  EXPECT_GT(coarse / fine, 2.5);
  EXPECT_LT(fine, 0.1 * std::abs(exact));
}

// A uniform axial strain is the same on both sides of the wall and Q1
// represents it exactly: the multiplier, which is the traction jump, does not
// change. The tissue stress on the wall does change, by lambda h.
TEST(VesselMultiplierValidation, BOTH_StaticCubeJumpIndependentOfAxialStrain)
{
  const double plane     = solve_cube(1, 0.);
  const double stretched = solve_cube(1, 0.25);
  EXPECT_NEAR(stretched, plane, 1e-6 * std::abs(plane));
}

#endif // DEAL_II_WITH_VTK
