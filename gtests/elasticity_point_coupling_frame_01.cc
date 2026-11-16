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

// Point coupling in 3D with a boundary function and no data line. The
// multiplier dof of the constant mode c tests the displacement along R e_c,
// where R rotates the z axis onto the vessel direction. The data of that dof
// must therefore be (R e_c) . g(x), the boundary function in the vessel
// frame. For a rigid translation u = g, the constraint B u = rhs then holds
// exactly whatever the vessel direction.

#include <deal.II/base/function_parser.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elasticity.h>

#include <string>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  constexpr int dim = 3;

  // Unit cube [-1,1]^3 without boundary conditions, and one inclusion of
  // radius 0.3 at the origin along `direction`, with the three constant
  // modes and the boundary data g = (0.1, -0.2, 0.3).
  void
  check_translation_satisfies_constraint(const std::string &direction)
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<dim> par;
    ElasticityProblem<dim>           problem(par);
    par.output_directory =
      TestPaths::output_directory("elasticity-point-coupling-frame-01");
    initialize_parameters_from_string(R"(
      subsection Immersed Problem
        set Initial refinement     = 2
        set Dirichlet boundary ids =
        subsection Grid generation
          set Domain type              = generate
          set Grid generator           = hyper_cube
          set Grid generator arguments = -1: 1: false
        end
        subsection Immersed inclusions
          set Inclusions                     = 0, 0, 0, )" +
                                      direction + R"(, 0.3, 0
          set Number of fourier coefficients = 1
          set Inclusions refinement          = 64
          subsection Boundary data
            set Function expression = 0.1; -0.2; 0.3
          end
        end
      end
    )");

    problem.make_grid();
    problem.setup_fe();
    problem.inclusions.setup_inclusions_particles(*problem.tria);
    problem.setup_dofs();
    problem.assemble_coupling();
    ASSERT_EQ(problem.inclusions.n_dofs(), 3u);

    FunctionParser<dim> translation(dim);
    translation.initialize("x,y,z", "0.1; -0.2; 0.3", {});
    LA::MPI::Vector u(problem.owned_dofs[0], problem.mpi_communicator);
    VectorTools::interpolate(problem.dh, translation, u);

    LA::MPI::Vector Bu(problem.solution.block(1));
    problem.coupling_matrix.Tvmult(Bu, u);
    const auto &rhs = problem.system_rhs.block(1);
    EXPECT_GT(rhs.linfty_norm(), 0.);
    for (const auto i : Bu.locally_owned_elements())
      EXPECT_NEAR(Bu[i], rhs[i], 1e-12) << "multiplier dof " << i;
  }
} // namespace



// The vessel frame is the global frame, so the component mix-up cannot show.
TEST(ElasticityProblem, BOTH_PointCouplingBoundaryFunctionVesselAlongZ)
{
  check_translation_satisfies_constraint("0, 0, 0.2");
}



// Before issue #456 was fixed, the data were the global components of g, so
// B u = R^T g differed from the rhs g.
TEST(ElasticityProblem, BOTH_PointCouplingBoundaryFunctionTiltedVessel)
{
  check_translation_satisfies_constraint("0.05, 0.1, 0.1");
}
