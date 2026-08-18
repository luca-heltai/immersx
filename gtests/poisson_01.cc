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
// full text of the license can be found in the LICENSE.md file at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>

#include <filesystem>

#include "poisson.h"
#include "utils.h"


using namespace dealii;


TEST(Poisson, Construction)
{
  ParameterAcceptor::clear();
  PoissonParameters<2> parameters;
  PoissonSolver<2>     problem(parameters);

  EXPECT_EQ(problem.n_dofs(), 0u);
  EXPECT_FALSE(problem.solution_is_finite());
}


TEST(Poisson, MPI_OneCycleSolve)
{
  ParameterAcceptor::clear();
  PoissonParameters<2> parameters;
  initialize_parameters_from_string(R"(
    subsection Poisson
      set FE degree                   = 1
      set Initial refinement          = 1
      set Dirichlet boundary ids      = 0
      set Output name                 = poisson_01
      subsection Grid generation
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Refinement and remeshing
        set Number of refinement cycles = 2
        set Strategy                   = global
      end
      subsection Right hand side
        set Function expression = 1
        set Variable names      = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names      = x,y,t
      end
      subsection Solver
        subsection Control
          set Max steps  = 100
          set Reduction  = 1.e-10
          set Tolerance  = 1.e-12
          set Log result = false
        end
      end
    end
  )");

  parameters.output_directory =
    (std::filesystem::temp_directory_path() / "immersx_poisson_01").string();

  PoissonSolver<2> problem(parameters);
  problem.run();

  EXPECT_GT(problem.n_dofs(), 0u);
  EXPECT_TRUE(problem.solution_is_finite());
  EXPECT_GT(problem.solution_l2_norm(), 0.);
}
