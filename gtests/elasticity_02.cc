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

#include <deal.II/base/mutex.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "elasticity.h"
#include "utils.h"

using namespace dealii;

template <int dim>
void
get_default_test_parameters(ElasticityProblemParameters<dim> &par)
{
#ifdef DEBUG
  par.output_directory = "tests_debug_output";
#else
  par.output_directory = "tests_release_output";
#endif
  par.output_name         = "solution";
  par.fe_degree           = 1;
  par.initial_refinement  = 5;
  par.domain_type         = "generate";
  par.name_of_grid        = "hyper_cube";
  par.arguments_for_grid  = "-1: 1: false";
  par.refinement_strategy = "fixed_fraction";
  par.coarsening_fraction = 0.0;
  par.refinement_fraction = 0.3;
  par.n_refinement_cycles = 1;
  par.max_cells           = 20000;

  par.default_material_properties.Lame_mu     = 1;
  par.default_material_properties.Lame_lambda = 1;

  par.displacement_solver_control.set_reduction(1e-12);
  par.displacement_solver_control.set_tolerance(1e-12);
  par.reduced_mass_solver_control.set_reduction(1e-12);
  par.reduced_mass_solver_control.set_tolerance(1e-12);
}



TEST(ElasticityTest, TwoInclusionsInCell)
{
  ParameterAcceptor::clear();
  static constexpr int             dim = 3;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  ElasticityProblem<dim> problem(par);


  initialize_parameters();
  ParameterAcceptor::prm.parse_input_from_string(
    R"(
      subsection Error
  set Enable computation of the errors = false
end
subsection Immersed Problem
  set Dirichlet boundary ids             = 0
  set Initial refinement                 = 3
  set Output name                        = TwoInclusionsInCell
  subsection Immersed inclusions
    set Inclusions                          = 0,0,0, 0,0,1, 0.5,0; 0,0,0.5, 0,0,1, 0.5,0; 
    set Reference inclusion data            = 0,0,0,0.1,0,0,0,0.1,0
    set Inclusions refinement               = 50
    set Number of fourier coefficients      = 2
    set Selection of Fourier coefficients   = 3,7
    set Cluster inclusions with segments    = true
  end
end
subsection Solvers
  subsection Augmented Lagrange
    set Log frequency = 1
    set Log history   = false
    set Log result    = true
    set Max steps     = 1000
    set Reduction     = 1.e-10
    set Tolerance     = 1.e-10
  end
  subsection Displacement
    set Log frequency = 1
    set Log history   = false
    set Log result    = true
    set Max steps     = 1000
    set Reduction     = 1.e-8
    set Tolerance     = 1.e-10
  end
  subsection Reduced mass
    set Log frequency = 1
    set Log history   = false
    set Log result    = true
    set Max steps     = 100
    set Reduction     = 1.e-10
    set Tolerance     = 1.e-12
  end
end
    )");

  ParameterAcceptor::parse_all_parameters();
  problem.run();

  const double tol = 1e-4;
  ASSERT_NEAR(problem.solution.block(0).l2_norm(), 0.677138, tol);
  ASSERT_NEAR(problem.solution.block(1).l2_norm(), 3.09713, tol);
}

TEST(ElasticityTest, MultiInclusionsInCell)
{
  ParameterAcceptor::clear();
  static constexpr int             dim = 3;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  ElasticityProblem<dim> problem(par);


  initialize_parameters();
  ParameterAcceptor::prm.parse_input_from_string(
    R"(
      subsection Error
  set Enable computation of the errors = false
end
subsection Immersed Problem
  set Dirichlet boundary ids             = 2,3,4,5
  set Initial refinement                 = 4
  set Normal flux boundary ids           = 0,1
  set Output name                        = MultiInclusionsInCell
  subsection Grid generation
    set Grid generator arguments = -2:2:true
  end
  subsection Immersed inclusions
    set Inclusions file                     = ../data/tests/cylinder_x.txt
    set Reference inclusion data            = 0,0,0,0.5,0,0,0,0.5,0
    set Inclusions refinement               = 50
    set Number of fourier coefficients      = 2
    set Selection of Fourier coefficients   = 3,7
    set Cluster inclusions with segments    = true
  end
end
subsection Solvers
  subsection Augmented Lagrange
    set Log frequency = 1
    set Log history   = false
    set Log result    = true
    set Max steps     = 1000
    set Reduction     = 1.e-10
    set Tolerance     = 1.e-10
  end
  subsection Displacement
    set Log frequency = 1
    set Log history   = false
    set Log result    = true
    set Max steps     = 1000
    set Reduction     = 1.e-8
    set Tolerance     = 1.e-10
  end
  subsection Reduced mass
    set Log frequency = 1
    set Log history   = false
    set Log result    = true
    set Max steps     = 100
    set Reduction     = 1.e-10
    set Tolerance     = 1.e-12
  end
end
    )");

  ParameterAcceptor::parse_all_parameters();
  problem.run();

  const double tol = 1e-4;
  ASSERT_NEAR(problem.solution.block(0).l2_norm(), 12.5827, tol);
  ASSERT_NEAR(problem.solution.block(1).l2_norm(), 25.2899, tol);
}