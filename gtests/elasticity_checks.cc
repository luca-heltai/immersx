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



TEST(ElasticityTest, DisplacementX)
{
  ParameterAcceptor::clear();
  static constexpr int             dim = 2;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  ElasticityProblem<dim> problem(par);
  initialize_parameters();

  ParameterAcceptor::prm.parse_input_from_string(
    R"(
    subsection Immersed Problem
      set Output name                           = displacement_x
      set Initial refinement                    = 5
      subsection Grid generation
        set Domain type              = generate
        set Grid generator           = hyper_cube
        set Grid generator arguments = -5: 5: false
      end
      subsection Immersed inclusions
        set Inclusions                          = 0, 0, 1.0
        set Number of fourier coefficients      = 1
        set Selection of Fourier coefficients   = 0
        set Inclusions refinement               = 100
        subsection Boundary data
          set Function expression = 1; 0
        end
      end
    end
  )");

  ParameterAcceptor::parse_all_parameters();
  problem.run();
  ASSERT_NEAR(problem.solution.block(0).linfty_norm(), 1.0, 5e-2);
}



TEST(ElasticityTest, DisplacementY)
{
  ParameterAcceptor::clear();
  static constexpr int             dim = 2;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  ElasticityProblem<dim> problem(par);
  initialize_parameters();

  ParameterAcceptor::prm.parse_input_from_string(
    R"(
    subsection Immersed Problem
      set Output name                           = displacement_y
      set Initial refinement                    = 5
      subsection Grid generation
        set Domain type              = generate
        set Grid generator           = hyper_cube
        set Grid generator arguments = -5: 5: false
      end
      subsection Immersed inclusions
        set Inclusions                          = 0, 0, 1.0
        set Number of fourier coefficients      = 1
        set Selection of Fourier coefficients   = 1
        set Inclusions refinement               = 100
        subsection Boundary data
          set Function expression = 0; 1
        end
      end
    end
  )");

  ParameterAcceptor::parse_all_parameters();
  problem.run();
  ASSERT_NEAR(problem.solution.block(0).linfty_norm(), 1.0, 6e-2);
}



TEST(ElasticityTest, DisplacementXScaled)
{
  ParameterAcceptor::clear();
  static constexpr int             dim = 2;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  ElasticityProblem<dim> problem(par);
  initialize_parameters();

  ParameterAcceptor::prm.parse_input_from_string(
    R"(
    subsection Immersed Problem
      set Output name                           = displacement_x_scaled
      set Initial refinement                    = 5
      subsection Grid generation
        set Domain type              = generate
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Immersed inclusions
        set Inclusions                          = 0, 0, .1
        set Number of fourier coefficients      = 1
        set Inclusions refinement               = 100
        set Selection of Fourier coefficients   = 0 
        subsection Boundary data
          set Function expression = .1; 0
        end
      end
    end
    subsection Solvers
      subsection Displacement
        set Max steps     = 100
        set Reduction     = 1.e-9
        set Tolerance     = 1.e-9
      end
    end
  )");

  ParameterAcceptor::parse_all_parameters();
  problem.run();
  ASSERT_NEAR(problem.solution.block(0).linfty_norm(), .1, 2e-1);
}



TEST(ElasticityTest, DisplacementYScaled)
{
  ParameterAcceptor::clear();
  static constexpr int             dim = 2;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  ElasticityProblem<dim> problem(par);
  initialize_parameters();

  ParameterAcceptor::prm.parse_input_from_string(
    R"(
    subsection Immersed Problem
      set Output name                           = displacement_y_scaled
      set Initial refinement                    = 5
      subsection Grid generation
        set Domain type              = generate
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Immersed inclusions
        set Inclusions                          = 0, 0, .1
        set Number of fourier coefficients      = 1
        set Inclusions refinement               = 100
        set Selection of Fourier coefficients   = 1
        subsection Boundary data
          set Function expression = 0;.1
        end
      end
    end
  )");

  ParameterAcceptor::parse_all_parameters();
  problem.run();
  ASSERT_NEAR(problem.solution.block(0).linfty_norm(), 0.1, 2e-1);
}


/**
 * We need the data_file_1d.txt file to be present somewhere
 */
TEST(ElasticityTest, DISABLED_CheckInclusionMatrix)
{
  ParameterAcceptor::clear();
  static constexpr int             dim = 2;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  ElasticityProblem<dim> problem(par);
  initialize_parameters();

  ParameterAcceptor::prm.parse_input_from_string(
    R"(
    subsection Immersed Problem
      set Output name                           = displacement_enflate
      set Initial refinement                    = 5
      subsection Grid generation
        set Domain type              = generate
        set Grid generator           = hyper_cube
        set Grid generator arguments = -4: 4: false
      end
      subsection Immersed inclusions
        set Inclusions                          = 0, 0, .5
        set Number of fourier coefficients      = 3
        set Data file                           = ../data/tests/data_file_1d.txt
        set Inclusions refinement               = 100
      end
    end
  )");

  ParameterAcceptor::parse_all_parameters();
  problem.run();
  // ASSERT_NEAR(problem.solution.block(0).linfty_norm(), 1.0, 6e-2);
  problem.inclusion_matrix.print(std::cout);
  // problem.coupling_matrix.print(std::cout);

  auto inclusions   = problem.solution.block(1);
  auto displacement = problem.solution.block(0);
  // problem.solution.block(0) = 1.0;
  const auto Bt = linear_operator<LA::MPI::Vector>(problem.coupling_matrix);
  const auto B  = transpose_operator(Bt);
  const auto M  = linear_operator<LA::MPI::Vector>(problem.inclusion_matrix);


  // for small radius you might need SolverFGMRES<LA::MPI::Vector>
  SolverCG<LA::MPI::Vector> cg_M(problem.par.displacement_solver_control);
  auto                      invM = inverse_operator(M, cg_M);


  inclusions = invM * B * displacement;

  inclusions.print(std::cout);
}

TEST(ElasticityTest3, Rotation3D)
{
  ParameterAcceptor::clear();
  static constexpr int             dim = 3;
  ElasticityProblemParameters<dim> parX;
  get_default_test_parameters(parX);
  ElasticityProblem<dim> problemX(parX);

  initialize_parameters();
  ParameterAcceptor::prm.parse_input_from_string(
    R"(
      subsection Immersed Problem
        set Dirichlet boundary ids             = 0
        set Initial refinement                 = 4
        set Output name                        = solution_3D_X
        subsection Immersed inclusions
          set Data file                         =
          set Inclusions                        = 0., 0., 0., 1., 0., 0., 0.5, 0
          set Inclusions file                   = 
          set Inclusions refinement             = 100 
          set Number of fourier coefficients    = 2
          set Reference inclusion data          = 0, 0, 0, 0.1, 0, 0, 0, 0.1, 0,0, 0,0 
          set Selection of Fourier coefficients = 3,7
        end
      end
    )");

  ParameterAcceptor::parse_all_parameters();
  problemX.run();

  ElasticityProblemParameters<dim> parZ;
  get_default_test_parameters(parZ);
  ElasticityProblem<dim> problemZ(parZ);

  initialize_parameters();
  ParameterAcceptor::prm.parse_input_from_string(
    R"(
      subsection Immersed Problem
        set Dirichlet boundary ids             = 0
        set Initial refinement                 = 4
        set Output name                        = solution_3D_Z
        subsection Immersed inclusions
          set Data file                         =
          set Inclusions                        = 0., 0., 0., 0., 0., 1., 0.5, 0
          set Inclusions file                   = 
          set Inclusions refinement             = 100 
          set Number of fourier coefficients    = 2
          set Reference inclusion data          = 0, 0, 0, 0.1, 0, 0, 0, 0.1, 0,0, 0,0 
          set Selection of Fourier coefficients = 3,7
        end
      end
    )");

  ParameterAcceptor::parse_all_parameters();
  problemZ.run();

  const double tol = 0.1;
  ASSERT_NEAR(problemX.solution.block(0).l2_norm(),
              problemZ.solution.block(0).l2_norm(),
              tol);
  ASSERT_NEAR(problemX.solution.block(1).l2_norm(),
              problemZ.solution.block(1).l2_norm(),
              tol);
}


TEST(ElasticityTest3, Displacement3D)
{
  ParameterAcceptor::clear();
  static constexpr int             dim = 3;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  ElasticityProblem<dim> problem(par);


  initialize_parameters();
  ParameterAcceptor::prm.parse_input_from_string(
    R"(
subsection Immersed Problem
  set Dirichlet boundary ids             = 0
  set Initial refinement                 = 3
  set Normal flux boundary ids           = 1,2
  set Output name                        = Displacement3D
  subsection Grid generation
    set Domain type              = generate
    set Grid generator           = subdivided_cylinder
    set Grid generator arguments = 2:2:2
  end
  subsection Immersed inclusions
    set Inclusions file                     = ../data/tests/cylinder_x.txt
    set Reference inclusion data            = 0,0,0,0.1,0,0,0,0.1,0
    set Inclusions refinement               = 50
    set Number of fourier coefficients      = 2
    set Selection of Fourier coefficients   = 3,7
  end
  subsection Material properties
    set Material tags by material id =
    subsection default
      set Lame lambda     = 50.0
      set Lame mu         = 2.0
    end
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
  ASSERT_NEAR(problem.solution.block(0).l2_norm(), 3.6763217, tol);
  ASSERT_NEAR(problem.solution.block(1).l2_norm(), 81.839722544, tol);
}