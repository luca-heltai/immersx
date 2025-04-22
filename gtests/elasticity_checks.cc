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

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "elasticity.h"

using namespace dealii;

template <int dim>
void
get_default_test_parameters(ElasticityProblemParameters<dim> &par)
{
  par.output_directory    = ".";
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

  par.Lame_mu     = 1;
  par.Lame_lambda = 1;

  par.inner_control.set_reduction(1e-12);
  par.inner_control.set_tolerance(1e-12);
  par.outer_control.set_reduction(1e-12);
  par.outer_control.set_tolerance(1e-12);
}



TEST(ElasticityTest, DisplacementX)
{
  static constexpr int             dim = 2;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  ElasticityProblem<dim> problem(par);
  ParameterAcceptor::initialize();
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
  static constexpr int             dim = 2;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  ElasticityProblem<dim> problem(par);
  ParameterAcceptor::initialize();
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
  static constexpr int             dim = 2;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  ElasticityProblem<dim> problem(par);
  ParameterAcceptor::initialize();
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
  )");
  ParameterAcceptor::parse_all_parameters();
  problem.run();
  ASSERT_NEAR(problem.solution.block(0).linfty_norm(), .1, 2e-1);
}



TEST(ElasticityTest, DisplacementYScaled)
{
  static constexpr int             dim = 2;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  ElasticityProblem<dim> problem(par);
  ParameterAcceptor::initialize();
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
  static constexpr int             dim = 2;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  ElasticityProblem<dim> problem(par);
  ParameterAcceptor::initialize();
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
        set Data file                           = ../data_file_1d.txt
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
  SolverCG<LA::MPI::Vector> cg_M(problem.par.inner_control);
  auto                      invM = inverse_operator(M, cg_M);


  inclusions = B * displacement;

  inclusions.print(std::cout);
}

TEST(ElasticityTest3, DisplacementD)
{
  static constexpr int             dim = 3;
  ElasticityProblemParameters<dim> par;
  get_default_test_parameters(par);
  par.initial_refinement = 4;
  ElasticityProblem<dim> problem(par);
  // std::string
  // prm_file("/home/camilla/Desktop/Project_Lagrangian_CODE/beam_3d.prm");
  //  ParameterAcceptor::initialize(prm_file);
  ParameterAcceptor::initialize();
  ParameterAcceptor::prm.parse_input_from_string(
    R"(
    subsection Immersed Problem
      set Dirichlet boundary ids             = 0
      set FE degree                          = 1
      set Initial refinement                 = 4
      set Output directory                   = ./
      set Output name                        = solution
      set Output results also before solving = false
      subsection Dirichlet boundary conditions
        set Function constants  = 
        set Function expression = 0;0;0
        set Variable names      = x,y,z,t
      end
      subsection Exact solution
        set Function constants  = 
        set Function expression = 0;0;0
        set Variable names      = x,y,z,t
        set Weight expression   = 1.
      end
      subsection Grid generation
        set Domain type              = generate
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1:1:false
      end
      subsection Immersed inclusions
        set Bounding boxes extraction level   = 1
        set Data file                         = 
        set Inclusions                        = 0., 0., 0., 1., 0., 0., 0.1, 0 
        set Inclusions file                   = 
        set Inclusions refinement             = 100
        set Number of fourier coefficients    = 2
        set Reference inclusion data          = 0, 0, 0, 0.01, 0, 0, 0, 0.01, 0,0,0,0
        set Selection of Fourier coefficients = 3,7
        subsection Boundary data
          set Function constants  = 
          set Function expression = 0;0;0
          set Variable names      = x,y,z,t
        end
      end
      subsection Neumann boundary conditions
        set Function constants  = p = 0.14
        set Function expression = 0;0;0
        set Variable names      = x,y,z,t
      end
      subsection Physical constants
        set Lame lambda = 1
        set Lame mu     = 1
      end
      subsection Refinement and remeshing
        set Coarsening fraction         = 0
        set Maximum number of cells     = 2000000
        set Number of refinement cycles = 3
        set Refinement fraction         = 0.3
        set Strategy                    = fixed_number
      end
      subsection Right hand side
        set Function constants  = 
        set Function expression = 0;0;0
        set Variable names      = x,y,z,t
      end
    end
  )");

  ParameterAcceptor::parse_all_parameters();
  problem.run();

  ASSERT_NEAR(problem.solution.block(0).l2_norm(), 0.0956977, 2e-1);
}
