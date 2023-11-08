#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "elasticity.h"

using namespace dealii;

template <int dim>
std::shared_ptr<ElasticityProblemParameters<dim>>
get_default_test_parameters()
{
  auto par = std::make_shared<ElasticityProblemParameters<dim>>();

  par->output_directory    = ".";
  par->output_name         = "solution";
  par->fe_degree           = 1;
  par->initial_refinement  = 5;
  par->domain_type         = "generate";
  par->name_of_grid        = "hyper_cube";
  par->arguments_for_grid  = "-1: 1: false";
  par->refinement_strategy = "fixed_fraction";
  par->coarsening_fraction = 0.0;
  par->refinement_fraction = 0.3;
  par->n_refinement_cycles = 1;
  par->max_cells           = 20000;

  par->Lame_mu     = 1;
  par->Lame_lambda = 1;

  par->inner_control.set_reduction(1e-12);
  par->inner_control.set_tolerance(1e-12);
  par->outer_control.set_reduction(1e-12);
  par->outer_control.set_tolerance(1e-12);
  return par;
}



TEST(ElasticityTest, DisplacementX)
{
  static constexpr int   dim = 2;
  auto                   par = get_default_test_parameters<dim>();
  ElasticityProblem<dim> problem(*par);
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
        set Start index of Fourier coefficients = 0
        set Inclusions refinement               = 100
        subsection Boundary data
          set Function expression = 1; 0
        end
      end
    end
  )");
  ParameterAcceptor::parse_all_parameters();
  problem.run();
  ASSERT_NEAR(problem.solution.block(0).linfty_norm(), 1.0, 6e-2);
}



TEST(ElasticityTest, DisplacementY)
{
  static constexpr int   dim = 2;
  auto                   par = get_default_test_parameters<dim>();
  ElasticityProblem<dim> problem(*par);
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
        set Start index of Fourier coefficients = 0
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
  static constexpr int   dim = 2;
  auto                   par = get_default_test_parameters<dim>();
  ElasticityProblem<dim> problem(*par);
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
        set Start index of Fourier coefficients = 0
        set Inclusions refinement               = 100
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
  static constexpr int   dim = 2;
  auto                   par = get_default_test_parameters<dim>();
  ElasticityProblem<dim> problem(*par);
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
        set Start index of Fourier coefficients = 0
        set Inclusions refinement               = 100
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
