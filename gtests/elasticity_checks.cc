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



TEST(ElasticityTest, Displacement)
{
  static constexpr int   dim = 2;
  auto                   par = get_default_test_parameters<dim>();
  ElasticityProblem<dim> problem(*par);
  ParameterAcceptor::initialize();
  problem.run();
}
