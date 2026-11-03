#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elasticity.h>
#include <immersx/physics/elasticity_problem_parameters.h>

#include "test_paths.h"

using namespace ImmersX;

using namespace dealii;

TEST(ElasticityCouplingParameters, ParseCouplingSelector)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2> par;

  // Initialize global defaults used by the test harness (this mirrors how the
  // existing tests call initialize_parameters() before parsing test-specific
  // snippets).
  initialize_parameters();

  // Minimal parameter snippet selecting tensor-product coupling. Do not
  // attempt to set the full tensor-product subsections here; just verify
  // selector parsing and that ParameterAcceptor does not throw.
  ParameterAcceptor::prm.parse_input_from_string(
    R"(
    subsection Immersed Problem
      set Coupling type = TensorProduct
    end
  )");

  EXPECT_NO_THROW(ParameterAcceptor::parse_all_parameters());
  EXPECT_EQ(par.coupling_type, CouplingType::TensorProduct);
}

#ifndef DEAL_II_WITH_VTK
TEST(ElasticityCoupling, TensorProductIsDisabledWithoutVTK)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2> par;
  par.coupling_type = CouplingType::TensorProduct;

  ElasticityProblem<2> problem(par);

  EXPECT_FALSE(problem.uses_tensor_product_coupling());
  EXPECT_EQ(problem.n_multiplier_dofs(), 0u);
}
#endif

namespace
{
  void
  check_affine_overlap(const bool outside, const bool cluster)
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<2> parameters;
    parameters.fe_degree           = 1;
    parameters.initial_refinement  = 2;
    parameters.domain_type         = "generate";
    parameters.name_of_grid        = "hyper_cube";
    parameters.arguments_for_grid  = "0:1:false";
    parameters.triangulation_type  = "distributed";
    parameters.dirichlet_ids       = {0};
    parameters.n_refinement_cycles = 1;
    for (auto *control : {&parameters.displacement_solver_control,
                          &parameters.reduced_mass_solver_control,
                          &parameters.augmented_lagrange_solver_control})
      {
        control->set_max_steps(1000);
        control->set_tolerance(1.e-13);
        control->set_reduction(1.e-13);
      }

    parameters.output_directory =
      TestPaths::output_directory("elasticity-overlap");
    parameters.output_name = std::string(outside ? "outside" : "partial") +
                             (cluster ? "-cluster" : "");
    ElasticityProblem<2> problem(parameters);
    initialize_parameters_from_string(
      "subsection Immersed Problem\n"
      " subsection Immersed inclusions\n set Inclusions = " +
      std::string(outside ? "2,2,0.25" : "-0.1,0.5,0.25") +
      "\n set Inclusions file =\n set Number of fourier coefficients = 3\n"
      " set Cluster inclusions with segments = " +
      std::string(cluster ? "true" : "false") +
      "\n subsection Boundary data\n set Function expression = 1+x;2+y\n end\n end\n"
      "end\n"
      "subsection Functions\n subsection Dirichlet boundary conditions\n"
      " set Function expression = 1+x;2+y\n end\n end\n");
    problem.run();
    LA::MPI::Vector expected(problem.solution.block(0));
    VectorTools::interpolate(problem.dh, parameters.bc, expected);
    expected -= problem.solution.block(0);
    EXPECT_LT(expected.l2_norm(), 1.e-8);
    EXPECT_LT(problem.solution.block(1).l2_norm(), 1.e-8);
  }
} // namespace

TEST(ElasticityCouplingIntegration, BOTH_ExteriorMultiplierLeavesAffineBulk)
{
  check_affine_overlap(true, false);
}
TEST(ElasticityCouplingIntegration,
     BOTH_PartialConstraintWithNonzeroDirichletData)
{
  check_affine_overlap(false, false);
}
TEST(ElasticityCouplingIntegration, BOTH_PartialWallWithExteriorCenterCluster)
{
  check_affine_overlap(false, true);
}
