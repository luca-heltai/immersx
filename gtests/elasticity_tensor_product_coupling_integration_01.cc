#include <gtest/gtest.h>
#include <immersx/physics/elasticity.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

using namespace ImmersX;
#include "test_paths.h"

using namespace dealii;

namespace
{
  void
  configure_tensor_product_parameters(
    ElasticityProblemParameters<2, 3> &par,
    const std::string &dirichlet_expression = "0; 0; 0.01*sin(2*pi*50*t)")
  {
    initialize_parameters();
    ParameterAcceptor::prm.parse_input_from_string(
      "subsection Functions\n"
      "  subsection Dirichlet boundary conditions\n"
      "    set Function expression = " +
      dirichlet_expression +
      "\n"
      "  end\n"
      "  subsection Right hand side\n"
      "    set Function expression = 0; 0; 0\n"
      "  end\n"
      "  subsection Exact solution\n"
      "    set Function expression = 0; 0; 0\n"
      "  end\n"
      "  subsection Initial displacement\n"
      "    set Function expression = 0; 0; 0\n"
      "  end\n"
      "  subsection Initial velocity\n"
      "    set Function expression = 0; 0; 0\n"
      "  end\n"
      "end\n");
    ParameterAcceptor::parse_all_parameters();

    par.coupling_type                           = CouplingType::TensorProduct;
    par.domain_type                             = "generate";
    par.name_of_grid                            = "hyper_cube";
    par.arguments_for_grid                      = "-1: 1: false";
    par.initial_refinement                      = 1;
    par.default_material_properties.Lame_mu     = 1;
    par.default_material_properties.Lame_lambda = 1;
    par.dirichlet_ids                           = {0, 1, 2, 3, 4, 5};
    par.tensor_product_coupling_parameters.tensor_product_space_parameters
      .reduced_grid_name =
      ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
    par.tensor_product_coupling_parameters.coupling_rhs_expressions = {"1",
                                                                       "0",
                                                                       "0"};
  }
} // namespace

TEST(ElasticityCouplingIntegration, SetupCreatesTensorProductMultiplierBlock)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  configure_tensor_product_parameters(par);
  ElasticityProblem<2, 3> problem(par);

  problem.make_grid();
  problem.setup_fe();
  ASSERT_NO_THROW(problem.setup_dofs());
  ASSERT_TRUE(problem.tensor_product_coupling != nullptr);
  EXPECT_GT(problem.solution.block(1).size(), 0);
  EXPECT_EQ(problem.solution.n_blocks(), 2u);
}

TEST(ElasticityCouplingIntegration, AssemblyProducesTensorProductRhs)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  configure_tensor_product_parameters(par);
  ElasticityProblem<2, 3> problem(par);

  problem.make_grid();
  problem.setup_fe();
  problem.setup_dofs();
  problem.assemble_elasticity_system();
  problem.assemble_coupling();

  EXPECT_GT(problem.system_rhs.block(1).l2_norm(), 0.0);
}

TEST(ElasticityCouplingIntegrationValidation, StaticSolveCompletes)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  configure_tensor_product_parameters(par);
  // The six multiplier constraints need more than the three free bulk DoFs
  // of the level-one mesh with homogeneous Dirichlet boundary conditions.
  par.initial_refinement = 2;
  // Resolve the augmented system accurately before checking the physical
  // residuals, which undo its scaling.
  par.augmented_lagrange_solver_control.set_tolerance(1.e-13);
  par.augmented_lagrange_solver_control.set_reduction(1.e-13);
  ElasticityProblem<2, 3> problem(par);

  problem.make_grid();
  problem.setup_fe();
  problem.setup_dofs();
  problem.assemble_elasticity_system();
  problem.assemble_coupling();
  ASSERT_NO_THROW(problem.solve_static());

  EXPECT_TRUE(std::isfinite(problem.solution.block(0).l2_norm()));
  EXPECT_TRUE(std::isfinite(problem.solution.block(1).l2_norm()));

  LA::MPI::Vector bulk_action(problem.system_rhs.block(0));
  LA::MPI::Vector multiplier_action(problem.system_rhs.block(0));
  problem.stiffness_matrix.vmult(bulk_action, problem.solution.block(0));
  problem.coupling_matrix.vmult(multiplier_action, problem.solution.block(1));
  const double bulk_scale = bulk_action.l2_norm() +
                            multiplier_action.l2_norm() +
                            problem.system_rhs.block(0).l2_norm();
  bulk_action += multiplier_action;
  bulk_action -= problem.system_rhs.block(0);
  EXPECT_LE(bulk_action.l2_norm(), 1.e-8 * std::max(1., bulk_scale));

  LA::MPI::Vector constraint_action(problem.system_rhs.block(1));
  LA::MPI::Vector outside_action(problem.system_rhs.block(1));
  problem.coupling_matrix.Tvmult(constraint_action, problem.solution.block(0));
  problem.outside_mass_matrix.vmult(outside_action, problem.solution.block(1));
  const double constraint_scale = constraint_action.l2_norm() +
                                  outside_action.l2_norm() +
                                  problem.system_rhs.block(1).l2_norm();
  constraint_action -= outside_action;
  constraint_action -= problem.system_rhs.block(1);
  EXPECT_LE(constraint_action.l2_norm(),
            1.e-8 * std::max(1., constraint_scale));
}

TEST(ElasticityCouplingIntegrationValidation,
     DynamicStrongConstraintsRemainFinite)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  configure_tensor_product_parameters(par);

  par.output_directory =
    TestPaths::output_directory("elasticity-issue-203-diagnostics");
  std::filesystem::create_directories(par.output_directory);
  par.time_parameters.initial_time                                = 0.;
  par.time_parameters.final_time                                  = 2.e-3;
  par.time_parameters.time_step                                   = 1.e-3;
  par.default_material_properties.rho                             = 1.;
  par.tensor_product_coupling_parameters.coupling_rhs_expressions = {
    "sin(2*pi*t)", "sin(2*pi*t)", "sin(2*pi*t)"};
  par.check_model_consistency();

  ElasticityProblem<2, 3> problem(par);
  ASSERT_NO_THROW(problem.run());

  EXPECT_TRUE(std::isfinite(problem.solution.block(0).l2_norm()));
  EXPECT_TRUE(std::isfinite(problem.solution.block(1).l2_norm()));
  EXPECT_LT(problem.solution.block(0).linfty_norm(), 1.e3);
}


TEST(ElasticityCouplingIntegrationValidation,
     BOTH_DynamicStrongConstraintsWithLocalRefinementRemainFinite)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  configure_tensor_product_parameters(par, "0; 0; 0");

  par.output_directory =
    TestPaths::output_directory("elasticity-issue-203-small");
  std::filesystem::create_directories(par.output_directory);
  par.time_parameters.initial_time    = 0.;
  par.time_parameters.final_time      = 2.e-2;
  par.time_parameters.time_step       = 1.e-3;
  par.default_material_properties.rho = 1.;
  par.tensor_product_coupling_parameters.refinement_parameters
    .refinement_strategy = "space";
  // On this small fixture, level two is the smallest cap that creates a
  // locally refined background mesh and therefore hanging-node constraints.
  par.tensor_product_coupling_parameters.refinement_parameters
    .max_refinement_level                                         = 2;
  par.tensor_product_coupling_parameters.coupling_rhs_expressions = {
    "sin(2*pi*t)", "sin(2*pi*t)", "sin(2*pi*t)"};
  par.check_model_consistency();

  ElasticityProblem<2, 3> problem(par);
  ASSERT_NO_THROW(problem.run());

  EXPECT_TRUE(std::isfinite(problem.solution.block(0).l2_norm()));
  EXPECT_TRUE(std::isfinite(problem.solution.block(1).l2_norm()));
  EXPECT_GT(problem.solution.block(0).linfty_norm(), 1.e-12);
  EXPECT_LT(problem.solution.block(0).linfty_norm(), 1.e3);
}


TEST(ElasticityCouplingIntegrationValidation,
     BOTH_NewmarkUsesDiscreteVelocityBoundaryData)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  configure_tensor_product_parameters(par, "0; 0; t");
  ParameterAcceptor::prm.parse_input_from_string(R"(
    subsection Functions
      subsection Right hand side
        set Function expression = 0; 0; 0
      end
      subsection Initial velocity
        set Function expression = 0; 0; 1
      end
    end
  )");
  ParameterAcceptor::parse_all_parameters();

  par.output_directory =
    TestPaths::output_directory("elasticity-newmark-discrete-bc");
  std::filesystem::create_directories(par.output_directory);
  par.time_parameters.initial_time                                = 0.;
  par.time_parameters.final_time                                  = 3.e-2;
  par.time_parameters.time_step                                   = 1.e-3;
  par.default_material_properties.rho                             = 1.;
  par.default_material_properties.rayleigh_alpha                  = 0.;
  par.tensor_product_coupling_parameters.coupling_rhs_expressions = {"0",
                                                                     "0",
                                                                     "0"};
  par.check_model_consistency();

  ElasticityProblem<2, 3> problem(par);
  ASSERT_NO_THROW(problem.run());

  EXPECT_TRUE(std::isfinite(problem.velocity.block(0).l2_norm()));
  EXPECT_TRUE(std::isfinite(problem.solution.block(0).l2_norm()));
  EXPECT_LT(problem.velocity.block(0).linfty_norm(), 2.);
  EXPECT_LT(problem.solution.block(0).linfty_norm(), 1.e-1);

  double       boundary_velocity_error = 0.;
  unsigned int nonzero_boundary_lines  = 0;
  for (const auto &line : problem.constraints.get_lines())
    if (line.inhomogeneity > 1.e-12 &&
        problem.velocity.block(0).in_local_range(line.index))
      {
        ++nonzero_boundary_lines;
        boundary_velocity_error =
          std::max(boundary_velocity_error,
                   std::abs(problem.velocity.block(0)[line.index] - 1.));
      }

  // Co-refinement can leave a rank without prescribed boundary DoFs. Retain
  // the nonzero-data oracle and check every owned row across all ranks.
  nonzero_boundary_lines =
    Utilities::MPI::sum(nonzero_boundary_lines, MPI_COMM_WORLD);
  boundary_velocity_error =
    Utilities::MPI::max(boundary_velocity_error, MPI_COMM_WORLD);
  ASSERT_GT(nonzero_boundary_lines, 0u);
  EXPECT_LT(boundary_velocity_error, 1.e-3);
}
