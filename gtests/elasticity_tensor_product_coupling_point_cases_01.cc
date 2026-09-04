#include <gtest/gtest.h>
#include <immersx/physics/elasticity.h>

#include <array>
#include <cmath>
#include <set>
#include <string>
#include <vector>

using namespace ImmersX;
#include <immersx/io/utils.h>

#include "test_paths.h"

#ifdef DEAL_II_WITH_VTK

using namespace dealii;

/*
 * These tests exercise tensor-product counterparts of point-coupling cases.
 * They use a VTK centerline together with a reference cross section and
 * therefore verify the corresponding load, material, backend, and solve
 * paths with the tensor-product representation. Tensor-product coupling
 * currently supports one setup/solve cycle, so the ExactLambda counterpart
 * uses a single solve instead of adaptive refinement.
 */
namespace
{
  class ElasticityTensorProductCouplingTriangulationTypeTest
    : public ::testing::TestWithParam<const char *>
  {};

  std::string
  current_triangulation_type(
    const ElasticityTensorProductCouplingTriangulationTypeTest &test_instance)
  {
    return test_instance.GetParam();
  }

  template <int dim, int spacedim>
  void
  set_tensor_product_defaults(ElasticityProblemParameters<dim, spacedim> &par,
                              const std::string              &output_name,
                              const std::vector<std::string> &rhs,
                              const std::string &grid_name = "one_cylinder.vtk")
  {
#  ifdef DEBUG
    par.output_directory =
      ImmersX::TestPaths::output_directory("elasticity-tensor-product-point");
#  else
    par.output_directory =
      ImmersX::TestPaths::output_directory("elasticity-tensor-product-point");
#  endif
    par.output_name         = output_name;
    par.fe_degree           = 1;
    par.initial_refinement  = 2;
    par.domain_type         = "generate";
    par.name_of_grid        = "hyper_cube";
    par.arguments_for_grid  = "-1: 1: false";
    par.triangulation_type  = "distributed";
    par.n_refinement_cycles = 1;
    par.max_cells           = 20000;
    par.dirichlet_ids.clear();
    for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
      par.dirichlet_ids.insert(boundary_id);

    par.default_material_properties.Lame_mu     = 1;
    par.default_material_properties.Lame_lambda = 1;
    par.displacement_solver_control.set_reduction(1e-10);
    par.displacement_solver_control.set_tolerance(1e-10);
    par.reduced_mass_solver_control.set_reduction(1e-10);
    par.reduced_mass_solver_control.set_tolerance(1e-10);

    par.coupling_type               = CouplingType::TensorProduct;
    auto &tensor_product_parameters = par.tensor_product_coupling_parameters;
    tensor_product_parameters.tensor_product_space_parameters
      .reduced_grid_name =
      ImmersX::TestPaths::data_filename("tests/" + grid_name);
    tensor_product_parameters.tensor_product_space_parameters.fe_degree  = 1;
    tensor_product_parameters.tensor_product_space_parameters.n_q_points = 2;
    tensor_product_parameters.tensor_product_space_parameters.thickness =
      "0.05";
    tensor_product_parameters.tensor_product_space_parameters.section
      .inclusion_type = "hyper_ball";
    tensor_product_parameters.tensor_product_space_parameters.section
      .inclusion_degree = 0;
    tensor_product_parameters.tensor_product_space_parameters.section
      .refinement_level = 1;
    tensor_product_parameters.tensor_product_space_parameters.section
      .selected_coefficients.clear();
    tensor_product_parameters.coupling_rhs_expressions = rhs;
  }

  template <int dim, int spacedim>
  void
  assert_tensor_product_solution(
    const ElasticityProblem<dim, spacedim> &problem)
  {
    ASSERT_EQ(problem.solution.n_blocks(), 2u);
    EXPECT_TRUE(std::isfinite(problem.solution.block(0).l2_norm()));
    EXPECT_TRUE(std::isfinite(problem.solution.block(1).l2_norm()));
    EXPECT_GT(problem.solution.block(1).l2_norm(), 0.0);
  }

  template <int dim, int spacedim>
  std::array<double, 2>
  run_tensor_product_case(ElasticityProblemParameters<dim, spacedim> &par)
  {
    ElasticityProblem<dim, spacedim> problem(par);
    initialize_parameters();
    ParameterAcceptor::parse_all_parameters();
    try
      {
        problem.run();
      }
    catch (const std::exception &exception)
      {
        ADD_FAILURE() << exception.what();
        return {};
      }
    assert_tensor_product_solution(problem);
    return {problem.solution.block(0).l2_norm(),
            problem.solution.block(1).l2_norm()};
  }

  void
  set_issue_165_parameters(ElasticityProblemParameters<3>     &par,
                           const std::string                  &output_name,
                           const std::string                  &grid_name,
                           const std::set<types::boundary_id> &dirichlet_ids,
                           const std::set<types::boundary_id> &normal_flux_ids)
  {
    par.output_directory =
      ImmersX::TestPaths::output_directory("elasticity-tensor-product-point");
    par.output_name         = output_name;
    par.fe_degree           = 1;
    par.initial_refinement  = 4;
    par.domain_type         = "generate";
    par.name_of_grid        = "hyper_cube";
    par.arguments_for_grid  = "-2: 2: true";
    par.triangulation_type  = "distributed";
    par.refinement_strategy = "global";
    par.refinement_fraction = 1.0;
    par.n_refinement_cycles = 1;
    par.max_cells           = 2000000;
    par.dirichlet_ids       = dirichlet_ids;
    par.normal_flux_ids     = normal_flux_ids;

    par.default_material_properties.Lame_mu     = 1;
    par.default_material_properties.Lame_lambda = 1;
    par.displacement_solver_control.set_reduction(1.e-8);
    par.displacement_solver_control.set_tolerance(1.e-8);
    par.reduced_mass_solver_control.set_reduction(1.e-8);
    par.reduced_mass_solver_control.set_tolerance(1.e-8);

    par.coupling_type               = CouplingType::TensorProduct;
    auto &tensor_product_parameters = par.tensor_product_coupling_parameters;
    auto &space_parameters =
      tensor_product_parameters.tensor_product_space_parameters;
    space_parameters.reduced_grid_name =
      ImmersX::TestPaths::data_filename("tests/" + grid_name);
    space_parameters.fe_degree                     = 1;
    space_parameters.n_q_points                    = 3;
    space_parameters.n_quadrature_repetitions      = 4;
    space_parameters.thickness                     = "0.2";
    space_parameters.input_file_fields             = "displacement";
    space_parameters.section.inclusion_type        = "hyper_ball";
    space_parameters.section.inclusion_degree      = 1;
    space_parameters.section.refinement_level      = 1;
    space_parameters.section.selected_coefficients = {3, 7};
    tensor_product_parameters.refinement_parameters
      .embedded_pre_refinement_cycles = 0;
    tensor_product_parameters.refinement_parameters
      .embedded_post_refinement_cycles = 0;
    tensor_product_parameters.particle_coupling_parameters
      .rtree_extraction_level                          = 1;
    tensor_product_parameters.coupling_rhs_expressions = {"0.1", "0.1"};
  }
} // namespace

TEST_P(ElasticityTensorProductCouplingTriangulationTypeTest, MPI_DisplacementX)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  set_tensor_product_defaults(par,
                              "tensor_product_displacement_x",
                              {"1", "0", "0"});
  par.triangulation_type = current_triangulation_type(*this);
  run_tensor_product_case(par);
}

TEST_P(ElasticityTensorProductCouplingTriangulationTypeTest, MPI_DisplacementY)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  set_tensor_product_defaults(par,
                              "tensor_product_displacement_y",
                              {"0", "1", "0"});
  par.triangulation_type = current_triangulation_type(*this);
  run_tensor_product_case(par);
}

TEST_P(ElasticityTensorProductCouplingTriangulationTypeTest,
       MPI_DisplacementXScaled)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  set_tensor_product_defaults(par,
                              "tensor_product_displacement_x_scaled",
                              {"0.1", "0", "0"});
  par.triangulation_type = current_triangulation_type(*this);
  run_tensor_product_case(par);
}

TEST_P(ElasticityTensorProductCouplingTriangulationTypeTest,
       MPI_DisplacementYScaled)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  set_tensor_product_defaults(par,
                              "tensor_product_displacement_y_scaled",
                              {"0", "0.1", "0"});
  par.triangulation_type = current_triangulation_type(*this);
  run_tensor_product_case(par);
}

TEST_P(ElasticityTensorProductCouplingTriangulationTypeTest,
       MPI_RotationAboutXAndZ)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par_x;
  set_issue_165_parameters(par_x,
                           "tensor_product_issue_165_rotation_x",
                           "one_cylinder_rotation_x.vtk",
                           {2, 3, 4, 5},
                           {0, 1});
  par_x.triangulation_type = current_triangulation_type(*this);
  const auto x_norms       = run_tensor_product_case(par_x);

  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par_z;
  set_issue_165_parameters(par_z,
                           "tensor_product_issue_165_rotation_z",
                           "one_cylinder_rotation_z.vtk",
                           {0, 1, 2, 3},
                           {4, 5});
  par_z.triangulation_type = current_triangulation_type(*this);
  const auto z_norms       = run_tensor_product_case(par_z);

  EXPECT_NEAR(x_norms[0], z_norms[0], 1.e-8);
  EXPECT_NEAR(x_norms[1], z_norms[1], 1.e-8);
}

TEST_P(ElasticityTensorProductCouplingTriangulationTypeTest,
       MPI_DisplacementAlongVtkCenterline)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par;
  set_tensor_product_defaults(par,
                              "tensor_product_displacement_vtk_centerline",
                              {"1", "0", "0"});
  par.triangulation_type                  = current_triangulation_type(*this);
  par.default_material_properties.Lame_mu = 2.0;
  par.default_material_properties.Lame_lambda = 50.0;
  run_tensor_product_case(par);
}

TEST_P(ElasticityTensorProductCouplingTriangulationTypeTest,
       MPI_DisplacementAlongVtkCenterlineWithSegmentClustering)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par;
  set_tensor_product_defaults(
    par,
    "tensor_product_displacement_vtk_centerline_segments",
    {"1", "0", "0"});
  par.triangulation_type                  = current_triangulation_type(*this);
  par.default_material_properties.Lame_mu = 2.0;
  par.default_material_properties.Lame_lambda = 50.0;
  run_tensor_product_case(par);
}

TEST(ElasticityTensorProductCoupling,
     DISABLED_DisplacementAlongVtkCenterlineWithSegmentClustering)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par;
  set_tensor_product_defaults(
    par,
    "tensor_product_displacement_vtk_centerline_segments_serial",
    {"1", "0", "0"});
  par.triangulation_type                      = "distributed";
  par.default_material_properties.Lame_mu     = 2.0;
  par.default_material_properties.Lame_lambda = 50.0;
  run_tensor_product_case(par);
}

TEST(ElasticityTensorProductCoupling, ExactLambdaSingleSolve)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  set_tensor_product_defaults(par,
                              "tensor_product_exact_lambda_single_solve",
                              {"0", "1", "0"});
  par.triangulation_type = "distributed";
  run_tensor_product_case(par);
}

TEST_P(ElasticityTensorProductCouplingTriangulationTypeTest,
       MPI_TwoSegmentsInCell)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par;
  set_tensor_product_defaults(par,
                              "tensor_product_two_segments_in_cell",
                              {"0.1", "0.2", "0.3"});
  par.triangulation_type = current_triangulation_type(*this);
  run_tensor_product_case(par);
}

TEST_P(ElasticityTensorProductCouplingTriangulationTypeTest,
       MPI_MultipleSegmentsInCell)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par;
  set_tensor_product_defaults(par,
                              "tensor_product_multiple_segments_in_cell",
                              {"0.1", "0.2", "0.3"});
  par.triangulation_type                  = current_triangulation_type(*this);
  par.default_material_properties.Lame_mu = 2.0;
  par.default_material_properties.Lame_lambda = 50.0;
  run_tensor_product_case(par);
}

INSTANTIATE_TEST_SUITE_P(TriangulationBackends,
                         ElasticityTensorProductCouplingTriangulationTypeTest,
                         ::testing::Values("distributed"));

#endif // DEAL_II_WITH_VTK
