#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "elasticity.h"
#include "utils.h"

#ifdef DEAL_II_WITH_VTK

using namespace dealii;

/*
 * The legacy tests describe point/segment inclusions through Inclusions. The
 * reduced path instead reads a VTK centerline and combines it with a reference
 * cross section, so the checked-in one_cylinder.vtk is the common
 * representative geometry for these counterparts. The legacy numerical norms
 * are therefore not reused here; these tests verify that the corresponding
 * load, material, backend, and solve paths remain operational. Reduced coupling
 * also supports only one setup/solve cycle at present, so the ExactLambda
 * counterpart uses a single solve rather than the legacy adaptive refinement
 * sequence.
 */
namespace
{
  class ReducedElasticityTriangulationTypeTest
    : public ::testing::TestWithParam<const char *>
  {};

  std::string
  current_triangulation_type(
    const ReducedElasticityTriangulationTypeTest &test_instance)
  {
    return test_instance.GetParam();
  }

  template <int dim, int spacedim>
  void
  set_reduced_defaults(ElasticityProblemParameters<dim, spacedim> &par,
                       const std::string                          &output_name,
                       const std::vector<std::string>             &rhs)
  {
#  ifdef DEBUG
    par.output_directory = "tests_debug_output";
#  else
    par.output_directory = "tests_release_output";
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

    par.use_reduced_coupling = true;
    auto &reduced_parameters = par.reduced_coupling_parameters;
    reduced_parameters.tensor_product_space_parameters.reduced_grid_name =
      SOURCE_DIR "/data/tests/one_cylinder.vtk";
    reduced_parameters.tensor_product_space_parameters.fe_degree  = 1;
    reduced_parameters.tensor_product_space_parameters.n_q_points = 4;
    reduced_parameters.tensor_product_space_parameters.thickness  = 0.05;
    reduced_parameters.tensor_product_space_parameters.section.inclusion_type =
      "hyper_ball";
    reduced_parameters.tensor_product_space_parameters.section
      .inclusion_degree = 0;
    reduced_parameters.tensor_product_space_parameters.section
      .refinement_level = 2;
    reduced_parameters.tensor_product_space_parameters.section
      .selected_coefficients.clear();
    reduced_parameters.coupling_rhs_expressions = rhs;
  }

  template <int dim, int spacedim>
  void
  assert_reduced_solution(const ElasticityProblem<dim, spacedim> &problem)
  {
    ASSERT_EQ(problem.solution.n_blocks(), 2u);
    EXPECT_TRUE(std::isfinite(problem.solution.block(0).l2_norm()));
    EXPECT_TRUE(std::isfinite(problem.solution.block(1).l2_norm()));
    EXPECT_GT(problem.solution.block(1).l2_norm(), 0.0);
  }

  template <int dim, int spacedim>
  void
  run_reduced_case(ElasticityProblemParameters<dim, spacedim> &par)
  {
    ElasticityProblem<dim, spacedim> problem(par);
    initialize_parameters();
    ParameterAcceptor::parse_all_parameters();
    ASSERT_NO_THROW(problem.run());
    assert_reduced_solution(problem);
  }
} // namespace

TEST_P(ReducedElasticityTriangulationTypeTest, MPI_DisplacementX)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  set_reduced_defaults(par, "reduced_displacement_x", {"1", "0", "0"});
  par.triangulation_type = current_triangulation_type(*this);
  run_reduced_case(par);
}

TEST_P(ReducedElasticityTriangulationTypeTest, MPI_DisplacementY)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  set_reduced_defaults(par, "reduced_displacement_y", {"0", "1", "0"});
  par.triangulation_type = current_triangulation_type(*this);
  run_reduced_case(par);
}

TEST_P(ReducedElasticityTriangulationTypeTest, MPI_DisplacementXScaled)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  set_reduced_defaults(par, "reduced_displacement_x_scaled", {"0.1", "0", "0"});
  par.triangulation_type = current_triangulation_type(*this);
  run_reduced_case(par);
}

TEST_P(ReducedElasticityTriangulationTypeTest, MPI_DisplacementYScaled)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  set_reduced_defaults(par, "reduced_displacement_y_scaled", {"0", "0.1", "0"});
  par.triangulation_type = current_triangulation_type(*this);
  run_reduced_case(par);
}

TEST_P(ReducedElasticityTriangulationTypeTest, MPI_RotationAboutXAndZ)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par_x;
  set_reduced_defaults(par_x, "reduced_rotation_x", {"1", "0", "0"});
  par_x.triangulation_type = current_triangulation_type(*this);
  run_reduced_case(par_x);

  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par_z;
  set_reduced_defaults(par_z, "reduced_rotation_z", {"0", "0", "1"});
  par_z.triangulation_type = current_triangulation_type(*this);
  run_reduced_case(par_z);
}

TEST_P(ReducedElasticityTriangulationTypeTest,
       MPI_DisplacementAlongVtkCenterline)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par;
  set_reduced_defaults(par,
                       "reduced_displacement_vtk_centerline",
                       {"1", "0", "0"});
  par.triangulation_type                  = current_triangulation_type(*this);
  par.default_material_properties.Lame_mu = 2.0;
  par.default_material_properties.Lame_lambda = 50.0;
  run_reduced_case(par);
}

TEST_P(ReducedElasticityTriangulationTypeTest,
       MPI_DisplacementAlongVtkCenterlineWithSegmentClustering)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par;
  set_reduced_defaults(par,
                       "reduced_displacement_vtk_centerline_segments",
                       {"1", "0", "0"});
  par.triangulation_type                  = current_triangulation_type(*this);
  par.default_material_properties.Lame_mu = 2.0;
  par.default_material_properties.Lame_lambda = 50.0;
  run_reduced_case(par);
}

TEST(ReducedElasticity, DisplacementAlongVtkCenterlineWithSegmentClustering)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par;
  set_reduced_defaults(par,
                       "reduced_displacement_vtk_centerline_segments_serial",
                       {"1", "0", "0"});
  par.triangulation_type                      = "distributed";
  par.default_material_properties.Lame_mu     = 2.0;
  par.default_material_properties.Lame_lambda = 50.0;
  run_reduced_case(par);
}

TEST(ReducedElasticity, ExactLambdaSingleSolve)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  set_reduced_defaults(par,
                       "reduced_exact_lambda_single_solve",
                       {"0", "1", "0"});
  par.triangulation_type = "distributed";
  run_reduced_case(par);
}

TEST_P(ReducedElasticityTriangulationTypeTest, MPI_TwoReducedSegmentsInCell)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par;
  set_reduced_defaults(par,
                       "reduced_two_segments_in_cell",
                       {"0.1", "0.2", "0.3"});
  par.triangulation_type = current_triangulation_type(*this);
  run_reduced_case(par);
}

TEST_P(ReducedElasticityTriangulationTypeTest,
       MPI_MultipleReducedSegmentsInCell)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par;
  set_reduced_defaults(par,
                       "reduced_multiple_segments_in_cell",
                       {"0.1", "0.2", "0.3"});
  par.triangulation_type                  = current_triangulation_type(*this);
  par.default_material_properties.Lame_mu = 2.0;
  par.default_material_properties.Lame_lambda = 50.0;
  run_reduced_case(par);
}

INSTANTIATE_TEST_SUITE_P(TriangulationBackends,
                         ReducedElasticityTriangulationTypeTest,
                         ::testing::Values("distributed", "fullydistributed"));

#endif // DEAL_II_WITH_VTK
