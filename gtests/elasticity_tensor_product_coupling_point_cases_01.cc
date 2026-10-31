#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/physics/elasticity.h>

#include <array>
#include <cmath>
#include <iomanip>
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
 * paths with the tensor-product representation.
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
    return {{problem.solution.block(0).l2_norm(),
             problem.solution.block(1).l2_norm()}};
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

TEST(ElasticityTensorProductCouplingValidation,
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

// assemble_tensor_product_coupling() divides the datum of each mode by the
// maximum of the mode on the cross section, so that a datum is the
// displacement amplitude of its mode, as with point coupling. Without
// solving, check that the assembled rhs is the unscaled one divided by that
// maximum, and that the selected modes cos(theta) e_x and sin(theta) e_y still
// have the normalization sqrt(2) cos(theta) that the scaling assumes.
TEST(ElasticityTensorProductCouplingValidation, BOTH_RhsScaledByModeMaximum)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> par;
  set_issue_165_parameters(par,
                           "tensor_product_rhs_scaling",
                           "one_cylinder_rotation_x.vtk",
                           {2, 3, 4, 5},
                           {0, 1});
  par.initial_refinement = 2;
  par.tensor_product_coupling_parameters.tensor_product_space_parameters
    .section.refinement_level = 3;
  ElasticityProblem<3> problem(par);
  initialize_parameters();
  ParameterAcceptor::parse_all_parameters();

  problem.make_grid();
  problem.setup_fe();
  problem.setup_dofs();
  problem.assemble_coupling();

  const auto &coupling = *problem.tensor_product_coupling;
  const auto &basis =
    coupling.get_reference_cross_section().get_basis_functions();
  const auto &reduced_dh = coupling.get_dof_handler();
  ASSERT_EQ(basis.size(), 2u);

  const auto &scaled   = problem.force_rhs.block(1);
  auto        unscaled = scaled;
  unscaled             = 0;
  coupling.assemble_reduced_rhs(unscaled);
  ASSERT_GT(unscaled.l2_norm(), 0.);

  for (unsigned int c = 0; c < basis.size(); ++c)
    {
      // On the 32-sided polygon of Refinement level 3 the maximum is 1.4188.
      const double max_value = basis[c].linfty_norm();
      EXPECT_NEAR(max_value, std::sqrt(2.), 1e-2) << "mode " << c;
      for (const auto i : DoFTools::extract_dofs(
             reduced_dh,
             reduced_dh.get_fe().component_mask(FEValuesExtractors::Scalar(c))))
        EXPECT_NEAR(scaled[i] * max_value,
                    unscaled[i],
                    1e-12 * std::abs(unscaled[i]))
          << "mode " << c << ", dof " << i;
    }
}

TEST(ElasticityTensorProductCouplingValidation, ExactLambdaSingleSolve)
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

namespace
{
  // Number of cells after run() with n_cycles cycles of fixed_number
  // refinement, with the given space post-refinement cycles of the
  // co-refinement of the tensor-product coupling.
  types::global_cell_index
  n_cells_after_adaptive_run(const unsigned int n_cycles,
                             const unsigned int space_post_refinement_cycles)
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<3> par;
    set_tensor_product_defaults(par,
                                "tensor_product_adaptive_" +
                                  std::to_string(n_cycles) + "_" +
                                  std::to_string(space_post_refinement_cycles),
                                {"0.1", "0", "0"});
    par.refinement_strategy = "fixed_number";
    par.refinement_fraction = 0.3;
    par.n_refinement_cycles = n_cycles;
    par.tensor_product_coupling_parameters.refinement_parameters
      .space_post_refinement_cycles = space_post_refinement_cycles;
    ElasticityProblem<3> problem(par);
    initialize_parameters();
    ParameterAcceptor::parse_all_parameters();
    problem.run();
    return problem.tria->n_global_active_cells();
  }
} // namespace

// Adaptive refinement with tensor-product coupling: run() refines the mesh and
// rebuilds the coupling on it. With the default co-refinement parameters the
// rebuild does not refine the mesh again, which would invalidate the solution
// transfer and throw.
TEST(ElasticityTensorProductCouplingValidation,
     AdaptiveRefinementRebuildsCoupling)
{
  EXPECT_GT(n_cells_after_adaptive_run(2, 0), n_cells_after_adaptive_run(1, 0));
}

// When the co-refinement of the tensor-product coupling has space pre- or
// post-refinement cycles, it prevails over adaptive refinement: run() skips
// the remaining refinement cycles, so three cycles give the mesh of one.
TEST(ElasticityTensorProductCouplingValidation,
     CoRefinementSkipsAdaptiveRefinement)
{
  EXPECT_EQ(n_cells_after_adaptive_run(3, 1), n_cells_after_adaptive_run(1, 1));
}

namespace
{
  // Errors of an InflatedVessel solution against the exact solution, and the
  // mean radial displacement of the solution on the wall.
  struct InflatedVesselResult
  {
    double l2;
    double linfty;
    double h1;
    double wall_displacement;
  };

  // A straight vessel of radius ri = 0.2 along the axis of the cylinder of
  // radius re = 2 and half length 2 around the x axis is inflated by
  // di = 0.1. With u_x = 0 on the end faces and no x dependence, the problem
  // is plane strain in (y, z), and the exact displacement is the radial Lame
  // solution u_r = ri di / (re^2 - ri^2) (re^2 / r - r) outside the vessel
  // and u_r = di r / ri inside. It vanishes on the hull, where the Dirichlet
  // data is zero.
  //
  // With tensor-product coupling the selected indices 3 and 7 are the modes
  // cos(theta) e_x and sin(theta) e_y of the reference circle; with point
  // coupling they are the same Fourier coefficients of 100 inclusions of
  // length 0.04. In both cases the data 0.1 give the inflation di = 0.1.
  //
  // The exact solution has a kink at the wall, which a mesh that does not
  // resolve the wall smooths over a cell: the solution then looks like the
  // one of a larger vessel, about 25% too large everywhere with Initial
  // refinement 2 or 3, and refining only along the axis does not reduce
  // this. The mesh is therefore refined n_cycles - 1 times with the Kelly
  // estimator and fixed_number refinement, which marks the cells crossed by
  // the wall. The result of every cycle is returned.
  //
  // With tensor-product coupling the quadrature along the axis has 32
  // repetitions of a 3-point Gauss rule per reduced cell of length 1, so each
  // repetition lies inside one bulk cell, of length 1/32 after three
  // refinements, and the coupling is integrated exactly. With 16 repetitions
  // on cells of length 1/32, the mean wall displacement is 2% below the
  // imposed one.
  std::vector<InflatedVesselResult>
  solve_inflated_vessel(const bool point_coupling, const unsigned int n_cycles)
  {
    std::string inclusions;
    for (unsigned int i = 0; i < 100; ++i)
      inclusions += (i == 0 ? "" : "; ") + std::to_string(-1.98 + 0.04 * i) +
                    ", 0, 0, 0.04, 0, 0, 0.2, 0";
    const std::string name =
      std::string(point_coupling ? "point" : "tensor_product") +
      "_inflated_vessel_" + std::to_string(n_cycles);

    ParameterAcceptor::clear();
    ElasticityProblemParameters<3> par;
    ElasticityProblem<3>           problem(par);
    initialize_parameters();
    ParameterAcceptor::prm.parse_input_from_string(
      TestPaths::expand_configured_paths(R"(
    subsection Functions
      subsection Dirichlet boundary conditions
        set Function expression = 0; 0; 0
      end
      subsection Exact solution
        set Function constants  = ri=0.2, re=2, di=0.1
        set Function expression = 0; if((y^2+z^2) > (ri*ri), ri*di / (re*re-ri*ri) * (re*re*y/(y^2+z^2) - y), di*y/ri); if((y^2+z^2) > (ri*ri), ri*di / (re*re-ri*ri) * (re*re*z/(y^2+z^2) - z), di*z/ri)
        set Variable names      = x,y,z,t
      end
    end
    subsection Immersed Problem
      set Coupling type                      = )" +
                                         std::string(point_coupling ?
                                                       "Point" :
                                                       "TensorProduct") +
                                         R"(
      set Dirichlet boundary ids             = 0
      set Normal flux boundary ids           = 1,2
      set FE degree                          = 1
      set Initial refinement                 = 2
      set Output name                        = )" +
                                         name + R"(
      set Output results also before solving = false
      subsection Grid generation
        set Domain type              = generate
        set Grid generator           = subdivided_cylinder
        set Grid generator arguments = 4: 2: 2
        set Triangulation type       = distributed
      end
      subsection Immersed inclusions
        set Inclusions                        = )" +
                                         (point_coupling ? inclusions : "") +
                                         R"(
        set Inclusions refinement             = 400
        set Selection of Fourier coefficients = 3,7
        set Reference inclusion data          = 0,0,0,0.1,0,0,0,0.1,0
      end
      subsection Material properties
        subsection default
          set Lame lambda = 1
          set Lame mu     = 1
        end
      end
      subsection Refinement and remeshing
        set Strategy                    = fixed_number
        set Refinement fraction         = 0.3
        set Coarsening fraction         = 0
        set Maximum number of cells     = 2000000
        set Number of refinement cycles = )" +
                                         std::to_string(n_cycles) + R"(
      end
    end
    subsection Tensor product coupling
      subsection Cross section
        set Inclusion type           = hyper_ball
        set Maximum inclusion degree = 1
        set Refinement level         = 3
        set Selected indices         = 3, 7
      end
      subsection Particle coupling
        set RTree extraction level = 1
      end
      subsection Representative domain
        set Finite element degree            = 1
        set Number of quadrature points      = 3
        set Number of quadrature repetitions = 32
        set Reduced grid name                = @TEST_DATA_DIR@/tests/one_cylinder_rotation_x.vtk
        set Reduced right hand side          = 0.1; 0.1
        set Thickness                        = 0.2
      end
    end
    subsection Solvers
      subsection Augmented Lagrange
        set Max steps = 10000
        set Reduction = 1.e-10
        set Tolerance = 1.e-12
      end
      subsection Displacement
        set Max steps = 10000
        set Reduction = 1.e-10
        set Tolerance = 1.e-12
      end
    end
    )"));
    ParameterAcceptor::parse_all_parameters();
    par.output_directory =
      TestPaths::output_directory("elasticity-tensor-product-point");

    // The steps of run(), without the output and the stresses of each cycle.
    problem.make_grid();
    problem.setup_fe();
    if (point_coupling)
      problem.inclusions.setup_inclusions_particles(*problem.tria);
    problem.setup_dofs();

    // The errors against the exact solution are measured for |x| <= 1.9, away
    // from the end faces x = -2 and x = 2: there the vessel ends on the
    // boundary of the domain, which the plane-strain solution does not
    // describe. setup_dofs() reinitializes the solution after each
    // refinement, so it is looked up at every evaluation.
    const QGauss<3> quadrature(par.fe_degree + 2);
    const ScalarFunctionFromFunctionObject<3> weight(
      [](const Point<3> &p) { return std::abs(p[0]) <= 1.9 ? 1. : 0.; });
    const auto error = [&](const VectorTools::NormType norm) {
      Vector<float> cellwise_errors(problem.tria->n_active_cells());
      VectorTools::integrate_difference(problem.dh,
                                        problem.locally_relevant_solution.block(
                                          0),
                                        par.exact_solution,
                                        cellwise_errors,
                                        quadrature,
                                        norm,
                                        &weight);
      return VectorTools::compute_global_error(*problem.tria,
                                               cellwise_errors,
                                               norm);
    };

    // Mean radial displacement on the wall in the plane x = 0. The points are
    // evaluated in serial: VectorTools::point_value() needs the cell that
    // contains the point.
    const auto wall_displacement = [&]() {
      double             mean     = 0.;
      const unsigned int n_points = 64;
      for (unsigned int k = 0; k < n_points; ++k)
        {
          const double   theta = 2. * numbers::PI * (k + 0.5) / n_points;
          const Point<3> p(0., 0.2 * std::cos(theta), 0.2 * std::sin(theta));
          Vector<double> value(3);
          VectorTools::point_value(problem.dh,
                                   problem.locally_relevant_solution.block(0),
                                   p,
                                   value);
          mean +=
            (value[1] * std::cos(theta) + value[2] * std::sin(theta)) / n_points;
        }
      return mean;
    };

    std::vector<InflatedVesselResult> results;
    for (problem.cycle = 0; problem.cycle < par.n_refinement_cycles;
         ++problem.cycle)
      {
        problem.assemble_elasticity_system();
        problem.assemble_coupling();
        problem.solve_static();

        results.push_back({error(VectorTools::L2_norm),
                           error(VectorTools::Linfty_norm),
                           error(VectorTools::H1_norm),
                           wall_displacement()});
        const auto &result = results.back();
        problem.pcout << std::setprecision(10) << name << ", cycle "
                      << problem.cycle << ": " << problem.dh.n_dofs()
                      << " dofs, L2 " << result.l2 << " Linfty "
                      << result.linfty << " H1 " << result.h1
                      << " wall displacement " << result.wall_displacement
                      << std::endl;

        if (problem.cycle != par.n_refinement_cycles - 1)
          problem.refine_and_transfer();
      }
    return results;
  }
} // namespace

// Tensor-product counterpart of ElasticityValidation.ExactLambda: the
// inflated vessel of solve_inflated_vessel(), with four cycles of adaptive
// refinement. From the third to the fourth cycle the errors decrease at the
// rates of an immersed method with a kink at the wall, and the wall moves by
// the imposed 0.1 on average. The test runs in serial only, because the wall
// mean uses VectorTools::point_value(), so adaptive refinement with
// tensor-product coupling is not tested here on more than one rank.
TEST(ElasticityTensorProductCouplingValidation, ExactSolution)
{
  const auto results = solve_inflated_vessel(false, 4);
  ASSERT_EQ(results.size(), 4u);
  const auto &coarse = results[2];
  const auto &fine   = results[3];

  EXPECT_GT(coarse.l2 / fine.l2, 2.);
  EXPECT_GT(coarse.h1 / fine.h1, 1.25);
  EXPECT_NEAR(fine.wall_displacement, 0.1, 3e-3);

  // Upper bounds on the errors of the finer solve, not theoretical estimates.
  // A serial Debug run with deal.II 9.7.1 (142215 dofs) gave L2 6.56e-3,
  // Linfty 5.63e-3 and H1 0.188; the bounds leave at least 20% margin.
  EXPECT_LT(fine.l2, 8.2e-3);
  EXPECT_LT(fine.linfty, 7.0e-3);
  EXPECT_LT(fine.h1, 0.235);
}

// With the same adaptive refinement, point and tensor-product coupling impose
// the same inflation of the vessel and approximate the exact solution equally
// well.
TEST(ElasticityTensorProductCouplingValidation, PointAndTensorProductExactSolution)
{
  const auto point  = solve_inflated_vessel(true, 2).back();
  const auto tensor = solve_inflated_vessel(false, 2).back();

  EXPECT_NEAR(point.wall_displacement, 0.1, 1e-3);
  EXPECT_NEAR(tensor.wall_displacement, 0.1, 1e-3);
  EXPECT_NEAR(tensor.l2, point.l2, 0.1 * point.l2);
  EXPECT_NEAR(tensor.linfty, point.linfty, 0.1 * point.linfty);
  EXPECT_NEAR(tensor.h1, point.h1, 0.1 * point.h1);
}

INSTANTIATE_TEST_SUITE_P(TriangulationBackendsValidation,
                         ElasticityTensorProductCouplingTriangulationTypeTest,
                         ::testing::Values("distributed"));

#endif // DEAL_II_WITH_VTK
