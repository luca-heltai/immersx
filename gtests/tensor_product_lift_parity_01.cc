// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>

#include <gtest/gtest.h>
#include <immersx/core/lifting.h>
#include <immersx/core/representation.h>
#include <immersx/core/state.h>
#include <immersx/coupling/tensor_product_lift.h>
#include <immersx/coupling/tensor_product_space.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>

#include "coupled_poisson_elasticity.h"
#include "test_paths.h"

#ifdef DEAL_II_WITH_VTK

using namespace ImmersX;
using namespace dealii;

namespace
{
  constexpr int reduced_dim  = 1;
  constexpr int surface_dim  = 2;
  constexpr int spacedim     = 3;
  constexpr int n_components = 1;

  /**
   * Configure both the legacy TensorProductSpace and the modern
   * TensorProductLift with exactly the same settings. The representative
   * domain is the one-cell straight line of data/tests/one_cylinder.vtk, from
   * (0.5, 0.5, 0) to (0.5, 0.5, 1).
   */
  void
  configure_common(
    TensorProductSpaceParameters<reduced_dim,
                                 surface_dim,
                                 spacedim,
                                 n_components> &old_parameters,
    TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>
      &new_lift)
  {
    // Representative (reduced) domain.
    old_parameters.fe_degree                = 1;
    old_parameters.quadrature_type          = "gauss";
    old_parameters.n_q_points               = 2;
    old_parameters.n_quadrature_repetitions = 1;
    old_parameters.thickness                = "0.2";
    old_parameters.reduced_grid_name =
      ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");

    // Cross section.
    old_parameters.section.inclusion_type           = "hyper_ball";
    old_parameters.section.refinement_level         = 1;
    old_parameters.section.inclusion_degree         = 0;
    old_parameters.section.selected_coefficients    = {0};
    old_parameters.section.quadrature_type          = "gauss";
    old_parameters.section.n_q_points               = 4;
    old_parameters.section.n_quadrature_repetitions = 1;

    new_lift.thickness                      = "0.2";
    new_lift.representative_quadrature_type = "gauss";
    new_lift.representative_n_q_points      = 2;
    new_lift.representative_n_repetitions   = 1;

    new_lift.section.inclusion_type           = "hyper_ball";
    new_lift.section.refinement_level         = 1;
    new_lift.section.inclusion_degree         = 0;
    new_lift.section.selected_coefficients    = {0};
    new_lift.section.quadrature_type          = "gauss";
    new_lift.section.n_q_points               = 4;
    new_lift.section.n_quadrature_repetitions = 1;
  }

  /** Build the Poisson source problem on the exact one_cylinder line mesh. */
  void
  configure_poisson(PoissonParameters<reduced_dim, spacedim> &parameters)
  {
    parameters.fe_degree          = 1;
    parameters.initial_refinement = 0;
    parameters.name_of_grid =
      ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
    parameters.arguments_for_grid = "";
    parameters.triangulation_type = "fullydistributed";
  }
} // namespace


/**
 * Phase-1 parity gate: with identical settings, the legacy
 * TensorProductSpace path and the modern TensorProductLift path must produce
 * the same representative quadrature, cross-section quadrature, lifted
 * physical points, lifted weights, total measure, and selected-mode values.
 */
TEST(TensorProductLiftParity, Mode0SingleLineCell) // NOLINT
{
  ParameterAcceptor::clear();

  TensorProductSpaceParameters<reduced_dim, surface_dim, spacedim, n_components>
    old_parameters;
  TensorProductLift<reduced_dim, surface_dim, spacedim, n_components> new_lift(
    "/Parity lift/");
  configure_common(old_parameters, new_lift);

  // ---- OLD PATH: reference TensorProductSpace implementation. ----
  TensorProductSpace<reduced_dim, surface_dim, spacedim, n_components>
    old_space(old_parameters);
  old_space.initialize();

  const auto &old_section = old_space.get_reference_cross_section();
  const auto &old_qpoints = old_space.get_locally_owned_qpoints();
  const auto &old_weights = old_space.get_locally_owned_weights();
  const auto &old_reduced_qpoints =
    old_space.get_locally_owned_reduced_qpoints();

  AffineConstraints<double> empty_constraints;
  empty_constraints.close();
  TensorProductRepresentation<reduced_dim, surface_dim, spacedim, n_components>
    old_representation(old_space,
                       old_space.get_dof_handler(),
                       old_space.get_dof_handler().locally_owned_dofs(),
                       DoFTools::extract_locally_relevant_dofs(
                         old_space.get_dof_handler()),
                       empty_constraints);

  // ---- NEW PATH: app pressure Representation on the same mesh, lifted with
  //      the modern parameterized descriptor. ----
  PoissonParameters<reduced_dim, spacedim> poisson_parameters;
  configure_poisson(poisson_parameters);
  initialize_parameters_from_string(R"(
    subsection Poisson
      subsection Right hand side
        set Function expression = 0
        set Variable names      = x,y,z,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names      = x,y,z,t
      end
      subsection Solver
        subsection Control
          set Max steps  = 10
          set Reduction  = 1.e-12
          set Tolerance  = 1.e-14
          set Log result = false
        end
      end
    end
  )");
  PoissonSolver<reduced_dim, spacedim> poisson_problem(poisson_parameters);
  poisson_problem.make_grid();
  poisson_problem.setup_fe();
  poisson_problem.setup_system();
  poisson_problem.assemble_system();

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor descriptor;
  descriptor.name           = "pressure";
  const auto pressure_field = layout.add_field(descriptor);

  CoupledPoissonElasticity::PressureRepresentation source_representation(
    pressure_field, poisson_problem, 1.0);
  const auto  lifted        = source_representation.lift(new_lift);
  const auto &lifted_points = lifted.lifted_points();
  const auto &support       = lifted.support();
  const auto &new_section   = support.reference_cross_section();
  const auto  source_points =
    source_representation.locally_owned_quadrature_points(
      support.representative_quadrature());

  // 1. Number of representative quadrature points.
  EXPECT_EQ(old_reduced_qpoints.size(), source_points.size());
  ASSERT_GT(old_reduced_qpoints.size(), 0u);

  // 2. Representative quadrature coordinates.
  for (unsigned int q = 0; q < old_reduced_qpoints.size(); ++q)
    for (unsigned int d = 0; d < spacedim; ++d)
      EXPECT_NEAR(old_reduced_qpoints[q][d], source_points[q].point[d], 1.e-12)
        << "Representative point " << q << ", coordinate " << d;

  // 3. Number of cross-section quadrature points.
  EXPECT_EQ(old_section.n_quadrature_points(),
            new_section.n_quadrature_points());

  // 4. Number of lifted physical points.
  EXPECT_EQ(old_qpoints.size(), lifted_points.size());
  ASSERT_GT(lifted_points.size(), 0u);

  // 5/6. Lifted physical coordinates and weights, in the same
  //      (representative qpoint, section qpoint) iteration order.
  ASSERT_EQ(old_qpoints.size(), old_weights.size());
  double old_total_measure = 0.;
  double new_total_measure = 0.;
  for (unsigned int q = 0; q < old_qpoints.size(); ++q)
    {
      for (unsigned int d = 0; d < spacedim; ++d)
        EXPECT_NEAR(old_qpoints[q][d], lifted_points[q].point[d], 1.e-12)
          << "Lifted point " << q << ", coordinate " << d;
      ASSERT_EQ(old_weights[q].size(), 1u);
      EXPECT_NEAR(old_weights[q][0], lifted_points[q].weight, 1.e-12)
        << "Lifted weight " << q;
      EXPECT_EQ(lifted_points[q].representative_qpoint,
                q / old_section.n_quadrature_points());
      EXPECT_EQ(lifted_points[q].section_qpoint,
                q % old_section.n_quadrature_points());
      old_total_measure += old_weights[q][0];
      new_total_measure += lifted_points[q].weight;
    }

  // 7. Total lifted measure.
  EXPECT_NEAR(old_total_measure, new_total_measure, 1.e-12);
  EXPECT_NEAR(new_total_measure, 1. * old_section.measure(0.2), 1.e-12);

  // 8. Selected modes.
  EXPECT_EQ(old_parameters.section.selected_coefficients,
            lifted_points.front().selected_modes);

  // 9. Cross-section mode values for mode 0 at every section quadrature point.
  for (unsigned int section_q = 0;
       section_q < old_section.n_quadrature_points();
       ++section_q)
    EXPECT_NEAR(old_section.shape_value(0, section_q, 0),
                lifted_points[section_q].mode_values[0],
                1.e-12)
      << "Mode value at section point " << section_q;

  // The legacy and modern algebraic spaces agree for a single scalar mode.
  EXPECT_EQ(old_space.get_dof_handler().n_dofs(), poisson_problem.n_dofs());

  // The legacy tensor-product view and the modern lift agree point-by-point.
  const auto old_view_points =
    old_representation.locally_owned_quadrature_points(
      Quadrature<surface_dim>());
  ASSERT_EQ(old_view_points.size(), lifted_points.size());
  for (unsigned int q = 0; q < old_view_points.size(); ++q)
    {
      EXPECT_NEAR(old_view_points[q].weight, lifted_points[q].weight, 1.e-12)
        << "View weight " << q;
      EXPECT_EQ(old_view_points[q].dof_indices.size(),
                lifted_points[q].selected_modes.size() *
                  source_representation.n_dofs_per_cell());
    }
}

#endif // DEAL_II_WITH_VTK
