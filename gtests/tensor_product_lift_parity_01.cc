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

#include <deal.II/fe/fe_system.h>

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

  /** Build the same one-cell straight line programmatically. */
  void
  make_representative_line(Triangulation<reduced_dim, spacedim> &tria)
  {
    std::vector<Point<spacedim>> vertices = {Point<spacedim>(0.5, 0.5, 0.0),
                                             Point<spacedim>(0.5, 0.5, 1.0)};
    std::vector<CellData<reduced_dim>> cells(1);
    // one_cylinder.vtk stores the cell with connectivity (1, 0), so the
    // programmatic mesh must use the same orientation (vertex 0 at z = 1) for
    // point-by-point parity.
    cells[0].vertices[0] = 1;
    cells[0].vertices[1] = 0;
    tria.create_triangulation(vertices, cells, SubCellData());
  }

  /** Owns the programmatic one-cell line mesh and a modal FE space. */
  struct ModalLineSource
  {
    explicit ModalLineSource(const unsigned int n_modes)
      : fe(FE_Q<reduced_dim, spacedim>(1), n_modes)
      , dof_handler(tria)
    {
      Triangulation<reduced_dim, spacedim> serial;
      make_representative_line(serial);
      tria.copy_triangulation(serial);
      dof_handler.distribute_dofs(fe);
      constraints.close();
    }

    parallel::fullydistributed::Triangulation<reduced_dim, spacedim> tria{
      MPI_COMM_SELF};
    FESystem<reduced_dim, spacedim>   fe;
    DoFHandler<reduced_dim, spacedim> dof_handler;
    AffineConstraints<double>         constraints;
  };

  /**
   * Build a modal (CASE B) source view that exposes one algebraic slot per
   * (local DoF, mode) in component-major order.
   */
  FiniteElementRepresentation<reduced_dim, spacedim>
  make_modal_source(ModalLineSource &fixture)
  {
    return FiniteElementRepresentation<reduced_dim, spacedim>(
      fixture.tria,
      fixture.dof_handler,
      fixture.dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(fixture.dof_handler),
      fixture.constraints,
      dealii::StaticMappingQ1<reduced_dim, spacedim>::mapping,
      dealii::FEValuesExtractors::Scalar(0),
      ImmersX::RepresentationMetadata(),
      /*all_components=*/true);
  }

  /** Configure a two-mode lift/section for the multi-mode parity tests. */
  void
  configure_two_modes(
    TensorProductSpaceParameters<reduced_dim,
                                 surface_dim,
                                 spacedim,
                                 n_components> &old_parameters,
    TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>
      &new_lift)
  {
    old_parameters.fe_degree                = 1;
    old_parameters.quadrature_type          = "gauss";
    old_parameters.n_q_points               = 2;
    old_parameters.n_quadrature_repetitions = 1;
    old_parameters.thickness                = "0.2";
    old_parameters.reduced_grid_name =
      ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");

    old_parameters.section.inclusion_type           = "hyper_ball";
    old_parameters.section.refinement_level         = 1;
    old_parameters.section.inclusion_degree         = 1;
    old_parameters.section.selected_coefficients    = {0, 1};
    old_parameters.section.quadrature_type          = "gauss";
    old_parameters.section.n_q_points               = 4;
    old_parameters.section.n_quadrature_repetitions = 1;

    new_lift.thickness                      = "0.2";
    new_lift.representative_quadrature_type = "gauss";
    new_lift.representative_n_q_points      = 2;
    new_lift.representative_n_repetitions   = 1;

    new_lift.section.inclusion_type           = "hyper_ball";
    new_lift.section.refinement_level         = 1;
    new_lift.section.inclusion_degree         = 1;
    new_lift.section.selected_coefficients    = {0, 1};
    new_lift.section.quadrature_type          = "gauss";
    new_lift.section.n_q_points               = 4;
    new_lift.section.n_quadrature_repetitions = 1;
  }

  /**
   * Minimal modern source that owns a symbolic thickness evaluation, used to
   * exercise the source-provider seam without any imported-field storage.
   */
  struct SourceProvidedThickness
  {
    using value_type      = dealii::Vector<double>;
    using state_type      = ImmersX::ImmersXLA::MPI::Vector;
    using QuadraturePoint = ImmersX::RepresentationQuadraturePoint<3, double>;
    using TriangulationType =
      dealii::parallel::TriangulationBase<reduced_dim, spacedim>;
    using DoFHandlerType = dealii::DoFHandler<reduced_dim, spacedim>;
    using ExtractorType  = dealii::FEValuesExtractors::Scalar;

    std::vector<QuadraturePoint>
    locally_owned_quadrature_points(
      const dealii::Quadrature<1> &quadrature) const
    {
      std::vector<QuadraturePoint> result;
      result.reserve(quadrature.size());
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          QuadraturePoint point;
          point.point      = dealii::Point<3>(0.5, 0.5, quadrature.point(q)[0]);
          point.tangent[2] = 1.;
          point.weight     = 0.5;
          point.dof_indices  = {0};
          point.basis_values = {1.};
          result.push_back(point);
        }
      return result;
    }

    double
    evaluate_thickness(const dealii::Point<3> &point,
                       const double,
                       const std::vector<double> &) const
    {
      return 0.1 + 0.5 * point[2];
    }
  };
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


/**
 * CASE A guard: a physical (non-modal) source field owns one coefficient set.
 * Requesting more than mode 0 on such a source must fail instead of silently
 * aliasing several modes to the same algebraic coefficient.
 */
TEST(TensorProductLift, ScalarSourceRejectsModalExpansion) // NOLINT
{
  ParameterAcceptor::clear();

  TensorProductLift<reduced_dim, surface_dim, spacedim, n_components> lift(
    "/Scalar source/");
  TensorProductSpaceParameters<reduced_dim, surface_dim, spacedim, n_components>
    old_parameters;
  configure_two_modes(old_parameters, lift);

  ModalLineSource fixture(1);
  const auto      source = make_modal_source(fixture);
  const auto      lifted = source.lift(lift);

  EXPECT_THROW(lifted.locally_owned_quadrature_points(
                 Quadrature<surface_dim>()),
               dealii::ExceptionBase);
}


/**
 * CASE B semantics: a modal source owns one coefficient per (representative
 * DoF, mode). The modal lift must keep those algebraic indices distinct and
 * multiply each mode's reduced basis by the corresponding section mode.
 */
TEST(TensorProductLift, ModalSourceDistinctCoefficients) // NOLINT
{
  ParameterAcceptor::clear();

  TensorProductLift<reduced_dim, surface_dim, spacedim, n_components> lift(
    "/Modal source/");
  TensorProductSpaceParameters<reduced_dim, surface_dim, spacedim, n_components>
    old_parameters;
  configure_two_modes(old_parameters, lift);

  ModalLineSource fixture(2);
  const auto      source     = make_modal_source(fixture);
  const auto      modal_lift = ImmersX::make_modal_lift(source, lift);
  EXPECT_TRUE(modal_lift.modal());

  const auto &support = modal_lift.support();
  const auto &section = support.reference_cross_section();
  const auto  source_points =
    source.locally_owned_quadrature_points(support.representative_quadrature());
  const auto lifted_points =
    modal_lift.locally_owned_quadrature_points(Quadrature<surface_dim>());

  ASSERT_FALSE(lifted_points.empty());
  ASSERT_FALSE(source_points.empty());
  const unsigned int n_modes = section.n_selected_basis();
  const unsigned int n_scalar_dofs =
    source_points.front().dof_indices.size() / n_modes;
  ASSERT_EQ(n_scalar_dofs * n_modes, source_points.front().dof_indices.size());

  for (unsigned int q = 0; q < lifted_points.size(); ++q)
    {
      const unsigned int representative_q = q / section.n_quadrature_points();
      const unsigned int section_q        = q % section.n_quadrature_points();
      ASSERT_EQ(lifted_points[q].dof_indices.size(), n_scalar_dofs * n_modes);
      for (unsigned int mode = 0; mode < n_modes; ++mode)
        for (unsigned int i = 0; i < n_scalar_dofs; ++i)
          {
            const unsigned int slot = mode * n_scalar_dofs + i;
            // Mode 0 and mode 1 must be independent algebraic unknowns.
            if (mode > 0)
              EXPECT_NE(lifted_points[q].dof_indices[slot],
                        lifted_points[q].dof_indices[i])
                << "Mode " << mode << " aliases mode 0 at point " << q;
            EXPECT_NEAR(lifted_points[q].basis_values[slot],
                        source_points[representative_q].basis_values[slot] *
                          section.shape_value(mode, section_q, 0),
                        1.e-12)
              << "Lifted basis at point " << q << ", slot " << slot;
          }
    }
}


/**
 * A symbolic thickness expression without a source-side provider must fail
 * with a clear error instead of silently using a fallback thickness.
 */
TEST(TensorProductLift, SymbolicThicknessWithoutProviderFails) // NOLINT
{
  ParameterAcceptor::clear();

  TensorProductLift<reduced_dim, surface_dim, spacedim, n_components> lift(
    "/No provider/");
  TensorProductSpaceParameters<reduced_dim, surface_dim, spacedim, n_components>
    old_parameters;
  configure_two_modes(old_parameters, lift);
  lift.section.selected_coefficients = {0};
  lift.thickness                     = "radius";

  ModalLineSource fixture(1);
  const auto      source = make_modal_source(fixture);
  EXPECT_THROW(source.lift(lift), dealii::ExceptionBase);
}


/**
 * The modern source-provider seam: a source that naturally owns a thickness
 * evaluation provides per-point values for a symbolic expression. The lift
 * never reads the file or stores the field; it only forwards the evaluation.
 */
TEST(TensorProductLift, SourceProvidedSymbolicThickness) // NOLINT
{
  ParameterAcceptor::clear();

  TensorProductLift<reduced_dim, surface_dim, spacedim, n_components> lift(
    "/Source thickness/");
  TensorProductSpaceParameters<reduced_dim, surface_dim, spacedim, n_components>
    old_parameters;
  configure_two_modes(old_parameters, lift);
  lift.section.selected_coefficients = {0};
  lift.thickness                     = "radius";

  SourceProvidedThickness source;
  const auto              lifted  = ImmersX::make_lift(source, lift);
  const auto             &support = lifted.support();
  const auto             &section = support.reference_cross_section();
  const auto              source_points =
    source.locally_owned_quadrature_points(support.representative_quadrature());
  const auto &lifted_points = lifted.lifted_points();

  ASSERT_FALSE(source_points.empty());
  ASSERT_EQ(lifted_points.size(),
            source_points.size() * section.n_quadrature_points());
  for (unsigned int q = 0; q < source_points.size(); ++q)
    {
      const double thickness =
        source.evaluate_thickness(source_points[q].point, 0., {});
      const auto expected =
        section.get_transformed_quadrature(source_points[q].point,
                                           source_points[q].tangent,
                                           thickness);
      for (unsigned int section_q = 0;
           section_q < section.n_quadrature_points();
           ++section_q)
        {
          const unsigned int index =
            q * section.n_quadrature_points() + section_q;
          for (unsigned int d = 0; d < spacedim; ++d)
            EXPECT_NEAR(lifted_points[index].point[d],
                        expected.point(section_q)[d],
                        1.e-12)
              << "Lifted point " << index << ", coordinate " << d;
          EXPECT_NEAR(lifted_points[index].weight,
                      expected.weight(section_q) * source_points[q].weight,
                      1.e-12)
            << "Lifted weight " << index;
        }
    }
}


/**
 * Phase-4 parity gate: with modes 0 and 1 and identical settings, the legacy
 * modal TensorProductSpace/TensorProductRepresentation path and the modern
 * modal lift must agree on the modal DoF count, the
 * (representative DoF, mode) -> algebraic index mapping, the lifted
 * tensor-product basis values, and the physical quadrature.
 */
TEST(TensorProductLiftParity, MultiModeSingleLineCell) // NOLINT
{
  ParameterAcceptor::clear();

  TensorProductSpaceParameters<reduced_dim, surface_dim, spacedim, n_components>
    old_parameters;
  TensorProductLift<reduced_dim, surface_dim, spacedim, n_components> new_lift(
    "/Multi-mode lift/");
  configure_two_modes(old_parameters, new_lift);

  // ---- OLD PATH: reference modal TensorProductSpace. ----
  TensorProductSpace<reduced_dim, surface_dim, spacedim, n_components>
    old_space(old_parameters);
  old_space.initialize();
  const auto &old_section = old_space.get_reference_cross_section();
  ASSERT_EQ(old_section.n_selected_basis(), 2u);

  AffineConstraints<double> empty_constraints;
  empty_constraints.close();
  TensorProductRepresentation<reduced_dim, surface_dim, spacedim, n_components>
    old_representation(old_space,
                       old_space.get_dof_handler(),
                       old_space.get_dof_handler().locally_owned_dofs(),
                       DoFTools::extract_locally_relevant_dofs(
                         old_space.get_dof_handler()),
                       empty_constraints);

  // ---- NEW PATH: modal source representation lifted in modal mode. ----
  ModalLineSource fixture(2);
  const auto      modal_source = make_modal_source(fixture);
  const auto      modal_lift = ImmersX::make_modal_lift(modal_source, new_lift);
  const auto     &support    = modal_lift.support();
  const auto     &new_section   = support.reference_cross_section();
  const auto      source_points = modal_source.locally_owned_quadrature_points(
    support.representative_quadrature());

  // Modal DoF count: 2 reduced DoFs x 2 modes.
  EXPECT_EQ(old_space.get_dof_handler().n_dofs(), fixture.dof_handler.n_dofs());
  ASSERT_EQ(fixture.dof_handler.n_dofs(), 4u);

  // Representative quadrature coordinates agree.
  ASSERT_EQ(old_space.get_locally_owned_reduced_qpoints().size(),
            source_points.size());
  for (unsigned int q = 0; q < source_points.size(); ++q)
    for (unsigned int d = 0; d < spacedim; ++d)
      EXPECT_NEAR(old_space.get_locally_owned_reduced_qpoints()[q][d],
                  source_points[q].point[d],
                  1.e-12);

  // Section mode values agree for both modes.
  ASSERT_EQ(old_section.n_quadrature_points(),
            new_section.n_quadrature_points());
  for (unsigned int mode = 0; mode < 2; ++mode)
    for (unsigned int section_q = 0;
         section_q < old_section.n_quadrature_points();
         ++section_q)
      EXPECT_NEAR(old_section.shape_value(mode, section_q, 0),
                  new_section.shape_value(mode, section_q, 0),
                  1.e-12);

  // Lifted physical points, weights, and algebraic indices agree.
  const auto old_points = old_representation.locally_owned_quadrature_points(
    Quadrature<surface_dim>());
  const auto new_points =
    modal_lift.locally_owned_quadrature_points(Quadrature<surface_dim>());
  ASSERT_EQ(old_points.size(), new_points.size());
  const unsigned int n_scalar_dofs =
    source_points.front().dof_indices.size() / 2;
  for (unsigned int q = 0; q < old_points.size(); ++q)
    {
      for (unsigned int d = 0; d < spacedim; ++d)
        EXPECT_NEAR(old_points[q].point[d], new_points[q].point[d], 1.e-12)
          << "Point " << q;
      EXPECT_NEAR(old_points[q].weight, new_points[q].weight, 1.e-12)
        << "Weight " << q;

      ASSERT_EQ(old_points[q].dof_indices.size(),
                new_points[q].dof_indices.size());
      for (unsigned int slot = 0; slot < old_points[q].dof_indices.size();
           ++slot)
        EXPECT_EQ(old_points[q].dof_indices[slot],
                  new_points[q].dof_indices[slot])
          << "Algebraic index " << slot << " at point " << q;

      // Mode 0 and mode 1 use different algebraic indices.
      for (unsigned int mode = 1; mode < 2; ++mode)
        EXPECT_NE(new_points[q].dof_indices[mode * n_scalar_dofs],
                  new_points[q].dof_indices[0]);

      // Lifted tensor-product basis values match the reference product
      // N_i(s) phi_m(xi) computed from the shared source data.
      const unsigned int representative_q =
        q / old_section.n_quadrature_points();
      const unsigned int section_q = q % old_section.n_quadrature_points();
      for (unsigned int mode = 0; mode < 2; ++mode)
        for (unsigned int i = 0; i < n_scalar_dofs; ++i)
          EXPECT_NEAR(new_points[q].basis_values[mode * n_scalar_dofs + i],
                      source_points[representative_q]
                          .basis_values[mode * n_scalar_dofs + i] *
                        new_section.shape_value(mode, section_q, 0),
                      1.e-12)
            << "Basis at point " << q << ", mode " << mode << ", dof " << i;
    }
}

#endif // DEAL_II_WITH_VTK
