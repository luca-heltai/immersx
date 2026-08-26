// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>

#include <gtest/gtest.h>
#include <immersx/core/lifting.h>
#include <immersx/core/representation.h>
#include <immersx/core/state.h>
#include <immersx/coupling/reduced_coupling.h>
#include <immersx/coupling/tensor_product_lift.h>
#include <immersx/coupling/tensor_product_space.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>

#include <cstdint>

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
      const auto representative_qpoint = q / old_section.n_quadrature_points();
      EXPECT_EQ(lifted_points[q].source_entity_id,
                source_points[representative_qpoint].source_entity_id);
      EXPECT_EQ(lifted_points[q].stable_id,
                static_cast<std::uint64_t>(lifted_points[q].source_entity_id) *
                    source_points.size() * old_section.n_quadrature_points() +
                  representative_qpoint * old_section.n_quadrature_points() +
                  q % old_section.n_quadrature_points());
      EXPECT_EQ(lifted_points[q].source_dof_indices,
                source_points[representative_qpoint].dof_indices);
      EXPECT_EQ(lifted_points[q].source_basis_values,
                source_points[representative_qpoint].basis_values);
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
            const unsigned int slot = i * n_modes + mode;
            // Mode 0 and mode 1 must be independent algebraic unknowns.
            if (mode > 0)
              EXPECT_NE(lifted_points[q].dof_indices[slot],
                        lifted_points[q].dof_indices[i * n_modes])
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
        EXPECT_NE(new_points[q].dof_indices[mode],
                  new_points[q].dof_indices[0]);

      // Lifted tensor-product basis values match the reference product
      // N_i(s) phi_m(xi) computed from the shared source data.
      const unsigned int representative_q =
        q / old_section.n_quadrature_points();
      const unsigned int section_q = q % old_section.n_quadrature_points();
      for (unsigned int mode = 0; mode < 2; ++mode)
        for (unsigned int i = 0; i < n_scalar_dofs; ++i)
          EXPECT_NEAR(
            new_points[q].basis_values[i * 2 + mode],
            source_points[representative_q].basis_values[i * 2 + mode] *
              new_section.shape_value(mode, section_q, 0),
            1.e-12)
            << "Basis at point " << q << ", mode " << mode << ", dof " << i;
    }
}


namespace
{
  /**
   * Configure the shared settings for the ReducedCoupling parity test. The
   * representative domain is one_cylinder.vtk and the modal source uses the
   * same geometry programmatically.
   */
  void
  configure_coupling_parity(
    ReducedCouplingParameters<reduced_dim, surface_dim, spacedim, n_components>
      &old_parameters,
    TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>
                                    &new_lift,
    const std::vector<unsigned int> &modes)
  {
    auto &space_parameters = old_parameters.tensor_product_space_parameters;
    old_parameters.coupling_rhs_expressions =
      std::vector<std::string>(std::max<std::size_t>(modes.size(), 1), "0");
    space_parameters.reduced_grid_name =
      ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
    space_parameters.fe_degree                = 1;
    space_parameters.quadrature_type          = "gauss";
    space_parameters.n_q_points               = 2;
    space_parameters.n_quadrature_repetitions = 1;
    space_parameters.thickness                = "0.2";

    space_parameters.section.inclusion_type   = "hyper_ball";
    space_parameters.section.refinement_level = 1;
    space_parameters.section.inclusion_degree =
      modes.empty() ? 0 : modes.back();
    space_parameters.section.selected_coefficients    = modes;
    space_parameters.section.quadrature_type          = "gauss";
    space_parameters.section.n_q_points               = 4;
    space_parameters.section.n_quadrature_repetitions = 1;

    new_lift.thickness                      = "0.2";
    new_lift.representative_quadrature_type = "gauss";
    new_lift.representative_n_q_points      = 2;
    new_lift.representative_n_repetitions   = 1;

    new_lift.section.inclusion_type        = "hyper_ball";
    new_lift.section.refinement_level      = 1;
    new_lift.section.inclusion_degree      = modes.empty() ? 0 : modes.back();
    new_lift.section.selected_coefficients = modes;
    new_lift.section.quadrature_type       = "gauss";
    new_lift.section.n_q_points            = 4;
    new_lift.section.n_quadrature_repetitions = 1;
  }

  /** Assemble the coupling matrix of the modern modal path. */
  template <typename ModalLift>
  void
  assemble_modal_coupling_matrix(const ModalLift            &modal_lift,
                                 const DoFHandler<3>        &background_dh,
                                 const unsigned int          n_immersed_dofs,
                                 dealii::FullMatrix<double> &matrix)
  {
    matrix.reinit(background_dh.n_dofs(), n_immersed_dofs);
    const auto lifted_points =
      modal_lift.locally_owned_quadrature_points(Quadrature<surface_dim>());
    const dealii::MappingQ1<3>                   mapping;
    std::vector<dealii::types::global_dof_index> background_dof_indices(
      background_dh.get_fe().n_dofs_per_cell());
    for (const auto &point : lifted_points)
      {
        const auto cell_and_unit_point =
          dealii::GridTools::find_active_cell_around_point(mapping,
                                                           background_dh,
                                                           point.point);
        const auto &cell = cell_and_unit_point.first;
        if (cell == background_dh.end())
          continue;
        const dealii::Quadrature<3> point_quadrature(
          std::vector<dealii::Point<3>>{cell_and_unit_point.second});
        dealii::FEValues<3> fe_values(mapping,
                                      background_dh.get_fe(),
                                      point_quadrature,
                                      dealii::update_values);
        fe_values.reinit(cell);
        cell->get_dof_indices(background_dof_indices);
        for (unsigned int i = 0; i < background_dof_indices.size(); ++i)
          for (unsigned int j = 0; j < point.dof_indices.size(); ++j)
            matrix(background_dof_indices[i], point.dof_indices[j]) +=
              fe_values.shape_value(i, 0) * point.basis_values[j] *
              point.weight;
      }
  }

  /**
   * Compare the action and transpose-action of the legacy ReducedCoupling
   * matrix and the modern modal-path coupling matrix.
   */
  void
  check_coupling_action_parity(const std::vector<unsigned int> &modes)
  {
    ParameterAcceptor::clear();

    // Shared background mesh.
    parallel::distributed::Triangulation<3> background_tria(MPI_COMM_WORLD);
    GridGenerator::hyper_cube(background_tria, -0.2, 1.2);
    background_tria.refine_global(2);

    // ---- OLD PATH: reference ReducedCoupling. ----
    ReducedCouplingParameters<reduced_dim, surface_dim, spacedim, n_components>
      old_parameters;
    TensorProductLift<reduced_dim, surface_dim, spacedim, n_components> lift(
      "/Coupling parity lift/");
    configure_coupling_parity(old_parameters, lift, modes);

    ReducedCoupling<reduced_dim, surface_dim, spacedim, n_components> coupling(
      background_tria, old_parameters);
    coupling.initialize();

    FE_Q<3>       background_fe(1);
    DoFHandler<3> background_dh(background_tria);
    background_dh.distribute_dofs(background_fe);
    const dealii::IndexSet background_owned =
      background_dh.locally_owned_dofs();
    const dealii::IndexSet background_relevant =
      DoFTools::extract_locally_relevant_dofs(background_dh);
    dealii::AffineConstraints<double> constraints(background_owned,
                                                  background_relevant);
    constraints.close();

    DynamicSparsityPattern dsp(background_dh.n_dofs(),
                               coupling.get_dof_handler().n_dofs());
    coupling.assemble_coupling_sparsity(dsp, background_dh, constraints);
    ImmersXLA::MPI::SparseMatrix coupling_old;
    coupling_old.reinit(background_owned,
                        coupling.get_dof_handler().locally_owned_dofs(),
                        dsp,
                        MPI_COMM_WORLD);
    coupling.assemble_coupling_matrix(coupling_old, background_dh, constraints);

    // ---- NEW PATH: modal representation lifted on the same geometry. ----
    ModalLineSource fixture(static_cast<unsigned int>(modes.size()));
    const auto      modal_source = make_modal_source(fixture);
    const auto      modal_lift   = ImmersX::make_modal_lift(modal_source, lift);
    EXPECT_EQ(coupling.get_dof_handler().n_dofs(),
              fixture.dof_handler.n_dofs());

    dealii::FullMatrix<double> coupling_new;
    assemble_modal_coupling_matrix(modal_lift,
                                   background_dh,
                                   fixture.dof_handler.n_dofs(),
                                   coupling_new);

    // ---- Forward action C * x. ----
    ImmersXLA::MPI::Vector x;
    x.reinit(coupling.get_dof_handler().locally_owned_dofs(), MPI_COMM_WORLD);
    for (const auto index : x.locally_owned_elements())
      x[index] = std::sin(1. + 0.7 * static_cast<double>(index)) +
                 0.2 * static_cast<double>(index);

    ImmersXLA::MPI::Vector forward_old;
    forward_old.reinit(background_owned, MPI_COMM_WORLD);
    coupling_old.vmult(forward_old, x);

    dealii::Vector<double> x_serial(fixture.dof_handler.n_dofs());
    for (const auto index : x.locally_owned_elements())
      x_serial[index] = x[index];
    dealii::Vector<double> forward_new(background_dh.n_dofs());
    coupling_new.vmult(forward_new, x_serial);

    for (const auto index : background_owned)
      EXPECT_NEAR(forward_old[index],
                  forward_new[index],
                  1.e-10 * std::max(1., std::abs(forward_old[index])))
        << "C x differs at background DoF " << index;

    // ---- Transpose action C^T * y. ----
    ImmersXLA::MPI::Vector y;
    y.reinit(background_owned, MPI_COMM_WORLD);
    for (const auto index : y.locally_owned_elements())
      y[index] = std::cos(0.3 * static_cast<double>(index)) +
                 0.1 * static_cast<double>(index);

    ImmersXLA::MPI::Vector transpose_old;
    transpose_old.reinit(coupling.get_dof_handler().locally_owned_dofs(),
                         MPI_COMM_WORLD);
    coupling_old.Tvmult(transpose_old, y);

    dealii::Vector<double> y_serial(background_dh.n_dofs());
    for (const auto index : y.locally_owned_elements())
      y_serial[index] = y[index];
    dealii::Vector<double> transpose_new(fixture.dof_handler.n_dofs());
    coupling_new.Tvmult(transpose_new, y_serial);

    for (const auto index : coupling.get_dof_handler().locally_owned_dofs())
      EXPECT_NEAR(transpose_old[index],
                  transpose_new[index],
                  1.e-10 * std::max(1., std::abs(transpose_old[index])))
        << "C^T y differs at immersed DoF " << index;
  }
} // namespace


/**
 * Phase-7 readiness gate: the legacy ReducedCoupling matrix and the modern
 * modal-path coupling matrix must have the same action and transpose-action
 * for identical settings. Both a single-mode and a multi-mode configuration
 * are exercised.
 */
TEST(TensorProductLiftParity, ReducedCouplingActionSingleMode) // NOLINT
{
  check_coupling_action_parity({0});
}


TEST(TensorProductLiftParity, ReducedCouplingActionMultiMode) // NOLINT
{
  check_coupling_action_parity({0, 1});
}

#endif // DEAL_II_WITH_VTK
