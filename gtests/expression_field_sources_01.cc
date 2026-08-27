// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <gtest/gtest.h>
#include <immersx/core/expression_representation.h>
#include <immersx/io/imported_finite_element_fields.h>
#include <immersx/io/vtk_utils.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace dealii;
using namespace ImmersX;

namespace
{
  double
  owned_dot(const ImmersXLA::MPI::Vector &left,
            const ImmersXLA::MPI::Vector &right)
  {
    double local = 0.;
    for (const auto index : left.locally_owned_elements())
      local += left[index] * right[index];
    return Utilities::MPI::sum(local, left.get_mpi_communicator());
  }

#ifdef DEAL_II_WITH_VTK
  std::unique_ptr<parallel::fullydistributed::Triangulation<3>>
  make_imported_mesh(const std::string &filename)
  {
    Triangulation<3>         serial_triangulation;
    DoFHandler<3>            serial_dof_handler(serial_triangulation);
    Vector<double>           serial_data;
    std::vector<std::string> names;
    VTKUtils::read_vtk(filename, serial_dof_handler, serial_data, names);
    GridTools::partition_triangulation(
      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), serial_triangulation);
    const auto description = TriangulationDescription::Utilities::
      create_description_from_triangulation(serial_triangulation,
                                            MPI_COMM_WORLD);
    auto mesh = std::make_unique<parallel::fullydistributed::Triangulation<3>>(
      MPI_COMM_WORLD);
    mesh->create_triangulation(description);
    return mesh;
  }
#endif

  void
  check_active_and_frozen(const unsigned int expected_processes)
  {
    ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),
              expected_processes);
    ASSERT_TRUE(SymbolicExpressionKernel::available());

    parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
    GridGenerator::hyper_cube(triangulation, 0., 1.);
    triangulation.refine_global(2);

    FE_Q<2>       finite_element(2);
    DoFHandler<2> dof_handler(triangulation);
    dof_handler.distribute_dofs(finite_element);
    const auto locally_owned = dof_handler.locally_owned_dofs();
    const auto locally_relevant =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    AffineConstraints<double> constraints(locally_owned, locally_relevant);
    constraints.close();

    const FiniteElementRepresentation<2> representation(
      triangulation, dof_handler, locally_owned, locally_relevant, constraints);
    const QGauss<2> quadrature(3);

    FieldDescriptor descriptor;
    descriptor.name             = "active";
    descriptor.locally_owned    = locally_owned;
    descriptor.locally_relevant = locally_relevant;
    StateLayout layout;
    const auto  active_field = layout.add_field(descriptor);

    ImmersXLA::MPI::Vector active_values;
    active_values.reinit(locally_owned, MPI_COMM_WORLD);
    for (const auto index : locally_owned)
      active_values[index] = 0.5 + 0.1 * static_cast<double>(index);

    auto frozen_coefficients = std::make_shared<ImmersXLA::MPI::Vector>();
    frozen_coefficients->reinit(locally_owned, MPI_COMM_WORLD);
    for (const auto index : locally_owned)
      (*frozen_coefficients)[index] = 1. + 0.05 * static_cast<double>(index);

    const auto frozen = frozen_field(representation, frozen_coefficients);
    const auto active = state_field(representation, active_field);

    const auto expression = make_expression_representation(
      representation,
      quadrature,
      {value(active, "A"), value(frozen, "P"), gradient(frozen, "grad_P_0", 0)},
      "A + P + grad_P_0");

    ASSERT_EQ(expression.dependencies().size(), 1u);
    EXPECT_EQ(expression.dependencies().front(), active_field);
    ASSERT_EQ(expression.n_sampling_sources(), 2u);
    EXPECT_EQ(expression.sampling_plan_for_source(0).update_flags() &
                update_gradients,
              UpdateFlags());
    EXPECT_NE(expression.sampling_plan_for_source(1).update_flags() &
                update_gradients,
              UpdateFlags());

    StateView<ImmersXLA::MPI::Vector> state_view(layout, 0.);
    state_view.bind(active_field, active_values);
    const EvaluationContext<ImmersXLA::MPI::Vector> context(0., state_view);
    const auto  expression_value = expression.evaluate(context);
    const auto &plan             = expression.sampling_plan();

    const auto             active_sampling = plan.linearize(active_values);
    ImmersXLA::MPI::Vector active_samples;
    active_sampling.reinit_range_vector(active_samples, false);
    active_sampling.vmult(active_samples, active_values);
    const auto &frozen_plan     = expression.sampling_plan(1);
    const auto  frozen_sampling = frozen_plan.linearize(*frozen_coefficients);
    ImmersXLA::MPI::Vector frozen_samples;
    frozen_sampling.reinit_range_vector(frozen_samples, false);
    frozen_sampling.vmult(frozen_samples, *frozen_coefficients);
    const auto &frozen_gradient_plan = expression.sampling_plan(2);
    const auto  frozen_gradient_sampling =
      frozen_gradient_plan.gradient_linearize(*frozen_coefficients, 0);
    ImmersXLA::MPI::Vector frozen_gradient_samples;
    frozen_gradient_sampling.reinit_range_vector(frozen_gradient_samples,
                                                 false);
    frozen_gradient_sampling.vmult(frozen_gradient_samples,
                                   *frozen_coefficients);

    for (const auto index : plan.locally_owned_points())
      EXPECT_NEAR(expression_value[index],
                  active_samples[index] + frozen_samples[index] +
                    frozen_gradient_samples[index],
                  1.e-14);

    const auto             jacobian = expression.linearize(context);
    ImmersXLA::MPI::Vector action;
    jacobian.reinit_range_vector(action, false);
    jacobian.vmult(action, active_values);
    for (const auto index : plan.locally_owned_points())
      EXPECT_DOUBLE_EQ(action[index], active_samples[index]);

    ImmersXLA::MPI::Vector direction        = active_values;
    const double           epsilon          = 1.e-7;
    const auto             original_value   = expression.evaluate(context);
    auto                   perturbed_values = active_values;
    for (const auto index : perturbed_values.locally_owned_elements())
      perturbed_values[index] += epsilon * direction[index];
    StateView<ImmersXLA::MPI::Vector> perturbed_state_view(layout, 0.);
    perturbed_state_view.bind(active_field, perturbed_values);
    const EvaluationContext<ImmersXLA::MPI::Vector> perturbed_context(
      0., perturbed_state_view);
    const auto perturbed_value = expression.evaluate(perturbed_context);
    for (const auto index : plan.locally_owned_points())
      EXPECT_NEAR((perturbed_value[index] - original_value[index]) / epsilon,
                  action[index],
                  2.e-7 * std::max(1., std::abs(action[index])));

    ImmersXLA::MPI::Vector dual;
    jacobian.reinit_range_vector(dual, false);
    for (const auto index : dual.locally_owned_elements())
      dual[index] = 0.4 + 0.03 * static_cast<double>(index);
    ImmersXLA::MPI::Vector transpose_action;
    jacobian.reinit_domain_vector(transpose_action, false);
    jacobian.Tvmult(transpose_action, dual);
    EXPECT_NEAR(owned_dot(action, dual),
                owned_dot(direction, transpose_action),
                2.e-10 * std::max(1., std::abs(owned_dot(action, dual))));

    const auto frozen_expression =
      make_expression_representation(representation,
                                     quadrature,
                                     {value(frozen, "P"),
                                      gradient(frozen, "grad_P_0", 0)},
                                     "P + grad_P_0");
    EXPECT_TRUE(frozen_expression.dependencies().empty());
    const auto frozen_value = frozen_expression.evaluate(context);
    for (const auto index : plan.locally_owned_points())
      EXPECT_NEAR(frozen_value[index],
                  frozen_samples[index] + frozen_gradient_samples[index],
                  1.e-14);

    const auto active_gradients =
      make_expression_representation(representation,
                                     quadrature,
                                     {gradient(active, "g0", 0),
                                      gradient(active, "g1", 1)},
                                     "g0 + g1");
    EXPECT_EQ(active_gradients.n_sampling_sources(), 1u);

    const auto frozen_value_and_gradient =
      make_expression_representation(representation,
                                     quadrature,
                                     {value(frozen, "P"),
                                      gradient(frozen, "g0", 0)},
                                     "P + g0");
    EXPECT_EQ(frozen_value_and_gradient.n_sampling_sources(), 1u);
  }

  void
  check_three_component_gradients()
  {
    parallel::distributed::Triangulation<3> triangulation(MPI_COMM_WORLD);
    GridGenerator::hyper_cube(triangulation, 0., 1.);
    triangulation.refine_global(1);
    FE_Q<3>       finite_element(2);
    DoFHandler<3> dof_handler(triangulation);
    dof_handler.distribute_dofs(finite_element);
    const auto locally_owned = dof_handler.locally_owned_dofs();
    const auto locally_relevant =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    AffineConstraints<double> constraints(locally_owned, locally_relevant);
    constraints.close();
    const FiniteElementRepresentation<3> representation(
      triangulation, dof_handler, locally_owned, locally_relevant, constraints);
    StateLayout     layout;
    FieldDescriptor descriptor;
    descriptor.name             = "active";
    descriptor.locally_owned    = locally_owned;
    descriptor.locally_relevant = locally_relevant;
    const auto active_field     = layout.add_field(descriptor);
    const auto active           = state_field(representation, active_field);
    const auto expression =
      make_expression_representation(representation,
                                     QGauss<3>(2),
                                     {gradient(active, "g0", 0),
                                      gradient(active, "g1", 1),
                                      gradient(active, "g2", 2)},
                                     "g0 + g1 + g2");
    EXPECT_EQ(expression.n_sampling_sources(), 1u);
    EXPECT_NE(expression.sampling_plan_for_source(0).update_flags() &
                update_gradients,
              UpdateFlags());
  }
} // namespace


TEST(ExpressionFieldSources, ActiveAndFrozenBindings)
{
  check_active_and_frozen(1);
}


TEST(ExpressionFieldSources, MPI_ActiveAndFrozenBindings)
{
  check_active_and_frozen(2);
}


TEST(ExpressionFieldSources, ThreeGradientBindingsShareOnePlan)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);
  check_three_component_gradients();
}


TEST(ExpressionFieldSources, MPI_ThreeGradientBindingsShareOnePlan)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  check_three_component_gradients();
}


TEST(ExpressionFieldSources, FrozenImportedFieldSurvivesParentHandle)
{
#ifdef DEAL_II_WITH_VTK
  const auto filename =
    std::string(TEST_DATA_DIR) + "/tests/imported_lambda_3d.vtk";
  auto mesh       = make_imported_mesh(filename);
  auto expression = [&] {
    auto imported =
      std::make_shared<ImportedFiniteElementFields<3>>(filename, *mesh);
    const auto frozen = frozen_field(imported->field("lambda"));
    return make_expression_representation(frozen.current_representation(),
                                          QGauss<3>(2),
                                          {value(frozen, "lambda")},
                                          "lambda");
  }();

  StateLayout                                     layout;
  StateView<ImmersXLA::MPI::Vector>               state_view(layout, 0.);
  const EvaluationContext<ImmersXLA::MPI::Vector> context(0., state_view);
  const auto values = expression.evaluate(context);
  EXPECT_GT(values.locally_owned_elements().n_elements(), 0u);
#else
  GTEST_SKIP() << "VTK support is required for imported fields.";
#endif
}

TEST(ExpressionFieldSources, FrozenImportedFieldTracksRefinement)
{
#ifdef DEAL_II_WITH_VTK
  const auto filename =
    std::string(TEST_DATA_DIR) + "/tests/imported_lambda_3d.vtk";
  parallel::distributed::Triangulation<3> mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(mesh, 0., 1.);
  auto imported =
    std::make_shared<ImportedFiniteElementFields<3>>(filename, mesh);
  const auto frozen = frozen_field(imported->field("lambda"));
  const auto before_dofs =
    frozen.current_representation().dof_handler().n_dofs();

  mesh.refine_global(1);

  const auto current = frozen.current_representation();
  EXPECT_GT(current.dof_handler().n_dofs(), before_dofs);
  const auto expression = make_expression_representation(
    current, QGauss<3>(2), {value(frozen, "lambda")}, "lambda");
  StateLayout                                     layout;
  StateView<ImmersXLA::MPI::Vector>               state_view(layout, 0.);
  const EvaluationContext<ImmersXLA::MPI::Vector> context(0., state_view);
  const auto             values   = expression.evaluate(context);
  const auto            &plan     = expression.sampling_plan();
  const auto             sampling = plan.linearize(*frozen.coefficients);
  ImmersXLA::MPI::Vector sampled;
  sampling.reinit_range_vector(sampled, false);
  sampling.vmult(sampled, *frozen.coefficients);
  for (const auto &point : plan.points())
    EXPECT_NEAR(sampled[plan.point_index(&point - plan.points().data())],
                point.point[0] + point.point[1] + point.point[2],
                1.e-12);
  auto difference = values;
  difference -= sampled;
  EXPECT_LT(difference.l2_norm(), 1.e-12);
#else
  GTEST_SKIP() << "VTK support is required for imported fields.";
#endif
}

TEST(ExpressionFieldSources, MPI_FrozenImportedFieldTracksRefinement)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
#ifdef DEAL_II_WITH_VTK
  const auto filename =
    std::string(TEST_DATA_DIR) + "/tests/imported_lambda_3d.vtk";
  parallel::distributed::Triangulation<3> mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(mesh, 0., 1.);
  auto imported =
    std::make_shared<ImportedFiniteElementFields<3>>(filename, mesh);
  const auto frozen = frozen_field(imported->field("lambda"));
  const auto before_dofs =
    frozen.current_representation().dof_handler().n_dofs();
  mesh.refine_global(1);
  EXPECT_GT(frozen.current_representation().dof_handler().n_dofs(),
            before_dofs);
#else
  GTEST_SKIP() << "VTK support is required for imported fields.";
#endif
}
