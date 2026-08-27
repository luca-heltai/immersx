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

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/core/expression_representation.h>

using namespace dealii;
using namespace ImmersX;

namespace
{
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
