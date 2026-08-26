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

#include <algorithm>
#include <cmath>
#include <map>

using namespace dealii;
using namespace ImmersX;

namespace
{
  struct ExpressionData
  {
    parallel::distributed::Triangulation<2>         triangulation;
    FE_Q<2>                                         finite_element;
    DoFHandler<2>                                   dof_handler;
    IndexSet                                        locally_owned;
    IndexSet                                        locally_relevant;
    AffineConstraints<double>                       constraints;
    std::unique_ptr<FiniteElementRepresentation<2>> representation;
    StateLayout                                     layout;
    FieldId                                         field;

    ExpressionData()
      : triangulation(MPI_COMM_WORLD)
      , finite_element(2)
      , dof_handler(triangulation)
    {
      GridGenerator::hyper_cube(triangulation, 0., 1.);
      triangulation.refine_global(2);
      dof_handler.distribute_dofs(finite_element);
      locally_owned    = dof_handler.locally_owned_dofs();
      locally_relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
      constraints.reinit(locally_owned, locally_relevant);
      constraints.close();
      representation =
        std::make_unique<FiniteElementRepresentation<2>>(triangulation,
                                                         dof_handler,
                                                         locally_owned,
                                                         locally_relevant,
                                                         constraints);

      FieldDescriptor descriptor;
      descriptor.name             = "A";
      descriptor.locally_owned    = locally_owned;
      descriptor.locally_relevant = locally_relevant;
      field                       = layout.add_field(descriptor);
    }
  };

  void
  fill_vector(const IndexSet &owned, ImmersXLA::MPI::Vector &vector)
  {
    vector.reinit(owned, MPI_COMM_WORLD);
    for (const auto index : owned)
      vector[index] = 0.5 + 0.25 * index;
  }

  double
  local_dot(const IndexSet               &owned,
            const ImmersXLA::MPI::Vector &left,
            const ImmersXLA::MPI::Vector &right)
  {
    double result = 0.;
    for (const auto index : owned)
      result += left[index] * right[index];
    return Utilities::MPI::sum(result, MPI_COMM_WORLD);
  }

  void
  check_expression(const bool mpi)
  {
    if (mpi)
      ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
    else
      ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);
    ASSERT_TRUE(SymbolicExpressionKernel::available());

    ExpressionData  data;
    const QGauss<2> quadrature(3);
    const auto      plan =
      make_retained_sampling_plan(*data.representation, quadrature);
    const double beta       = 1.5;
    const auto   expression = make_expression_representation(
      data.field, plan, "A*A + beta*A", "A", {{"beta", beta}});

    ImmersXLA::MPI::Vector state;
    fill_vector(data.locally_owned, state);
    ImmersXLA::MPI::Vector direction;
    fill_vector(data.locally_owned, direction);

    StateView<ImmersXLA::MPI::Vector> state_view(data.layout, 0.);
    state_view.bind(data.field, state);
    EvaluationContext<ImmersXLA::MPI::Vector> context(0., state_view);

    const auto             value             = expression.evaluate(context);
    const auto             sampling_operator = plan.linearize(state);
    ImmersXLA::MPI::Vector samples;
    sampling_operator.reinit_range_vector(samples, false);
    sampling_operator.vmult(samples, state);
    for (std::size_t q = 0; q < plan.points().size(); ++q)
      {
        const double a = samples[plan.point_index(q)];
        EXPECT_DOUBLE_EQ(value[plan.point_index(q)], a * a + beta * a);
      }

    const auto             jacobian = expression.linearize(context);
    ImmersXLA::MPI::Vector jacobian_action;
    jacobian.reinit_range_vector(jacobian_action, false);
    jacobian.vmult(jacobian_action, direction);
    ImmersXLA::MPI::Vector sampled_direction;
    sampling_operator.reinit_range_vector(sampled_direction, false);
    sampling_operator.vmult(sampled_direction, direction);
    for (std::size_t q = 0; q < plan.points().size(); ++q)
      {
        const double a = samples[plan.point_index(q)];
        EXPECT_DOUBLE_EQ(jacobian_action[plan.point_index(q)],
                         (2. * a + beta) *
                           sampled_direction[plan.point_index(q)]);
      }

    const double           epsilon   = 1.e-7;
    ImmersXLA::MPI::Vector perturbed = state;
    perturbed.add(epsilon, direction);
    StateView<ImmersXLA::MPI::Vector> perturbed_view(data.layout, 0.);
    perturbed_view.bind(data.field, perturbed);
    EvaluationContext<ImmersXLA::MPI::Vector> perturbed_context(0.,
                                                                perturbed_view);
    const auto perturbed_value = expression.evaluate(perturbed_context);
    for (std::size_t q = 0; q < plan.points().size(); ++q)
      EXPECT_NEAR(
        (perturbed_value[plan.point_index(q)] - value[plan.point_index(q)]) /
          epsilon,
        jacobian_action[plan.point_index(q)],
        1.e-7 * std::max(1., std::abs(jacobian_action[plan.point_index(q)])));

    ImmersXLA::MPI::Vector weights;
    weights.reinit(plan.locally_owned_points(), MPI_COMM_WORLD);
    for (const auto index : plan.locally_owned_points())
      weights[index] = 0.75 + 0.1 * index;
    ImmersXLA::MPI::Vector transpose;
    jacobian.reinit_domain_vector(transpose, false);
    jacobian.Tvmult(transpose, weights);
    EXPECT_NEAR(local_dot(plan.locally_owned_points(),
                          jacobian_action,
                          weights),
                local_dot(data.locally_owned, direction, transpose),
                1.e-8);
  }
} // namespace


TEST(ExpressionRepresentation, ValueAndJacobian) // NOLINT
{
  check_expression(false);
}


TEST(ExpressionRepresentation, MPI_ValueAndJacobian) // NOLINT
{
  check_expression(true);
}
