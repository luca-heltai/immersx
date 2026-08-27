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

#include <gtest/gtest.h>
#include <immersx/core/expression_representation.h>
#include <immersx/core/lifting.h>

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
    FieldId                                         second_field;

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
      descriptor.name             = "U";
      second_field                = layout.add_field(descriptor);
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
    const double    beta       = 1.5;
    const auto      expression = make_expression_representation(
      *data.representation,
      quadrature,
      {value(state_field(*data.representation, data.field), "A")},
      "A*A + beta*A",
      {{"beta", beta}});
    const auto &plan = expression.sampling_plan();
    ASSERT_FALSE(plan.points().empty());
    EXPECT_EQ(plan.update_flags() & update_gradients, update_default);
    EXPECT_TRUE(plan.points().front().basis_gradients.empty());

    ImmersXLA::MPI::Vector state;
    fill_vector(data.locally_owned, state);
    ImmersXLA::MPI::Vector direction;
    fill_vector(data.locally_owned, direction);
    ImmersXLA::MPI::Vector second_state;
    second_state.reinit(data.locally_owned, MPI_COMM_WORLD);
    for (const auto index : data.locally_owned)
      second_state[index] = 0.25 + 0.125 * index;

    StateView<ImmersXLA::MPI::Vector> state_view(data.layout, 0.);
    state_view.bind(data.field, state);
    state_view.bind(data.second_field, second_state);
    EvaluationContext<ImmersXLA::MPI::Vector> context(0., state_view);

    const auto             expression_value  = expression.evaluate(context);
    const auto             sampling_operator = plan.linearize(state);
    ImmersXLA::MPI::Vector samples;
    sampling_operator.reinit_range_vector(samples, false);
    sampling_operator.vmult(samples, state);
    for (std::size_t q = 0; q < plan.points().size(); ++q)
      {
        const double a = samples[plan.point_index(q)];
        EXPECT_DOUBLE_EQ(expression_value[plan.point_index(q)],
                         a * a + beta * a);
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
      EXPECT_NEAR((perturbed_value[plan.point_index(q)] -
                   expression_value[plan.point_index(q)]) /
                    epsilon,
                  jacobian_action[plan.point_index(q)],
                  1.e-7 *
                    std::max(1.,
                             std::abs(jacobian_action[plan.point_index(q)])));

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

    const auto multi = make_expression_representation(
      *data.representation,
      quadrature,
      {value(state_field(*data.representation, data.field), "A"),
       value(state_field(*data.representation, data.second_field), "U")},
      "A*A + 2*U");
    ASSERT_EQ(multi.dependencies().size(), 2u);
    EXPECT_EQ(multi.dependencies()[0], data.field);
    EXPECT_EQ(multi.dependencies()[1], data.second_field);

    const auto             multi_value     = multi.evaluate(context);
    const auto             second_sampling = plan.linearize(second_state);
    ImmersXLA::MPI::Vector second_samples;
    second_sampling.reinit_range_vector(second_samples, false);
    second_sampling.vmult(second_samples, second_state);
    for (std::size_t q = 0; q < plan.points().size(); ++q)
      {
        const auto   index = plan.point_index(q);
        const double a     = samples[index];
        EXPECT_DOUBLE_EQ(multi_value[index],
                         a * a + 2. * second_samples[index]);
      }

    const auto multi_jacobian = multi.linearize(context, data.second_field);
    ImmersXLA::MPI::Vector multi_action;
    multi_jacobian.reinit_range_vector(multi_action, false);
    multi_jacobian.vmult(multi_action, direction);
    ImmersXLA::MPI::Vector second_direction;
    second_sampling.reinit_range_vector(second_direction, false);
    second_sampling.vmult(second_direction, direction);
    for (std::size_t q = 0; q < plan.points().size(); ++q)
      EXPECT_DOUBLE_EQ(multi_action[plan.point_index(q)],
                       2. * second_direction[plan.point_index(q)]);

    const auto gradient_expression = make_expression_representation(
      *data.representation,
      quadrature,
      {value(state_field(*data.representation, data.field), "A"),
       gradient(state_field(*data.representation, data.field), "grad_A_0", 0),
       gradient(state_field(*data.representation, data.field), "grad_A_1", 1)},
      "A*A + grad_A_0 + 2*grad_A_1");
    const auto &gradient_plan = gradient_expression.sampling_plan();
    EXPECT_EQ(gradient_plan.update_flags() & update_gradients,
              update_gradients);
    ASSERT_FALSE(gradient_plan.points().front().basis_gradients.empty());
    const auto gradient_value      = gradient_expression.evaluate(context);
    const auto gradient_sampling   = gradient_plan.linearize(state);
    const auto gradient_x_sampling = gradient_plan.gradient_linearize(state, 0);
    const auto gradient_y_sampling = gradient_plan.gradient_linearize(state, 1);
    ImmersXLA::MPI::Vector gradient_samples;
    ImmersXLA::MPI::Vector gradient_x_samples;
    ImmersXLA::MPI::Vector gradient_y_samples;
    gradient_sampling.reinit_range_vector(gradient_samples, false);
    gradient_x_sampling.reinit_range_vector(gradient_x_samples, false);
    gradient_y_sampling.reinit_range_vector(gradient_y_samples, false);
    gradient_sampling.vmult(gradient_samples, state);
    gradient_x_sampling.vmult(gradient_x_samples, state);
    gradient_y_sampling.vmult(gradient_y_samples, state);
    for (std::size_t q = 0; q < gradient_plan.points().size(); ++q)
      {
        const auto index = gradient_plan.point_index(q);
        EXPECT_DOUBLE_EQ(gradient_value[index],
                         gradient_samples[index] * gradient_samples[index] +
                           gradient_x_samples[index] +
                           2. * gradient_y_samples[index]);
      }

    const auto gradient_jacobian = gradient_expression.linearize(context);
    ImmersXLA::MPI::Vector gradient_action;
    gradient_jacobian.reinit_range_vector(gradient_action, false);
    gradient_jacobian.vmult(gradient_action, direction);
    const auto gradient_direction = gradient_plan.linearize(direction);
    const auto gradient_x_direction =
      gradient_plan.gradient_linearize(direction, 0);
    const auto gradient_y_direction =
      gradient_plan.gradient_linearize(direction, 1);
    ImmersXLA::MPI::Vector gradient_direction_values;
    ImmersXLA::MPI::Vector gradient_x_direction_values;
    ImmersXLA::MPI::Vector gradient_y_direction_values;
    gradient_direction.reinit_range_vector(gradient_direction_values, false);
    gradient_x_direction.reinit_range_vector(gradient_x_direction_values,
                                             false);
    gradient_y_direction.reinit_range_vector(gradient_y_direction_values,
                                             false);
    gradient_direction.vmult(gradient_direction_values, direction);
    gradient_x_direction.vmult(gradient_x_direction_values, direction);
    gradient_y_direction.vmult(gradient_y_direction_values, direction);
    for (std::size_t q = 0; q < gradient_plan.points().size(); ++q)
      {
        const auto index = gradient_plan.point_index(q);
        EXPECT_DOUBLE_EQ(gradient_action[index],
                         2. * gradient_samples[index] *
                             gradient_direction_values[index] +
                           gradient_x_direction_values[index] +
                           2. * gradient_y_direction_values[index]);
      }

    ImmersXLA::MPI::Vector gradient_perturbed = state;
    gradient_perturbed.add(epsilon, direction);
    StateView<ImmersXLA::MPI::Vector> gradient_perturbed_view(data.layout, 0.);
    gradient_perturbed_view.bind(data.field, gradient_perturbed);
    gradient_perturbed_view.bind(data.second_field, second_state);
    EvaluationContext<ImmersXLA::MPI::Vector> gradient_perturbed_context(
      0., gradient_perturbed_view);
    const auto gradient_perturbed_value =
      gradient_expression.evaluate(gradient_perturbed_context);
    for (std::size_t q = 0; q < gradient_plan.points().size(); ++q)
      EXPECT_NEAR((gradient_perturbed_value[gradient_plan.point_index(q)] -
                   gradient_value[gradient_plan.point_index(q)]) /
                    epsilon,
                  gradient_action[gradient_plan.point_index(q)],
                  3.e-7 *
                    std::max(1.,
                             std::abs(
                               gradient_action[gradient_plan.point_index(q)])));

    ImmersXLA::MPI::Vector gradient_weights;
    gradient_weights.reinit(gradient_plan.locally_owned_points(),
                            MPI_COMM_WORLD);
    for (const auto index : gradient_plan.locally_owned_points())
      gradient_weights[index] = 0.5 + 0.05 * index;
    ImmersXLA::MPI::Vector gradient_transpose;
    gradient_jacobian.reinit_domain_vector(gradient_transpose, false);
    gradient_jacobian.Tvmult(gradient_transpose, gradient_weights);
    EXPECT_NEAR(local_dot(gradient_plan.locally_owned_points(),
                          gradient_action,
                          gradient_weights),
                local_dot(data.locally_owned, direction, gradient_transpose),
                1.e-8);
  }

  void
  check_deferred_sampling(const bool mpi)
  {
    if (mpi)
      ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
    else
      ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);

    ExpressionData  data;
    const QGauss<2> quadrature(3);
    const auto      deferred =
      make_fe_expression(*data.representation,
                         {value(state_field(*data.representation, data.field),
                                "A")},
                         "factor*A",
                         {{"factor", 2.5}});

    // The description has no sampling plan until this explicit conversion.
    const auto sampled = sample(deferred, quadrature);
    EXPECT_EQ(deferred.domain(),
              (RepresentationDomain(2, 2, "finite-element-expression")));
    EXPECT_EQ(sampled.quantity_space().domain(),
              (RepresentationDomain(2, 2, "retained-fe-sampling")));

    ImmersXLA::MPI::Vector state;
    fill_vector(data.locally_owned, state);
    ImmersXLA::MPI::Vector direction;
    fill_vector(data.locally_owned, direction);
    StateView<ImmersXLA::MPI::Vector> state_view(data.layout, 0.);
    state_view.bind(data.field, state);
    EvaluationContext<ImmersXLA::MPI::Vector> context(0., state_view);

    const auto             value_result = sampled.evaluate(context);
    const auto             jacobian     = sampled.linearize(context);
    ImmersXLA::MPI::Vector action;
    jacobian.reinit_range_vector(action, false);
    jacobian.vmult(action, direction);

    const auto            &plan               = sampled.sampling_plan();
    const auto             sampling           = plan.linearize(state);
    const auto             direction_sampling = plan.linearize(direction);
    ImmersXLA::MPI::Vector samples;
    ImmersXLA::MPI::Vector direction_values;
    sampling.reinit_range_vector(samples, false);
    direction_sampling.reinit_range_vector(direction_values, false);
    sampling.vmult(samples, state);
    direction_sampling.vmult(direction_values, direction);
    for (std::size_t q = 0; q < plan.points().size(); ++q)
      {
        const auto index = plan.point_index(q);
        EXPECT_DOUBLE_EQ(value_result[index], 2.5 * samples[index]);
        EXPECT_DOUBLE_EQ(action[index], 2.5 * direction_values[index]);
      }

    ImmersXLA::MPI::Vector weights;
    weights.reinit(plan.locally_owned_points(), MPI_COMM_WORLD);
    for (const auto index : plan.locally_owned_points())
      weights[index] = 1. + 0.1 * index;
    ImmersXLA::MPI::Vector transpose;
    jacobian.reinit_domain_vector(transpose, false);
    jacobian.Tvmult(transpose, weights);
    EXPECT_NEAR(local_dot(plan.locally_owned_points(), action, weights),
                local_dot(data.locally_owned, direction, transpose),
                1.e-10);
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


TEST(ExpressionRepresentation, DeferredSampling) // NOLINT
{
  check_deferred_sampling(false);
}


TEST(ExpressionRepresentation, MPI_DeferredSampling) // NOLINT
{
  check_deferred_sampling(true);
}


namespace
{
  struct LiftExpressionData
  {
    parallel::fullydistributed::Triangulation<1, 2>    triangulation;
    FE_Q<1, 2>                                         finite_element;
    DoFHandler<1, 2>                                   dof_handler;
    IndexSet                                           locally_owned;
    IndexSet                                           locally_relevant;
    AffineConstraints<double>                          constraints;
    std::unique_ptr<FiniteElementRepresentation<1, 2>> representation;
    StateLayout                                        layout;
    FieldId                                            field;

    LiftExpressionData()
      : triangulation(MPI_COMM_WORLD)
      , finite_element(2)
      , dof_handler(triangulation)
    {
      Triangulation<1, 2> serial;
      GridGenerator::hyper_cube(serial, 0., 1.);
      serial.refine_global(2);
      triangulation.copy_triangulation(serial);
      dof_handler.distribute_dofs(finite_element);
      locally_owned    = dof_handler.locally_owned_dofs();
      locally_relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
      constraints.reinit(locally_owned, locally_relevant);
      constraints.close();
      representation =
        std::make_unique<FiniteElementRepresentation<1, 2>>(triangulation,
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
  check_expression_lift(const bool mpi)
  {
    if (mpi)
      ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
    else
      ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);
    ASSERT_TRUE(SymbolicExpressionKernel::available());

    ParameterAcceptor::clear();
    LiftExpressionData         data;
    TensorProductLift<1, 2, 2> lift("/Expression lift/");
    lift.thickness                        = "0.2";
    lift.representative_quadrature_type   = "gauss";
    lift.representative_n_q_points        = 3;
    lift.representative_n_repetitions     = 1;
    lift.section.inclusion_type           = "hyper_ball";
    lift.section.refinement_level         = 1;
    lift.section.inclusion_degree         = 0;
    lift.section.selected_coefficients    = {0};
    lift.section.quadrature_type          = "gauss";
    lift.section.n_q_points               = 3;
    lift.section.n_quadrature_repetitions = 1;

    const auto expression =
      make_fe_expression(*data.representation,
                         {value(state_field(*data.representation, data.field),
                                "A")},
                         "A*A + 1.25*A");
    const auto  lifted  = expression.lift(lift);
    const auto &sampled = lifted.source_representation();
    const auto &plan    = sampled.sampling_plan();

    ImmersXLA::MPI::Vector state;
    state.reinit(data.locally_owned, MPI_COMM_WORLD);
    for (const auto index : data.locally_owned)
      state[index] = 0.5 + 0.2 * index;
    ImmersXLA::MPI::Vector direction;
    direction.reinit(data.locally_owned, MPI_COMM_WORLD);
    for (const auto index : data.locally_owned)
      direction[index] = 0.25 + 0.1 * index;

    StateView<ImmersXLA::MPI::Vector> state_view(data.layout, 0.);
    state_view.bind(data.field, state);
    EvaluationContext<ImmersXLA::MPI::Vector> context(0., state_view);
    const auto source_value    = sampled.evaluate(context);
    const auto source_jacobian = sampled.linearize(context);
    const auto lifted_value    = lifted.evaluate(context);
    for (std::size_t q = 0; q < lifted.lifted_points().size(); ++q)
      {
        const auto &point = lifted.lifted_points()[q];
        ASSERT_LT(point.source_representative_qpoint, plan.points().size());
        const auto source_index =
          plan.point_index(point.source_representative_qpoint);
        EXPECT_DOUBLE_EQ(lifted_value[lifted.point_index(q)],
                         point.mode_values[0] * source_value[source_index]);
      }

    const auto             lifted_jacobian = lifted.linearize(context);
    ImmersXLA::MPI::Vector lifted_action;
    lifted_jacobian.reinit_range_vector(lifted_action, false);
    lifted_jacobian.vmult(lifted_action, direction);
    ImmersXLA::MPI::Vector source_action;
    source_jacobian.reinit_range_vector(source_action, false);
    source_jacobian.vmult(source_action, direction);
    for (std::size_t q = 0; q < lifted.lifted_points().size(); ++q)
      {
        const auto &point = lifted.lifted_points()[q];
        const auto  source_index =
          plan.point_index(point.source_representative_qpoint);
        EXPECT_DOUBLE_EQ(lifted_action[lifted.point_index(q)],
                         point.mode_values[0] * source_action[source_index]);
      }

    ImmersXLA::MPI::Vector weights;
    weights.reinit(lifted.locally_owned_points(), MPI_COMM_WORLD);
    for (const auto index : lifted.locally_owned_points())
      weights[index] = 0.75 + 0.05 * index;
    ImmersXLA::MPI::Vector transpose;
    lifted_jacobian.reinit_domain_vector(transpose, false);
    lifted_jacobian.Tvmult(transpose, weights);
    double forward = 0.;
    for (const auto index : lifted.locally_owned_points())
      forward += lifted_action[index] * weights[index];
    EXPECT_NEAR(Utilities::MPI::sum(forward, MPI_COMM_WORLD),
                local_dot(data.locally_owned, direction, transpose),
                1.e-8);
  }
} // namespace


TEST(ExpressionRepresentation, TensorProductLiftValueAndJacobian) // NOLINT
{
  check_expression_lift(false);
}


TEST(ExpressionRepresentation, MPI_TensorProductLiftValueAndJacobian) // NOLINT
{
  check_expression_lift(true);
}
