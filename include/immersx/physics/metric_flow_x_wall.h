// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of the ImmersX application, based on deal.II.
// ---------------------------------------------------------------------
#ifndef immersx_metric_flow_x_wall_h
#define immersx_metric_flow_x_wall_h

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <immersx/core/constraint.h>
#  include <immersx/core/observable_lift.h>
#  include <immersx/core/weak_term.h>
#  include <immersx/physics/metric_flow_x_pressure.h>

#  include <memory>
#  include <type_traits>
#  include <vector>

namespace ImmersX
{
  /** Couple native flow radius to lifted displacement with radial traction.
   * The reaction field selects the solid equilibrium row: displacement for
   * statics, velocity for the first-order elastodynamics formulation.
   * Geometry, reference fields and the native flow must outlive the adapter.
   */
  template <typename Flow,
            typename Displacement,
            typename FlowFields,
            typename Multiplier,
            typename WallLift,
            typename Radius,
            typename Supports,
            typename Projection,
            typename Metric,
            typename Reaction>
  auto
  metric_flow_x_wall_constraint(Flow                &flow_problem,
                                Displacement         displacement,
                                FlowFields           flow_fields,
                                Multiplier           lambda,
                                const WallLift      &wall_lift,
                                Radius               reference_radius,
                                const Supports      &supports,
                                Projection           pressure_projection,
                                Metric               multiplier_metric,
                                Reaction             reaction_field,
                                std::vector<FieldId> primal_ids,
                                const bool           pressure_feedback = true)
  {
    using namespace dealii;
    const auto reference_wall =
      lift(value(lambda), wall_lift, reference_radius);
    return [=, &flow_problem, &wall_lift](auto &builder) {
      const auto id          = builder.algebraic_field(lambda.name(),
                                              lambda.locally_owned_dofs(),
                                              lambda.locally_relevant_dofs());
      const auto multiplier  = lambda.with_id(id);
      const auto radius      = reference_radius;
      const auto lambda_wall = reference_wall.with_id(id);
      const auto negative_wall_displacement =
        lift(radius - sqrt(value(flow_fields.area) / numbers::PI),
             wall_lift,
             radius);

      weak_term(value(displacement), test(lambda_wall))
        .on(supports.overlap)
        .with_cached_assembly()
        .add_constraint_terms(builder, id, 1., 0, reaction_field);
      weak_term(negative_wall_displacement, test(multiplier))
        .on(supports.overlap)
        .add(builder);
      weak_term(-1. * value(multiplier), test(multiplier))
        .on(supports.outside)
        .add(builder);

      if (pressure_feedback)
        {
          const auto pressure      = -1. * value(multiplier);
          const auto flow_id       = flow_fields.state;
          auto       pressure_term = builder.term(flow_id, "native-pressure");
          pressure_term.residual(
            [&flow_problem, pressure_projection, pressure, flow_id, id](
              const auto &context) {
              auto contribution = std::make_shared<MetricFlowX::VectorType>(
                flow_problem.make_state());
              *contribution = 0.;
              flow_problem.add_external_pressure_residual(
                context.time(),
                context.state(flow_id),
                metric_flow_x_pressure_provider(pressure,
                                                context.state(id),
                                                pressure_projection),
                *contribution);
              dealii::PackagedOperation<MetricFlowX::VectorType> result;
              result.reinit_vector = [&flow_problem](auto &dst, const bool) {
                flow_problem.reinit_state(dst);
              };
              result.apply = [contribution](auto &dst) { dst = *contribution; };
              result.apply_add = [contribution](auto &dst) {
                dst += *contribution;
              };
              return result;
            });
          typename std::decay_t<decltype(builder)>::Model::MatrixOperatorFactory
            pressure_jacobian =
              [&flow_problem, pressure_projection, pressure, flow_id](
                const auto &context) {
                return metric_flow_x_pressure_operator(flow_problem,
                                                       pressure,
                                                       context.state(flow_id),
                                                       context.time(),
                                                       pressure_projection);
              };
          pressure_term.state(id, std::move(pressure_jacobian));
          if constexpr (detail::has_external_pressure_state_jacobian<
                          std::decay_t<decltype(flow_problem)>>::value)
            {
              typename std::decay_t<decltype(builder)>::Model::
                MatrixOperatorFactory pressure_state_jacobian =
                  [&flow_problem, pressure_projection, pressure, flow_id, id](
                    const auto &context) {
                    return metric_flow_x_pressure_state_operator(
                      flow_problem,
                      pressure,
                      context.state(id),
                      context.state(flow_id),
                      context.time(),
                      pressure_projection);
                  };
              pressure_term.state(flow_id, std::move(pressure_state_jacobian));
            }
        }


      auto metric        = builder.matrix_operator(*multiplier_metric);
      metric.materialize = [multiplier_metric] { return multiplier_metric; };
      metric.materialize_into = [multiplier_metric](auto &destination) {
        destination.copy_from(*multiplier_metric);
      };
      builder.saddle_point(id, primal_ids, metric);
      return ConstraintFields<std::decay_t<decltype(lambda)>>{multiplier};
    };
  }
} // namespace ImmersX
#endif // IMMERSX_WITH_METRIC_FLOW_X
#endif // immersx_metric_flow_x_wall_h
