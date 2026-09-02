// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_metric_flow_x_h
#define immersx_metric_flow_x_h

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <immersx/core/contributor.h>
#  include <immersx/core/representation.h>
#  include <metric_flow_x/blood_flow_system.h>

#  include <utility>

namespace ImmersX
{
  /** Non-owning description of a MetricFlowX Problem contribution.
   *
   * BloodFlowSystem owns the mesh, physics, and native vectors.  This value
   * only supplies the mutable access needed by native assembly while allowing
   * the public adapter::add() API to retain its logically-const contributor
   * contract.
   */
  template <int dim, int spacedim = dim>
  struct MetricFlowXProblem
  {
    using Problem = ::MetricFlowX::BloodFlowSystem<dim, spacedim>;

    Problem *problem = nullptr;
  };

  /** Semantic fields and native component views for one BloodFlowSystem. */
  struct MetricFlowXFields
  {
    FieldId            state;
    FieldComponentView area;
    FieldComponentView velocity;
  };

  /** Make a non-owning MetricFlowX contribution descriptor. */
  template <int dim, int spacedim>
  MetricFlowXProblem<dim, spacedim>
  metric_flow_x(::MetricFlowX::BloodFlowSystem<dim, spacedim> &problem)
  {
    return {&problem};
  }

  /** Register BloodFlowSystem as one mixed differential/algebraic Field. */
  template <typename Builder, int dim, int spacedim>
  MetricFlowXFields
  contribute(Builder                                 &builder,
             const MetricFlowXProblem<dim, spacedim> &description)
  {
    using Problem = typename MetricFlowXProblem<dim, spacedim>::Problem;
    using Vector  = ::MetricFlowX::VectorType;
    using Matrix  = ::MetricFlowX::MatrixType;

    AssertThrow(description.problem != nullptr,
                dealii::ExcMessage("A MetricFlowX Problem cannot be null."));
    Problem &problem = *description.problem;

    const FieldId state = builder.field("state",
                                        problem.locally_owned_dofs(),
                                        problem.locally_relevant_dofs(),
                                        problem.differential_dofs());

    auto term = builder.term(state, "blood-flow");
    term.residual([description, state](const auto &context) {
      const auto *y    = &context.state(state);
      const auto *ydot = &context.derivative(state);
      const auto  time = context.time();

      dealii::PackagedOperation<Vector> result;
      result.reinit_vector = [description](Vector &vector, const bool) {
        description.problem->reinit_state(vector);
      };
      result.apply = [description, y, ydot, time](Vector &destination) {
        description.problem->assemble_residual(time, *y, *ydot, destination);
      };
      result.apply_add = [description, y, ydot, time](Vector &destination) {
        Vector contribution;
        description.problem->reinit_state(contribution);
        description.problem->assemble_residual(time, *y, *ydot, contribution);
        destination += contribution;
      };
      return result;
    });

    typename Builder::Model::MatrixOperatorFactory state_factory =
      [description, state](const auto &context) {
        AssertThrow(context.has_state_derivative(),
                    dealii::ExcMessage(
                      "MetricFlowX Jacobians require a state derivative."));
        description.problem->assemble_state_jacobian(context.time(),
                                                     context.state(state),
                                                     context.derivative(state));
        return matrix_operator<Vector, Matrix>(
          description.problem->state_jacobian_matrix());
      };
    term.state(state, std::move(state_factory));

    typename Builder::Model::MatrixOperatorFactory derivative_factory =
      [description, state](const auto &context) {
        AssertThrow(context.has_state_derivative(),
                    dealii::ExcMessage(
                      "MetricFlowX Jacobians require a state derivative."));
        description.problem->assemble_derivative_jacobian(
          context.time(), context.state(state), context.derivative(state));
        return matrix_operator<Vector, Matrix>(
          description.problem->derivative_jacobian_matrix());
      };
    term.derivative(state, std::move(derivative_factory));

    return {
      state,
      FieldComponentView(state,
                         problem.component_dofs(Problem::Component::area)),
      FieldComponentView(state,
                         problem.component_dofs(Problem::Component::velocity))};
  }
} // namespace ImmersX

#endif // IMMERSX_WITH_METRIC_FLOW_X

#endif // immersx_metric_flow_x_h
