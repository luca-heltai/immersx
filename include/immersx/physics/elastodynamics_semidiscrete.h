// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_elastodynamics_semidiscrete_h
#define immersx_elastodynamics_semidiscrete_h

#include <immersx/core/contributor.h>
#include <immersx/core/semidiscrete_pde_models.h>
#include <immersx/physics/elastodynamics.h>

namespace ImmersX
{
  struct ElastodynamicsFields
  {
    FieldId displacement;
    FieldId velocity;
  };

  template <typename Builder, int dim, int spacedim = dim>
  ElastodynamicsFields
  contribute(Builder                                   &builder,
             const ElastodynamicsSolver<dim, spacedim> &problem)
  {
    using VectorType = typename ElastodynamicsSolver<dim, spacedim>::VectorType;

    const auto displacement =
      builder.differential_field("displacement",
                                 problem.locally_owned_dofs(),
                                 problem.locally_relevant_dofs());
    const auto velocity =
      builder.differential_field("velocity",
                                 problem.locally_owned_dofs(),
                                 problem.locally_relevant_dofs());

    const auto mass = ImmersX::payload_free(
      dealii::linear_operator<VectorType, VectorType>(problem.mass_matrix()));
    const auto stiffness =
      ImmersX::payload_free(dealii::linear_operator<VectorType, VectorType>(
        problem.stiffness_matrix()));
    const auto damping =
      ImmersX::payload_free(dealii::linear_operator<VectorType, VectorType>(
        problem.damping_matrix()));

    auto kinematic = builder.term(displacement, "kinematic");
    kinematic
      .residual([displacement, velocity, &problem, mass](const auto &context) {
        return semidiscrete_detail::constrained_operation(
          mass * context.derivative(displacement) -
            mass * context.state(velocity),
          problem.constraints());
      })
      .state(velocity,
             semidiscrete_detail::constrained_operator(-1. * mass,
                                                       problem.constraints()))
      .derivative(
        displacement,
        semidiscrete_detail::constrained_operator(mass, problem.constraints()));

    auto dynamics = builder.term(velocity, "dynamics");
    dynamics
      .residual([velocity, displacement, &problem, mass, stiffness, damping](
                  const auto &context) {
        const auto &v_dot = context.derivative(velocity);
        auto result = mass * v_dot + stiffness * context.state(displacement) +
                      damping * context.state(velocity);
        typename SemiDiscreteModel<VectorType>::Operation forcing;
        forcing.reinit_vector = [v_dot](VectorType &vector, const bool omit) {
          vector.reinit(v_dot, omit);
        };
        forcing.apply = [&problem, time = context.time()](VectorType &vector) {
          problem.body_force_at_time(time, vector);
        };
        forcing.apply_add = [&problem,
                             time = context.time()](VectorType &vector) {
          VectorType force;
          problem.body_force_at_time(time, force);
          vector += force;
        };
        return semidiscrete_detail::constrained_operation(
          result - forcing, problem.velocity_constraints());
      })
      .state(displacement,
             semidiscrete_detail::constrained_operator(
               stiffness, problem.velocity_constraints()))
      .state(velocity,
             semidiscrete_detail::constrained_operator(
               damping, problem.velocity_constraints()))
      .derivative(velocity,
                  semidiscrete_detail::constrained_operator(
                    mass, problem.velocity_constraints()));

    return {displacement, velocity};
  }
} // namespace ImmersX

#endif // immersx_elastodynamics_semidiscrete_h
