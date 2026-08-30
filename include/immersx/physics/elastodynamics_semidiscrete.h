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

#include <immersx/algebra/local_preconditioner.h>
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

    const auto free_components =
      [](const dealii::IndexSet                  &owned,
         const dealii::AffineConstraints<double> &constraints) {
        dealii::IndexSet result(owned.size());
        for (const auto index : owned)
          if (!constraints.is_constrained(index))
            result.add_index(index);
        result.compress();
        return result;
      };

    const auto displacement =
      builder.field("displacement",
                    problem.locally_owned_dofs(),
                    problem.locally_relevant_dofs(),
                    free_components(problem.locally_owned_dofs(),
                                    problem.constraints()));
    const auto velocity =
      builder.field("velocity",
                    problem.locally_owned_dofs(),
                    problem.locally_relevant_dofs(),
                    free_components(problem.locally_owned_dofs(),
                                    problem.velocity_constraints()));

    const auto mass =
      ImmersX::matrix_operator<VectorType>(problem.mass_matrix());
    const auto stiffness =
      ImmersX::matrix_operator<VectorType>(problem.stiffness_matrix());
    const auto damping =
      ImmersX::matrix_operator<VectorType>(problem.damping_matrix());

    builder.preconditioner(
      displacement, [](const auto &linearized_matrix, const auto &prototype) {
        return make_amg_preconditioner(linearized_matrix, prototype);
      });
    builder.preconditioner(
      velocity, [](const auto &linearized_matrix, const auto &prototype) {
        return make_amg_preconditioner(linearized_matrix, prototype);
      });

    auto kinematic = builder.term(displacement, "kinematic");
    kinematic
      .residual([displacement, velocity, &problem, mass](const auto &context) {
        return semidiscrete_detail::constrained_operation(
          mass.view * context.derivative(displacement) -
            mass.view * context.state(velocity),
          problem.constraints());
      })
      .state(velocity,
             semidiscrete_detail::constrained_matrix_operator(
               -1. * mass, problem.constraints()))
      .derivative(displacement,
                  semidiscrete_detail::constrained_matrix_operator(
                    mass, problem.constraints()));

    auto dynamics = builder.term(velocity, "dynamics");
    dynamics
      .residual([velocity, displacement, &problem, mass, stiffness, damping](
                  const auto &context) {
        const auto &v_dot  = context.derivative(velocity);
        auto        result = mass.view * v_dot +
                      stiffness.view * context.state(displacement) +
                      damping.view * context.state(velocity);
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
             semidiscrete_detail::constrained_matrix_operator(
               stiffness, problem.velocity_constraints()))
      .state(velocity,
             semidiscrete_detail::constrained_matrix_operator(
               damping, problem.velocity_constraints()))
      .derivative(velocity,
                  semidiscrete_detail::constrained_matrix_operator(
                    mass, problem.velocity_constraints()));

    return {displacement, velocity};
  }
} // namespace ImmersX

#endif // immersx_elastodynamics_semidiscrete_h
