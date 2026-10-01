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
#include <immersx/core/fe_space.h>
#include <immersx/core/semidiscrete_pde_models.h>
#include <immersx/physics/elastodynamics.h>

namespace ImmersX
{
  template <int dim, int spacedim = dim>
  struct ElastodynamicsFields
  {
    using Space = FiniteElementSpaceView<dim, spacedim>;
    using VectorField =
      Field<dim, spacedim, dealii::FEValuesExtractors::Vector>;

    VectorField                  displacement;
    VectorField                  velocity;
    std::shared_ptr<const Space> space;
  };

  template <typename Builder, int dim, int spacedim = dim>
  ElastodynamicsFields<dim, spacedim>
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

    const auto displacement_id =
      builder.field("displacement",
                    problem.locally_owned_dofs(),
                    problem.locally_relevant_dofs(),
                    free_components(problem.locally_owned_dofs(),
                                    problem.constraints()));
    const auto velocity_id =
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
    builder.preconditioner(displacement_id,
                           [](const auto &linearized_matrix,
                              const auto &prototype) {
                             return make_amg_preconditioner(linearized_matrix,
                                                            prototype);
                           });
    builder.preconditioner(
      velocity_id, [](const auto &linearized_matrix, const auto &prototype) {
        return make_amg_preconditioner(linearized_matrix, prototype);
      });

    using Space      = typename ElastodynamicsFields<dim, spacedim>::Space;
    auto       space = std::make_shared<Space>(problem.dof_handler(),
                                         problem.mapping(),
                                         problem.constraints(),
                                         &problem.locally_relevant_dofs());
    const auto displacement =
      space->field(displacement_id,
                   "displacement",
                   dealii::FEValuesExtractors::Vector(0));
    const auto velocity = space->field(velocity_id,
                                       "velocity",
                                       dealii::FEValuesExtractors::Vector(0));

    auto kinematic = builder.term(displacement_id, "kinematic");
    kinematic
      .residual(
        [displacement_id, velocity_id, &problem, mass](const auto &context) {
          problem.update_constraints(context.time());
          return semidiscrete_detail::constrained_residual(
            mass.view * context.derivative(displacement_id) -
              mass.view * context.state(velocity_id),
            context.state(displacement_id),
            problem.constraints());
        })
      .state(velocity_id,
             semidiscrete_detail::constrained_matrix_operator(
               -1. * mass, problem.constraints()))
      .state(displacement_id,
             semidiscrete_detail::constrained_matrix_identity_operator(
               mass, problem.constraints()))
      .derivative(displacement_id,
                  semidiscrete_detail::constrained_matrix_operator(
                    mass, problem.constraints()));

    auto dynamics = builder.term(velocity_id, "dynamics");
    dynamics
      .residual(
        [velocity_id, displacement_id, &problem, mass, stiffness, damping](
          const auto &context) {
          problem.update_constraints(context.time());
          const auto &v_dot  = context.derivative(velocity_id);
          auto        result = mass.view * v_dot +
                        stiffness.view * context.state(displacement_id) +
                        damping.view * context.state(velocity_id);
          typename SemiDiscreteModel<VectorType>::Operation forcing;
          forcing.reinit_vector = [v_dot](VectorType &vector, const bool omit) {
            vector.reinit(v_dot, omit);
          };
          forcing.apply = [&problem,
                           time = context.time()](VectorType &vector) {
            problem.body_force_at_time(time, vector);
          };
          forcing.apply_add = [&problem,
                               time = context.time()](VectorType &vector) {
            VectorType force;
            problem.body_force_at_time(time, force);
            vector += force;
          };
          return semidiscrete_detail::constrained_residual(
            result - forcing,
            context.state(velocity_id),
            problem.velocity_constraints());
        })
      .state(displacement_id,
             semidiscrete_detail::constrained_matrix_operator(
               stiffness, problem.velocity_constraints()))
      .state(velocity_id,
             semidiscrete_detail::constrained_matrix_operator_with_identity(
               damping, problem.velocity_constraints()))
      .derivative(velocity_id,
                  semidiscrete_detail::constrained_matrix_operator(
                    mass, problem.velocity_constraints()));

    return {displacement, velocity, std::move(space)};
  }

  /** Initialize the two-field adapter state from the problem state. */
  template <typename Adapter,
            typename Fields,
            int dim,
            int spacedim,
            typename GlobalVector>
  void
  initialize_elastodynamics_adapter_state(
    Adapter                                   &adapter,
    const Fields                              &fields,
    const ElastodynamicsSolver<dim, spacedim> &problem,
    GlobalVector                              &state,
    GlobalVector                              &state_dot)
  {
    adapter.field(state, fields.fields().displacement) = problem.displacement();
    adapter.field(state, fields.fields().velocity)     = problem.velocity();
    adapter.field(state_dot, fields.fields().displacement) = problem.velocity();
    typename ElastodynamicsSolver<dim, spacedim>::VectorType acceleration;
    problem.initial_acceleration(acceleration);
    adapter.field(state_dot, fields.fields().velocity) = acceleration;
  }
} // namespace ImmersX

#endif // immersx_elastodynamics_semidiscrete_h
