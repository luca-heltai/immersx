// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_navier_stokes_semidiscrete_h
#define immersx_navier_stokes_semidiscrete_h

#include <immersx/core/contributor.h>
#include <immersx/core/semidiscrete_pde_models.h>
#include <immersx/physics/navier_stokes.h>

namespace ImmersX
{
  struct NavierStokesFields
  {
    FieldId velocity;
    FieldId pressure;
  };

  template <typename Builder, int dim, int spacedim = dim>
  NavierStokesFields
  contribute(Builder &builder, const NavierStokesSolver<dim, spacedim> &problem)
  {
    using VectorType = typename NavierStokesSolver<dim, spacedim>::VectorType;

    AssertThrow(problem.locally_owned_dofs_by_block().size() == 2,
                dealii::ExcMessage(
                  "Navier-Stokes semantic fields need two DoF blocks."));

    const auto velocity =
      builder.differential_field("velocity",
                                 problem.locally_owned_dofs_by_block()[0],
                                 problem.locally_relevant_dofs_by_block()[0]);
    const auto pressure =
      builder.algebraic_field("pressure",
                              problem.locally_owned_dofs_by_block()[1],
                              problem.locally_relevant_dofs_by_block()[1]);
    const auto mass = problem.density() * ImmersX::matrix_operator<VectorType>(
                                            problem.velocity_mass_matrix());
    const auto viscosity = ImmersX::matrix_operator<VectorType>(
      problem.continuous_operator().block(0, 0));
    const auto pressure_operator = ImmersX::matrix_operator<VectorType>(
      problem.continuous_operator().block(0, 1));
    const auto divergence = ImmersX::matrix_operator<VectorType>(
      problem.continuous_operator().block(1, 0));
    const auto pressure_metric =
      ImmersX::matrix_operator<VectorType>(problem.pressure_metric_matrix());

    builder.saddle_point(pressure, {velocity}, pressure_metric);

    builder.preconditioner(
      velocity, [](const auto &linearized_matrix, const auto &prototype) {
        return make_amg_preconditioner(linearized_matrix, prototype);
      });
    builder.preconditioner(pressure,
                           [&problem](const auto &, const auto &prototype) {
                             return make_amg_preconditioner(
                               problem.pressure_metric_matrix(), prototype);
                           });

    auto stokes = builder.term(velocity, "stokes");
    stokes
      .residual(
        [velocity, pressure, &problem, mass, viscosity, pressure_operator](
          const auto &context) {
          const auto &v_dot  = context.derivative(velocity);
          auto        result = mass.view * v_dot +
                        viscosity.view * context.state(velocity) +
                        pressure_operator.view * context.state(pressure);
          typename SemiDiscreteModel<VectorType>::Operation forcing;
          forcing.reinit_vector = [v_dot](VectorType &vector, const bool omit) {
            vector.reinit(v_dot, omit);
          };
          forcing.apply = [&problem,
                           time = context.time()](VectorType &vector) {
            problem.velocity_forcing_at_time(time, vector);
            vector *= problem.density();
          };
          forcing.apply_add = [&problem,
                               time = context.time()](VectorType &vector) {
            VectorType force;
            problem.velocity_forcing_at_time(time, force);
            force *= problem.density();
            vector += force;
          };
          return semidiscrete_detail::constrained_operation(
            result - forcing, problem.constraints());
        })
      .state(velocity,
             semidiscrete_detail::constrained_matrix_operator(
               viscosity, problem.constraints()))
      .state(pressure,
             semidiscrete_detail::mixed_constrained_matrix_operator(
               pressure_operator, problem.constraints(), 0))
      .derivative(velocity,
                  semidiscrete_detail::constrained_matrix_operator(
                    mass, problem.constraints()));

    auto incompressibility = builder.term(pressure, "incompressibility");
    incompressibility
      .residual([velocity, &problem, divergence](const auto &context) {
        return semidiscrete_detail::mixed_constrained_operation(
          divergence.view * context.state(velocity),
          problem.constraints(),
          problem.velocity_block_size());
      })
      .state(velocity,
             semidiscrete_detail::mixed_constrained_matrix_operator(
               divergence,
               problem.constraints(),
               problem.velocity_block_size()));

    return {velocity, pressure};
  }
} // namespace ImmersX

#endif // immersx_navier_stokes_semidiscrete_h
