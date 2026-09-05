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
#include <immersx/core/fe_space.h>
#include <immersx/core/semidiscrete_pde_models.h>
#include <immersx/core/weak_term.h>
#include <immersx/physics/navier_stokes.h>

#include <limits>
#include <memory>
#include <vector>

namespace ImmersX
{
  namespace navier_stokes_detail
  {
    template <int dim, int spacedim, typename Extractor>
    Field<dim, spacedim, Extractor>
    make_block_field(
      const FESpaceView<dim, spacedim>        &space,
      const FieldId                            id,
      const std::string                       &name,
      const Extractor                         &extractor,
      const dealii::types::global_dof_index    block_offset,
      const dealii::types::global_dof_index    block_size,
      const dealii::IndexSet                  &owned,
      const dealii::IndexSet                  &relevant,
      const dealii::AffineConstraints<double> &native_constraints)
    {
      using GlobalIndex               = dealii::types::global_dof_index;
      const auto               n_dofs = space.dof_handler().n_dofs();
      std::vector<GlobalIndex> execution_indices(
        n_dofs, std::numeric_limits<GlobalIndex>::max());
      for (GlobalIndex i = 0; i < block_size; ++i)
        execution_indices[block_offset + i] = i;

      auto constraints = std::make_shared<dealii::AffineConstraints<double>>();
      constraints->reinit(owned, relevant);
      for (const auto &line : native_constraints.get_lines())
        if (line.index >= block_offset &&
            line.index < block_offset + block_size)
          {
            const auto row = line.index - block_offset;
            constraints->add_line(row);
            for (const auto &entry : line.entries)
              if (entry.first >= block_offset &&
                  entry.first < block_offset + block_size)
                constraints->add_entry(row,
                                       entry.first - block_offset,
                                       entry.second);
            constraints->set_inhomogeneity(row, line.inhomogeneity);
          }
      constraints->close();

      return space.field(id, name, extractor)
        .reindexed(
          name, owned, relevant, std::move(execution_indices), constraints);
    }
  } // namespace navier_stokes_detail

  template <int dim, int spacedim = dim>
  struct NavierStokesFields
  {
    FieldId                                           velocity;
    FieldId                                           pressure;
    std::shared_ptr<const FESpaceView<dim, spacedim>> space;
  };

  template <typename Builder, int dim, int spacedim = dim>
  NavierStokesFields<dim, spacedim>
  contribute(Builder &builder, const NavierStokesSolver<dim, spacedim> &problem)
  {
    using VectorType = typename NavierStokesSolver<dim, spacedim>::VectorType;

    AssertThrow(problem.locally_owned_dofs_by_block().size() == 2,
                dealii::ExcMessage(
                  "Navier-Stokes semantic fields need two DoF blocks."));

    const auto velocity_id =
      builder.differential_field("velocity",
                                 problem.locally_owned_dofs_by_block()[0],
                                 problem.locally_relevant_dofs_by_block()[0]);
    const auto pressure_id =
      builder.algebraic_field("pressure",
                              problem.locally_owned_dofs_by_block()[1],
                              problem.locally_relevant_dofs_by_block()[1]);
    const auto mass = problem.density() * ImmersX::matrix_operator<VectorType>(
                                            problem.velocity_mass_matrix());
    const auto pressure_metric =
      ImmersX::matrix_operator<VectorType>(problem.pressure_metric_matrix());

    builder.saddle_point(pressure_id, {velocity_id}, pressure_metric);

    builder.preconditioner(
      velocity_id, [](const auto &linearized_matrix, const auto &prototype) {
        return make_amg_preconditioner(linearized_matrix, prototype);
      });
    builder.preconditioner(pressure_id,
                           [&problem](const auto &, const auto &prototype) {
                             return make_amg_preconditioner(
                               problem.pressure_metric_matrix(), prototype);
                           });

    auto space = std::make_shared<FESpaceView<dim, spacedim>>(
      problem.dof_handler(),
      problem.mapping(),
      problem.constraints(),
      &problem.locally_relevant_dofs());
    const auto u = navier_stokes_detail::make_block_field(
      *space,
      velocity_id,
      "velocity",
      problem.velocity_extractor(),
      0,
      problem.velocity_block_size(),
      problem.locally_owned_dofs_by_block()[0],
      problem.locally_relevant_dofs_by_block()[0],
      problem.constraints());
    const auto p = navier_stokes_detail::make_block_field(
      *space,
      pressure_id,
      "pressure",
      problem.pressure_extractor(),
      problem.velocity_block_size(),
      problem.locally_owned_dofs_by_block()[1].size(),
      problem.locally_owned_dofs_by_block()[1],
      problem.locally_relevant_dofs_by_block()[1],
      problem.constraints());
    const auto v = test(u);
    const auto q = test(p);

    auto stokes = builder.term(velocity_id, "stokes");
    stokes
      .residual([velocity_id, &problem, mass](const auto &context) {
        const auto &v_dot  = context.derivative(velocity_id);
        auto        result = mass.view * v_dot;
        typename SemiDiscreteModel<VectorType>::Operation forcing;
        forcing.reinit_vector = [v_dot](VectorType &vector, const bool omit) {
          vector.reinit(v_dot, omit);
        };
        forcing.apply = [&problem, time = context.time()](VectorType &vector) {
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
      .derivative(velocity_id,
                  semidiscrete_detail::constrained_matrix_operator(
                    mass, problem.constraints()));

    weak_term(2. * problem.viscosity() * symmetric_gradient(u),
              symmetric_gradient(v))
      .add(builder);
    weak_term(-1. * value(p), divergence(v)).add(builder);
    weak_term(divergence(u), q).add(builder);
    if (problem.include_convective_term())
      weak_term(problem.density() * (gradient(u) * u), v).add(builder);

    return {velocity_id, pressure_id, std::move(space)};
  }
} // namespace ImmersX

#endif // immersx_navier_stokes_semidiscrete_h
