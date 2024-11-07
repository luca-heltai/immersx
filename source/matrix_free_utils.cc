// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 */



#include <matrix_free_utils.h>


template <int dim, typename number, int n_components>
CouplingOperator<dim, number, n_components>::CouplingOperator(
  const Inclusions<dim>           &inclusions_,
  const DoFHandler<dim>           &dof_handler_,
  const AffineConstraints<number> &constraints_,
  const MappingQ<dim>             &mapping_,
  const FiniteElement<dim>        &fe_)
  : rpe{{1e-9, true, inclusions_.rtree_extraction_level, {}}}
  , n_coefficients{inclusions_.n_coefficients}
{
  [[maybe_unused]] auto parallel_triangulation =
    dynamic_cast<const parallel::distributed::Triangulation<dim, dim> *>(
      &dof_handler_.get_triangulation());
  Assert((parallel_triangulation != nullptr),
         ExcMessage("Only p::d::T triangulations are supported."));

  mapping     = &mapping_;
  fe          = &fe_;
  constraints = &constraints_;
  dof_handler = &dof_handler_;
  inclusions  = &inclusions_;

  //  Get all locally owned support points from the inclusions
  std::vector<Point<dim>> locations(
    inclusions->inclusions_as_particles.n_locally_owned_particles());
  const_cast<Particles::ParticleHandler<dim> &>(
    inclusions->inclusions_as_particles)
    .get_particle_positions(locations);

  rpe.reinit(locations, dof_handler->get_triangulation(), *mapping);

  // We should make sure all points have been found. Do this only in debug mode
  Assert(rpe.all_points_found(),
         ExcInternalError("Not all points have been"
                          "found during the geometric search procedure."));
}



template <int dim, typename number, int n_components>
void
CouplingOperator<dim, number, n_components>::initialize_dof_vector(
  VectorType &vec) const
{
  (void)vec;
}



template <int dim, typename number, int n_components>
void
CouplingOperator<dim, number, n_components>::vmult(VectorType       &dst,
                                                   const VectorType &src) const
{
  Assert(rpe.is_ready(),
         ExcInternalError("Remote evaluator has not been initialized."));

  src.update_ghost_values();
  dst = 0.;

  std::vector<number> integration_values;
  const unsigned int  n_dofs_per_cell = fe->n_dofs_per_cell();

  // collect
  std::vector<types::global_dof_index> inclusion_dof_indices(
    inclusions->n_coefficients);

  auto particle = inclusions->inclusions_as_particles.begin();
  while (particle != inclusions->inclusions_as_particles.end())
    {
      const auto id                   = particle->get_id();
      inclusion_dof_indices           = inclusions->get_dof_indices(id);
      const auto &inclusion_fe_values = inclusions->get_fe_values(id);

      double value_per_particle = 0.;
      for (unsigned int j = 0; j < inclusions->n_coefficients; ++j)
        {
          value_per_particle +=
            inclusion_fe_values[j] * src(inclusion_dof_indices[j]);
        }
      integration_values.push_back(value_per_particle);
      ++particle;
    }


  const auto integration_function = [&](const auto &values,
                                        const auto &cell_data) {
    FEPointEvaluation<1, dim, dim> phi_force(*mapping, *fe, update_values);

    std::vector<double>                  local_values;
    std::vector<types::global_dof_index> local_dof_indices;

    for (const auto cell : cell_data.cell_indices())
      {
        const auto cell_dh =
          cell_data.get_active_cell_iterator(cell)->as_dof_handler_iterator(
            *dof_handler);


        const auto unit_points      = cell_data.get_unit_points(cell);
        const auto inclusion_values = cell_data.get_data_view(cell, values);

        phi_force.reinit(cell_dh, unit_points);

        for (const auto q : phi_force.quadrature_point_indices())
          phi_force.submit_value(inclusion_values[q], q);

        local_values.resize(n_dofs_per_cell);
        phi_force.test_and_sum(local_values, EvaluationFlags::values);

        local_dof_indices.resize(n_dofs_per_cell);
        cell_dh->get_dof_indices(local_dof_indices);
        constraints->distribute_local_to_global(local_values,
                                                local_dof_indices,
                                                dst);
      }
  };

  rpe.template process_and_evaluate<number>(integration_values,
                                            integration_function);
  dst.compress(VectorOperation::add);
}



template <int dim, typename number, int n_components>
void
CouplingOperator<dim, number, n_components>::Tvmult(VectorType       &dst,
                                                    const VectorType &src) const
{
  Assert(rpe.is_ready(),
         ExcInternalError("Remote evaluator has not been initialized."));

  src.update_ghost_values();
  dst = 0.;

  const auto evaluation_function = [&](const auto &values,
                                       const auto &cell_data) {
    FEPointEvaluation<n_components, dim, dim> evaluator(*mapping,
                                                        *fe,
                                                        update_values);

    std::vector<double> local_values;
    const unsigned int  n_dofs_per_cell = fe->n_dofs_per_cell();

    for (const auto cell : cell_data.cell_indices())
      {
        const auto cell_dh =
          cell_data.get_active_cell_iterator(cell)->as_dof_handler_iterator(
            *dof_handler);

        const auto unit_points = cell_data.get_unit_points(cell);
        const auto local_value = cell_data.get_data_view(cell, values);

        local_values.resize(n_dofs_per_cell);
        cell_dh->get_dof_values(*constraints,
                                src,
                                local_values.begin(),
                                local_values.end());

        evaluator.reinit(cell_dh, unit_points);
        evaluator.evaluate(local_values, EvaluationFlags::values);

        for (unsigned int q = 0; q < unit_points.size(); ++q)
          local_value[q] = evaluator.get_value(q);
      }
  };


  const std::vector<number> output =
    rpe.template evaluate_and_process<number>(evaluation_function);

  std::vector<types::global_dof_index> inclusion_dof_indices(
    inclusions->n_coefficients);

  auto particle = inclusions->inclusions_as_particles.begin();
  while (particle != inclusions->inclusions_as_particles.end())
    {
      const auto id                   = particle->get_id();
      const auto local_id             = particle->get_local_index();
      inclusion_dof_indices           = inclusions->get_dof_indices(id);
      const auto &inclusion_fe_values = inclusions->get_fe_values(id);

      for (unsigned int j = 0; j < inclusions->n_coefficients; ++j)
        dst(inclusion_dof_indices[j]) +=
          inclusion_fe_values[j] * output[local_id];

      ++particle;
    }

  dst.compress(VectorOperation::add);
}



template <int dim, typename number, int n_components>
void
CouplingOperator<dim, number, n_components>::vmult_add(
  VectorType       &dst,
  const VectorType &src) const
{
  Assert(rpe.is_ready(),
         ExcInternalError("Remote evaluator has not been initialized."));
  VectorType tmp_vector;
  tmp_vector.reinit(dst, true);
  vmult(tmp_vector, src);
  dst += tmp_vector;
}

template <int dim, typename number, int n_components>
void
CouplingOperator<dim, number, n_components>::Tvmult_add(
  VectorType       &dst,
  const VectorType &src) const
{
  Assert(rpe.is_ready(),
         ExcInternalError("Remote evaluator has not been initialized."));
  VectorType tmp_vector;
  tmp_vector.reinit(dst, true);
  Tvmult(tmp_vector, src);
  dst += tmp_vector;
}



// Template instantiations
template class CouplingOperator<2, double, 1>;
template class CouplingOperator<3, double, 1>;
template class CouplingOperator<2, float, 1>;
template class CouplingOperator<3, float, 1>;