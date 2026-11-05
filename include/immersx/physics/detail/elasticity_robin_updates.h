// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on the deal.II library.
// ---------------------------------------------------------------------
#ifndef immersx_elasticity_robin_updates_h
#define immersx_elasticity_robin_updates_h

#include <deal.II/base/function_lib.h>

#include <deal.II/lac/affine_constraints.h>

#include <immersx/core/robin_boundary.h>

#include <vector>

namespace ImmersX::robin_detail
{
  /** Update boundary springs without rebuilding bulk mass or damping. */
  template <int spacedim, typename Parameters, typename Matrix>
  bool
  update_elasticity_springs(
    const dealii::DoFHandler<spacedim>      &dofs,
    const dealii::Mapping<spacedim>         &mapping,
    const dealii::FiniteElement<spacedim>   &fe,
    const dealii::Quadrature<spacedim - 1>  &quadrature,
    const dealii::AffineConstraints<double> &constraints,
    const Parameters                        &parameters,
    const double                             old_time,
    const double                             new_time,
    Matrix                                  &stiffness,
    Matrix                                  *effective,
    const double                             effective_scale,
    const MPI_Comm                           communicator)
  {
    using namespace dealii;
    if (parameters.robin_ids.empty() || old_time == new_time)
      return false;
    FEFaceValues<spacedim>           values(mapping,
                                  fe,
                                  quadrature,
                                  update_values | update_quadrature_points |
                                    update_JxW_values);
    const FEValuesExtractors::Vector displacement(0);
    FullMatrix<double> before(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
    FullMatrix<double> after(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
    Vector<double>     unused(fe.n_dofs_per_cell());
    Functions::ZeroFunction<spacedim>    zero(spacedim);
    std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
    std::vector<std::pair<types::global_dof_index, double>> diagonal;
    std::vector<std::pair<types::global_dof_index, double>> effective_diagonal;
    for (const auto &line : constraints.get_lines())
      if (stiffness.locally_owned_range_indices().is_element(line.index))
        {
          diagonal.emplace_back(line.index,
                                stiffness.el(line.index, line.index));
          if (effective)
            effective_diagonal.emplace_back(line.index,
                                            effective->el(line.index,
                                                          line.index));
        }
    unsigned invalid = 0;
    double   change  = 0.;
    for (const auto &cell : dofs.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          before = 0.;
          after  = 0.;
          unused = 0.;
          for (const auto face : cell->face_indices())
            if (cell->face(face)->at_boundary() &&
                parameters.robin_ids.count(cell->face(face)->boundary_id()))
              {
                const auto id = cell->face(face)->boundary_id();
                values.reinit(cell, face);
                parameters.set_boundary_condition_times(old_time);
                invalid |= add_face_terms(values,
                                          displacement,
                                          parameters.get_robin_coefficient(id),
                                          zero,
                                          before,
                                          unused);
                parameters.set_boundary_condition_times(new_time);
                invalid |= add_face_terms(values,
                                          displacement,
                                          parameters.get_robin_coefficient(id),
                                          zero,
                                          after,
                                          unused);
              }
          after.add(-1., before);
          change = std::max(change, after.frobenius_norm());
          cell->get_dof_indices(ids);
          constraints.distribute_local_to_global(after, ids, stiffness);
          if (effective)
            {
              after *= effective_scale;
              constraints.distribute_local_to_global(after, ids, *effective);
            }
        }
    parameters.set_boundary_condition_times(new_time);
    assert_valid_boundary_data(invalid, communicator);
    stiffness.compress(VectorOperation::add);
    if (effective)
      effective->compress(VectorOperation::add);
    // Affine projection supplies artificial essential diagonals. Their values
    // belong to the original bulk operator, not to this boundary difference.
    for (const auto &[index, value] : diagonal)
      stiffness.set(index, index, value);
    if (effective)
      for (const auto &[index, value] : effective_diagonal)
        effective->set(index, index, value);
    stiffness.compress(VectorOperation::insert);
    if (effective)
      effective->compress(VectorOperation::insert);
    return Utilities::MPI::max(change, communicator) != 0.;
  }
} // namespace ImmersX::robin_detail
#endif
