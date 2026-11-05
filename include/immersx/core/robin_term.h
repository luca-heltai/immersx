// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_robin_term_h
#define immersx_robin_term_h

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <immersx/core/robin_boundary.h>
#include <immersx/core/weak_term.h>

#include <memory>
#include <utility>
#include <vector>

namespace ImmersX
{
  /** Assemble a static scalar or vector Robin weak term on the selected ids.
   * Function lookup callbacks return references to beta and g for a boundary
   * id; the scale callback permits modulated loads. Callers set function times
   * and validate disjoint boundary condition ids before adding this term.
   */
  template <typename FieldType,
            typename Ids,
            typename Coefficient,
            typename LoadFunction,
            typename LoadScale>
  auto
  robin_boundary_term(FieldType    displacement,
                      Ids          ids,
                      Coefficient  coefficient,
                      LoadFunction load_function,
                      LoadScale    load_scale,
                      const double matrix_scale = 1.)
  {
    return [displacement,
            ids           = std::move(ids),
            coefficient   = std::move(coefficient),
            load_function = std::move(load_function),
            load_scale    = std::move(load_scale),
            matrix_scale](auto &builder) {
      using namespace dealii;
      constexpr int dim       = FieldType::dimension();
      constexpr int spacedim  = FieldType::spacedimension();
      using Builder           = std::decay_t<decltype(builder)>;
      using Matrix            = ImmersXLA::MPI::SparseMatrix;
      using Vector            = ImmersXLA::MPI::Vector;
      const auto communicator = displacement.space().mpi_communicator();
      if (ids.empty())
        return displacement;
      auto                   matrix   = std::make_shared<Matrix>();
      auto                   rhs      = std::make_shared<Vector>();
      const auto            &owned    = displacement.locally_owned_dofs();
      const auto            &relevant = displacement.locally_relevant_dofs();
      DynamicSparsityPattern sparsity(relevant);
      DoFTools::make_sparsity_pattern(displacement.dof_handler(),
                                      sparsity,
                                      displacement.constraints(),
                                      true);
      SparsityTools::distribute_sparsity_pattern(sparsity,
                                                 owned,
                                                 communicator,
                                                 relevant);
      auto matrix_sparsity = detail::initialize_weak_matrix(
        *matrix, owned, owned, sparsity, communicator);
      rhs->reinit(owned, communicator);
      const auto                 &fe = displacement.finite_element();
      FEFaceValues<dim, spacedim> values(
        displacement.mapping(),
        fe,
        compatibility::gauss_quadrature<dim - 1>(
          fe.reference_cell().face_reference_cell(0), fe.degree + 1),
        update_values | update_quadrature_points | update_JxW_values);
      const auto                           n = fe.n_dofs_per_cell();
      FullMatrix<double>                   local(n, n);
      dealii::Vector<double>               load(n);
      std::vector<types::global_dof_index> indices(n);
      unsigned int                         invalid = 0;
      for (const auto &cell :
           displacement.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            local = 0.;
            load  = 0.;
            for (const auto face : cell->face_indices())
              if (cell->face(face)->at_boundary() &&
                  ids.count(cell->face(face)->boundary_id()))
                {
                  values.reinit(cell, face);
                  invalid |= robin_detail::add_face_terms(
                    values,
                    displacement.extractor(),
                    coefficient(cell->face(face)->boundary_id()),
                    load_function(cell->face(face)->boundary_id()),
                    local,
                    load,
                    matrix_scale,
                    load_scale(cell->face(face)->boundary_id()));
                }
            cell->get_dof_indices(indices);
            displacement.constraints().distribute_local_to_global(
              local, load, indices, *matrix, *rhs);
          }
      robin_detail::assert_valid_boundary_data(invalid, communicator);
      matrix->compress(VectorOperation::add);
      rhs->compress(VectorOperation::add);
      auto op             = builder.matrix_operator(*matrix);
      op.materialize      = [matrix, matrix_sparsity] { return matrix; };
      op.materialize_into = [matrix](Matrix &destination) {
        destination.copy_from(*matrix);
      };
      builder.term(displacement.id(), "robin")
        .residual([displacement, matrix, rhs](const auto &context) {
          const auto &state = context.state(displacement.id());
          typename Builder::Model::Operation result;
          result.reinit_vector = [rhs](auto &v, const bool) { v.reinit(*rhs); };
          result.apply         = [matrix, rhs, &state](auto &v) {
            matrix->vmult(v, state);
            v -= *rhs;
          };
          result.apply_add = [matrix, rhs, &state](auto &v) {
            Vector temporary;
            temporary.reinit(*rhs);
            matrix->vmult(temporary, state);
            temporary -= *rhs;
            v += temporary;
          };
          return result;
        })
        .state(displacement.id(), op);
      return displacement;
    };
  }
} // namespace ImmersX
#endif
