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


#ifndef rdlm_reduced_coupling_h
#define rdlm_reduced_coupling_h

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/polynomials_p.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <fstream>

#include "particle_coupling.h"
#include "tensor_product_space.h"

using namespace dealii;

/**
 *
 */
template <int reduced_dim, int dim, int spacedim = dim, int n_components = 1>
struct ReducedCouplingParameters : public ParameterAcceptor
{
  /// Constructor that registers parameters.
  ReducedCouplingParameters();

  /// TensorProductSpace parameters.
  TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components>
    tensor_product_space_parameters;

  /// ParticleCoupling parameters.
  ParticleCouplingParameters<spacedim> particle_coupling_parameters;

  /// Name of the grid to read from a file.
  std::string reduced_grid_name = "";

  /// Number of pre_refinements to apply to the grid, before transforming it to
  /// a fully distributed grid.
  unsigned int pre_refinement = 0;

  /// Reduced coupling right hand side
  std::vector<std::string> coupling_rhs_expressions = {"0"};
};

template <int reduced_dim, int dim, int spacedim = dim, int n_components = 1>
struct ReducedCoupling
  : public TensorProductSpace<reduced_dim, dim, spacedim, n_components>,
    public ParticleCoupling<spacedim>
{
  /// Constructor that initializes the parameters.
  ReducedCoupling(
    const parallel::TriangulationBase<spacedim> &background_tria,
    const ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components>
      &par);

  /// Initialize the tensor product space and particle coupling.
  void
  initialize(
    const Mapping<spacedim> &mapping = StaticMappingQ1<spacedim>::mapping);

  void
  assemble_coupling_sparsities(
    DynamicSparsityPattern          &dsp,
    DynamicSparsityPattern          &dsp_transpose,
    const DoFHandler<spacedim>      &dh,
    const AffineConstraints<double> &constraints) const;

  template <typename MatrixType>
  void
  assemble_coupling_matrices(
    MatrixType                      &coupling_matrix,
    MatrixType                      &coupling_matrix_transpose,
    const DoFHandler<spacedim>      &dh,
    const AffineConstraints<double> &constraints) const;


  template <typename VectorType>
  void
  assemble_reduced_rhs(VectorType &reduced_rhs) const;

  const AffineConstraints<double> &
  get_coupling_constraints() const;

private:
  const MPI_Comm mpi_communicator;

  const ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components>
    &par;

  /// The triangulation of the background domain.
  SmartPointer<const parallel::TriangulationBase<spacedim>> background_tria;

  AffineConstraints<double> coupling_constraints;

  /// The coupling rhs expression
  std::unique_ptr<FunctionParser<spacedim>> coupling_rhs;
};


// Template specializations

template <int reduced_dim, int dim, int spacedim, int n_components>
template <typename MatrixType>
inline void
ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
  assemble_coupling_matrices(MatrixType &coupling_matrix,
                             MatrixType &coupling_matrix_transpose,
                             const DoFHandler<spacedim>      &dh,
                             const AffineConstraints<double> &constraints) const
{
  const auto &fe          = dh.get_fe();
  const auto &immersed_fe = this->get_dof_handler().get_fe();

  std::vector<types::global_dof_index> background_dof_indices(
    fe.n_dofs_per_cell());

  FullMatrix<double> local_coupling_matrix(fe.n_dofs_per_cell(),
                                           immersed_fe.n_dofs_per_cell());

  FullMatrix<double> local_coupling_matrix_transpose(
    immersed_fe.n_dofs_per_cell(), fe.n_dofs_per_cell());

  auto particle = this->get_particles().begin();
  while (particle != this->get_particles().end())
    {
      const auto &cell = particle->get_surrounding_cell();
      const auto  dh_cell =
        typename DoFHandler<spacedim>::cell_iterator(*cell, &dh);
      dh_cell->get_dof_indices(background_dof_indices);

      const auto pic = this->get_particles().particles_in_cell(cell);
      Assert(pic.begin() == particle, ExcInternalError());

      types::global_cell_index previous_cell_id = numbers::invalid_unsigned_int;
      types::global_cell_index last_cell_id     = numbers::invalid_unsigned_int;
      local_coupling_matrix                     = 0;
      for (const auto &p : pic)
        {
          const auto [immersed_cell_id, immersed_q, section_q] =
            this->particle_id_to_cell_and_qpoint_indices(p.get_id());
          const auto &background_p = p.get_reference_location();
          const auto  immersed_p   = this->get_quadrature().point(immersed_q);
          const auto &JxW          = p.get_properties()[0];
          last_cell_id             = immersed_cell_id;
          if (immersed_cell_id != previous_cell_id &&
              previous_cell_id != numbers::invalid_unsigned_int)
            {
              // Distribute the matrix to the previous dofs
              const auto &immersed_dof_indices =
                this->get_dof_indices(previous_cell_id);
              constraints.distribute_local_to_global(local_coupling_matrix,
                                                     background_dof_indices,
                                                     coupling_constraints,
                                                     immersed_dof_indices,
                                                     coupling_matrix);
              local_coupling_matrix = 0;
              previous_cell_id      = immersed_cell_id;
            }

          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            {
              const auto comp_i = fe.system_to_component_index(i).first;
              if (comp_i < n_components)
                {
                  const auto v_i_comp_i = fe.shape_value(i, background_p);

                  for (unsigned int j = 0; j < immersed_fe.n_dofs_per_cell();
                       ++j)
                    {
                      const auto comp_j =
                        immersed_fe.system_to_component_index(j).first;

                      const auto phi_comp_j_comp_i =
                        this->get_reference_cross_section().shape_value(
                          comp_j, section_q, comp_i);

                      const auto w_j_comp_j =
                        immersed_fe.shape_value(j, immersed_p);

                      local_coupling_matrix(i, j) +=
                        v_i_comp_i * phi_comp_j_comp_i * w_j_comp_j * JxW;
                    }
                }
            }
        }
      const auto &immersed_dof_indices = this->get_dof_indices(last_cell_id);
      constraints.distribute_local_to_global(local_coupling_matrix,
                                             background_dof_indices,
                                             coupling_constraints,
                                             immersed_dof_indices,
                                             coupling_matrix);
      //  transpose
      local_coupling_matrix_transpose.copy_transposed(local_coupling_matrix);
      coupling_constraints.distribute_local_to_global(
        local_coupling_matrix_transpose,
        immersed_dof_indices,
        constraints,
        background_dof_indices,
        coupling_matrix_transpose);
      particle = pic.end();
    }
  coupling_matrix.compress(VectorOperation::add);
  coupling_matrix_transpose.compress(VectorOperation::add);
}


template <int reduced_dim, int dim, int spacedim, int n_components>
template <typename VectorType>
inline void
ReducedCoupling<reduced_dim, dim, spacedim, n_components>::assemble_reduced_rhs(
  VectorType &reduced_rhs) const
{
  const auto &immersed_fe = this->get_dof_handler().get_fe();

  const auto &all_weights     = this->get_locally_owned_weights();
  const auto &reduced_qpoints = this->get_locally_owned_reduced_qpoints();

  const auto &locally_owned_cell_ids = this->get_triangulation()
                                         .global_active_cell_index_partitioner()
                                         .lock()
                                         ->locally_owned_range();

  Vector<double> cell_rhs(immersed_fe.n_dofs_per_cell());
  Vector<double> reduced_rhs_value(immersed_fe.n_components());

  unsigned int reduced_q = 0;
  unsigned int all_q     = 0;
  for (const auto &cell_id : locally_owned_cell_ids)
    {
      cell_rhs = 0;
      for (unsigned int q = 0; q < this->get_quadrature().size(); ++q)
        {
          this->coupling_rhs->vector_value(reduced_qpoints[reduced_q++],
                                           reduced_rhs_value);
          for (unsigned int qs = 0;
               qs < this->get_reference_cross_section().n_quadrature_points();
               ++qs)
            {
              for (unsigned int i = 0; i < immersed_fe.n_dofs_per_cell(); ++i)
                {
                  const auto comp_i =
                    immersed_fe.system_to_component_index(i).first;

                  const auto w_i =
                    immersed_fe.shape_value(i, this->get_quadrature().point(q));

                  for (unsigned int comp_j = 0; comp_j < n_components; ++comp_j)
                    {
                      const auto phi_i_comp_j =
                        this->get_reference_cross_section().shape_value(comp_i,
                                                                        qs,
                                                                        comp_j);

                      cell_rhs(i) += reduced_rhs_value(comp_j) * phi_i_comp_j *
                                     phi_i_comp_j * w_i * all_weights[all_q][0];
                    }
                }
              all_q++;
            }
        }
      const auto &immersed_dof_indices = this->get_dof_indices(cell_id);
      coupling_constraints.distribute_local_to_global(cell_rhs,
                                                      immersed_dof_indices,
                                                      reduced_rhs);
    }
  reduced_rhs.compress(VectorOperation::add);
}

#endif