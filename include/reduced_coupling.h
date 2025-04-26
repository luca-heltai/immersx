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

#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "particle_coupling.h"
#include "tensor_product_space.h"

using namespace dealii;

/**
 * @class ReducedCouplingParameters
 * @brief Parameter structure for configuring ReducedCoupling objects.
 *
 * This structure holds all parameters required to set up a ReducedCoupling
 * object, including parameters for the tensor product space, particle coupling,
 * grid name, pre-refinement steps, and right-hand side expressions for the
 * coupling.
 *
 * @tparam reduced_dim The reduced dimension of the problem.
 * @tparam dim The dimension of the domain we are approximating.
 * @tparam spacedim The space dimension (default: dim).
 * @tparam n_components Number of components (default: 1).
 */
template <int reduced_dim, int dim, int spacedim = dim, int n_components = 1>
struct ReducedCouplingParameters : public ParameterAcceptor
{
  /**
   * @brief Constructor that registers parameters with the ParameterAcceptor.
   */
  ReducedCouplingParameters();

  /**
   * @brief Parameters for the tensor product space.
   */
  TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components>
    tensor_product_space_parameters;

  /**
   * @brief Parameters for the particle coupling.
   */
  ParticleCouplingParameters<spacedim> particle_coupling_parameters;

  /**
   * @brief Name of the grid to read from a file.
   */
  std::string reduced_grid_name = "";

  /**
   * @brief Number of pre-refinements to apply to the grid before distribution.
   */
  unsigned int pre_refinement = 0;

  /**
   * @brief Right hand side expressions for the reduced coupling.
   */
  std::vector<std::string> coupling_rhs_expressions = {"0"};
};

/**
 * @class ReducedCoupling
 * @brief Combines tensor product space and particle coupling for reduced Lagrange multipliers.
 *
 * This class inherits from TensorProductSpace and ParticleCoupling, providing
 * methods to initialize, assemble coupling matrices, and handle constraints for
 * reduced coupling problems in the context of immersed or embedded finite
 * element methods.
 *
 * @tparam reduced_dim The reduced dimension of the problem.
 * @tparam dim The dimension of the background domain.
 * @tparam spacedim The space dimension (default: dim).
 * @tparam n_components Number of components (default: 1).
 */
template <int reduced_dim, int dim, int spacedim = dim, int n_components = 1>
struct ReducedCoupling
  : public TensorProductSpace<reduced_dim, dim, spacedim, n_components>,
    public ParticleCoupling<spacedim>
{
  /**
   * @brief Constructor that initializes the ReducedCoupling object with background triangulation and parameters.
   * @param background_tria The background domain triangulation.
   * @param par The parameters for reduced coupling.
   */
  ReducedCoupling(
    const parallel::TriangulationBase<spacedim> &background_tria,
    const ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components>
      &par);

  /**
   * @brief Initialize the tensor product space and particle coupling.
   * @param mapping The mapping to use (default: StaticMappingQ1).
   */
  void
  initialize(
    const Mapping<spacedim> &mapping = StaticMappingQ1<spacedim>::mapping);

  /**
   * @brief Assemble the sparsity pattern for the coupling matrix.
   * @param dsp The dynamic sparsity pattern to fill.
   * @param dh The DoFHandler for the background domain.
   * @param constraints The affine constraints to apply.
   */
  void
  assemble_coupling_sparsity(
    DynamicSparsityPattern          &dsp,
    const DoFHandler<spacedim>      &dh,
    const AffineConstraints<double> &constraints) const;

  /**
   * @brief Assemble the coupling matrix between background and reduced spaces.
   * @tparam MatrixType The matrix type (e.g., SparseMatrix<double>).
   * @param coupling_matrix The matrix to assemble.
   * @param dh The DoFHandler for the background domain.
   * @param constraints The affine constraints to apply.
   */
  template <typename MatrixType>
  void
  assemble_coupling_matrix(MatrixType                      &coupling_matrix,
                           const DoFHandler<spacedim>      &dh,
                           const AffineConstraints<double> &constraints) const;

  /**
   * @brief Assemble the right-hand side vector for the reduced space.
   * @tparam VectorType The vector type (e.g., Vector<double>).
   * @param reduced_rhs The right-hand side vector to assemble.
   */
  template <typename VectorType>
  void
  assemble_reduced_rhs(VectorType &reduced_rhs) const;

  /**
   * @brief Get the affine constraints associated with the coupling.
   * @return The affine constraints.
   */
  const AffineConstraints<double> &
  get_coupling_constraints() const;

private:
  /**
   * @brief The MPI communicator used for parallel operations.
   */
  const MPI_Comm mpi_communicator;

  /**
   * @brief Reference to the parameters used for this coupling.
   */
  const ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components>
    &par;

  /**
   * @brief The triangulation of the background domain.
   */
  SmartPointer<const parallel::TriangulationBase<spacedim>> background_tria;

  /**
   * @brief Affine constraints for the coupling.
   */
  AffineConstraints<double> coupling_constraints;

  /**
   * @brief The right-hand side function for the coupling.
   */
  std::unique_ptr<FunctionParser<spacedim>> coupling_rhs;
};


// Template specializations

template <int reduced_dim, int dim, int spacedim, int n_components>
template <typename MatrixType>
inline void
ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
  assemble_coupling_matrix(MatrixType                      &coupling_matrix,
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

          // FIXME: this is not correct, we need to use the local
          // immersed_cell_id index.
          const auto global_reduced_qpoint_index = 0;
          // immersed_cell_id * this->get_quadrature().size() + immersed_q;

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

                      const auto phi_comp_j_comp_i = this->weight_shape_value(
                        comp_j, global_reduced_qpoint_index, section_q, comp_i);

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
      particle = pic.end();
    }
  coupling_matrix.compress(VectorOperation::add);
}


template <int reduced_dim, int dim, int spacedim, int n_components>
template <typename VectorType>
inline void
ReducedCoupling<reduced_dim, dim, spacedim, n_components>::assemble_reduced_rhs(
  VectorType &reduced_rhs) const
{
  VectorTools::create_right_hand_side(this->get_dof_handler(),
                                      this->get_quadrature(),
                                      *coupling_rhs,
                                      reduced_rhs,
                                      coupling_constraints);

  reduced_rhs.compress(VectorOperation::add);
}

#endif