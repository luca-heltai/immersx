// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
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
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <immersx/core/symbolic_field_evaluator.h>
#include <immersx/coupling/immersed_repartitioner.h>
#include <immersx/coupling/particle_coupling.h>
#include <immersx/coupling/tensor_product_space.h>
#include <immersx/io/utils.h>

#include <fstream>
#include <functional>
#include <map>
#include <optional>

namespace ImmersX
{
  using namespace dealii;

  template <int reduced_dim, int spacedim>
  struct ImmersedRepartitionerStorage
  {
    ImmersedRepartitionerStorage(
      const parallel::TriangulationBase<spacedim> &tria)
      : value(tria)
    {}
    ImmersedRepartitioner<reduced_dim, spacedim> value;
  };

  template <int spacedim>
  struct ImmersedRepartitionerStorage<0, spacedim>
  {
    ImmersedRepartitionerStorage(const parallel::TriangulationBase<spacedim> &)
    {}
  };

  /**
   * @class ReducedCouplingParameters
   * @brief Parameters for configuring tensor-product coupling objects.
   *
   * This structure holds all parameters required to set up a `ReducedCoupling`
   * object, including parameters for the tensor-product space, particle
   * coupling, grid name, pre-refinement steps, and right-hand side expressions
   * for the coupling.
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
     * Refinement parameters for the tensor product space.
     */
    RefinementParameters<reduced_dim> refinement_parameters;

    /**
     * @brief Right hand side expressions for tensor-product coupling.
     */
    std::vector<std::string> coupling_rhs_expressions = {"0"};
  };

  /**
   * @class ReducedCoupling
   * @brief Combines tensor-product space and particle coupling for
   *        lower-dimensional Lagrange multipliers.
   *
   * This class inherits from TensorProductSpace and ParticleCoupling, providing
   * methods to initialize, assemble coupling matrices, and handle constraints
   * for reduced coupling problems in the context of immersed or embedded finite
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
     * @param par The tensor-product coupling parameters.
     */
    ReducedCoupling(
      parallel::TriangulationBase<spacedim> &background_tria,
      const ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components>
        &par);

    /**
     * @brief Initialize the tensor product space and particle coupling.
     * @param mapping The background geometry mapping.
     */
    void
    initialize(const Mapping<spacedim> &mapping);

    /** Optional preparation of a 0D background before particle ownership.
     * The default refines the existing background. A Problem may instead
     * refine a serial preparation mesh and finalize its distributed Domain.
     */
    std::function<void(const std::vector<Point<spacedim>> &,
                       const std::vector<double> &)>
      prepare_point_cloud_background;

    /** Initialize using the background mesh's linear reference-cell mapping. */
    void
    initialize();

    /**
     * @brief Set the time used by the reduced right-hand-side function.
     */
    void
    set_time(const double time);

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
     * @brief Assemble the matrix mapping multiplier coefficients to background loads.
     *
     * The entries are B_ij = int_gamma int_D(s) v_i . (psi phi_c)_j dD ds, a
     * surface integral over the cross sections D(s) along the reduced domain
     * gamma. Point coupling instead uses the plain average
     * 1/|S| int_S v_i . phi_j over the circle S of each inclusion, which does
     * not grow with the size of the inclusion.
     *
     * @tparam MatrixType The matrix type (e.g., SparseMatrix<double>).
     * @param multiplier_load_matrix The matrix to assemble.
     * @param dh The DoFHandler for the background domain.
     * @param constraints The affine constraints to apply.
     */
    template <typename MatrixType>
    void
    assemble_multiplier_load_matrix(
      MatrixType                      &multiplier_load_matrix,
      const DoFHandler<spacedim>      &dh,
      const AffineConstraints<double> &constraints) const;

    /** Assemble the matrix and add the load from prescribed background values.
     * reduced_rhs must already be initialized on the representative DoFs.
     * Finish insert writes collectively before calling this additive assembly.
     * Its prescribed data are retained; the affine background action is
     * subtracted before the distributed vector is compressed.
     */
    template <typename MatrixType, typename VectorType>
    void
    assemble_multiplier_load_matrix(
      MatrixType                      &multiplier_load_matrix,
      const DoFHandler<spacedim>      &dh,
      const AffineConstraints<double> &constraints,
      VectorType                      &reduced_rhs) const;

    /** Compatibility spelling for assemble_multiplier_load_matrix(). */
    template <typename MatrixType>
    void
    assemble_coupling_matrix(MatrixType                      &matrix,
                             const DoFHandler<spacedim>      &dh,
                             const AffineConstraints<double> &constraints) const
    {
      assemble_multiplier_load_matrix(matrix, dh, constraints);
    }

    /** Compatibility overload retaining prescribed background values. */
    template <typename MatrixType, typename VectorType>
    void
    assemble_coupling_matrix(MatrixType                      &matrix,
                             const DoFHandler<spacedim>      &dh,
                             const AffineConstraints<double> &constraints,
                             VectorType                      &reduced_rhs) const
    {
      assemble_multiplier_load_matrix(matrix, dh, constraints, reduced_rhs);
    }

    /**
     * @brief Assemble the mass matrix of the multiplier space.
     *
     * The complete sampled Gram matrix integrates products of the lifted
     * representative basis, including products of distinct transverse modes.
     * It is independent of background coverage and remains the multiplier
     * metric when only part of the physical support is coupled.
     *
     * @tparam MatrixType The matrix type (e.g., SparseMatrix<double>).
     * @param mass_matrix The matrix to assemble.
     */
    template <typename MatrixType>
    void
    assemble_coupling_mass_matrix(MatrixType &mass_matrix) const;

    /** Assemble the sampled Gram matrix on points inside the background. */
    template <typename MatrixType>
    void
    assemble_overlap_mass_matrix(MatrixType &mass_matrix) const;

    /** Assemble the sampled Gram matrix on points outside the background. */
    template <typename MatrixType>
    void
    assemble_outside_mass_matrix(MatrixType &mass_matrix) const;

    /** Source-owned membership, in get_locally_owned_qpoints() order. */
    const std::vector<bool> &
    source_point_membership() const
    {
      return located_source_points;
    }

    /** Prescribed source coefficients, indexed by cell/entity, quadrature
     * point, and selected mode. These values are not projected onto overlap.
     */
    auto
    source_modal_data_values() const
    {
      return modal_data_values();
    }

    /**
     * @brief Assemble the right-hand side vector for the reduced space.
     *
     * Reconstructs the prescribed field from its modal coefficient functions
     * and pairs it with the lifted multiplier basis at located physical
     * samples. No contribution is added at points outside the background.
     *
     * @tparam VectorType The vector type (e.g., Vector<double>).
     * @param reduced_rhs The right-hand side vector to assemble.
     */
    template <typename VectorType>
    void
    assemble_reduced_rhs(VectorType &reduced_rhs) const;

    /** Integrate prescribed modal data on the complete nominal support. */
    template <typename VectorType>
    void
    assemble_prescribed_rhs(VectorType &rhs) const;

    /**
     * @brief Get the affine constraints associated with the coupling.
     * @return The affine constraints.
     */
    const AffineConstraints<double> &
    get_coupling_constraints() const;

  private:
    /** Visit each physical sample with its representative indices and basis. */
    template <typename Callback>
    void
    for_each_source_sample(const Callback &callback) const;

    template <typename MatrixType>
    void
    assemble_sampled_mass_matrix(MatrixType               &mass_matrix,
                                 const std::optional<bool> inside) const;

    std::map<types::global_cell_index, std::vector<std::vector<double>>>
    modal_data_values() const;

    template <typename VectorType>
    void
    assemble_sampled_rhs(VectorType &rhs, const bool restrict_to_overlap) const;

    std::vector<bool> located_source_points;

    template <typename MatrixType, typename VectorType>
    void
    assemble_multiplier_load_matrix_impl(
      MatrixType                      &multiplier_load_matrix,
      const DoFHandler<spacedim>      &dh,
      const AffineConstraints<double> &constraints,
      VectorType                      *reduced_rhs) const;

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
    ObserverPointer<parallel::TriangulationBase<spacedim>> background_tria;

    /** Whether zero-dimensional bulk pre/post refinement was applied. */
    bool zero_dimensional_background_refined = false;

    /**
     * @brief Affine constraints for the coupling.
     */
    AffineConstraints<double> coupling_constraints;

    /**
     * @brief The right-hand side function for the coupling.
     */
    std::unique_ptr<FunctionParser<spacedim>> coupling_rhs;
    std::unique_ptr<SymbolicFieldEvaluator>   symbolic_coupling_rhs;
    double                                    rhs_time = 0.;

    /**
     * An ImmersedRepartitioner object that handles the repartitioning of the
     * triangulation.
     */
    ImmersedRepartitionerStorage<reduced_dim, spacedim> immersed_partitioner;
  };


// Template specializations
#ifndef DOXYGEN
  template <int reduced_dim, int dim, int spacedim, int n_components>
  template <typename MatrixType>
  inline void
  ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
    assemble_multiplier_load_matrix(
      MatrixType                      &matrix,
      const DoFHandler<spacedim>      &dh,
      const AffineConstraints<double> &constraints) const
  {
    assemble_multiplier_load_matrix_impl(
      matrix, dh, constraints, static_cast<Vector<double> *>(nullptr));
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  template <typename MatrixType, typename VectorType>
  inline void
  ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
    assemble_multiplier_load_matrix(
      MatrixType                      &matrix,
      const DoFHandler<spacedim>      &dh,
      const AffineConstraints<double> &constraints,
      VectorType                      &rhs) const
  {
    assemble_multiplier_load_matrix_impl(matrix, dh, constraints, &rhs);
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  template <typename MatrixType, typename VectorType>
  inline void
  ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
    assemble_multiplier_load_matrix_impl(
      MatrixType                      &multiplier_load_matrix,
      const DoFHandler<spacedim>      &dh,
      const AffineConstraints<double> &constraints,
      VectorType                      *reduced_rhs) const
  {
    const auto add_affine_load =
      [&](const FullMatrix<double>                   &local,
          const std::vector<types::global_dof_index> &background_indices,
          const std::vector<types::global_dof_index> &reduced_indices) {
        if (reduced_rhs == nullptr)
          return;
        Vector<double> local_rhs(reduced_indices.size());
        for (unsigned int i = 0; i < background_indices.size(); ++i)
          {
            const double prescribed =
              constraints.get_inhomogeneity(background_indices[i]);
            if (prescribed != 0.)
              for (unsigned int j = 0; j < reduced_indices.size(); ++j)
                local_rhs[j] -= local(i, j) * prescribed;
          }
        coupling_constraints.distribute_local_to_global(local_rhs,
                                                        reduced_indices,
                                                        *reduced_rhs);
      };
    if constexpr (reduced_dim == 0)
      {
        const auto                          &fe = dh.get_fe();
        std::vector<types::global_dof_index> background_dof_indices(
          fe.n_dofs_per_cell());
        for (const auto &particle : this->get_particles())
          {
            const auto &cell = particle.get_surrounding_cell();
            const typename DoFHandler<spacedim>::cell_iterator dh_cell(*cell,
                                                                       &dh);
            dh_cell->get_dof_indices(background_dof_indices);
            const auto [entity_id, unused_q, section_q] =
              this->particle_id_to_representative_indices(particle.get_id());
            (void)unused_q;
            const auto &entity_dofs =
              this->get_representative_dof_indices(entity_id);
            FullMatrix<double> local(fe.n_dofs_per_cell(), entity_dofs.size());
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              {
                const auto comp_i = fe.system_to_component_index(i).first;
                if (comp_i >= n_components)
                  continue;
                for (unsigned int j = 0; j < entity_dofs.size(); ++j)
                  {
                    const auto basis_j = j;
                    local(i, j) =
                      fe.shape_value(i, particle.get_reference_location()) *
                      particle
                        .get_properties()[1 + basis_j * n_components + comp_i] *
                      particle.get_properties()[0];
                  }
              }
            add_affine_load(local, background_dof_indices, entity_dofs);
            constraints.distribute_local_to_global(local,
                                                   background_dof_indices,
                                                   coupling_constraints,
                                                   entity_dofs,
                                                   multiplier_load_matrix);
          }
      }
    else
      {
        const auto &fe          = dh.get_fe();
        const auto &immersed_fe = this->get_dof_handler().get_fe();

        std::vector<types::global_dof_index> background_dof_indices(
          fe.n_dofs_per_cell());

        auto particle = this->get_particles().begin();
        while (particle != this->get_particles().end())
          {
            const auto &cell = particle->get_surrounding_cell();
            const auto  dh_cell =
              typename DoFHandler<spacedim>::cell_iterator(*cell, &dh);
            dh_cell->get_dof_indices(background_dof_indices);

            const auto pic = this->get_particles().particles_in_cell(cell);
            Assert(pic.begin() == particle, ExcInternalError());
            std::map<types::global_cell_index, FullMatrix<double>>
              local_coupling_matrices;

            for (const auto &p : pic)
              {
                const auto [immersed_cell_id, immersed_q, section_q] =
                  this->particle_id_to_representative_indices(p.get_id());

                auto [matrix_it, inserted] =
                  local_coupling_matrices.try_emplace(
                    immersed_cell_id,
                    fe.n_dofs_per_cell(),
                    immersed_fe.n_dofs_per_cell());
                if (inserted)
                  matrix_it->second = 0;
                auto &local_coupling_matrix = matrix_it->second;

                const auto &background_p = p.get_reference_location();
                const auto  immersed_p =
                  this->get_quadrature().point(immersed_q);
                const double JxW = p.get_properties()[0];
                for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                  {
                    const auto comp_i = fe.system_to_component_index(i).first;
                    if (comp_i < n_components)
                      {
                        const auto v_i_comp_i = fe.shape_value(i, background_p);

                        for (unsigned int j = 0;
                             j < immersed_fe.n_dofs_per_cell();
                             ++j)
                          {
                            const auto comp_j =
                              immersed_fe.system_to_component_index(j).first;

                            const auto phi_comp_j_comp_i =
                              p.get_properties()[1 + comp_j * n_components +
                                                 comp_i];

                            const auto w_j_comp_j =
                              immersed_fe.shape_value(j, immersed_p);

                            local_coupling_matrix(i, j) +=
                              v_i_comp_i * phi_comp_j_comp_i * w_j_comp_j * JxW;
                          }
                      }
                  }
              }
            for (const auto &[immersed_cell_id, local_coupling_matrix] :
                 local_coupling_matrices)
              {
                const auto &immersed_dof_indices =
                  this->get_dof_indices(immersed_cell_id);
                add_affine_load(local_coupling_matrix,
                                background_dof_indices,
                                immersed_dof_indices);
                constraints.distribute_local_to_global(local_coupling_matrix,
                                                       background_dof_indices,
                                                       coupling_constraints,
                                                       immersed_dof_indices,
                                                       multiplier_load_matrix);
              }
            particle = pic.end();
          }
      }
    multiplier_load_matrix.compress(VectorOperation::add);
    if (reduced_rhs != nullptr)
      reduced_rhs->compress(VectorOperation::add);
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  template <typename Callback>
  void
  ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
    for_each_source_sample(const Callback &callback) const
  {
    const auto &points  = this->get_locally_owned_qpoints();
    const auto &weights = this->get_locally_owned_weights();
    const auto &modes   = this->get_locally_owned_mode_values();
    const auto  owned   = this->locally_owned_qpoints();
    AssertDimension(owned.n_elements(), points.size());
    AssertDimension(located_source_points.size(), points.size());
    for (unsigned int q = 0; q < points.size(); ++q)
      {
        const auto [entity, representative_q, section_q] =
          this->particle_id_to_representative_indices(
            owned.nth_index_in_set(q));
        (void)section_q;
        std::vector<types::global_dof_index> indices;
        std::vector<Tensor<1, n_components>> basis;
        if constexpr (reduced_dim == 0)
          {
            indices = this->get_representative_dof_indices(entity);
            basis.resize(indices.size());
            for (unsigned int i = 0; i < indices.size(); ++i)
              for (unsigned int d = 0; d < n_components; ++d)
                basis[i][d] = modes[q][i * n_components + d];
          }
        else
          {
            indices = this->get_dof_indices(entity);
            basis.resize(indices.size());
            const auto &fe   = this->get_dof_handler().get_fe();
            const auto &unit = this->get_quadrature().point(representative_q);
            for (unsigned int i = 0; i < indices.size(); ++i)
              {
                const auto component = fe.system_to_component_index(i).first;
                for (unsigned int d = 0; d < n_components; ++d)
                  basis[i][d] = fe.shape_value(i, unit) *
                                modes[q][component * n_components + d];
              }
          }
        callback(q, entity, representative_q, indices, basis, weights[q][0]);
      }
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  template <typename MatrixType>
  void
  ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
    assemble_sampled_mass_matrix(MatrixType               &matrix,
                                 const std::optional<bool> inside) const
  {
    AssertDimension(matrix.m(), this->n_representative_dofs());
    AssertDimension(matrix.n(), this->n_representative_dofs());
    matrix = 0.;
    for_each_source_sample([&](const auto q,
                               const auto,
                               const auto,
                               const auto  &indices,
                               const auto  &basis,
                               const double weight) {
      if (inside.has_value() && located_source_points[q] != *inside)
        return;
      FullMatrix<double> local(indices.size(), indices.size());
      for (unsigned int i = 0; i < indices.size(); ++i)
        for (unsigned int j = 0; j < indices.size(); ++j)
          local(i, j) = weight * (basis[i] * basis[j]);
      coupling_constraints.distribute_local_to_global(local, indices, matrix);
    });
    matrix.compress(VectorOperation::add);
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  template <typename MatrixType>
  void
  ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
    assemble_coupling_mass_matrix(MatrixType &matrix) const
  {
    assemble_sampled_mass_matrix(matrix, std::nullopt);
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  template <typename MatrixType>
  void
  ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
    assemble_overlap_mass_matrix(MatrixType &matrix) const
  {
    assemble_sampled_mass_matrix(matrix, true);
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  template <typename MatrixType>
  void
  ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
    assemble_outside_mass_matrix(MatrixType &matrix) const
  {
    assemble_sampled_mass_matrix(matrix, false);
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  template <typename VectorType>
  void
  ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
    assemble_sampled_rhs(VectorType &rhs, const bool restrict_to_overlap) const
  {
    const auto  data  = modal_data_values();
    const auto &modes = this->get_locally_owned_mode_values();
    rhs               = 0.;
    for_each_source_sample([&](const auto   q,
                               const auto   entity,
                               const auto   representative_q,
                               const auto  &indices,
                               const auto  &basis,
                               const double weight) {
      if (restrict_to_overlap && !located_source_points[q])
        return;
      const auto &coefficients = data.at(entity).at(representative_q);
      Tensor<1, n_components> imposed;
      for (unsigned int c = 0; c < coefficients.size(); ++c)
        for (unsigned int d = 0; d < n_components; ++d)
          imposed[d] += coefficients[c] * modes[q][c * n_components + d];
      Vector<double> local(indices.size());
      for (unsigned int i = 0; i < indices.size(); ++i)
        local[i] = weight * (basis[i] * imposed);
      coupling_constraints.distribute_local_to_global(local, indices, rhs);
    });
    rhs.compress(VectorOperation::add);
  }
  template <int reduced_dim, int dim, int spacedim, int n_components>
  template <typename VectorType>
  void
  ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
    assemble_reduced_rhs(VectorType &rhs) const
  {
    assemble_sampled_rhs(rhs, true);
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  template <typename VectorType>
  void
  ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
    assemble_prescribed_rhs(VectorType &rhs) const
  {
    assemble_sampled_rhs(rhs, false);
  }

#endif

} // namespace ImmersX

#endif
