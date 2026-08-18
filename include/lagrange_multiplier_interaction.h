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
// either version 3.0 of the License or (at your option) any later version. The
// full text of the license can be found in the LICENSE.md file at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#ifndef immersx_lagrange_multiplier_interaction_h
#define immersx_lagrange_multiplier_interaction_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "linear_algebra.h"
#include "particle_coupling.h"
#include "representation.h"

/**
 * Representation-driven continuity interaction with a Lagrange multiplier.
 *
 * The matrix orientation is deliberately chosen so that `coupling_matrix()`
 * has background rows and multiplier columns:
 *
 * @code
 *   [ A_bulk       0          C ] [u]       [f]
 *   [ 0            A_surface  -M^T] [w]  =   [g]
 *   [ C^T         -M           0 ] [lambda] [0]
 * @endcode
 *
 * Thus C is assembled as the trace pairing between the background field and
 * the embedded representation, while M is the embedded mass matrix. The
 * interaction owns both matrices and all point-location/assembly state. It
 * does not know about PoissonSolver or any other concrete PDE class.
 *
 * The class is intentionally a two-representation building block. Adding a
 * third problem later means constructing another representation and another
 * interaction; neither existing problem nor representation stores a unique
 * coupling partner.
 */
template <typename BackgroundRepresentation, typename EmbeddedRepresentation>
class LagrangeMultiplierInteraction
{
public:
  static constexpr unsigned int spacedim =
    BackgroundRepresentation::space_dimension;

  static_assert(spacedim == EmbeddedRepresentation::space_dimension,
                "Representations must live in the same physical space.");
  static_assert(BackgroundRepresentation::dimension >=
                  EmbeddedRepresentation::dimension,
                "The embedded representation cannot have higher dimension.");

  using MatrixType = ImmersXLA::MPI::SparseMatrix;
  using VectorType = ImmersXLA::MPI::Vector;
  using PointType  = RepresentationQuadraturePoint<spacedim>;

  LagrangeMultiplierInteraction(
    const BackgroundRepresentation             &background,
    const EmbeddedRepresentation               &embedded,
    const ParticleCouplingParameters<spacedim> &search_parameters)
    : background(background)
    , embedded(embedded)
    , particle_coupling(search_parameters)
  {
    AssertThrow(background.triangulation().get_mpi_communicator() ==
                  embedded.triangulation().get_mpi_communicator(),
                ExcMessage("Interaction representations must use the same "
                           "MPI communicator."));
  }

  /** Assemble point ownership, the trace matrix C, and the mass matrix M. */
  void
  assemble()
  {
    const unsigned int degree = std::max(background.finite_element().degree,
                                         embedded.finite_element().degree);
    quadrature =
      std::make_unique<dealii::QGauss<EmbeddedRepresentation::dimension>>(
        degree + 1);
    embedded_quadrature_points =
      embedded.locally_owned_quadrature_points(*quadrature);

    const unsigned int n_embedded_dofs = embedded.n_dofs_per_cell();
    const unsigned int n_properties    = 1 + 2 * n_embedded_dofs;
    particle_coupling.initialize_particle_handler(background.triangulation(),
                                                  background.mapping(),
                                                  n_properties);

    std::vector<dealii::Point<spacedim>> points;
    std::vector<std::vector<double>>     properties;
    points.reserve(embedded_quadrature_points.size());
    properties.reserve(embedded_quadrature_points.size());

    for (const auto &point : embedded_quadrature_points)
      {
        points.push_back(point.point);

        // ParticleHandler transports scalar properties. For this first direct
        // FE adapter we use them for the trace quadrature weight, represented
        // DoF indices, and basis values. The explicit range check preserves a
        // safe failure mode on machines where a global index cannot be carried
        // exactly by a double. A future TensorProductSpace transport adapter
        // should replace this packing with a typed metadata exchange.
        std::vector<double> point_properties;
        point_properties.reserve(n_properties);
        point_properties.push_back(point.weight);
        for (const auto dof : point.dof_indices)
          {
            const double encoded = static_cast<double>(dof);
            AssertThrow(static_cast<dealii::types::global_dof_index>(encoded) ==
                          dof,
                        dealii::ExcMessage(
                          "A representation DoF index cannot be transported "
                          "exactly through ParticleHandler properties."));
            point_properties.push_back(encoded);
          }
        point_properties.insert(point_properties.end(),
                                point.basis_values.begin(),
                                point.basis_values.end());
        properties.emplace_back(std::move(point_properties));
      }

    particle_coupling.insert_points(points, properties);

    assemble_coupling_matrix();
    assemble_mass_matrix();
  }

  /** Return C, with background rows and embedded multiplier columns. */
  const MatrixType &
  coupling_matrix() const
  {
    return coupling_matrix_storage;
  }

  /** Return M, with multiplier rows and embedded field columns. */
  const MatrixType &
  multiplier_mass_matrix() const
  {
    return multiplier_mass_matrix_storage;
  }

  const dealii::IndexSet &
  multiplier_locally_owned_dofs() const
  {
    return embedded.locally_owned_dofs();
  }

  const dealii::IndexSet &
  multiplier_locally_relevant_dofs() const
  {
    return embedded.locally_relevant_dofs();
  }

private:
  template <typename ParticleType>
  void
  read_particle_data(
    const ParticleType                           &particle,
    std::vector<dealii::types::global_dof_index> &embedded_dof_indices,
    std::vector<double>                          &embedded_basis_values,
    double                                       &weight) const
  {
    const auto         properties = particle.get_properties();
    const unsigned int n          = embedded.n_dofs_per_cell();
    AssertDimension(properties.size(), 1 + 2 * n);

    weight = properties[0];
    embedded_dof_indices.resize(n);
    embedded_basis_values.resize(n);
    for (unsigned int i = 0; i < n; ++i)
      embedded_dof_indices[i] =
        static_cast<dealii::types::global_dof_index>(properties[1 + i]);
    for (unsigned int i = 0; i < n; ++i)
      embedded_basis_values[i] = properties[1 + n + i];
  }

  template <typename ParticleType>
  void
  make_background_dof_indices(
    const ParticleType                           &particle,
    std::vector<dealii::types::global_dof_index> &background_dof_indices) const
  {
    const auto &background_cell = particle.get_surrounding_cell();
    const typename BackgroundRepresentation::DoFHandlerType::cell_iterator
      background_dof_cell(*background_cell, &background.dof_handler());
    background_dof_cell->get_dof_indices(background_dof_indices);
  }

  void
  assemble_coupling_matrix()
  {
    const unsigned int n_background_dofs = background.n_dofs_per_cell();
    const unsigned int n_embedded_dofs   = embedded.n_dofs_per_cell();

    dealii::DynamicSparsityPattern dsp(background.dof_handler().n_dofs(),
                                       embedded.dof_handler().n_dofs(),
                                       background.locally_relevant_dofs());

    std::vector<dealii::types::global_dof_index> background_dof_indices(
      n_background_dofs);
    std::vector<dealii::types::global_dof_index> embedded_dof_indices(
      n_embedded_dofs);
    std::vector<double> unused_basis_values;
    double              unused_weight = 0.;
    for (const auto &particle : particle_coupling.get_particles())
      {
        make_background_dof_indices(particle, background_dof_indices);
        read_particle_data(particle,
                           embedded_dof_indices,
                           unused_basis_values,
                           unused_weight);
        background.constraints().add_entries_local_to_global(
          background_dof_indices, embedded_dof_indices, dsp);
      }

    dealii::SparsityTools::distribute_sparsity_pattern(
      dsp,
      background.locally_owned_dofs(),
      background.triangulation().get_mpi_communicator(),
      background.locally_relevant_dofs());
    coupling_matrix_storage.reinit(
      background.locally_owned_dofs(),
      embedded.locally_owned_dofs(),
      dsp,
      background.triangulation().get_mpi_communicator());

    dealii::FullMatrix<double> local_matrix(n_background_dofs, n_embedded_dofs);
    std::vector<double>        embedded_basis_values;
    double                     weight = 0.;
    for (const auto &particle : particle_coupling.get_particles())
      {
        make_background_dof_indices(particle, background_dof_indices);
        read_particle_data(particle,
                           embedded_dof_indices,
                           embedded_basis_values,
                           weight);
        local_matrix = 0.;

        const auto &background_fe   = background.finite_element();
        const auto &reference_point = particle.get_reference_location();
        for (unsigned int i = 0; i < n_background_dofs; ++i)
          for (unsigned int j = 0; j < n_embedded_dofs; ++j)
            local_matrix(i, j) = background_fe.shape_value(i, reference_point) *
                                 embedded_basis_values[j] * weight;

        // The multiplier columns are unconstrained algebraic variables. The
        // background constraints still have to eliminate constrained trace
        // rows exactly as they do for the independent Poisson operator.
        background.constraints().distribute_local_to_global(
          local_matrix,
          background_dof_indices,
          embedded_dof_indices,
          coupling_matrix_storage);
      }
    coupling_matrix_storage.compress(dealii::VectorOperation::add);
  }

  void
  assemble_mass_matrix()
  {
    dealii::DynamicSparsityPattern dsp(embedded.dof_handler().n_dofs(),
                                       embedded.dof_handler().n_dofs(),
                                       embedded.locally_relevant_dofs());
    std::vector<dealii::types::global_dof_index> dof_indices(
      embedded.n_dofs_per_cell());
    for (const auto &cell : embedded.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(dof_indices);
          for (const auto row : dof_indices)
            for (const auto col : dof_indices)
              dsp.add(row, col);
        }

    dealii::SparsityTools::distribute_sparsity_pattern(
      dsp,
      embedded.locally_owned_dofs(),
      embedded.triangulation().get_mpi_communicator(),
      embedded.locally_relevant_dofs());
    multiplier_mass_matrix_storage.reinit(
      embedded.locally_owned_dofs(),
      embedded.locally_owned_dofs(),
      dsp,
      embedded.triangulation().get_mpi_communicator());

    // M maps the embedded field into the multiplier equation, so its columns
    // must represent the constrained embedded algebraic space. Its rows are
    // multiplier DoFs and are intentionally unconstrained.
    dealii::AffineConstraints<double> no_row_constraints;
    no_row_constraints.close();
    dealii::FullMatrix<double> local_mass(embedded.n_dofs_per_cell(),
                                          embedded.n_dofs_per_cell());
    for (const auto &point : embedded_quadrature_points)
      {
        local_mass = 0.;
        for (unsigned int i = 0; i < point.dof_indices.size(); ++i)
          for (unsigned int j = 0; j < point.dof_indices.size(); ++j)
            local_mass(i, j) =
              point.weight * point.basis_values[i] * point.basis_values[j];

        no_row_constraints.distribute_local_to_global(
          local_mass,
          point.dof_indices,
          embedded.constraints(),
          point.dof_indices,
          multiplier_mass_matrix_storage);
      }
    multiplier_mass_matrix_storage.compress(dealii::VectorOperation::add);
  }

  const BackgroundRepresentation &background;
  const EmbeddedRepresentation   &embedded;
  ParticleCoupling<spacedim>      particle_coupling;

  std::unique_ptr<dealii::Quadrature<EmbeddedRepresentation::dimension>>
                         quadrature;
  std::vector<PointType> embedded_quadrature_points;

  MatrixType coupling_matrix_storage;
  MatrixType multiplier_mass_matrix_storage;
};

#endif // immersx_lagrange_multiplier_interaction_h
