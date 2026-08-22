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

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/constraint_contributor.h>
#include <immersx/core/constraint_equation.h>
#include <immersx/core/representation.h>
#include <immersx/coupling/particle_coupling.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <type_traits>
#include <vector>

namespace ImmersX
{
  /**
   * Representation-driven continuity interaction with a Lagrange multiplier.
   *
   * The matrix orientation is deliberately chosen so that `coupling_matrix()`
   * has first-representation rows and second-representation multiplier columns:
   *
   * @code
   *   [ A_0          0          C ] [u_0]     [f_0]
   *   [ 0            A_1        -M^T] [u_1] = [f_1]
   *   [ C^T         -M           0 ] [lambda] [0]
   * @endcode
   *
   * Thus C is assembled as the trace pairing between the two representations,
   * while M is the second-representation mass matrix. The first representation
   * supplies the mesh used internally by the current particle search; this is
   * an assembly role, not a privileged Problem endpoint. The interaction owns
   * both matrices and all point-location/assembly state. It does not know about
   * PoissonSolver or any other concrete PDE class. Both representation
   * arguments are non-owning and must outlive the interaction.
   *
   * The class is intentionally a two-representation building block. Adding a
   * third problem later means constructing another representation and another
   * interaction; neither existing problem nor representation stores a unique
   * coupling partner.
   */
  template <typename FirstRepresentation, typename SecondRepresentation>
  class LagrangeMultiplierInteraction
  {
  public:
    static constexpr unsigned int spacedim =
      FirstRepresentation::ambient_dimension;

    static_assert(RepresentationConcept<FirstRepresentation>::value,
                  "The first interaction endpoint does not satisfy the "
                  "representation contract.");
    static_assert(RepresentationConcept<SecondRepresentation>::value,
                  "The second interaction endpoint does not satisfy the "
                  "representation contract.");

    // This continuity interaction assembles scalar pairings into scalar sparse
    // matrices. Vector-valued observables need a separate concrete interaction
    // (for example velocity continuity or traction), rather than runtime
    // branching through this scalar path.
    static_assert(
      std::is_same<typename FirstRepresentation::value_type, double>::value,
      "LagrangeMultiplierInteraction only accepts scalar first representations; "
      "use a vector-specific interaction for vector observables.");
    static_assert(
      std::is_same<typename SecondRepresentation::value_type, double>::value,
      "LagrangeMultiplierInteraction only accepts scalar second representations; "
      "use a vector-specific interaction for vector observables.");

    static_assert(spacedim == SecondRepresentation::ambient_dimension,
                  "Representations must live in the same physical space.");
    static_assert(FirstRepresentation::support_dimension >=
                    SecondRepresentation::support_dimension,
                  "The second representation cannot have higher support "
                  "dimension than the first search representation.");

    using MatrixType = ImmersXLA::MPI::SparseMatrix;
    using VectorType = ImmersXLA::MPI::Vector;
    using PointType  = typename SecondRepresentation::QuadraturePoint;

    LagrangeMultiplierInteraction(
      const FirstRepresentation                  &first,
      const SecondRepresentation                 &second,
      const ParticleCouplingParameters<spacedim> &search_parameters)
      : first(first)
      , second(second)
      , constraint_equation_storage(second.locally_owned_dofs(),
                                    second.mpi_communicator())
      , particle_coupling(search_parameters)
    {
      AssertThrow(first.triangulation().get_mpi_communicator() ==
                    second.triangulation().get_mpi_communicator(),
                  ExcMessage("Interaction representations must use the same "
                             "MPI communicator."));
    }

    /** Assemble point ownership, the trace matrix C, and the mass matrix M. */
    void
    assemble()
    {
      const unsigned int degree =
        std::max(first.finite_element().degree, second.finite_element().degree);
      quadrature = std::make_unique<
        dealii::QGauss<SecondRepresentation::support_dimension>>(degree + 1);
      second_quadrature_points =
        second.locally_owned_quadrature_points(*quadrature);

      const unsigned int n_second_dofs = second.n_dofs_per_cell();
      const unsigned int n_properties  = 1 + 2 * n_second_dofs;
      particle_coupling.initialize_particle_handler(first.triangulation(),
                                                    first.mapping(),
                                                    n_properties);

      std::vector<dealii::Point<spacedim>> points;
      std::vector<std::vector<double>>     properties;
      points.reserve(second_quadrature_points.size());
      properties.reserve(second_quadrature_points.size());

      for (const auto &point : second_quadrature_points)
        {
          points.push_back(point.point);

          // ParticleHandler transports scalar properties. For this first direct
          // FE adapter we use them for the trace quadrature weight, represented
          // DoF indices, and basis values. The explicit range check preserves a
          // safe failure mode on machines where a global index cannot be
          // carried exactly by a double. A future TensorProductSpace transport
          // adapter should replace this packing with a typed metadata exchange.
          std::vector<double> point_properties;
          point_properties.reserve(n_properties);
          point_properties.push_back(point.weight);
          for (const auto dof : point.dof_indices)
            {
              const double encoded = static_cast<double>(dof);
              AssertThrow(static_cast<dealii::types::global_dof_index>(
                            encoded) == dof,
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

      constraint_equation_storage.clear_contributions();
      constraint_equation_storage.clear_rhs();
      constraint_equation_storage.add_contribution(
        0,
        coupling_matrix_storage,
        ConstraintContributionOrientation::transpose);
      constraint_equation_storage.add_contribution(
        1,
        multiplier_mass_matrix_storage,
        ConstraintContributionOrientation::direct,
        -1.);
      constraint_equation_storage.set_multiplier_metric(
        multiplier_mass_matrix_storage);
    }

    /** Return C, with first rows and second-representation multiplier columns.
     */
    const MatrixType &
    coupling_matrix() const
    {
      return coupling_matrix_storage;
    }

    /** Return M, with multiplier rows and second-representation field columns.
     */
    const MatrixType &
    multiplier_mass_matrix() const
    {
      return multiplier_mass_matrix_storage;
    }

    const dealii::IndexSet &
    multiplier_locally_owned_dofs() const
    {
      return second.locally_owned_dofs();
    }

    const dealii::IndexSet &
    multiplier_locally_relevant_dofs() const
    {
      return second.locally_relevant_dofs();
    }

    /** Return the generic two-representation constraint equation. */
    const ConstraintEquation &
    constraint_equation() const
    {
      return constraint_equation_storage;
    }

    /** Return the first representation without transferring ownership. */
    const FirstRepresentation &
    first_representation() const
    {
      return first;
    }

    /** Return the second representation without transferring ownership. */
    const SecondRepresentation &
    second_representation() const
    {
      return second;
    }

    /** Metadata for adapter-level dependency discovery. */
    const ImmersX::RepresentationMetadata &
    first_metadata() const
    {
      return first.metadata();
    }

    /** Metadata for adapter-level dependency discovery. */
    const ImmersX::RepresentationMetadata &
    second_metadata() const
    {
      return second.metadata();
    }

    /**
     * Assemble the multiplier-dual right hand side for prescribed data.
     *
     * The coefficients belong to the second representation.  Applying M
     * performs the physical pairing with the multiplier basis, so this is the
     * reduced analogue of assembling <R g, R mu> without ever materializing R
     * or R^T.
     */
    void
    assemble_prescribed_rhs(
      const PrescribedFieldDatum<SecondRepresentation> &datum,
      VectorType                                       &rhs) const
    {
      const auto equation = prescribed_constraint_equation(datum);
      rhs.reinit(equation.multiplier_locally_owned_dofs(),
                 second.mpi_communicator());
      rhs = equation.rhs();
    }

    /**
     * Build the one-problem version of this interaction's constraint.
     *
     * The represented field is prescribed, so the second contribution from the
     * continuity equation is removed and its value is moved to the right-hand
     * side: C^T u = M g.  No Problem is manufactured for g.
     */
    ConstraintEquation
    prescribed_constraint_equation(
      const PrescribedFieldDatum<SecondRepresentation> &datum) const
    {
      AssertThrow(
        &datum.representation() == &second,
        dealii::ExcMessage(
          "Prescribed data must use the interaction's second representation."));
      AssertDimension(datum.coefficients().size(),
                      second.dof_handler().n_dofs());

      VectorType rhs;
      rhs.reinit(second.locally_owned_dofs(), second.mpi_communicator());
      multiplier_mass_matrix_storage.vmult(rhs, datum.coefficients());

      ConstraintEquation equation(second.locally_owned_dofs(),
                                  second.mpi_communicator());
      equation.add_contribution(0,
                                coupling_matrix_storage,
                                ConstraintContributionOrientation::transpose);
      equation.set_multiplier_metric(multiplier_mass_matrix_storage);
      equation.set_rhs(rhs);
      return equation;
    }

  private:
    template <typename ParticleType>
    void
    read_particle_data(
      const ParticleType                           &particle,
      std::vector<dealii::types::global_dof_index> &second_dof_indices,
      std::vector<double>                          &second_basis_values,
      double                                       &weight) const
    {
      const auto         properties = particle.get_properties();
      const unsigned int n          = second.n_dofs_per_cell();
      AssertDimension(properties.size(), 1 + 2 * n);

      weight = properties[0];
      second_dof_indices.resize(n);
      second_basis_values.resize(n);
      for (unsigned int i = 0; i < n; ++i)
        second_dof_indices[i] =
          static_cast<dealii::types::global_dof_index>(properties[1 + i]);
      for (unsigned int i = 0; i < n; ++i)
        second_basis_values[i] = properties[1 + n + i];
    }

    template <typename ParticleType>
    void
    make_first_dof_indices(
      const ParticleType                           &particle,
      std::vector<dealii::types::global_dof_index> &first_dof_indices) const
    {
      const auto &first_cell = particle.get_surrounding_cell();
      const typename FirstRepresentation::DoFHandlerType::cell_iterator
        first_dof_cell(*first_cell, &first.dof_handler());
      first_dof_cell->get_dof_indices(first_dof_indices);
    }

    void
    assemble_coupling_matrix()
    {
      const unsigned int n_first_dofs  = first.n_dofs_per_cell();
      const unsigned int n_second_dofs = second.n_dofs_per_cell();

      dealii::DynamicSparsityPattern dsp(first.dof_handler().n_dofs(),
                                         second.dof_handler().n_dofs(),
                                         first.locally_relevant_dofs());

      std::vector<dealii::types::global_dof_index> first_dof_indices(
        n_first_dofs);
      std::vector<dealii::types::global_dof_index> second_dof_indices(
        n_second_dofs);
      std::vector<double>               unused_basis_values;
      double                            unused_weight = 0.;
      dealii::AffineConstraints<double> no_column_constraints;
      no_column_constraints.close();
      for (const auto &particle : particle_coupling.get_particles())
        {
          make_first_dof_indices(particle, first_dof_indices);
          read_particle_data(particle,
                             second_dof_indices,
                             unused_basis_values,
                             unused_weight);
          first.constraints().add_entries_local_to_global(first_dof_indices,
                                                          no_column_constraints,
                                                          second_dof_indices,
                                                          dsp);
        }

      dealii::SparsityTools::distribute_sparsity_pattern(
        dsp,
        first.locally_owned_dofs(),
        first.triangulation().get_mpi_communicator(),
        first.locally_relevant_dofs());
      coupling_matrix_storage.reinit(
        first.locally_owned_dofs(),
        second.locally_owned_dofs(),
        dsp,
        first.triangulation().get_mpi_communicator());

      dealii::FullMatrix<double> local_matrix(n_first_dofs, n_second_dofs);
      std::vector<double>        second_basis_values;
      double                     weight = 0.;
      for (const auto &particle : particle_coupling.get_particles())
        {
          make_first_dof_indices(particle, first_dof_indices);
          read_particle_data(particle,
                             second_dof_indices,
                             second_basis_values,
                             weight);
          local_matrix = 0.;

          const auto &first_fe        = first.finite_element();
          const auto &reference_point = particle.get_reference_location();
          for (unsigned int i = 0; i < n_first_dofs; ++i)
            for (unsigned int j = 0; j < n_second_dofs; ++j)
              local_matrix(i, j) = first_fe.shape_value(i, reference_point) *
                                   second_basis_values[j] * weight;

          // The multiplier columns are unconstrained algebraic variables. The
          // first representation's constraints still have to eliminate
          // constrained trace rows exactly as they do for its own operator.
          first.constraints().distribute_local_to_global(
            local_matrix,
            first_dof_indices,
            no_column_constraints,
            second_dof_indices,
            coupling_matrix_storage);
        }
      coupling_matrix_storage.compress(dealii::VectorOperation::add);
    }

    void
    assemble_mass_matrix()
    {
      dealii::DynamicSparsityPattern dsp(second.dof_handler().n_dofs(),
                                         second.dof_handler().n_dofs(),
                                         second.locally_relevant_dofs());
      std::vector<dealii::types::global_dof_index> dof_indices(
        second.n_dofs_per_cell());
      for (const auto &cell : second.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(dof_indices);
            for (const auto row : dof_indices)
              for (const auto col : dof_indices)
                dsp.add(row, col);
          }

      dealii::SparsityTools::distribute_sparsity_pattern(
        dsp,
        second.locally_owned_dofs(),
        second.triangulation().get_mpi_communicator(),
        second.locally_relevant_dofs());
      multiplier_mass_matrix_storage.reinit(
        second.locally_owned_dofs(),
        second.locally_owned_dofs(),
        dsp,
        second.triangulation().get_mpi_communicator());

      // M maps the second field into the multiplier equation, so its columns
      // must represent the constrained second algebraic space. Its rows are
      // multiplier DoFs and are intentionally unconstrained.
      dealii::AffineConstraints<double> no_row_constraints;
      no_row_constraints.close();
      dealii::FullMatrix<double> local_mass(second.n_dofs_per_cell(),
                                            second.n_dofs_per_cell());
      for (const auto &point : second_quadrature_points)
        {
          local_mass = 0.;
          for (unsigned int i = 0; i < point.dof_indices.size(); ++i)
            for (unsigned int j = 0; j < point.dof_indices.size(); ++j)
              local_mass(i, j) =
                point.weight * point.basis_values[i] * point.basis_values[j];

          no_row_constraints.distribute_local_to_global(
            local_mass,
            point.dof_indices,
            second.constraints(),
            point.dof_indices,
            multiplier_mass_matrix_storage);
        }
      multiplier_mass_matrix_storage.compress(dealii::VectorOperation::add);
    }

    const FirstRepresentation  &first;
    const SecondRepresentation &second;
    ConstraintEquation          constraint_equation_storage;
    ParticleCoupling<spacedim>  particle_coupling;

    std::unique_ptr<dealii::Quadrature<SecondRepresentation::support_dimension>>
                           quadrature;
    std::vector<PointType> second_quadrature_points;

    MatrixType coupling_matrix_storage;
    MatrixType multiplier_mass_matrix_storage;
  };

  template <typename Builder,
            typename FirstRepresentation,
            typename SecondRepresentation>
  ConstraintEquationFields
  contribute(
    Builder                                                   &builder,
    const LagrangeMultiplierInteraction<FirstRepresentation,
                                        SecondRepresentation> &interaction,
    const FieldId                                              first,
    const FieldId                                              second)
  {
    AssertThrow(!interaction.constraint_equation().contributions_view().empty(),
                dealii::ExcMessage(
                  "A Lagrange multiplier interaction must be assembled "
                  "before it is contributed."));
    return contribute_constraint_equation(builder,
                                          interaction.constraint_equation(),
                                          first,
                                          second);
  }

} // namespace ImmersX

#endif // immersx_lagrange_multiplier_interaction_h
