// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_vector_lagrange_multiplier_interaction_h
#define immersx_vector_lagrange_multiplier_interaction_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/constraint_equation.h>
#include <immersx/core/contributor.h>
#include <immersx/core/representation.h>
#include <immersx/core/semidiscrete_pde_models.h>
#include <immersx/coupling/particle_coupling.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ImmersX
{
  /**
   * Distributed, nonmatching vector velocity continuity interaction.
   *
   * The first representation is the background/matrix vector space and the
   * second representation is both the fiber vector space and the multiplier
   * space.  The matrices have the following orientations:
   *
   * @code
   *   C : matrix DoFs x multiplier DoFs
   *   Q : multiplier DoFs x fiber DoFs
   *
   *   A_m v_m + C lambda       = r_m
   *   A_f v_f - Q^T lambda     = r_f
   *   C^T v_m - Q v_f          = 0
   * @endcode
   *
   * `Q` is an interaction pairing matrix.  It is a vector-valued L2 mass
   * matrix when the multiplier and fiber representations use the same FE; it
   * is deliberately not called a physical mass matrix.  Physical mass belongs
   * to the fiber Problem and is exposed by its ElastodynamicsSolver.
   *
   * Fiber quadrature points are transported through ParticleCoupling.  The
   * particle properties contain only scalar data (weight, global DoF indices,
   * and flattened vector basis values); the typed vector values are restored
   * before assembly.  This keeps the existing distributed point-search path
   * while making component separation explicit at the matrix assembly site.
   *
   * Both representations are non-owning.  The geometry is fixed for the
   * initial application, but the representation geometry versions are recorded
   * so a later moving-geometry caller can detect stale interaction matrices.
   */
  template <typename FirstRepresentation, typename SecondRepresentation>
  class VectorLagrangeMultiplierInteraction
  {
  public:
    /** Semantic binding for the two velocity rows and multiplier row. */
    struct Fields
    {
      FieldId     first;
      FieldId     second;
      FieldId     multiplier;
      std::string prefix;

      std::string
      term(const char *local_name) const
      {
        return prefix + "." + local_name;
      }
    };

    static constexpr unsigned int spacedim =
      FirstRepresentation::ambient_dimension;

    static_assert(RepresentationConcept<FirstRepresentation>::value,
                  "The first vector interaction endpoint does not satisfy "
                  "the representation contract.");
    static_assert(RepresentationConcept<SecondRepresentation>::value,
                  "The second vector interaction endpoint does not satisfy "
                  "the representation contract.");
    static_assert(
      std::is_same<typename FirstRepresentation::value_type,
                   typename SecondRepresentation::value_type>::value,
      "Vector interaction endpoints must carry the same typed value.");
    static_assert(std::is_same<typename FirstRepresentation::value_type,
                               dealii::Tensor<1, spacedim>>::value,
                  "VectorLagrangeMultiplierInteraction requires vector-valued "
                  "representations.");
    static_assert(spacedim == SecondRepresentation::ambient_dimension,
                  "Representations must live in the same physical space.");
    static_assert(FirstRepresentation::support_dimension >=
                    SecondRepresentation::support_dimension,
                  "The second representation cannot have higher support "
                  "dimension than the first search representation.");

    using MatrixType = ImmersXLA::MPI::SparseMatrix;
    using VectorType = ImmersXLA::MPI::Vector;
    using ValueType  = typename FirstRepresentation::value_type;
    using PointType  = typename SecondRepresentation::QuadraturePoint;

    VectorLagrangeMultiplierInteraction(
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
                  dealii::ExcMessage("Interaction representations must use "
                                     "the same MPI communicator."));
    }

    /** Assemble distributed point ownership, C, and the pairing Q. */
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
      const unsigned int n_properties =
        1 + n_second_dofs + spacedim * n_second_dofs;
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

          std::vector<double> point_properties;
          point_properties.reserve(n_properties);
          point_properties.push_back(point.weight);
          for (const auto dof : point.dof_indices)
            {
              const double encoded = static_cast<double>(dof);
              AssertThrow(
                static_cast<dealii::types::global_dof_index>(encoded) == dof,
                dealii::ExcMessage(
                  "A representation DoF index cannot be transported exactly "
                  "through ParticleHandler properties."));
              point_properties.push_back(encoded);
            }
          for (const auto &basis_value : point.basis_values)
            for (unsigned int component = 0; component < spacedim; ++component)
              point_properties.push_back(basis_value[component]);
          properties.emplace_back(std::move(point_properties));
        }

      particle_coupling.insert_points(points, properties);
      assemble_coupling_matrix();
      assemble_pairing_matrix();

      constraint_equation_storage.clear_contributions();
      constraint_equation_storage.clear_rhs();
      constraint_equation_storage.add_contribution(
        0,
        coupling_matrix_storage,
        ConstraintContributionOrientation::transpose);
      constraint_equation_storage.add_contribution(
        1,
        pairing_matrix_storage,
        ConstraintContributionOrientation::direct,
        -1.);
      constraint_equation_storage.set_multiplier_metric(pairing_matrix_storage);

      assembled_first_geometry_version  = first.geometry_version();
      assembled_second_geometry_version = second.geometry_version();
    }

    /**
     * Register the interaction multiplier in a caller-owned StateLayout.
     *
     * The first and second fields are supplied explicitly because a
     * Representation may depend on more than one semantic field. The
     * multiplier field is algebraic and uses the IndexSets of the second
     * representation, which is the native multiplier space for this
     * interaction. The layout and the returned binding must outlive all model
     * callbacks.
     */
    Fields
    register_fields(StateLayout         &layout,
                    const FieldId        first_field,
                    const FieldId        second_field,
                    const std::string   &prefix,
                    const HistoryGroupId history_group = HistoryGroupId()) const
    {
      AssertThrow(layout.contains(first_field) && layout.contains(second_field),
                  dealii::ExcMessage(
                    "Interaction participant fields must belong to the "
                    "StateLayout."));
      AssertThrow(!prefix.empty(),
                  dealii::ExcMessage(
                    "An interaction field prefix cannot be empty."));

      FieldDescriptor multiplier_descriptor;
      multiplier_descriptor.name          = prefix + ".lambda";
      multiplier_descriptor.time_role     = TimeRole::algebraic;
      multiplier_descriptor.history_group = history_group;
      multiplier_descriptor.locally_owned = multiplier_locally_owned_dofs();
      multiplier_descriptor.locally_relevant =
        multiplier_locally_relevant_dofs();

      Fields fields;
      fields.first      = first_field;
      fields.second     = second_field;
      fields.multiplier = layout.add_field(std::move(multiplier_descriptor));
      fields.prefix     = prefix;
      return fields;
    }

    /** Add C lambda, -Q^T lambda, and C^T v_first - Q v_second. */
    void
    add_semidiscrete_terms(SemiDiscreteModel<VectorType> &model,
                           const Fields                  &fields) const
    {
      AssertThrow(assembly_is_current(),
                  dealii::ExcMessage(
                    "Cannot register an interaction with stale matrices."));

      model.add_term(
        fields.term("first"),
        [this, fields](const auto &context, auto &residual) {
          const auto &lambda =
            context.state().field(fields.multiplier, context.time());
          auto      &row = residual.field(fields.first);
          VectorType contribution;
          contribution.reinit(row);
          coupling_matrix_storage.vmult(contribution, lambda);
          row += contribution;
        },
        [this, fields](const auto &linearization,
                       const auto &increment,
                       auto       &residual) {
          const double a   = linearization.state_weight();
          const double t   = linearization.evaluation().time();
          auto        &row = residual.field(fields.first);
          VectorType   contribution;
          contribution.reinit(row);
          coupling_matrix_storage.vmult(contribution,
                                        increment.field(fields.multiplier, t));
          contribution *= a;
          row += contribution;
        });

      model.add_term(
        fields.term("second"),
        [this, fields](const auto &context, auto &residual) {
          const auto &lambda =
            context.state().field(fields.multiplier, context.time());
          auto      &row = residual.field(fields.second);
          VectorType contribution;
          contribution.reinit(row);
          pairing_matrix_storage.Tvmult(contribution, lambda);
          contribution *= -1.;
          row += contribution;
        },
        [this, fields](const auto &linearization,
                       const auto &increment,
                       auto       &residual) {
          const double a   = linearization.state_weight();
          const double t   = linearization.evaluation().time();
          auto        &row = residual.field(fields.second);
          VectorType   contribution;
          contribution.reinit(row);
          pairing_matrix_storage.Tvmult(contribution,
                                        increment.field(fields.multiplier, t));
          contribution *= -a;
          row += contribution;
        });

      model.add_term(
        fields.term("constraint"),
        [this, fields](const auto &context, auto &residual) {
          const auto &first_velocity =
            context.state().field(fields.first, context.time());
          const auto &second_velocity =
            context.state().field(fields.second, context.time());
          auto      &row = residual.field(fields.multiplier);
          VectorType contribution;
          contribution.reinit(row);
          coupling_matrix_storage.Tvmult(contribution, first_velocity);
          row += contribution;
          pairing_matrix_storage.vmult(contribution, second_velocity);
          row -= contribution;
        },
        [this, fields](const auto &linearization,
                       const auto &increment,
                       auto       &residual) {
          const double a   = linearization.state_weight();
          const double t   = linearization.evaluation().time();
          auto        &row = residual.field(fields.multiplier);
          VectorType   contribution;
          contribution.reinit(row);
          coupling_matrix_storage.Tvmult(contribution,
                                         increment.field(fields.first, t));
          contribution *= a;
          row += contribution;
          pairing_matrix_storage.vmult(contribution,
                                       increment.field(fields.second, t));
          contribution *= -a;
          row += contribution;
        });
    }

    /** Return C, with matrix rows and multiplier columns. */
    const MatrixType &
    coupling_matrix() const
    {
      return coupling_matrix_storage;
    }

    /** Return Q, with multiplier rows and fiber columns. */
    const MatrixType &
    pairing_matrix() const
    {
      return pairing_matrix_storage;
    }

    /** Descriptive compatibility alias for the interaction pairing Q. */
    const MatrixType &
    multiplier_mass_matrix() const
    {
      return pairing_matrix_storage;
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

    const ConstraintEquation &
    constraint_equation() const
    {
      return constraint_equation_storage;
    }

    const FirstRepresentation &
    first_representation() const
    {
      return first;
    }

    const SecondRepresentation &
    second_representation() const
    {
      return second;
    }

    /** Whether the assembled matrices correspond to current endpoint geometry.
     */
    bool
    assembly_is_current() const
    {
      return assembled_first_geometry_version == first.geometry_version() &&
             assembled_second_geometry_version == second.geometry_version();
    }

    /** Expose the search object for diagnostics and distributed tests. */
    const ParticleCoupling<spacedim> &
    particle_search() const
    {
      return particle_coupling;
    }

  private:
    template <typename ParticleType>
    void
    read_particle_data(
      const ParticleType                           &particle,
      std::vector<dealii::types::global_dof_index> &dof_indices,
      std::vector<ValueType>                       &basis_values,
      double                                       &weight) const
    {
      const auto         properties = particle.get_properties();
      const unsigned int n          = second.n_dofs_per_cell();
      AssertDimension(properties.size(), 1 + n + spacedim * n);

      weight = properties[0];
      dof_indices.resize(n);
      for (unsigned int i = 0; i < n; ++i)
        {
          dof_indices[i] =
            static_cast<dealii::types::global_dof_index>(properties[1 + i]);
          AssertThrow(static_cast<double>(dof_indices[i]) == properties[1 + i],
                      dealii::ExcMessage(
                        "A transported representation DoF index is not an "
                        "exact global index."));
        }

      basis_values.assign(n, ValueType());
      const unsigned int basis_offset = 1 + n;
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int component = 0; component < spacedim; ++component)
          basis_values[i][component] =
            properties[basis_offset + i * spacedim + component];
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

    ValueType
    first_basis_value(
      const unsigned int                                           index,
      const dealii::Point<FirstRepresentation::support_dimension> &reference)
      const
    {
      ValueType value;
      value = 0.;
      for (unsigned int component = 0; component < spacedim; ++component)
        value[component] =
          first.finite_element().shape_value_component(index,
                                                       reference,
                                                       component);
      return value;
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
      std::vector<ValueType>            unused_basis_values;
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
      std::vector<ValueType>     second_basis_values;
      double                     weight = 0.;
      for (const auto &particle : particle_coupling.get_particles())
        {
          make_first_dof_indices(particle, first_dof_indices);
          read_particle_data(particle,
                             second_dof_indices,
                             second_basis_values,
                             weight);
          local_matrix = 0.;

          const auto &reference_point = particle.get_reference_location();
          for (unsigned int i = 0; i < n_first_dofs; ++i)
            {
              const auto first_value = first_basis_value(i, reference_point);
              for (unsigned int j = 0; j < n_second_dofs; ++j)
                local_matrix(i, j) =
                  (first_value * second_basis_values[j]) * weight;
            }

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
    assemble_pairing_matrix()
    {
      const unsigned int             n_dofs = second.n_dofs_per_cell();
      dealii::DynamicSparsityPattern dsp(second.dof_handler().n_dofs(),
                                         second.dof_handler().n_dofs(),
                                         second.locally_relevant_dofs());
      std::vector<dealii::types::global_dof_index> dof_indices(n_dofs);
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
      pairing_matrix_storage.reinit(
        second.locally_owned_dofs(),
        second.locally_owned_dofs(),
        dsp,
        second.triangulation().get_mpi_communicator());

      dealii::AffineConstraints<double> no_row_constraints;
      no_row_constraints.close();
      dealii::FullMatrix<double> local_matrix(n_dofs, n_dofs);
      for (const auto &point : second_quadrature_points)
        {
          local_matrix = 0.;
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              local_matrix(i, j) =
                point.weight * (point.basis_values[i] * point.basis_values[j]);

          no_row_constraints.distribute_local_to_global(local_matrix,
                                                        point.dof_indices,
                                                        second.constraints(),
                                                        point.dof_indices,
                                                        pairing_matrix_storage);
        }
      pairing_matrix_storage.compress(dealii::VectorOperation::add);
    }

    const FirstRepresentation  &first;
    const SecondRepresentation &second;
    ConstraintEquation          constraint_equation_storage;
    ParticleCoupling<spacedim>  particle_coupling;

    std::unique_ptr<dealii::Quadrature<SecondRepresentation::support_dimension>>
                           quadrature;
    std::vector<PointType> second_quadrature_points;
    MatrixType             coupling_matrix_storage;
    MatrixType             pairing_matrix_storage;
    std::uint64_t          assembled_first_geometry_version =
      std::numeric_limits<std::uint64_t>::max();
    std::uint64_t assembled_second_geometry_version =
      std::numeric_limits<std::uint64_t>::max();
  };

  /** Contributor for a vector Lagrange-multiplier interaction. */
  template <typename Interaction>
  class VectorLagrangeMultiplierContributor
  {
  public:
    using VectorType = typename Interaction::VectorType;
    using Fields     = typename Interaction::Fields;

    VectorLagrangeMultiplierContributor(const Interaction &interaction,
                                        const FieldId      first,
                                        const FieldId      second)
      : interaction_(interaction)
      , first_(first)
      , second_(second)
    {}

    template <typename Builder>
    Fields
    operator()(Builder &builder, const std::string &prefix) const
    {
      AssertThrow(interaction_.assembly_is_current(),
                  dealii::ExcMessage(
                    "Cannot register an interaction with stale matrices."));

      FieldDescriptor descriptor;
      descriptor.name          = prefix + ".lambda";
      descriptor.time_role     = TimeRole::algebraic;
      descriptor.locally_owned = interaction_.multiplier_locally_owned_dofs();
      descriptor.locally_relevant =
        interaction_.multiplier_locally_relevant_dofs();
      const auto multiplier = builder.add_field(std::move(descriptor));
      Fields     fields{first_, second_, multiplier, prefix};

      const auto C =
        ImmersX::payload_free(dealii::linear_operator<VectorType, VectorType>(
          interaction_.coupling_matrix()));
      const auto Q =
        ImmersX::payload_free(dealii::linear_operator<VectorType, VectorType>(
          interaction_.pairing_matrix()));
      const auto Ct = dealii::transpose_operator(C);
      const auto Qt = dealii::transpose_operator(Q);

      builder.add_residual(
        first_, fields.term("first"), [fields, C](const auto &context) {
          return C * context.state().field(fields.multiplier, context.time());
        });
      builder.add_residual(
        second_, fields.term("second"), [fields, Qt](const auto &context) {
          return (-1. * Qt) *
                 context.state().field(fields.multiplier, context.time());
        });
      builder.add_residual(multiplier,
                           fields.term("constraint"),
                           [fields, Ct, Q](const auto &context) {
                             return Ct * context.state().field(fields.first,
                                                               context.time()) -
                                    Q * context.state().field(fields.second,
                                                              context.time());
                           });

      builder.add_state_operator(first_, multiplier, fields.term("first"), C);
      builder.add_state_operator(second_,
                                 multiplier,
                                 fields.term("second"),
                                 -1. * Qt);
      builder.add_state_operator(multiplier,
                                 first_,
                                 fields.term("constraint"),
                                 Ct);
      builder.add_state_operator(multiplier,
                                 second_,
                                 fields.term("constraint"),
                                 -1. * Q);
      return fields;
    }

  private:
    const Interaction &interaction_;
    FieldId            first_;
    FieldId            second_;
  };

  template <typename Interaction>
  VectorLagrangeMultiplierContributor<Interaction>
  vector_lagrange_multiplier(const Interaction &interaction,
                             const FieldId      first,
                             const FieldId      second)
  {
    return VectorLagrangeMultiplierContributor<Interaction>(interaction,
                                                            first,
                                                            second);
  }

} // namespace ImmersX

#endif // immersx_vector_lagrange_multiplier_interaction_h
