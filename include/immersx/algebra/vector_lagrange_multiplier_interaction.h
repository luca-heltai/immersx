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

#include <deal.II/numerics/data_out.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/constraint_contributor.h>
#include <immersx/core/constraint_equation.h>
#include <immersx/core/representation.h>
#include <immersx/coupling/particle_coupling.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
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
                                    second.mpi_communicator(),
                                    second.locally_relevant_dofs())
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

    /** Accept the multiplier state associated with an accepted solve. */
    void
    set_multiplier(const VectorType &new_multiplier)
    {
      AssertDimension(new_multiplier.size(), second.dof_handler().n_dofs());
      multiplier_storage.reinit(second.locally_owned_dofs(),
                                second.locally_relevant_dofs(),
                                second.mpi_communicator());
      multiplier_storage = new_multiplier;
      multiplier_storage.update_ghost_values();
      multiplier_initialized = true;
    }

    /** Return the accepted multiplier state, including locally relevant DoFs.
     */
    const VectorType &
    multiplier() const
    {
      AssertThrow(multiplier_initialized,
                  dealii::ExcMessage("No multiplier state has been accepted."));
      return multiplier_storage;
    }

    /** Write the accepted vector multiplier on its own representation mesh. */
    void
    output_results(const std::string &output_directory,
                   const std::string &output_name = "vector_multiplier",
                   const unsigned int cycle       = 0) const
    {
      AssertThrow(multiplier_initialized,
                  dealii::ExcMessage("No multiplier state has been accepted."));
      std::filesystem::create_directories(output_directory);

      using DataOutType =
        dealii::DataOut<SecondRepresentation::representative_dimension,
                        SecondRepresentation::ambient_dimension>;
      DataOutType data_out;
      data_out.attach_dof_handler(second.dof_handler());
      const std::vector<std::string> names(spacedim, "lagrange_multiplier");
      const std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>
        interpretation(
          spacedim,
          dealii::DataComponentInterpretation::component_is_part_of_vector);
      data_out.add_data_vector(multiplier_storage,
                               names,
                               DataOutType::type_dof_data,
                               interpretation);

      dealii::Vector<float> subdomain(second.triangulation().n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = second.triangulation().locally_owned_subdomain();
      data_out.add_data_vector(subdomain, "subdomain");
      data_out.build_patches(second.mapping());

      const std::string filename =
        output_name + "_" + std::to_string(cycle) + ".vtu";
      data_out.write_vtu_in_parallel(output_directory + "/" + filename,
                                     second.mpi_communicator());
      output_records.emplace_back(static_cast<double>(cycle), filename);

      if (dealii::Utilities::MPI::this_mpi_process(second.mpi_communicator()) ==
          0)
        {
          std::ofstream pvd(output_directory + "/" + output_name + ".pvd");
          dealii::DataOutBase::write_pvd_record(pvd, output_records);
        }
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
    mutable VectorType     multiplier_storage;
    bool                   multiplier_initialized = false;
    mutable std::vector<std::pair<double, std::string>> output_records;
    std::uint64_t assembled_first_geometry_version =
      std::numeric_limits<std::uint64_t>::max();
    std::uint64_t assembled_second_geometry_version =
      std::numeric_limits<std::uint64_t>::max();
  };

  template <typename Builder,
            typename FirstRepresentation,
            typename SecondRepresentation>
  ConstraintFields
  contribute(Builder &builder,
             const VectorLagrangeMultiplierInteraction<FirstRepresentation,
                                                       SecondRepresentation>
                          &interaction,
             const FieldId first,
             const FieldId second)
  {
    AssertThrow(!interaction.constraint_equation().contributions_view().empty(),
                dealii::ExcMessage(
                  "A vector Lagrange multiplier interaction must be assembled "
                  "before it is contributed."));
    return contribute_constraint_equation(builder,
                                          interaction.constraint_equation(),
                                          std::vector<FieldId>{first, second});
  }

} // namespace ImmersX

#endif // immersx_vector_lagrange_multiplier_interaction_h
