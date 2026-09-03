// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_vessel_wall_interaction_h
#define immersx_vessel_wall_interaction_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/constraint_contributor.h>
#include <immersx/core/contributor.h>
#include <immersx/core/representation.h>
#include <immersx/coupling/particle_coupling.h>
#include <immersx/physics/metric_flow_x_vessel_wall_representation.h>

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
   * One-way MetricFlowX vessel-wall constraint.
   *
   * With `B` the scalar-multiplier/vector-displacement pairing and `P(A)` the
   * nonlinear Area-to-wall pairing, the current residual rows are
   *
   *   F_solid + B lambda = 0,
   *   B^T displacement - P(A) = 0,
   *   F_flow(A, U, ...) = 0.
   *
   * The multiplier `lambda` is intended to represent the pressure difference
   * Delta p.  The sign convention is fixed by the equations above: positive
   * lambda contributes `+B lambda` to the Elastodynamics dynamics residual and
   * positive radial displacement gives a positive constraint value.  Thus the
   * physical traction associated with a positive lambda is the negative of the
   * residual reaction under the usual `internal force - applied force` form.
   *
   * Deliberately absent is the future flow-row term
   * `F_DeltaP(lambda)`.  No lambda term is added to the MetricFlowX state row
   * until BloodFlowSystem exposes the required external/transmural pressure
   * forcing capability.  This class therefore does not guess whether that API
   * should consume quadrature pressures, coefficients, a residual operation,
   * or a source vector.
   */
  template <typename SolidRepresentation,
#ifdef IMMERSX_WITH_METRIC_FLOW_X
            typename WallRepresentation =
              MetricFlowXAreaRadialDisplacementRepresentation>
#else
            typename WallRepresentation>
#endif
  class VesselWallInteraction
  {
  public:
    static constexpr unsigned int spacedim =
      SolidRepresentation::ambient_dimension;
    static_assert(spacedim == 3,
                  "The initial MetricFlowX vessel wall is embedded in 3D.");
    static_assert(RepresentationConcept<SolidRepresentation>::value,
                  "The solid endpoint does not satisfy the representation "
                  "contract.");
    static_assert(std::is_same<typename SolidRepresentation::value_type,
                               dealii::Tensor<1, spacedim>>::value,
                  "VesselWallInteraction needs a vector-valued solid "
                  "representation.");
    static_assert(std::is_same<typename WallRepresentation::value_type,
                               std::vector<dealii::Tensor<1, spacedim>>>::value,
                  "The vessel-wall endpoint must be a vector-valued sampled "
                  "Area representation.");

    using MatrixType = ImmersXLA::MPI::SparseMatrix;
    using VectorType = ImmersXLA::MPI::Vector;
    using PointType  = typename WallRepresentation::WallPoint;

    VesselWallInteraction(
      const SolidRepresentation                  &solid,
      const WallRepresentation                   &wall,
      const ParticleCouplingParameters<spacedim> &search_parameters)
      : solid_(solid)
      , wall_(wall)
      , constraint_equation_storage(wall.multiplier_locally_owned_dofs(),
                                    wall.mpi_communicator(),
                                    wall.multiplier_locally_relevant_dofs())
      , particle_coupling_(search_parameters)
    {
      AssertThrow(solid.triangulation().get_mpi_communicator() ==
                    wall.mpi_communicator(),
                  dealii::ExcMessage(
                    "The solid and MetricFlowX wall must use the same MPI "
                    "communicator."));
    }

    /** Assemble the solid reaction, scalar wall metric, and point search. */
    void
    assemble()
    {
      const unsigned int degree =
        std::max(solid_.finite_element().degree, wall_.finite_element().degree);
      quadrature_  = std::make_unique<dealii::QGauss<2>>(degree + 1);
      wall_points_ = wall_.locally_owned_quadrature_points(*quadrature_);

      const unsigned int n_lambda = wall_.n_dofs_per_cell();
      if (!wall_points_.empty())
        AssertThrow(n_lambda == wall_points_.front().dof_indices.size(),
                    dealii::ExcMessage(
                      "Wall representation DoF metadata is inconsistent."));
      const unsigned int n_properties = 1 + 2 * n_lambda + spacedim;
      particle_coupling_.initialize_particle_handler(solid_.triangulation(),
                                                     solid_.mapping(),
                                                     n_properties);

      std::vector<dealii::Point<spacedim>> points;
      std::vector<std::vector<double>>     properties;
      points.reserve(wall_points_.size());
      properties.reserve(wall_points_.size());
      for (const auto &point : wall_points_)
        {
          points.push_back(point.point);
          std::vector<double> data;
          data.reserve(n_properties);
          data.push_back(point.weight);
          for (const auto dof : point.multiplier_dof_indices)
            {
              const double encoded = static_cast<double>(dof);
              AssertThrow(
                static_cast<dealii::types::global_dof_index>(encoded) == dof,
                dealii::ExcMessage(
                  "A wall DoF index cannot be transported exactly through "
                  "ParticleHandler properties."));
              data.push_back(encoded);
            }
          for (const auto basis : point.area_basis_values)
            data.push_back(basis);
          for (unsigned int component = 0; component < spacedim; ++component)
            data.push_back(point.normal[component]);
          properties.emplace_back(std::move(data));
        }
      particle_coupling_.insert_points(points, properties);

      assemble_solid_coupling();
      assemble_multiplier_metric();

      constraint_equation_storage.clear_contributions();
      constraint_equation_storage.clear_rhs();
      constraint_equation_storage.add_contribution(
        0,
        solid_coupling_storage,
        ConstraintContributionOrientation::transpose);
      constraint_equation_storage.set_multiplier_metric(
        multiplier_metric_storage);
      assembled_ = true;
    }

    bool
    assembled() const
    {
      return assembled_;
    }

    const MatrixType &
    solid_coupling_matrix() const
    {
      return solid_coupling_storage;
    }

    const MatrixType &
    multiplier_metric_matrix() const
    {
      return multiplier_metric_storage;
    }

    const ConstraintEquation &
    constraint_equation() const
    {
      return constraint_equation_storage;
    }

    const SolidRepresentation &
    solid_representation() const
    {
      return solid_;
    }

    const WallRepresentation &
    wall_representation() const
    {
      return wall_;
    }

    const dealii::IndexSet &
    multiplier_locally_owned_dofs() const
    {
      return wall_.multiplier_locally_owned_dofs();
    }

    const dealii::IndexSet &
    multiplier_locally_relevant_dofs() const
    {
      return wall_.multiplier_locally_relevant_dofs();
    }

    /** Accept the scalar pressure-difference state at an accepted step. */
    void
    set_multiplier(const VectorType &new_multiplier)
    {
      multiplier_storage.reinit(wall_.multiplier_locally_owned_dofs(),
                                wall_.multiplier_locally_relevant_dofs(),
                                wall_.mpi_communicator());
      multiplier_storage = new_multiplier;
      multiplier_storage.update_ghost_values();
      multiplier_initialized_ = true;
    }

    const VectorType &
    multiplier() const
    {
      AssertThrow(multiplier_initialized_,
                  dealii::ExcMessage("No accepted vessel-wall multiplier."));
      return multiplier_storage;
    }

    /** Write the accepted scalar multiplier on the MetricFlowX centerline. */
    void
    output_results(
      const std::string &output_directory,
      const std::string &output_name = "pressure_jump",
      const unsigned int cycle       = 0,
      const double       time = std::numeric_limits<double>::quiet_NaN()) const
    {
      AssertThrow(multiplier_initialized_,
                  dealii::ExcMessage("No accepted vessel-wall multiplier."));
      std::filesystem::create_directories(output_directory);
      const auto centerline_owned = wall_.dof_handler().locally_owned_dofs();
      dealii::IndexSet centerline_relevant(centerline_owned.size());
      for (const auto index : wall_.area_dof_numbers())
        centerline_relevant.add_index(index);
      centerline_relevant.compress();
      VectorType centerline_multiplier;
      centerline_multiplier.reinit(centerline_owned,
                                   centerline_relevant,
                                   wall_.mpi_communicator());
      centerline_multiplier = 0.;
      for (const auto index : multiplier_storage.locally_owned_elements())
        centerline_multiplier[wall_.area_dof_numbers()[index]] =
          multiplier_storage[index];
      centerline_multiplier.compress(dealii::VectorOperation::insert);
      dealii::DataOut<1, 3> data_out;
      data_out.attach_dof_handler(wall_.dof_handler());
      data_out.add_data_vector(centerline_multiplier, output_name);
      data_out.build_patches(wall_.mapping());
      const std::string filename =
        output_name + "_" + std::to_string(cycle) + ".vtu";
      data_out.write_vtu_in_parallel(output_directory + "/" + filename,
                                     wall_.mpi_communicator());
      output_records_.emplace_back(
        std::isfinite(time) ? time : static_cast<double>(cycle), filename);
      if (dealii::Utilities::MPI::this_mpi_process(wall_.mpi_communicator()) ==
          0)
        {
          std::ofstream pvd(output_directory + "/" + output_name + ".pvd");
          dealii::DataOutBase::write_pvd_record(pvd, output_records_);
        }
    }

    /**
     * Explicit status for the intentionally absent two-way pressure term.
     * This is a physical capability flag, not a guessed MetricFlowX API.
     */
    static constexpr bool flow_pressure_feedback_is_implemented = false;

  private:
    template <typename ParticleType>
    void
    read_particle_data(const ParticleType                           &particle,
                       std::vector<double>                          &basis,
                       dealii::Tensor<1, spacedim>                  &normal,
                       double                                       &weight,
                       std::vector<dealii::types::global_dof_index> &dofs) const
    {
      const auto         properties = particle.get_properties();
      const unsigned int n          = wall_.n_dofs_per_cell();
      AssertDimension(properties.size(), 1 + 2 * n + spacedim);
      weight = properties[0];
      dofs.resize(n);
      for (unsigned int i = 0; i < n; ++i)
        {
          dofs[i] =
            static_cast<dealii::types::global_dof_index>(properties[1 + i]);
          AssertThrow(static_cast<double>(dofs[i]) == properties[1 + i],
                      dealii::ExcMessage(
                        "A transported wall DoF index is not an exact "
                        "global index."));
        }
      basis.assign(properties.begin() + 1 + n, properties.begin() + 1 + 2 * n);
      normal = 0.;
      for (unsigned int component = 0; component < spacedim; ++component)
        normal[component] = properties[1 + 2 * n + component];
    }

    template <typename ParticleType>
    void
    make_solid_dof_indices(
      const ParticleType                           &particle,
      std::vector<dealii::types::global_dof_index> &dofs) const
    {
      const auto &cell = particle.get_surrounding_cell();
      const typename SolidRepresentation::DoFHandlerType::cell_iterator
        solid_cell(*cell, &solid_.dof_handler());
      solid_cell->get_dof_indices(dofs);
    }

    dealii::Tensor<1, spacedim>
    solid_basis_value(
      const unsigned int                                           i,
      const dealii::Point<SolidRepresentation::support_dimension> &reference)
      const
    {
      dealii::Tensor<1, spacedim> value;
      value = 0.;
      for (unsigned int component = 0; component < spacedim; ++component)
        value[component] =
          solid_.finite_element().shape_value_component(i,
                                                        reference,
                                                        component);
      return value;
    }

    void
    assemble_solid_coupling()
    {
      const unsigned int             n_first  = solid_.n_dofs_per_cell();
      const unsigned int             n_second = wall_.n_dofs_per_cell();
      dealii::DynamicSparsityPattern dsp(
        solid_.dof_handler().n_dofs(),
        wall_.multiplier_locally_owned_dofs().size(),
        solid_.locally_relevant_dofs());
      std::vector<dealii::types::global_dof_index> first_dofs(n_first);
      std::vector<dealii::types::global_dof_index> second_dofs(n_second);
      dealii::AffineConstraints<double>            no_column_constraints;
      no_column_constraints.close();
      for (const auto &particle : particle_coupling_.get_particles())
        {
          make_solid_dof_indices(particle, first_dofs);
          std::vector<double>         unused_basis;
          dealii::Tensor<1, spacedim> unused_normal;
          double                      unused_weight = 0.;
          read_particle_data(
            particle, unused_basis, unused_normal, unused_weight, second_dofs);
          solid_.constraints().add_entries_local_to_global(
            first_dofs, no_column_constraints, second_dofs, dsp);
        }
      dealii::SparsityTools::distribute_sparsity_pattern(
        dsp,
        solid_.locally_owned_dofs(),
        solid_.triangulation().get_mpi_communicator(),
        solid_.locally_relevant_dofs());
      solid_coupling_storage.reinit(solid_.locally_owned_dofs(),
                                    wall_.multiplier_locally_owned_dofs(),
                                    dsp,
                                    solid_.mpi_communicator());

      dealii::FullMatrix<double>  local_matrix(n_first, n_second);
      std::vector<double>         basis;
      dealii::Tensor<1, spacedim> normal;
      double                      weight = 0.;
      for (const auto &particle : particle_coupling_.get_particles())
        {
          make_solid_dof_indices(particle, first_dofs);
          std::vector<dealii::types::global_dof_index> unused_dofs;
          read_particle_data(particle, basis, normal, weight, second_dofs);
          const auto &reference = particle.get_reference_location();
          local_matrix          = 0.;
          for (unsigned int i = 0; i < n_first; ++i)
            for (unsigned int j = 0; j < n_second; ++j)
              local_matrix(i, j) =
                weight * (solid_basis_value(i, reference) * normal) * basis[j];
          solid_.constraints().distribute_local_to_global(
            local_matrix,
            first_dofs,
            no_column_constraints,
            second_dofs,
            solid_coupling_storage);
        }
      solid_coupling_storage.compress(dealii::VectorOperation::add);
    }

    void
    assemble_multiplier_metric()
    {
      const unsigned int             n = wall_.n_dofs_per_cell();
      dealii::DynamicSparsityPattern dsp(
        wall_.multiplier_locally_owned_dofs().size(),
        wall_.multiplier_locally_owned_dofs().size(),
        wall_.multiplier_locally_relevant_dofs());
      for (const auto &point : wall_points_)
        for (const auto row : point.multiplier_dof_indices)
          for (const auto column : point.multiplier_dof_indices)
            dsp.add(row, column);
      dealii::SparsityTools::distribute_sparsity_pattern(
        dsp,
        wall_.multiplier_locally_owned_dofs(),
        wall_.mpi_communicator(),
        wall_.multiplier_locally_relevant_dofs());
      multiplier_metric_storage.reinit(wall_.multiplier_locally_owned_dofs(),
                                       wall_.multiplier_locally_owned_dofs(),
                                       dsp,
                                       wall_.mpi_communicator());
      dealii::FullMatrix<double>        local_mass(n, n);
      dealii::AffineConstraints<double> no_constraints;
      no_constraints.close();
      for (const auto &point : wall_points_)
        {
          local_mass = 0.;
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              local_mass(i, j) = point.weight * point.area_basis_values[i] *
                                 point.area_basis_values[j];
          no_constraints.distribute_local_to_global(
            local_mass,
            point.multiplier_dof_indices,
            multiplier_metric_storage);
        }
      multiplier_metric_storage.compress(dealii::VectorOperation::add);
    }

  public:
    dealii::PackagedOperation<VectorType>
    constraint_residual(const EvaluationContext<VectorType> &context,
                        const FieldId solid_displacement) const
    {
      const auto *solid_state = &context.state(solid_displacement);
      const auto *interaction = this;
      dealii::PackagedOperation<VectorType> result;
      result.reinit_vector = [this](VectorType &vector, const bool omit) {
        vector.reinit(wall_.multiplier_locally_owned_dofs(),
                      wall_.multiplier_locally_relevant_dofs(),
                      wall_.mpi_communicator());
        if (!omit)
          vector = 0.;
      };
      result.apply = [interaction, solid_state, &context](VectorType &value) {
        interaction->solid_coupling_storage.Tvmult(value, *solid_state);
        const auto wall_values = interaction->wall_.evaluate(context);
        for (std::size_t q = 0; q < interaction->wall_points_.size(); ++q)
          {
            const double radial_value =
              interaction->wall_points_[q].normal * wall_values[q];
            for (unsigned int i = 0;
                 i < interaction->wall_points_[q].multiplier_dof_indices.size();
                 ++i)
              value[interaction->wall_points_[q].multiplier_dof_indices[i]] -=
                interaction->wall_points_[q].weight * radial_value *
                interaction->wall_points_[q].area_basis_values[i];
          }
        value.compress(dealii::VectorOperation::add);
      };
      result.apply_add = [interaction, solid_state, &context](
                           VectorType &value) {
        VectorType contribution;
        contribution.reinit(
          interaction->wall_.multiplier_locally_owned_dofs(),
          interaction->wall_.multiplier_locally_relevant_dofs(),
          interaction->wall_.mpi_communicator());
        contribution = 0.;
        interaction->solid_coupling_storage.Tvmult(contribution, *solid_state);
        const auto wall_values = interaction->wall_.evaluate(context);
        for (std::size_t q = 0; q < interaction->wall_points_.size(); ++q)
          {
            const double radial_value =
              interaction->wall_points_[q].normal * wall_values[q];
            for (unsigned int i = 0;
                 i < interaction->wall_points_[q].multiplier_dof_indices.size();
                 ++i)
              contribution[interaction->wall_points_[q]
                             .multiplier_dof_indices[i]] -=
                interaction->wall_points_[q].weight * radial_value *
                interaction->wall_points_[q].area_basis_values[i];
          }
        contribution.compress(dealii::VectorOperation::add);
        value += contribution;
      };
      return result;
    }

    dealii::LinearOperator<VectorType, VectorType>
    area_constraint_jacobian(const EvaluationContext<VectorType> &context) const
    {
      const auto representation_derivative = wall_.linearize(context);
      const auto points                    = wall_points_;
      const auto owned        = wall_.multiplier_locally_owned_dofs();
      const auto relevant     = wall_.multiplier_locally_relevant_dofs();
      const auto communicator = wall_.mpi_communicator();
      dealii::LinearOperator<VectorType, VectorType> result;
      result.reinit_range_vector =
        [owned, relevant, communicator](VectorType &vector, const bool omit) {
          vector.reinit(owned, relevant, communicator);
          if (!omit)
            vector = 0.;
        };
      result.reinit_domain_vector =
        [reference = context.state(wall_.source())](VectorType &vector,
                                                    const bool  omit) {
          vector.reinit(reference, omit);
        };
      result.vmult = [representation_derivative,
                      points](VectorType       &destination,
                              const VectorType &direction) {
        std::vector<dealii::Tensor<1, 3>> values;
        representation_derivative.reinit_range_vector(values, false);
        representation_derivative.vmult(values, direction);
        destination = 0.;
        for (std::size_t q = 0; q < points.size(); ++q)
          {
            const double value =
              points[q].weight * (points[q].normal * values[q]);
            for (unsigned int i = 0;
                 i < points[q].multiplier_dof_indices.size();
                 ++i)
              destination[points[q].multiplier_dof_indices[i]] +=
                value * points[q].area_basis_values[i];
          }
        destination.compress(dealii::VectorOperation::add);
      };
      result.vmult_add = [representation_derivative,
                          points](VectorType       &destination,
                                  const VectorType &direction) {
        std::vector<dealii::Tensor<1, 3>> values;
        representation_derivative.reinit_range_vector(values, false);
        representation_derivative.vmult(values, direction);
        for (std::size_t q = 0; q < points.size(); ++q)
          {
            const double value =
              points[q].weight * (points[q].normal * values[q]);
            for (unsigned int i = 0;
                 i < points[q].multiplier_dof_indices.size();
                 ++i)
              destination[points[q].multiplier_dof_indices[i]] +=
                value * points[q].area_basis_values[i];
          }
        destination.compress(dealii::VectorOperation::add);
      };
      result.Tvmult = [representation_derivative,
                       points](VectorType       &destination,
                               const VectorType &source) {
        std::vector<dealii::Tensor<1, 3>> values(points.size());
        for (std::size_t q = 0; q < points.size(); ++q)
          {
            double value = 0.;
            for (unsigned int i = 0;
                 i < points[q].multiplier_dof_indices.size();
                 ++i)
              value += points[q].area_basis_values[i] *
                       source[points[q].multiplier_dof_indices[i]];
            values[q] = points[q].normal * (points[q].weight * value);
          }
        representation_derivative.Tvmult(destination, values);
      };
      result.Tvmult_add = [representation_derivative,
                           points](VectorType       &destination,
                                   const VectorType &source) {
        std::vector<dealii::Tensor<1, 3>> values(points.size());
        for (std::size_t q = 0; q < points.size(); ++q)
          {
            double value = 0.;
            for (unsigned int i = 0;
                 i < points[q].multiplier_dof_indices.size();
                 ++i)
              value += points[q].area_basis_values[i] *
                       source[points[q].multiplier_dof_indices[i]];
            values[q] = points[q].normal * (points[q].weight * value);
          }
        representation_derivative.Tvmult_add(destination, values);
      };
      return result;
    }

  private:
    const SolidRepresentation             &solid_;
    const WallRepresentation              &wall_;
    ConstraintEquation                     constraint_equation_storage;
    ParticleCoupling<spacedim>             particle_coupling_;
    std::unique_ptr<dealii::Quadrature<2>> quadrature_;
    std::vector<PointType>                 wall_points_;
    MatrixType                             solid_coupling_storage;
    MatrixType                             multiplier_metric_storage;
    mutable VectorType                     multiplier_storage;
    bool                                   multiplier_initialized_ = false;
    bool                                   assembled_              = false;
    mutable std::vector<std::pair<double, std::string>> output_records_;
  };

  template <typename Builder,
            typename SolidRepresentation,
            typename WallRepresentation>
  ConstraintFields
  contribute(Builder                                         &builder,
             const VesselWallInteraction<SolidRepresentation,
                                         WallRepresentation> &interaction,
             const FieldId solid_displacement,
             const FieldId solid_velocity,
             const FieldId flow_state)
  {
    AssertThrow(interaction.assembled(),
                dealii::ExcMessage(
                  "A vessel-wall interaction must be assembled before it is "
                  "contributed."));

    const auto multiplier =
      builder.algebraic_field("lambda",
                              interaction.multiplier_locally_owned_dofs(),
                              interaction.multiplier_locally_relevant_dofs());
    builder.preconditioner(
      flow_state, [](const auto &matrix, const auto &reinit_vector) {
        return make_amg_preconditioner<
          typename WallRepresentation::StateType,
          typename VesselWallInteraction<SolidRepresentation,
                                         WallRepresentation>::MatrixType>(
          matrix, reinit_vector);
      });
    builder.saddle_point(multiplier,
                         std::vector<FieldId>{solid_displacement, flow_state},
                         builder.matrix_operator(
                           interaction.multiplier_metric_matrix()));

    const auto solid_reaction =
      builder.matrix_operator(interaction.solid_coupling_matrix());
    builder.term(solid_velocity, "vessel-wall-reaction")
      .residual([solid_reaction, multiplier](const auto &context) {
        return solid_reaction.view * context.state(multiplier);
      })
      .state(multiplier, solid_reaction.view);

    const auto solid_constraint =
      dealii::transpose_operator(solid_reaction.view);
    builder.term(multiplier, "vessel-wall-constraint")
      .residual([&interaction, solid_displacement](const auto &context) {
        return interaction.constraint_residual(context, solid_displacement);
      })
      .state(solid_displacement, solid_constraint)
      .state(flow_state,
             typename Builder::Model::OperatorFactory(
               [&interaction](const auto &context) {
                 return -1. * interaction.area_constraint_jacobian(context);
               }));

    // There is intentionally no term on `flow_state`.  This is the explicit
    // future hook for F_DeltaP(lambda): once BloodFlowSystem exposes an
    // external/transmural-pressure forcing capability, add that physical term
    // and its Jacobian here without guessing its eventual API today.
    (void)flow_state;
    return {multiplier};
  }
} // namespace ImmersX

#endif // immersx_vessel_wall_interaction_h
