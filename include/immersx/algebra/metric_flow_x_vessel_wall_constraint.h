// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_metric_flow_x_vessel_wall_constraint_h
#define immersx_metric_flow_x_vessel_wall_constraint_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/constraint.h>
#include <immersx/core/contributor.h>
#include <immersx/physics/metric_flow_x_vessel_wall_observable.h>

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
   * MetricFlowX vessel-wall interaction.
   *
   * With `B` the scalar-multiplier/vector-displacement pairing and `P(A)` the
   * nonlinear Area-to-wall pairing, the current residual rows are
   *
   *   F_solid + B lambda = 0,
   *   B^T displacement - P(A) = 0,
   *   F_flow(A, U, ...) + F_DeltaP(lambda) = 0.
   *
   * The multiplier `lambda` is the pressure difference
   *
   *   lambda = p_external - p_internal,tube.
   *
   * Consequently positive lambda produces an inward wall traction through
   * `+B lambda`, while the MetricFlowX external-pressure convention receives
   * `p_external_pressure = -lambda`.  This is the action/reaction sign implied
   * by the outward wall normal and the residual convention
   * `internal force - applied force`.
   */
  template <typename SolidField,
#ifdef IMMERSX_WITH_METRIC_FLOW_X
            typename WallObservable = MetricFlowXVesselWallGeometry>
#else
            typename WallObservable>
#endif
  class MetricFlowXVesselWallConstraint
  {
  public:
    static constexpr unsigned int spacedim = SolidField::spacedimension();
    static_assert(spacedim == 3,
                  "The initial MetricFlowX vessel wall is embedded in 3D.");
    static_assert(std::is_same<typename SolidField::value_type,
                               dealii::Tensor<1, spacedim>>::value,
                  "MetricFlowX vessel constraint needs a vector-valued solid "
                  "representation.");

    using MatrixType = ImmersXLA::MPI::SparseMatrix;
    using VectorType = ImmersXLA::MPI::Vector;

    MetricFlowXVesselWallConstraint(const SolidField     &solid,
                                    const WallObservable &wall)
      : wall_(wall)
    {
      AssertThrow(solid.space().mpi_communicator() == wall.mpi_communicator(),
                  dealii::ExcMessage(
                    "The solid and MetricFlowX wall must use the same MPI "
                    "communicator."));
    }

    template <typename Lift>
    auto
    radial_displacement(const Lift &lift) const
    {
      return wall_.radial_displacement(lift);
    }

    auto
    multiplier_field() const
    {
      return wall_.multiplier_field();
    }

    using ExternalPressureProvider =
      typename WallObservable::ExternalPressureProvider;

    ExternalPressureProvider
    make_external_pressure_provider(const VectorType &multiplier) const
    {
      const auto lambda_provider =
        wall_.make_external_pressure_provider(multiplier);
      return [lambda_provider](const auto &evaluation) {
        return -lambda_provider(evaluation);
      };
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
      VectorType centerline_owned_values;
      centerline_owned_values.reinit(centerline_owned,
                                     wall_.mpi_communicator());
      centerline_owned_values = 0.;
      for (const auto index : multiplier_storage.locally_owned_elements())
        centerline_owned_values[wall_.area_dof_numbers()[index]] =
          multiplier_storage[index];
      centerline_owned_values.compress(dealii::VectorOperation::insert);
      VectorType centerline_multiplier;
      centerline_multiplier.reinit(centerline_owned,
                                   centerline_relevant,
                                   wall_.mpi_communicator());
      centerline_multiplier = centerline_owned_values;
      centerline_multiplier.update_ghost_values();
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

    static constexpr bool flow_pressure_feedback_is_implemented = true;

  public:
    dealii::PackagedOperation<VectorType>
    flow_pressure_residual(const EvaluationContext<VectorType> &context,
                           const FieldId                        flow_state,
                           const FieldId multiplier) const
    {
      const auto *interaction = this;
      const auto *flow        = &context.state(flow_state);
      const auto  provider =
        make_external_pressure_provider(context.state(multiplier));
      const double                          time = context.time();
      dealii::PackagedOperation<VectorType> result;
      result.reinit_vector = [this](VectorType &vector, const bool omit) {
        wall_.problem().reinit_state(vector);
        if (!omit)
          vector = 0.;
      };
      result.apply = [interaction, flow, provider, time](VectorType &value) {
        value = 0.;
        interaction->wall_.problem().add_external_pressure_residual(time,
                                                                    *flow,
                                                                    provider,
                                                                    value);
      };
      result.apply_add =
        [interaction, flow, provider, time](VectorType &value) {
          VectorType contribution;
          interaction->wall_.problem().reinit_state(contribution);
          interaction->wall_.problem().add_external_pressure_residual(
            time, *flow, provider, contribution);
          value += contribution;
        };
      return result;
    }

    dealii::LinearOperator<VectorType, VectorType>
    flow_pressure_jacobian(const EvaluationContext<VectorType> &context,
                           const FieldId                        flow_state,
                           const FieldId multiplier) const
    {
      (void)multiplier;
      const auto *interaction  = this;
      const auto  flow         = context.state(flow_state);
      const auto  time         = context.time();
      const auto  owned        = multiplier_locally_owned_dofs();
      const auto  relevant     = multiplier_locally_relevant_dofs();
      const auto  communicator = wall_.mpi_communicator();
      dealii::LinearOperator<VectorType, VectorType> result;
      result.reinit_range_vector =
        [problem = &wall_.problem()](VectorType &vector, const bool omit) {
          problem->reinit_state(vector);
          if (!omit)
            vector = 0.;
        };
      result.reinit_domain_vector =
        [owned, relevant, communicator](VectorType &vector, const bool omit) {
          vector.reinit(owned, relevant, communicator);
          if (!omit)
            vector = 0.;
        };
      result.vmult = [interaction, flow, time](VectorType       &destination,
                                               const VectorType &direction) {
        destination = 0.;
        const auto provider =
          interaction->make_external_pressure_provider(direction);
        interaction->wall_.problem().add_external_pressure_residual(
          time, flow, provider, destination);
      };
      result.vmult_add =
        [interaction, flow, time](VectorType       &destination,
                                  const VectorType &direction) {
          const auto provider =
            interaction->make_external_pressure_provider(direction);
          interaction->wall_.problem().add_external_pressure_residual(
            time, flow, provider, destination);
        };
      return result;
    }

  private:
    const WallObservable &wall_;
    mutable VectorType    multiplier_storage;
    bool                  multiplier_initialized_ = false;
    mutable std::vector<std::pair<double, std::string>> output_records_;
  };

  template <typename Builder, typename SolidField, typename WallObservable>
  ConstraintFields
  contribute(Builder &builder,
             const MetricFlowXVesselWallConstraint<SolidField, WallObservable>
                          &interaction,
             const FieldId flow_state,
             const FieldId multiplier)
  {
    builder.preconditioner(
      flow_state, [](const auto &matrix, const auto &reinit_vector) {
        return make_amg_preconditioner<
          typename WallObservable::StateType,
          typename MetricFlowXVesselWallConstraint<SolidField,
                                                   WallObservable>::MatrixType>(
          matrix, reinit_vector);
      });
    builder.term(flow_state, "vessel-wall-pressure-feedback")
      .residual([&interaction, flow_state, multiplier](const auto &context) {
        return interaction.flow_pressure_residual(context,
                                                  flow_state,
                                                  multiplier);
      })
      .state(multiplier,
             typename Builder::Model::OperatorFactory(
               [&interaction, flow_state, multiplier](const auto &context) {
                 return interaction.flow_pressure_jacobian(context,
                                                           flow_state,
                                                           multiplier);
               }));
    return {multiplier};
  }
} // namespace ImmersX

#endif // immersx_metric_flow_x_vessel_wall_constraint_h
