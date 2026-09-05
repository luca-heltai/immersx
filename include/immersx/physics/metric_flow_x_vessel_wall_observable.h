// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_metric_flow_x_vessel_wall_observable_h
#define immersx_metric_flow_x_vessel_wall_observable_h

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <deal.II/base/exceptions.h>
#  include <deal.II/base/quadrature.h>
#  include <deal.II/base/tensor.h>
#  include <deal.II/base/utilities.h>

#  include <deal.II/fe/fe_values.h>
#  include <deal.II/fe/fe_values_extractors.h>

#  include <immersx/algebra/linear_algebra.h>
#  include <immersx/core/fe_space.h>
#  include <immersx/core/observable.h>
#  include <immersx/core/state.h>
#  include <immersx/coupling/detail/coupling_point.h>
#  include <immersx/coupling/tensor_product_lift.h>
#  include <immersx/physics/metric_flow_x.h>

#  include <algorithm>
#  include <array>
#  include <cmath>
#  include <cstdint>
#  include <limits>
#  include <map>
#  include <memory>
#  include <string>
#  include <utility>
#  include <vector>

namespace ImmersX
{
  /**
   * Nonlinear radial displacement observable for a MetricFlowX vessel wall.
   *
   * The source is the existing mixed MetricFlowX state Field.  Its Area
   * component is sampled on the cells owned by the BloodFlowSystem and lifted
   * to the vessel surface by TensorProductLiftSupport.  The reference
   * cross-section is configured with the two vector modes 3 and 7 and the
   * same scalar coefficient at both modes.  The selected, orthogonalized
   * modes determine the radial direction; the direction is normalized so that
   * the represented observable is exactly
   *
   *   g(A) = (sqrt(A/pi) - sqrt(A0/pi)) n.
   *
   * `A0` is read from the MetricFlowX vessel metadata associated with each
   * centerline cell.  No second mesh, Area Field, or resting-area parameter is
   * owned here.
   */
  class MetricFlowXAreaRadialDisplacementObservable
  {
  public:
    static constexpr unsigned int support_dimension        = 2;
    static constexpr unsigned int ambient_dimension        = 3;
    static constexpr unsigned int representative_dimension = 1;

    using Problem           = ::MetricFlowX::BloodFlowSystem<1, 3>;
    using StateType         = ::MetricFlowX::VectorType;
    using Value             = dealii::Tensor<1, 3>;
    using value_type        = std::vector<Value>;
    using state_type        = StateType;
    using Operator          = dealii::LinearOperator<value_type, state_type>;
    using ExtractorType     = dealii::FEValuesExtractors::Scalar;
    using TriangulationType = dealii::parallel::TriangulationBase<1, 3>;
    using DoFHandlerType    = dealii::DoFHandler<1, 3>;
    using Lift              = TensorProductLift<1, 2, 3, 3>;
    using Support           = TensorProductLiftSupport<1, 2, 3, 3>;

    struct ObservableEvaluationRequest
    {};

    struct WallPoint : detail::CouplingPoint<3, double>
    {
      double                                       a0 = 0.;
      dealii::CellId                               cell_id;
      std::vector<double>                          area_basis_values;
      std::vector<dealii::types::global_dof_index> multiplier_dof_indices;
      dealii::Tensor<1, 3>                         normal;
    };

    using ExternalPressureProvider = typename Problem::ExternalPressureProvider;

    MetricFlowXAreaRadialDisplacementObservable(
      const Problem                                         &problem,
      const Field<1, 3, dealii::FEValuesExtractors::Scalar> &area,
      const dealii::IndexSet                                &area_components,
      const Lift                                            &lift,
      const double mode_tolerance = 1.e-10)
      : problem_(problem)
      , area_(area)
      , area_components_(area_components)
      , support_(lift.parameters())
      , mode_tolerance_(mode_tolerance)
      , area_owned_(
          make_component_subset(problem.locally_owned_dofs(), area_components_))
      , area_relevant_(make_component_subset(problem.locally_relevant_dofs(),
                                             area_components_))
    {
      initialize_multiplier_space();
      AssertThrow(area_.field_id().is_valid(),
                  dealii::ExcMessage(
                    "The vessel-wall observable needs a valid Area Field."));
      AssertThrow(area_components_.size() ==
                    problem.locally_owned_dofs().size(),
                  dealii::ExcMessage(
                    "The Area component set does not match the MetricFlowX "
                    "state space."));
      const std::vector<unsigned int> radial_modes{3u, 7u};
      AssertThrow(support_.selected_modes() == radial_modes,
                  dealii::ExcMessage(
                    "The radial vessel-wall observable requires cross-"
                    "section modes 3 and 7 in that order."));
      build_points();
    }

    FieldId
    source() const
    {
      return area_.field_id();
    }

    std::vector<FieldId>
    dependencies() const
    {
      return {source()};
    }

    const Problem &
    problem() const
    {
      return problem_;
    }

    const Field<1, 3, dealii::FEValuesExtractors::Scalar> &
    area_view() const
    {
      return area_;
    }

    const Support &
    support() const
    {
      return support_;
    }

    const std::vector<WallPoint> &
    points() const
    {
      return points_;
    }

    /** The two modal coefficients used for one scalar radius increment. */
    std::array<double, 2>
    mode_coefficients(const double delta_radius) const
    {
      return {delta_radius, delta_radius};
    }

    /** Return the current multiplier as a native MetricFlowX pressure field.
     *
     * The returned provider owns a relevant ghosted copy of the supplied
     * candidate vector.  It evaluates the scalar FE field on the incident
     * centerline cell identified by `PressureEvaluationPoint::cell_id`; no
     * global point search or global vector gather is performed.
     */
    ExternalPressureProvider
    make_external_pressure_provider(const StateType &multiplier) const
    {
      auto values = std::make_shared<StateType>();
      values->reinit(multiplier_owned_,
                     multiplier_relevant_,
                     mpi_communicator());
      *values = multiplier;
      values->update_ghost_values();

      return [this, values](
               const typename Problem::PressureEvaluationPoint &evaluation) {
        return evaluate_multiplier(*values, evaluation);
      };
    }

    const TriangulationType &
    triangulation() const
    {
      return problem_.triangulation();
    }

    const DoFHandlerType &
    dof_handler() const
    {
      return problem_.dof_handler();
    }

    const dealii::FiniteElement<1, 3> &
    finite_element() const
    {
      return problem_.finite_element();
    }

    const dealii::Mapping<1, 3> &
    mapping() const
    {
      return dealii::StaticMappingQ1<1, 3>::mapping;
    }

    const dealii::IndexSet &
    locally_owned_dofs() const
    {
      return area_owned_;
    }

    const dealii::IndexSet &
    locally_relevant_dofs() const
    {
      return area_relevant_;
    }

    const dealii::IndexSet &
    multiplier_locally_owned_dofs() const
    {
      return multiplier_owned_;
    }

    const dealii::IndexSet &
    multiplier_locally_relevant_dofs() const
    {
      return multiplier_relevant_;
    }

    const std::vector<dealii::types::global_dof_index> &
    area_dof_numbers() const
    {
      return area_dof_numbers_;
    }

    const dealii::AffineConstraints<double> &
    constraints() const
    {
      return problem_.constraints();
    }

    ExtractorType
    extractor() const
    {
      return problem_.area_extractor();
    }

    MPI_Comm
    mpi_communicator() const
    {
      return problem_.mpi_communicator();
    }

    unsigned int
    n_dofs_per_cell() const
    {
      return n_area_dofs_per_cell_;
    }

    /**
     * Return the retained vessel-wall points and the vector basis
     * `n * psi_A` used by both the nonlinear observable and its derivative.
     */
    const std::vector<WallPoint> &
    locally_owned_quadrature_points(
      const dealii::Quadrature<support_dimension> &) const
    {
      return points_;
    }

    value_type
    evaluate(const EvaluationContext<state_type> &context,
             const ObservableEvaluationRequest & = {}) const
    {
      const auto relevant = relevant_state(context.state(source()));
      value_type result(points_.size());
      for (std::size_t q = 0; q < points_.size(); ++q)
        {
          const double area = evaluate_stencil(relevant, points_[q]);
          result[q] = points_[q].normal * delta_radius(area, points_[q].a0);
        }
      return result;
    }

    Operator
    linearize(const EvaluationContext<state_type> &context,
              const ObservableEvaluationRequest & = {}) const
    {
      return linearize(context, source());
    }

    Operator
    linearize(const EvaluationContext<state_type> &context,
              const FieldId                        field,
              const ObservableEvaluationRequest & = {}) const
    {
      AssertThrow(field == source(),
                  dealii::ExcMessage(
                    "The requested field is not a vessel-wall dependency."));

      const auto          reference    = context.state(source());
      const auto          relevant     = problem_.locally_relevant_dofs();
      const auto          owned        = problem_.locally_owned_dofs();
      const auto          communicator = problem_.mpi_communicator();
      std::vector<double> derivative(points_.size());
      const auto          current = relevant_state(reference);
      for (std::size_t q = 0; q < points_.size(); ++q)
        {
          const double area = evaluate_stencil(current, points_[q]);
          derivative[q]     = radius_derivative(area);
        }
      const auto points = points_;

      Operator result;
      result.reinit_range_vector = [n = points.size()](value_type &vector,
                                                       const bool) {
        vector.resize(n);
        for (auto &value : vector)
          value = 0.;
      };
      result.reinit_domain_vector = [reference](state_type &vector,
                                                const bool  omit) {
        vector.reinit(reference, omit);
      };
      result.vmult = [points, derivative, owned, relevant, communicator](
                       value_type &destination, const state_type &direction) {
        const auto source =
          make_relevant(direction, owned, relevant, communicator);
        destination.resize(points.size());
        for (std::size_t q = 0; q < points.size(); ++q)
          destination[q] =
            points[q].normal *
            (derivative[q] * evaluate_stencil(source, points[q]));
      };
      result.vmult_add = [points, derivative, owned, relevant, communicator](
                           value_type       &destination,
                           const state_type &direction) {
        const auto source =
          make_relevant(direction, owned, relevant, communicator);
        for (std::size_t q = 0; q < points.size(); ++q)
          destination[q] +=
            points[q].normal *
            (derivative[q] * evaluate_stencil(source, points[q]));
      };
      result.Tvmult = [points, derivative](state_type       &destination,
                                           const value_type &values) {
        destination = 0.;
        for (std::size_t q = 0; q < points.size(); ++q)
          {
            const double radial_value =
              derivative[q] * (points[q].normal * values[q]);
            for (unsigned int i = 0; i < points[q].dof_indices.size(); ++i)
              destination[points[q].dof_indices[i]] +=
                radial_value * points[q].area_basis_values[i];
          }
        destination.compress(dealii::VectorOperation::add);
      };
      result.Tvmult_add = [points, derivative](state_type       &destination,
                                               const value_type &values) {
        for (std::size_t q = 0; q < points.size(); ++q)
          {
            const double radial_value =
              derivative[q] * (points[q].normal * values[q]);
            for (unsigned int i = 0; i < points[q].dof_indices.size(); ++i)
              destination[points[q].dof_indices[i]] +=
                radial_value * points[q].area_basis_values[i];
          }
        destination.compress(dealii::VectorOperation::add);
      };
      return result;
    }

    double
    delta_radius(const double area, const double a0) const
    {
      AssertThrow(std::isfinite(area) && area > 0.,
                  dealii::ExcMessage(
                    "MetricFlowX Area must be finite and positive."));
      AssertThrow(std::isfinite(a0) && a0 > 0.,
                  dealii::ExcMessage(
                    "MetricFlowX resting Area must be finite and positive."));
      return std::sqrt(area / dealii::numbers::PI) -
             std::sqrt(a0 / dealii::numbers::PI);
    }

    double
    radius_derivative(const double area) const
    {
      AssertThrow(std::isfinite(area) && area > 0.,
                  dealii::ExcMessage(
                    "MetricFlowX Area must be finite and positive."));
      return 1. / (2. * std::sqrt(dealii::numbers::PI * area));
    }

  private:
    struct CellData
    {
      dealii::Point<3>                                   first_vertex;
      dealii::Point<3>                                   last_vertex;
      std::vector<std::pair<unsigned int, unsigned int>> area_basis;
    };

    static dealii::IndexSet
    make_component_subset(const dealii::IndexSet &source,
                          const dealii::IndexSet &component)
    {
      dealii::IndexSet result(source.size());
      for (const auto index : source)
        if (component.is_element(index))
          result.add_index(index);
      result.compress();
      return result;
    }

    void
    initialize_multiplier_space()
    {
      using GlobalIndex = dealii::types::global_dof_index;
      std::vector<GlobalIndex> local_area_dofs(area_owned_.begin(),
                                               area_owned_.end());
      const auto               batches =
        dealii::Utilities::MPI::all_gather(mpi_communicator(), local_area_dofs);
      area_dof_numbers_.clear();
      for (const auto &batch : batches)
        area_dof_numbers_.insert(area_dof_numbers_.end(),
                                 batch.begin(),
                                 batch.end());
      std::sort(area_dof_numbers_.begin(), area_dof_numbers_.end());
      area_dof_numbers_.erase(std::unique(area_dof_numbers_.begin(),
                                          area_dof_numbers_.end()),
                              area_dof_numbers_.end());
      area_to_multiplier_.clear();
      for (unsigned int i = 0; i < area_dof_numbers_.size(); ++i)
        area_to_multiplier_[area_dof_numbers_[i]] = i;

      multiplier_owned_.set_size(area_dof_numbers_.size());
      for (const auto index : area_owned_)
        multiplier_owned_.add_index(area_to_multiplier_.at(index));
      multiplier_owned_.compress();
      multiplier_relevant_.set_size(area_dof_numbers_.size());
      for (const auto index : area_relevant_)
        multiplier_relevant_.add_index(area_to_multiplier_.at(index));
      multiplier_relevant_.compress();
    }

    static StateType
    make_relevant(const StateType        &source,
                  const dealii::IndexSet &owned,
                  const dealii::IndexSet &relevant,
                  const MPI_Comm          communicator)
    {
      StateType result;
      result.reinit(owned, relevant, communicator);
      result = source;
      result.update_ghost_values();
      return result;
    }

    StateType
    relevant_state(const StateType &source) const
    {
      return make_relevant(source,
                           problem_.locally_owned_dofs(),
                           problem_.locally_relevant_dofs(),
                           problem_.mpi_communicator());
    }

    static double
    evaluate_stencil(const StateType &state, const WallPoint &point)
    {
      double result = 0.;
      for (unsigned int i = 0; i < point.dof_indices.size(); ++i)
        result += point.area_basis_values[i] * state[point.dof_indices[i]];
      return result;
    }

    double
    evaluate_multiplier(
      const StateType                                 &multiplier,
      const typename Problem::PressureEvaluationPoint &evaluation) const
    {
      const auto cell = cell_data_.find(evaluation.cell_id);
      AssertThrow(cell != cell_data_.end(),
                  dealii::ExcMessage(
                    "MetricFlowX pressure evaluation has no incident cell "
                    "metadata."));

      const auto tangent = cell->second.last_vertex - cell->second.first_vertex;
      const double tangent_squared = tangent * tangent;
      AssertThrow(tangent_squared > 0.,
                  dealii::ExcMessage("A centerline cell has zero length."));
      const auto   displacement = evaluation.point - cell->second.first_vertex;
      const double coordinate =
        std::clamp((displacement * tangent) / tangent_squared, 0., 1.);
      const dealii::Point<1> reference_point(coordinate);

      double value = 0.;
      for (const auto &[local_dof, multiplier_dof] : cell->second.area_basis)
        value +=
          problem_.finite_element().shape_value_component(local_dof,
                                                          reference_point,
                                                          0) *
          multiplier[multiplier_dof];
      return value;
    }

    void
    build_points()
    {
      AssertThrow(support_.reference_cross_section().n_selected_basis() == 2,
                  dealii::ExcMessage(
                    "The radial cross-section must contain exactly two "
                    "selected modes."));
      const auto            &quadrature = support_.representative_quadrature();
      dealii::FEValues<1, 3> fe_values(dealii::StaticMappingQ1<1, 3>::mapping,
                                       problem_.finite_element(),
                                       quadrature,
                                       dealii::update_values |
                                         dealii::update_quadrature_points |
                                         dealii::update_JxW_values);
      const unsigned int n_dofs = problem_.finite_element().n_dofs_per_cell();
      std::vector<dealii::types::global_dof_index> local_dofs(n_dofs);

      for (const auto &cell : problem_.dof_handler().active_cell_iterators())
        if (!cell->is_artificial())
          {
            cell->get_dof_indices(local_dofs);
            std::vector<std::pair<unsigned int, unsigned int>> area_basis;
            for (unsigned int i = 0; i < n_dofs; ++i)
              if (area_components_.is_element(local_dofs[i]))
                area_basis.emplace_back(i,
                                        area_to_multiplier_.at(local_dofs[i]));
            cell_data_.emplace(cell->id(),
                               CellData{cell->vertex(0),
                                        cell->vertex(1),
                                        std::move(area_basis)});

            if (!cell->is_locally_owned())
              continue;
            fe_values.reinit(cell);
            const double a0 =
              problem_.vessel_properties(cell->material_id()).a0;
            AssertThrow(std::isfinite(a0) && a0 > 0.,
                        dealii::ExcMessage(
                          "MetricFlowX vessel metadata contains an invalid "
                          "resting Area."));
            for (const auto q : fe_values.quadrature_point_indices())
              {
                dealii::Tensor<1, 3> tangent =
                  cell->vertex(1) - cell->vertex(0);
                const auto transformed =
                  support_.transform(fe_values.quadrature_point(q),
                                     tangent,
                                     fe_values.JxW(q),
                                     std::sqrt(a0 / dealii::numbers::PI),
                                     q,
                                     {});
                for (unsigned int section_q = 0; section_q < transformed.size();
                     ++section_q)
                  {
                    const auto &lifted = transformed[section_q];
                    WallPoint   point;
                    point.point                = lifted.point;
                    point.representative_point = lifted.representative_point;
                    point.weight               = lifted.weight;
                    point.cell_id              = cell->id();
                    point.source_entity_id = cell->global_active_cell_index();
                    point.representative_qpoint = q;
                    point.section_qpoint        = section_q;
                    point.stable_id =
                      static_cast<std::uint64_t>(point.source_entity_id) *
                        quadrature.size() * transformed.size() +
                      q * transformed.size() + section_q;
                    const auto radial =
                      point.point - point.representative_point;
                    AssertThrow(radial.norm() > mode_tolerance_,
                                dealii::ExcMessage(
                                  "A vessel-wall cross-section point has no "
                                  "radial direction."));
                    point.normal = radial / radial.norm();
                    dealii::Tensor<1, 3> modal_direction;
                    modal_direction = 0.;
                    for (unsigned int mode = 0; mode < 2; ++mode)
                      for (unsigned int component = 0; component < 3;
                           ++component)
                        modal_direction[component] +=
                          lifted.mode_values[mode * 3 + component];
                    AssertThrow(modal_direction.norm() > mode_tolerance_,
                                dealii::ExcMessage(
                                  "Selected vessel-wall modes do not define "
                                  "a radial direction."));
                    AssertThrow(std::abs(modal_direction * point.normal) /
                                    modal_direction.norm() >
                                  1. - 1.e-8,
                                dealii::ExcMessage(
                                  "Selected modes 3 and 7 are not radial for "
                                  "the current ReferenceCrossSection."));
                    point.a0 = a0;
                    point.area_basis_values.clear();
                    for (unsigned int i = 0; i < n_dofs; ++i)
                      if (area_components_.is_element(local_dofs[i]))
                        {
                          const double basis =
                            fe_values[problem_.area_extractor()].value(i, q);
                          point.dof_indices.push_back(local_dofs[i]);
                          point.multiplier_dof_indices.push_back(
                            area_to_multiplier_.at(local_dofs[i]));
                          point.area_basis_values.push_back(basis);
                        }
                    points_.push_back(std::move(point));
                  }
              }
          }
      const auto n_local_area_dofs =
        points_.empty() ?
          0u :
          static_cast<unsigned int>(points_.front().dof_indices.size());
      n_area_dofs_per_cell_ =
        dealii::Utilities::MPI::max(n_local_area_dofs, mpi_communicator());
    }

    const Problem                                        &problem_;
    const Field<1, 3, dealii::FEValuesExtractors::Scalar> area_;
    const dealii::IndexSet                                area_components_;
    Support                                               support_;
    const double                                          mode_tolerance_;
    dealii::IndexSet                                      area_owned_;
    dealii::IndexSet                                      area_relevant_;
    dealii::IndexSet                                      multiplier_owned_;
    dealii::IndexSet                                      multiplier_relevant_;
    std::vector<dealii::types::global_dof_index>          area_dof_numbers_;
    std::map<dealii::types::global_dof_index, dealii::types::global_dof_index>
                                       area_to_multiplier_;
    std::map<dealii::CellId, CellData> cell_data_;
    std::vector<WallPoint>             points_;
    unsigned int                       n_area_dofs_per_cell_ = 0;
  };
} // namespace ImmersX

#endif // IMMERSX_WITH_METRIC_FLOW_X

#endif // immersx_metric_flow_x_vessel_wall_observable_h
