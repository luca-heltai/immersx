// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_observable_lift_h
#define immersx_observable_lift_h

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/linear_operator.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/observable.h>
#include <immersx/coupling/detail/coupling_point.h>
#include <immersx/coupling/detail/fe_stencil.h>
#include <immersx/coupling/tensor_product_lift.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace ImmersX
{
  /**
   * An Observable sampled on a tensor-product lift.
   *
   * This is deliberately a small value-space adapter.  It retains the source
   * FE field and its stencils, while the coupling backend owns point search
   * and redistribution when the quantity is paired with a target field.
   */
  template <typename SourceObservable,
            int reduced_dim,
            int surface_dim,
            int spacedim,
            int n_components = 1>
  class LiftedObservable
  {
    static_assert(
      reduced_dim >= 1 && reduced_dim < surface_dim &&
        surface_dim <= spacedim && spacedim <= 3,
      "Observable lifting requires 1 <= dim < support <= spacedim <= 3.");
    static_assert(std::is_arithmetic_v<typename SourceObservable::value_type>,
                  "The initial Observable lift supports scalar quantities.");

  public:
    static constexpr unsigned int support_dimension        = surface_dim;
    static constexpr unsigned int ambient_dimension        = spacedim;
    static constexpr unsigned int representative_dimension = reduced_dim;

    using source_field_type = typename SourceObservable::source_field_type;
    static constexpr bool vector_lift =
      n_components == spacedim && n_components > 1;
    using value_type =
      std::conditional_t<vector_lift,
                         std::vector<dealii::Tensor<1, spacedim>>,
                         ImmersXLA::MPI::Vector>;
    using point_value_type =
      std::conditional_t<vector_lift, dealii::Tensor<1, spacedim>, double>;
    using state_type = ImmersXLA::MPI::Vector;
    using Point      = detail::CouplingPoint<spacedim, double>;
    using Lift =
      TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>;
    using Support  = TensorProductLiftSupport<reduced_dim,
                                             surface_dim,
                                             spacedim,
                                             n_components>;
    using Operator = dealii::LinearOperator<value_type, state_type>;

    LiftedObservable(SourceObservable source, const Lift &lift)
      : source_(std::move(source))
      , support_(std::make_shared<Support>(lift.parameters()))
    {
      build_points();
    }

    LiftedObservable(SourceObservable                   source,
                     const Lift                        &lift,
                     SourceThicknessEvaluator<spacedim> thickness_evaluator)
      : source_(std::move(source))
      , support_(std::make_shared<Support>(lift.parameters()))
    {
      support_->set_thickness_evaluator(std::move(thickness_evaluator));
      build_points();
    }

    template <typename ThicknessObservable,
              std::enable_if_t<
                detail::is_observable<std::decay_t<ThicknessObservable>>::value,
                int> = 0>
    LiftedObservable(SourceObservable    source,
                     const Lift         &lift,
                     ThicknessObservable thickness)
      : source_(std::move(source))
      , support_(std::make_shared<Support>(lift.parameters()))
    {
      AssertThrow(thickness.dependencies().empty(),
                  dealii::ExcMessage(
                    "Lift thickness Observables must be frozen."));
      thickness_evaluator_ = [thickness](const dealii::CellId          &cell_id,
                                         const dealii::Point<spacedim> &point,
                                         const double                   time) {
        return evaluate_frozen_expression(
          thickness,
          cell_id,
          point,
          time,
          [](const auto &input) -> const auto & {
            return input.template frozen_values<state_type>();
          });
      };
      build_points();
    }

    const source_field_type &
    source() const
    {
      return source_.source();
    }

    FieldId
    source_field() const
    {
      return source_.source_field();
    }

    const source_field_type &
    source_for(const FieldId field) const
    {
      return source_.source_for(field);
    }

    static constexpr unsigned int
    dimension()
    {
      return source_field_type::dimension();
    }

    static constexpr unsigned int
    spacedimension()
    {
      return source_field_type::spacedimension();
    }

    static constexpr bool
    is_linear()
    {
      return false;
    }

    const std::vector<FieldId> &
    dependencies() const
    {
      return source_.dependencies();
    }

    bool
    is_frozen() const
    {
      return source_.is_frozen();
    }

    template <typename VectorType>
    const VectorType &
    frozen_values() const
    {
      return source_.template frozen_values<VectorType>();
    }

    double
    scale() const
    {
      return source_.scale();
    }

    LiftedObservable
    with_id(const FieldId id) const
    {
      LiftedObservable result = *this;
      result.source_          = source_.with_id(id);
      return result;
    }

    /** Select one cross-section basis function without changing its amplitude.
     *
     * Separate scalar fields can share a representative FE space while each
     * carries an independent modal coefficient. The index is a slot in the
     * configured selected coefficients, not a global basis index.
     */
    LiftedObservable
    mode(const unsigned int slot) const
    {
      AssertThrow(slot < support_->reference_cross_section().n_selected_basis(),
                  dealii::ExcMessage("Invalid selected-mode slot."));
      LiftedObservable result = *this;
      result.mode_slot_       = slot;
      result.points_.clear();
      result.point_indices_.clear();
      result.point_owned_.clear();
      result.point_relevant_.clear();
      result.build_points();
      return result;
    }

    const std::vector<Point> &
    lifted_points() const
    {
      return points_;
    }

    const std::vector<Point> &
    locally_owned_quadrature_points(
      const dealii::Quadrature<surface_dim> &) const
    {
      return points_;
    }

    const dealii::IndexSet &
    locally_owned_points() const
    {
      return point_owned_;
    }

    const dealii::IndexSet &
    locally_relevant_points() const
    {
      return point_relevant_;
    }

    dealii::types::global_dof_index
    point_index(const std::size_t local_point) const
    {
      AssertIndexRange(local_point, point_indices_.size());
      return point_indices_[local_point];
    }

    MPI_Comm
    mpi_communicator() const
    {
      return source().space().mpi_communicator();
    }

    template <typename Context>
    value_type
    evaluate(const Context &context) const
    {
      value_type result;
      if constexpr (vector_lift)
        {
          result.resize(points_.size());
          for (std::size_t q = 0; q < points_.size(); ++q)
            result[q] = evaluate_point(context, q);
        }
      else
        {
          result.reinit(point_owned_, point_relevant_, mpi_communicator());
          if constexpr (detail::is_transformed_observable<
                          SourceObservable>::value)
            for (std::size_t q = 0; q < points_.size(); ++q)
              result[point_indices_[q]] =
                mode_value(points_[q]) * composed_value(context, points_[q]);
          else if (is_frozen())
            detail::apply_stencils(
              points_,
              point_indices_,
              source().locally_owned_dofs(),
              source().locally_relevant_dofs(),
              point_owned_,
              point_relevant_,
              &source().constraints(),
              mpi_communicator(),
              frozen_values<state_type>(),
              result,
              [](const auto &point) -> const auto & {
                return point.dof_indices;
              },
              [](const auto &point, const unsigned int i) {
                return point.basis_values[i];
              },
              false,
              false);
          else
            detail::apply_stencils(
              points_,
              point_indices_,
              source().locally_owned_dofs(),
              source().locally_relevant_dofs(),
              point_owned_,
              point_relevant_,
              &source().constraints(),
              mpi_communicator(),
              context.state(source_field()),
              result,
              [](const auto &point) -> const auto & {
                return point.dof_indices;
              },
              [](const auto &point, const unsigned int i) {
                return point.basis_values[i];
              },
              false,
              false);
          result *= scale();
        }
      return result;
    }

    template <typename Context>
    point_value_type
    evaluate_point(const Context &context, const std::size_t point) const
    {
      AssertIndexRange(point, points_.size());
      double scalar = 0.;
      if constexpr (detail::is_transformed_observable<SourceObservable>::value)
        scalar = composed_value(context, points_[point]);
      else
        scalar = primitive_value(context, points_[point]);
      if constexpr (vector_lift)
        return mode_vector(points_[point]) * (scale() * scalar);
      else
        return scale() * mode_value(points_[point]) * scalar;
    }

    template <typename Context>
    point_value_type
    linearize_point(const Context     &context,
                    const FieldId      field,
                    const std::size_t  point,
                    const unsigned int basis) const
    {
      AssertIndexRange(point, points_.size());
      double derivative = 0.;
      if constexpr (detail::is_transformed_observable<SourceObservable>::value)
        derivative = composed_basis_derivative(
          source_, context, field, points_[point], basis, context.time());
      else if (!source_.is_frozen() && field == source_field())
        derivative = points_[point].source_basis_values[basis];

      if constexpr (vector_lift)
        return mode_vector(points_[point]) * (scale() * derivative);
      else
        return scale() * mode_value(points_[point]) * derivative;
    }

    template <typename Context>
    Operator
    linearize(const Context &context, const FieldId field) const
    {
      AssertThrow(field == source_field(),
                  dealii::ExcMessage(
                    "The requested field is not a lift dependency."));
      if constexpr (vector_lift)
        return linearize_vector(context, field);
      else if constexpr (detail::is_transformed_observable<
                           SourceObservable>::value)
        return linearize_composed(context, field);
      else
        {
          const auto owned            = point_owned_;
          const auto relevant         = point_relevant_;
          const auto point_ids        = point_indices_;
          const auto points           = points_;
          const auto source_field_ref = source();
          const auto communicator     = mpi_communicator();
          const auto coefficient      = scale();

          Operator result;
          result.reinit_range_vector =
            [owned, relevant, communicator](value_type &vector,
                                            const bool  omit) {
              vector.reinit(owned, relevant, communicator);
              if (!omit)
                vector = 0.;
            };
          result.reinit_domain_vector = [source_field_ref,
                                         communicator](state_type &vector,
                                                       const bool  omit) {
            vector.reinit(source_field_ref.locally_owned_dofs(),
                          source_field_ref.locally_relevant_dofs(),
                          communicator);
            if (!omit)
              vector = 0.;
          };
          result.vmult = [points,
                          point_ids,
                          source_field_ref,
                          owned,
                          relevant,
                          communicator,
                          coefficient](value_type       &destination,
                                       const state_type &values) {
            detail::apply_stencils(
              points,
              point_ids,
              source_field_ref.locally_owned_dofs(),
              source_field_ref.locally_relevant_dofs(),
              owned,
              relevant,
              &source_field_ref.constraints(),
              communicator,
              values,
              destination,
              [](const auto &point) -> const auto & {
                return point.dof_indices;
              },
              [coefficient](const auto &point, const unsigned int i) {
                return coefficient * point.basis_values[i];
              },
              false,
              false);
          };
          result.vmult_add = [points,
                              point_ids,
                              source_field_ref,
                              owned,
                              relevant,
                              communicator,
                              coefficient](value_type       &destination,
                                           const state_type &values) {
            detail::apply_stencils(
              points,
              point_ids,
              source_field_ref.locally_owned_dofs(),
              source_field_ref.locally_relevant_dofs(),
              owned,
              relevant,
              &source_field_ref.constraints(),
              communicator,
              values,
              destination,
              [](const auto &point) -> const auto & {
                return point.dof_indices;
              },
              [coefficient](const auto &point, const unsigned int i) {
                return coefficient * point.basis_values[i];
              },
              false,
              true);
          };
          result.Tvmult =
            [points, point_ids, source_field_ref, communicator, coefficient](
              state_type &destination, const value_type &values) {
              detail::apply_stencils_transpose(
                points,
                point_ids,
                source_field_ref.locally_owned_dofs(),
                &source_field_ref.constraints(),
                communicator,
                values,
                destination,
                [](const auto &point) -> const auto & {
                  return point.dof_indices;
                },
                [coefficient](const auto &point, const unsigned int i) {
                  return coefficient * point.basis_values[i];
                },
                false);
            };
          result.Tvmult_add =
            [points, point_ids, source_field_ref, communicator, coefficient](
              state_type &destination, const value_type &values) {
              detail::apply_stencils_transpose(
                points,
                point_ids,
                source_field_ref.locally_owned_dofs(),
                &source_field_ref.constraints(),
                communicator,
                values,
                destination,
                [](const auto &point) -> const auto & {
                  return point.dof_indices;
                },
                [coefficient](const auto &point, const unsigned int i) {
                  return coefficient * point.basis_values[i];
                },
                true);
            };
          return result;
        }
    }

    template <typename Context>
    Operator
    linearize(const Context &context) const
    {
      return linearize(context, source_field());
    }

  private:
    template <typename ObservableType, typename CoefficientProvider>
    static auto
    primitive_value_at(const ObservableType &observable,
                       const Point          &point,
                       const double          time,
                       CoefficientProvider &&coefficients_for)
    {
      const auto &field = observable.source();
      const auto  tria_cell =
        field.dof_handler().get_triangulation().create_cell_iterator(
          point.source_cell_id);
      const auto cell = tria_cell->as_dof_handler_iterator(field.dof_handler());
      const auto reference_point =
        field.mapping().transform_real_to_unit_cell(cell,
                                                    point.representative_point);
      const dealii::Quadrature<reduced_dim> quadrature(
        std::vector<dealii::Point<reduced_dim>>{reference_point},
        std::vector<double>{1.});
      dealii::FEValues<reduced_dim, spacedim> values(
        field.mapping(),
        cell->get_fe(),
        quadrature,
        dealii::update_values | ObservableType::update_flags());
      values.reinit(cell);

      std::vector<dealii::types::global_dof_index> native_indices(
        cell->get_fe().n_dofs_per_cell());
      cell->get_dof_indices(native_indices);
      std::vector<dealii::types::global_dof_index> execution_indices;
      std::vector<unsigned int>                    positions;
      for (unsigned int i = 0; i < native_indices.size(); ++i)
        if (field.has_execution_index(native_indices[i]))
          {
            positions.push_back(i);
            execution_indices.push_back(
              field.execution_index(native_indices[i]));
          }
      const auto         &coefficients = coefficients_for(observable);
      std::vector<double> local_values(execution_indices.size());
      field.constraints().get_dof_values(coefficients,
                                         execution_indices.begin(),
                                         local_values.begin(),
                                         local_values.end());

      using Value = typename ObservableType::value_type;
      Value       result{};
      const auto &view = values[field.extractor()];
      for (unsigned int i = 0; i < positions.size(); ++i)
        result +=
          local_values[i] * observable.operation()(view, positions[i], 0);
      (void)time;
      return observable.scale() * result;
    }

    template <typename ObservableType>
    static auto
    primitive_direction_at(const ObservableType &observable,
                           const Point          &point,
                           const state_type     &direction)
    {
      const auto &field = observable.source();
      const auto  tria_cell =
        field.dof_handler().get_triangulation().create_cell_iterator(
          point.source_cell_id);
      const auto cell = tria_cell->as_dof_handler_iterator(field.dof_handler());
      const auto reference_point =
        field.mapping().transform_real_to_unit_cell(cell,
                                                    point.representative_point);
      const dealii::Quadrature<reduced_dim> quadrature(
        std::vector<dealii::Point<reduced_dim>>{reference_point},
        std::vector<double>{1.});
      dealii::FEValues<reduced_dim, spacedim> values(
        field.mapping(),
        cell->get_fe(),
        quadrature,
        dealii::update_values | ObservableType::update_flags());
      values.reinit(cell);

      std::vector<dealii::types::global_dof_index> native_indices(
        cell->get_fe().n_dofs_per_cell());
      cell->get_dof_indices(native_indices);
      std::vector<dealii::types::global_dof_index> execution_indices;
      std::vector<unsigned int>                    positions;
      for (unsigned int i = 0; i < native_indices.size(); ++i)
        if (field.has_execution_index(native_indices[i]))
          {
            positions.push_back(i);
            execution_indices.push_back(
              field.execution_index(native_indices[i]));
          }
      std::vector<double> local_values(execution_indices.size());
      field.constraints().get_dof_values(direction,
                                         execution_indices.begin(),
                                         local_values.begin(),
                                         local_values.end());
      using Value = typename ObservableType::value_type;
      Value       result{};
      const auto &view = values[field.extractor()];
      for (unsigned int i = 0; i < positions.size(); ++i)
        result +=
          local_values[i] * observable.operation()(view, positions[i], 0);
      return result;
    }

    template <typename ObservableType>
    static auto
    primitive_basis_at(const ObservableType &observable,
                       const Point          &point,
                       const unsigned int    basis)
    {
      const auto &field = observable.source();
      const auto  tria_cell =
        field.dof_handler().get_triangulation().create_cell_iterator(
          point.source_cell_id);
      const auto cell = tria_cell->as_dof_handler_iterator(field.dof_handler());
      AssertIndexRange(basis, cell->get_fe().n_dofs_per_cell());
      std::vector<dealii::types::global_dof_index> native_indices(
        cell->get_fe().n_dofs_per_cell());
      cell->get_dof_indices(native_indices);
      if (!field.has_execution_index(native_indices[basis]))
        return typename ObservableType::value_type{};
      const auto reference_point =
        field.mapping().transform_real_to_unit_cell(cell,
                                                    point.representative_point);
      const dealii::Quadrature<reduced_dim> quadrature(
        std::vector<dealii::Point<reduced_dim>>{reference_point},
        std::vector<double>{1.});
      dealii::FEValues<reduced_dim, spacedim> values(
        field.mapping(),
        cell->get_fe(),
        quadrature,
        dealii::update_values | ObservableType::update_flags());
      values.reinit(cell);
      const auto &view = values[field.extractor()];
      return observable.operation()(view, basis, 0);
    }

    template <typename ObservableType, typename CoefficientProvider>
    static auto
    evaluate_frozen_expression(const ObservableType          &observable,
                               const dealii::CellId          &cell_id,
                               const dealii::Point<spacedim> &point,
                               const double                   time,
                               CoefficientProvider          &&coefficients_for)
    {
      const auto evaluator = [&](const auto &input, const auto &, double) {
        Point source_point;
        source_point.source_cell_id       = cell_id;
        source_point.representative_point = point;
        return primitive_value_at(input, source_point, time, coefficients_for);
      };
      return evaluate_observable_input(observable, point, time, evaluator);
    }

    template <typename Context>
    double
    primitive_value(const Context &context, const Point &point) const
    {
      const auto &coefficients =
        source_.is_frozen() ? source_.template frozen_values<state_type>() :
                              context.state(source_.source_field());
      const auto          indices = execution_indices(source_.source(), point);
      std::vector<double> local(point.dof_indices.size());
      source_.source().constraints().get_dof_values(coefficients,
                                                    indices.begin(),
                                                    local.begin(),
                                                    local.end());
      double result = 0.;
      for (unsigned int i = 0; i < local.size(); ++i)
        result += local[i] * point.source_basis_values[i];
      return result;
    }

    static dealii::Tensor<1, spacedim>
    mode_vector(const Point &point)
    {
      if (point.mode_vector.norm_square() > 0.)
        return point.mode_vector;

      dealii::Tensor<1, spacedim> result;
      result = 0.;
      if constexpr (vector_lift)
        for (unsigned int component = 0; component < spacedim; ++component)
          if (component < point.mode_values.size())
            result[component] = point.mode_values[component];
      return result;
    }

    static std::vector<dealii::types::global_dof_index>
    execution_indices(const source_field_type &field, const Point &point)
    {
      std::vector<dealii::types::global_dof_index> result;
      result.reserve(point.dof_indices.size());
      for (const auto index : point.dof_indices)
        result.push_back(field.execution_index(index));
      return result;
    }

    template <typename Context>
    Operator
    linearize_vector(const Context &context, const FieldId field) const
    {
      const auto owned         = point_owned_;
      const auto relevant      = point_relevant_;
      const auto points        = points_;
      const auto column_source = source_for(field);
      const auto communicator  = mpi_communicator();
      const auto coefficient   = scale();
      const auto observable    = source_;
      const auto context_copy  = context;
      const auto time          = context.time();

      Operator result;
      result.reinit_range_vector = [n = points.size()](value_type &vector,
                                                       const bool) {
        vector.resize(n);
        for (auto &entry : vector)
          entry = 0.;
      };
      result.reinit_domain_vector =
        [column_source, communicator](state_type &vector, const bool omit) {
          vector.reinit(column_source.locally_owned_dofs(),
                        column_source.locally_relevant_dofs(),
                        communicator);
          if (!omit)
            vector = 0.;
        };
      result.vmult =
        [points, coefficient, observable, context_copy, field, time](
          value_type &destination, const state_type &direction) {
          destination.resize(points.size());
          for (std::size_t q = 0; q < points.size(); ++q)
            {
              double derivative = 0.;
              derivative        = composed_derivative(
                observable, context_copy, direction, field, points[q], time);
              destination[q] =
                mode_vector(points[q]) * (coefficient * derivative);
            }
        };
      result.vmult_add =
        [points, coefficient, observable, context_copy, field, time](
          value_type &destination, const state_type &direction) {
          for (std::size_t q = 0; q < points.size(); ++q)
            {
              const double derivative = composed_derivative(
                observable, context_copy, direction, field, points[q], time);
              destination[q] +=
                mode_vector(points[q]) * (coefficient * derivative);
            }
        };
      result.Tvmult = [points,
                       coefficient,
                       observable,
                       column_source,
                       context_copy,
                       field,
                       time](state_type       &destination,
                             const value_type &values) {
        destination = 0.;
        for (std::size_t q = 0; q < points.size(); ++q)
          {
            const double weight =
              coefficient * (mode_vector(points[q]) * values[q]);
            const auto tria_cell =
              column_source.dof_handler()
                .get_triangulation()
                .create_cell_iterator(points[q].source_cell_id);
            const auto cell =
              tria_cell->as_dof_handler_iterator(column_source.dof_handler());
            std::vector<dealii::types::global_dof_index> native_indices(
              cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(native_indices);
            std::vector<dealii::types::global_dof_index> execution_indices;
            std::vector<double>                          local;
            for (unsigned int i = 0; i < native_indices.size(); ++i)
              if (column_source.has_execution_index(native_indices[i]))
                {
                  execution_indices.push_back(
                    column_source.execution_index(native_indices[i]));
                  local.push_back(
                    weight *
                    composed_basis_derivative(
                      observable, context_copy, field, points[q], i, time));
                }
            column_source.constraints().distribute_local_to_global(
              local, execution_indices, destination);
          }
        destination.compress(dealii::VectorOperation::add);
      };
      result.Tvmult_add = [transpose =
                             result.Tvmult](state_type       &destination,
                                            const value_type &values) {
        state_type contribution;
        contribution.reinit(destination);
        contribution = 0.;
        transpose(contribution, values);
        destination += contribution;
      };
      return result;
    }

    template <typename Context>
    Operator
    linearize_composed(const Context &context, const FieldId field) const
    {
      const auto owned         = point_owned_;
      const auto relevant      = point_relevant_;
      const auto point_ids     = point_indices_;
      const auto points        = points_;
      const auto column_source = source_for(field);
      const auto communicator  = mpi_communicator();
      const auto coefficient   = scale();
      const auto observable    = source_;
      const auto context_copy  = context;
      const auto time          = context.time();

      Operator result;
      result.reinit_range_vector =
        [owned, relevant, communicator](value_type &vector, const bool omit) {
          vector.reinit(owned, relevant, communicator);
          if (!omit)
            vector = 0.;
        };
      result.reinit_domain_vector =
        [column_source, communicator](state_type &vector, const bool omit) {
          vector.reinit(column_source.locally_owned_dofs(),
                        column_source.locally_relevant_dofs(),
                        communicator);
          if (!omit)
            vector = 0.;
        };
      result.vmult = [points,
                      point_ids,
                      owned,
                      relevant,
                      communicator,
                      coefficient,
                      observable,
                      context_copy,
                      field,
                      time](value_type       &destination,
                            const state_type &direction) {
        destination.reinit(owned, relevant, communicator);
        destination = 0.;
        for (std::size_t q = 0; q < points.size(); ++q)
          destination[point_ids[q]] =
            coefficient * mode_value(points[q]) *
            composed_derivative(
              observable, context_copy, direction, field, points[q], time);
      };
      result.vmult_add =
        [points, point_ids, coefficient, observable, context_copy, field, time](
          value_type &destination, const state_type &direction) {
          for (std::size_t q = 0; q < points.size(); ++q)
            destination[point_ids[q]] +=
              coefficient * mode_value(points[q]) *
              composed_derivative(
                observable, context_copy, direction, field, points[q], time);
        };
      result.Tvmult = [points,
                       point_ids,
                       coefficient,
                       observable,
                       column_source,
                       context_copy,
                       field,
                       time](state_type       &destination,
                             const value_type &values) {
        destination = 0.;
        for (std::size_t q = 0; q < points.size(); ++q)
          {
            const double weight =
              coefficient * mode_value(points[q]) * values[point_ids[q]];
            const auto tria_cell =
              column_source.dof_handler()
                .get_triangulation()
                .create_cell_iterator(points[q].source_cell_id);
            const auto cell =
              tria_cell->as_dof_handler_iterator(column_source.dof_handler());
            std::vector<dealii::types::global_dof_index> native_indices(
              cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(native_indices);
            std::vector<dealii::types::global_dof_index> execution_indices;
            std::vector<double>                          local;
            for (unsigned int i = 0; i < native_indices.size(); ++i)
              if (column_source.has_execution_index(native_indices[i]))
                {
                  execution_indices.push_back(
                    column_source.execution_index(native_indices[i]));
                  local.push_back(
                    weight *
                    composed_basis_derivative(
                      observable, context_copy, field, points[q], i, time));
                }
            column_source.constraints().distribute_local_to_global(
              local, execution_indices, destination);
          }
        destination.compress(dealii::VectorOperation::add);
      };
      result.Tvmult_add = [transpose =
                             result.Tvmult](state_type       &destination,
                                            const value_type &values) {
        state_type contribution;
        contribution.reinit(destination);
        contribution = 0.;
        transpose(contribution, values);
        destination += contribution;
      };
      return result;
    }

    template <typename ObservableType, typename Context>
    static auto
    composed_derivative(const ObservableType &observable,
                        const Context        &context,
                        const state_type     &direction,
                        const FieldId         requested_field,
                        const Point          &point,
                        const double          time)
    {
      const auto evaluate = [&](const auto &input, const auto &, double) {
        const auto coefficients = [&](const auto &primitive) -> const auto & {
          return primitive.is_frozen() ?
                   primitive.template frozen_values<state_type>() :
                   context.state(primitive.source_field());
        };
        return primitive_value_at(input, point, time, coefficients);
      };
      const auto derivative = [&](const auto   &input,
                                  const FieldId input_field,
                                  const auto &,
                                  double) {
        if (input.is_frozen() || input_field != requested_field)
          return typename std::decay_t<decltype(input)>::value_type{};
        return primitive_direction_at(input, point, direction);
      };
      return linearize_observable_input(observable,
                                        requested_field,
                                        point.representative_point,
                                        time,
                                        evaluate,
                                        derivative);
    }

    template <typename ObservableType, typename Context>
    static auto
    composed_basis_derivative(const ObservableType &observable,
                              const Context        &context,
                              const FieldId         requested_field,
                              const Point          &point,
                              const unsigned int    basis,
                              const double          time)
    {
      const auto evaluate = [&](const auto &input, const auto &, double) {
        const auto coefficients = [&](const auto &primitive) -> const auto & {
          return primitive.is_frozen() ?
                   primitive.template frozen_values<state_type>() :
                   context.state(primitive.source_field());
        };
        return primitive_value_at(input, point, time, coefficients);
      };
      const auto derivative = [&](const auto   &input,
                                  const FieldId input_field,
                                  const auto &,
                                  double) {
        if (input.is_frozen() || input_field != requested_field)
          return typename std::decay_t<decltype(input)>::value_type{};
        return primitive_basis_at(input, point, basis);
      };
      return linearize_observable_input(observable,
                                        requested_field,
                                        point.representative_point,
                                        time,
                                        evaluate,
                                        derivative);
    }

    static double
    mode_value(const Point &point)
    {
      return point.mode_values.empty() ? 1. : point.mode_values.front();
    }

    template <typename Context>
    double
    composed_value(const Context &context, const Point &point) const
    {
      const auto coefficients = [&](const auto &input) -> const auto & {
        return input.is_frozen() ? input.template frozen_values<state_type>() :
                                   context.state(input.source_field());
      };
      return evaluate_frozen_expression(source_,
                                        point.source_cell_id,
                                        point.representative_point,
                                        context.time(),
                                        coefficients);
    }

    void
    build_points()
    {
      const auto &field      = source();
      const auto  quadrature = support_->representative_quadrature(
        field.finite_element().reference_cell());
      const auto flags = dealii::update_values | dealii::update_JxW_values |
                         dealii::update_quadrature_points |
                         dealii::update_jacobians;
      const dealii::hp::MappingCollection<reduced_dim, spacedim> mappings(
        field.mapping());
      const dealii::hp::QCollection<reduced_dim>  quadratures(quadrature);
      dealii::hp::FEValues<reduced_dim, spacedim> hp_values(
        mappings, field.dof_handler().get_fe_collection(), quadratures, flags);
      for (const auto &cell : field.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            hp_values.reinit(cell);
            const auto &values = hp_values.get_present_fe_values();
            std::vector<dealii::types::global_dof_index> indices(
              cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(indices);
            const auto &view = values[field.extractor()];
            for (const auto q : values.quadrature_point_indices())
              {
                const auto   representative_point = values.quadrature_point(q);
                const double thickness =
                  thickness_evaluator_ ?
                    thickness_evaluator_(cell->id(), representative_point, 0.) :
                    support_->thickness(representative_point, 0.);
                AssertThrow(std::isfinite(thickness) && thickness > 0.,
                            dealii::ExcMessage(
                              "Lift thickness must be finite and positive."));
                const auto transformed =
                  support_->transform(representative_point,
                                      values.jacobian(q),
                                      values.JxW(q),
                                      thickness,
                                      q);
                for (unsigned int section_q = 0; section_q < transformed.size();
                     ++section_q)
                  {
                    const auto &lifted = transformed[section_q];
                    Point       point;
                    point.point                = lifted.point;
                    point.representative_point = lifted.representative_point;
                    point.weight               = lifted.weight;
                    point.source_entity_id = cell->global_active_cell_index();
                    point.source_cell_id   = cell->id();
                    point.representative_qpoint = q;
                    point.section_qpoint        = section_q;
                    point.stable_id =
                      static_cast<std::uint64_t>(point.source_entity_id) *
                        quadrature.size() * transformed.size() +
                      q * transformed.size() + section_q;
                    point.dof_indices = indices;
                    point.basis_values.resize(indices.size());
                    if constexpr (vector_lift)
                      {
                        point.mode_vector = 0.;
                        if (mode_slot_ != dealii::numbers::invalid_unsigned_int)
                          {
                            AssertIndexRange(mode_slot_,
                                             lifted.mode_values.size() /
                                               spacedim);
                            for (unsigned int component = 0;
                                 component < spacedim;
                                 ++component)
                              point.mode_vector[component] =
                                lifted.mode_values[mode_slot_ * spacedim +
                                                   component];
                          }
                        else
                          for (unsigned int mode = 0;
                               mode * spacedim < lifted.mode_values.size();
                               ++mode)
                            for (unsigned int component = 0;
                                 component < spacedim;
                                 ++component)
                              point.mode_vector[component] +=
                                lifted.mode_values[mode * spacedim + component];
                        for (unsigned int i = 0; i < indices.size(); ++i)
                          point.basis_values[i] = view.value(i, q);
                        if (mode_slot_ == dealii::numbers::invalid_unsigned_int)
                          {
                            const double mode_norm = point.mode_vector.norm();
                            AssertThrow(
                              mode_norm > 0.,
                              dealii::ExcMessage(
                                "A vector lift requires a nonzero mode "
                                "direction."));
                            point.mode_vector /= mode_norm;
                          }
                      }
                    else
                      for (unsigned int i = 0; i < indices.size(); ++i)
                        point.basis_values[i] =
                          view.value(i, q) *
                          lifted.mode_values
                            [mode_slot_ ==
                                 dealii::numbers::invalid_unsigned_int ?
                               0 :
                               mode_slot_];
                    point.source_basis_values.resize(indices.size());
                    for (unsigned int i = 0; i < indices.size(); ++i)
                      point.source_basis_values[i] = view.value(i, q);
                    if (field.is_reindexed())
                      {
                        std::vector<dealii::types::global_dof_index> dofs;
                        std::vector<double>                          basis;
                        std::vector<double> source_basis;
                        dofs.reserve(point.dof_indices.size());
                        basis.reserve(point.basis_values.size());
                        source_basis.reserve(point.source_basis_values.size());
                        for (unsigned int i = 0; i < indices.size(); ++i)
                          if (field.has_execution_index(indices[i]))
                            {
                              dofs.push_back(indices[i]);
                              basis.push_back(point.basis_values[i]);
                              source_basis.push_back(
                                point.source_basis_values[i]);
                            }
                        point.dof_indices         = std::move(dofs);
                        point.basis_values        = std::move(basis);
                        point.source_basis_values = std::move(source_basis);
                      }
                    point.mode_values = lifted.mode_values;
                    if (mode_slot_ != dealii::numbers::invalid_unsigned_int)
                      {
                        const unsigned int width = vector_lift ? spacedim : 1;
                        AssertIndexRange(mode_slot_,
                                         lifted.mode_values.size() / width);
                        point.mode_values.assign(lifted.mode_values.begin() +
                                                   mode_slot_ * width,
                                                 lifted.mode_values.begin() +
                                                   (mode_slot_ + 1) * width);
                      }
                    points_.push_back(std::move(point));
                  }
              }
          }

      const auto [offset, size] =
        dealii::Utilities::MPI::partial_and_total_sum(points_.size(),
                                                      mpi_communicator());
      point_owned_.set_size(size);
      point_owned_.add_range(offset, offset + points_.size());
      point_owned_.compress();
      point_relevant_ = point_owned_;
      point_indices_.reserve(points_.size());
      for (std::size_t i = 0; i < points_.size(); ++i)
        point_indices_.push_back(offset + i);
    }

    SourceObservable         source_;
    std::shared_ptr<Support> support_;
    unsigned int             mode_slot_ = dealii::numbers::invalid_unsigned_int;
    std::function<
      double(const dealii::CellId &, const dealii::Point<spacedim> &, double)>
                                                 thickness_evaluator_;
    std::vector<Point>                           points_;
    dealii::IndexSet                             point_owned_;
    dealii::IndexSet                             point_relevant_;
    std::vector<dealii::types::global_dof_index> point_indices_;
  };

  /** A lifted Observable used only as an explicit residual test expression. */
  template <typename LiftedObservableType>
  class TestLiftedObservable : public LiftedObservableType
  {
  public:
    using LiftedObservableType::LiftedObservableType;

    explicit TestLiftedObservable(const LiftedObservableType &observable)
      : LiftedObservableType(observable)
    {}

    const std::vector<FieldId> &
    dependencies() const
    {
      static const std::vector<FieldId> none;
      return none;
    }

    TestLiftedObservable
    with_id(const FieldId id) const
    {
      return TestLiftedObservable(LiftedObservableType::with_id(id));
    }
  };

  namespace detail
  {
    template <typename SourceObservable,
              int reduced_dim,
              int surface_dim,
              int spacedim,
              int n_components>
    struct is_observable<LiftedObservable<SourceObservable,
                                          reduced_dim,
                                          surface_dim,
                                          spacedim,
                                          n_components>> : std::true_type
    {};

    template <typename SourceObservable,
              int reduced_dim,
              int surface_dim,
              int spacedim,
              int n_components>
    struct is_lifted_observable<LiftedObservable<SourceObservable,
                                                 reduced_dim,
                                                 surface_dim,
                                                 spacedim,
                                                 n_components>> : std::true_type
    {};

    template <typename LiftedObservableType>
    struct is_test_expression<TestLiftedObservable<LiftedObservableType>>
      : std::true_type
    {};

    template <typename LiftedObservableType>
    struct is_lifted_observable<TestLiftedObservable<LiftedObservableType>>
      : std::true_type
    {};
  } // namespace detail

  template <typename LiftedObservableType>
  auto
  test(const LiftedObservableType &observable) -> std::enable_if_t<
    detail::is_lifted_observable<LiftedObservableType>::value,
    TestLiftedObservable<LiftedObservableType>>
  {
    return TestLiftedObservable<LiftedObservableType>(observable);
  }

  template <
    typename SourceObservable,
    int reduced_dim,
    int surface_dim,
    int spacedim,
    int n_components,
    std::enable_if_t<detail::is_observable<SourceObservable>::value, int> = 0>
  auto
  make_lift(
    const SourceObservable &source,
    const TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>
      &lift)
  {
    return LiftedObservable<SourceObservable,
                            reduced_dim,
                            surface_dim,
                            spacedim,
                            n_components>(source, lift);
  }

  template <
    typename SourceObservable,
    int reduced_dim,
    int surface_dim,
    int spacedim,
    int n_components,
    std::enable_if_t<detail::is_observable<SourceObservable>::value, int> = 0>
  auto
  make_lift(
    const SourceObservable &source,
    const TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>
                                      &lift,
    SourceThicknessEvaluator<spacedim> thickness_evaluator)
  {
    return LiftedObservable<SourceObservable,
                            reduced_dim,
                            surface_dim,
                            spacedim,
                            n_components>(source,
                                          lift,
                                          std::move(thickness_evaluator));
  }

  template <typename SourceObservable,
            typename ThicknessObservable,
            int reduced_dim,
            int surface_dim,
            int spacedim,
            int n_components,
            std::enable_if_t<
              detail::is_observable<SourceObservable>::value &&
                detail::is_observable<std::decay_t<ThicknessObservable>>::value,
              int> = 0>
  auto
  make_lift(
    const SourceObservable &source,
    const TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>
                              &lift,
    const ThicknessObservable &thickness)
  {
    return LiftedObservable<SourceObservable,
                            reduced_dim,
                            surface_dim,
                            spacedim,
                            n_components>(source, lift, thickness);
  }

  template <
    typename SourceObservable,
    int reduced_dim,
    int surface_dim,
    int spacedim,
    int n_components,
    std::enable_if_t<detail::is_observable<SourceObservable>::value, int> = 0>
  auto
  lift(const SourceObservable &source,
       const TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>
         &descriptor)
  {
    return make_lift(source, descriptor);
  }

  template <
    typename SourceObservable,
    int reduced_dim,
    int surface_dim,
    int spacedim,
    int n_components,
    std::enable_if_t<detail::is_observable<SourceObservable>::value, int> = 0>
  auto
  lift(const SourceObservable &source,
       const TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>
                                         &descriptor,
       SourceThicknessEvaluator<spacedim> thickness_evaluator)
  {
    return make_lift(source, descriptor, std::move(thickness_evaluator));
  }

  template <typename SourceObservable,
            typename ThicknessObservable,
            int reduced_dim,
            int surface_dim,
            int spacedim,
            int n_components,
            std::enable_if_t<
              detail::is_observable<SourceObservable>::value &&
                detail::is_observable<std::decay_t<ThicknessObservable>>::value,
              int> = 0>
  auto
  lift(const SourceObservable &source,
       const TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>
                                 &descriptor,
       const ThicknessObservable &thickness)
  {
    return make_lift(source, descriptor, thickness);
  }
} // namespace ImmersX

#endif // immersx_observable_lift_h
