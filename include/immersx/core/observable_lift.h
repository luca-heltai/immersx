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

#include <deal.II/lac/linear_operator.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/observable.h>
#include <immersx/coupling/detail/coupling_point.h>
#include <immersx/coupling/detail/fe_stencil.h>
#include <immersx/coupling/tensor_product_lift.h>

#include <cstdint>
#include <limits>
#include <memory>
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
    static_assert(reduced_dim == 1,
                  "The Observable lift currently supports curve sources.");
    static_assert(surface_dim == 2,
                  "The Observable lift currently supports surface targets.");
    static_assert(std::is_arithmetic_v<typename SourceObservable::value_type>,
                  "The initial Observable lift supports scalar quantities.");

  public:
    static constexpr unsigned int support_dimension        = surface_dim;
    static constexpr unsigned int ambient_dimension        = spacedim;
    static constexpr unsigned int representative_dimension = reduced_dim;

    using source_field_type = typename SourceObservable::source_field_type;
    using value_type        = ImmersXLA::MPI::Vector;
    using state_type        = ImmersXLA::MPI::Vector;
    using Point             = detail::CouplingPoint<spacedim, double>;
    using Lift =
      TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>;
    using Support  = TensorProductLiftSupport<reduced_dim,
                                             surface_dim,
                                             spacedim,
                                             n_components>;
    using Operator = dealii::LinearOperator<value_type, state_type>;

    LiftedObservable(SourceObservable source, const Lift &lift)
      : source_(std::move(source))
      , support_(lift.parameters())
      , constant_thickness_(lift.parameters().constant_thickness)
    {
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
      result.reinit(point_owned_, point_relevant_, mpi_communicator());
      if (is_frozen())
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
          [](const auto &point) -> const auto & { return point.dof_indices; },
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
          [](const auto &point) -> const auto & { return point.dof_indices; },
          [](const auto &point, const unsigned int i) {
            return point.basis_values[i];
          },
          false,
          false);
      result *= scale();
      return result;
    }

    template <typename Context>
    Operator
    linearize(const Context &context, const FieldId field) const
    {
      (void)context;
      AssertThrow(field == source_field(),
                  dealii::ExcMessage(
                    "The requested field is not a lift dependency."));
      const auto owned            = point_owned_;
      const auto relevant         = point_relevant_;
      const auto point_ids        = point_indices_;
      const auto points           = points_;
      const auto source_field_ref = source();
      const auto communicator     = mpi_communicator();
      const auto coefficient      = scale();

      Operator result;
      result.reinit_range_vector =
        [owned, relevant, communicator](value_type &vector, const bool omit) {
          vector.reinit(owned, relevant, communicator);
          if (!omit)
            vector = 0.;
        };
      result.reinit_domain_vector =
        [source_field_ref, communicator](state_type &vector, const bool omit) {
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
          [](const auto &point) -> const auto & { return point.dof_indices; },
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
          [](const auto &point) -> const auto & { return point.dof_indices; },
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
            [](const auto &point) -> const auto & { return point.dof_indices; },
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
            [](const auto &point) -> const auto & { return point.dof_indices; },
            [coefficient](const auto &point, const unsigned int i) {
              return coefficient * point.basis_values[i];
            },
            true);
        };
      return result;
    }

    template <typename Context>
    Operator
    linearize(const Context &context) const
    {
      return linearize(context, source_field());
    }

  private:
    void
    build_points()
    {
      const auto &field      = source();
      const auto &quadrature = support_.representative_quadrature();
      const auto  flags = dealii::update_values | dealii::update_JxW_values |
                         dealii::update_quadrature_points;
      dealii::FEValues<reduced_dim, spacedim> values(
        field.mapping(), field.space().finite_element(), quadrature, flags);
      for (const auto &cell : field.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            values.reinit(cell);
            std::vector<dealii::types::global_dof_index> indices(
              cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(indices);
            dealii::Tensor<1, spacedim> tangent;
            tangent = cell->vertex(1) - cell->vertex(0);
            tangent /= tangent.norm();
            const auto &view = values[field.extractor()];
            for (const auto q : values.quadrature_point_indices())
              {
                const auto transformed =
                  support_.transform(values.quadrature_point(q),
                                     tangent,
                                     values.JxW(q),
                                     constant_thickness_,
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
                    point.representative_qpoint = q;
                    point.section_qpoint        = section_q;
                    point.stable_id =
                      static_cast<std::uint64_t>(point.source_entity_id) *
                        quadrature.size() * transformed.size() +
                      q * transformed.size() + section_q;
                    point.dof_indices = indices;
                    point.basis_values.resize(indices.size());
                    for (unsigned int i = 0; i < indices.size(); ++i)
                      point.basis_values[i] =
                        view.value(i, q) * lifted.mode_values.front();
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

    SourceObservable                             source_;
    Support                                      support_;
    const double                                 constant_thickness_;
    std::vector<Point>                           points_;
    dealii::IndexSet                             point_owned_;
    dealii::IndexSet                             point_relevant_;
    std::vector<dealii::types::global_dof_index> point_indices_;
  };

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
  lift(const SourceObservable &source,
       const TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>
         &descriptor)
  {
    return make_lift(source, descriptor);
  }
} // namespace ImmersX

#endif // immersx_observable_lift_h
