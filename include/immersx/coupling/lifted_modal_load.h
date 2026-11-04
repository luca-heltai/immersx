// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of the ImmersX application, based on deal.II.
// ---------------------------------------------------------------------
#ifndef immersx_lifted_modal_load_h
#define immersx_lifted_modal_load_h

#include <immersx/core/constraint.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/weak_term.h>

#include <tuple>
#include <type_traits>
#include <vector>

namespace ImmersX
{
  namespace detail
  {
    template <typename Field,
              typename Points,
              typename Basis,
              typename Datum,
              typename Selection>
    auto
    assemble_modal_gram(const Field     &field,
                        const Points    &points,
                        const Basis     &basis,
                        const Datum     &datum,
                        const Selection &selected)
    {
      using Matrix                           = ImmersXLA::MPI::SparseMatrix;
      using Vector                           = ImmersXLA::MPI::Vector;
      auto                           metric  = std::make_shared<Matrix>();
      auto                           outside = std::make_shared<Matrix>();
      auto                           rhs     = std::make_shared<Vector>();
      dealii::DynamicSparsityPattern sparsity(field.locally_relevant_dofs());
      dealii::DoFTools::make_sparsity_pattern(field.dof_handler(),
                                              sparsity,
                                              field.constraints(),
                                              true);
      dealii::SparsityTools::distribute_sparsity_pattern(
        sparsity,
        field.locally_owned_dofs(),
        field.space().mpi_communicator(),
        field.locally_relevant_dofs());
      initialize_weak_matrix(*metric,
                             field.locally_owned_dofs(),
                             field.locally_owned_dofs(),
                             sparsity,
                             field.space().mpi_communicator());
      initialize_weak_matrix(*outside,
                             field.locally_owned_dofs(),
                             field.locally_owned_dofs(),
                             sparsity,
                             field.space().mpi_communicator());
      rhs->reinit(field.locally_owned_dofs(), field.space().mpi_communicator());
      using BasisValue = std::decay_t<decltype(basis(std::size_t{}, 0u))>;
      using DatumValue = std::decay_t<decltype(datum(std::size_t{}))>;
      std::vector<BasisValue>    basis_values;
      dealii::FullMatrix<double> local, exterior;
      dealii::Vector<double>     load;
      for (std::size_t q = 0; q < points.size(); ++q)
        {
          const auto &point = points[q];
          const auto  n     = point.dof_indices.size();
          local.reinit(n, n);
          exterior.reinit(n, n);
          load.reinit(n);
          basis_values.resize(n);
          for (unsigned int i = 0; i < n; ++i)
            basis_values[i] = basis(q, i);
          const bool inside     = selected(point);
          const auto prescribed = inside ? datum(q) : DatumValue{};
          for (unsigned int i = 0; i < n; ++i)
            {
              const auto &vi = basis_values[i];
              if (inside)
                load[i] = point.weight * natural_pairing(vi, prescribed);
              for (unsigned int j = 0; j < n; ++j)
                {
                  local(i, j) =
                    point.weight * natural_pairing(vi, basis_values[j]);
                  if (!inside)
                    exterior(i, j) = local(i, j);
                }
            }
          field.constraints().distribute_local_to_global(
            local, load, point.dof_indices, *metric, *rhs);
          field.constraints().distribute_local_to_global(exterior,
                                                         point.dof_indices,
                                                         *outside);
        }
      metric->compress(dealii::VectorOperation::add);
      outside->compress(dealii::VectorOperation::add);
      rhs->compress(dealii::VectorOperation::add);
      return std::make_tuple(metric, outside, rhs);
    }

    template <typename Builder, typename Field, typename Matrix>
    void
    add_modal_outside_mass(Builder                       &builder,
                           const Field                   &field,
                           const std::shared_ptr<Matrix> &outside)
    {
      auto op             = builder.matrix_operator(*outside);
      op.materialize      = [outside] { return outside; };
      op.materialize_into = [outside](auto &destination) {
        destination.copy_from(*outside);
      };
      builder.term(field.id(), "multiplier.outside")
        .residual([op, id = field.id()](const auto &context) {
          return -1. * (op.view * context.state(id));
        })
        .state(field.id(), -1. * op);
    }
  } // namespace detail

  /** Assemble all modes together, retaining cross-mode products on a cut
   * section. Return the full metric, its exterior part, and the load on the
   * overlap. The overlap must be built from the full source mesh.
   */
  template <typename Lifted, typename Function, typename Support>
  auto
  lifted_modal_load(const std::vector<Lifted> &lifts,
                    const Function            &function,
                    const Support             &overlap)
  {
    AssertThrow(!lifts.empty(),
                dealii::ExcMessage("A modal load needs at least one lift."));
    const auto &points = lifts.front().lifted_points();
    const auto &field  = lifts.front().source();
    overlap.assert_matches(field.dof_handler().get_triangulation());
    AssertThrow(!overlap.has_material_filter(),
                dealii::ExcMessage(
                  "A joint modal Gram matrix requires the full source mesh."));
    for (const auto &lifted : lifts)
      {
        AssertThrow(&lifted.source().dof_handler() == &field.dof_handler(),
                    dealii::ExcMessage("Modal lifts must share one FE space."));
        AssertThrow(
          lifted.lifted_points().size() == points.size(),
          dealii::ExcMessage(
            "Modal lifts must use the same number of quadrature points."));
        for (std::size_t q = 0; q < points.size(); ++q)
          {
            const auto &point = lifted.lifted_points()[q];
            AssertThrow(
              point.dof_indices == points[q].dof_indices &&
                point.source_entity_id == points[q].source_entity_id &&
                point.representative_qpoint ==
                  points[q].representative_qpoint &&
                point.section_qpoint == points[q].section_qpoint &&
                point.point == points[q].point &&
                point.weight == points[q].weight,
              dealii::ExcMessage(
                "Modal lifts must use identical physical quadrature points."));
          }
      }
    const auto basis = [&](const std::size_t q, const unsigned int i) {
      dealii::Tensor<1, Lifted::ambient_dimension> value;
      for (const auto &lifted : lifts)
        {
          const auto &point = lifted.lifted_points()[q];
          value += point.source_basis_values[i] * point.mode_vector;
        }
      return value;
    };
    const auto datum = [&](const std::size_t q) {
      dealii::Tensor<1, Lifted::ambient_dimension> value;
      for (unsigned int c = 0; c < lifts.size(); ++c)
        {
          const auto  &point     = lifts[c].lifted_points()[q];
          const double amplitude = [&] {
            if constexpr (std::is_invocable_r_v<double,
                                                const Function &,
                                                const typename Lifted::Point &,
                                                unsigned int>)
              return function(point, c);
            else
              return function.value(point.representative_point, c);
          }();
          value += amplitude * point.mode_vector;
        }
      return value;
    };
    return detail::assemble_modal_gram(
      field, points, basis, datum, [&](const auto &point) {
        return overlap.includes_lifted_point(point);
      });
  }

  /** Assemble a physical modal mass and the load of a modal datum.
   * The datum is a modal coefficient on the representative curve, matching
   * ReducedCoupling's static right-hand-side convention, not a physical
   * vector evaluated around the cross section. The modes have maximum one, so
   * the datum is the amplitude of the mode, and the load is the integral of
   * (datum phi) . phi, weighted like the mass matrix. Accept either a deal.II
   * Function or a callback taking the retained point and component; the latter
   * can evaluate imported fields on the known source cell without point search.
   */
  template <typename Lifted, typename Function>
  auto
  lifted_modal_load(const Lifted      &lifted,
                    const Function    &function,
                    const unsigned int component)
  {
    using Matrix                          = ImmersXLA::MPI::SparseMatrix;
    using Vector                          = ImmersXLA::MPI::Vector;
    const auto                    &field  = lifted.source();
    auto                           matrix = std::make_shared<Matrix>();
    auto                           rhs    = std::make_shared<Vector>();
    dealii::DynamicSparsityPattern sparsity(field.locally_relevant_dofs());
    dealii::DoFTools::make_sparsity_pattern(field.dof_handler(),
                                            sparsity,
                                            field.constraints(),
                                            true);
    dealii::SparsityTools::distribute_sparsity_pattern(
      sparsity,
      field.locally_owned_dofs(),
      field.space().mpi_communicator(),
      field.locally_relevant_dofs());
    detail::initialize_weak_matrix(*matrix,
                                   field.locally_owned_dofs(),
                                   field.locally_owned_dofs(),
                                   sparsity,
                                   field.space().mpi_communicator());
    rhs->reinit(field.locally_owned_dofs(), field.space().mpi_communicator());
    for (const auto &point : lifted.lifted_points())
      {
        const auto                 n = point.dof_indices.size();
        dealii::FullMatrix<double> local(n, n);
        dealii::Vector<double>     load(n);
        const double mode_squared = point.mode_vector * point.mode_vector;
        const double datum        = [&]() {
          if constexpr (std::is_invocable_r_v<double,
                                              const Function &,
                                              const typename Lifted::Point &,
                                              unsigned int>)
            return function(point, component);
          else
            return function.value(point.representative_point, component);
        }();
        for (unsigned int i = 0; i < n; ++i)
          {
            load[i] = point.weight * mode_squared *
                      point.source_basis_values[i] * datum;
            for (unsigned int j = 0; j < n; ++j)
              local(i, j) = point.weight * mode_squared *
                            point.source_basis_values[i] *
                            point.source_basis_values[j];
          }
        field.constraints().distribute_local_to_global(
          local, load, point.dof_indices, *matrix, *rhs);
      }
    matrix->compress(dealii::VectorOperation::add);
    rhs->compress(dealii::VectorOperation::add);
    return std::make_pair(matrix, rhs);
  }

  /** A unified modal constraint with its physical section-weighted metric. */
  template <typename Weak>
  struct LiftedModalInteraction
  {
    using Matrix = ImmersXLA::MPI::SparseMatrix;
    using Vector = ImmersXLA::MPI::Vector;
    Constraint<ConstraintSum<Weak>, Vector> constraint;
    std::shared_ptr<Matrix>                 metric;
    std::shared_ptr<Matrix>                 outside;

    template <typename Builder>
    auto
    operator()(Builder &builder) const
    {
      auto op             = builder.matrix_operator(*metric);
      op.materialize      = [metric = metric] { return metric; };
      op.materialize_into = [metric = metric](auto &destination) {
        destination.copy_from(*metric);
      };
      auto fields = constraint.add(builder, &op);
      if (outside)
        detail::add_modal_outside_mass(builder, fields.multiplier, outside);
      return fields;
    }
  };

  template <typename Weak>
  LiftedModalInteraction<Weak>
  lifted_modal_constraint(
    ConstraintSum<Weak>                           terms,
    std::shared_ptr<ImmersXLA::MPI::SparseMatrix> metric,
    ImmersXLA::MPI::Vector                        rhs,
    std::shared_ptr<ImmersXLA::MPI::SparseMatrix> outside = {})
  {
    return {make_constraint(std::move(terms), std::move(rhs)),
            std::move(metric),
            std::move(outside)};
  }

  /** Add an owned affine FE operator without exposing execution block numbers.
   */
  template <typename Field, typename Matrix, typename Vector>
  auto
  affine_field_term(Field                   field,
                    std::shared_ptr<Matrix> matrix,
                    std::shared_ptr<Vector> rhs)
  {
    return [field, matrix, rhs](auto &builder) {
      using Builder       = std::decay_t<decltype(builder)>;
      auto op             = builder.matrix_operator(*matrix);
      op.materialize      = [matrix] { return matrix; };
      op.materialize_into = [matrix](Matrix &destination) {
        destination.copy_from(*matrix);
      };
      builder.term(field.id(), "affine")
        .residual([field, matrix, rhs](const auto &context) {
          const auto                        &state = context.state(field.id());
          typename Builder::Model::Operation result;
          result.reinit_vector = [rhs](auto &v, const bool) { v.reinit(*rhs); };
          result.apply         = [matrix, rhs, &state](auto &v) {
            matrix->vmult(v, state);
            v -= *rhs;
          };
          result.apply_add = [matrix, rhs, &state](auto &v) {
            Vector temporary;
            temporary.reinit(*rhs);
            matrix->vmult(temporary, state);
            temporary -= *rhs;
            v += temporary;
          };
          return result;
        })
        .state(field.id(), op);
      return field;
    };
  }
} // namespace ImmersX
#endif
