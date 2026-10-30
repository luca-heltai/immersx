// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_metric_flow_x_pressure_h
#define immersx_metric_flow_x_pressure_h

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X
#  include <immersx/core/matrix_operator.h>
#  include <immersx/core/observable.h>
#  include <immersx/physics/metric_flow_x.h>

#  include <map>
#  include <tuple>

namespace ImmersX
{
  namespace detail
  {
    template <typename Flow, typename = void>
    struct has_external_pressure_state_jacobian : std::false_type
    {};

    template <typename Flow>
    struct has_external_pressure_state_jacobian<
      Flow,
      std::void_t<decltype(std::declval<const Flow &>()
                             .add_external_pressure_state_jacobian(
                               std::declval<double>(),
                               std::declval<const MetricFlowX::VectorType &>(),
                               std::declval<const typename Flow::
                                              ExternalPressureProvider &>(),
                               std::declval<MetricFlowX::MatrixType &>()))>>
      : std::true_type
    {};
  } // namespace detail

  /** Evaluate a scalar value observable at native pressure evaluation points.
   * CellId selects the DG trace at faces and junctions. Referenced spaces must
   * outlive the provider; coefficients are copied into a relevant vector.
   */
  template <typename Pressure>
  auto
  metric_flow_x_pressure_provider(const Pressure                &pressure,
                                  const MetricFlowX::VectorType &coefficients)
  {
    const auto field   = pressure.source();
    auto       ghosted = std::make_shared<MetricFlowX::VectorType>();
    ghosted->reinit(field.locally_owned_dofs(),
                    field.locally_relevant_dofs(),
                    field.space().mpi_communicator());
    *ghosted = coefficients;
    ghosted->update_ghost_values();
    using Cell = dealii::DoFHandler<1, 3>::active_cell_iterator;
    auto cells = std::make_shared<std::map<dealii::CellId, Cell>>();
    for (const auto &cell : field.dof_handler().active_cell_iterators())
      if (!cell->is_artificial())
        cells->emplace(cell->id(), cell);
    return [field, ghosted, cells, scale = pressure.scale()](
             const MetricFlowX::BloodFlowSystem<1, 3>::PressureEvaluationPoint
               &point) {
      const auto iterator = cells->find(point.cell_id);
      AssertThrow(
        iterator != cells->end(),
        dealii::ExcMessage(
          "A native pressure evaluation requested an unavailable multiplier cell."));
      const auto &cell = iterator->second;
      const auto  unit =
        field.mapping().transform_real_to_unit_cell(cell, point.point);
      const auto &fe = field.space().finite_element();
      AssertDimension(fe.n_components(), 1);
      std::vector<dealii::types::global_dof_index> indices(
        fe.n_dofs_per_cell());
      cell->get_dof_indices(indices);
      double value = 0.;
      for (unsigned int j = 0; j < indices.size(); ++j)
        value += (*ghosted)[field.execution_index(indices[j])] *
                 fe.shape_value(j, unit);
      return scale * value;
    };
  }

  /** Linearize the native pressure contribution at fixed pressure coefficients.
   * HLL weights depend on the flow state even when pressure is prescribed.
   * The returned actions retain the assembled matrix for their full lifetime.
   */
  template <typename Flow, typename Pressure>
  MaterializedOperator<MetricFlowX::VectorType, MetricFlowX::MatrixType>
  metric_flow_x_pressure_state_operator(
    Flow                          &problem,
    const Pressure                &pressure,
    const MetricFlowX::VectorType &pressure_coefficients,
    const MetricFlowX::VectorType &flow_state,
    const double                   time)
  {
    using Vector      = MetricFlowX::VectorType;
    using Matrix      = MetricFlowX::MatrixType;
    const auto matrix = detail::clone_matrix(problem.state_jacobian_matrix());
    *matrix           = 0.;
    if constexpr (detail::has_external_pressure_state_jacobian<Flow>::value)
      problem.add_external_pressure_state_jacobian(
        time,
        flow_state,
        metric_flow_x_pressure_provider(pressure, pressure_coefficients),
        *matrix);
    else
      AssertThrow(
        false,
        dealii::ExcMessage(
          "The installed MetricFlowX does not expose the prescribed-pressure state Jacobian."));
    MaterializedOperator<Vector, Matrix> result;
    result.view.reinit_range_vector = [&problem, matrix](Vector &dst,
                                                         const bool) {
      problem.reinit_state(dst);
    };
    result.view.reinit_domain_vector = result.view.reinit_range_vector;
    result.view.vmult = [matrix](Vector &dst, const Vector &src) {
      matrix->vmult(dst, src);
    };
    result.view.vmult_add = [matrix](Vector &dst, const Vector &src) {
      matrix->vmult_add(dst, src);
    };
    result.view.Tvmult = [matrix](Vector &dst, const Vector &src) {
      matrix->Tvmult(dst, src);
    };
    result.view.Tvmult_add = [matrix](Vector &dst, const Vector &src) {
      matrix->Tvmult_add(dst, src);
    };
    result.materialize = [matrix]() { return detail::clone_matrix(*matrix); };
    result.materialize_into = [matrix](Matrix &dst) {
      detail::copy_matrix(dst, *matrix);
    };
    return result;
  }

  /** Linearize native pressure with respect to its scalar coefficients.
   * The forward action uses native assembly. Transpose and matrix requests
   * collectively probe the coefficient basis once; their cost is linear in
   * the global number of pressure coefficients.
   */
  template <typename Pressure>
  MaterializedOperator<MetricFlowX::VectorType, MetricFlowX::MatrixType>
  metric_flow_x_pressure_operator(MetricFlowX::BloodFlowSystem<1, 3> &problem,
                                  const Pressure                     &pressure,
                                  const MetricFlowX::VectorType &flow_state,
                                  const double                   time)
  {
    using Vector            = MetricFlowX::VectorType;
    using Matrix            = MetricFlowX::MatrixType;
    const auto field        = pressure.source();
    const auto state        = std::make_shared<Vector>(flow_state);
    const auto rows         = problem.locally_owned_dofs();
    const auto columns      = field.locally_owned_dofs();
    const auto communicator = field.space().mpi_communicator();
    MaterializedOperator<Vector, Matrix> result;
    result.view.reinit_range_vector = [&problem](Vector &dst, const bool) {
      problem.reinit_state(dst);
    };
    result.view.reinit_domain_vector = [columns, communicator](Vector &dst,
                                                               const bool) {
      dst.reinit(columns, communicator);
    };
    result.view.vmult = [&problem, pressure, state, time](Vector       &dst,
                                                          const Vector &src) {
      dst = 0.;
      problem.add_external_pressure_residual(
        time, *state, metric_flow_x_pressure_provider(pressure, src), dst);
    };
    const auto action  = result.view;
    auto       cached  = std::make_shared<std::shared_ptr<Matrix>>();
    result.materialize = [action, cached, rows, columns, communicator]() {
      if (!*cached)
        {
          Vector basis, image;
          action.reinit_domain_vector(basis, false);
          action.reinit_range_vector(image, false);
          dealii::DynamicSparsityPattern sparsity(rows.size(),
                                                  columns.size(),
                                                  rows);
          std::vector<std::tuple<dealii::types::global_dof_index,
                                 dealii::types::global_dof_index,
                                 double>>
            entries;
          for (dealii::types::global_dof_index j = 0; j < columns.size(); ++j)
            {
              basis = 0.;
              if (columns.is_element(j))
                basis[j] = 1.;
              basis.compress(dealii::VectorOperation::insert);
              action.vmult(image, basis);
              for (const auto i : rows)
                if (image[i] != 0.)
                  {
                    sparsity.add(i, j);
                    entries.emplace_back(i, j, image[i]);
                  }
            }
          *cached = std::make_shared<Matrix>();
          (*cached)->reinit(rows, columns, sparsity, communicator, false);
          for (const auto &[i, j, value] : entries)
            (*cached)->set(i, j, value);
          (*cached)->compress(dealii::VectorOperation::insert);
        }
      return detail::clone_matrix(**cached);
    };
    const auto matrix       = result.materialize;
    result.materialize_into = [matrix](Matrix &dst) {
      detail::copy_matrix(dst, *matrix());
    };
    result.view.Tvmult = [matrix](Vector &dst, const Vector &src) {
      matrix()->Tvmult(dst, src);
    };
    result.view.Tvmult_add = [matrix](Vector &dst, const Vector &src) {
      matrix()->Tvmult_add(dst, src);
    };
    result.view.vmult_add = [action](Vector &dst, const Vector &src) {
      Vector image;
      action.reinit_range_vector(image, false);
      action.vmult(image, src);
      dst += image;
    };
    return result;
  }
} // namespace ImmersX
#endif
#endif
