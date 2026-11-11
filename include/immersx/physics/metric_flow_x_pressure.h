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
#  include <deal.II/dofs/dof_tools.h>

#  include <deal.II/fe/fe_values.h>

#  include <deal.II/grid/grid_tools.h>

#  include <deal.II/lac/dynamic_sparsity_pattern.h>
#  include <deal.II/lac/full_matrix.h>
#  include <deal.II/lac/sparsity_tools.h>

#  include <immersx/compatibility/reference_cell.h>
#  include <immersx/core/integration_support.h>
#  include <immersx/core/matrix_operator.h>
#  include <immersx/core/observable.h>
#  include <immersx/physics/metric_flow_x.h>

#  include <algorithm>
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
    template <typename Flow, typename = void>
    struct has_external_pressure_jacobian_visitor : std::false_type
    {};

    template <typename Flow>
    struct has_external_pressure_jacobian_visitor<
      Flow,
      std::void_t<
        decltype(std::declval<const Flow &>().visit_external_pressure_jacobian(
          std::declval<double>(),
          std::declval<const MetricFlowX::VectorType &>(),
          std::declval<const typename Flow::ExternalPressureBasisProvider &>(),
          std::declval<
            const typename Flow::ExternalPressureJacobianEntryVisitor &>()))>>
      : std::true_type
    {};
  } // namespace detail

  /** Project the covered wall reaction onto the native discontinuous basis.
   * The full nominal section is the denominator. A fully exterior section
   * contributes zero pressure, and full coverage gives the identity map.
   */
  template <typename Field>
  auto
  make_metric_flow_x_pressure_projection(
    const Field                    &field,
    const IntegrationSupport<1, 3> &overlap)
  {
    using Matrix    = MetricFlowX::MatrixType;
    auto projection = std::make_shared<Matrix>();
    auto metric     = std::make_shared<Matrix>();
    overlap.assert_matches(field.dof_handler().get_triangulation());
    const bool valid = field.finite_element().n_components() == 1 &&
                       field.finite_element().n_dofs_per_vertex() == 0 &&
                       field.constraints().n_constraints() == 0 &&
                       !field.is_reindexed() && !overlap.has_material_filter();
    AssertThrow(
      dealii::Utilities::MPI::min(static_cast<unsigned int>(valid),
                                  field.space().mpi_communicator()) != 0,
      dealii::ExcMessage(
        "Native wall-pressure projection requires the full unconstrained discontinuous scalar field."));
    const auto fallback = compatibility::gauss_quadrature<1>(
      field.finite_element().reference_cell(),
      field.finite_element().degree + 1);
    const auto                    &quadrature = overlap.quadrature(fallback);
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
    projection->reinit(field.locally_owned_dofs(),
                       field.locally_owned_dofs(),
                       sparsity,
                       field.space().mpi_communicator(),
                       false);
    metric->reinit(field.locally_owned_dofs(),
                   field.locally_owned_dofs(),
                   sparsity,
                   field.space().mpi_communicator(),
                   false);
    dealii::FEValues<1, 3> values(field.mapping(),
                                  field.finite_element(),
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_JxW_values);
    const auto             n = field.finite_element().n_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> indices(n);
    for (const auto &cell : field.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          values.reinit(cell);
          cell->get_dof_indices(indices);
          dealii::FullMatrix<double> total(n, n), covered(n, n), local(n, n);
          for (const auto q : values.quadrature_point_indices())
            for (unsigned int i = 0; i < n; ++i)
              for (unsigned int j = 0; j < n; ++j)
                {
                  const double product = values.shape_value(i, q) *
                                         values.shape_value(j, q) *
                                         values.JxW(q);
                  total(i, j) += product * overlap.full_point_weight(cell, q);
                  covered(i, j) += product * overlap.point_weight(cell, q);
                }
          field.constraints().distribute_local_to_global(total,
                                                         indices,
                                                         *metric);
          total.gauss_jordan();
          total.mmult(local, covered);
          field.constraints().distribute_local_to_global(local,
                                                         indices,
                                                         *projection);
        }
    metric->compress(dealii::VectorOperation::add);
    projection->compress(dealii::VectorOperation::add);
    return std::make_pair(projection, metric);
  }

  /** Evaluate a scalar value observable at native pressure evaluation points.
   * CellId selects the DG trace at faces and junctions. Referenced spaces must
   * outlive the provider; coefficients are copied into a relevant vector.
   */
  template <typename Pressure>
  auto
  metric_flow_x_pressure_provider(
    const Pressure                                &pressure,
    const MetricFlowX::VectorType                 &coefficients,
    std::shared_ptr<const MetricFlowX::MatrixType> projection = {})
  {
    const auto field   = pressure.source();
    auto       ghosted = std::make_shared<MetricFlowX::VectorType>();
    ghosted->reinit(field.locally_owned_dofs(),
                    field.locally_relevant_dofs(),
                    field.space().mpi_communicator());
    if (projection)
      {
        MetricFlowX::VectorType projected;
        projected.reinit(field.locally_owned_dofs(),
                         field.space().mpi_communicator());
        projection->vmult(projected, coefficients);
        *ghosted = projected;
      }
    else
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

  /** Return coefficient stencils on native pressure evaluation points.
   * Projection rows are cell-local for the discontinuous wall pressure space.
   * Their small dense blocks are exchanged with ghost cells once, rather than
   * importing a distributed matrix during every point evaluation.
   */
  template <typename Pressure>
  auto
  metric_flow_x_pressure_basis_provider(
    const Pressure                                &pressure,
    std::shared_ptr<const MetricFlowX::MatrixType> projection = {})
  {
    const auto  field = pressure.source();
    const auto &dofs  = field.dof_handler();
    const auto &fe    = field.space().finite_element();
    AssertThrow(
      fe.n_components() == 1 && fe.n_dofs_per_vertex() == 0 &&
        field.constraints().n_constraints() == 0 && !field.is_reindexed(),
      dealii::ExcMessage(
        "Native pressure coefficient assembly requires an unconstrained discontinuous scalar field."));
    using Cell  = dealii::DoFHandler<1, 3>::active_cell_iterator;
    using Block = std::vector<double>;
    auto cells  = std::make_shared<std::map<dealii::CellId, Cell>>();
    auto blocks = std::make_shared<std::map<dealii::CellId, Block>>();
    for (const auto &cell : dofs.active_cell_iterators())
      if (!cell->is_artificial())
        {
          cells->emplace(cell->id(), cell);
          if (projection && cell->is_locally_owned())
            {
              std::vector<dealii::types::global_dof_index> indices(
                fe.n_dofs_per_cell());
              cell->get_dof_indices(indices);
              auto &block = (*blocks)[cell->id()];
              block.resize(indices.size() * indices.size());
              for (unsigned int i = 0; i < indices.size(); ++i)
                {
                  for (auto entry = projection->begin(indices[i]);
                       entry != projection->end(indices[i]);
                       ++entry)
                    AssertThrow(
                      entry->value() == 0. ||
                        std::find(indices.begin(),
                                  indices.end(),
                                  entry->column()) != indices.end(),
                      dealii::ExcMessage(
                        "Native pressure projection must be cell-local."));
                  for (unsigned int j = 0; j < indices.size(); ++j)
                    block[i * indices.size() + j] =
                      projection->el(indices[i], indices[j]);
                }
            }
        }
    if (projection &&
        dynamic_cast<const dealii::parallel::TriangulationBase<1, 3> *>(
          &dofs.get_triangulation()))
      dealii::GridTools::exchange_cell_data_to_ghosts<Block>(
        dofs,
        [&](const Cell &cell) { return blocks->at(cell->id()); },
        [&](const Cell &cell, const Block &block) {
          (*blocks)[cell->id()] = block;
        });
    return [field,
            cells,
            blocks,
            projected = bool(projection),
            scale     = pressure.scale()](
             const MetricFlowX::BloodFlowSystem<1, 3>::PressureEvaluationPoint
               &point) {
      const auto &cell = cells->at(point.cell_id);
      const auto  unit =
        field.mapping().transform_real_to_unit_cell(cell, point.point);
      const auto &fe = field.space().finite_element();
      std::vector<dealii::types::global_dof_index> indices(
        fe.n_dofs_per_cell());
      cell->get_dof_indices(indices);
      std::vector<std::pair<dealii::types::global_dof_index, double>> stencil;
      for (unsigned int j = 0; j < indices.size(); ++j)
        {
          double weight = fe.shape_value(j, unit);
          if (projected)
            {
              weight            = 0.;
              const auto &block = blocks->at(point.cell_id);
              for (unsigned int i = 0; i < indices.size(); ++i)
                weight +=
                  fe.shape_value(i, unit) * block[i * indices.size() + j];
            }
          if (weight != 0.)
            stencil.emplace_back(field.execution_index(indices[j]),
                                 scale * weight);
        }
      return stencil;
    };
  }

  /** Linearize the native pressure contribution at fixed pressure coefficients.
   * HLL weights depend on the flow state even when pressure is prescribed.
   * The returned actions retain the assembled matrix for their full lifetime.
   */
  template <typename Flow, typename Pressure>
  MaterializedOperator<MetricFlowX::VectorType, MetricFlowX::MatrixType>
  metric_flow_x_pressure_state_operator(
    Flow                                          &problem,
    const Pressure                                &pressure,
    const MetricFlowX::VectorType                 &pressure_coefficients,
    const MetricFlowX::VectorType                 &flow_state,
    const double                                   time,
    std::shared_ptr<const MetricFlowX::MatrixType> projection = {})
  {
    using Vector      = MetricFlowX::VectorType;
    using Matrix      = MetricFlowX::MatrixType;
    const auto matrix = detail::clone_matrix(problem.state_jacobian_matrix());
    *matrix           = 0.;
    if constexpr (detail::has_external_pressure_state_jacobian<Flow>::value)
      problem.add_external_pressure_state_jacobian(
        time,
        flow_state,
        metric_flow_x_pressure_provider(pressure,
                                        pressure_coefficients,
                                        projection),
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
   * Native local derivative assembly constructs transpose and matrix actions.
   * Older MetricFlowX packages without the derivative visitor collectively
   * probe every global coefficient instead. The candidate flow state is owned
   * by this operator because HLL pressure weights depend on that state.
   */
  template <typename Pressure,
            typename Flow = MetricFlowX::BloodFlowSystem<1, 3>>
  MaterializedOperator<MetricFlowX::VectorType, MetricFlowX::MatrixType>
  metric_flow_x_pressure_operator(
    Flow                                          &problem,
    const Pressure                                &pressure,
    const MetricFlowX::VectorType                 &flow_state,
    const double                                   time,
    std::shared_ptr<const MetricFlowX::MatrixType> projection = {})
  {
    using Vector     = MetricFlowX::VectorType;
    using Matrix     = MetricFlowX::MatrixType;
    const auto field = pressure.source();
    const auto state = std::make_shared<Vector>(flow_state);
    const auto rows =
      std::make_shared<const dealii::IndexSet>(problem.locally_owned_dofs());
    const auto columns =
      std::make_shared<const dealii::IndexSet>(field.locally_owned_dofs());
    const auto communicator = field.space().mpi_communicator();
    MaterializedOperator<Vector, Matrix> result;
    result.view.reinit_range_vector = [&problem](Vector &dst, const bool) {
      problem.reinit_state(dst);
    };
    result.view.reinit_domain_vector = [columns, communicator](Vector &dst,
                                                               const bool) {
      dst.reinit(*columns, communicator);
    };
    const auto pressure_view = std::make_shared<const Pressure>(pressure);
    result.view.vmult =
      [&problem, pressure_view, state, time, projection](Vector       &dst,
                                                         const Vector &src) {
        // Scalar zeroing preserves a Trilinos destination's pending Insert
        // mode; native pressure assembly needs a fresh vector before adding
        // entries.
        problem.reinit_state(dst);
        dst = 0.;
        problem.add_external_pressure_residual(
          time,
          *state,
          metric_flow_x_pressure_provider(*pressure_view, src, projection),
          dst);
      };
    const auto action =
      std::make_shared<const decltype(result.view)>(result.view);
    auto cached        = std::make_shared<std::shared_ptr<Matrix>>();
    result.materialize = [action,
                          cached,
                          rows,
                          columns,
                          communicator,
                          &problem,
                          pressure_view,
                          state,
                          time,
                          projection]() {
      if (!*cached)
        {
          dealii::DynamicSparsityPattern sparsity(rows->size(),
                                                  columns->size(),
                                                  *rows);
          std::vector<std::tuple<dealii::types::global_dof_index,
                                 dealii::types::global_dof_index,
                                 double>>
            entries;
          if constexpr (detail::has_external_pressure_jacobian_visitor<
                          Flow>::value)
            problem.visit_external_pressure_jacobian(
              time,
              *state,
              metric_flow_x_pressure_basis_provider(*pressure_view, projection),
              [&](const auto i, const auto j, const double value) {
                Assert(
                  rows->is_element(i),
                  dealii::ExcMessage(
                    "Native pressure derivative rows must be locally owned."));
                if (value != 0.)
                  {
                    sparsity.add(i, j);
                    entries.emplace_back(i, j, value);
                  }
              });
          else
            {
              Vector basis, image;
              action->reinit_domain_vector(basis, false);
              action->reinit_range_vector(image, false);
              for (dealii::types::global_dof_index j = 0; j < columns->size();
                   ++j)
                {
                  basis = 0.;
                  if (columns->is_element(j))
                    basis[j] = 1.;
                  basis.compress(dealii::VectorOperation::insert);
                  action->vmult(image, basis);
                  for (const auto i : *rows)
                    if (image[i] != 0.)
                      {
                        sparsity.add(i, j);
                        entries.emplace_back(i, j, image[i]);
                      }
                }
            }
          *cached = std::make_shared<Matrix>();
          (*cached)->reinit(*rows, *columns, sparsity, communicator, false);
          for (const auto &[i, j, value] : entries)
            (*cached)->add(i, j, value);
          (*cached)->compress(dealii::VectorOperation::add);
        }
      return detail::clone_matrix(**cached);
    };
    const auto matrix =
      std::make_shared<const decltype(result.materialize)>(result.materialize);
    result.materialize_into = [matrix](Matrix &dst) {
      detail::copy_matrix(dst, *(*matrix)());
    };
    result.view.Tvmult = [matrix](Vector &dst, const Vector &src) {
      (*matrix)()->Tvmult(dst, src);
    };
    result.view.Tvmult_add = [matrix](Vector &dst, const Vector &src) {
      (*matrix)()->Tvmult_add(dst, src);
    };
    result.view.vmult_add = [action](Vector &dst, const Vector &src) {
      Vector image;
      action->reinit_range_vector(image, false);
      action->vmult(image, src);
      dst += image;
    };
    return result;
  }
} // namespace ImmersX
#endif
#endif
