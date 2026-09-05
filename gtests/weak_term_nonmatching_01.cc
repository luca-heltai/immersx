// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <gtest/gtest.h>
#include <immersx/core/contributor.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/state.h>
#include <immersx/core/weak_term.h>

#include <cmath>
#include <functional>

using namespace dealii;
using namespace ImmersX;

namespace
{
  struct DistributedScalarSpace
  {
    DistributedScalarSpace(parallel::distributed::Triangulation<2> &tria,
                           const FiniteElement<2>                  &fe,
                           const unsigned int                       refinements)
      : dof_handler(tria)
    {
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(refinements);
      dof_handler.distribute_dofs(fe);
      constraints.close();
    }

    DoFHandler<2>             dof_handler;
    AffineConstraints<double> constraints;
  };

  template <typename SourceField, typename TargetField>
  void
  check_nonmatching_scalar_pairing(SourceField  source,
                                   TargetField  target,
                                   const double expected_preparations,
                                   const std::function<void()> &change = {})
  {
    using Vector = ImmersXLA::MPI::Vector;
    using Matrix = ImmersXLA::MPI::SparseMatrix;
    using Model  = SemiDiscreteModel<Vector, Matrix>;

    StateLayout layout;
    source = source.space().field(layout, "source");
    target = target.space().field(layout, "target");

    Model                               model;
    SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
    const auto before = detail::weak_term_nonmatching_preparations.load();
    const auto expected =
      before + static_cast<unsigned int>(expected_preparations);
    weak_term(value(source), target).add(builder);
    EXPECT_EQ(detail::weak_term_nonmatching_preparations.load(), expected);

    Vector state;
    state.reinit(source.locally_owned_dofs(), MPI_COMM_WORLD);
    for (const auto index : state.locally_owned_elements())
      state[index] = 1. + std::sin(static_cast<double>(index));
    state.compress(VectorOperation::insert);

    StateView<Vector> state_view(layout, 0.);
    state_view.bind(source.field_id(), state);
    const EvaluationContext<Vector> context(0., state_view);

    if (change)
      {
        change();
        Vector changed_residual;
        changed_residual.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
        model.evaluate_row(target.field_id(), context, changed_residual);
        EXPECT_EQ(detail::weak_term_nonmatching_preparations.load(),
                  expected + 1u);
      }

    Vector residual;
    residual.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
    model.evaluate_row(target.field_id(), context, residual);

    const auto matrix = model.state_matrix_operator(target.field_id(),
                                                    source.field_id(),
                                                    context);
    ASSERT_TRUE(matrix.has_value());
    Vector action;
    action.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
    matrix->view.vmult(action, state);
    Vector forward = action;
    action -= residual;
    EXPECT_LT(action.l2_norm(), 1.e-12);

    Vector dual;
    dual.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
    for (const auto index : dual.locally_owned_elements())
      dual[index] = 2. - 0.25 * static_cast<double>(index);
    dual.compress(VectorOperation::insert);
    Vector transpose;
    transpose.reinit(source.locally_owned_dofs(), MPI_COMM_WORLD);
    matrix->view.Tvmult(transpose, dual);
    EXPECT_NEAR(forward * dual, state * transpose, 1.e-11);

    Vector second_residual;
    second_residual.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
    model.evaluate_row(target.field_id(), context, second_residual);
    second_residual -= residual;
    EXPECT_LT(second_residual.l2_norm(), 1.e-12);
    EXPECT_EQ(detail::weak_term_nonmatching_preparations.load(),
              expected + (change ? 1u : 0u));
  }

  template <typename SourceField, typename TargetField>
  void
  check_nonmatching_vector_pairing(SourceField source, TargetField target)
  {
    using Vector = ImmersXLA::MPI::Vector;
    using Matrix = ImmersXLA::MPI::SparseMatrix;
    using Model  = SemiDiscreteModel<Vector, Matrix>;

    StateLayout layout;
    source = source.space().field(layout, "source", source.extractor());
    target = target.space().field(layout, "target", target.extractor());

    Model                               model;
    SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
    weak_term(value(source), target).add(builder);

    Vector state;
    state.reinit(source.locally_owned_dofs(), MPI_COMM_WORLD);
    for (const auto index : state.locally_owned_elements())
      state[index] = 0.5 + static_cast<double>(index);
    state.compress(VectorOperation::insert);

    StateView<Vector> state_view(layout, 0.);
    state_view.bind(source.field_id(), state);
    const EvaluationContext<Vector> context(0., state_view);
    Vector                          residual;
    residual.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
    model.evaluate_row(target.field_id(), context, residual);
    EXPECT_GT(residual.l2_norm(), 0.);

    const auto matrix = model.state_matrix_operator(target.field_id(),
                                                    source.field_id(),
                                                    context);
    ASSERT_TRUE(matrix.has_value());
    Vector action;
    action.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
    matrix->view.vmult(action, state);
    action -= residual;
    EXPECT_LT(action.l2_norm(), 1.e-12);
  }
} // namespace

TEST(WeakTermNonmatching, NonmatchingScalarDifferentDegrees)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);
  parallel::distributed::Triangulation<2> source_tria(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<2> target_tria(MPI_COMM_WORLD);
  FE_Q<2>                                 source_fe(1);
  FE_Q<2>                                 target_fe(2);
  DistributedScalarSpace source_space(source_tria, source_fe, 1);
  DistributedScalarSpace target_space(target_tria, target_fe, 2);
  const auto             source_view = fe_space(source_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    source_space.constraints);
  const auto             target_view = fe_space(target_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    target_space.constraints);
  const auto             source      = source_view.field("source");
  const auto             target      = target_view.field("target");
  check_nonmatching_scalar_pairing(source, target, 1., [&source_tria] {
    GridTools::transform(
      [](const Point<2> &point) {
        auto moved = point;
        moved[0] += 0.01;
        return moved;
      },
      source_tria);
  });
}

TEST(WeakTermNonmatching, NonmatchingScalarGradientUsesPointSearch)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);
  parallel::distributed::Triangulation<2> source_tria(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<2> target_tria(MPI_COMM_WORLD);
  FE_Q<2>                                 source_fe(1);
  FE_Q<2>                                 target_fe(2);
  DistributedScalarSpace source_space(source_tria, source_fe, 1);
  DistributedScalarSpace target_space(target_tria, target_fe, 2);
  const auto             source_view = fe_space(source_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    source_space.constraints);
  const auto             target_view = fe_space(target_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    target_space.constraints);

  StateLayout layout;
  const auto  source = source_view.field(layout, "source");
  const auto  target = target_view.field(layout, "target");
  using Vector       = ImmersXLA::MPI::Vector;
  using Matrix       = ImmersXLA::MPI::SparseMatrix;
  using Model        = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  weak_term(gradient(source), gradient(target)).add(builder);

  StateView<Vector>               state_view(layout, 0.);
  const EvaluationContext<Vector> context(0., state_view);
  const auto                      actual =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(actual.has_value());
  ASSERT_TRUE(actual->is_materializable());

  const QGauss<2>        quadrature(target_fe.degree + 1);
  DynamicSparsityPattern reference_sparsity(target_space.dof_handler.n_dofs(),
                                            source_space.dof_handler.n_dofs());
  for (unsigned int i = 0; i < target_space.dof_handler.n_dofs(); ++i)
    for (unsigned int j = 0; j < source_space.dof_handler.n_dofs(); ++j)
      reference_sparsity.add(i, j);
  SparsityPattern reference_pattern;
  reference_pattern.copy_from(reference_sparsity);
  SparseMatrix<double> reference(reference_pattern);

  FEValues<2> target_values(StaticMappingQ1<2>::mapping,
                            target_fe,
                            quadrature,
                            update_gradients | update_JxW_values |
                              update_quadrature_points);
  const auto &target_view_values = target_values[FEValuesExtractors::Scalar(0)];
  std::vector<types::global_dof_index> target_indices(
    target_fe.n_dofs_per_cell());
  std::vector<types::global_dof_index> source_indices(
    source_fe.n_dofs_per_cell());
  for (const auto &target_cell :
       target_space.dof_handler.active_cell_iterators())
    {
      if (!target_cell->is_locally_owned())
        continue;
      target_values.reinit(target_cell);
      target_cell->get_dof_indices(target_indices);
      for (const unsigned int q : target_values.quadrature_point_indices())
        {
          const auto source_cell_and_reference =
            GridTools::find_active_cell_around_point(
              StaticMappingQ1<2>::mapping,
              source_space.dof_handler,
              target_values.quadrature_point(q));
          ASSERT_NE(source_cell_and_reference.first,
                    source_space.dof_handler.end());

          const Quadrature<2> source_quadrature(
            std::vector<Point<2>>{source_cell_and_reference.second});
          FEValues<2> source_values(StaticMappingQ1<2>::mapping,
                                    source_fe,
                                    source_quadrature,
                                    update_gradients);
          source_values.reinit(source_cell_and_reference.first);
          source_cell_and_reference.first->get_dof_indices(source_indices);
          const auto &source_view_values =
            source_values[FEValuesExtractors::Scalar(0)];
          for (unsigned int i = 0; i < target_indices.size(); ++i)
            for (unsigned int j = 0; j < source_indices.size(); ++j)
              reference.add(
                target_indices[i],
                source_indices[j],
                dealii::scalar_product(source_view_values.gradient(j, 0),
                                       target_view_values.gradient(i, q)) *
                  target_values.JxW(q));
        }
    }

  const auto actual_matrix = actual->matrix();
  ASSERT_EQ(actual_matrix->m(), reference.m());
  ASSERT_EQ(actual_matrix->n(), reference.n());
  for (unsigned int i = 0; i < reference.m(); ++i)
    for (unsigned int j = 0; j < reference.n(); ++j)
      EXPECT_NEAR((*actual_matrix)(i, j), reference(i, j), 1.e-12);
}

TEST(WeakTermNonmatching, MPI_NonmatchingPartitionAndTranspose)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  parallel::distributed::Triangulation<2> source_tria(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<2> target_tria(MPI_COMM_WORLD);
  FE_Q<2>                                 source_fe(1);
  FE_Q<2>                                 target_fe(2);
  DistributedScalarSpace source_space(source_tria, source_fe, 2);
  DistributedScalarSpace target_space(target_tria, target_fe, 3);
  const auto             source_view = fe_space(source_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    source_space.constraints);
  const auto             target_view = fe_space(target_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    target_space.constraints);
  const auto             source      = source_view.field("source");
  const auto             target      = target_view.field("target");
  check_nonmatching_scalar_pairing(source, target, 1.);
}

TEST(WeakTermNonmatching, MPI_ScaledPairing)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  parallel::distributed::Triangulation<2> source_tria(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<2> target_tria(MPI_COMM_WORLD);
  FE_Q<2>                                 source_fe(1);
  FE_Q<2>                                 target_fe(2);
  DistributedScalarSpace source_space(source_tria, source_fe, 2);
  DistributedScalarSpace target_space(target_tria, target_fe, 3);
  const auto             source_view = fe_space(source_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    source_space.constraints);
  const auto             target_view = fe_space(target_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    target_space.constraints);

  using Vector = ImmersXLA::MPI::Vector;
  using Matrix = ImmersXLA::MPI::SparseMatrix;
  using Model  = SemiDiscreteModel<Vector, Matrix>;

  StateLayout base_layout;
  const auto  base_source = source_view.field(base_layout, "base-source");
  const auto  base_target = target_view.field(base_layout, "base-target");
  Model       base_model;
  SemidiscreteBuilder<Vector, Matrix> base_builder(base_layout, base_model);
  weak_term(value(base_source), base_target).add(base_builder);

  StateLayout scaled_layout;
  const auto  scaled_source = source_view.field(scaled_layout, "scaled-source");
  const auto  scaled_target = target_view.field(scaled_layout, "scaled-target");
  Model       scaled_model;
  SemidiscreteBuilder<Vector, Matrix> scaled_builder(scaled_layout,
                                                     scaled_model);
  weak_term(2.0 * value(scaled_source), 3.0 * value(scaled_target))
    .add(scaled_builder);

  Vector base_state;
  Vector scaled_state;
  base_state.reinit(base_source.locally_owned_dofs(), MPI_COMM_WORLD);
  scaled_state.reinit(scaled_source.locally_owned_dofs(), MPI_COMM_WORLD);
  for (const auto index : base_state.locally_owned_elements())
    {
      const double value  = 0.5 + static_cast<double>(index);
      base_state[index]   = value;
      scaled_state[index] = value;
    }
  base_state.compress(VectorOperation::insert);
  scaled_state.compress(VectorOperation::insert);

  StateView<Vector> base_view(base_layout, 0.);
  StateView<Vector> scaled_view(scaled_layout, 0.);
  base_view.bind(base_source.field_id(), base_state);
  scaled_view.bind(scaled_source.field_id(), scaled_state);
  const EvaluationContext<Vector> base_context(0., base_view);
  const EvaluationContext<Vector> scaled_context(0., scaled_view);

  Vector base_residual;
  Vector scaled_residual;
  base_residual.reinit(base_target.locally_owned_dofs(), MPI_COMM_WORLD);
  scaled_residual.reinit(scaled_target.locally_owned_dofs(), MPI_COMM_WORLD);
  base_model.evaluate_row(base_target.field_id(), base_context, base_residual);
  scaled_model.evaluate_row(scaled_target.field_id(),
                            scaled_context,
                            scaled_residual);
  scaled_residual.add(-6., base_residual);
  EXPECT_LT(scaled_residual.l2_norm(), 1.e-12);

  const auto base_matrix =
    base_model.state_matrix_operator(base_target.field_id(),
                                     base_source.field_id(),
                                     base_context);
  const auto scaled_matrix =
    scaled_model.state_matrix_operator(scaled_target.field_id(),
                                       scaled_source.field_id(),
                                       scaled_context);
  ASSERT_TRUE(base_matrix.has_value());
  ASSERT_TRUE(scaled_matrix.has_value());
  Vector base_action;
  Vector scaled_action;
  base_action.reinit(base_target.locally_owned_dofs(), MPI_COMM_WORLD);
  scaled_action.reinit(scaled_target.locally_owned_dofs(), MPI_COMM_WORLD);
  base_matrix->view.vmult(base_action, base_state);
  scaled_matrix->view.vmult(scaled_action, scaled_state);
  scaled_action.add(-6., base_action);
  EXPECT_LT(scaled_action.l2_norm(), 1.e-12);
}

TEST(WeakTermNonmatching, NonmatchingVectorPairing)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);
  parallel::distributed::Triangulation<2> source_tria(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<2> target_tria(MPI_COMM_WORLD);
  FESystem<2>                             source_fe(FE_Q<2>(1), 2);
  FESystem<2>                             target_fe(FE_Q<2>(2), 2);
  DistributedScalarSpace source_space(source_tria, source_fe, 1);
  DistributedScalarSpace target_space(target_tria, target_fe, 2);
  const auto             source_view = fe_space(source_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    source_space.constraints);
  const auto             target_view = fe_space(target_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    target_space.constraints);
  const auto             source =
    source_view.field("source", FEValuesExtractors::Vector(0));
  const auto target =
    target_view.field("target", FEValuesExtractors::Vector(0));
  check_nonmatching_vector_pairing(source, target);
}
