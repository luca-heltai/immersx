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

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/non_matching/coupling.h>

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
    const auto preparations_per_build =
      static_cast<unsigned int>(expected_preparations);
    const auto expected = before;
    weak_term(value(source), test(target)).add(builder);
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
                  expected + preparations_per_build);
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
              expected + (4u + (change ? 1u : 0u)) * preparations_per_build);
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
    weak_term(value(source), test(target)).add(builder);

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

  template <typename ValueType>
  void
  check_transport_roundtrip()
  {
    ValueType           value;
    std::vector<double> expected;
    expected.reserve(detail::n_stored_components<ValueType>());
    for (unsigned int i = 0; i < detail::n_stored_components<ValueType>(); ++i)
      {
        const double entry = 1. + static_cast<double>(i);
        if constexpr (detail::is_tensor<ValueType>::value)
          value[ValueType::unrolled_to_component_indices(i)] = entry;
        else
          value.access_raw_entry(i) = entry;
        expected.push_back(entry);
      }

    std::vector<double> flattened;
    detail::flatten_value(value, flattened);
    EXPECT_EQ(flattened, expected);

    unsigned int offset = 0;
    const auto   roundtrip =
      detail::unflatten_value<ValueType>(flattened, offset);
    EXPECT_EQ(offset, flattened.size());
    std::vector<double> roundtrip_flattened;
    detail::flatten_value(roundtrip, roundtrip_flattened);
    EXPECT_EQ(roundtrip_flattened, expected);
  }
} // namespace

TEST(WeakTermNonmatching, TensorValueTransportRoundTripsAllComponents)
{
  check_transport_roundtrip<Tensor<2, 2>>();
  check_transport_roundtrip<Tensor<2, 3>>();
  check_transport_roundtrip<SymmetricTensor<2, 2>>();
  check_transport_roundtrip<SymmetricTensor<2, 3>>();
}

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
  weak_term(gradient(source), gradient(test(target))).add(builder);

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
      EXPECT_NEAR(actual_matrix->el(i, j), reference.el(i, j), 1.e-12);
}

TEST(WeakTermNonmatching, EmbeddedSupportMatchesDealIINonMatchingCoupling)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);
  parallel::distributed::Triangulation<2> bulk_tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bulk_tria, 0., 1.);
  bulk_tria.refine_global(1);
  bulk_tria.begin_active()->set_refine_flag();
  bulk_tria.execute_coarsening_and_refinement();
  FE_Q<2>       bulk_fe(1);
  DoFHandler<2> bulk_dof_handler(bulk_tria);
  bulk_dof_handler.distribute_dofs(bulk_fe);
  AffineConstraints<double> bulk_constraints;
  DoFTools::make_hanging_node_constraints(bulk_dof_handler, bulk_constraints);
  bulk_constraints.close();

  Triangulation<1, 2> embedded_tria;
  GridGenerator::hyper_cube(embedded_tria, 0., 1.);
  GridTools::transform(
    [](const Point<2> &point) {
      auto shifted = point;
      shifted[1] += 0.5;
      return shifted;
    },
    embedded_tria);
  embedded_tria.refine_global(2);
  FE_Q<1, 2>       embedded_fe(1);
  DoFHandler<1, 2> embedded_dof_handler(embedded_tria);
  embedded_dof_handler.distribute_dofs(embedded_fe);
  AffineConstraints<double> embedded_constraints;
  embedded_constraints.close();

  const auto bulk_view =
    fe_space(bulk_dof_handler, StaticMappingQ1<2>::mapping, bulk_constraints);
  const auto             embedded_view = fe_space(embedded_dof_handler,
                                      StaticMappingQ1<1, 2>::mapping,
                                      embedded_constraints);
  GridTools::Cache<2, 2> cache(bulk_tria, StaticMappingQ1<2>::mapping);
  const QGauss<1>        quadrature(2);
  DynamicSparsityPattern native_sparsity(bulk_dof_handler.n_dofs(),
                                         embedded_dof_handler.n_dofs());
  NonMatching::create_coupling_sparsity_pattern(cache,
                                                bulk_dof_handler,
                                                embedded_dof_handler,
                                                quadrature,
                                                native_sparsity,
                                                bulk_constraints,
                                                ComponentMask(),
                                                ComponentMask(),
                                                StaticMappingQ1<1, 2>::mapping,
                                                embedded_constraints);
  SparsityPattern native_pattern;
  native_pattern.copy_from(native_sparsity);
  SparseMatrix<double> native_matrix(native_pattern);
  NonMatching::create_coupling_mass_matrix(cache,
                                           bulk_dof_handler,
                                           embedded_dof_handler,
                                           quadrature,
                                           native_matrix,
                                           bulk_constraints,
                                           ComponentMask(),
                                           ComponentMask(),
                                           StaticMappingQ1<1, 2>::mapping,
                                           embedded_constraints);

  using Vector = ImmersXLA::MPI::Vector;
  using Matrix = ImmersXLA::MPI::SparseMatrix;
  using Model  = SemiDiscreteModel<Vector, Matrix>;
  const IntegrationSupport<1, 2> support(embedded_tria);

  StateLayout reverse_layout;
  const auto  bulk_source = bulk_view.field(reverse_layout, "bulk-source");
  const auto  embedded_target =
    embedded_view.field(reverse_layout, "embedded-target");
  Model                               reverse_model;
  SemidiscreteBuilder<Vector, Matrix> reverse_builder(reverse_layout,
                                                      reverse_model);
  weak_term(value(bulk_source), test(embedded_target), support)
    .add(reverse_builder);
  StateView<Vector>               reverse_state(reverse_layout, 0.);
  const EvaluationContext<Vector> reverse_context(0., reverse_state);
  const auto                      reverse_matrix =
    reverse_model.state_matrix_operator(embedded_target.field_id(),
                                        bulk_source.field_id(),
                                        reverse_context);
  ASSERT_TRUE(reverse_matrix.has_value());
  const auto reverse_native = reverse_matrix->matrix();
  ASSERT_EQ(reverse_native->m(), native_matrix.n());
  ASSERT_EQ(reverse_native->n(), native_matrix.m());
  for (unsigned int i = 0; i < reverse_native->m(); ++i)
    for (unsigned int j = 0; j < reverse_native->n(); ++j)
      EXPECT_NEAR(reverse_native->el(i, j), native_matrix.el(j, i), 1.e-12);

  StateLayout forward_layout;
  const auto  embedded_source =
    embedded_view.field(forward_layout, "embedded-source");
  const auto bulk_target = bulk_view.field(forward_layout, "bulk-target");
  Model      forward_model;
  SemidiscreteBuilder<Vector, Matrix> forward_builder(forward_layout,
                                                      forward_model);
  weak_term(value(embedded_source), test(bulk_target), support)
    .add(forward_builder);
  StateView<Vector>               forward_state(forward_layout, 0.);
  const EvaluationContext<Vector> forward_context(0., forward_state);
  const auto                      forward_matrix =
    forward_model.state_matrix_operator(bulk_target.field_id(),
                                        embedded_source.field_id(),
                                        forward_context);
  ASSERT_TRUE(forward_matrix.has_value());
  const auto forward_native = forward_matrix->matrix();
  ASSERT_EQ(forward_native->m(), native_matrix.m());
  ASSERT_EQ(forward_native->n(), native_matrix.n());
  for (unsigned int i = 0; i < forward_native->m(); ++i)
    for (unsigned int j = 0; j < forward_native->n(); ++j)
      EXPECT_NEAR(forward_native->el(i, j), native_matrix.el(i, j), 1.e-12);

  FESystem<2>   vector_bulk_fe(FE_Q<2>(1), 2);
  DoFHandler<2> vector_bulk_dof_handler(bulk_tria);
  vector_bulk_dof_handler.distribute_dofs(vector_bulk_fe);
  AffineConstraints<double> vector_bulk_constraints;
  DoFTools::make_hanging_node_constraints(vector_bulk_dof_handler,
                                          vector_bulk_constraints);
  vector_bulk_constraints.close();
  FESystem<1, 2>   vector_embedded_fe(FE_Q<1, 2>(1), 2);
  DoFHandler<1, 2> vector_embedded_dof_handler(embedded_tria);
  vector_embedded_dof_handler.distribute_dofs(vector_embedded_fe);
  AffineConstraints<double> vector_embedded_constraints;
  vector_embedded_constraints.close();
  const auto  vector_bulk_view     = fe_space(vector_bulk_dof_handler,
                                         StaticMappingQ1<2>::mapping,
                                         vector_bulk_constraints);
  const auto  vector_embedded_view = fe_space(vector_embedded_dof_handler,
                                             StaticMappingQ1<1, 2>::mapping,
                                             vector_embedded_constraints);
  StateLayout vector_layout;
  const auto  vector_bulk =
    vector_bulk_view.field(vector_layout,
                           "vector-bulk",
                           FEValuesExtractors::Vector(0));
  const auto vector_embedded =
    vector_embedded_view.field(vector_layout,
                               "vector-embedded",
                               FEValuesExtractors::Vector(0));
  // Both spaces represent ambient 2D vectors. Do not use the embedded FE's
  // dim-sized vector mask or the implementation's Field mask in this oracle.
  const ComponentMask    ambient_vector_mask(2, true);
  DynamicSparsityPattern vector_native_sparsity(
    vector_bulk_dof_handler.n_dofs(), vector_embedded_dof_handler.n_dofs());
  NonMatching::create_coupling_sparsity_pattern(cache,
                                                vector_bulk_dof_handler,
                                                vector_embedded_dof_handler,
                                                quadrature,
                                                vector_native_sparsity,
                                                vector_bulk_constraints,
                                                ambient_vector_mask,
                                                ambient_vector_mask,
                                                StaticMappingQ1<1, 2>::mapping,
                                                vector_embedded_constraints);
  SparsityPattern vector_native_pattern;
  vector_native_pattern.copy_from(vector_native_sparsity);
  SparseMatrix<double> vector_native_matrix(vector_native_pattern);
  NonMatching::create_coupling_mass_matrix(cache,
                                           vector_bulk_dof_handler,
                                           vector_embedded_dof_handler,
                                           quadrature,
                                           vector_native_matrix,
                                           vector_bulk_constraints,
                                           ambient_vector_mask,
                                           ambient_vector_mask,
                                           StaticMappingQ1<1, 2>::mapping,
                                           vector_embedded_constraints);

  Model                               vector_reverse_model;
  SemidiscreteBuilder<Vector, Matrix> vector_reverse_builder(
    vector_layout, vector_reverse_model);
  weak_term(value(vector_bulk), test(vector_embedded), support)
    .add(vector_reverse_builder);
  StateView<Vector>               vector_reverse_state(vector_layout, 0.);
  const EvaluationContext<Vector> vector_reverse_context(0.,
                                                         vector_reverse_state);
  const auto                      vector_reverse_matrix =
    vector_reverse_model.state_matrix_operator(vector_embedded.field_id(),
                                               vector_bulk.field_id(),
                                               vector_reverse_context);
  ASSERT_TRUE(vector_reverse_matrix.has_value());
  for (unsigned int i = 0; i < vector_reverse_matrix->matrix()->m(); ++i)
    for (unsigned int j = 0; j < vector_reverse_matrix->matrix()->n(); ++j)
      EXPECT_NEAR(vector_reverse_matrix->matrix()->el(i, j),
                  vector_native_matrix.el(j, i),
                  1.e-12);

  Model                               vector_forward_model;
  SemidiscreteBuilder<Vector, Matrix> vector_forward_builder(
    vector_layout, vector_forward_model);
  weak_term(value(vector_embedded), test(vector_bulk), support)
    .add(vector_forward_builder);
  StateView<Vector>               vector_forward_state(vector_layout, 0.);
  const EvaluationContext<Vector> vector_forward_context(0.,
                                                         vector_forward_state);
  const auto                      vector_forward_matrix =
    vector_forward_model.state_matrix_operator(vector_bulk.field_id(),
                                               vector_embedded.field_id(),
                                               vector_forward_context);
  ASSERT_TRUE(vector_forward_matrix.has_value());
  for (unsigned int i = 0; i < vector_forward_matrix->matrix()->m(); ++i)
    for (unsigned int j = 0; j < vector_forward_matrix->matrix()->n(); ++j)
      EXPECT_NEAR(vector_forward_matrix->matrix()->el(i, j),
                  vector_native_matrix.el(i, j),
                  1.e-12);
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
  weak_term(value(base_source), test(base_target)).add(base_builder);

  StateLayout scaled_layout;
  const auto  scaled_source = source_view.field(scaled_layout, "scaled-source");
  const auto  scaled_target = target_view.field(scaled_layout, "scaled-target");
  Model       scaled_model;
  SemidiscreteBuilder<Vector, Matrix> scaled_builder(scaled_layout,
                                                     scaled_model);
  weak_term(2.0 * value(scaled_source), 3.0 * test(scaled_target))
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
