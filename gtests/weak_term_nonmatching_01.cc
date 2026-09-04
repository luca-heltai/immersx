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
