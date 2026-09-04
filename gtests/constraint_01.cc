// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/core/constraint.h>
#include <immersx/core/contributor.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/state.h>

using namespace dealii;
using namespace ImmersX;

namespace
{
  template <typename FiniteElement>
  struct Space
  {
    Space(Triangulation<2> &tria, const FiniteElement &finite_element)
      : dof_handler(tria)
    {
      dof_handler.distribute_dofs(finite_element);
      constraints.close();
    }

    DoFHandler<2>             dof_handler;
    AffineConstraints<double> constraints;
  };

  template <bool with_rhs, typename ValueType, typename Extractor>
  void
  check_constraint(const Extractor &extractor)
  {
    Triangulation<2> tria;
    GridGenerator::hyper_cube(tria);
    tria.refine_global(1);

    using FiniteElement = std::
      conditional_t<std::is_same_v<ValueType, double>, FE_Q<2>, FESystem<2>>;
    const FiniteElement source_fe = [&] {
      if constexpr (std::is_same_v<ValueType, double>)
        return FiniteElement(1);
      else
        return FiniteElement(FE_Q<2>(1), 2);
    }();
    const FiniteElement target_fe = [&] {
      if constexpr (std::is_same_v<ValueType, double>)
        return FiniteElement(2);
      else
        return FiniteElement(FE_Q<2>(2), 2);
    }();

    Space<FiniteElement> source_space_1(tria, source_fe);
    Space<FiniteElement> source_space_2(tria, source_fe);
    Space<FiniteElement> multiplier_space(tria, target_fe);

    const auto source_view_1   = fe_space(source_space_1.dof_handler,
                                        StaticMappingQ1<2>::mapping,
                                        source_space_1.constraints);
    const auto source_view_2   = fe_space(source_space_2.dof_handler,
                                        StaticMappingQ1<2>::mapping,
                                        source_space_2.constraints);
    const auto multiplier_view = fe_space(multiplier_space.dof_handler,
                                          StaticMappingQ1<2>::mapping,
                                          multiplier_space.constraints);

    StateLayout layout;
    const auto  source_1 = source_view_1.field(layout, "u1", extractor);
    const auto  source_2 = source_view_2.field(layout, "u2", extractor);
    const auto  lambda   = multiplier_view.field("lambda", extractor);

    using Vector = Vector<double>;
    using Matrix = SparseMatrix<double>;
    using Model  = SemiDiscreteModel<Vector, Matrix>;

    Model                               model;
    SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
    const auto                          c1 = weak_term(value(source_1), lambda);
    const auto                          c2 = weak_term(value(source_2), lambda);
#ifdef IMMERSX_WEAK_TERM_TESTING
    const auto preparations = detail::weak_term_nonmatching_preparations.load();
#endif
    Vector prescribed(multiplier_space.dof_handler.n_dofs());
    for (unsigned int i = 0; i < prescribed.size(); ++i)
      prescribed[i] = 0.5 * i;
    const auto fields = [&] {
      if constexpr (with_rhs)
        return make_constraint(c1 - c2, prescribed).add(builder);
      else
        return make_constraint(c1 - c2).add(builder);
    }();
#ifdef IMMERSX_WEAK_TERM_TESTING
    EXPECT_EQ(detail::weak_term_nonmatching_preparations.load(), preparations);
#endif

    EXPECT_TRUE(fields.multiplier.is_valid());
    EXPECT_EQ(layout.field(fields.multiplier).locally_owned.size(),
              multiplier_space.dof_handler.n_dofs());
    ASSERT_EQ(model.saddle_points().size(), 1u);
    EXPECT_EQ(model.saddle_points().front().multiplier, fields.multiplier);
    EXPECT_EQ(model.saddle_points().front().participants.size(), 2u);
    EXPECT_TRUE(model.has_multiplier_metric(fields.multiplier));

    Vector u1(source_space_1.dof_handler.n_dofs());
    Vector u2(source_space_2.dof_handler.n_dofs());
    Vector l(multiplier_space.dof_handler.n_dofs());
    for (unsigned int i = 0; i < u1.size(); ++i)
      u1[i] = 1. + i;
    for (unsigned int i = 0; i < u2.size(); ++i)
      u2[i] = 2. - 0.5 * i;
    for (unsigned int i = 0; i < l.size(); ++i)
      l[i] = -1. + 0.25 * i;

    StateView<Vector> state_view(layout, 0.);
    state_view.bind(source_1.field_id(), u1);
    state_view.bind(source_2.field_id(), u2);
    state_view.bind(fields.multiplier, l);
    const EvaluationContext<Vector> context(0., state_view);
    const auto metric = model.multiplier_metric(fields.multiplier, context);
    ASSERT_TRUE(metric.has_value());
    ASSERT_TRUE(metric->is_materializable());
    EXPECT_GT(metric->matrix()->frobenius_norm(), 1.e-12);

    const auto b1 = model.state_matrix_operator(fields.multiplier,
                                                source_1.field_id(),
                                                context);
    const auto b2 = model.state_matrix_operator(fields.multiplier,
                                                source_2.field_id(),
                                                context);
    ASSERT_TRUE(b1.has_value());
    ASSERT_TRUE(b2.has_value());

    Vector expected_constraint(l.size());
    b1->view.vmult(expected_constraint, u1);
    Vector contribution(l.size());
    b2->view.vmult(contribution, u2);
    expected_constraint += contribution;
    if constexpr (with_rhs)
      expected_constraint -= prescribed;

    Vector constraint_residual(l.size());
    model.evaluate_row(fields.multiplier, context, constraint_residual);
    constraint_residual -= expected_constraint;
    EXPECT_LT(constraint_residual.l2_norm(), 1.e-12);

    const auto reaction = model.state_matrix_operator(source_1.field_id(),
                                                      fields.multiplier,
                                                      context);
    ASSERT_TRUE(reaction.has_value());
    Vector expected_reaction(u1.size());
    b1->view.Tvmult(expected_reaction, l);
    Vector reaction_residual(u1.size());
    model.evaluate_row(source_1.field_id(), context, reaction_residual);
    reaction_residual -= expected_reaction;
    EXPECT_LT(reaction_residual.l2_norm(), 1.e-12);
  }
} // namespace

TEST(Constraint, IndependentScalarMultiplierSpace)
{
  check_constraint<false, double>(FEValuesExtractors::Scalar(0));
}

TEST(Constraint, IndependentVectorMultiplierSpace)
{
  check_constraint<false, Tensor<1, 2>>(FEValuesExtractors::Vector(0));
}

TEST(Constraint, FrozenScalarRightHandSide)
{
  check_constraint<true, double>(FEValuesExtractors::Scalar(0));
}


TEST(Constraint, SingleTerm)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);

  FE_Q<2>       source_fe(1);
  FE_Q<2>       multiplier_fe(2);
  DoFHandler<2> source_dh(tria);
  DoFHandler<2> multiplier_dh(tria);
  source_dh.distribute_dofs(source_fe);
  multiplier_dh.distribute_dofs(multiplier_fe);
  AffineConstraints<double> source_constraints;
  AffineConstraints<double> multiplier_constraints;
  source_constraints.close();
  multiplier_constraints.close();

  const auto source_view =
    fe_space(source_dh, StaticMappingQ1<2>::mapping, source_constraints);
  const auto  multiplier_view = fe_space(multiplier_dh,
                                        StaticMappingQ1<2>::mapping,
                                        multiplier_constraints);
  StateLayout layout;
  const auto  source = source_view.field(layout, "source");
  const auto  lambda = multiplier_view.field("lambda");

  using Vector = Vector<double>;
  using Matrix = SparseMatrix<double>;
  using Model  = SemiDiscreteModel<Vector, Matrix>;
  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  const auto                          fields =
    make_constraint(weak_term(value(source), lambda)).add(builder);

  ASSERT_TRUE(fields.multiplier.is_valid());
  ASSERT_EQ(model.saddle_points().size(), 1u);
  ASSERT_EQ(model.saddle_points().front().participants.size(), 1u);

  Vector source_state(source_dh.n_dofs());
  Vector lambda_state(multiplier_dh.n_dofs());
  source_state = 1.;
  lambda_state = 0.25;
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), source_state);
  state_view.bind(fields.multiplier, lambda_state);
  const EvaluationContext<Vector> context(0., state_view);

  const auto pairing =
    model.state_matrix_operator(fields.multiplier, source.field_id(), context);
  ASSERT_TRUE(pairing.has_value());
  Vector expected(lambda_state.size());
  pairing->view.vmult(expected, source_state);
  Vector residual(lambda_state.size());
  model.evaluate_row(fields.multiplier, context, residual);
  residual -= expected;
  EXPECT_LT(residual.l2_norm(), 1.e-12);
}

TEST(Constraint, NonmatchingGeometryUsesCachedBackend)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);

  parallel::distributed::Triangulation<2> source_tria(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<2> second_source_tria(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<2> multiplier_tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(source_tria);
  GridGenerator::hyper_cube(second_source_tria);
  GridGenerator::hyper_cube(multiplier_tria);
  source_tria.refine_global(1);
  second_source_tria.refine_global(1);
  multiplier_tria.refine_global(2);

  FE_Q<2>       source_fe(1);
  FE_Q<2>       multiplier_fe(1);
  DoFHandler<2> source_dh(source_tria);
  DoFHandler<2> second_source_dh(second_source_tria);
  DoFHandler<2> multiplier_dh(multiplier_tria);
  source_dh.distribute_dofs(source_fe);
  second_source_dh.distribute_dofs(source_fe);
  multiplier_dh.distribute_dofs(multiplier_fe);
  AffineConstraints<double> source_constraints;
  AffineConstraints<double> second_source_constraints;
  AffineConstraints<double> multiplier_constraints;
  source_constraints.close();
  second_source_constraints.close();
  multiplier_constraints.close();

  const auto source_view =
    fe_space(source_dh, StaticMappingQ1<2>::mapping, source_constraints);
  const auto  second_source_view = fe_space(second_source_dh,
                                           StaticMappingQ1<2>::mapping,
                                           second_source_constraints);
  const auto  multiplier_view    = fe_space(multiplier_dh,
                                        StaticMappingQ1<2>::mapping,
                                        multiplier_constraints);
  StateLayout layout;
  const auto  source        = source_view.field(layout, "source");
  const auto  second_source = second_source_view.field(layout, "second_source");
  const auto  lambda        = multiplier_view.field("lambda");

  using Vector = ImmersXLA::MPI::Vector;
  using Matrix = ImmersXLA::MPI::SparseMatrix;
  using Model  = SemiDiscreteModel<Vector, Matrix>;
  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  const auto                          c1 = weak_term(value(source), lambda);
  const auto c2 = weak_term(value(second_source), lambda);
#ifdef IMMERSX_WEAK_TERM_TESTING
  const auto preparations = detail::weak_term_nonmatching_preparations.load();
#endif
  const auto fields = make_constraint(c1 - c2).add(builder);
#ifdef IMMERSX_WEAK_TERM_TESTING
  EXPECT_EQ(detail::weak_term_nonmatching_preparations.load(),
            preparations + 2);
#endif

  Vector source_state;
  Vector second_source_state;
  Vector lambda_state;
  source_state.reinit(source.locally_owned_dofs(), MPI_COMM_WORLD);
  second_source_state.reinit(second_source.locally_owned_dofs(),
                             MPI_COMM_WORLD);
  lambda_state.reinit(lambda.locally_owned_dofs(), MPI_COMM_WORLD);
  source_state        = 1.;
  second_source_state = 2.;
  lambda_state        = 0.25;
  source_state.compress(VectorOperation::insert);
  second_source_state.compress(VectorOperation::insert);
  lambda_state.compress(VectorOperation::insert);

  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), source_state);
  state_view.bind(second_source.field_id(), second_source_state);
  state_view.bind(fields.multiplier, lambda_state);
  const EvaluationContext<Vector> context(0., state_view);
  Vector                          residual;
  residual.reinit(lambda.locally_owned_dofs(), MPI_COMM_WORLD);
  model.evaluate_row(fields.multiplier, context, residual);
  Vector repeated_residual;
  repeated_residual.reinit(lambda.locally_owned_dofs(), MPI_COMM_WORLD);
  model.evaluate_row(fields.multiplier, context, repeated_residual);
  repeated_residual -= residual;
  EXPECT_LT(repeated_residual.l2_norm(), 1.e-12);
  const auto pairing =
    model.state_matrix_operator(fields.multiplier, source.field_id(), context);
  const auto reaction =
    model.state_matrix_operator(source.field_id(), fields.multiplier, context);
  ASSERT_TRUE(pairing.has_value());
  ASSERT_TRUE(reaction.has_value());
  Vector expected_reaction;
  expected_reaction.reinit(source.locally_owned_dofs(), MPI_COMM_WORLD);
  pairing->view.Tvmult(expected_reaction, lambda_state);
  Vector reaction_residual;
  reaction_residual.reinit(source.locally_owned_dofs(), MPI_COMM_WORLD);
  model.evaluate_row(source.field_id(), context, reaction_residual);
  reaction_residual -= expected_reaction;
  EXPECT_LT(reaction_residual.l2_norm(), 1.e-12);
#ifdef IMMERSX_WEAK_TERM_TESTING
  EXPECT_EQ(detail::weak_term_nonmatching_preparations.load(),
            preparations + 2);
#endif
}


TEST(Constraint, MPI_NonmatchingDistributedReaction)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);

  parallel::distributed::Triangulation<2> source_tria(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<2> multiplier_tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(source_tria);
  GridGenerator::hyper_cube(multiplier_tria);
  source_tria.refine_global(2);
  multiplier_tria.refine_global(3);

  FE_Q<2>       source_fe(1);
  FE_Q<2>       multiplier_fe(2);
  DoFHandler<2> source_dh(source_tria);
  DoFHandler<2> multiplier_dh(multiplier_tria);
  source_dh.distribute_dofs(source_fe);
  multiplier_dh.distribute_dofs(multiplier_fe);
  const auto source_owned = source_dh.locally_owned_dofs();
  const auto source_relevant =
    DoFTools::extract_locally_relevant_dofs(source_dh);
  const auto multiplier_owned = multiplier_dh.locally_owned_dofs();
  const auto multiplier_relevant =
    DoFTools::extract_locally_relevant_dofs(multiplier_dh);
  AffineConstraints<double> source_constraints;
  AffineConstraints<double> multiplier_constraints;
  source_constraints.reinit(source_owned, source_relevant);
  multiplier_constraints.reinit(multiplier_owned, multiplier_relevant);
  source_constraints.close();
  multiplier_constraints.close();

  const auto  source_view     = fe_space(source_dh,
                                    StaticMappingQ1<2>::mapping,
                                    source_constraints,
                                    source_relevant);
  const auto  multiplier_view = fe_space(multiplier_dh,
                                        StaticMappingQ1<2>::mapping,
                                        multiplier_constraints,
                                        multiplier_relevant);
  StateLayout layout;
  const auto  source = source_view.field(layout, "source");
  const auto  lambda = multiplier_view.field("lambda");

  using Vector = ImmersXLA::MPI::Vector;
  using Matrix = ImmersXLA::MPI::SparseMatrix;
  using Model  = SemiDiscreteModel<Vector, Matrix>;
  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  const auto                          fields =
    make_constraint(weak_term(value(source), lambda)).add(builder);

  Vector source_state(source_owned, MPI_COMM_WORLD);
  Vector lambda_state(multiplier_owned, MPI_COMM_WORLD);
  source_state = 1.;
  lambda_state = 0.25;
  source_state.compress(VectorOperation::insert);
  lambda_state.compress(VectorOperation::insert);
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), source_state);
  state_view.bind(fields.multiplier, lambda_state);
  const EvaluationContext<Vector> context(0., state_view);

  Vector residual;
  residual.reinit(multiplier_owned, MPI_COMM_WORLD);
  model.evaluate_row(fields.multiplier, context, residual);
  const auto pairing =
    model.state_matrix_operator(fields.multiplier, source.field_id(), context);
  ASSERT_TRUE(pairing.has_value());
  Vector expected;
  expected.reinit(multiplier_owned, MPI_COMM_WORLD);
  pairing->view.vmult(expected, source_state);
  residual -= expected;
  EXPECT_LT(residual.l2_norm(), 1.e-12);

  Vector reaction;
  reaction.reinit(source_owned, MPI_COMM_WORLD);
  model.evaluate_row(source.field_id(), context, reaction);
  Vector expected_reaction;
  expected_reaction.reinit(source_owned, MPI_COMM_WORLD);
  pairing->view.Tvmult(expected_reaction, lambda_state);
  reaction -= expected_reaction;
  EXPECT_LT(reaction.l2_norm(), 1.e-12);
}


TEST(Constraint, MPI_MixedDimensionalReverseNonmatching)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);

  parallel::distributed::Triangulation<2>         bulk_tria(MPI_COMM_WORLD);
  Triangulation<1, 2>                             serial_line;
  parallel::fullydistributed::Triangulation<1, 2> line_tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bulk_tria);
  bulk_tria.refine_global(2);

  std::vector<Point<2>> line_vertices{Point<2>(0.15, 0.5), Point<2>(0.85, 0.5)};
  std::vector<CellData<1>> line_cells(1);
  line_cells[0].vertices[0] = 0;
  line_cells[0].vertices[1] = 1;
  serial_line.create_triangulation(line_vertices, line_cells, SubCellData());
  serial_line.refine_global(2);
  line_tria.copy_triangulation(serial_line);

  FE_Q<2>          bulk_fe(1);
  FE_Q<1, 2>       line_fe(1);
  DoFHandler<2>    bulk_dh(bulk_tria);
  DoFHandler<1, 2> line_dh(line_tria);
  bulk_dh.distribute_dofs(bulk_fe);
  line_dh.distribute_dofs(line_fe);

  const auto bulk_owned    = bulk_dh.locally_owned_dofs();
  const auto bulk_relevant = DoFTools::extract_locally_relevant_dofs(bulk_dh);
  const auto line_owned    = line_dh.locally_owned_dofs();
  const auto line_relevant = DoFTools::extract_locally_relevant_dofs(line_dh);
  AffineConstraints<double> bulk_constraints;
  AffineConstraints<double> line_constraints;
  bulk_constraints.reinit(bulk_owned, bulk_relevant);
  line_constraints.reinit(line_owned, line_relevant);
  bulk_constraints.close();
  line_constraints.close();

  const auto  bulk_view = fe_space(bulk_dh,
                                  StaticMappingQ1<2>::mapping,
                                  bulk_constraints,
                                  bulk_relevant);
  const auto  line_view = fe_space(line_dh,
                                  StaticMappingQ1<1, 2>::mapping,
                                  line_constraints,
                                  line_relevant);
  StateLayout layout;
  const auto  source = bulk_view.field(layout, "source");
  const auto  lambda = line_view.field("lambda");

  using Vector = ImmersXLA::MPI::Vector;
  using Matrix = ImmersXLA::MPI::SparseMatrix;
  using Model  = SemiDiscreteModel<Vector, Matrix>;
  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  const auto                          fields =
    make_constraint(weak_term(value(source), lambda)).add(builder);

  Vector source_state(bulk_owned, MPI_COMM_WORLD);
  Vector lambda_state(line_owned, MPI_COMM_WORLD);
  source_state = 1.;
  lambda_state = 0.25;
  source_state.compress(VectorOperation::insert);
  lambda_state.compress(VectorOperation::insert);
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), source_state);
  state_view.bind(fields.multiplier, lambda_state);
  const EvaluationContext<Vector> context(0., state_view);

  Vector residual;
  residual.reinit(line_owned, MPI_COMM_WORLD);
  model.evaluate_row(fields.multiplier, context, residual);
  const auto pairing =
    model.state_matrix_operator(fields.multiplier, source.field_id(), context);
  ASSERT_TRUE(pairing.has_value());
  Vector expected;
  expected.reinit(line_owned, MPI_COMM_WORLD);
  pairing->view.vmult(expected, source_state);
  residual -= expected;
  EXPECT_LT(residual.l2_norm(), 1.e-12);

  Vector reaction;
  reaction.reinit(bulk_owned, MPI_COMM_WORLD);
  model.evaluate_row(source.field_id(), context, reaction);
  Vector expected_reaction;
  expected_reaction.reinit(bulk_owned, MPI_COMM_WORLD);
  pairing->view.Tvmult(expected_reaction, lambda_state);
  reaction -= expected_reaction;
  EXPECT_LT(reaction.l2_norm(), 1.e-12);
}
