// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/quadrature_lib.h>

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
