// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/core/contributor.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/known_term.h>
#include <immersx/core/observable.h>
#include <immersx/core/state.h>

#include <cmath>
#include <set>

using namespace dealii;
using namespace ImmersX;

namespace
{
  class TimeFunction : public Function<2>
  {
  public:
    TimeFunction()
      : Function<2>(1)
    {}

    void
    set_time(const double time) override
    {
      time_ = time;
    }

    double
    value(const Point<2> &, const unsigned int = 0) const override
    {
      return time_;
    }

  private:
    double time_ = 0.;
  };

  struct ScalarSpace
  {
    ScalarSpace()
    {
      GridGenerator::hyper_cube(triangulation);
      triangulation.refine_global(1);
      dof_handler.distribute_dofs(finite_element);
      constraints.close();
    }

    Triangulation<2>          triangulation;
    FE_Q<2>                   finite_element{1};
    DoFHandler<2>             dof_handler{triangulation};
    AffineConstraints<double> constraints;
  };

  using LocalVector = dealii::Vector<double>;
  using LocalMatrix = dealii::SparseMatrix<double>;
  using Model       = SemiDiscreteModel<LocalVector, LocalMatrix>;

} // namespace

TEST(KnownTerm, TimeDependentVolumeSourceHasNoJacobian)
{
  ScalarSpace space;
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto target = V.field(layout, "target");

  TimeFunction function;
  const auto   source = known_function<2>(function);
  const auto   term   = known_term(source, test(target));
  EXPECT_TRUE(term.dependencies().empty());

  Model                                         model;
  SemidiscreteBuilder<LocalVector, LocalMatrix> builder(layout, model);
  term.add(builder);

  const StateView<LocalVector>         view_at_one(layout, 1.);
  const StateView<LocalVector>         view_at_two(layout, 2.);
  const EvaluationContext<LocalVector> context_at_one(1., view_at_one);
  const EvaluationContext<LocalVector> context_at_two(2., view_at_two);
  LocalVector residual_at_one(target.dof_handler().n_dofs());
  LocalVector residual_at_two(target.dof_handler().n_dofs());
  model.evaluate_row(target.field_id(), context_at_one, residual_at_one);
  model.evaluate_row(target.field_id(), context_at_two, residual_at_two);

  LocalVector expected = residual_at_one;
  expected *= 2.;
  residual_at_two -= expected;
  EXPECT_LT(residual_at_two.l2_norm(), 1.e-12);
  EXPECT_FALSE(model.has_state_operator(target.field_id(), target.field_id()));
  EXPECT_FALSE(model.has_derivative_terms());
}

TEST(KnownTerm, MaterialAndBoundaryOverridesUseFallbackContext)
{
  ScalarSpace space;
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto target = V.field(layout, "target");

  bool                   saw_material = false;
  bool                   saw_boundary = false;
  double                 boundary_h   = 0.;
  Tensor<1, 2>           boundary_normal;
  KnownSource<double, 2> source([](const KnownTermContext<2> &) { return 1.; });
  source.on_material(0, [&saw_material](const KnownTermContext<2> &data) {
    saw_material = data.material_id == 0;
    return 3.;
  });
  source.on_boundary(0,
                     [&saw_boundary, &boundary_h, &boundary_normal](
                       const KnownTermContext<2> &data) {
                       saw_boundary =
                         data.boundary_id.has_value() && *data.boundary_id == 0;
                       boundary_h      = data.h;
                       boundary_normal = data.normal;
                       return 5.;
                     });

  Model                                         model;
  SemidiscreteBuilder<LocalVector, LocalMatrix> builder(layout, model);
  known_term(source, test(target)).add(builder);
  known_term(source, test(target)).on_boundary(0).add(builder);

  const StateView<LocalVector>         view(layout, 0.);
  const EvaluationContext<LocalVector> evaluation(0., view);
  LocalVector                          residual(target.dof_handler().n_dofs());
  model.evaluate_row(target.field_id(), evaluation, residual);

  EXPECT_TRUE(saw_material);
  EXPECT_TRUE(saw_boundary);
  EXPECT_GT(boundary_h, 0.);
  EXPECT_GT(boundary_normal.norm(), 0.);
  EXPECT_GT(residual.l2_norm(), 0.);
}

TEST(KnownTerm, FrozenFiniteElementSourceReusesObservableMachinery)
{
  ScalarSpace space;
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source_field = V.field(layout, "source");
  const auto target       = V.field(layout, "target");

  LocalVector prescribed(source_field.dof_handler().n_dofs());
  prescribed      = 1.;
  const auto term = known_term(frozen(source_field, prescribed), test(target));

  Model                                         model;
  SemidiscreteBuilder<LocalVector, LocalMatrix> builder(layout, model);
  term.add(builder);

  const StateView<LocalVector>         view(layout, 0.);
  const EvaluationContext<LocalVector> evaluation(0., view);
  LocalVector                          residual(target.dof_handler().n_dofs());
  model.evaluate_row(target.field_id(), evaluation, residual);

  EXPECT_GT(residual.l2_norm(), 0.);
  EXPECT_FALSE(
    model.has_state_operator(target.field_id(), source_field.field_id()));
}

TEST(WeakTerm, BoundaryRegionMatchesFaceMassPairing)
{
  ScalarSpace space;
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source = V.field(layout, "source");
  const auto target = V.field(layout, "target");

  Model                                         model;
  SemidiscreteBuilder<LocalVector, LocalMatrix> builder(layout, model);
  weak_term(value(source), test(target)).on_boundary(0).add(builder);

  LocalVector source_state(source.dof_handler().n_dofs());
  source_state = 1.;
  LocalVector target_state(target.dof_handler().n_dofs());
  target_state = 0.;
  StateView<LocalVector> state_view(layout, 0.);
  state_view.bind(source.field_id(), source_state);
  state_view.bind(target.field_id(), target_state);
  const EvaluationContext<LocalVector> evaluation(0., state_view);

  LocalVector residual(target.dof_handler().n_dofs());
  model.evaluate_row(target.field_id(), evaluation, residual);

  const QGauss<1> quadrature(space.finite_element.degree + 1);
  FEFaceValues<2> face_values(StaticMappingQ1<2>::mapping,
                              space.finite_element,
                              quadrature,
                              update_values | update_JxW_values);
  LocalVector     expected(residual.size());
  expected = 0.;
  std::vector<types::global_dof_index> indices(
    space.finite_element.n_dofs_per_cell());
  for (const auto &cell : space.dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto face : cell->face_indices())
        if (cell->face(face)->at_boundary() &&
            cell->face(face)->boundary_id() == 0)
          {
            face_values.reinit(cell, face);
            cell->get_dof_indices(indices);
            for (const auto q : face_values.quadrature_point_indices())
              {
                double source_value = 0.;
                for (unsigned int j = 0; j < indices.size(); ++j)
                  source_value += face_values.shape_value(j, q);
                for (unsigned int i = 0; i < indices.size(); ++i)
                  expected[indices[i]] += source_value *
                                          face_values.shape_value(i, q) *
                                          face_values.JxW(q);
              }
          }

  residual -= expected;
  EXPECT_LT(residual.l2_norm(), 1.e-12);
  EXPECT_TRUE(model.has_state_operator(target.field_id(), source.field_id()));
}
