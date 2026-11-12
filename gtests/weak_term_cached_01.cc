// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/core/contributor.h>
#include <immersx/core/domain.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/weak_term.h>

#include <cmath>
#include <map>
#include <utility>

using namespace dealii;
using namespace ImmersX;

namespace
{
  using V     = ImmersXLA::MPI::Vector;
  using M     = ImmersXLA::MPI::SparseMatrix;
  using Model = SemiDiscreteModel<V, M>;

  struct Q1Space
  {
    Q1Space()
      : tria(MPI_COMM_WORLD)
      , fe(1)
      , dofs(tria)
    {
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(1);
      dofs.distribute_dofs(fe);
      constraints.close();
    }

    auto
    points() const
    {
      std::map<types::global_dof_index, Point<2>> result;
      const auto                                  local =
        DoFTools::map_dofs_to_support_points(StaticMappingQ1<2>::mapping, dofs);
      for (const auto &part : Utilities::MPI::all_gather(MPI_COMM_WORLD, local))
        result.insert(part.begin(), part.end());
      return result;
    }

    parallel::distributed::Triangulation<2> tria;
    FE_Q<2>                                 fe;
    DoFHandler<2>                           dofs;
    AffineConstraints<double>               constraints;
  };

  // Exact one-dimensional Q1 mass matrix on [0,.5] and [.5,1].
  double
  mass_entry(const Point<2> &a, const Point<2> &b)
  {
    double result = 1.;
    for (unsigned int d = 0; d < 2; ++d)
      {
        const int i = std::lround(2. * a[d]), j = std::lround(2. * b[d]);
        result *= i == j ? (i == 1 ? 1. / 3. : 1. / 6.) :
                           (std::abs(i - j) == 1 ? 1. / 12. : 0.);
      }
    return result;
  }

  double
  nodal_value(const Point<2> &point, const double scale)
  {
    return scale * (1. + point[0] + 2. * point[1]);
  }
} // namespace

TEST(WeakTermNonmatching, BOTH_CachedAssemblyRetainsAnalyticPairing)
{
  Q1Space                   source_space, target_space;
  const auto                source_view = fe_space(source_space.dofs,
                                    StaticMappingQ1<2>::mapping,
                                    source_space.constraints);
  const auto                target_view = fe_space(target_space.dofs,
                                    StaticMappingQ1<2>::mapping,
                                    target_space.constraints);
  StateLayout               layout;
  const auto                source = source_view.field(layout, "source");
  const auto                target = target_view.field(layout, "target");
  Model                     model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  const auto before = detail::weak_term_matrix_initializations.load();
  weak_term(value(source), test(target))
    .with_cached_assembly()
    .on(IntegrationSupport<2>(target_space.tria))
    .add(builder);
  EXPECT_EQ(detail::weak_term_matrix_initializations.load(), before);
  EXPECT_ANY_THROW(weak_term(value(source) * value(source), test(target))
                     .with_cached_assembly());

  const auto   source_points = source_space.points();
  const auto   target_points = target_space.points();
  V            state(source.locally_owned_dofs(), MPI_COMM_WORLD);
  V            residual(target.locally_owned_dofs(), MPI_COMM_WORLD);
  StateView<V> view(layout, 0.);
  view.bind(source.field_id(), state);
  const EvaluationContext<V> context(0., view);
  state    = 0.;
  residual = 0.;
  model.evaluate_row(target.field_id(), context, residual);
  EXPECT_EQ(detail::weak_term_matrix_initializations.load(), before);

  const auto op =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(op.has_value());
  for (const double scale : {1., -2.5})
    {
      for (const auto i : source.locally_owned_dofs())
        state[i] = nodal_value(source_points.at(i), scale);
      state.compress(VectorOperation::insert);
      residual = 0.;
      model.evaluate_row(target.field_id(), context, residual);
      V action(residual);
      op->view.vmult(action, state);
      for (const auto i : target.locally_owned_dofs())
        {
          double exact = 0.;
          for (const auto &[j, point] : source_points)
            exact += mass_entry(target_points.at(i), point) *
                     nodal_value(point, scale);
          EXPECT_NEAR(residual[i], exact, 1.e-12);
          EXPECT_NEAR(action[i], exact, 1.e-12);
        }
    }
  V dual(target.locally_owned_dofs(), MPI_COMM_WORLD);
  for (const auto i : target.locally_owned_dofs())
    dual[i] = nodal_value(target_points.at(i), .75);
  dual.compress(VectorOperation::insert);
  V transpose(source.locally_owned_dofs(), MPI_COMM_WORLD);
  op->view.Tvmult(transpose, dual);
  for (const auto j : source.locally_owned_dofs())
    {
      double exact = 0.;
      for (const auto &[i, point] : target_points)
        exact +=
          mass_entry(point, source_points.at(j)) * nodal_value(point, .75);
      EXPECT_NEAR(transpose[j], exact, 1.e-12);
    }
  EXPECT_NEAR(residual * dual, state * transpose, 1.e-12);
  for (unsigned int repeat = 0; repeat < 2; ++repeat)
    {
      const auto matrix = op->matrix();
      for (const auto i : target.locally_owned_dofs())
        for (const auto &[j, point] : source_points)
          EXPECT_NEAR(matrix->el(i, j),
                      mass_entry(target_points.at(i), point),
                      1.e-12);
    }
  EXPECT_EQ(detail::weak_term_matrix_initializations.load(), before + 1u);
}

TEST(WeakTermNonmatching, BOTH_CachedAssemblyKeepsCurrentAffineOffsets)
{
  Q1Space space;
  space.constraints.clear();
  space.constraints.add_line(0);
  space.constraints.set_inhomogeneity(0, 1.);
  space.constraints.close();
  AffineConstraints<double> unconstrained;
  unconstrained.close();
  const auto source_view =
    fe_space(space.dofs, StaticMappingQ1<2>::mapping, space.constraints);
  const auto target_view =
    fe_space(space.dofs, StaticMappingQ1<2>::mapping, unconstrained);
  StateLayout               layout;
  const auto                source = source_view.field(layout, "source");
  const auto                target = target_view.field(layout, "target");
  Model                     model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  weak_term(value(source), test(target))
    .on(IntegrationSupport<2>(space.tria))
    .with_cached_assembly()
    .with_context_update([&](const double time) {
      space.constraints.set_inhomogeneity(0, 1. + time);
    })
    .add(builder);
  const auto points = space.points();
  V          state(source.locally_owned_dofs(), MPI_COMM_WORLD);
  state               = 0.;
  const auto   before = detail::weak_term_matrix_initializations.load();
  StateView<V> initial_view(layout, 0.);
  initial_view.bind(source.field_id(), state);
  const auto op =
    model.state_matrix_operator(target.field_id(),
                                source.field_id(),
                                EvaluationContext<V>(0., initial_view));
  ASSERT_TRUE(op.has_value());
  (void)op->matrix();
  for (const double time : {0., 2.})
    {
      StateView<V> view(layout, time);
      view.bind(source.field_id(), state);
      const EvaluationContext<V> context(time, view);
      model.update_context(context);
      V residual(target.locally_owned_dofs(), MPI_COMM_WORLD);
      residual = 0.;
      model.evaluate_row(target.field_id(), context, residual);
      for (const auto i : target.locally_owned_dofs())
        EXPECT_NEAR(residual[i],
                    mass_entry(points.at(i), points.at(0)) * (1. + time),
                    1.e-12);
    }
  // A zero homogeneous state must still evaluate the current affine offset.
  (void)op->matrix();
  EXPECT_EQ(detail::weak_term_matrix_initializations.load(), before + 1u);
}

TEST(WeakTermNonmatching, BOTH_CachedLiftedReactionRetainsPairing)
{
  ParameterAcceptor::clear();
  parallel::distributed::Triangulation<3> bulk_tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bulk_tria, -1., 1.);
  bulk_tria.refine_global(1);
  FESystem<3>   bulk_fe(FE_Q<3>(1), 3);
  DoFHandler<3> bulk_dofs(bulk_tria);
  bulk_dofs.distribute_dofs(bulk_fe);
  AffineConstraints<double> bulk_constraints;
  bulk_constraints.close();
  const auto bulk_space =
    fe_space(bulk_dofs, StaticMappingQ1<3>::mapping, bulk_constraints);
  DomainParameters<1, 3> parameters("/Cached line/");
  parameters.name_of_grid       = "hyper_cube";
  parameters.arguments_for_grid = "-.5:.5:false";
  parameters.initial_refinement = 1;
  Domain<1, 3> line_domain(parameters);
  line_domain.make_grid();
  FE_Q<1, 3>       line_fe(1);
  DoFHandler<1, 3> line_dofs(line_domain.triangulation());
  line_dofs.distribute_dofs(line_fe);
  AffineConstraints<double> line_constraints;
  line_constraints.close();
  const auto line_space =
    fe_space(line_dofs, StaticMappingQ1<1, 3>::mapping, line_constraints);
  StateLayout layout;
  const auto  u = bulk_space.field(layout, "u", FEValuesExtractors::Vector(0));
  const auto  lambda = line_space.field(layout, "lambda");
  TensorProductLift<1, 2, 3, 3> geometry("/Cached lift/");
  geometry.section.inclusion_degree      = 0;
  geometry.section.inclusion_type        = "hyper_cube";
  geometry.section.refinement_level      = 0;
  geometry.section.selected_coefficients = {0};
  geometry.section.n_q_points            = 2;
  geometry.representative_n_q_points     = 2;
  geometry.thickness                     = "0.25";
  const auto lifted                      = lift(value(lambda), geometry);
  EXPECT_ANY_THROW(
    weak_term(lift(value(lambda) * value(lambda), geometry), test(lambda))
      .with_cached_assembly());
  Model                     model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  const auto before = detail::weak_term_matrix_initializations.load();
  const IntegrationSupport<1, 3> support(line_domain.triangulation());
  weak_term(value(u), test(lifted))
    .with_cached_assembly()
    .on(support)
    .add(builder);
  weak_term(lifted, test(u)).on(support).with_cached_assembly().add(builder);
  EXPECT_EQ(detail::weak_term_matrix_initializations.load(), before);
  V          solid(u.locally_owned_dofs(), MPI_COMM_WORLD);
  V          multiplier(lambda.locally_owned_dofs(), MPI_COMM_WORLD);
  V          bulk_residual(u.locally_owned_dofs(), MPI_COMM_WORLD);
  V          line_residual(lambda.locally_owned_dofs(), MPI_COMM_WORLD);
  const auto points =
    DoFTools::map_dofs_to_support_points(StaticMappingQ1<3>::mapping,
                                         bulk_dofs);
  for (const double scale : {1., 3.})
    {
      for (const auto &cell : bulk_dofs.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            std::vector<types::global_dof_index> indices(
              bulk_fe.n_dofs_per_cell());
            cell->get_dof_indices(indices);
            for (unsigned int i = 0; i < indices.size(); ++i)
              if (u.locally_owned_dofs().is_element(indices[i]))
                {
                  const auto component =
                    bulk_fe.system_to_component_index(i).first;
                  solid[indices[i]] =
                    scale * (component + 1. + points.at(indices[i])[component]);
                }
          }
      solid.compress(VectorOperation::insert);
      multiplier = 2.;
      StateView<V> view(layout, 0.);
      view.bind(u.field_id(), solid);
      view.bind(lambda.field_id(), multiplier);
      const EvaluationContext<V> context(0., view);
      bulk_residual = 0.;
      line_residual = 0.;
      model.evaluate_row(u.field_id(), context, bulk_residual);
      model.evaluate_row(lambda.field_id(), context, line_residual);
      V expected(lambda.locally_owned_dofs(), MPI_COMM_WORLD);
      expected = 0.;
      for (const auto &point : lifted.lifted_points())
        {
          Tensor<1, 3> exact;
          for (unsigned int d = 0; d < 3; ++d)
            exact[d] = scale * (d + 1. + point.point[d]);
          Vector<double> local(point.dof_indices.size());
          for (unsigned int i = 0; i < local.size(); ++i)
            local[i] = point.weight * point.source_basis_values[i] *
                       (exact * point.mode_vector);
          line_constraints.distribute_local_to_global(local,
                                                      point.dof_indices,
                                                      expected);
        }
      expected.compress(VectorOperation::add);
      EXPECT_GT(expected.l2_norm(), 1.e-8);
      V error(line_residual);
      error -= expected;
      EXPECT_LT(error.l2_norm(), 1.e-12);
      EXPECT_NEAR(solid * bulk_residual, multiplier * line_residual, 1.e-12);
      for (const auto &[row, column] :
           {std::pair{u.field_id(), lambda.field_id()},
            std::pair{lambda.field_id(), u.field_id()}})
        {
          const auto op = model.state_matrix_operator(row, column, context);
          ASSERT_TRUE(op.has_value());
          (void)op->matrix();
        }
    }
  EXPECT_EQ(detail::weak_term_matrix_initializations.load(), before + 2u);
}
