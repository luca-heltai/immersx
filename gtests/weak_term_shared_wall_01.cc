// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/core/contributor.h>
#include <immersx/core/domain.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/weak_term.h>

#include <cmath>
#include <map>
#include <numeric>

using namespace dealii;
using namespace ImmersX;

namespace
{
  using V     = ImmersXLA::MPI::Vector;
  using M     = ImmersXLA::MPI::SparseMatrix;
  using Model = SemiDiscreteModel<V, M>;

  void
  check_shared_wall(const bool empty_owner)
  {
    ParameterAcceptor::clear();
    parallel::distributed::Triangulation<3> bulk(MPI_COMM_WORLD);
    GridGenerator::hyper_cube(bulk, -1., 1.);
    bulk.refine_global(1);
    FESystem<3>   fe(FE_Q<3>(1), 6);
    DoFHandler<3> dofs(bulk);
    dofs.distribute_dofs(fe);
    AffineConstraints<double> constraints;
    constraints.close();
    const auto space = fe_space(dofs, StaticMappingQ1<3>::mapping, constraints);
    const auto other_space =
      fe_space(dofs, StaticMappingQ1<3>::mapping, constraints);
    StateLayout layout;
    const auto  u = space.field(layout, "u", FEValuesExtractors::Vector(0));
    const auto  v = space.field(layout, "v", FEValuesExtractors::Vector(0));
    DomainParameters<1, 3> parameters("/Shared wall/Line/");
    parameters.triangulation_type = "fullydistributed";
    parameters.name_of_grid       = "hyper_cube";
    parameters.arguments_for_grid = "-.5:.5:false";
    parameters.initial_refinement = empty_owner ? 0 : 1;
    Domain<1, 3> line(parameters);
    line.make_grid();
    FE_DGQ<1, 3>     line_fe(1);
    DoFHandler<1, 3> line_dofs(line.triangulation());
    line_dofs.distribute_dofs(line_fe);
    AffineConstraints<double> line_constraints;
    line_constraints.close();
    const auto line_space =
      fe_space(line_dofs, StaticMappingQ1<1, 3>::mapping, line_constraints);
    const auto                    lambda = line_space.field(layout, "lambda");
    TensorProductLift<1, 2, 3, 3> geometry("/Shared wall/Lift/");
    geometry.section.inclusion_degree      = 0;
    geometry.section.inclusion_type        = "hyper_cube";
    geometry.section.refinement_level      = 0;
    geometry.section.selected_coefficients = {0};
    geometry.section.n_q_points            = 2;
    geometry.representative_n_q_points     = 3;
    geometry.thickness                     = "0.25";
    const auto                     lifted  = lift(value(lambda), geometry);
    const IntegrationSupport<1, 3> support(line.triangulation());
    if (empty_owner)
      {
        EXPECT_EQ(Utilities::MPI::min(static_cast<unsigned int>(
                                        lifted.lifted_points().size()),
                                      MPI_COMM_WORLD),
                  0u);
      }
    Model                     shared, independent;
    SemidiscreteBuilder<V, M> builder(layout, shared),
      old_builder(layout, independent);
    const auto term =
      weak_term(value(u), test(lifted)).on(support).with_cached_assembly();
    term.add_constraint_terms(builder, lambda.id(), 1., 0, v);
    term.add(old_builder);
    weak_term(lifted, test(v))
      .on(support)
      .with_cached_assembly()
      .add(old_builder);

    // All ranks must reject before any deferred distributed assembly, even
    // when only one rank supplies a different Space.
    const auto other =
      other_space.field(v.id(), "other", FEValuesExtractors::Vector(0));
    const auto wrong_space =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 ? other : v;
    EXPECT_ANY_THROW(
      term.add_constraint_terms(builder, lambda.id(), 1., 1, wrong_space));
    const auto wrong_mask =
      space.field(v.id(), "wrong mask", FEValuesExtractors::Vector(3));
    EXPECT_ANY_THROW(
      term.add_constraint_terms(builder, lambda.id(), 1., 1, wrong_mask));
    std::vector<types::global_dof_index> identity(dofs.n_dofs());
    std::iota(identity.begin(), identity.end(), 0);
    const auto reindexed = v.reindexed("reindexed",
                                       v.locally_owned_dofs(),
                                       v.locally_relevant_dofs(),
                                       identity);
    EXPECT_ANY_THROW(
      term.add_constraint_terms(builder, lambda.id(), 1., 1, reindexed));
    EXPECT_ANY_THROW(term.add_constraint_terms(
      builder, lambda.id(), 1., 1, v.with_constraints()));

    const auto local_points =
      DoFTools::map_dofs_to_support_points(StaticMappingQ1<3>::mapping, dofs);
    std::map<types::global_dof_index, Point<3>> points;
    for (const auto &part :
         Utilities::MPI::all_gather(MPI_COMM_WORLD, local_points))
      points.insert(part.begin(), part.end());
    std::map<types::global_dof_index, unsigned int> local_components,
      components;
    for (const auto &cell : dofs.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            local_components[indices[i]] =
              fe.system_to_component_index(i).first;
        }
    for (const auto &part :
         Utilities::MPI::all_gather(MPI_COMM_WORLD, local_components))
      components.insert(part.begin(), part.end());
    V state(u.locally_owned_dofs(), MPI_COMM_WORLD), velocity(state), du(state);
    V multiplier(lambda.locally_owned_dofs(), MPI_COMM_WORLD), dl(multiplier);
    for (const auto i : u.locally_owned_dofs())
      {
        const auto c = components.at(i);
        state[i]     = c < 3 ? 1. + c + points.at(i)[c] : 0.;
        velocity[i]  = -3. + .013 * i;
        du[i]        = std::sin(.27 * (i + 1));
      }
    for (const auto i : lambda.locally_owned_dofs())
      {
        multiplier[i] = 1. + .11 * i;
        dl[i]         = std::cos(.31 * (i + 1));
      }
    for (V *vector : {&state, &velocity, &du, &multiplier, &dl})
      vector->compress(VectorOperation::insert);
    const auto view_for = [&](const V &x, const V &l) {
      StateView<V> view(layout, 0.);
      view.bind(u.id(), x);
      view.bind(v.id(), velocity);
      view.bind(lambda.id(), l);
      return view;
    };
    const auto residual =
      [&](const Model &model, const FieldId row, const V &x, const V &l) {
        V result(row == lambda.id() ? lambda.locally_owned_dofs() :
                                      u.locally_owned_dofs(),
                 MPI_COMM_WORLD);
        result          = 0.;
        const auto view = view_for(x, l);
        model.evaluate_row(row, EvaluationContext<V>(0., view), result);
        return result;
      };
    const auto                 view = view_for(state, multiplier);
    const EvaluationContext<V> context(0., view);
    const auto before = detail::weak_term_matrix_initializations.load();
    const auto forward =
      shared.state_matrix_operator(lambda.id(), u.id(), context);
    const auto reverse =
      shared.state_matrix_operator(v.id(), lambda.id(), context);
    ASSERT_TRUE(forward.has_value());
    ASSERT_TRUE(reverse.has_value());
    EXPECT_FALSE(
      shared.state_matrix_operator(u.id(), lambda.id(), context).has_value());
    EXPECT_EQ(residual(shared, u.id(), state, multiplier).l2_norm(), 0.);
    const auto retained = forward->matrix();
    (void)reverse->matrix();
    V line_result = residual(shared, lambda.id(), state, multiplier);
    V bulk_result = residual(shared, v.id(), state, multiplier);
    EXPECT_EQ(detail::weak_term_matrix_initializations.load(), before + 1u);

    // Physical-row oracle: explicit Cartesian Q1 hats, independent of FEValues
    // and point location. The grid has spacing one in each direction.
    std::vector<double> local_reaction(dofs.n_dofs(), 0.),
      reaction(dofs.n_dofs());
    V exact_line(lambda.locally_owned_dofs(), MPI_COMM_WORLD);
    exact_line = 0.;
    std::vector<double> local_line(line_dofs.n_dofs(), 0.),
      reference_line(line_dofs.n_dofs());
    for (const auto &q : lifted.lifted_points())
      {
        Tensor<1, 3> exact_u;
        for (unsigned int d = 0; d < 3; ++d)
          exact_u[d] = 1. + d + q.point[d];
        double l = 0.;
        for (unsigned int i = 0; i < q.dof_indices.size(); ++i)
          {
            l += (1. + .11 * q.dof_indices[i]) * q.source_basis_values[i];
            local_line[q.dof_indices[i]] +=
              q.weight * q.source_basis_values[i] * (exact_u * q.mode_vector);
          }
        for (const auto &[i, node] : points)
          if (components.at(i) < 3)
            {
              double hat = 1.;
              for (unsigned int d = 0; d < 3; ++d)
                hat *= std::max(0., 1. - std::abs(q.point[d] - node[d]));
              local_reaction[i] +=
                q.weight * l * hat * q.mode_vector[components.at(i)];
            }
      }
    Utilities::MPI::sum(local_line, MPI_COMM_WORLD, reference_line);
    for (const auto i : lambda.locally_owned_dofs())
      exact_line[i] = reference_line[i];
    exact_line.compress(VectorOperation::insert);
    EXPECT_GT(exact_line.l2_norm(), 1.e-8);
    Utilities::MPI::sum(local_reaction, MPI_COMM_WORLD, reaction);
    V error(line_result);
    error -= exact_line;
    EXPECT_LT(error.l2_norm(), 1.e-11);
    for (const auto i : u.locally_owned_dofs())
      EXPECT_NEAR(bulk_result[i], reaction[i], 1.e-11);
    EXPECT_GT(bulk_result.l2_norm(), 1.e-8);
    EXPECT_NEAR(state * bulk_result, multiplier * line_result, 1.e-11);
    V action(line_result), transpose(bulk_result), reaction_action(bulk_result);
    forward->view.vmult(action, du);
    forward->view.Tvmult(transpose, dl);
    reverse->view.vmult(reaction_action, dl);
    reaction_action -= transpose;
    EXPECT_LT(reaction_action.l2_norm(), 1.e-12);
    EXPECT_NEAR(action * dl, du * transpose, 1.e-11);
    const double eps = 1.e-6;
    V            perturbed(state);
    perturbed.add(eps, du);
    V fd = residual(shared, lambda.id(), perturbed, multiplier);
    fd -= line_result;
    fd *= 1. / eps;
    fd -= action;
    EXPECT_LT(fd.l2_norm(), 1.e-8);
    V lp(multiplier);
    lp.add(eps, dl);
    fd = residual(shared, v.id(), state, lp);
    fd -= bulk_result;
    fd *= 1. / eps;
    fd -= transpose;
    EXPECT_LT(fd.l2_norm(), 1.e-8);
    velocity.add(7.);
    error = residual(shared, lambda.id(), state, multiplier);
    error -= line_result;
    EXPECT_LT(error.l2_norm(), 1.e-12);
    (void)residual(shared, lambda.id(), perturbed, lp);
    V retained_action(line_result);
    retained->vmult(retained_action, du);
    retained_action -= action;
    EXPECT_LT(retained_action.l2_norm(), 1.e-12);
    EXPECT_EQ(detail::weak_term_matrix_initializations.load(), before + 1u);
    error = residual(independent, lambda.id(), state, multiplier);
    error -= line_result;
    EXPECT_LT(error.l2_norm(), 1.e-11);
    V bulk_error = residual(independent, v.id(), state, multiplier);
    bulk_error -= bulk_result;
    EXPECT_LT(bulk_error.l2_norm(), 1.e-11);
    EXPECT_EQ(detail::weak_term_matrix_initializations.load(), before + 3u);
  }
} // namespace

TEST(WeakTermNonmatching, BOTH_SharedWallPairingPreservesPhysicalRows)
{
  check_shared_wall(false);
}

TEST(WeakTermNonmatching, MPI_SharedWallPairingWithEmptyRepresentativeOwner)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  check_shared_wall(true);
}

TEST(WeakTermNonmatching, BOTH_SharedReactionKeepsCurrentSourceAffineOffsets)
{
  parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(1);
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(tria);
  dofs.distribute_dofs(fe);
  AffineConstraints<double> constraints, target_constraints;
  constraints.add_line(0);
  constraints.set_inhomogeneity(0, 1.);
  constraints.close();
  target_constraints.close();
  const auto space = fe_space(dofs, StaticMappingQ1<2>::mapping, constraints);
  const auto target_space =
    fe_space(dofs, StaticMappingQ1<2>::mapping, target_constraints);
  StateLayout layout;
  const auto  u = space.field(layout, "u"), v = space.field(layout, "v");
  const auto  lambda = target_space.field(layout, "lambda");
  Model       model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  weak_term(value(u), test(lambda))
    .on(IntegrationSupport<2>(tria))
    .with_cached_assembly()
    .add_constraint_terms(builder, lambda.id(), 1., 0, v);
  std::map<types::global_dof_index, Point<2>> points;
  const auto                                  local =
    DoFTools::map_dofs_to_support_points(StaticMappingQ1<2>::mapping, dofs);
  for (const auto &part : Utilities::MPI::all_gather(MPI_COMM_WORLD, local))
    points.insert(part.begin(), part.end());
  V state(u.locally_owned_dofs(), MPI_COMM_WORLD),
    multiplier(lambda.locally_owned_dofs(), MPI_COMM_WORLD);
  state             = 0.;
  multiplier        = 2.;
  const auto before = detail::weak_term_matrix_initializations.load();
  for (const double offset : {1., 3.})
    {
      constraints.set_inhomogeneity(0, offset);
      StateView<V> view(layout, 0.);
      view.bind(u.id(), state);
      view.bind(lambda.id(), multiplier);
      const EvaluationContext<V> context(0., view);
      V result(lambda.locally_owned_dofs(), MPI_COMM_WORLD);
      result = 0.;
      model.evaluate_row(lambda.id(), context, result);
      for (const auto i : lambda.locally_owned_dofs())
        {
          double mass = 1.;
          for (unsigned int d = 0; d < 2; ++d)
            {
              const int a = std::lround(2. * points.at(i)[d]);
              const int b = std::lround(2. * points.at(0)[d]);
              mass *= a == b ? (a == 1 ? 1. / 3. : 1. / 6.) :
                               (std::abs(a - b) == 1 ? 1. / 12. : 0.);
            }
          EXPECT_NEAR(result[i], mass * offset, 1.e-12);
        }
      const auto forward =
        model.state_matrix_operator(lambda.id(), u.id(), context);
      const auto reverse =
        model.state_matrix_operator(v.id(), lambda.id(), context);
      ASSERT_TRUE(forward.has_value());
      ASSERT_TRUE(reverse.has_value());
      (void)forward->matrix();
      (void)reverse->matrix();
      V reaction(v.locally_owned_dofs(), MPI_COMM_WORLD);
      reaction = 0.;
      model.evaluate_row(v.id(), context, reaction);
      if (v.locally_owned_dofs().is_element(0))
        {
          EXPECT_NEAR(reaction[0], 0., 1.e-12);
        }
    }
  EXPECT_EQ(detail::weak_term_matrix_initializations.load(), before + 1u);
}
