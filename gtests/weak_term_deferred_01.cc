// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on the deal.II library.
// ---------------------------------------------------------------------
#include <deal.II/base/function_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/weak_term.h>

using namespace dealii;
using namespace ImmersX;

TEST(WeakTerm, BOTH_DeferredBuildAndBoundaryOnlyAffineAssembly)
{
  parallel::distributed::Triangulation<2> mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(mesh);
  mesh.refine_global(2);
  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center().norm() < 0.4)
      cell->set_refine_flag();
  mesh.execute_coarsening_and_refinement();
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  const auto relevant = DoFTools::extract_locally_relevant_dofs(dofs);
  AffineConstraints<double> constraints(dofs.locally_owned_dofs(), relevant);
  DoFTools::make_hanging_node_constraints(dofs, constraints);
  VectorTools::interpolate_boundary_values(dofs,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           constraints);
  constraints.close();
  const auto  space = fe_space(dofs, StaticMappingQ1<2>::mapping, constraints);
  StateLayout layout;
  const auto  u = space.field(layout, "u");
  using V       = ImmersXLA::MPI::Vector;
  using M       = ImmersXLA::MPI::SparseMatrix;
  SemiDiscreteModel<V, M>   model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  const auto before = detail::weak_term_matrix_initializations.load();
  weak_term(value(u), test(u)).add(builder);
  EXPECT_EQ(detail::weak_term_matrix_initializations.load(), before);
  V state;
  state.reinit(dofs.locally_owned_dofs(), MPI_COMM_WORLD);
  state = 0.;
  StateView<V> view(layout, 0.);
  view.bind(u.id(), state);
  const EvaluationContext<V> context(0., view);
  const auto op = model.state_matrix_operator(u.id(), u.id(), context);
  ASSERT_TRUE(op.has_value());
  EXPECT_EQ(detail::weak_term_matrix_initializations.load(), before);
  const auto matrix = op->matrix();
  EXPECT_GT(detail::weak_term_matrix_initializations.load(), before);

  const auto trial  = value(u);
  const auto target = test(u);
  using Assembly    = detail::WeakAssembly<decltype(trial), decltype(target)>;
  std::vector<types::global_dof_index> prescribed;
  for (const auto &line : constraints.get_lines())
    if (line.entries.empty())
      prescribed.push_back(line.index);
  for (const double offset : {2., -3., 0.})
    {
      for (const auto i : prescribed)
        constraints.set_inhomogeneity(i, offset);
      const auto matrices_before =
        detail::weak_term_matrix_initializations.load();
      const auto cells_before = detail::weak_term_affine_cells.load();
      const auto rhs          = Assembly::template affine_rhs<V, M>(
        trial, target, InferredIntegrationSupport{}, false, {});
      EXPECT_EQ(detail::weak_term_matrix_initializations.load(),
                matrices_before);
      unsigned int                         expected_cells = 0;
      std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(indices);
            if (std::any_of(indices.begin(), indices.end(), [&](const auto i) {
                  return constraints.get_inhomogeneity(i) != 0.;
                }))
              ++expected_cells;
          }
      EXPECT_EQ(detail::weak_term_affine_cells.load() - cells_before,
                expected_cells);
      if (offset != 0.)
        EXPECT_GT(rhs->l2_norm(), 0.);
      else
        EXPECT_EQ(rhs->l2_norm(), 0.);
      // Reuse the explicitly materialized matrix after changing Dirichlet data.
      V expected, actual;
      expected.reinit(state);
      actual.reinit(state);
      matrix->vmult(expected, state);
      expected -= *rhs;
      model.evaluate_row(u.id(), context, actual);
      actual -= expected;
      EXPECT_LT(actual.l2_norm(), 1.e-12);
      // A fresh build has the same action despite the changed boundary values.
      auto fresh = op->matrix();
      state      = 1.;
      fresh->vmult(actual, state);
      matrix->vmult(expected, state);
      actual -= expected;
      EXPECT_LT(actual.l2_norm(), 1.e-12);
      state = 0.;
    }
}

TEST(WeakTerm, DeferredSumReleasesTemporaryMatrices)
{
  using V               = dealii::Vector<double>;
  using M               = dealii::SparseMatrix<double>;
  using Op              = MaterializedOperator<V, M>;
  unsigned int    alive = 0, peak = 0, builds = 0;
  std::vector<Op> recipes;
  for (unsigned int i = 0; i < 3; ++i)
    {
      Op op;
      op.materialize = [&, i] {
        ++builds;
        ++alive;
        peak         = std::max(peak, alive);
        auto pattern = std::make_shared<SparsityPattern>(2, 2, 2);
        pattern->add(i % 2, (i + 1) % 2);
        pattern->compress();
        auto matrix = std::shared_ptr<M>(new M(), [&, pattern](M *p) {
          delete p;
          --alive;
        });
        matrix->reinit(*pattern);
        matrix->set(i % 2, (i + 1) % 2, i + 1.);
        return matrix;
      };
      recipes.push_back(std::move(op));
    }
  const auto sum = detail::sum_matrices(recipes);
  EXPECT_EQ(alive, 0u);
  EXPECT_EQ(peak, 1u);
  EXPECT_EQ(builds, 6u);
  EXPECT_DOUBLE_EQ(sum->el(0, 1), 4.);
  EXPECT_DOUBLE_EQ(sum->el(1, 0), 2.);
}

TEST(WeakTerm, SerialDestinationKeepsCallerOwnedSparsity)
{
  Triangulation<2> mesh;
  GridGenerator::hyper_cube(mesh);
  mesh.refine_global(1);
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();
  const auto  space = fe_space(dofs, StaticMappingQ1<2>::mapping, constraints);
  StateLayout layout;
  const auto  u = space.field(layout, "u");
  using V       = dealii::Vector<double>;
  using M       = dealii::SparseMatrix<double>;
  SemiDiscreteModel<V, M>   model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  weak_term(value(u), test(u)).add(builder);
  weak_term(2. * value(u), test(u)).add(builder);
  V state(dofs.n_dofs());
  state = 1.;
  StateView<V> view(layout, 0.);
  view.bind(u.id(), state);
  const EvaluationContext<V> context(0., view);
  const auto op = model.state_matrix_operator(u.id(), u.id(), context);
  ASSERT_TRUE(op.has_value());
  DynamicSparsityPattern graph(dofs.n_dofs());
  DoFTools::make_sparsity_pattern(dofs, graph);
  SparsityPattern sparsity;
  sparsity.copy_from(graph);
  M destination(sparsity);
  op->materialize_into_matrix(destination);
  EXPECT_EQ(&destination.get_sparsity_pattern(), &sparsity);
  V action(dofs.n_dofs());
  destination.vmult(action, state);
  EXPECT_NEAR(action.l1_norm(), 3., 1.e-12);
}
