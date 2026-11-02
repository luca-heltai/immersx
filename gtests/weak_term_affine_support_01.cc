// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/weak_term.h>

using namespace dealii;
using namespace ImmersX;

namespace
{
  template <typename VectorType,
            typename MatrixType,
            typename FieldType,
            typename Oracle>
  void
  check_affine_model(SemiDiscreteModel<VectorType, MatrixType> &model,
                     const StateLayout                         &layout,
                     const FieldType                           &source,
                     const FieldType                           &target,
                     const Oracle                              &oracle)
  {
    VectorType state, direction, residual, expected, shifted, action;
    detail::initialize_weak_vector(state,
                                   source.locally_owned_dofs(),
                                   source.space().mpi_communicator());
    detail::initialize_weak_vector(direction,
                                   source.locally_owned_dofs(),
                                   source.space().mpi_communicator());
    for (const auto index : source.locally_owned_dofs())
      if (!source.constraints().is_constrained(index))
        {
          state[index]     = 0.3 + 0.01 * index;
          direction[index] = 0.7 - 0.003 * index;
        }
    if constexpr (detail::has_vector_compress<VectorType>::value)
      {
        state.compress(VectorOperation::insert);
        direction.compress(VectorOperation::insert);
      }
    for (auto *vector : {&residual, &expected, &shifted, &action})
      detail::initialize_weak_vector(*vector,
                                     target.locally_owned_dofs(),
                                     target.space().mpi_communicator());
    StateView<VectorType> view(layout, 0.);
    view.bind(source.field_id(), state);
    EvaluationContext<VectorType> context(0., view);
    model.evaluate_row(target.field_id(), context, residual);
    oracle(expected, 1.);
    EXPECT_GT(expected.l2_norm(), 1.e-5);
    expected -= residual;
    EXPECT_LT(expected.l2_norm(), 1.e-12);

    const auto jacobian = model.state_matrix_operator(target.field_id(),
                                                      source.field_id(),
                                                      context);
    ASSERT_TRUE(jacobian.has_value());
    jacobian->view.vmult(action, direction);
    const double epsilon = 1.e-6;
    state.add(epsilon, direction);
    model.evaluate_row(target.field_id(), context, shifted);
    shifted -= residual;
    shifted *= 1. / epsilon;
    shifted -= action;
    EXPECT_LT(shifted.l2_norm(), 1.e-9);

    // Isolate the affine lift from the condensed matrix action.
    state    = 0.;
    residual = 0.;
    expected = 0.;
    model.evaluate_row(target.field_id(), context, residual);
    oracle(expected, 0.);
    expected -= residual;
    EXPECT_LT(expected.l2_norm(), 1.e-12);
  }

  // Reconstruct the physical FE coefficients independently of weak-term
  // storage.
  template <typename FieldType>
  Vector<double>
  physical_state(const FieldType &source, const double free_scale)
  {
    Vector<double> state(source.locally_owned_dofs().size());
    for (unsigned int i = 0; i < state.size(); ++i)
      state[i] = free_scale * (0.3 + 0.01 * i);
    source.constraints().distribute(state);
    return state;
  }

  template <typename VectorType, typename FieldType>
  void
  cell_oracle(VectorType                  &expected,
              const FieldType             &source,
              const FieldType             &target,
              const IntegrationSupport<2> &support,
              const double                 free_scale)
  {
    const auto  state = physical_state(source, free_scale);
    FEValues<2> trial(source.mapping(),
                      source.space().finite_element(),
                      QGauss<2>(4),
                      update_values);
    FEValues<2> test_values(target.mapping(),
                            target.space().finite_element(),
                            QGauss<2>(4),
                            update_values | update_JxW_values);
    for (const auto &cell : source.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned() && support.includes_cell(cell))
        {
          const auto target_cell =
            cell->as_dof_handler_iterator(target.dof_handler());
          trial.reinit(cell);
          test_values.reinit(target_cell);
          std::vector<double> values(trial.n_quadrature_points);
          trial.get_function_values(state, values);
          Vector<double> local(target_cell->get_fe().n_dofs_per_cell());
          std::vector<types::global_dof_index> indices(local.size());
          target_cell->get_dof_indices(indices);
          Vector<double> diagonal(local.size());
          for (unsigned int q = 0; q < values.size(); ++q)
            for (unsigned int i = 0; i < local.size(); ++i)
              {
                const double phi = test_values.shape_value(i, q);
                local[i] += phi * values[q] * test_values.JxW(q);
                diagonal[i] += phi * phi * test_values.JxW(q);
              }
          target.constraints().distribute_local_to_global(local,
                                                          indices,
                                                          expected);
          // deal.II square assembly also enforces constrained diagonal rows.
          // The execution state has zero constrained entries in this test.
          if (&source.constraints() == &target.constraints() &&
              &source.dof_handler() == &target.dof_handler())
            for (unsigned int i = 0; i < indices.size(); ++i)
              if (source.constraints().is_constrained(indices[i]))
                expected[indices[i]] -=
                  diagonal[i] *
                  source.constraints().get_inhomogeneity(indices[i]);
        }
    detail::compress_weak_vector(expected);
  }

  template <typename VectorType, typename FieldType>
  void
  boundary_oracle(VectorType                         &expected,
                  const FieldType                    &source,
                  const FieldType                    &target,
                  const std::set<types::boundary_id> &ids,
                  const bool                          normal_derivative,
                  const double                        free_scale)
  {
    constexpr int               dim      = FieldType::dimension();
    constexpr int               spacedim = FieldType::spacedimension();
    const auto                  state    = physical_state(source, free_scale);
    FEFaceValues<dim, spacedim> values(source.mapping(),
                                       source.space().finite_element(),
                                       QGauss<dim - 1>(4),
                                       update_values | update_gradients |
                                         update_normal_vectors |
                                         update_JxW_values);
    for (const auto &cell : source.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          std::vector<types::global_dof_index> indices(
            cell->get_fe().n_dofs_per_cell());
          cell->get_dof_indices(indices);
          for (const auto face : cell->face_indices())
            if (cell->face(face)->at_boundary() &&
                ids.count(cell->face(face)->boundary_id()))
              {
                values.reinit(cell, face);
                std::vector<double> field_values(values.n_quadrature_points);
                std::vector<Tensor<1, spacedim>> gradients(
                  values.n_quadrature_points);
                values.get_function_values(state, field_values);
                values.get_function_gradients(state, gradients);
                Vector<double> local(indices.size());
                for (unsigned int q = 0; q < values.n_quadrature_points; ++q)
                  for (unsigned int i = 0; i < indices.size(); ++i)
                    local[i] += values.shape_value(i, q) *
                                (normal_derivative ?
                                   gradients[q] * values.normal_vector(q) :
                                   field_values[q]) *
                                values.JxW(q);
                target.constraints().distribute_local_to_global(local,
                                                                indices,
                                                                expected);
              }
        }
    detail::compress_weak_vector(expected);
  }
} // namespace

TEST(WeakTerm, AffineCellSupportRectangularAndDynamic)
{
  Triangulation<2> mesh;
  GridGenerator::hyper_cube(mesh);
  mesh.refine_global(1);
  for (const auto &cell : mesh.active_cell_iterators())
    cell->set_material_id(cell->center()[0] < 0.5 ? 1 : 2);
  FE_Q<2>       source_fe(2), target_fe(1);
  DoFHandler<2> source_dofs(mesh), target_dofs(mesh);
  source_dofs.distribute_dofs(source_fe);
  target_dofs.distribute_dofs(target_fe);
  AffineConstraints<double> source_constraints, target_constraints;
  source_constraints.add_line(0);
  source_constraints.set_inhomogeneity(0, 2.);
  source_constraints.add_line(1);
  source_constraints.add_entry(1, 2, 0.5);
  source_constraints.set_inhomogeneity(1, 0.4);
  source_constraints.close();
  target_constraints.add_line(3);
  target_constraints.add_entry(3, 4, 0.25);
  target_constraints.set_inhomogeneity(
    3, 7.); // Test offsets must not enter the residual.
  target_constraints.close();
  const auto source_space =
    fe_space(source_dofs, StaticMappingQ1<2>::mapping, source_constraints);
  const auto target_space =
    fe_space(target_dofs, StaticMappingQ1<2>::mapping, target_constraints);
  StateLayout                 layout;
  const auto                  source = source_space.field(layout, "source");
  const auto                  target = target_space.field(layout, "target");
  const IntegrationSupport<2> support(mesh, 1);
  using V = Vector<double>;
  using M = SparseMatrix<double>;
  SemiDiscreteModel<V, M>   model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  weak_term(value(source), test(target), support).add(builder);
  for (const double offset : {2., -1., 0.})
    {
      source_constraints.set_inhomogeneity(0, offset);
      check_affine_model(model,
                         layout,
                         source,
                         target,
                         [&](V &expected, const double free_scale) {
                           cell_oracle(
                             expected, source, target, support, free_scale);
                         });
    }
}

TEST(WeakTerm, BOTH_AffineCellSupportHangingAndAsymmetric)
{
  parallel::distributed::Triangulation<2> mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(mesh);
  mesh.refine_global(2);
  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[0] < 0.3 &&
        cell->center()[1] < 0.3)
      cell->set_refine_flag();
  mesh.execute_coarsening_and_refinement();
  for (const auto &cell : mesh.active_cell_iterators())
    if (!cell->is_artificial())
      cell->set_material_id(cell->center()[0] < 0.5 ? 1 : 2);
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  const auto relevant = DoFTools::extract_locally_relevant_dofs(dofs);
  AffineConstraints<double> constraints(dofs.locally_owned_dofs(), relevant);
  DoFTools::make_hanging_node_constraints(dofs, constraints);
  EXPECT_GT(Utilities::MPI::sum(constraints.n_constraints(), MPI_COMM_WORLD),
            0u);
  const auto points =
    DoFTools::map_dofs_to_support_points(StaticMappingQ1<2>::mapping, dofs);
  types::global_dof_index affine_node = numbers::invalid_dof_index;
  // This fine-cell interior node contributes to cells with hanging test rows.
  const Point<2> affine_point(0.125, 0.125);
  for (const auto &[index, point] : points)
    if (relevant.is_element(index) && point.distance(affine_point) < 1.e-12)
      {
        affine_node = index;
        constraints.add_line(index);
        constraints.set_inhomogeneity(index, 0.);
      }
  constraints.close();
  EXPECT_FALSE(constraints.has_inhomogeneities());
  const auto space = fe_space(dofs, StaticMappingQ1<2>::mapping, constraints);
  EXPECT_EQ(space.locally_relevant_dofs(), relevant);
  StateLayout layout;
  const auto  source = space.field(layout, "source");
  const auto  target = space.field(layout, "target");
  using V            = ImmersXLA::MPI::Vector;
  using M            = ImmersXLA::MPI::SparseMatrix;
  SemiDiscreteModel<V, M>   explicit_model, inferred_model;
  SemidiscreteBuilder<V, M> explicit_builder(layout, explicit_model),
    inferred_builder(layout, inferred_model);
  const IntegrationSupport<2> support(mesh, 1), whole_mesh(mesh);
  weak_term(value(source), test(target), support).add(explicit_builder);
  weak_term(value(source), test(target)).add(inferred_builder);
  for (const double offset : {2., -3., 0.})
    {
      if (affine_node != numbers::invalid_dof_index)
        constraints.set_inhomogeneity(affine_node, offset);
      if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 2)
        {
          EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(
                                          constraints.has_inhomogeneities()),
                                        MPI_COMM_WORLD),
                    offset == 0. ? 0u : 1u);
        }
      check_affine_model(explicit_model,
                         layout,
                         source,
                         target,
                         [&](V &expected, const double free_scale) {
                           cell_oracle(
                             expected, source, target, support, free_scale);
                         });
      check_affine_model(inferred_model,
                         layout,
                         source,
                         target,
                         [&](V &expected, const double free_scale) {
                           cell_oracle(
                             expected, source, target, whole_mesh, free_scale);
                         });
    }
}

TEST(WeakTerm, AffineFrozenAndConstraintCellSupports)
{
  Triangulation<2> mesh;
  GridGenerator::hyper_cube(mesh);
  mesh.refine_global(1);
  for (const auto &cell : mesh.active_cell_iterators())
    cell->set_material_id(cell->center()[0] < 0.5 ? 1 : 2);
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  AffineConstraints<double> source_constraints, target_constraints;
  source_constraints.add_line(0);
  source_constraints.set_inhomogeneity(0, 2.);
  source_constraints.close();
  target_constraints.close();
  const auto source_space =
    fe_space(dofs, StaticMappingQ1<2>::mapping, source_constraints);
  const auto target_space =
    fe_space(dofs, StaticMappingQ1<2>::mapping, target_constraints);
  StateLayout layout;
  const auto  source = source_space.field(layout, "source");
  const auto  target = target_space.field(layout, "multiplier");
  using V            = Vector<double>;
  using M            = SparseMatrix<double>;
  V coefficients(dofs.n_dofs());
  for (unsigned int i = 0; i < coefficients.size(); ++i)
    if (!source_constraints.is_constrained(i))
      coefficients[i] = 0.3 + 0.01 * i;
  const auto observable = frozen(source, coefficients);
  EXPECT_TRUE(observable.dependencies().empty());
  SemiDiscreteModel<V, M>   frozen_model, constraint_model;
  SemidiscreteBuilder<V, M> frozen_builder(layout, frozen_model),
    constraint_builder(layout, constraint_model);
  const IntegrationSupport<2> cells(mesh, 1);
  weak_term(observable, test(target), cells).add(frozen_builder);
  const double sign = -1.25;
  weak_term(value(source), test(target), cells)
    .add_constraint_terms(constraint_builder, target.field_id(), sign, 0);
  StateView<V>         view(layout, 0.);
  EvaluationContext<V> context(0., view);
  for (const double offset : {2., -1., 0.})
    {
      source_constraints.set_inhomogeneity(0, offset);
      V residual(dofs.n_dofs()), expected(dofs.n_dofs());
      frozen_model.evaluate_row(target.field_id(), context, residual);
      cell_oracle(expected, source, target, cells, 1.);
      EXPECT_GT(expected.l2_norm(), 1.e-5);
      expected -= residual;
      EXPECT_LT(expected.l2_norm(), 1.e-12);
      check_affine_model(constraint_model,
                         layout,
                         source,
                         target,
                         [&](V &result, const double free_scale) {
                           cell_oracle(
                             result, source, target, cells, free_scale);
                           result *= sign;
                         });
    }
}

TEST(WeakTerm, AffineFrozenCellAndBoundarySupports)
{
  Triangulation<2> mesh;
  GridGenerator::hyper_cube(mesh, 0., 1., true);
  mesh.refine_global(1);
  for (const auto &cell : mesh.active_cell_iterators())
    cell->set_material_id(cell->center()[0] < 0.5 ? 1 : 2);
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  AffineConstraints<double> source_constraints, target_constraints;
  source_constraints.add_line(0);
  source_constraints.set_inhomogeneity(0, 2.);
  source_constraints.close();
  target_constraints.close();
  const auto source_space =
    fe_space(dofs, StaticMappingQ1<2>::mapping, source_constraints);
  const auto target_space =
    fe_space(dofs, StaticMappingQ1<2>::mapping, target_constraints);
  StateLayout layout;
  const auto  source = source_space.field("frozen-source");
  const auto  target = target_space.field(layout, "target");
  using V            = Vector<double>;
  using M            = SparseMatrix<double>;
  V coefficients(dofs.n_dofs());
  for (unsigned int i = 0; i < coefficients.size(); ++i)
    if (!source_constraints.is_constrained(i))
      coefficients[i] = 0.3 + 0.01 * i;
  const auto observable = frozen(source, coefficients);
  EXPECT_TRUE(observable.dependencies().empty());
  SemiDiscreteModel<V, M>   cell_model, boundary_model, region_model;
  SemidiscreteBuilder<V, M> cell_builder(layout, cell_model),
    boundary_builder(layout, boundary_model),
    region_builder(layout, region_model);
  const IntegrationSupport<2>         cells(mesh, 1);
  const BoundaryIntegrationSupport<2> boundary(mesh, 0);
  weak_term(observable, test(target), cells).add(cell_builder);
  weak_term(observable, test(target), boundary).add(boundary_builder);
  weak_term(observable, test(target)).on_boundary(0).add(region_builder);
  StateView<V>         view(layout, 0.);
  EvaluationContext<V> context(0., view);
  for (const double offset : {2., -1., 0.})
    {
      source_constraints.set_inhomogeneity(0, offset);
      for (auto *model : {&cell_model, &boundary_model, &region_model})
        {
          V residual(dofs.n_dofs()), expected(dofs.n_dofs());
          model->evaluate_row(target.field_id(), context, residual);
          if (model == &cell_model)
            cell_oracle(expected, source, target, cells, 1.);
          else if (model == &boundary_model || model == &region_model)
            boundary_oracle(expected, source, target, {0}, false, 1.);
          EXPECT_GT(expected.l2_norm(), 1.e-5);
          expected -= residual;
          EXPECT_LT(expected.l2_norm(), 1.e-12);
        }
    }
}

TEST(WeakTerm, AffineConstraintCellAndBoundarySupports)
{
  Triangulation<2> mesh;
  GridGenerator::hyper_cube(mesh, 0., 1., true);
  mesh.refine_global(1);
  for (const auto &cell : mesh.active_cell_iterators())
    cell->set_material_id(cell->center()[0] < 0.5 ? 1 : 2);
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  AffineConstraints<double> source_constraints, target_constraints;
  source_constraints.add_line(0);
  source_constraints.set_inhomogeneity(0, 2.);
  source_constraints.close();
  target_constraints.close();
  const auto source_space =
    fe_space(dofs, StaticMappingQ1<2>::mapping, source_constraints);
  const auto target_space =
    fe_space(dofs, StaticMappingQ1<2>::mapping, target_constraints);
  StateLayout layout;
  const auto  source = source_space.field(layout, "source");
  const auto  target = target_space.field(layout, "multiplier");
  using V            = Vector<double>;
  using M            = SparseMatrix<double>;
  SemiDiscreteModel<V, M>   cell_model, boundary_model, region_model;
  SemidiscreteBuilder<V, M> cell_builder(layout, cell_model),
    boundary_builder(layout, boundary_model),
    region_builder(layout, region_model);
  const IntegrationSupport<2>         cells(mesh, 1);
  const BoundaryIntegrationSupport<2> boundary(mesh, 0);
  const double                        sign = -1.25;
  weak_term(value(source), test(target), cells)
    .add_constraint_terms(cell_builder, target.field_id(), sign, 0);
  weak_term(value(source), test(target), boundary)
    .add_constraint_terms(boundary_builder, target.field_id(), sign, 0);
  weak_term(value(source), test(target))
    .on_boundary(0)
    .add_constraint_terms(region_builder, target.field_id(), sign, 0);
  for (const double offset : {2., -1., 0.})
    {
      source_constraints.set_inhomogeneity(0, offset);
      for (auto *model : {&cell_model, &boundary_model, &region_model})
        check_affine_model(
          *model,
          layout,
          source,
          target,
          [&](V &expected, const double free_scale) {
            if (model == &cell_model)
              cell_oracle(expected, source, target, cells, free_scale);
            else if (model == &boundary_model || model == &region_model)
              boundary_oracle(expected, source, target, {0}, false, free_scale);
            expected *= sign;
          });
    }
}

TEST(WeakTerm, AffineBoundarySupportIdsAndDynamic)
{
  Triangulation<2> mesh;
  GridGenerator::hyper_cube(mesh, 0., 1., true);
  mesh.refine_global(1);
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.add_line(0);
  constraints.set_inhomogeneity(0, 0.);
  constraints.add_line(1);
  constraints.add_entry(1, 2, 0.25);
  constraints.set_inhomogeneity(1, 0.4);
  constraints.close();
  const auto  space = fe_space(dofs, StaticMappingQ1<2>::mapping, constraints);
  StateLayout layout;
  const auto  source = space.field(layout, "source");
  const auto  target = space.field(layout, "target");
  using V            = Vector<double>;
  using M            = SparseMatrix<double>;
  SemiDiscreteModel<V, M>   explicit_model, inferred_model, normal_model;
  SemidiscreteBuilder<V, M> explicit_builder(layout, explicit_model),
    inferred_builder(layout, inferred_model),
    normal_builder(layout, normal_model);
  const BoundaryIntegrationSupport<2> support(mesh, 0);
  weak_term(value(source), test(target), support).add(explicit_builder);
  weak_term(value(source), test(target)).on_boundary(0).add(inferred_builder);
  weak_term(gradient(source) * support.normal(), test(target), support)
    .add(normal_builder);
  for (const double offset : {2., -3., 0.})
    {
      constraints.set_inhomogeneity(0, offset);
      for (auto *model : {&explicit_model, &inferred_model})
        check_affine_model(
          *model,
          layout,
          source,
          target,
          [&](V &expected, const double free_scale) {
            boundary_oracle(expected, source, target, {0}, false, free_scale);
          });
      check_affine_model(normal_model,
                         layout,
                         source,
                         target,
                         [&](V &expected, const double free_scale) {
                           boundary_oracle(
                             expected, source, target, {0}, true, free_scale);
                         });
    }
}

TEST(WeakTerm, AffineBoundarySupportEmbeddedConormal)
{
  Triangulation<1, 2> mesh;
  GridGenerator::hyper_cube(mesh, 0., 1., true);
  mesh.refine_global(1);
  FE_Q<1, 2>       fe(1);
  DoFHandler<1, 2> dofs(mesh);
  dofs.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.add_line(0);
  constraints.set_inhomogeneity(0, 2.);
  constraints.close();
  AffineConstraints<double> target_constraints;
  target_constraints.close();
  const auto space =
    fe_space(dofs, StaticMappingQ1<1, 2>::mapping, constraints);
  const auto target_space =
    fe_space(dofs, StaticMappingQ1<1, 2>::mapping, target_constraints);
  StateLayout layout;
  const auto  source = space.field(layout, "source");
  const auto  target = target_space.field(layout, "target");
  using V            = Vector<double>;
  using M            = SparseMatrix<double>;
  SemiDiscreteModel<V, M>                model;
  SemidiscreteBuilder<V, M>              builder(layout, model);
  const BoundaryIntegrationSupport<1, 2> support(mesh);
  weak_term(gradient(source) * support.conormal(), test(target), support)
    .add(builder);
  for (const double offset : {2., -1.})
    {
      constraints.set_inhomogeneity(0, offset);
      check_affine_model(model,
                         layout,
                         source,
                         target,
                         [&](V &expected, const double free_scale) {
                           boundary_oracle(expected,
                                           source,
                                           target,
                                           support.boundary_ids(),
                                           true,
                                           free_scale);
                         });
    }
}

TEST(WeakTerm, BOTH_AffineBoundarySupportHangingAndAsymmetric)
{
  parallel::distributed::Triangulation<2> mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(mesh, 0., 1., true);
  mesh.refine_global(2);
  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[0] < 0.3 &&
        cell->center()[1] < 0.3)
      cell->set_refine_flag();
  mesh.execute_coarsening_and_refinement();
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  const auto relevant = DoFTools::extract_locally_relevant_dofs(dofs);
  AffineConstraints<double> constraints(dofs.locally_owned_dofs(), relevant);
  DoFTools::make_hanging_node_constraints(dofs, constraints);
  EXPECT_GT(Utilities::MPI::sum(constraints.n_constraints(), MPI_COMM_WORLD),
            0u);
  const auto points =
    DoFTools::map_dofs_to_support_points(StaticMappingQ1<2>::mapping, dofs);
  types::global_dof_index corner = numbers::invalid_dof_index;
  for (const auto &[index, point] : points)
    if (relevant.is_element(index) && point.norm() < 1.e-12)
      {
        corner = index;
        constraints.add_line(index);
      }
  constraints.close();
  EXPECT_FALSE(constraints.has_inhomogeneities());
  const auto  space = fe_space(dofs, StaticMappingQ1<2>::mapping, constraints);
  StateLayout layout;
  const auto  source = space.field(layout, "source");
  const auto  target = space.field(layout, "target");
  using V            = ImmersXLA::MPI::Vector;
  using M            = ImmersXLA::MPI::SparseMatrix;
  SemiDiscreteModel<V, M>   explicit_model, inferred_model, normal_model;
  SemidiscreteBuilder<V, M> explicit_builder(layout, explicit_model),
    inferred_builder(layout, inferred_model),
    normal_builder(layout, normal_model);
  const BoundaryIntegrationSupport<2> support(mesh, 0);
  weak_term(value(source), test(target), support).add(explicit_builder);
  weak_term(value(source), test(target)).on_boundary(0).add(inferred_builder);
  weak_term(gradient(source) * support.normal(), test(target), support)
    .add(normal_builder);
  for (const double offset : {2., -3., 0.})
    {
      if (corner != numbers::invalid_dof_index)
        constraints.set_inhomogeneity(corner, offset);
      if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 2)
        {
          EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(
                                          constraints.has_inhomogeneities()),
                                        MPI_COMM_WORLD),
                    offset == 0. ? 0u : 1u);
        }
      for (auto *model : {&explicit_model, &inferred_model})
        check_affine_model(
          *model,
          layout,
          source,
          target,
          [&](V &expected, const double free_scale) {
            boundary_oracle(expected, source, target, {0}, false, free_scale);
          });
      check_affine_model(normal_model,
                         layout,
                         source,
                         target,
                         [&](V &expected, const double free_scale) {
                           boundary_oracle(
                             expected, source, target, {0}, true, free_scale);
                         });
    }
}
