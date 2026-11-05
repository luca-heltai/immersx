// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on deal.II.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/matrix_creator.h>

#include <gtest/gtest.h>
#include <immersx/core/contributor.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/weak_term.h>

#include <algorithm>
#include <array>
#include <cmath>

using namespace dealii;
using namespace ImmersX;

TEST(WeakTerm, BOTH_VectorCellAssemblyMatchesMatrixCreatorOnHangingMesh)
{
  parallel::distributed::Triangulation<2> mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_rectangle(mesh, Point<2>(0., 0.), Point<2>(2., 1.));
  mesh.refine_global(2);
  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[0] < .5 &&
        cell->center()[1] < .5)
      cell->set_refine_flag();
  mesh.execute_coarsening_and_refinement();
  for (const unsigned int degree : {1u, 2u})
    {
      SCOPED_TRACE(degree);
      FESystem<2>   fe(FE_Q<2>(degree), 2);
      DoFHandler<2> dofs(mesh);
      dofs.distribute_dofs(fe);
      const auto owned    = dofs.locally_owned_dofs();
      const auto relevant = DoFTools::extract_locally_relevant_dofs(dofs);
      AffineConstraints<double> constraints(owned, relevant);
      DoFTools::make_hanging_node_constraints(dofs, constraints);
      constraints.close();
      EXPECT_GT(Utilities::MPI::sum(constraints.n_constraints(),
                                    MPI_COMM_WORLD),
                0u);
      const auto &mapping = StaticMappingQ1<2>::mapping;
      const auto  space   = fe_space(dofs, mapping, constraints, &relevant);
      StateLayout layout;
      const auto  u = space.field(layout, "u", FEValuesExtractors::Vector(0));
      using V       = ImmersXLA::MPI::Vector;
      using M       = ImmersXLA::MPI::SparseMatrix;
      const StateView<V>         view(layout, 0.);
      const EvaluationContext<V> context(0., view);
      DynamicSparsityPattern sparsity(dofs.n_dofs(), dofs.n_dofs(), relevant);
      DoFTools::make_sparsity_pattern(dofs, sparsity, constraints, true);
      SparsityTools::distribute_sparsity_pattern(sparsity,
                                                 owned,
                                                 MPI_COMM_WORLD,
                                                 relevant);
      const QGauss<2> quadrature(degree + 1);
      for (const bool gradient_term : {false, true})
        {
          SCOPED_TRACE(gradient_term);
          SemiDiscreteModel<V, M>   model;
          SemidiscreteBuilder<V, M> builder(layout, model);
          if (gradient_term)
            weak_term(gradient(u), gradient(test(u))).add(builder);
          else
            weak_term(value(u), test(u)).add(builder);
          const auto op = model.state_matrix_operator(u.id(), u.id(), context);
          ASSERT_TRUE(op.has_value());
          const auto actual = op->matrix();
          M          expected;
          expected.reinit(owned, owned, sparsity, MPI_COMM_WORLD);
          if (gradient_term)
            MatrixCreator::create_laplace_matrix(
              mapping,
              dofs,
              quadrature,
              expected,
              static_cast<const Function<2> *>(nullptr),
              constraints);
          else
            MatrixCreator::create_mass_matrix(mapping,
                                              dofs,
                                              quadrature,
                                              expected,
                                              static_cast<const Function<2> *>(
                                                nullptr),
                                              constraints);
          V probe, result, reference;
          probe.reinit(owned, MPI_COMM_WORLD);
          result.reinit(probe);
          reference.reinit(probe);
          for (unsigned int sample = 1; sample <= 3; ++sample)
            {
              for (const auto i : owned)
                probe[i] = std::sin(sample * (i + 1.));
              actual->vmult(result, probe);
              expected.vmult(reference, probe);
              result -= reference;
              EXPECT_LE(result.l2_norm(), 1.e-11 * reference.l2_norm());
            }
        }
    }
}

TEST(WeakTerm, BoundaryMassMatchesMatrixCreator)
{
  for (const unsigned int degree : {1u, 2u})
    for (const bool refined : {false, true})
      {
        SCOPED_TRACE("degree=" + std::to_string(degree) +
                     " local refinement=" + std::to_string(refined));
        Triangulation<2> mesh;
        GridGenerator::hyper_cube(mesh, 0., 1., true);
        mesh.refine_global(1);
        if (refined)
          {
            mesh.begin_active()->set_refine_flag();
            mesh.execute_coarsening_and_refinement();
          }
        FE_Q<2>       fe(degree);
        DoFHandler<2> dh(mesh);
        dh.distribute_dofs(fe);
        AffineConstraints<double> constraints;
        DoFTools::make_hanging_node_constraints(dh, constraints);
        constraints.close();
        if (refined)
          {
            ASSERT_GT(constraints.n_constraints(), 0u);
          }
        const auto &mapping = StaticMappingQ1<2>::mapping;
        const auto  space   = fe_space(dh, mapping, constraints);
        StateLayout layout;
        const auto  u = space.field(layout, "u");
        const StateView<Vector<double>>         state(layout, 0.);
        const EvaluationContext<Vector<double>> context(0., state);

        for (const std::set<types::boundary_id> &ids :
             {std::set<types::boundary_id>{0}, {1}, {0, 1, 2, 3}})
          {
            using Model =
              SemiDiscreteModel<Vector<double>, SparseMatrix<double>>;
            Model                                                     model;
            SemidiscreteBuilder<Vector<double>, SparseMatrix<double>> builder(
              layout, model);
            for (const auto id : ids)
              weak_term(value(u),
                        test(u),
                        BoundaryIntegrationSupport<2>(mesh, id))
                .add(builder);
            const auto op =
              model.state_matrix_operator(u.field_id(), u.field_id(), context);
            ASSERT_TRUE(op.has_value());
            const auto matrix = op->matrix();
            ASSERT_NE(matrix, nullptr);

            std::vector<types::global_dof_index> boundary_indices(dh.n_dofs());
            DoFTools::map_dof_to_boundary_indices(dh, ids, boundary_indices);
            const auto                 n_boundary = dh.n_boundary_dofs(ids);
            Functions::ZeroFunction<2> zero;
            std::map<types::boundary_id, const Function<2> *> functions;
            for (const auto id : ids)
              functions[id] = &zero;
            DynamicSparsityPattern dsp(n_boundary, n_boundary);
            DoFTools::make_boundary_sparsity_pattern(dh,
                                                     functions,
                                                     boundary_indices,
                                                     dsp);
            SparsityPattern sparsity;
            sparsity.copy_from(dsp);
            SparseMatrix<double> native(sparsity);
            Vector<double>       rhs(n_boundary);
            MatrixCreator::create_boundary_mass_matrix(mapping,
                                                       dh,
                                                       QGauss<1>(degree + 1),
                                                       native,
                                                       functions,
                                                       rhs,
                                                       boundary_indices);

            // MatrixCreator returns the unscaled boundary-only mass matrix.
            // Restrict/prolong its numbering and apply C^T M C algebraically.
            // Ignore constrained columns: their artificial diagonals are not
            // part of the variational operator on the unconstrained space.
            Vector<double> probe(dh.n_dofs()), expected(dh.n_dofs()),
              actual(dh.n_dofs());
            Vector<double> boundary_probe(n_boundary),
              boundary_action(n_boundary);
            const auto native_action = [&]() {
              auto distributed_probe = probe;
              constraints.distribute(distributed_probe);
              for (unsigned int i = 0; i < dh.n_dofs(); ++i)
                if (boundary_indices[i] != numbers::invalid_dof_index)
                  boundary_probe[boundary_indices[i]] = distributed_probe[i];
              native.vmult(boundary_action, boundary_probe);
              expected = 0.;
              for (unsigned int i = 0; i < dh.n_dofs(); ++i)
                if (boundary_indices[i] != numbers::invalid_dof_index)
                  expected[i] = boundary_action[boundary_indices[i]];
              constraints.condense(expected);
            };
            for (unsigned int j = 0; j < dh.n_dofs(); ++j)
              if (!constraints.is_constrained(j))
                {
                  probe    = 0.;
                  probe[j] = 1.;
                  native_action();
                  matrix->vmult(actual, probe);
                  for (unsigned int i = 0; i < dh.n_dofs(); ++i)
                    if (!constraints.is_constrained(i))
                      {
                        EXPECT_NEAR(actual[i], expected[i], 2.e-13);
                      }
                }
            // A nonconstant, non-polynomial nodal vector also checks actions
            // and the transpose, not only the constant-field row sums.
            for (unsigned int j = 0; j < dh.n_dofs(); ++j)
              probe[j] = constraints.is_constrained(j) ? 0. : std::sin(1.7 * j);
            native_action();
            op->view.vmult(actual, probe);
            actual -= expected;
            EXPECT_LT(actual.l2_norm(), 2.e-12);
            op->view.Tvmult(actual, probe);
            actual -= expected;
            EXPECT_LT(actual.l2_norm(), 2.e-12);
          }
      }
}

TEST(WeakTerm, IndependentSupportMatchesPiecewiseFETrace)
{
  Triangulation<2> background;
  GridGenerator::hyper_cube(background, -1., 1.);
  background.refine_global(2);
  FE_Q<2>       fe(1);
  DoFHandler<2> dh(background);
  dh.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();
  const auto  space = fe_space(dh, StaticMappingQ1<2>::mapping, constraints);
  StateLayout layout;
  const auto  u = space.field(layout, "u");

  // The trace is continuous piecewise linear, with slope jumps at three
  // background interfaces. A single global polynomial would hide aliasing.
  const auto trace = [](const double x) {
    const std::array<double, 5> values{{0., 1., -0.4, 1.7, 0.3}};
    const unsigned int          cell =
      std::min(3u, static_cast<unsigned int>(2. * (x + 1.)));
    const double t = 2. * (x + 1.) - cell;
    return (1. - t) * values[cell] + t * values[cell + 1];
  };
  std::vector<Point<2>> points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points(StaticMappingQ1<2>::mapping, dh, points);
  Vector<double> coefficients(dh.n_dofs());
  for (unsigned int i = 0; i < dh.n_dofs(); ++i)
    coefficients[i] = trace(points[i][0]);

  // Independent oracle: a one-dimensional FE space fitted to the background
  // interfaces, integrated exactly by MatrixCreator, not another trace loop.
  Triangulation<1>            reference_mesh;
  const std::vector<Point<1>> vertices{Point<1>(-0.83),
                                       Point<1>(-0.5),
                                       Point<1>(0.),
                                       Point<1>(0.5),
                                       Point<1>(0.79)};
  std::vector<CellData<1>>    cells(vertices.size() - 1);
  for (unsigned int c = 0; c < cells.size(); ++c)
    cells[c].vertices = {{c, c + 1}};
  reference_mesh.create_triangulation(vertices, cells, SubCellData());
  FE_Q<1>       reference_fe(1);
  DoFHandler<1> reference_dh(reference_mesh);
  reference_dh.distribute_dofs(reference_fe);
  DynamicSparsityPattern dsp(reference_dh.n_dofs());
  DoFTools::make_sparsity_pattern(reference_dh, dsp);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);
  SparseMatrix<double> reference_mass(sparsity);
  MatrixCreator::create_mass_matrix(StaticMappingQ1<1>::mapping,
                                    reference_dh,
                                    QGauss<1>(2),
                                    reference_mass);
  std::vector<Point<1>> reference_points(reference_dh.n_dofs());
  DoFTools::map_dofs_to_support_points(StaticMappingQ1<1>::mapping,
                                       reference_dh,
                                       reference_points);
  Vector<double> reference_coefficients(reference_dh.n_dofs()),
    reference_action(reference_dh.n_dofs());
  for (unsigned int i = 0; i < reference_dh.n_dofs(); ++i)
    reference_coefficients[i] = trace(reference_points[i][0]);
  reference_mass.vmult(reference_action, reference_coefficients);
  const double reference_energy = reference_coefficients * reference_action;

  // Resolve the three slope jumps explicitly: this is an exact quadrature
  // regression on four segments, rather than a mesh-convergence study.
  std::vector<Point<2>> support_vertices;
  for (const auto &vertex : vertices)
    support_vertices.emplace_back(vertex[0], 0.17);
  Triangulation<1, 2> support_mesh;
  support_mesh.create_triangulation(support_vertices, cells, SubCellData());
  const IntegrationSupport<1, 2> support(support_mesh);
  using Model = SemiDiscreteModel<Vector<double>, SparseMatrix<double>>;
  Model                                                     model;
  SemidiscreteBuilder<Vector<double>, SparseMatrix<double>> builder(layout,
                                                                    model);
  weak_term(value(u), test(u), support).add(builder);
  const StateView<Vector<double>>         state(layout, 0.);
  const EvaluationContext<Vector<double>> context(0., state);
  const auto                              op =
    model.state_matrix_operator(u.field_id(), u.field_id(), context);
  ASSERT_TRUE(op.has_value());
  Vector<double> action(dh.n_dofs());
  op->view.vmult(action, coefficients);
  EXPECT_NEAR(coefficients * action, reference_energy, 1.e-12);
}

TEST(WeakTerm, IndependentSupportRejectsDifferentBackgroundsAndMappings)
{
  Triangulation<2> mesh, other_mesh;
  GridGenerator::hyper_cube(mesh);
  GridGenerator::hyper_cube(other_mesh);
  FE_Q<2>       fe(1);
  DoFHandler<2> dh(mesh), other_dh(other_mesh);
  dh.distribute_dofs(fe);
  other_dh.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();
  MappingQ1<2> other_mapping;
  const auto   source_space =
    fe_space(dh, StaticMappingQ1<2>::mapping, constraints);
  const auto other_space =
    fe_space(other_dh, StaticMappingQ1<2>::mapping, constraints);
  const auto          mapped_space = fe_space(dh, other_mapping, constraints);
  StateLayout         layout;
  const auto          u      = source_space.field(layout, "u");
  const auto          other  = other_space.field(layout, "other");
  const auto          mapped = mapped_space.field(layout, "mapped");
  Triangulation<1, 2> support_mesh;
  CellData<1>         cell;
  cell.vertices = {{0, 1}};
  support_mesh.create_triangulation({Point<2>(0.1, 0.2), Point<2>(0.9, 0.2)},
                                    {cell},
                                    SubCellData());
  const IntegrationSupport<1, 2> support(support_mesh);
  for (const auto &target : {other, mapped})
    {
      using Model = SemiDiscreteModel<Vector<double>, SparseMatrix<double>>;
      Model                                                     model;
      SemidiscreteBuilder<Vector<double>, SparseMatrix<double>> builder(layout,
                                                                        model);
      const auto assemble = [&]() {
        weak_term(value(u), test(target), support).add(builder);
        const StateView<Vector<double>>         state(layout, 0.);
        const EvaluationContext<Vector<double>> context(0., state);
        const auto                              op =
          model.state_matrix_operator(target.field_id(), u.field_id(), context);
        return op->matrix();
      };
      EXPECT_THROW(assemble(), ExceptionBase);
    }
}
