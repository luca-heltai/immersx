// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/sparse_matrix.h>

#include <gtest/gtest.h>
#include <immersx/core/weak_term.h>

using namespace dealii;
using namespace ImmersX;

namespace
{
  template <int dim, int spacedim>
  void
  check_vector_coupling(const bool locally_refined = false)
  {
    SCOPED_TRACE("dim=" + std::to_string(dim) +
                 ", spacedim=" + std::to_string(spacedim));
    Triangulation<spacedim> background;
    GridGenerator::hyper_cube(background, -1., 1.);
    Triangulation<dim, spacedim> embedded;
    // Equal-dimensional pairings use the same domain in either direction.
    GridGenerator::hyper_cube(embedded,
                              dim == spacedim ? -1. : -0.6,
                              dim == spacedim ? 1. : 0.6);
    if (locally_refined)
      {
        background.refine_global(1);
        background.begin_active()->set_refine_flag();
        background.execute_coarsening_and_refinement();
        GridTools::transform(
          [](const Point<spacedim> &point) {
            auto shifted = point;
            for (unsigned int d = dim; d < spacedim; ++d)
              shifted[d] = -0.3;
            return shifted;
          },
          embedded);
        embedded.refine_global(1);
        embedded.begin_active()->set_refine_flag();
        embedded.execute_coarsening_and_refinement();
      }
    FESystem<spacedim>        background_fe(FE_Q<spacedim>(1), spacedim + 3);
    FESystem<dim, spacedim>   embedded_fe(FE_Q<dim, spacedim>(1), spacedim + 3);
    DoFHandler<spacedim>      background_dh(background);
    DoFHandler<dim, spacedim> embedded_dh(embedded);
    background_dh.distribute_dofs(background_fe);
    embedded_dh.distribute_dofs(embedded_fe);
    AffineConstraints<double> background_constraints, embedded_constraints;
    DoFTools::make_hanging_node_constraints(background_dh,
                                            background_constraints);
    DoFTools::make_hanging_node_constraints(embedded_dh, embedded_constraints);
    background_constraints.close();
    embedded_constraints.close();
    if (locally_refined)
      {
        EXPECT_GT(background_constraints.n_constraints(), 0u);
        if constexpr (dim > 1)
          {
            EXPECT_GT(embedded_constraints.n_constraints(), 0u);
          }
      }
    MappingQ1<spacedim>      background_mapping;
    MappingQ1<dim, spacedim> embedded_mapping;
    const auto               background_space =
      fe_space(background_dh, background_mapping, background_constraints);
    const auto embedded_space =
      fe_space(embedded_dh, embedded_mapping, embedded_constraints);
    const auto b = value(background_space.field(FieldId(0),
                                                "background",
                                                FEValuesExtractors::Vector(1)));
    const auto e = value(embedded_space.field(FieldId(0),
                                              "embedded",
                                              FEValuesExtractors::Vector(2)));
    using Matrix = SparseMatrix<double>;
    using Vector = dealii::Vector<double>;
    const auto reverse =
      detail::WeakAssembly<decltype(b),
                           decltype(e)>::template assemble<Vector, Matrix>(b,
                                                                           e);
    const auto forward =
      detail::WeakAssembly<decltype(e),
                           decltype(b)>::template assemble<Vector, Matrix>(e,
                                                                           b);

    // Direct FEValuesViews quadrature does not use component masks or the
    // NonMatching assembler. Check transverse and unselected components too.
    const QGauss<dim>       quadrature(2);
    FEValues<dim, spacedim> embedded_values(embedded_mapping,
                                            embedded_fe,
                                            quadrature,
                                            update_values |
                                              update_quadrature_points |
                                              update_JxW_values);
    DynamicSparsityPattern  sparsity(embedded_dh.n_dofs(),
                                    background_dh.n_dofs());
    for (unsigned int i = 0; i < embedded_dh.n_dofs(); ++i)
      for (unsigned int j = 0; j < background_dh.n_dofs(); ++j)
        sparsity.add(i, j);
    SparsityPattern pattern;
    pattern.copy_from(sparsity);
    Matrix                               reference(pattern);
    std::vector<types::global_dof_index> bi(background_fe.n_dofs_per_cell()),
      ei(embedded_fe.n_dofs_per_cell());
    for (const auto &embedded_cell : embedded_dh.active_cell_iterators())
      {
        embedded_values.reinit(embedded_cell);
        embedded_cell->get_dof_indices(ei);
        for (const auto q : embedded_values.quadrature_point_indices())
          {
            const auto background_point =
              GridTools::find_active_cell_around_point(
                background_mapping,
                background_dh,
                embedded_values.quadrature_point(q));
            ASSERT_NE(background_point.first, background_dh.end());
            FEValues<spacedim> background_values(background_mapping,
                                                 background_fe,
                                                 Quadrature<spacedim>(
                                                   std::vector<Point<spacedim>>{
                                                     background_point.second}),
                                                 update_values);
            background_values.reinit(background_point.first);
            background_point.first->get_dof_indices(bi);
            FullMatrix<double> local(ei.size(), bi.size());
            for (unsigned int i = 0; i < ei.size(); ++i)
              for (unsigned int j = 0; j < bi.size(); ++j)
                local(i, j) =
                  embedded_values[FEValuesExtractors::Vector(2)].value(i, q) *
                  background_values[FEValuesExtractors::Vector(1)].value(j, 0) *
                  embedded_values.JxW(q);
            embedded_constraints.distribute_local_to_global(
              local, ei, background_constraints, bi, reference);
          }
      }
    ASSERT_EQ(reverse.matrix->m(), reference.m());
    ASSERT_EQ(reverse.matrix->n(), reference.n());
    ASSERT_EQ(forward.matrix->m(), reference.n());
    ASSERT_EQ(forward.matrix->n(), reference.m());
    EXPECT_GT(reference.frobenius_norm(), 0.);
    for (unsigned int i = 0; i < reference.m(); ++i)
      for (unsigned int j = 0; j < reference.n(); ++j)
        {
          EXPECT_NEAR(reverse.matrix->el(i, j), reference.el(i, j), 1.e-13);
          EXPECT_NEAR(forward.matrix->el(j, i), reference.el(i, j), 1.e-13);
        }
  }
} // namespace

TEST(WeakTermNonmatching, NativeVectorCouplingPreservesAmbientComponents)
{
  check_vector_coupling<1, 1>();
  check_vector_coupling<1, 2>();
  check_vector_coupling<1, 3>();
  check_vector_coupling<2, 3>();
}

TEST(WeakTermNonmatching, NativeVectorCouplingWithHangingNodes)
{
  check_vector_coupling<1, 2>(true);
  check_vector_coupling<1, 3>(true);
  check_vector_coupling<2, 3>(true);
}
