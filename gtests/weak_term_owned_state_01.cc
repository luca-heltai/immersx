// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_lib.h>

#include <deal.II/fe/fe_simplex_p.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/domain.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/weak_term.h>
using namespace dealii;
using namespace ImmersX;

namespace
{
  void
  check_owned_nonlinear(const bool single_cell)
  {
    ParameterAcceptor::clear();
    Triangulation<2>            coarse;
    const std::vector<Point<2>> vertices =
      single_cell ?
        std::vector<Point<2>>{{0., 0.}, {1., 0.}, {0., 1.}} :
        std::vector<Point<2>>{{0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}, {.5, .5}};
    std::vector<CellData<2>> cells(single_cell ? 1 : 4, CellData<2>(3));
    if (single_cell)
      cells[0].vertices = {0, 1, 2};
    else
      for (unsigned int i = 0; i < 4; ++i)
        cells[i].vertices = {i, (i + 1) % 4, 4};
    coarse.create_triangulation(vertices, cells, SubCellData());
    if (single_cell)
      {
        const auto cell = coarse.begin_active();
        for (const auto &face : cell->face_iterators())
          face->set_boundary_id(1);
        cell->face(0)->set_boundary_id(0);
      }
    DomainParameters<2> parameters("/Tests/Owned nonlinear state/");
    parameters.triangulation_type = "fullydistributed";
    Domain<2> domain(parameters);
    domain.prepare_grid(coarse);
    domain.finalize_grid();
    if (single_cell)
      {
        EXPECT_EQ(Utilities::MPI::min(
                    domain.triangulation().n_locally_owned_active_cells(),
                    MPI_COMM_WORLD),
                  0u);
      }
    FiniteElementSpace<2> space(domain.triangulation(),
                                std::make_unique<FE_SimplexP<2>>(1));
    space.constraints().clear();
    space.constraints().reinit(space.locally_owned_dofs(),
                               space.locally_relevant_dofs());
    VectorTools::interpolate_boundary_values(space.view().mapping(),
                                             space.dof_handler(),
                                             0,
                                             Functions::ConstantFunction<2>(2.),
                                             space.constraints());
    space.constraints().close();
    StateLayout layout;
    const auto  view = space.view();
    const auto  u    = view.field(layout, "u");
    const auto  v    = view.field(layout, "v");
    using VectorType = ImmersXLA::MPI::Vector;
    using MatrixType = ImmersXLA::MPI::SparseMatrix;
    VectorType state, zero, result, ones, direction;
    for (auto *vector : {&state, &zero, &result, &ones, &direction})
      vector->reinit(space.locally_owned_dofs(), MPI_COMM_WORLD);
    state     = 2.;
    zero      = 0.;
    ones      = 1.;
    direction = 1.;
    space.constraints().set_zero(direction);
    StateView<VectorType> state_view(layout, 0.);
    state_view.bind(u.id(), state).bind(v.id(), zero);
    EvaluationContext<VectorType>               context(0., state_view);
    SemiDiscreteModel<VectorType, MatrixType>   model;
    SemidiscreteBuilder<VectorType, MatrixType> builder(layout, model);
    weak_term(value(u) * value(u), test(v)).add(builder);
    model.evaluate_row(v.id(), context, result);
    // Each triangle contributes area/3 for the free vertex shape function.
    EXPECT_NEAR(ones * result, single_cell ? 2. / 3. : 4. / 3., 1.e-12);
    model.state_operator(v.id(), u.id(), context).vmult(result, direction);
    // Integral of the square of that shape function is total area/6.
    EXPECT_NEAR(ones * result, single_cell ? 1. / 3. : 2. / 3., 1.e-12);
  }
} // namespace
TEST(WeakTerm, BOTH_ConstrainedNonlinearOwnedState)
{
  check_owned_nonlinear(false);
}
TEST(WeakTerm, MPI_ConstrainedNonlinearWithEmptyRank)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  check_owned_nonlinear(true);
}
