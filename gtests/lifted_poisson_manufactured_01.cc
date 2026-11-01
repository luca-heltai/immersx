// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_parser.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/core/boundary_conditions.h>
#include <immersx/core/domain.h>
#include <immersx/core/field_contributor.h>
#include <immersx/core/known_term.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/core/modal_lift.h>

using namespace ImmersX;
using namespace dealii;

TEST(LiftedPoisson, BOTH_ManufacturedForcingAndHangingNodes)
{
  ParameterAcceptor::clear();
  DomainParameters<3> dp("/Manufactured lift/Bulk/");
  dp.arguments_for_grid = "-1:1:false";
  dp.initial_refinement = 2;
  Domain<3> bulk(dp);
  bulk.prepare_grid(true);
  auto &mesh = bulk.refinement_triangulation();
  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->center()[0] < 0. && cell->center()[1] < 0. &&
        cell->center()[2] < 0.)
      cell->set_refine_flag();
  mesh.execute_coarsening_and_refinement();
  bulk.finalize_grid();
  DomainParameters<1, 3> rp("/Manufactured lift/Curve/");
  Domain<1, 3>           curve(rp);
  curve.prepare_grid(true);
  curve.refinement_triangulation().clear();
  std::vector<CellData<1>> cells(2);
  cells[0].vertices[0] = 0;
  cells[0].vertices[1] = 1;
  cells[1].vertices[0] = 1;
  cells[1].vertices[1] = 2;
  curve.refinement_triangulation().create_triangulation({Point<3>(0., 0., -.5),
                                                         Point<3>(0., 0., 0.),
                                                         Point<3>(0., 0., .5)},
                                                        cells,
                                                        SubCellData());
  curve.finalize_grid();
  FiniteElementSpaceParameters<3>    fp("/Manufactured lift/Bulk space/",
                                     "FE_Q<3>(2)");
  FiniteElementSpaceParameters<1, 3> mp("/Manufactured lift/Modal space/",
                                        "FESystem<1,3>[FE_Q<1,3>(2)^3]");
  FiniteElementSpace<3>              space(bulk.triangulation(), fp);
  FiniteElementSpace<1, 3>           modes(curve.triangulation(), mp);
  AffineConstraints<double>          hanging;
  DoFTools::make_hanging_node_constraints(space.dof_handler(), hanging);
  hanging.close();
  EXPECT_GT(Utilities::MPI::sum(hanging.n_constraints(), MPI_COMM_WORLD), 0u)
    << "Background cells: " << bulk.triangulation().n_global_active_cells();
  FunctionParser<3> exact;
  exact.initialize("x,y,z", "2+z-z*z", {});
  BoundaryConditions<3> bc(space);
  bc.add_dirichlet(0, exact);
  bc.update(0.);
  LinearAdapterParameters options("/Manufactured lift/Solver/");
  options.solver = LinearSolver::direct;
  LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector> adapter(
    options, MPI_COMM_WORLD);
  const auto view       = space.view();
  const auto modal_view = modes.view();
  const auto u = adapter.add(algebraic_field(view.field("u"))).fields();
  adapter.add(weak_term(gradient(u), gradient(test(u))));
  // -Delta(2+z-z^2)=2. The known term belongs to F=Au+B^T lambda-f.
  const KnownSource<double, 3> forcing([](const auto &) { return -2.; });
  adapter.add(known_term(forcing, test(u)));
  TensorProductLift<1, 2, 3> lift("/Manufactured lift/Lift/");
  lift.thickness                     = "0.2";
  lift.section.inclusion_degree      = 1;
  lift.section.selected_coefficients = {0, 1, 2};
  const auto geometry = modal_lift(modal_view.field("lambda"), lift);
  // u is constant across each transverse circle, so only mode zero is nonzero.
  auto data = std::make_shared<FunctionParser<3>>(3);
  data->initialize("x,y,z", "2+z-z*z;0;0", {});
  const auto lambda =
    adapter.add(LiftedMultiplierInteraction(u, geometry, data))
      .fields()
      .multiplier;
  auto state = adapter.make_state();
  adapter.solve(state);
  auto residual = adapter.make_state();
  adapter.evaluate_residual(state, residual);
  EXPECT_LT(residual.l2_norm(), options.tolerance);
  EXPECT_LT(adapter.field(state, lambda).l2_norm(), 1.e-8);
  auto solution = adapter.field(state, u);
  view.constraints().distribute(solution);
  const auto points =
    DoFTools::map_dofs_to_support_points(view.mapping(), space.dof_handler());
  double error = 0.;
  for (const auto i : solution.locally_owned_elements())
    error = std::max(error, std::abs(solution[i] - exact.value(points.at(i))));
  EXPECT_LT(Utilities::MPI::max(error, MPI_COMM_WORLD), 1.e-8);
}
