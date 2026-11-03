// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/core/boundary_conditions.h>
#include <immersx/core/domain.h>
#include <immersx/core/field_contributor.h>
#include <immersx/core/known_term.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/core/modal_lift.h>

#include <array>

using namespace ImmersX;
using namespace dealii;

TEST(LiftedPoisson, BOTH_NonzeroBoundaryAndMultipleModes)
{
  ParameterAcceptor::clear();
  DomainParameters<3> dp("/Lifted test/Bulk/");
  dp.arguments_for_grid = "-1:1:false";
  dp.initial_refinement = 2;
  Domain<3> bulk(dp);
  bulk.make_grid();
  DomainParameters<1, 3> rp("/Lifted test/Curve/");
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
  FiniteElementSpaceParameters<3>    fp("/Lifted test/Bulk space/");
  FiniteElementSpaceParameters<1, 3> mp("/Lifted test/Modal space/",
                                        "FESystem<1,3>[FE_Q<1,3>(1)^3]");
  FiniteElementSpace<3>              fs(bulk.triangulation(), fp);
  FiniteElementSpace<1, 3>           ms(curve.triangulation(), mp);
  BoundaryConditions<3>              bc(fs);
  Functions::ConstantFunction<3>     boundary(2.);
  bc.add_dirichlet(0, boundary);
  bc.update(0.);
  auto                    view = fs.view();
  auto                    mv   = ms.view();
  LinearAdapterParameters options("/Lifted test/Solver/");
  options.solver = LinearSolver::direct;
  LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector> adapter(
    options, MPI_COMM_WORLD);
  const auto u = adapter.add(algebraic_field(view.field("u"))).fields();
  adapter.add(weak_term(gradient(u), gradient(test(u))));
  TensorProductLift<1, 2, 3> lift("/Lifted test/Lift/");
  lift.thickness                     = "0.2";
  lift.section.inclusion_degree      = 1;
  lift.section.selected_coefficients = {0, 1, 2};
  auto geometry                      = modal_lift(mv.field("lambda"), lift);
  // The constant trace has modal coefficient 2 in the constant mode and zero
  // coefficients in the two degree-one modes. Nonhomogeneous BC enter B u.
  auto data = std::make_shared<Functions::ConstantFunction<3>>(
    std::vector<double>{2., 0., 0.});
  const auto lambda =
    adapter.add(LiftedMultiplierInteraction(u, geometry, data))
      .fields()
      .multiplier;
  auto state = adapter.make_state();
  adapter.solve(state);
  auto residual = adapter.make_state();
  adapter.evaluate_residual(state, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-9);
  EXPECT_LT(adapter.field(state, lambda).l2_norm(), 1.e-8);
  double error = 0.;
  for (auto i : adapter.field(state, u).locally_owned_elements())
    error = std::max(error, std::abs(adapter.field(state, u)[i] - 2.));
  EXPECT_LT(Utilities::MPI::max(error, MPI_COMM_WORLD), 1.e-9);
  boundary.set_time(1.);
  // Refresh the same persistent constraints after assembly, without rebuilding
  // B. The modal row must see the changed affine trace as well as the bulk row.
  Functions::ConstantFunction<3> changed_boundary(3.);
  bc.clear();
  bc.add_dirichlet(0, changed_boundary);
  bc.update(0.);
  adapter.evaluate_residual(state, residual);
  EXPECT_GT(residual.l2_norm(), 1.e-3);
  adapter.solve(state);
  adapter.evaluate_residual(state, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-9);
}


namespace
{
  void
  check_modal_overlap(const double center_x)
  {
    ParameterAcceptor::clear();
    DomainParameters<3> bulk_parameters("/Modal overlap/Bulk/");
    bulk_parameters.arguments_for_grid = "-1:1:false";
    bulk_parameters.initial_refinement = 1;
    Domain<3> bulk(bulk_parameters);
    bulk.make_grid();
    DomainParameters<1, 3> curve_parameters("/Modal overlap/Curve/");
    Domain<1, 3>           curve(curve_parameters);
    curve.prepare_grid(true);
    curve.refinement_triangulation().clear();
    std::vector<CellData<1>> cells(1);
    cells[0].vertices = {{0, 1}};
    curve.refinement_triangulation().create_triangulation(
      {Point<3>(center_x, 0., -.5), Point<3>(center_x, 0., .5)},
      cells,
      SubCellData());
    curve.finalize_grid();
    FiniteElementSpace<3>    bulk_space(bulk.triangulation(),
                                     std::make_unique<FE_Q<3>>(1));
    FiniteElementSpace<1, 3> modal_space(
      curve.triangulation(),
      std::make_unique<FESystem<1, 3>>(FE_DGQ<1, 3>(0), 3));
    BoundaryConditions<3>      boundary(bulk_space);
    Functions::ZeroFunction<3> zero;
    boundary.add_dirichlet(0, zero);
    boundary.update(0.);
    const auto                 bulk_view  = bulk_space.view();
    const auto                 modal_view = modal_space.view();
    TensorProductLift<1, 2, 3> descriptor("/Modal overlap/Lift/");
    descriptor.thickness                     = "0.2";
    descriptor.section.inclusion_degree      = 1;
    descriptor.section.selected_coefficients = {0, 1, 2};
    const auto geometry = modal_lift(modal_view.field("lambda"), descriptor);
    GridTools::Cache<3> cache(bulk.triangulation(), bulk_view.mapping());
    const auto          supports =
      make_overlap_supports(curve.support(),
                            modal_view.mapping(),
                            geometry.support().representative_quadrature(),
                            geometry.lifted_points(),
                            cache);
    ModalDataEvaluator data = [](const auto &, const auto &) {
      return std::vector<double>{2., 1., -.5};
    };
    const auto [metric, outside, rhs] =
      geometry.assemble_overlap(data, supports.overlap);
    ImmersXLA::MPI::Vector amplitudes;
    amplitudes.reinit(modal_space.locally_owned_dofs(), MPI_COMM_WORLD);
    const auto                          &fe = modal_space.finite_element();
    std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
    for (const auto &cell : modal_space.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            amplitudes[indices[i]] = std::array<double, 3>{
              {2., 1., -.5}}[fe.system_to_component_index(i).first];
        }
    amplitudes.compress(VectorOperation::insert);
    ImmersXLA::MPI::SparseMatrix reference_metric;
    ImmersXLA::MPI::Vector       unused_rhs;
    const auto                   retained_sparsity =
      geometry.assemble_representative(reference_metric, unused_rhs, data);
    auto full_action = amplitudes, reference_action = amplitudes;
    metric->vmult(full_action, amplitudes);
    reference_metric.vmult(reference_action, amplitudes);
    full_action -= reference_action;
    EXPECT_LT(full_action.l2_norm(), 1.e-12);
    auto expected = amplitudes, exterior = amplitudes;
    metric->vmult(expected, amplitudes);
    outside->vmult(exterior, amplitudes);
    expected -= exterior;
    expected -= *rhs;
    EXPECT_LT(expected.l2_norm(), 1.e-12);
    double cross = 0.;
    for (const auto &cell : modal_space.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(indices);
          cross += std::abs(outside->el(indices[0], indices[1])) +
                   std::abs(outside->el(indices[0], indices[2]));
        }
    if (center_x > 1.2)
      {
        EXPECT_LT(rhs->l2_norm(), 1.e-12);
        auto difference = amplitudes;
        metric->vmult(difference, amplitudes);
        difference -= exterior;
        EXPECT_LT(difference.l2_norm(), 1.e-12);
      }
    else
      {
        EXPECT_GT(rhs->l2_norm(), 1.e-3);
        EXPECT_GT(Utilities::MPI::sum(cross, MPI_COMM_WORLD), 1.e-3);
      }

    LinearAdapterParameters options("/Modal overlap/Solver/");
    options.solver = LinearSolver::direct;
    LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector> adapter(
      options, MPI_COMM_WORLD);
    const auto u = adapter.add(algebraic_field(bulk_view.field("u"))).fields();
    adapter.add(weak_term(gradient(u), gradient(test(u))));
    const KnownSource<double, 3> force([](const auto &) { return -1.; });
    adapter.add(known_term(force, test(u)));
    const auto multiplier =
      adapter.add(LiftedMultiplierInteraction(u, geometry, data))
        .fields()
        .multiplier;
    auto state = adapter.make_state();
    adapter.solve(state);
    auto residual = adapter.make_state();
    adapter.evaluate_residual(state, residual);
    EXPECT_LT(residual.l2_norm(), 1.e-9);
    EXPECT_GT(adapter.field(state, u).l2_norm(), 1.e-6);
    if (center_x > 1.2)
      {
        EXPECT_LT(adapter.field(state, multiplier).l2_norm(), 1.e-10);
      }
  }
} // namespace

TEST(LiftedPoisson, BOTH_PartialOverlapRetainsCrossModeGram)
{
  check_modal_overlap(.95);
}

TEST(LiftedPoisson, BOTH_ExteriorWallHasZeroMultiplierWithNonzeroForcing)
{
  check_modal_overlap(2.);
}
