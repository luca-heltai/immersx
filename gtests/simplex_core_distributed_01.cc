// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_lib.h>

#include <deal.II/fe/fe_simplex_p.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/constraint.h>
#include <immersx/core/domain.h>
#include <immersx/core/known_term.h>

using namespace dealii;
using namespace ImmersX;

namespace
{
  template <int dim>
  void
  check_distributed_simplex_integrals()
  {
    ParameterAcceptor::clear();
    Triangulation<dim> cube, coarse;
    GridGenerator::hyper_cube(cube);
    GridGenerator::convert_hypercube_to_simplex_mesh(cube, coarse);
    DomainParameters<dim> parameters("/Tests/DistributedSimplex/");
    parameters.triangulation_type = "fullydistributed";
    parameters.initial_refinement = 0;
    Domain<dim> domain(parameters);
    domain.prepare_grid(coarse);
    domain.finalize_grid();
    FiniteElementSpace<dim> space(domain.triangulation(),
                                  std::make_unique<FE_SimplexP<dim>>(1));
    const auto              V = space.view();
    StateLayout             layout;
    const auto              source = V.field(layout, "source");
    const auto              target = V.field(layout, "target");
    using VectorType               = ImmersXLA::MPI::Vector;
    using MatrixType               = ImmersXLA::MPI::SparseMatrix;
    using Model                    = SemiDiscreteModel<VectorType, MatrixType>;
    VectorType ones, zero;
    ones.reinit(space.locally_owned_dofs(), MPI_COMM_WORLD);
    zero.reinit(space.locally_owned_dofs(), MPI_COMM_WORLD);
    ones = 1.;
    zero = 0.;
    StateView<VectorType> view(layout, 0.);
    view.bind(source.id(), ones);
    view.bind(target.id(), zero);
    const EvaluationContext<VectorType> context(0., view);
    auto check_integral = [&](const auto &term, const double expected) {
      Model                                       model;
      SemidiscreteBuilder<VectorType, MatrixType> builder(layout, model);
      term.add(builder);
      VectorType result;
      result.reinit(space.locally_owned_dofs(), MPI_COMM_WORLD);
      model.evaluate_row(target.id(), context, result);
      EXPECT_NEAR(ones * result, expected, 1.e-12);
    };
    check_integral(weak_term(value(source), test(target)), 1.);
    check_integral(weak_term(value(source), test(target)).on_boundary(0),
                   2. * dim);
    Functions::ConstantFunction<dim> two(2.);
    check_integral(known_term(known_function<dim>(two), test(target)), 2.);
    VectorType frozen_ones;
    frozen_ones.reinit(space.locally_owned_dofs(),
                       space.locally_relevant_dofs(),
                       MPI_COMM_WORLD);
    frozen_ones = ones;
    check_integral(known_term(frozen(source, frozen_ones), test(target)), 1.);
    check_integral(
      known_term(frozen(source, frozen_ones), test(target)).on_boundary(0),
      2. * dim);

    Model                                       model;
    SemidiscreteBuilder<VectorType, MatrixType> builder(layout, model);
    const auto                                  fields =
      make_constraint(weak_term(value(source), test(V.field("lambda"))))
        .add(builder);
    StateView<VectorType> constraint_view(layout, 0.);
    constraint_view.bind(source.id(), ones);
    constraint_view.bind(fields.multiplier.id(), ones);
    const EvaluationContext<VectorType> constraint_context(0., constraint_view);
    VectorType                          result;
    result.reinit(space.locally_owned_dofs(), MPI_COMM_WORLD);
    model.evaluate_row(fields.multiplier.id(), constraint_context, result);
    EXPECT_NEAR(ones * result, 1., 1.e-12);
    const auto metric =
      model.multiplier_metric(fields.multiplier.id(), constraint_context);
    ASSERT_TRUE(metric.has_value());
    metric->matrix()->vmult(result, ones);
    EXPECT_NEAR(ones * result, 1., 1.e-12);
  }
} // namespace

TEST(WeakTerm, BOTH_DistributedSimplexTriangleIntegrals)
{
  check_distributed_simplex_integrals<2>();
}

TEST(WeakTerm, BOTH_DistributedSimplexTetrahedronIntegrals)
{
  check_distributed_simplex_integrals<3>();
}
