// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0

#include <deal.II/fe/fe_dgq.h>

#include <gtest/gtest.h>
#include <immersx/core/domain.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/weak_term.h>

#include <cmath>
#include <memory>

using namespace dealii;
using namespace ImmersX;

namespace
{
  void
  check_lifted_radius(const bool single_cell)
  {
    ParameterAcceptor::clear();
    DomainParameters<1, 3> parameters("/Tests/Lifted radius/");
    parameters.triangulation_type = "fullydistributed";
    parameters.name_of_grid       = "hyper_cube";
    parameters.arguments_for_grid = "-.5:.5:false";
    parameters.initial_refinement = single_cell ? 0 : 2;
    Domain<1, 3> domain(parameters);
    domain.make_grid();
    if (single_cell)
      {
        EXPECT_EQ(Utilities::MPI::min(
                    domain.triangulation().n_locally_owned_active_cells(),
                    MPI_COMM_WORLD),
                  0u);
      }

    FE_DGQ<1, 3>     fe(1);
    DoFHandler<1, 3> dofs(domain.triangulation());
    dofs.distribute_dofs(fe);
    AffineConstraints<double> constraints;
    constraints.close();
    const auto space =
      fe_space(dofs, StaticMappingQ1<1, 3>::mapping, constraints);
    StateLayout                layout;
    const auto                 area   = space.field(layout, "area");
    const auto                 target = space.field(layout, "radius");
    TensorProductLift<1, 2, 3> geometry("/Tests/Radius lift/");
    geometry.section.inclusion_degree      = 0;
    geometry.section.inclusion_type        = "hyper_cube";
    geometry.section.refinement_level      = 0;
    geometry.section.selected_coefficients = {0};
    geometry.section.n_q_points            = 2;
    geometry.representative_n_q_points     = 3;
    geometry.thickness                     = "0.25";
    const auto radius = lift(sqrt(value(area) / numbers::PI), geometry);
    if (single_cell)
      {
        EXPECT_EQ(Utilities::MPI::min(static_cast<unsigned int>(
                                        radius.lifted_points().size()),
                                      MPI_COMM_WORLD),
                  0u);
      }

    using V     = ImmersXLA::MPI::Vector;
    using M     = ImmersXLA::MPI::SparseMatrix;
    using Model = SemiDiscreteModel<V, M>;
    Model                     model;
    SemidiscreteBuilder<V, M> builder(layout, model);
    weak_term(radius, test(target))
      .on(IntegrationSupport<1, 3>(domain.triangulation()))
      .add(builder);
    V state(area.locally_owned_dofs(), MPI_COMM_WORLD);
    V direction(area.locally_owned_dofs(), MPI_COMM_WORLD);
    V test_vector(target.locally_owned_dofs(), MPI_COMM_WORLD);
    for (const auto i : area.locally_owned_dofs())
      {
        state[i]       = 2. + .1 * i;
        direction[i]   = .3 + .07 * i;
        test_vector[i] = 1. - .04 * i;
      }
    state.compress(VectorOperation::insert);
    direction.compress(VectorOperation::insert);
    test_vector.compress(VectorOperation::insert);

    const auto view_for = [&](const V &candidate) {
      StateView<V> view(layout, 0.);
      view.bind(area.id(), candidate);
      return view;
    };
    const auto residual_for = [&](const V &candidate) {
      V          result(target.locally_owned_dofs(), MPI_COMM_WORLD);
      const auto view = view_for(candidate);
      const EvaluationContext<V> context(0., view);
      model.evaluate_row(target.id(), context, result);
      return result;
    };
    std::shared_ptr<M> first_matrix;
    V first_action(target.locally_owned_dofs(), MPI_COMM_WORLD);
    for (const double shift : {0., 1.})
      {
        V candidate(state);
        candidate.add(shift);
        const auto                 view = view_for(candidate);
        const EvaluationContext<V> context(0., view);
        const auto                 op =
          model.state_matrix_operator(target.id(), area.id(), context);
        ASSERT_TRUE(op.has_value());
        const auto matrix = op->matrix();
        V          action(target.locally_owned_dofs(), MPI_COMM_WORLD);
        matrix->vmult(action, direction);
        V expected(target.locally_owned_dofs(), MPI_COMM_WORLD);
        V exact_residual(target.locally_owned_dofs(), MPI_COMM_WORLD);
        expected       = 0.;
        exact_residual = 0.;
        // Independent quadrature oracle visits the stored samples directly.
        for (const auto &point : radius.lifted_points())
          {
            double a = 0., da = 0.;
            for (unsigned int j = 0; j < point.dof_indices.size(); ++j)
              {
                a += point.source_basis_values[j] *
                     candidate[point.dof_indices[j]];
                da += point.source_basis_values[j] *
                      direction[point.dof_indices[j]];
              }
            for (unsigned int i = 0; i < point.dof_indices.size(); ++i)
              {
                const double weight =
                  point.weight * point.source_basis_values[i];
                exact_residual[point.dof_indices[i]] +=
                  weight * std::sqrt(a / numbers::PI);
                expected[point.dof_indices[i]] +=
                  weight * da / (2. * std::sqrt(numbers::PI * a));
              }
          }
        expected.compress(VectorOperation::add);
        exact_residual.compress(VectorOperation::add);
        V error(action);
        error -= expected;
        EXPECT_LT(error.l2_norm(), 1.e-12);
        error = residual_for(candidate);
        error -= exact_residual;
        EXPECT_LT(error.l2_norm(), 1.e-12);

        const double epsilon = 1.e-6;
        V            plus(candidate), minus(candidate);
        plus.add(epsilon, direction);
        minus.add(-epsilon, direction);
        V difference = residual_for(plus);
        difference -= residual_for(minus);
        difference *= .5 / epsilon;
        difference -= action;
        EXPECT_LT(difference.l2_norm(), 1.e-9);

        V transpose(area.locally_owned_dofs(), MPI_COMM_WORLD);
        matrix->Tvmult(transpose, test_vector);
        EXPECT_NEAR(test_vector * action, transpose * direction, 1.e-12);
        if (shift == 0.)
          {
            first_matrix = matrix;
            first_action = action;
          }
      }
    V retained(target.locally_owned_dofs(), MPI_COMM_WORLD);
    first_matrix->vmult(retained, direction);
    retained -= first_action;
    EXPECT_LT(retained.l2_norm(), 1.e-12);
  }
} // namespace

TEST(WeakTerm, BOTH_LiftedRadiusJacobianMatchesQuadratureAndFiniteDifference)
{
  check_lifted_radius(false);
}

TEST(WeakTerm, MPI_LiftedRadiusJacobianWithEmptyOwner)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  check_lifted_radius(true);
}
