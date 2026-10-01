// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/core/boundary_conditions.h>
#include <immersx/core/contributor.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/observable.h>
#include <immersx/core/state.h>
#include <immersx/core/weak_term.h>

#include <algorithm>
#include <limits>

using namespace dealii;
using namespace ImmersX;

namespace
{
  class TimeFunction : public Function<2>
  {
  public:
    TimeFunction()
      : Function<2>(1)
    {}

    void
    set_time(const double time) override
    {
      time_ = time;
    }

    double
    value(const Point<2> &, const unsigned int = 0) const override
    {
      return time_;
    }

  private:
    double time_ = 0.;
  };

  class VectorFunction : public Function<2>
  {
  public:
    VectorFunction()
      : Function<2>(2)
    {}

    double
    value(const Point<2> &, const unsigned int component = 0) const override
    {
      return component == 0 ? 1. : 2.;
    }
  };

  using LocalVector = Vector<double>;
  using LocalMatrix = SparseMatrix<double>;
} // namespace

TEST(BoundaryConditions, AggregatesRulesAndUpdatesStableConstraints)
{
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  FiniteElementSpaceParameters<2> parameters;
  parameters.finite_element = "FE_Q<2>(1)";
  FiniteElementSpace<2> space(triangulation, parameters);

  Functions::ConstantFunction<2> one(1.);
  BoundaryConditions<2>          aggregated(space);
  aggregated.add_dirichlet(0, one);
  aggregated.add_dirichlet(1, one);
  aggregated.update(0.);
  EXPECT_EQ(aggregated.n_dirichlet_rules(), 2u);
  EXPECT_GT(space.constraints().n_constraints(), 0u);

  parallel::distributed::Triangulation<2> time_triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(time_triangulation);
  time_triangulation.refine_global(1);
  FiniteElementSpace<2> time_space(time_triangulation, parameters);
  const auto           *constraints_address = &time_space.constraints();
  TimeFunction          time_function;
  BoundaryConditions<2> time_dependent(time_space);
  time_dependent.add_dirichlet(0, time_function);
  time_dependent.update(1.);
  double minimum_at_one = std::numeric_limits<double>::max();
  for (const auto &line : time_space.constraints().get_lines())
    minimum_at_one =
      std::min(minimum_at_one,
               time_space.constraints().get_inhomogeneity(line.index));
  time_dependent.update(2.);
  double minimum_at_two = std::numeric_limits<double>::max();
  for (const auto &line : time_space.constraints().get_lines())
    minimum_at_two =
      std::min(minimum_at_two,
               time_space.constraints().get_inhomogeneity(line.index));

  EXPECT_EQ(&time_space.constraints(), constraints_address);
  EXPECT_NE(minimum_at_one, minimum_at_two);
  EXPECT_DOUBLE_EQ(minimum_at_one, 1.);
  EXPECT_DOUBLE_EQ(minimum_at_two, 2.);
  EXPECT_EQ(time_dependent.value_revision(), 2u);
  EXPECT_EQ(time_dependent.structure_revision(), 1u);
}

TEST(BoundaryConditions, VectorMaskAndNormalFlux)
{
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  auto finite_element = std::make_unique<FESystem<2>>(FE_Q<2>(1), 2);
  FiniteElementSpace<2> space(triangulation, std::move(finite_element));

  VectorFunction        function;
  BoundaryConditions<2> conditions(space);
  conditions.add_dirichlet(0,
                           function,
                           ComponentMask(std::vector<bool>{true, false}));
  conditions.add_nonzero_normal_flux(2, function);
  conditions.update(0.);

  EXPECT_EQ(conditions.n_dirichlet_rules(), 1u);
  EXPECT_EQ(conditions.n_normal_flux_rules(), 1u);
  EXPECT_GT(space.constraints().n_constraints(), 0u);
}

TEST(BoundaryConditions, WeakAffineContributionFollowsEvaluationTime)
{
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  FiniteElementSpaceParameters<2> parameters;
  parameters.finite_element = "FE_Q<2>(1)";
  FiniteElementSpace<2> space(triangulation, parameters);

  TimeFunction          time_function;
  BoundaryConditions<2> conditions(space);
  conditions.add_dirichlet(0, time_function);
  conditions.update(0.);

  StateLayout                                   layout;
  const auto                                    V = space.view();
  const auto                                    u = V.field(layout, "u");
  SemiDiscreteModel<LocalVector, LocalMatrix>   model;
  SemidiscreteBuilder<LocalVector, LocalMatrix> builder(layout, model);
  weak_term(gradient(u), gradient(test(u)))
    .with_context_update(
      [&conditions](const double time) { conditions.update(time); })
    .add(builder);

  LocalVector state(u.dof_handler().n_dofs());
  state = 0.;
  StateView<LocalVector> state_view_at_zero(layout, 0.);
  state_view_at_zero.bind(u.field_id(), state);
  StateView<LocalVector> state_view_at_one(layout, 1.);
  state_view_at_one.bind(u.field_id(), state);
  const EvaluationContext<LocalVector> context_at_zero(0., state_view_at_zero);
  const EvaluationContext<LocalVector> context_at_one(1., state_view_at_one);
  LocalVector                          residual_at_zero(state.size());
  LocalVector                          residual_at_one(state.size());
  model.update_context(context_at_zero);
  model.evaluate_row(u.field_id(), context_at_zero, residual_at_zero);
  model.update_context(context_at_one);
  model.evaluate_row(u.field_id(), context_at_one, residual_at_one);

  EXPECT_LT(residual_at_zero.l2_norm(), 1.e-12);
  EXPECT_GT(residual_at_one.l2_norm(), 0.);
}
