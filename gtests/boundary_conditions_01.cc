// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>

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
#include <immersx/physics/modulated_parsed_function.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

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

  class ThreeComponentFunction : public Function<2>
  {
  public:
    ThreeComponentFunction()
      : Function<2>(3)
    {}

    double
    value(const Point<2> &, const unsigned int component = 0) const override
    {
      return 1. + component;
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
  auto copied = aggregated;
  auto moved  = std::move(copied);
  moved.update(0.);
  EXPECT_EQ(moved.n_dirichlet_rules(), 2u);

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

TEST(BoundaryConditions, OwnedFunctionSurvivesSourceScope)
{
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  FiniteElementSpaceParameters<2> parameters;
  parameters.finite_element = "FE_Q<2>(1)";
  FiniteElementSpace<2> space(triangulation, parameters);

  BoundaryConditions<2> conditions(space);
  {
    const auto function =
      std::make_shared<const Functions::ConstantFunction<2>>(3.);
    conditions.add_dirichlet(0, function);
  }

  conditions.update(0.);
  ASSERT_GT(space.constraints().n_constraints(), 0u);
  for (const auto &line : space.constraints().get_lines())
    EXPECT_DOUBLE_EQ(space.constraints().get_inhomogeneity(line.index), 3.);
}

TEST(BoundaryConditions, OwnedParsedFunctionsSurviveBoundaryCopies)
{
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  FiniteElementSpaceParameters<2> parameters;
  parameters.finite_element = "FE_Q<2>(1)";
  FiniteElementSpace<2> space(triangulation, parameters);

  using FunctionHandle = std::shared_ptr<const Function<2>>;
  FunctionHandle expression;
  FunctionHandle parameterized;
  {
    expression          = std::make_shared<const FunctionParser<2>>("x + t");
    const auto function = std::make_shared<ModulatedParsedFunction<2>>(
      "/Boundary condition parameterized function/");
    parameterized = function->function_handle();
  }

  BoundaryConditions<2> conditions(space);
  conditions.add_dirichlet(0, expression);
  conditions.add_dirichlet(1, parameterized);
  const auto copied_conditions = conditions;

  conditions.update(0.5);
  copied_conditions.update(0.5);
  EXPECT_GT(space.constraints().n_constraints(), 0u);
  EXPECT_EQ(conditions.n_dirichlet_rules(), 2u);
  EXPECT_EQ(copied_conditions.n_dirichlet_rules(), 2u);
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

TEST(BoundaryConditions, FieldConstraintsAreIndependentAndTimeAware)
{
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  FiniteElementSpaceParameters<2> parameters;
  parameters.finite_element = "FE_Q<2>(1)";
  FiniteElementSpace<2> space(triangulation, parameters);
  const auto            V = space.view();

  const auto            displacement = V.field("displacement");
  const auto            velocity     = V.field("velocity");
  TimeFunction          displacement_value;
  TimeFunction          velocity_value;
  BoundaryConditions<2> displacement_conditions(displacement);
  BoundaryConditions<2> velocity_conditions(velocity);
  displacement_conditions.add_dirichlet(0, displacement_value);
  velocity_conditions.add_dirichlet(0, velocity_value);

  const auto *displacement_address = &displacement.constraints();
  const auto *velocity_address     = &velocity.constraints();
  EXPECT_NE(displacement_address, &space.constraints());
  EXPECT_NE(velocity_address, &space.constraints());
  EXPECT_NE(displacement_address, velocity_address);

  const auto minimum_inhomogeneity = [](const auto &constraints) {
    double result = std::numeric_limits<double>::max();
    for (const auto &line : constraints.get_lines())
      result = std::min(result, constraints.get_inhomogeneity(line.index));
    return result;
  };

  displacement_conditions.update(1.);
  velocity_conditions.update(2.);
  EXPECT_DOUBLE_EQ(minimum_inhomogeneity(displacement.constraints()), 1.);
  EXPECT_DOUBLE_EQ(minimum_inhomogeneity(velocity.constraints()), 2.);
  EXPECT_EQ(&displacement.constraints(), displacement_address);
  EXPECT_EQ(&velocity.constraints(), velocity_address);

  velocity_conditions.update(4.);
  displacement_conditions.update(3.);
  EXPECT_DOUBLE_EQ(minimum_inhomogeneity(displacement.constraints()), 3.);
  EXPECT_DOUBLE_EQ(minimum_inhomogeneity(velocity.constraints()), 4.);
  EXPECT_EQ(space.constraints().n_constraints(), 0u);
}

TEST(BoundaryConditions, FieldConstraintsIncludeHangingNodes)
{
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  triangulation.begin_active()->set_refine_flag();
  triangulation.execute_coarsening_and_refinement();

  FiniteElementSpaceParameters<2> parameters;
  parameters.finite_element = "FE_Q<2>(1)";
  FiniteElementSpace<2> space(triangulation, parameters);
  const auto            V            = space.view();
  const auto            with_hanging = V.field("with-hanging");
  const auto            without      = V.field("without-hanging");

  BoundaryConditions<2> with_hanging_conditions(with_hanging);
  with_hanging_conditions.update(0.);
  BoundaryConditions<2> without_conditions(without);
  without_conditions.include_hanging_node_constraints(false);
  without_conditions.update(0.);

  EXPECT_GT(with_hanging.constraints().n_constraints(), 0u);
  EXPECT_EQ(without.constraints().n_constraints(), 0u);
}

TEST(BoundaryConditions, FieldMaskSubmaskAndAutomaticNormalFlux)
{
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  auto finite_element =
    std::make_unique<FESystem<2>>(FE_Q<2>(1), 1, FE_Q<2>(1), 2);
  FiniteElementSpace<2>  space(triangulation, std::move(finite_element));
  const auto             V = space.view();
  ThreeComponentFunction function;
  VectorFunction         flux_function;

  const auto vector = V.field("vector", FEValuesExtractors::Vector(1));
  BoundaryConditions<2> vector_conditions(vector);
  vector_conditions.include_hanging_node_constraints(false);
  vector_conditions.add_dirichlet(0, function);
  vector_conditions.update(0.);
  const auto vector_dofs =
    DoFTools::extract_dofs(vector.dof_handler(), vector.component_mask());
  ASSERT_GT(vector.constraints().n_constraints(), 0u);
  for (const auto &line : vector.constraints().get_lines())
    EXPECT_TRUE(vector_dofs.is_element(line.index));

  const auto subvector = V.field("subvector", FEValuesExtractors::Vector(1));
  BoundaryConditions<2> subvector_conditions(subvector);
  subvector_conditions.include_hanging_node_constraints(false);
  const ComponentMask submask(std::vector<bool>{false, true, false});
  subvector_conditions.add_dirichlet(0, function, submask);
  subvector_conditions.update(0.);
  const auto subvector_dofs =
    DoFTools::extract_dofs(subvector.dof_handler(),
                           ComponentMask(
                             std::vector<bool>{false, true, false}));
  ASSERT_GT(subvector.constraints().n_constraints(), 0u);
  for (const auto &line : subvector.constraints().get_lines())
    EXPECT_TRUE(subvector_dofs.is_element(line.index));
  EXPECT_THROW(subvector_conditions.add_dirichlet(
                 0,
                 function,
                 ComponentMask(std::vector<bool>{true, false, false})),
               ExceptionBase);

  const auto            flux = V.field("flux", FEValuesExtractors::Vector(1));
  BoundaryConditions<2> flux_conditions(flux);
  flux_conditions.include_hanging_node_constraints(false);
  flux_conditions.add_nonzero_normal_flux(0, flux_function);
  flux_conditions.update(0.);

  const auto explicit_flux =
    V.field("explicit-flux", FEValuesExtractors::Vector(1));
  BoundaryConditions<2> explicit_flux_conditions(explicit_flux);
  explicit_flux_conditions.include_hanging_node_constraints(false);
  explicit_flux_conditions.add_nonzero_normal_flux(0, flux_function, 1);
  explicit_flux_conditions.update(0.);

  ASSERT_GT(flux.constraints().n_constraints(), 0u);
  ASSERT_EQ(flux.constraints().n_constraints(),
            explicit_flux.constraints().n_constraints());
  for (unsigned int i = 0; i < flux.constraints().n_constraints(); ++i)
    {
      const auto &automatic_line = flux.constraints().get_lines()[i];
      const auto &explicit_line  = explicit_flux.constraints().get_lines()[i];
      EXPECT_EQ(automatic_line.index, explicit_line.index);
      EXPECT_DOUBLE_EQ(
        flux.constraints().get_inhomogeneity(automatic_line.index),
        explicit_flux.constraints().get_inhomogeneity(explicit_line.index));
    }
}

TEST(BoundaryConditions, WeakAffineContributionFollowsEvaluationTime)
{
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  FiniteElementSpaceParameters<2> parameters;
  parameters.finite_element = "FE_Q<2>(1)";
  FiniteElementSpace<2> space(triangulation, parameters);

  const auto            V = space.view();
  StateLayout           layout;
  const auto            u = V.field(layout, "u");
  TimeFunction          time_function;
  BoundaryConditions<2> conditions(u);
  conditions.add_dirichlet(0, time_function);
  conditions.update(0.);

  SemiDiscreteModel<LocalVector, LocalMatrix>   model;
  SemidiscreteBuilder<LocalVector, LocalMatrix> builder(layout, model);
  conditions.register_with(builder);
  weak_term(gradient(u), gradient(test(u))).add(builder);

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
