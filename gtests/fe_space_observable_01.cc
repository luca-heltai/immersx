// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/core/observable.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/state.h>
#include <immersx/core/symbolic_expression_kernel.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>

#include <type_traits>

namespace
{
  using namespace dealii;

  struct ExternalFESystem
  {
    Triangulation<2>          triangulation;
    FESystem<2>               finite_element{FE_Q<2>(1), 2};
    DoFHandler<2>             dof_handler{triangulation};
    MappingQ1<2>              mapping;
    AffineConstraints<double> constraints;
    IndexSet                  relevant;

    ExternalFESystem()
    {
      GridGenerator::hyper_cube(triangulation);
      dof_handler.distribute_dofs(finite_element);
      relevant = dof_handler.locally_owned_dofs();
    }
  };
} // namespace

TEST(FESpace, IsANonOwningViewAndSupportsSubspaces)
{
  ExternalFESystem     system;
  ImmersX::StateLayout layout;
  const auto           V = ImmersX::fe_space(system.dof_handler,
                                   system.mapping,
                                   system.constraints,
                                   system.relevant);

  const auto displacement =
    V.field(layout, "displacement", FEValuesExtractors::Vector(0));
  const auto pressure =
    V.field(layout, "pressure", FEValuesExtractors::Scalar(0));
  const auto displacement_again =
    V[FEValuesExtractors::Vector(0)].field("displacement-preview");

  EXPECT_EQ(&V.dof_handler(), &system.dof_handler);
  EXPECT_EQ(&V.mapping(), &system.mapping);
  EXPECT_EQ(&V.constraints(), &system.constraints);
  EXPECT_EQ(&V.locally_relevant_dofs(), &system.relevant);
  EXPECT_EQ(displacement.field_id(), ImmersX::FieldId(0));
  EXPECT_EQ(pressure.field_id(), ImmersX::FieldId(1));
  EXPECT_NE(displacement.field_id(), pressure.field_id());
  EXPECT_FALSE(displacement_again.is_registered());
  EXPECT_EQ(&displacement.dof_handler(), &system.dof_handler);
  EXPECT_EQ(displacement.extractor().first_vector_component, 0u);
}

TEST(FESpace, FieldLocalConstraintsArePersistentAndIndependent)
{
  ExternalFESystem system;
  const auto       V = ImmersX::fe_space(system.dof_handler,
                                   system.mapping,
                                   system.constraints,
                                   system.relevant);

  const auto displacement =
    V.field("displacement", FEValuesExtractors::Vector(0)).with_constraints();
  const auto velocity =
    V.field("velocity", FEValuesExtractors::Vector(0)).with_constraints();

  ASSERT_TRUE(displacement.has_field_constraints());
  ASSERT_TRUE(velocity.has_field_constraints());
  EXPECT_NE(&displacement.constraints(), &system.constraints);
  EXPECT_NE(&velocity.constraints(), &system.constraints);
  EXPECT_NE(&displacement.constraints(), &velocity.constraints());

  auto *const displacement_constraints = &displacement.mutable_constraints();
  auto *const velocity_constraints     = &velocity.mutable_constraints();
  displacement_constraints->clear();
  velocity_constraints->clear();
  displacement_constraints->add_line(0);
  displacement_constraints->close();
  velocity_constraints->add_line(1);
  velocity_constraints->close();

  EXPECT_TRUE(displacement.constraints().is_constrained(0));
  EXPECT_FALSE(displacement.constraints().is_constrained(1));
  EXPECT_FALSE(velocity.constraints().is_constrained(0));
  EXPECT_TRUE(velocity.constraints().is_constrained(1));

  displacement.mutable_constraints().clear();
  displacement.mutable_constraints().close();
  EXPECT_EQ(displacement_constraints, &displacement.constraints());
  EXPECT_FALSE(displacement.constraints().is_constrained(0));
  EXPECT_TRUE(velocity.constraints().is_constrained(1));
}

TEST(FESpace, FieldComponentMaskComesFromNonzeroExtractor)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  FESystem<2>   fe(FE_Q<2>(1), 1, FE_Q<2>(1), 2);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();

  const auto V =
    ImmersX::fe_space(dof_handler, StaticMappingQ1<2>::mapping, constraints);
  const auto vector = V.field("vector", FEValuesExtractors::Vector(1));
  const auto mask   = vector.component_mask();

  ASSERT_EQ(mask.size(), 3u);
  EXPECT_FALSE(mask[0]);
  EXPECT_TRUE(mask[1]);
  EXPECT_TRUE(mask[2]);
}

TEST(FESpace, ScalarAndVectorFieldCPOsUseDealIIExtractors)
{
  ExternalFESystem     system;
  ImmersX::StateLayout layout;
  const auto           V = ImmersX::fe_space(system.dof_handler,
                                   system.mapping,
                                   system.constraints,
                                   system.relevant);

  const auto scalar = ImmersX::scalar_field(V, "temperature");
  const auto vector = ImmersX::vector_field(V, "velocity");
  const auto registered_scalar =
    ImmersX::scalar_field(V, layout, "registered-temperature");
  const auto registered_vector =
    ImmersX::vector_field(V, layout, "registered-velocity");

  using ScalarField = ImmersX::Field<2, 2, dealii::FEValuesExtractors::Scalar>;
  using VectorField = ImmersX::Field<2, 2, dealii::FEValuesExtractors::Vector>;
  static_assert(
    std::is_same_v<std::remove_cv_t<decltype(scalar)>, ScalarField>);
  static_assert(
    std::is_same_v<std::remove_cv_t<decltype(vector)>, VectorField>);

  EXPECT_EQ(scalar.name(), "temperature");
  EXPECT_EQ(scalar.extractor().component, 0u);
  EXPECT_EQ(vector.name(), "velocity");
  EXPECT_EQ(vector.extractor().first_vector_component, 0u);
  EXPECT_FALSE(scalar.is_registered());
  EXPECT_FALSE(vector.is_registered());
  EXPECT_TRUE(registered_scalar.is_registered());
  EXPECT_TRUE(registered_vector.is_registered());
  EXPECT_NE(registered_scalar.field_id(), registered_vector.field_id());
}

TEST(FESpace, ValueAndGradientExposeTypedDependencies)
{
  ExternalFESystem     system;
  ImmersX::StateLayout layout;
  const auto           V = ImmersX::fe_space(system.dof_handler,
                                   system.mapping,
                                   system.constraints,
                                   system.relevant);
  const auto           displacement =
    V.field(layout, "displacement", FEValuesExtractors::Vector(0));
  const auto pressure =
    V.field(layout, "pressure", FEValuesExtractors::Scalar(0));

  const auto displacement_value = ImmersX::value(displacement);
  const auto pressure_value     = ImmersX::value(pressure);
  const auto displacement_grad  = ImmersX::gradient(displacement);
  const auto pressure_grad      = ImmersX::gradient(pressure);
  const auto displacement_div   = ImmersX::divergence(displacement);
  const auto displacement_sym   = ImmersX::symmetric_gradient(displacement);
  const auto displacement_curl  = ImmersX::curl(displacement);

  using DisplacementView =
    FEValuesViews::View<2, 2, FEValuesExtractors::Vector>;
  using PressureView = FEValuesViews::View<2, 2, FEValuesExtractors::Scalar>;
  static_assert(
    std::is_same_v<typename DisplacementView::value_type,
                   typename decltype(displacement_value)::value_type>);
  static_assert(std::is_same_v<typename PressureView::value_type,
                               typename decltype(pressure_value)::value_type>);
  static_assert(
    std::is_same_v<typename DisplacementView::gradient_type,
                   typename decltype(displacement_grad)::value_type>);
  static_assert(std::is_same_v<typename PressureView::gradient_type,
                               typename decltype(pressure_grad)::value_type>);
  static_assert(
    std::is_same_v<typename DisplacementView::divergence_type,
                   typename decltype(displacement_div)::value_type>);
  static_assert(
    std::is_same_v<typename DisplacementView::symmetric_gradient_type,
                   typename decltype(displacement_sym)::value_type>);
  static_assert(
    std::is_same_v<typename DisplacementView::curl_type,
                   typename decltype(displacement_curl)::value_type>);

  EXPECT_EQ(displacement_value.dependencies(),
            std::vector<ImmersX::FieldId>{displacement.field_id()});
  EXPECT_EQ(pressure_grad.dependencies(),
            std::vector<ImmersX::FieldId>{pressure.field_id()});
  EXPECT_EQ(displacement_value.update_flags() & update_values, update_values);
  EXPECT_EQ(pressure_grad.update_flags() & update_gradients, update_gradients);
  EXPECT_EQ(displacement_value.source_field(), displacement.field_id());
}

TEST(FESpace, TestExpressionsAndNativeAlgebra)
{
  ExternalFESystem     system;
  ImmersX::StateLayout layout;
  const auto           V = ImmersX::fe_space(system.dof_handler,
                                   system.mapping,
                                   system.constraints,
                                   system.relevant);
  const auto           displacement =
    V.field(layout, "displacement", FEValuesExtractors::Vector(0));

  const auto v          = ImmersX::test(displacement);
  const auto grad_v     = ImmersX::gradient(v);
  const auto value_u    = ImmersX::value(displacement);
  const auto sum        = value_u + value_u;
  const auto difference = value_u - value_u;
  const auto convection = ImmersX::gradient(displacement) * value_u;

  EXPECT_TRUE(v.dependencies().empty());
  EXPECT_TRUE(grad_v.dependencies().empty());
  EXPECT_EQ(v.source_field(), displacement.field_id());
  static_assert(
    std::is_same_v<typename decltype(grad_v)::value_type,
                   typename FEValuesViews::
                     View<2, 2, FEValuesExtractors::Vector>::gradient_type>);
  static_assert(std::is_same_v<typename decltype(sum)::value_type,
                               typename decltype(value_u)::value_type>);
  static_assert(std::is_same_v<typename decltype(difference)::value_type,
                               typename decltype(value_u)::value_type>);
  static_assert(
    std::is_same_v<typename decltype(convection)::value_type, Tensor<1, 2>>);
}

TEST(FESpace, TensorExtractorsUseDealIIViewTypes)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  FESystem<2>   fe(FE_Q<2>(1), 4);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();
  const auto V =
    ImmersX::fe_space(dof_handler, StaticMappingQ1<2>::mapping, constraints);

  const auto tensor = V[FEValuesExtractors::Tensor<2>(0)].field("tensor");
  const auto symmetric_tensor =
    V[FEValuesExtractors::SymmetricTensor<2>(0)].field("symmetric-tensor");
  const auto tensor_value         = ImmersX::value(tensor);
  const auto tensor_gradient      = ImmersX::gradient(tensor);
  const auto tensor_divergence    = ImmersX::divergence(tensor);
  const auto symmetric_value      = ImmersX::value(symmetric_tensor);
  const auto symmetric_divergence = ImmersX::divergence(symmetric_tensor);

  using TensorView = FEValuesViews::View<2, 2, FEValuesExtractors::Tensor<2>>;
  using SymmetricView =
    FEValuesViews::View<2, 2, FEValuesExtractors::SymmetricTensor<2>>;
  static_assert(std::is_same_v<typename TensorView::value_type,
                               typename decltype(tensor_value)::value_type>);
  static_assert(std::is_same_v<typename TensorView::gradient_type,
                               typename decltype(tensor_gradient)::value_type>);
  static_assert(
    std::is_same_v<typename TensorView::divergence_type,
                   typename decltype(tensor_divergence)::value_type>);
  static_assert(std::is_same_v<typename SymmetricView::value_type,
                               typename decltype(symmetric_value)::value_type>);
  static_assert(
    std::is_same_v<typename SymmetricView::divergence_type,
                   typename decltype(symmetric_divergence)::value_type>);
}

TEST(FESpace, FrozenObservablesReuseFEOperations)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  FE_Q<2>       fe(1);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();
  const auto V =
    ImmersX::fe_space(dof_handler, StaticMappingQ1<2>::mapping, constraints);
  const auto     field = V.field("frozen");
  Vector<double> coefficients(dof_handler.n_dofs());
  const auto     frozen_value = ImmersX::frozen(field, coefficients);
  const auto     frozen_grad  = ImmersX::gradient(frozen_value);

  EXPECT_TRUE(frozen_value.is_frozen());
  EXPECT_TRUE(frozen_grad.is_frozen());
  EXPECT_TRUE(frozen_value.dependencies().empty());
  EXPECT_TRUE(frozen_grad.dependencies().empty());
  EXPECT_EQ(frozen_grad.update_flags() & update_gradients, update_gradients);
}

TEST(FESpace, NonlinearTransformChainsActiveAndFrozenInputs)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  FE_Q<2>       fe(1);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();
  const auto V =
    ImmersX::fe_space(dof_handler, StaticMappingQ1<2>::mapping, constraints);
  const auto     field = V.field("active");
  Vector<double> coefficients(dof_handler.n_dofs());
  const auto     active = ImmersX::value(field);
  const auto     fixed  = ImmersX::frozen(field, coefficients);

  struct AffineKernel
  {
    struct Evaluation
    {
      double              value;
      std::vector<double> derivatives;
    };

    Evaluation
    evaluate(const std::vector<double> &values) const
    {
      return {values[0] + 3. * values[1], {1., 3.}};
    }
  };

  auto       expression = ImmersX::transform(active, fixed, AffineKernel{});
  const auto point      = Point<2>(0.25, 0.5);
  const auto value      = expression.evaluate_point(
    point, 0., [](const auto &, const auto &, const double) { return 2.; });
  EXPECT_DOUBLE_EQ(value, 8.);
  EXPECT_EQ(expression.dependencies(),
            std::vector<ImmersX::FieldId>{field.field_id()});
  EXPECT_DOUBLE_EQ(
    expression.linearize_point(
      field.field_id(),
      point,
      0.,
      [](const auto &, const auto &, const double) { return 0.; },
      [](const auto &, const auto, const auto &, const double) { return 1.; }),
    1.);
}

TEST(FESpace, SymbolicKernelSuppliesTransformDerivatives)
{
  if (!ImmersX::SymbolicExpressionKernel::available())
    GTEST_SKIP() << "deal.II was built without SymEngine";

  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  FE_Q<2>       fe(1);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();
  const auto V =
    ImmersX::fe_space(dof_handler, StaticMappingQ1<2>::mapping, constraints);
  const auto     field = V.field("active");
  Vector<double> coefficients(dof_handler.n_dofs());
  auto           kernel = ImmersX::SymbolicExpressionKernel();
  kernel.initialize("u + 2*v", {"u", "v"});
  const auto expression =
    ImmersX::transform(ImmersX::value(field),
                       ImmersX::frozen(field, coefficients),
                       std::move(kernel));
  const auto point = Point<2>(0.25, 0.5);
  EXPECT_DOUBLE_EQ(expression.evaluate_point(
                     point,
                     0.,
                     [](const auto &input, const auto &, const double) {
                       return input.is_frozen() ? 3. : 4.;
                     }),
                   10.);
  EXPECT_DOUBLE_EQ(
    expression.linearize_point(
      field.field_id(),
      point,
      0.,
      [](const auto &, const auto &, const double) { return 0.; },
      [](const auto &input, const auto, const auto &, const double) {
        return input.is_frozen() ? 0. : 1.;
      }),
    1.);
}

TEST(FESpace, NonlinearLiftUsesComposedSource)
{
  struct AffineKernel
  {
    struct Evaluation
    {
      double              value;
      std::vector<double> derivatives;
    };

    Evaluation
    evaluate(const std::vector<double> &values) const
    {
      return {values[0] + values[1], {1., 1.}};
    }
  };

  Triangulation<1, 2> tria;
  GridGenerator::hyper_cube(tria);
  FE_Q<1, 2>       fe(1);
  DoFHandler<1, 2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();
  const auto V =
    ImmersX::fe_space(dof_handler, StaticMappingQ1<1, 2>::mapping, constraints);
  ImmersX::StateLayout layout;
  const auto           source = V.field(layout, "source");
  using State                 = ImmersX::ImmersXLA::MPI::Vector;
  State frozen_values;
  frozen_values.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
  const auto expression =
    ImmersX::transform(ImmersX::value(source),
                       ImmersX::frozen(source, frozen_values),
                       AffineKernel{});
  ImmersX::TensorProductLift<1, 2, 2, 1> descriptor;
  descriptor.section.selected_coefficients = {0u};
  descriptor.section.inclusion_degree      = 1;
  descriptor.section.refinement_level      = 0;
  descriptor.section.n_q_points            = 2;
  descriptor.representative_n_q_points     = 2;
  const auto lifted = ImmersX::lift(expression, descriptor);

  State state;
  state.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
  state = 1.;
  ImmersX::StateView<State> state_view(layout, 0.);
  state_view.bind(source.field_id(), state);
  const ImmersX::EvaluationContext<State> context(0., state_view);
  const auto                              values = lifted.evaluate(context);
  EXPECT_EQ(values.size(), lifted.lifted_points().size());
  const auto linearization = lifted.linearize(context, source.field_id());
  State      direction;
  direction.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
  direction = 1.;
  ImmersX::ImmersXLA::MPI::Vector action;
  linearization.vmult(action, direction);
  EXPECT_EQ(action.size(), values.size());
}

TEST(FESpace, WrapsAnExistingProblemFromTheOutside)
{
  ParameterAcceptor::clear();
  ImmersX::PoissonParameters<2> parameters;
  parameters.domain_parameters.initial_refinement = 0;
  parameters.n_refinement_cycles                  = 1;
  ImmersX::PoissonSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();

  const auto V        = ImmersX::finite_element_space_view(problem);
  const auto pressure = V.field("pressure");

  EXPECT_EQ(&pressure.dof_handler(), &problem.dof_handler());
  EXPECT_EQ(&V.mapping(), &StaticMappingQ1<2>::mapping);
  EXPECT_EQ(pressure.name(), "pressure");
  EXPECT_FALSE(pressure.is_registered());
}

TEST(FESpace, ParametersSelectFiniteElementByName)
{
  ParameterAcceptor::clear();
  ImmersX::FiniteElementSpaceParameters<2> parameters("/Finite element space/");
  ImmersX::initialize_parameters_from_string(R"(
    subsection Finite element space
      set Finite element = FE_DGQ<2>(2)
    end
  )");

  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  ImmersX::FiniteElementSpace<2> space(triangulation, parameters);

  EXPECT_NE(dynamic_cast<const FE_DGQ<2> *>(&space.finite_element()), nullptr);
  EXPECT_EQ(space.dof_handler().n_dofs(), 36u);
  const auto view = space.view();
  EXPECT_EQ(&view.dof_handler(), &space.dof_handler());
  EXPECT_EQ(&view.mapping(), &StaticMappingQ1<2>::mapping);
}
