// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/algebra/metric_flow_x_vessel_wall_constraint.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>
#include <immersx/physics/metric_flow_x.h>
#include <immersx/physics/metric_flow_x_vessel_wall_observable.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <optional>
#include <utility>

#include "metric_flow_x_elastodynamics_mms.h"
#include "test_paths.h"

#if defined(IMMERSX_WITH_METRIC_FLOW_X) && defined(DEAL_II_WITH_SUNDIALS)

TEST(MetricFlowXRadialLaw, ValueAndDerivative)
{
  const auto evaluation = ImmersX::MetricFlowXRadialLaw{}.evaluate(
    {4. * dealii::numbers::PI, dealii::numbers::PI});
  EXPECT_DOUBLE_EQ(evaluation.value, 1.);
  ASSERT_EQ(evaluation.derivatives.size(), 2u);
  EXPECT_DOUBLE_EQ(evaluation.derivatives[0], 1. / (4. * dealii::numbers::PI));
  EXPECT_DOUBLE_EQ(evaluation.derivatives[1], 0.);
}

namespace
{
  using FlowProblem  = MetricFlowX::BloodFlowSystem<1, 3>;
  using FlowVector   = MetricFlowX::VectorType;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter      = ImmersX::IDAAdapter<FlowVector, GlobalVector>;
  using SolidProblem = ImmersX::ElastodynamicsSolver<3>;
  using SolidField   = ImmersX::Field<3, 3, dealii::FEValuesExtractors::Vector>;
  using WallObservable = ImmersX::MetricFlowXAreaRadialDisplacementObservable;
  using Interaction =
    ImmersX::MetricFlowXVesselWallConstraint<SolidField, WallObservable>;
  using SolidFields = decltype(std::declval<Adapter &>().add(
    std::declval<const SolidProblem &>()));
  using FlowDescriptor =
    decltype(ImmersX::metric_flow_x(std::declval<FlowProblem &>()));
  using FlowFields = decltype(std::declval<Adapter &>().add(
    std::declval<const FlowDescriptor &>()));

  class LinearRadialVirtualDisplacement : public dealii::Function<3>
  {
  public:
    explicit LinearRadialVirtualDisplacement(const double scale)
      : dealii::Function<3>(3)
      , scale(scale)
    {}

    double
    value(const dealii::Point<3> &point,
          const unsigned int      component = 0) const override
    {
      if (component == 1)
        return scale * point[1];
      if (component == 2)
        return scale * point[2];
      return 0.;
    }

  private:
    const double scale;
  };

  struct Fixture
  {
    explicit Fixture(const bool high_resolution_section = false)
    {
      const auto parameter_file = ImmersX::TestPaths::parameter_path(
        "gtests/parameters/metric_flow_x_elastodynamics.prm");
      dealii::ParameterAcceptor::clear();
      flow_problem = std::make_unique<FlowProblem>(MPI_COMM_WORLD);
      ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);
      flow_time = std::make_unique<ImmersX::TimeParameters>();
      ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);
      solid_parameters = std::make_unique<ImmersX::ElastodynamicsParameters<3>>(
        "/Elastodynamics/", flow_time.get());
      ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);
      wall_lift = std::make_unique<WallObservable::Lift>(
        "/MetricFlowX vessel wall lift/");
      wall_lift->section.inclusion_degree      = 1;
      wall_lift->section.refinement_level      = 5;
      wall_lift->section.selected_coefficients = {3u, 7u};
      wall_lift->section.n_q_points            = 8;
      wall_lift->representative_n_q_points     = 2;
      ImmersX::ParticleCouplingParameters<3> search_parameters;
      ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);

      ImmersX::initialize_parameters(parameter_file);
      flow_problem->initialize_params(parameter_file);

      wall_lift->section.refinement_level = high_resolution_section ? 5 : 1;

      solid_parameters->dirichlet_ids.clear();
      solid_problem = std::make_unique<SolidProblem>(*solid_parameters);
      solid_problem->make_grid();
      solid_problem->setup_fe();
      solid_problem->setup_system();
      solid_problem->assemble_operators();
      solid_problem->set_initial_conditions();
      flow_problem->setup();

      adapter = std::make_unique<Adapter>(*flow_time, MPI_COMM_WORLD);
      solid_fields.emplace(adapter->add(*solid_problem, "elastodynamics"));
      flow_fields.emplace(
        adapter->add(ImmersX::metric_flow_x(*flow_problem), "blood-flow"));
      solid_space = std::make_unique<ImmersX::FESpaceView<3, 3>>(
        solid_problem->dof_handler(),
        solid_problem->mapping(),
        solid_problem->constraints(),
        &solid_problem->locally_relevant_dofs());
      solid_field = std::make_unique<SolidField>(
        solid_space->field(solid_fields->fields().displacement,
                           "displacement",
                           dealii::FEValuesExtractors::Vector(0)));
      wall_observable =
        std::make_unique<WallObservable>(*flow_problem,
                                         flow_fields->fields().area,
                                         flow_fields->fields().area_components,
                                         *wall_lift);
      interaction = std::make_unique<Interaction>(*solid_field,
                                                  *wall_observable,
                                                  search_parameters);
      interaction->assemble();
      const auto lambda_field = interaction->multiplier_field();
      const auto wall_displacement =
        interaction->radial_displacement(*wall_lift);
      const auto radial_law = wall_observable->radial_law();
      const auto lambda_wall =
        ImmersX::lift(ImmersX::value(lambda_field),
                      *wall_lift,
                      ImmersX::SourceThicknessEvaluator<3>(
                        [radial_law](const dealii::Point<3> &point,
                                     const double            time,
                                     const std::vector<double> &) {
                          return std::sqrt(
                            radial_law.resting_area(point, time) /
                            dealii::numbers::PI);
                        }));
      const auto kinematic_constraint = ImmersX::make_constraint(
        ImmersX::weak_term(ImmersX::value(*solid_field), lambda_wall) -
        ImmersX::weak_term(wall_displacement, lambda_field));
      coupling_fields =
        adapter->add(kinematic_constraint, "vessel-wall").fields();
      adapter->add(*interaction,
                   "vessel-wall-pressure",
                   flow_fields->fields().state,
                   coupling_fields.multiplier);
    }

    void
    initialize(GlobalVector &state, GlobalVector &state_dot) const
    {
      adapter->field(state, solid_fields->fields().displacement) =
        solid_problem->displacement();
      adapter->field(state, solid_fields->fields().velocity) =
        solid_problem->velocity();
      auto &flow_state = adapter->field(state, flow_fields->fields().state);
      flow_problem->initialize_state(flow_state, flow_time->initial_time);
      for (const auto index :
           flow_problem->component_dofs(FlowProblem::Component::area))
        flow_state[index] = flow_problem->vessel_properties(0).a0;
      flow_problem->initialize_state_derivative(
        adapter->field(state_dot, flow_fields->fields().state),
        flow_time->initial_time);
      adapter->field(state, coupling_fields.multiplier)     = 0.;
      adapter->field(state_dot, coupling_fields.multiplier) = 0.;
    }

    std::unique_ptr<FlowProblem>                          flow_problem;
    std::unique_ptr<ImmersX::TimeParameters>              flow_time;
    std::unique_ptr<ImmersX::ElastodynamicsParameters<3>> solid_parameters;
    std::unique_ptr<SolidProblem>                         solid_problem;
    std::unique_ptr<WallObservable::Lift>                 wall_lift;
    std::unique_ptr<ImmersX::FESpaceView<3, 3>>           solid_space;
    std::unique_ptr<SolidField>                           solid_field;
    std::unique_ptr<WallObservable>                       wall_observable;
    std::unique_ptr<Interaction>                          interaction;
    std::unique_ptr<Adapter>                              adapter;
    std::optional<SolidFields>                            solid_fields;
    std::optional<FlowFields>                             flow_fields;
    ImmersX::ConstraintFields                             coupling_fields;
  };
} // namespace

TEST(MetricFlowXVesselWallConstraint, MPI_TwoWayResidualAndPressureSign)
{
  Fixture fixture;
  auto    state     = fixture.adapter->make_state();
  auto    state_dot = fixture.adapter->make_state();
  fixture.initialize(state, state_dot);

  auto residual_zero = fixture.adapter->make_state();
  fixture.adapter->solver().residual(0., state, state_dot, residual_zero);

  auto lambda_state = state;
  fixture.adapter->field(lambda_state, fixture.coupling_fields.multiplier) = 1.;
  auto residual_lambda = fixture.adapter->make_state();
  fixture.adapter->solver().residual(0.,
                                     lambda_state,
                                     state_dot,
                                     residual_lambda);

  auto flow_difference =
    fixture.adapter->field(residual_lambda,
                           fixture.flow_fields->fields().state);
  flow_difference -=
    fixture.adapter->field(residual_zero, fixture.flow_fields->fields().state);
  const auto lambda =
    fixture.adapter->field(lambda_state, fixture.coupling_fields.multiplier);
  auto expected_flow_difference = fixture.flow_problem->make_state();
  expected_flow_difference      = 0.;
  const auto provider =
    fixture.interaction->make_external_pressure_provider(lambda);
  fixture.flow_problem->add_external_pressure_residual(
    0.,
    fixture.adapter->field(state, fixture.flow_fields->fields().state),
    provider,
    expected_flow_difference);
  flow_difference -= expected_flow_difference;
  EXPECT_NEAR(dealii::Utilities::MPI::max(flow_difference.l2_norm(),
                                          MPI_COMM_WORLD),
              0.,
              1.e-12);
  EXPECT_GT(dealii::Utilities::MPI::max(expected_flow_difference.l2_norm(),
                                        MPI_COMM_WORLD),
            0.);

  const double epsilon      = 1.e-7;
  auto         lambda_plus  = state;
  auto         lambda_minus = state;
  fixture.adapter->field(lambda_plus, fixture.coupling_fields.multiplier) =
    epsilon;
  fixture.adapter->field(lambda_minus, fixture.coupling_fields.multiplier) =
    -epsilon;
  auto residual_plus  = fixture.adapter->make_state();
  auto residual_minus = fixture.adapter->make_state();
  fixture.adapter->solver().residual(0., lambda_plus, state_dot, residual_plus);
  fixture.adapter->solver().residual(0.,
                                     lambda_minus,
                                     state_dot,
                                     residual_minus);
  auto finite_difference =
    fixture.adapter->field(residual_plus, fixture.flow_fields->fields().state);
  finite_difference -=
    fixture.adapter->field(residual_minus, fixture.flow_fields->fields().state);
  finite_difference *= 1. / (2. * epsilon);
  finite_difference -= expected_flow_difference;
  EXPECT_NEAR(dealii::Utilities::MPI::max(finite_difference.l2_norm(),
                                          MPI_COMM_WORLD),
              0.,
              1.e-10);

  auto direction = fixture.adapter->make_state();
  fixture.adapter->field(direction, fixture.coupling_fields.multiplier) = 1.;
  fixture.adapter->solver().setup_jacobian(0., state, state_dot, 0.);
  auto jacobian_action = fixture.adapter->make_state();
  fixture.adapter->current_jacobian().vmult(jacobian_action, direction);
  auto jacobian_flow =
    fixture.adapter->field(jacobian_action,
                           fixture.flow_fields->fields().state);
  jacobian_flow -= expected_flow_difference;
  EXPECT_NEAR(dealii::Utilities::MPI::max(jacobian_flow.l2_norm(),
                                          MPI_COMM_WORLD),
              0.,
              1.e-10);

  auto displacement_difference =
    fixture.adapter->field(residual_lambda,
                           fixture.solid_fields->fields().displacement);
  displacement_difference -=
    fixture.adapter->field(residual_zero,
                           fixture.solid_fields->fields().displacement);
  EXPECT_NEAR(displacement_difference.l2_norm(), 0., 1.e-12);

  auto velocity_difference =
    fixture.adapter->field(residual_lambda,
                           fixture.solid_fields->fields().velocity);
  velocity_difference -=
    fixture.adapter->field(residual_zero,
                           fixture.solid_fields->fields().velocity);
  FlowVector expected;
  expected.reinit(fixture.solid_problem->locally_owned_dofs(),
                  fixture.solid_problem->locally_relevant_dofs(),
                  MPI_COMM_WORLD);
  fixture.interaction->solid_coupling_matrix().vmult(
    expected,
    fixture.adapter->field(lambda_state, fixture.coupling_fields.multiplier));
  velocity_difference -= expected;
  EXPECT_NEAR(velocity_difference.l2_norm(), 0., 1.e-12);

  auto area_perturbed = state;
  for (const auto index :
       fixture.flow_problem->component_dofs(FlowProblem::Component::area))
    fixture.adapter->field(area_perturbed,
                           fixture.flow_fields->fields().state)[index] += 1.e-8;
  auto residual_area = fixture.adapter->make_state();
  fixture.adapter->solver().residual(0.,
                                     area_perturbed,
                                     state_dot,
                                     residual_area);
  auto area_constraint_difference =
    fixture.adapter->field(residual_area, fixture.coupling_fields.multiplier);
  area_constraint_difference -=
    fixture.adapter->field(residual_zero, fixture.coupling_fields.multiplier);
  EXPECT_GT(dealii::Utilities::MPI::max(area_constraint_difference.l2_norm(),
                                        MPI_COMM_WORLD),
            0.);

  auto  solid_perturbed = state;
  auto &solid_displacement =
    fixture.adapter->field(solid_perturbed,
                           fixture.solid_fields->fields().displacement);
  solid_displacement  = expected;
  auto residual_solid = fixture.adapter->make_state();
  fixture.adapter->solver().residual(0.,
                                     solid_perturbed,
                                     state_dot,
                                     residual_solid);
  auto solid_constraint_difference =
    fixture.adapter->field(residual_solid, fixture.coupling_fields.multiplier);
  solid_constraint_difference -=
    fixture.adapter->field(residual_zero, fixture.coupling_fields.multiplier);
  EXPECT_GT(dealii::Utilities::MPI::max(solid_constraint_difference.l2_norm(),
                                        MPI_COMM_WORLD),
            0.);

  EXPECT_TRUE(Interaction::flow_pressure_feedback_is_implemented);
}

TEST(MetricFlowXVesselWallConstraint, MPI_TwoWayCompositionResidualSmoke)
{
  Fixture fixture;
  auto    state     = fixture.adapter->make_state();
  auto    state_dot = fixture.adapter->make_state();
  fixture.initialize(state, state_dot);
  auto residual = fixture.adapter->make_state();
  fixture.adapter->solver().residual(0., state, state_dot, residual);

  EXPECT_TRUE(std::isfinite(residual.l2_norm()));
  EXPECT_TRUE(Interaction::flow_pressure_feedback_is_implemented);
}

TEST(MetricFlowXVesselWallConstraint, MPI_DiscreteVirtualWorkNormalization)
{
  Fixture fixture(true);

  const ImmersX::OneVesselMMS::Parameters mms_parameters;
  const auto   properties = fixture.flow_problem->vessel_properties(0);
  const double formula_area =
    ImmersX::OneVesselMMS::reference_area(mms_parameters);
  const double formula_radius = std::sqrt(formula_area / dealii::numbers::PI);
  double       local_geometry_length = 0.;
  for (const auto &cell :
       fixture.flow_problem->triangulation().active_cell_iterators())
    if (cell->is_locally_owned())
      local_geometry_length += cell->diameter();
  const double geometry_length =
    dealii::Utilities::MPI::sum(local_geometry_length, MPI_COMM_WORLD);
  EXPECT_NEAR(properties.a0, formula_area, 1.e-15);
  EXPECT_NEAR(std::sqrt(properties.a0 / dealii::numbers::PI),
              formula_radius,
              1.e-15);
  EXPECT_NEAR(properties.L, mms_parameters.length, 1.e-14);
  EXPECT_NEAR(geometry_length, properties.L, 1.e-14);

  using SolidVector      = typename SolidProblem::VectorType;
  using MultiplierVector = typename Interaction::VectorType;
  MultiplierVector multiplier;
  multiplier.reinit(fixture.interaction->multiplier_locally_owned_dofs(),
                    fixture.interaction->multiplier_locally_relevant_dofs(),
                    MPI_COMM_WORLD);
  constexpr double pressure = 2.;
  multiplier                = pressure;
  multiplier.compress(dealii::VectorOperation::insert);
  multiplier.update_ghost_values();

  SolidVector displacement_owned;
  displacement_owned.reinit(fixture.solid_problem->locally_owned_dofs(),
                            MPI_COMM_WORLD);
  constexpr double                radial_scale = 1.e-3;
  LinearRadialVirtualDisplacement virtual_displacement(radial_scale);
  ASSERT_EQ(fixture.wall_observable->support().selected_modes(),
            (std::vector<unsigned int>{3u, 7u}));
  ASSERT_EQ(
    fixture.wall_observable->support().representative_quadrature().size(), 2u);
  EXPECT_GT(dealii::Utilities::MPI::sum(
              fixture.wall_observable->points().size(), MPI_COMM_WORLD),
            0u);
  dealii::VectorTools::interpolate(fixture.solid_problem->dof_handler(),
                                   virtual_displacement,
                                   displacement_owned);
  displacement_owned.compress(dealii::VectorOperation::insert);
  SolidVector displacement;
  displacement.reinit(fixture.solid_problem->locally_owned_dofs(),
                      fixture.solid_problem->locally_relevant_dofs(),
                      MPI_COMM_WORLD);
  displacement = displacement_owned;
  displacement.update_ghost_values();

  SolidVector reaction;
  reaction.reinit(fixture.solid_problem->locally_owned_dofs(), MPI_COMM_WORLD);
  fixture.interaction->solid_coupling_matrix().vmult(reaction, multiplier);
  double discrete_work = 0.;
  for (const auto index : fixture.solid_problem->locally_owned_dofs())
    discrete_work += displacement[index] * reaction[index];
  discrete_work = dealii::Utilities::MPI::sum(discrete_work, MPI_COMM_WORLD);

  const double expected_work = pressure * 2. * dealii::numbers::PI *
                               radial_scale * formula_radius * formula_radius *
                               properties.L;
  double           expected_metric = 0.;
  MultiplierVector displacement_pairing;
  displacement_pairing.reinit(
    fixture.interaction->multiplier_locally_owned_dofs(),
    fixture.interaction->multiplier_locally_relevant_dofs(),
    MPI_COMM_WORLD);
  fixture.interaction->solid_coupling_matrix().Tvmult(displacement_pairing,
                                                      displacement);
  double transposed_work = 0.;
  for (const auto index : multiplier.locally_owned_elements())
    transposed_work += multiplier[index] * displacement_pairing[index];
  transposed_work =
    dealii::Utilities::MPI::sum(transposed_work, MPI_COMM_WORLD);

  for (const auto &point : fixture.wall_observable->points())
    {
      double basis_sum = 0.;
      for (const auto basis : point.area_basis_values)
        basis_sum += basis;
      EXPECT_NEAR(basis_sum, 1., 1.e-12);
      expected_metric += pressure * pressure * point.weight;
    }

  MultiplierVector metric_image;
  metric_image.reinit(multiplier);
  fixture.interaction->multiplier_metric_matrix().vmult(metric_image,
                                                        multiplier);
  double metric_work = 0.;
  for (const auto index : multiplier.locally_owned_elements())
    metric_work += multiplier[index] * metric_image[index];
  metric_work = dealii::Utilities::MPI::sum(metric_work, MPI_COMM_WORLD);

  expected_metric =
    dealii::Utilities::MPI::sum(expected_metric, MPI_COMM_WORLD);
  const double work_scale = std::max(1., std::abs(discrete_work));
  // The production convention is F_solid + B lambda = 0.  The corresponding
  // virtual-work identity is therefore the positive adjoint pairing
  // <B lambda, delta u> = <lambda, B^T delta u>; the physical interpretation
  // is p*delta A*L when delta A is the radial area variation.
  EXPECT_NEAR(discrete_work, transposed_work, 2.e-10 * work_scale);
  EXPECT_NEAR(discrete_work,
              expected_work,
              2.e-10 * std::max(1., std::abs(expected_work)));
  EXPECT_NEAR(metric_work,
              expected_metric,
              2.e-12 * std::max(1., std::abs(expected_metric)));
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Independent geometry virtual work: discrete="
              << std::setprecision(17) << discrete_work
              << ", analytical=" << expected_work << ", relative defect="
              << std::abs(discrete_work - expected_work) /
                   std::max(1.e-300, std::abs(expected_work))
              << std::endl;
}

TEST(MetricFlowXVesselWallConstraint, MPI_FullCoupledJacobianFiniteDifference)
{
  Fixture fixture;
  auto    state     = fixture.adapter->make_state();
  auto    state_dot = fixture.adapter->make_state();
  fixture.initialize(state, state_dot);

  auto direction     = fixture.adapter->make_state();
  auto dot_direction = fixture.adapter->make_state();
  direction          = 0.;
  dot_direction      = 0.;

  auto fill_direction = [](auto &vector, const double value) {
    for (const auto index : vector.locally_owned_elements())
      vector[index] = value;
    vector.compress(dealii::VectorOperation::insert);
    vector.update_ghost_values();
  };

  fill_direction(fixture.adapter->field(direction,
                                        fixture.flow_fields->fields().state),
                 0.2);
  fill_direction(fixture.adapter->field(
                   direction, fixture.solid_fields->fields().displacement),
                 0.3);
  fill_direction(
    fixture.adapter->field(direction, fixture.solid_fields->fields().velocity),
    0.4);
  fill_direction(fixture.adapter->field(direction,
                                        fixture.coupling_fields.multiplier),
                 0.5);

  const double alpha   = 0.37;
  const double epsilon = 1.e-7;
  dot_direction        = direction;
  dot_direction *= alpha;
  auto state_plus  = state;
  auto state_minus = state;
  auto dot_plus    = state_dot;
  auto dot_minus   = state_dot;
  state_plus.add(epsilon, direction);
  state_minus.add(-epsilon, direction);
  dot_plus.add(epsilon, dot_direction);
  dot_minus.add(-epsilon, dot_direction);
  const auto refresh_ghosts = [&fixture](GlobalVector &value) {
    fixture.adapter->field(value, fixture.flow_fields->fields().state)
      .update_ghost_values();
    fixture.adapter->field(value, fixture.solid_fields->fields().displacement)
      .update_ghost_values();
    fixture.adapter->field(value, fixture.solid_fields->fields().velocity)
      .update_ghost_values();
    fixture.adapter->field(value, fixture.coupling_fields.multiplier)
      .update_ghost_values();
  };
  refresh_ghosts(state_plus);
  refresh_ghosts(state_minus);
  refresh_ghosts(dot_plus);
  refresh_ghosts(dot_minus);

  auto residual_plus  = fixture.adapter->make_state();
  auto residual_minus = fixture.adapter->make_state();
  fixture.adapter->solver().residual(0.13, state_plus, dot_plus, residual_plus);
  fixture.adapter->solver().residual(0.13,
                                     state_minus,
                                     dot_minus,
                                     residual_minus);
  residual_plus -= residual_minus;
  residual_plus *= 1. / (2. * epsilon);

  fixture.adapter->solver().setup_jacobian(0.13, state, state_dot, alpha);
  auto jacobian_action = fixture.adapter->make_state();
  fixture.adapter->current_jacobian().vmult(jacobian_action, direction);
  jacobian_action -= residual_plus;

  const auto check_block =
    [](const auto &error_block, const auto &reference_block, const char *name) {
      const double error =
        dealii::Utilities::MPI::max(error_block.l2_norm(), MPI_COMM_WORLD);
      const double reference =
        dealii::Utilities::MPI::max(reference_block.l2_norm(), MPI_COMM_WORLD);
      const double relative_error = error / std::max(1., reference);
      EXPECT_LT(relative_error, 1.e-6) << name << " Jacobian error = " << error
                                       << ", reference = " << reference;
    };
  check_block(fixture.adapter->field(jacobian_action,
                                     fixture.flow_fields->fields().state),
              fixture.adapter->field(residual_plus,
                                     fixture.flow_fields->fields().state),
              "flow");
  check_block(
    fixture.adapter->field(jacobian_action,
                           fixture.solid_fields->fields().displacement),
    fixture.adapter->field(residual_plus,
                           fixture.solid_fields->fields().displacement),
    "solid displacement");
  check_block(fixture.adapter->field(jacobian_action,
                                     fixture.solid_fields->fields().velocity),
              fixture.adapter->field(residual_plus,
                                     fixture.solid_fields->fields().velocity),
              "solid velocity");
  check_block(
    fixture.adapter->field(jacobian_action, fixture.coupling_fields.multiplier),
    fixture.adapter->field(residual_plus, fixture.coupling_fields.multiplier),
    "constraint");
}

#else

TEST(MetricFlowXVesselWallConstraint, FeatureMacroIsEnabled)
{
  GTEST_SKIP() << "MetricFlowX or deal.II SUNDIALS support is unavailable.";
}

#endif
