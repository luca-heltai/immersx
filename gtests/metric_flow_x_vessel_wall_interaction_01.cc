// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>
#include <immersx/algebra/vessel_wall_interaction.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>
#include <immersx/physics/metric_flow_x.h>
#include <immersx/physics/metric_flow_x_vessel_wall_representation.h>

#include <optional>
#include <utility>

#include "test_paths.h"

#if defined(IMMERSX_WITH_METRIC_FLOW_X) && defined(DEAL_II_WITH_SUNDIALS)

namespace
{
  using FlowProblem         = MetricFlowX::BloodFlowSystem<1, 3>;
  using FlowVector          = MetricFlowX::VectorType;
  using GlobalVector        = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter             = ImmersX::IDAAdapter<FlowVector, GlobalVector>;
  using SolidProblem        = ImmersX::ElastodynamicsSolver<3>;
  using SolidRepresentation = ImmersX::VectorFiniteElementRepresentation<3, 3>;
  using WallRepresentation =
    ImmersX::MetricFlowXAreaRadialDisplacementRepresentation;
  using Interaction =
    ImmersX::VesselWallInteraction<SolidRepresentation, WallRepresentation>;
  using SolidFields = decltype(std::declval<Adapter &>().add(
    std::declval<const SolidProblem &>()));
  using FlowDescriptor =
    decltype(ImmersX::metric_flow_x(std::declval<FlowProblem &>()));
  using FlowFields = decltype(std::declval<Adapter &>().add(
    std::declval<const FlowDescriptor &>()));

  struct Fixture
  {
    Fixture()
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
      wall_lift = std::make_unique<WallRepresentation::Lift>(
        "/MetricFlowX vessel wall lift/");
      wall_lift->section.inclusion_degree      = 1;
      wall_lift->section.refinement_level      = 1;
      wall_lift->section.selected_coefficients = {3u, 7u};
      wall_lift->section.n_q_points            = 8;
      wall_lift->representative_n_q_points     = 2;
      ImmersX::ParticleCouplingParameters<3> search_parameters;
      ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);

      ImmersX::initialize_parameters(parameter_file);
      flow_problem->initialize_params(parameter_file);

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
      solid_representation = std::make_unique<SolidRepresentation>(
        solid_problem->triangulation(),
        solid_problem->dof_handler(),
        solid_problem->locally_owned_dofs(),
        solid_problem->locally_relevant_dofs(),
        solid_problem->constraints(),
        solid_problem->mapping(),
        dealii::FEValuesExtractors::Vector(0));
      wall_representation =
        std::make_unique<WallRepresentation>(*flow_problem,
                                             flow_fields->fields().area,
                                             *wall_lift);
      interaction = std::make_unique<Interaction>(*solid_representation,
                                                  *wall_representation,
                                                  search_parameters);
      interaction->assemble();
      coupling_fields = adapter
                          ->add(*interaction,
                                "vessel-wall",
                                solid_fields->fields().displacement,
                                solid_fields->fields().velocity,
                                flow_fields->fields().state)
                          .fields();
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
    std::unique_ptr<WallRepresentation::Lift>             wall_lift;
    std::unique_ptr<SolidRepresentation>                  solid_representation;
    std::unique_ptr<WallRepresentation>                   wall_representation;
    std::unique_ptr<Interaction>                          interaction;
    std::unique_ptr<Adapter>                              adapter;
    std::optional<SolidFields>                            solid_fields;
    std::optional<FlowFields>                             flow_fields;
    ImmersX::ConstraintFields                             coupling_fields;
  };
} // namespace

TEST(MetricFlowXVesselWallInteraction, MPI_TwoWayResidualAndPressureSign)
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

#else

TEST(MetricFlowXVesselWallInteraction, FeatureMacroIsEnabled)
{
  GTEST_SKIP() << "MetricFlowX or deal.II SUNDIALS support is unavailable.";
}

#endif
