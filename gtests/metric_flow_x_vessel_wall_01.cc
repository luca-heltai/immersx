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
#include <deal.II/base/utilities.h>

#include <gtest/gtest.h>
#include <immersx/core/state.h>
#include <immersx/core/time_parameters.h>
#include <immersx/physics/metric_flow_x.h>
#include <immersx/physics/metric_flow_x_vessel_wall_representation.h>

#include <cmath>
#include <memory>

#include "test_paths.h"

#if defined(IMMERSX_WITH_METRIC_FLOW_X) && defined(DEAL_II_WITH_SUNDIALS)

namespace
{
  using Problem = MetricFlowX::BloodFlowSystem<1, 3>;
  using State   = MetricFlowX::VectorType;
  using Representation =
    ImmersX::MetricFlowXAreaRadialDisplacementRepresentation;
  using Lift = Representation::Lift;

  struct Fixture
  {
    Fixture()
    {
      dealii::ParameterAcceptor::clear();
      problem         = std::make_unique<Problem>(MPI_COMM_WORLD);
      time_parameters = std::make_unique<ImmersX::TimeParameters>(
        "/MetricFlowSystem<1, 3>/Time parameters/");
      problem->initialize_params(ImmersX::TestPaths::parameter_path(
        "gtests/parameters/metric_flow_x.prm"));
      problem->setup();

      ImmersX::FieldDescriptor descriptor;
      descriptor.name             = "flow";
      descriptor.locally_owned    = problem->locally_owned_dofs();
      descriptor.locally_relevant = problem->locally_relevant_dofs();
      field                       = layout.add_field(std::move(descriptor));
      area = std::make_unique<ImmersX::FieldComponentView>(
        field, problem->component_dofs(Problem::Component::area));

      lift = std::make_unique<Lift>("/Vessel wall test lift/");
      lift->section.inclusion_degree      = 1;
      lift->section.refinement_level      = 1;
      lift->section.selected_coefficients = {3u, 7u};
      lift->section.n_q_points            = 8;
      lift->representative_n_q_points     = 2;
      representation = std::make_unique<Representation>(*problem, *area, *lift);
    }

    void
    bind(State &state, const double time = 0.)
    {
      state_view = std::make_unique<ImmersX::StateView<State>>(layout, time);
      state_view->bind(field, state);
      context = std::make_unique<ImmersX::EvaluationContext<State>>(time,
                                                                    *state_view,
                                                                    nullptr);
    }

    std::unique_ptr<ImmersX::TimeParameters>           time_parameters;
    std::unique_ptr<Problem>                           problem;
    ImmersX::StateLayout                               layout;
    ImmersX::FieldId                                   field;
    std::unique_ptr<ImmersX::FieldComponentView>       area;
    std::unique_ptr<Lift>                              lift;
    std::unique_ptr<Representation>                    representation;
    std::unique_ptr<ImmersX::StateView<State>>         state_view;
    std::unique_ptr<ImmersX::EvaluationContext<State>> context;
  };

  State
  state_with_constant_area(Fixture &fixture, const double area)
  {
    State result = fixture.problem->make_state();
    fixture.problem->initialize_state(result, 0.);
    for (const auto index :
         fixture.problem->component_dofs(Problem::Component::area))
      result[index] = area;
    return result;
  }
} // namespace

TEST(MetricFlowXVesselWallRepresentation, RestStateIsZero)
{
  Fixture      fixture;
  const double a0    = fixture.problem->vessel_properties(0).a0;
  auto         state = state_with_constant_area(fixture, a0);
  fixture.bind(state);
  const auto values = fixture.representation->evaluate(*fixture.context);
  ASSERT_FALSE(values.empty());
  for (const auto &value : values)
    EXPECT_NEAR(value.norm(), 0., 2.e-12);
}

TEST(MetricFlowXVesselWallRepresentation, RadiusIncrementIsRadial)
{
  Fixture      fixture;
  const double a0 = fixture.problem->vessel_properties(0).a0;
  const double r0 = std::sqrt(a0 / dealii::numbers::PI);
  const double dr = 0.001;
  auto         state =
    state_with_constant_area(fixture,
                             dealii::numbers::PI * (r0 + dr) * (r0 + dr));
  fixture.bind(state);
  const auto values = fixture.representation->evaluate(*fixture.context);
  ASSERT_FALSE(values.empty());
  for (std::size_t q = 0; q < values.size(); ++q)
    {
      EXPECT_NEAR(values[q].norm(), dr, 2.e-11);
      EXPECT_NEAR(values[q] * fixture.representation->points()[q].normal,
                  dr,
                  2.e-11);
    }
  EXPECT_DOUBLE_EQ(fixture.representation->mode_coefficients(dr)[0], dr);
  EXPECT_DOUBLE_EQ(fixture.representation->mode_coefficients(dr)[1], dr);
}

TEST(MetricFlowXVesselWallRepresentation, AreaLinearizationAndTranspose)
{
  Fixture      fixture;
  const double a0      = fixture.problem->vessel_properties(0).a0;
  const double r0      = std::sqrt(a0 / dealii::numbers::PI);
  const double area    = dealii::numbers::PI * (r0 + 0.001) * (r0 + 0.001);
  const double epsilon = 1.e-8;
  auto         state   = state_with_constant_area(fixture, area);
  auto         plus    = state;
  auto         minus   = state;
  for (const auto index :
       fixture.problem->component_dofs(Problem::Component::area))
    {
      plus[index] += epsilon;
      minus[index] -= epsilon;
    }
  fixture.bind(state);
  const auto derivative = fixture.representation->linearize(*fixture.context);
  State      direction  = state;
  direction             = 0.;
  for (const auto index :
       fixture.problem->component_dofs(Problem::Component::area))
    direction[index] = 1.;
  std::vector<dealii::Tensor<1, 3>> analytical;
  derivative.reinit_range_vector(analytical, false);
  derivative.vmult(analytical, direction);

  fixture.bind(plus);
  const auto plus_values = fixture.representation->evaluate(*fixture.context);
  fixture.bind(minus);
  const auto minus_values = fixture.representation->evaluate(*fixture.context);
  ASSERT_EQ(analytical.size(), plus_values.size());
  for (std::size_t q = 0; q < analytical.size(); ++q)
    {
      auto finite_difference = plus_values[q];
      finite_difference -= minus_values[q];
      finite_difference *= 1. / (2. * epsilon);
      EXPECT_NEAR((analytical[q] - finite_difference).norm(), 0., 2.e-8);
    }

  fixture.bind(state);
  const auto transpose = fixture.representation->linearize(*fixture.context);
  std::vector<dealii::Tensor<1, 3>> values(analytical.size());
  for (std::size_t q = 0; q < values.size(); ++q)
    values[q] = fixture.representation->points()[q].normal;
  State transposed;
  transpose.reinit_domain_vector(transposed, false);
  transpose.Tvmult(transposed, values);
  double left = 0.;
  for (std::size_t q = 0; q < analytical.size(); ++q)
    left += values[q] * analytical[q];
  double right = transposed * direction;
  EXPECT_NEAR(left, right, 2.e-9);
}

TEST(MetricFlowXVesselWallRepresentation, MPI_CompactMultiplierSpace)
{
  Fixture     fixture;
  const auto &owned = fixture.representation->multiplier_locally_owned_dofs();
  const auto &relevant =
    fixture.representation->multiplier_locally_relevant_dofs();
  const auto global_owned =
    dealii::Utilities::MPI::sum(owned.n_elements(), MPI_COMM_WORLD);
  EXPECT_EQ(global_owned, owned.size());
  EXPECT_EQ(relevant.size(), owned.size());
  for (const auto &point : fixture.representation->points())
    {
      ASSERT_EQ(point.multiplier_dof_indices.size(),
                point.area_basis_values.size());
      for (const auto index : point.multiplier_dof_indices)
        EXPECT_TRUE(relevant.is_element(index));
    }
}

#else

TEST(MetricFlowXVesselWallRepresentation, FeatureMacroIsEnabled)
{
  GTEST_SKIP() << "MetricFlowX or deal.II SUNDIALS support is unavailable.";
}

#endif
