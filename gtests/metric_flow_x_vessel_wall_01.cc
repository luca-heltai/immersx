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
#include <immersx/physics/metric_flow_x_vessel_wall_observable.h>

#include <cmath>
#include <map>
#include <memory>

#include "test_paths.h"

#if defined(IMMERSX_WITH_METRIC_FLOW_X) && defined(DEAL_II_WITH_SUNDIALS)

namespace
{
  using Problem    = MetricFlowX::BloodFlowSystem<1, 3>;
  using State      = MetricFlowX::VectorType;
  using Observable = ImmersX::MetricFlowXAreaRadialDisplacementObservable;
  using Lift       = Observable::Lift;

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
      space = std::make_shared<ImmersX::FESpaceView<1, 3>>(
        problem->dof_handler(),
        dealii::StaticMappingQ1<1, 3>::mapping,
        problem->constraints(),
        &problem->locally_relevant_dofs());
      area = std::make_unique<
        decltype(space->field(field, "area", problem->area_extractor()))>(
        space->field(field, "area", problem->area_extractor()));
      area_components = problem->component_dofs(Problem::Component::area);

      lift = std::make_unique<Lift>("/Vessel wall test lift/");
      lift->section.inclusion_degree      = 1;
      lift->section.refinement_level      = 5;
      lift->section.selected_coefficients = {3u, 7u};
      lift->section.n_q_points            = 32;
      lift->representative_n_q_points     = 2;
      representation =
        std::make_unique<Observable>(*problem, *area, area_components, *lift);
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

    std::unique_ptr<ImmersX::TimeParameters>          time_parameters;
    std::unique_ptr<Problem>                          problem;
    ImmersX::StateLayout                              layout;
    ImmersX::FieldId                                  field;
    std::shared_ptr<const ImmersX::FESpaceView<1, 3>> space;
    std::unique_ptr<ImmersX::Field<1, 3, dealii::FEValuesExtractors::Scalar>>
                                                       area;
    dealii::IndexSet                                   area_components;
    std::unique_ptr<Lift>                              lift;
    std::unique_ptr<Observable>                        representation;
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

TEST(MetricFlowXVesselWallObservable, RestStateIsZero)
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

TEST(MetricFlowXVesselWallObservable, RadiusIncrementIsRadial)
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

TEST(MetricFlowXVesselWallObservable, AreaLinearizationAndTranspose)
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
  EXPECT_NEAR(left,
              right,
              2.e-9 * std::max(1., std::max(std::abs(left), std::abs(right))));
}

TEST(MetricFlowXVesselWallObservable, MPI_CompactMultiplierSpace)
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

TEST(MetricFlowXVesselWallObservable, MPI_ExternalPressureInterpolation)
{
  Fixture fixture;
  State   multiplier;
  multiplier.reinit(fixture.representation->multiplier_locally_owned_dofs(),
                    fixture.representation->multiplier_locally_relevant_dofs(),
                    MPI_COMM_WORLD);
  multiplier = 0.;

  std::map<dealii::types::global_dof_index, dealii::types::global_dof_index>
    area_to_multiplier;
  for (const auto &point : fixture.representation->points())
    for (unsigned int i = 0; i < point.dof_indices.size(); ++i)
      area_to_multiplier.emplace(point.dof_indices[i],
                                 point.multiplier_dof_indices[i]);

  const unsigned int n_dofs =
    fixture.problem->finite_element().n_dofs_per_cell();
  std::vector<dealii::types::global_dof_index> local_dofs(n_dofs);
  for (const auto &cell :
       fixture.problem->dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell->get_dof_indices(local_dofs);
        for (unsigned int i = 0; i < n_dofs; ++i)
          if (fixture.problem->finite_element()
                .system_to_component_index(i)
                .first == 0)
            {
              const auto full   = local_dofs[i];
              const auto map_it = area_to_multiplier.find(full);
              if (map_it != area_to_multiplier.end() &&
                  multiplier.locally_owned_elements().is_element(
                    map_it->second))
                {
                  const double value =
                    fixture.problem->finite_element().shape_value_component(
                      i, dealii::Point<1>(1.), 0.);
                  multiplier[map_it->second] = value;
                }
            }
      }
  multiplier.compress(dealii::VectorOperation::insert);

  const auto provider =
    fixture.representation->make_external_pressure_provider(multiplier);
  ASSERT_GT(dealii::Utilities::MPI::max(fixture.representation->points().size(),
                                        MPI_COMM_WORLD),
            0u);
  double local_min = std::numeric_limits<double>::max();
  double local_max = -std::numeric_limits<double>::max();
  for (const auto &point : fixture.representation->points())
    {
      const MetricFlowX::BloodFlowSystem<1, 3>::PressureEvaluationPoint
             evaluation{0., point.point, 0u, point.cell_id};
      double expected = 0.;
      for (unsigned int i = 0; i < point.multiplier_dof_indices.size(); ++i)
        expected += point.area_basis_values[i] *
                    multiplier[point.multiplier_dof_indices[i]];
      EXPECT_NEAR(provider(evaluation), expected, 1.e-12);
      local_min = std::min(local_min, expected);
      local_max = std::max(local_max, expected);
    }
  EXPECT_GT(dealii::Utilities::MPI::max(local_max, MPI_COMM_WORLD) -
              dealii::Utilities::MPI::min(local_min, MPI_COMM_WORLD),
            1.e-8);
}

TEST(MetricFlowXVesselWallObservable, AreaPressureNormalization)
{
  Fixture     fixture;
  const auto &points = fixture.representation->points();
  ASSERT_FALSE(points.empty());

  dealii::FEValues<1, 3> fe_values(
    dealii::StaticMappingQ1<1, 3>::mapping,
    fixture.problem->finite_element(),
    fixture.representation->support().representative_quadrature(),
    dealii::update_JxW_values);
  for (const auto &cell :
       fixture.problem->dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        for (const auto q : fe_values.quadrature_point_indices())
          {
            double section_measure = 0.;
            for (const auto &point : points)
              if (point.cell_id == cell->id() &&
                  point.representative_qpoint == q)
                {
                  const double radius =
                    std::sqrt(point.a0 / dealii::numbers::PI);
                  section_measure += point.weight;
                  EXPECT_NEAR(radius * 2. * dealii::numbers::PI *
                                fixture.representation->radius_derivative(
                                  point.a0),
                              1.,
                              2.e-14);
                }
            const double normalized_measure =
              section_measure / fe_values.JxW(q);
            EXPECT_NEAR(normalized_measure,
                        fixture.representation->support()
                          .reference_cross_section()
                          .measure(
                            std::sqrt(points.front().a0 / dealii::numbers::PI)),
                        2.e-10);
            // The TensorProduct quadrature integrates the circular
            // cross-section measure 2*pi*R.  Multiplying it by dR/dA must
            // produce one: this is the pressure-work normalization, not an
            // empirical circumference correction.
            EXPECT_NEAR(normalized_measure *
                          fixture.representation->radius_derivative(
                            points.front().a0),
                        1.,
                        5.e-4);
            EXPECT_NEAR(
              normalized_measure *
                fixture.representation->radius_derivative(points.front().a0),
              section_measure / fe_values.JxW(q) /
                (2. * std::sqrt(dealii::numbers::PI * points.front().a0)),
              2.e-14);
          }
      }
}

#else

TEST(MetricFlowXVesselWallObservable, FeatureMacroIsEnabled)
{
  GTEST_SKIP() << "MetricFlowX or deal.II SUNDIALS support is unavailable.";
}

#endif
