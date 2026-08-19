// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>

#include "state_history.h"
#include "time_residual.h"


using dealii::Vector;


TEST(StateHistory, InterpolatesAcceptedSnapshots) // NOLINT
{
  StateHistory<Vector<double>> history;

  Vector<double> first(2);
  first[0] = 0.;
  first[1] = 10.;
  Vector<double> second(2);
  second[0] = 4.;
  second[1] = 2.;

  history.accept(0., first);
  history.accept(2., second);

  const auto middle = history.at(0.5);
  EXPECT_DOUBLE_EQ(middle[0], 1.);
  EXPECT_DOUBLE_EQ(middle[1], 8.);
  EXPECT_DOUBLE_EQ(history.latest()[0], 4.);
  EXPECT_EQ(history.size(), 2u);
}


TEST(StateHistory, SubsystemsKeepIndependentTimeGrids) // NOLINT
{
  StateHistoryRegistry<double>  histories;
  const ImmersX::HistoryGroupId fluid(0);
  const ImmersX::HistoryGroupId network(1);
  histories.accept(fluid, 0., 0.);
  histories.accept(fluid, 2., 20.);
  histories.accept(network, 0., 10.);
  histories.accept(network, 0.5, 15.);
  histories.accept(network, 1., 20.);

  EXPECT_DOUBLE_EQ(histories.at(fluid, 1.), 10.);
  EXPECT_DOUBLE_EQ(histories.at(network, 0.25), 12.5);
  EXPECT_DOUBLE_EQ(histories.history(fluid).last_time(), 2.);
  EXPECT_DOUBLE_EQ(histories.history(network).last_time(), 1.);
}


TEST(StateHistory, SameResidualWorksWithIntermediateHistoryQuery) // NOLINT
{
  using VectorType = Vector<double>;

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor descriptor;
  descriptor.name   = "active";
  const auto active = layout.add_field(descriptor);

  ImmersX::TimeResidualModel<VectorType> model;

  StateHistoryRegistry<VectorType> histories;
  const ImmersX::HistoryGroupId    network(1);
  VectorType                       subsystem_one_start(1);
  subsystem_one_start[0] = 0.;
  VectorType subsystem_one_end(1);
  subsystem_one_end[0] = 4.;
  histories.accept(network, 0., subsystem_one_start);
  histories.accept(network, 1., subsystem_one_end);

  VectorType active_stage(1);
  active_stage[0] = 2.;
  VectorType active_stage_dot(1);
  active_stage_dot[0] = 0.;
  ImmersX::StateView<VectorType> state(layout, 0.25);
  state.bind(active, active_stage);
  ImmersX::StateView<VectorType> state_dot(layout, 0.25);
  state_dot.bind(active, active_stage_dot);
  ImmersX::EvaluationContext<VectorType> context(
    0.25, state, &state_dot, {}, histories.query());

  model.add_term(time_residual_terms::diffusion,
                 [active, network](const auto &evaluation, auto &residual) {
                   const auto other =
                     evaluation.historical_state(network, evaluation.time());
                   residual.field(active)[0] +=
                     evaluation.state().field(active, evaluation.time())[0] +
                     other[0];
                 });

  VectorType residual(1);
  residual = 0.;
  ImmersX::ResidualAccumulator<VectorType> accumulator(layout);
  accumulator.bind(active, residual);
  model.evaluate(context, accumulator);
  EXPECT_DOUBLE_EQ(residual[0], 3.);
}
