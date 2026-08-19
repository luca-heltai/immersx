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
  StateHistoryRegistry<double> histories;
  histories.accept(0, 0., 0.);
  histories.accept(0, 2., 20.);
  histories.accept(1, 0., 10.);
  histories.accept(1, 0.5, 15.);
  histories.accept(1, 1., 20.);

  EXPECT_DOUBLE_EQ(histories.at(0, 1.), 10.);
  EXPECT_DOUBLE_EQ(histories.at(1, 0.25), 12.5);
  EXPECT_DOUBLE_EQ(histories.history(0).last_time(), 2.);
  EXPECT_DOUBLE_EQ(histories.history(1).last_time(), 1.);
}


TEST(StateHistory, SameResidualWorksWithIntermediateHistoryQuery) // NOLINT
{
  using VectorType = Vector<double>;

  TimeResidualModel<VectorType> model;
  model.add_term(time_residual_terms::diffusion,
                 [](const auto &context, auto &residual) {
                   const auto other = context.historical_state(1, context.time);
                   residual[0] += context.state[0] + other[0];
                 });

  StateHistoryRegistry<VectorType> histories;
  VectorType                       subsystem_one_start(1);
  subsystem_one_start[0] = 0.;
  VectorType subsystem_one_end(1);
  subsystem_one_end[0] = 4.;
  histories.accept(1, 0., subsystem_one_start);
  histories.accept(1, 1., subsystem_one_end);

  VectorType active_stage(1);
  active_stage[0] = 2.;
  VectorType active_stage_dot(1);
  active_stage_dot[0] = 0.;
  TimeResidualContext<VectorType> context(0.25, active_stage, active_stage_dot);
  context.history_query = histories.query();

  VectorType residual(1);
  model.residual(context, residual);
  EXPECT_DOUBLE_EQ(residual[0], 3.);
}
