// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of the ImmersX application, based on deal.II.
// ---------------------------------------------------------------------
#include <gtest/gtest.h>
#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X
#  include <metric_flow_x/blood_flow_system.h>

#  include <array>
#  include <map>

#  include "../apps/metric_flow_x_history.h"
#  include "test_paths.h"

namespace
{
  template <class Flow>
  auto
  check_history(int) -> decltype(std::declval<Flow &>().load_coupling_field(
                                   std::string(),
                                   std::string(),
                                   std::declval<MetricFlowX::VectorType &>()),
                                 void())
  {
    using namespace dealii;
    const auto comm = MPI_COMM_WORLD;
    const auto root = ImmersX::TestPaths::output_path(
      "flow-history/" + std::to_string(Utilities::MPI::n_mpi_processes(comm)));
    if (Utilities::MPI::this_mpi_process(comm) == 0)
      std::filesystem::remove_all(root);
    MPI_Barrier(comm);
    const std::array<double, 2> times{{0., .001}};
    using Values = std::map<types::global_dof_index, double>;
    std::array<std::array<Values, 3>, 2> expected;
    {
      ParameterAcceptor::clear();
      Flow flow(comm);
      flow.initialize_params(ImmersX::TestPaths::parameter_path(
                               "gtests/parameters/metric_flow_x.prm"),
                             "");
      flow.setup();
      auto state = flow.make_state(), rate = flow.make_state(),
           pressure = flow.make_state();
      flow.initialize_state(state, 0.);
      rate = 0.;
      for (unsigned int step = 0; step < times.size(); ++step)
        {
          for (auto i : flow.component_dofs(Flow::Component::area))
            state[i] *= 1. + .01 * step;
          for (auto i : flow.component_dofs(Flow::Component::velocity))
            state[i] = .25 + step;
          state.compress(VectorOperation::insert);
          flow.compute_pressure(state, rate, pressure, times[step]);
          for (auto i : flow.component_dofs(Flow::Component::area))
            {
              expected[step][0][i] = state[i];
              expected[step][2][i] = pressure[i] + 17.;
            }
          for (auto i : flow.component_dofs(Flow::Component::velocity))
            expected[step][1][i] = state[i];
          flow.set_external_pressure_provider([](const auto &) { return 17.; });
          ImmersX::metric_flow_x_app_detail::write_flow_history(
            flow, root.string(), times[step], state, rate, step == 0, 0);
          flow.clear_external_pressure_provider();
        }
    }
    ParameterAcceptor::clear();
    Flow restored(comm);
    restored.load_checkpoint((root / "layout").string());
    auto values = restored.make_state();
    for (unsigned int step = 0; step < times.size(); ++step)
      {
        std::ostringstream name;
        name << "fields-" << std::hexfloat << times[step];
        unsigned int field = 0;
        for (const auto &label : {"area", "velocity", "pressure"})
          {
            EXPECT_DOUBLE_EQ(restored.load_coupling_field(
                               (root / name.str()).string(), label, values),
                             times[step]);
            for (const auto &[i, value] : expected[step][field])
              EXPECT_NEAR(values[i],
                          value,
                          1.e-11 * std::max(1., std::abs(value)));
            ++field;
          }
      }
  }
  template <class Flow>
  void
  check_history(long)
  {
    GTEST_SKIP() << "MetricFlowX coupling-field API unavailable";
  }
} // namespace

TEST(MetricFlowX, BOTH_FlowHistoryReplaysChangingFieldsAndExternalPressure)
{
  check_history<MetricFlowX::BloodFlowSystem<1, 3>>(0);
}
#endif
