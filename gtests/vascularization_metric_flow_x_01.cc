// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/metric_flow_x.h>

#include "test_paths.h"

#if defined(IMMERSX_WITH_METRIC_FLOW_X) && defined(DEAL_II_WITH_SUNDIALS)
namespace
{
  using namespace dealii;
  using namespace ImmersX::Vascularization;
  constexpr Id base = Id(1) << 60;

  template <class Flow>
  void
  exercise_native_transfer()
  {
    if constexpr (!has_metric_flow_x_network_input<Flow>)
      GTEST_SKIP() << "Installed MetricFlowX has no in-memory input API.";
    else
      {
        const auto comm = MPI_COMM_WORLD;
        const auto rank = Utilities::MPI::this_mpi_process(comm);
        const auto last = Utilities::MPI::n_mpi_processes(comm) - 1;
        ParameterAcceptor::clear();
        parallel::distributed::Triangulation<3> background(comm);
        GridGenerator::hyper_cube(background);
        background.refine_global(2);
        Flow flow(comm);
        flow.initialize_params(
          ImmersX::TestPaths::parameter_path(
            "gtests/parameters/vascularization_metric_flow_x.prm"),
          ImmersX::TestPaths::output_path(
            "vascular-metric-flow-x/last_used_parameters.prm")
            .string());
        GenerationResult result;
        if (rank == 0 || rank == last)
          {
            result.network.nodes[base + 1] = {Point<3>(.1, .3, .5),
                                              0,
                                              Port::arterial_inlet};
            result.network.nodes[base + 3] = {Point<3>(.9, .3, .5), 0};
            result.network.nodes[base + 5] = {Point<3>(.9, .7, .5),
                                              last,
                                              Port::venous_outlet};
            result.network.nodes[base + 7] = {Point<3>(.1, .7, .5), last};
            Vessel artery, vein;
            artery.nodes                        = {{base + 1, base + 3}};
            artery.radii                        = {{.003, .0025}};
            artery.viscosity                    = .004;
            vein.nodes                          = {{base + 5, base + 7}};
            vein.radii                          = {{.004, .0035}};
            vein.viscosity                      = .005;
            vein.system                         = System::venous;
            vein.owner                          = last;
            result.network.vessels[base + 11]   = artery;
            result.network.vessels[base + 13]   = vein;
            result.network.terminals[base + 17] = {
              base + 3, base + 21, System::arterial, 0, 1.};
            result.network.terminals[base + 19] = {
              base + 7, base + 21, System::venous, last, 1.};
            if (rank == 0)
              result.terminal_pairs.push_back(
                {base + 21, base + 17, base + 19, 1.e9});
          }
        result.validation =
          validate_data(result.network, Utilities::MPI::n_mpi_processes(comm));
        ASSERT_TRUE(result.validation.valid());
        ASSERT_TRUE(
          validate_terminal_pairs(result.network, result.terminal_pairs)
            .valid());
        result.discretization = discretize(background,
                                           MappingQ1<3>(),
                                           result.network,
                                           DiscretizationParameters{},
                                           comm);
        ASSERT_TRUE(result.discretization.validation.valid());
        const auto   original     = result.network;
        unsigned int vessel_calls = 0, boundary_calls = 0;
        const auto   transfer = prepare_metric_flow_x(
          flow,
          result,
          [&](Id, const Vessel &vessel) {
            EXPECT_EQ(vessel.owner, rank);
            ++vessel_calls;
            typename Flow::VesselProperties properties;
            properties.E      = 5.e5;
            properties.h_wall = .0003;
            return properties;
          },
          [&](Id, const Node &node, const Terminal *terminal) {
            EXPECT_NE(node.port, Port::arterial_inlet);
            if (terminal)
              EXPECT_EQ(terminal->owner, rank);
            else
              EXPECT_EQ(node.owner, rank);
            ++boundary_calls;
            using Physics = typename Flow::NetworkPhysics;
            using RCR = typename decltype(Physics{}.terminals)::mapped_type;
            return RCR{0., 1.e8, 0., 0., "synthetic"};
          });
        EXPECT_EQ(Utilities::MPI::sum(vessel_calls, comm), 2u);
        EXPECT_EQ(Utilities::MPI::sum(boundary_calls, comm), 3u);
        ASSERT_EQ(transfer.vessel_ids.size(), 2u);
        ASSERT_EQ(transfer.terminal_ids.size(), 2u);
        EXPECT_EQ(transfer.reversed_vessels, std::set<Id>({base + 13}));
        EXPECT_EQ(transfer.boundary_ids.at(base + 1), 0u);
        EXPECT_NE(transfer.terminal_ids.at(base + 17),
                  transfer.terminal_ids.at(base + 19));
        EXPECT_DOUBLE_EQ(
          flow.vessel_properties(transfer.vessel_ids.at(base + 11)).r_in, .003);
        EXPECT_DOUBLE_EQ(
          flow.vessel_properties(transfer.vessel_ids.at(base + 13)).r_in,
          .0035);
        EXPECT_DOUBLE_EQ(
          flow.vessel_properties(transfer.vessel_ids.at(base + 13)).r_out,
          .004);
        EXPECT_DOUBLE_EQ(flow
                           .vessel_properties(transfer.vessel_ids.at(base + 13))
                           .dynamic_viscosity,
                         .005);
        EXPECT_EQ(result.network.vessels.size(), original.vessels.size());
        for (const auto &[id, vessel] : original.vessels)
          {
            EXPECT_EQ(result.network.vessels.at(id).nodes, vessel.nodes);
            EXPECT_EQ(result.network.vessels.at(id).radii, vessel.radii);
            EXPECT_EQ(result.network.vessels.at(id).system, vessel.system);
          }
        auto state = flow.make_state(), rate = flow.make_state(),
             residual = flow.make_state();
        flow.initialize_consistent_state(state, rate, .17);
        EXPECT_GT(rate.l2_norm(), 0.);
        flow.assemble_residual(.17, state, rate, residual);
        flow.assemble_state_jacobian(.17, state, rate);
        flow.assemble_derivative_jacobian(.17, state, rate);
        EXPECT_TRUE(std::isfinite(residual.l2_norm()));
        unsigned int inlet = 0, venous_outlet = 0;
        for (const auto &cell : flow.dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto reference = transfer.reference(cell->id());
              const bool vein      = reference.vessel == base + 13;
              EXPECT_TRUE(vein || reference.vessel == base + 11);
              const Point<3> a =
                vein ? Point<3>(.9, .7, .5) : Point<3>(.1, .3, .5);
              const Point<3> b =
                vein ? Point<3>(.1, .7, .5) : Point<3>(.9, .3, .5);
              for (unsigned int f = 0; f < 2; ++f)
                {
                  const double t =
                    (f == 0) != vein ? reference.begin : reference.end;
                  EXPECT_LT(cell->vertex(f).distance((1. - t) * a + t * b),
                            1.e-13);
                  if (cell->face(f)->at_boundary())
                    {
                      inlet += cell->face(f)->boundary_id() == 0;
                      if (cell->face(f)->boundary_id() ==
                          transfer.boundary_ids.at(base + 5))
                        {
                          ++venous_outlet;
                          EXPECT_EQ(f, 1u);
                        }
                    }
                }
              std::vector<types::global_dof_index> indices(
                cell->get_fe().n_dofs_per_cell());
              cell->get_dof_indices(indices);
              if (cell->face(0)->at_boundary() &&
                  cell->face(0)->boundary_id() == 0)
                {
                  double area = 0., velocity = 0.;
                  for (unsigned int i = 0; i < indices.size(); ++i)
                    if (cell->get_fe().unit_support_point(i)[0] == 0.)
                      {
                        const auto component =
                          cell->get_fe().system_to_component_index(i).first;
                        if (component == 2)
                          area = state(indices[i]);
                        else if (component == 3)
                          velocity = state(indices[i]);
                      }
                  EXPECT_GT(area, 0.);
                  EXPECT_NEAR(area * velocity, 2.e-9 * (1. + .17), 1.e-16);
                }
              for (unsigned int i = 0; i < indices.size(); ++i)
                if (cell->get_fe().system_to_component_index(i).first == 0)
                  {
                    const double xi = cell->get_fe().unit_support_point(i)[0];
                    const double t =
                      vein ?
                        reference.end - xi * (reference.end - reference.begin) :
                        reference.begin +
                          xi * (reference.end - reference.begin);
                    const double radius = (vein ? .004 : .003) - .0005 * t;
                    EXPECT_NEAR(state(indices[i]),
                                numbers::PI * radius * radius,
                                1.e-14);
                  }
            }
        EXPECT_EQ(Utilities::MPI::sum(inlet, comm), 1u);
        EXPECT_EQ(Utilities::MPI::sum(venous_outlet, comm), 1u);
        EXPECT_EQ(flow.triangulation().n_global_active_cells(),
                  Utilities::MPI::sum(
                    std::uint64_t(result.discretization.cells.size()), comm));
      }
  }
} // namespace

TEST(MetricFlowX, BOTH_PreparesVascularInputWithNativeBeds)
{
  exercise_native_transfer<MetricFlowX::BloodFlowSystem<1, 3>>();
}
#endif
