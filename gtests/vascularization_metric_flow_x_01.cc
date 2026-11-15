// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/metric_flow_x.h>

#include <limits>
#include <set>

#include "../apps/metric_flow_x_physical_input.h"
#include "test_paths.h"

#if defined(IMMERSX_WITH_METRIC_FLOW_X) && defined(DEAL_II_WITH_SUNDIALS)
namespace
{
  using namespace dealii;
  using namespace ImmersX::Vascularization;
  constexpr Id base = Id(1) << 60;

  template <class Flow>
  void
  exercise_native_transfer(const bool three = false)
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
            if (three)
              {
                result.network.nodes[base + 8] = {Point<3>(.1, .5, .5),
                                                  last,
                                                  Port::portal_inlet};
                result.network.nodes[base + 9] = {Point<3>(.9, .5, .5), last};
                Vessel portal                  = artery;
                portal.nodes                   = {{base + 8, base + 9}};
                portal.owner                   = last;
                portal.system                  = System::portal;
                result.network.vessels[base + 15]   = portal;
                result.network.terminals[base + 20] = {
                  base + 9, base + 21, System::portal, last, 1.};
                if (rank == 0)
                  result.terminal_triplets.push_back({base + 21,
                                                      base + 17,
                                                      base + 20,
                                                      base + 19,
                                                      {{1.e9, 2.e9, 3.e9}}});
              }
            if (rank == 0 && !three)
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
        unsigned int vessel_calls = 0, boundary_calls = 0, portal_calls = 0;
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
            if (node.port == Port::portal_inlet)
              {
                EXPECT_EQ(terminal, nullptr);
                ++portal_calls;
                return RCR{0., 1.e8, 0., 5000., "portal"};
              }
            return RCR{0., 1.e8, 0., 0., "synthetic"};
          });
        EXPECT_EQ(Utilities::MPI::sum(vessel_calls, comm), three ? 3u : 2u);
        EXPECT_EQ(Utilities::MPI::sum(boundary_calls, comm), three ? 5u : 3u);
        ASSERT_EQ(transfer.vessel_ids.size(), three ? 3u : 2u);
        ASSERT_EQ(transfer.terminal_ids.size(), three ? 3u : 2u);
        ASSERT_EQ(transfer.reversed_vessels.size(), 1u);
        EXPECT_EQ(transfer.reversed_vessels.count(base + 13), 1u);
        EXPECT_EQ(transfer.boundary_ids.at(base + 1), 0u);
        EXPECT_EQ(Utilities::MPI::sum(portal_calls, comm), three ? 1u : 0u);
        if (three)
          {
            EXPECT_NE(transfer.boundary_ids.at(base + 8), 0u);
            EXPECT_NE(transfer.boundary_ids.at(base + 8),
                      transfer.boundary_ids.at(base + 5));
            EXPECT_FALSE(transfer.reversed_vessels.count(base + 15));
          }
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
        for (const auto &cell : flow.triangulation().active_cell_iterators())
          if (!cell->is_artificial())
            {
              const auto interval =
                physical_reference(cell->id(), transfer.flow_cells);
              if (interval.begin != 0. && cell->face(0)->at_boundary())
                {
                  EXPECT_EQ(cell->face(0)->boundary_id(), 255u);
                }
              if (interval.end != 1. && cell->face(1)->at_boundary())
                {
                  EXPECT_EQ(cell->face(1)->boundary_id(), 255u);
                }
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
              const bool portal    = reference.vessel == base + 15;
              EXPECT_TRUE(vein || reference.vessel == base + 11 ||
                          (three && portal));
              const Point<3> a = vein   ? Point<3>(.9, .7, .5) :
                                 portal ? Point<3>(.1, .5, .5) :
                                          Point<3>(.1, .3, .5);
              const Point<3> b = vein   ? Point<3>(.1, .7, .5) :
                                 portal ? Point<3>(.9, .5, .5) :
                                          Point<3>(.9, .3, .5);
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
  template <class Flow>
  void
  exercise_sealed_transfer()
  {
    if constexpr (!has_metric_flow_x_network_input<Flow>)
      GTEST_SKIP() << "Installed MetricFlowX has no in-memory input API.";
    else
      {
        using Physics = typename Flow::NetworkPhysics;
        if constexpr (!ImmersX::Vascularization::internal::HasSealedTerminals<
                        Physics>::value)
          GTEST_SKIP() << "Installed MetricFlowX has no sealed-terminal API.";
        else
          {
            const auto comm = MPI_COMM_WORLD;
            const auto rank = Utilities::MPI::this_mpi_process(comm);
            const auto last = Utilities::MPI::n_mpi_processes(comm) - 1;
            ParameterAcceptor::clear();
            parallel::distributed::Triangulation<3> background(comm);
            GridGenerator::hyper_cube(background);
            background.refine_global(1);
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
                artery.nodes                      = {{base + 1, base + 3}};
                artery.radii                      = {{.003, .0025}};
                artery.viscosity                  = .004;
                vein.nodes                        = {{base + 5, base + 7}};
                vein.radii                        = {{.004, .0035}};
                vein.viscosity                    = .004;
                vein.owner                        = last;
                vein.system                       = System::venous;
                result.network.vessels[base + 11] = artery;
                result.network.vessels[base + 13] = vein;
                result.network.terminals[Id(0)]   = {
                  base + 3, invalid_id, System::arterial, 0, 1.};
                result.network.terminals[base + 19] = {
                  base + 7, invalid_id, System::venous, last, 1.};
              }
            result.validation =
              validate_data(result.network,
                            Utilities::MPI::n_mpi_processes(comm));
            ASSERT_TRUE(result.validation.valid());
            result.discretization = discretize(background,
                                               MappingQ1<3>(),
                                               result.network,
                                               DiscretizationParameters{},
                                               comm);
            ASSERT_TRUE(result.discretization.validation.valid());
            Flow flow(comm);
            flow.initialize_params(
              ImmersX::TestPaths::parameter_path(
                "gtests/parameters/vascularization_metric_flow_x.prm"),
              ImmersX::TestPaths::output_path(
                "vascular-metric-flow-x/sealed_last_used_parameters.prm")
                .string());
            const auto material = [](Id, const Vessel &) {
              typename Flow::VesselProperties properties;
              properties.E      = 5.e5;
              properties.h_wall = .0003;
              return properties;
            };
            unsigned int boundary_calls = 0;
            const auto   boundary =
              [&](Id, const Node &, const Terminal *terminal) {
                ++boundary_calls;
                EXPECT_EQ(terminal, nullptr);
                using RCR = typename decltype(Physics{}.terminals)::mapped_type;
                return RCR{0., 1.e8, 0., 0., "real-port"};
              };
            std::set<Id> sealed;
            // Deliberately request each end on a rank other than its owner.
            if (rank == last)
              sealed.insert(Id(0));
            if (rank == 0)
              sealed.insert(base + 19);
            if (rank == 0)
              result.terminal_pairs.push_back(
                {base + 21, Id(0), base + 19, 1.e9});
            EXPECT_ANY_THROW(
              prepare_metric_flow_x(flow, result, material, boundary, sealed));
            EXPECT_EQ(flow.triangulation().n_global_active_cells(), 0u);
            result.terminal_pairs.clear();
            auto unknown = sealed;
            if (rank == 0)
              unknown.insert(base + 99);
            EXPECT_ANY_THROW(
              prepare_metric_flow_x(flow, result, material, boundary, unknown));
            EXPECT_EQ(flow.triangulation().n_global_active_cells(), 0u);
            boundary_calls = 0;
            const auto transfer =
              prepare_metric_flow_x(flow, result, material, boundary, sealed);
            EXPECT_EQ(Utilities::MPI::sum(boundary_calls, comm), 1u);
            unsigned int artery_ends = 0, vein_ends = 0;
            for (const auto &cell :
                 flow.triangulation().active_cell_iterators())
              if (cell->is_locally_owned())
                for (unsigned int f = 0; f < 2; ++f)
                  if (cell->face(f)->at_boundary())
                    {
                      const auto id = cell->face(f)->boundary_id();
                      if (id == transfer.terminal_ids.at(Id(0)))
                        {
                          EXPECT_EQ(f, 1u);
                          ++artery_ends;
                        }
                      else if (id == transfer.terminal_ids.at(base + 19))
                        {
                          EXPECT_EQ(f, 0u);
                          ++vein_ends;
                        }
                    }
            EXPECT_EQ(Utilities::MPI::sum(artery_ends, comm), 1u);
            EXPECT_EQ(Utilities::MPI::sum(vein_ends, comm), 1u);
            auto state = flow.make_state(), rate = flow.make_state(),
                 residual = flow.make_state();
            flow.initialize_consistent_state(state, rate, .17);
            EXPECT_GT(rate.l2_norm(), 0.);
            flow.assemble_residual(.17, state, rate, residual);
            flow.assemble_state_jacobian(.17, state, rate);
            flow.assemble_derivative_jacobian(.17, state, rate);
            EXPECT_TRUE(std::isfinite(residual.l2_norm()));
          }
      }
  }

  // Inspect the actual native coefficient input without solving a transient.
  struct RecordedNativeFlow : MetricFlowX::BloodFlowSystem<1, 3>
  {
    using MetricFlowX::BloodFlowSystem<1, 3>::BloodFlowSystem;
    NetworkPhysics physics;
    void
    setup(const NetworkPhysics &input)
    {
      physics = input;
    }
  };

  void
  exercise_triplet_resistance_factors()
  {
    using namespace ImmersX::metric_flow_x_app_detail;
    const auto comm = MPI_COMM_WORLD;
    const auto rank = Utilities::MPI::this_mpi_process(comm);
    const auto last = Utilities::MPI::n_mpi_processes(comm) - 1;
    ParameterAcceptor::clear();
    parallel::distributed::Triangulation<3> background(comm);
    GridGenerator::hyper_cube(background);
    background.refine_global(1);
    GenerationResult generation;
    if (rank == 0)
      {
        for (unsigned int i = 0; i < 3; ++i)
          {
            const auto system               = i == 0 ? System::arterial :
                                              i == 1 ? System::portal :
                                                       System::venous;
            const auto port                 = i == 0 ? Port::arterial_inlet :
                                              i == 1 ? Port::portal_inlet :
                                                       Port::venous_outlet;
            generation.network.nodes[2 * i] = {Point<3>(.1, .2 + .2 * i, .5),
                                               0,
                                               port};
            generation.network.nodes[2 * i + 1] = {
              Point<3>(.9, .2 + .2 * i, .5), 0};
            Vessel vessel;
            vessel.nodes                    = {{2 * i, 2 * i + 1}};
            vessel.radii                    = {{.003, .0025}};
            vessel.viscosity                = .004;
            vessel.system                   = system;
            generation.network.vessels[i]   = vessel;
            generation.network.terminals[i] = {2 * i + 1, 7, system, 0, 1.};
          }
        generation.terminal_triplets.push_back(
          {7, 0, 1, 2, {{1.e9, 2.e9, 3.e9}}});
      }
    generation.validation =
      validate_data(generation.network, Utilities::MPI::n_mpi_processes(comm));
    generation.discretization = discretize(background,
                                           MappingQ1<3>(),
                                           generation.network,
                                           DiscretizationParameters{},
                                           comm);
    ASSERT_TRUE(generation.validation.valid());
    ASSERT_TRUE(generation.discretization.validation.valid());
    ArchivedFlowParameters parameters;
    parameters.young_moduli        = {4.e5, 4.e5, 4.e5};
    parameters.thickness_ratios    = {.1, .1, .1};
    parameters.reference_pressures = {1000., 100., 900.};
    parameters.portal_pressure     = 900.;
    parameters.venous_pressure     = 100.;
    parameters.root_resistance     = 1.e8;
    auto material                  = [](Id, const Vessel &) {
      RecordedNativeFlow::VesselProperties p;
      p.E      = 4.e5;
      p.h_wall = .0003;
      return p;
    };
    auto boundary = [](Id, const Node &, const Terminal *) {
      return RecordedNativeFlow::RCRPhysics{0., 1.e8, 2.e-12, 100., "test"};
    };
    std::vector<RecordedNativeFlow::TerminalConnection> default_links;
    for (const auto &factors : {std::array<double, 3>{{1., 1., 1.}},
                                std::array<double, 3>{{2., 3., 5.}},
                                std::array<double, 3>{{2., 3., 5.}}})
      {
        RecordedNativeFlow flow(comm);
        const auto         transfer = prepare_metric_flow_x(
          flow, generation, material, boundary, {}, factors);
        const std::array<double, 3> pressures{{1200., 800., 100.}};
        const std::array<double, 3> arms{
          {1.e9 * factors[0], 2.e9 * factors[1], 3.e9 * factors[2]}};
        const double pc = (pressures[0] / arms[0] + pressures[1] / arms[1] +
                           pressures[2] / arms[2]) /
                          (1. / arms[0] + 1. / arms[1] + 1. / arms[2]);
        std::array<double, 3> native_flows{};
        for (const auto &link : flow.physics.terminal_connections)
          {
            unsigned int a = 0, b = 0;
            while (transfer.terminal_ids.at(a) != link.boundary0)
              ++a;
            while (transfer.terminal_ids.at(b) != link.boundary1)
              ++b;
            const double q = (pressures[a] - pressures[b]) / link.resistance;
            native_flows[a] += q;
            native_flows[b] -= q;
          }
        for (unsigned int i = 0; i < 3; ++i)
          EXPECT_NEAR(native_flows[i], (pressures[i] - pc) / arms[i], 1.e-20);
        for (const auto &[id, rcr] : flow.physics.terminals)
          EXPECT_DOUBLE_EQ(rcr.C, 2.e-12);
        if (factors[0] == 1.)
          default_links = flow.physics.terminal_connections;
        if (rank == 0)
          {
            EXPECT_EQ(generation.terminal_triplets[0].resistances,
                      (std::array<double, 3>{{1.e9, 2.e9, 3.e9}}));
          }
      }
    // The omitted argument is exactly the original identity transfer.
    {
      RecordedNativeFlow flow(comm);
      prepare_metric_flow_x(flow, generation, material, boundary);
      ASSERT_EQ(flow.physics.terminal_connections.size(), default_links.size());
      for (unsigned int i = 0; i < default_links.size(); ++i)
        EXPECT_DOUBLE_EQ(flow.physics.terminal_connections[i].resistance,
                         default_links[i].resistance);
    }
    // Both applications call the same archived-input function and parameter.
    {
      RecordedNativeFlow flow(comm);
      parameters.triplet_resistance_factors = {{2., 3., 5.}};
      prepare_archived_flow(flow, generation, parameters);
      ASSERT_EQ(flow.physics.terminal_connections.size(), 3u);
      EXPECT_NEAR(flow.physics.terminal_connections[0].resistance,
                  8.8e9,
                  1.e-5);
    }
    for (const double bad : {0.,
                             -1.,
                             std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::max()})
      {
        RecordedNativeFlow    flow(comm);
        std::array<double, 3> factors{{1., 1., 1.}};
        // Invalid controls on one rank and overflow on an owning rank must
        // fail collectively before any mesh construction.
        if (rank == last || bad == std::numeric_limits<double>::max())
          factors[0] = bad;
        EXPECT_ANY_THROW(prepare_metric_flow_x(
          flow, generation, material, boundary, {}, factors));
        EXPECT_EQ(flow.triangulation().n_global_active_cells(), 0u);
      }
    if (last > 0)
      {
        RecordedNativeFlow          flow(comm);
        const std::array<double, 3> factors{{rank == last ? 2. : 1., 1., 1.}};
        EXPECT_ANY_THROW(prepare_metric_flow_x(
          flow, generation, material, boundary, {}, factors));
      }
    if (rank == 0)
      generation.terminal_triplets[0].resistances[0] =
        std::numeric_limits<double>::denorm_min();
    {
      RecordedNativeFlow flow(comm);
      EXPECT_ANY_THROW(prepare_metric_flow_x(
        flow, generation, material, boundary, {}, {{.5, 1., 1.}}));
    }
  }
} // namespace

TEST(MetricFlowX, BOTH_ScalesArchivedTripletResistances)
{
  exercise_triplet_resistance_factors();
}

TEST(MetricFlowX, BOTH_PreparesVascularInputWithNativeBeds)
{
  exercise_native_transfer<MetricFlowX::BloodFlowSystem<1, 3>>();
}
TEST(MetricFlowX,
     BOTH_PreparesThreeCompartmentInputWithIndependentPortalBoundary)
{
  exercise_native_transfer<MetricFlowX::BloodFlowSystem<1, 3>>(true);
}
TEST(MetricFlowX, BOTH_PreparesExplicitSealedInputTerminals)
{
  exercise_sealed_transfer<MetricFlowX::BloodFlowSystem<1, 3>>();
}
#endif
