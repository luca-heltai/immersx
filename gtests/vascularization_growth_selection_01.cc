// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/distribution.h>
#include <immersx/vascularization/generation.h>

#include "../source/vascularization/growth_selection.h"

using namespace dealii;
using namespace ImmersX::Vascularization;

TEST(VascularForest, BOTH_AcceptedSelectionMatchesFullNativeGeometryVote)
{
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  for (unsigned int cells : {1u, 4u})
    for (unsigned int frozen : {0u, 64u})
      {
        parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
        GridGenerator::subdivided_hyper_rectangle(bg,
                                                  {cells, 1, 1},
                                                  Point<3>(),
                                                  Point<3>(1., 1., 1.));
        MappingQ1<3> mapping;
        NetworkData  input;
        if (rank == 0)
          {
            input.nodes[0] = {{.2, .5, .5}, 0, Port::arterial_inlet};
            input.nodes[1] = {{.45, .5, .5}, 0};
            Vessel vessel;
            vessel.nodes       = {{0, 1}};
            vessel.radii       = {{.003, .003}};
            vessel.viscosity   = .004;
            vessel.provenance  = Provenance::generated;
            input.vessels[0]   = vessel;
            input.terminals[0] = {1, 0, System::arterial, 0, 1.};
            for (unsigned int i = 0; i < frozen; ++i)
              {
                const Id a = 1000 + 2 * i, b = a + 1;
                input.nodes[a]          = {{3. + .1 * i, 2., .5}, 0};
                input.nodes[b]          = {{3.05 + .1 * i, 2., .5}, 0};
                vessel.nodes            = {{a, b}};
                vessel.provenance       = Provenance::imported;
                input.vessels[1000 + i] = vessel;
              }
          }
        auto native = redistribute(bg, mapping, input, 0., MPI_COMM_WORLD);
        std::set<Id>         mutable_owned;
        std::map<Id, double> roots;
        if (native.vessels.count(0) && native.vessels.at(0).owner == rank)
          {
            mutable_owned.insert(0);
            roots.emplace(0, .003);
          }
        Parameters p("/Native accepted selector/");
        p.length_scale   = "meter";
        p.minimum_length = 0.;
        p.minimum_angle  = 15.;
        GrowthWork                                             work;
        ImmersX::Vascularization::internal::BackgroundGeometry geometry(bg,
                                                                        mapping,
                                                                        &work);
        auto initial = ImmersX::Vascularization::internal::commit_growth_splits(
          bg,
          mapping,
          native,
          mutable_owned,
          roots,
          {},
          p,
          MPI_COMM_WORLD,
          &geometry);
        ASSERT_TRUE(initial.committed);
        using namespace ImmersX::Vascularization::internal;
        GrowthGeometrySegments segments;
        for (const auto &[id, v] : initial.network.vessels)
          if (v.owner == rank)
            segments.emplace(
              id,
              Segment<3>{initial.network.nodes.at(v.nodes[0]).position,
                         initial.network.nodes.at(v.nodes[1]).position});
        const auto measures =
          geometry.measure_segments(segments, MPI_COMM_WORLD);
        GrowthExecution execution(
          initial.network,
          initial.physiology,
          p,
          1,
          geometry,
          measures,
          [&](const GrowthGeometrySegments &query) {
            return geometry.measure_segments(query, MPI_COMM_WORLD);
          },
          [&](const GrowthGeometrySegments &query) {
            geometry.release_profiles(query);
          },
          0.,
          MPI_COMM_WORLD,
          &work);
        ASSERT_TRUE(execution.validation().valid());
        const auto                   accepted = execution.accepted_snapshot();
        std::map<Id, const Vessel *> addresses;
        for (const auto &[id, vessel] : accepted.network().vessels)
          addresses.emplace(id, &vessel);
        const auto owner =
          Utilities::MPI::min(initial.mutable_owned.count(0) ? rank : size,
                              MPI_COMM_WORLD);
        for (unsigned int scenario = 0; scenario < 3; ++scenario)
          {
            std::vector<BatchProposal> proposals;
            if (rank == size - 1)
              {
                const Point<3> junction(.32, .5, .5);
                const Point<3> tip(.32, scenario == 1 ? 1.1 : .7, .5);
                BatchProposal  proposal;
                proposal.id   = 20;
                proposal.cost = 1.;
                proposal.owners.emplace(0, owner);
                proposal.old_nodes.emplace(
                  0, std::array<Id, 2>{{scenario == 2 ? Id{7} : Id{0}, 1}});
                proposal.changed = {
                  {0, {{0, 10}}, {{Point<3>(.2, .5, .5), junction}}, .003},
                  {10, {{10, 1}}, {{junction, Point<3>(.45, .5, .5)}}, .002},
                  {11, {{10, 11}}, {{junction, tip}}, .002}};
                proposals.push_back(proposal);
              }
            const auto before = work;
            const auto sparse = select_growth_batch(
              geometry, accepted, proposals, p, MPI_COMM_WORLD);
            const auto sparse_work = work - before;
            EXPECT_EQ(Utilities::MPI::sum(
                        sparse_work[GrowthOperation::physical_copy_attempts],
                        MPI_COMM_WORLD),
                      3u);
            EXPECT_EQ(Utilities::MPI::sum(
                        sparse_work[GrowthOperation::validation_record_visits],
                        MPI_COMM_WORLD),
                      3u);
            const auto full =
              select_growth_batch(geometry,
                                  initial.network,
                                  proposals,
                                  p,
                                  MPI_COMM_WORLD,
                                  [&](Id id) {
                                    const auto &v =
                                      initial.network.vessels.at(id);
                                    return v.radii;
                                  });
            EXPECT_EQ(sparse.local, full.local);
            EXPECT_EQ(sparse.relevant, full.relevant);
            if (rank == size - 1)
              {
                EXPECT_EQ(sparse.local.at(20), scenario == 0);
              }
            EXPECT_TRUE(accepted.current());
            for (const auto &[id, address] : addresses)
              EXPECT_EQ(&accepted.network().vessels.at(id), address);
          }
      }
}
