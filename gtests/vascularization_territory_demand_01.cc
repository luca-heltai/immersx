// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/mpi.h>

#include <gtest/gtest.h>

#include <cmath>

#include "../source/vascularization/territory_demand.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
using namespace ImmersX::Vascularization::internal;
namespace
{
  constexpr Id base = (Id{1} << 59) + 133;
  unsigned int
  rank()
  {
    return Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  }
  unsigned int
  size()
  {
    return Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  }
  unsigned int
  active()
  {
    return size() > 2 ? size() - 1 : size();
  }
  TissueCentre
  centre(Id offset, double x, unsigned int owner)
  {
    return {base + offset, base + 1000 + offset, Point<3>(x, .5, .5), owner};
  }
  std::vector<TissueCentre>
  local(const std::vector<TissueCentre> &centres)
  {
    std::vector<TissueCentre> result;
    for (const auto &c : centres)
      if (c.owner == rank())
        result.push_back(c);
    return result;
  }
  std::map<Id, TissueDemandSample>
  samples(unsigned int count, Id offset = 0)
  {
    std::map<Id, TissueDemandSample> result;
    for (unsigned int i = 0; i < count; ++i)
      if (i % active() == rank())
        result.emplace(offset + i,
                       TissueDemandSample{Point<3>((i + .5) / count, .5, .5),
                                          1. / count});
    return result;
  }
  std::vector<TissueCentre>
  initial_centres()
  {
    return {centre(1, .25, 0), centre(2, .75, 1 % active())};
  }
  double
  demand(const std::vector<TerminalDemandUpdate> &values, Id terminal)
  {
    double own = 0.;
    for (const auto &d : values)
      if (d.terminal == terminal)
        own = d.demand;
    return Utilities::MPI::sum(own, MPI_COMM_WORLD);
  }
  void
  bootstrap(TerritoryDemand &state)
  {
    const auto p = state.preview({}, 7);
    ASSERT_TRUE(p.validation.valid());
    state.prepare_adopt(p);
    state.adopt();
  }
  void
  oracle(const TerritoryDemand                  &state,
         const std::map<Id, TissueDemandSample> &points,
         const std::vector<TissueCentre>        &centres)
  {
    const auto assignments = state.owned_assignments();
    ASSERT_EQ(assignments.size(), points.size());
    for (const auto &[id, point] : points)
      {
        auto best = centres.front();
        for (const auto &c : centres)
          if (std::make_pair(point.position.distance_square(c.position),
                             c.territory) <
              std::make_pair(point.position.distance_square(best.position),
                             best.territory))
            best = c;
        EXPECT_EQ(assignments.at(id), best.territory);
      }
  }
} // namespace
TEST(VascularTissueDemand,
     BOTH_TerritoryPreviewIsJointAndLeavesAcceptedStateUnchanged)
{
  const auto      initial = initial_centres();
  const auto      points  = samples(16);
  TerritoryDemand state(points, local(initial), MPI_COMM_WORLD);
  bootstrap(state);
  const auto before = state.owned_assignments();
  const auto a = centre(3, .5, 1 % active()), b = centre(4, .625, 0);
  const auto isolated = state.preview(local({a}), 11);
  ASSERT_TRUE(isolated.validation.valid());
  EXPECT_DOUBLE_EQ(demand(isolated.demand.new_terminals, a.terminal), .25);
  EXPECT_DOUBLE_EQ(demand(isolated.demand.update.terminals,
                          initial[0].terminal),
                   .375);
  EXPECT_DOUBLE_EQ(demand(isolated.demand.update.terminals,
                          initial[1].terminal),
                   .375);
  const auto joint = state.preview(local({a, b}), 11);
  ASSERT_TRUE(joint.validation.valid());
  EXPECT_DOUBLE_EQ(demand(joint.demand.new_terminals, a.terminal), 3. / 16.);
  EXPECT_DOUBLE_EQ(demand(joint.demand.new_terminals, b.terminal), 2. / 16.);
  EXPECT_DOUBLE_EQ(demand(joint.demand.update.terminals, initial[1].terminal),
                   5. / 16.);
  EXPECT_EQ(state.owned_assignments(), before);
  EXPECT_DOUBLE_EQ(state.total_demand(), 1.);
  state.prepare_adopt(joint);
  EXPECT_THROW(state.owned_assignments(), ExceptionBase);
  EXPECT_THROW(state.preview({}, 12), ExceptionBase);
  state.discard();
  EXPECT_EQ(state.owned_assignments(), before);
  const auto replay = state.preview(local({a, b}), 11);
  ASSERT_TRUE(replay.validation.valid());
  EXPECT_DOUBLE_EQ(demand(replay.demand.new_terminals, a.terminal), 3. / 16.);
  state.prepare_adopt(replay);
  state.adopt();
  oracle(state, points, {initial[0], initial[1], a, b});
  EXPECT_THROW(state.prepare_adopt(isolated), ExceptionBase);
  EXPECT_DOUBLE_EQ(state.total_demand(), 1.);
}
TEST(VascularTissueDemand,
     BOTH_QuadratureReplacementHasItsOwnRevisionAndRollsBack)
{
  const auto      initial = initial_centres();
  const auto      points  = samples(16);
  TerritoryDemand state(points, local(initial), MPI_COMM_WORLD);
  bootstrap(state);
  TissueQuadratureChanges changes;
  for (const auto &[id, sample] : points)
    changes.removed.push_back(id);
  changes.added      = samples(32, 100);
  const auto preview = state.preview({}, 22, changes);
  ASSERT_TRUE(preview.validation.valid());
  EXPECT_EQ(preview.demand.update.base_quadrature_revision, 0u);
  EXPECT_EQ(preview.demand.update.quadrature_revision, 1u);
  EXPECT_DOUBLE_EQ((*preview.demand.update.canonical_total_demand)[0], 1.);
  state.prepare_adopt(preview);
  state.discard();
  EXPECT_EQ(state.quadrature_revision(), 0u);
  oracle(state, points, initial);
  state.prepare_adopt(preview);
  state.adopt();
  EXPECT_EQ(state.quadrature_revision(), 1u);
  oracle(state, changes.added, initial);
  TissueQuadratureChanges revised;
  for (const auto &[id, point] : changes.added)
    {
      revised.removed.push_back(id);
      revised.added.emplace(id,
                            TissueDemandSample{point.position,
                                               2. * point.mass});
    }
  const auto update = state.preview({}, 23, revised);
  ASSERT_TRUE(update.validation.valid());
  EXPECT_DOUBLE_EQ((*update.demand.update.canonical_total_demand)[0], 2.);
  state.prepare_adopt(update);
  state.adopt();
  EXPECT_DOUBLE_EQ(state.total_demand(), 2.);
  EXPECT_DOUBLE_EQ(demand(state.owned_demands(), initial[0].terminal), 1.);
}
TEST(VascularTissueDemand,
     BOTH_RefiningQuadratureResolvesAZeroMassTerritoryWithoutAFloor)
{
  const auto                       initial = initial_centres();
  std::map<Id, TissueDemandSample> points;
  for (unsigned int i = 0; i < 2; ++i)
    if (i % active() == rank())
      points.emplace(i, TissueDemandSample{initial[i].position, .5});
  TerritoryDemand state(points, local(initial), MPI_COMM_WORLD);
  bootstrap(state);
  const auto a      = centre(3, .5, 0);
  const auto sparse = state.preview(local({a}), 31);
  EXPECT_FALSE(sparse.validation.valid());
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(
                                  sparse.unresolved.size()),
                                MPI_COMM_WORLD),
            1u);
  EXPECT_THROW(state.prepare_adopt(sparse), ExceptionBase);
  TissueQuadratureChanges changes;
  for (const auto &[id, sample] : points)
    changes.removed.push_back(id);
  changes.added       = samples(16, 100);
  const auto resolved = state.preview(local({a}), 31, changes);
  ASSERT_TRUE(resolved.validation.valid());
  EXPECT_DOUBLE_EQ(demand(resolved.demand.new_terminals, a.terminal), .25);
  state.prepare_adopt(resolved);
  state.adopt();
  EXPECT_DOUBLE_EQ(state.total_demand(), 1.);
  oracle(state, changes.added, {initial[0], initial[1], a});
}
TEST(VascularTissueDemand,
     BOTH_RemovingDominantMassKeepsTheTinyOldTerritoryPositive)
{
  const auto a = centre(1, .25, 0), b = centre(2, .75, 1 % active());
  std::map<Id, TissueDemandSample> points;
  if (rank() == 0)
    points.emplace(0, TissueDemandSample{a.position, 1.e-300});
  if (rank() == 1 % active())
    points.emplace(1, TissueDemandSample{b.position, 1.e300});
  TerritoryDemand state(points, local({a}), MPI_COMM_WORLD);
  bootstrap(state);
  const auto preview = state.preview(local({b}), 41);
  ASSERT_TRUE(preview.validation.valid());
  EXPECT_DOUBLE_EQ(demand(preview.demand.update.terminals, a.terminal),
                   1.e-300);
  EXPECT_DOUBLE_EQ(demand(preview.demand.new_terminals, b.terminal), 1.e300);
  state.prepare_adopt(preview);
  state.adopt();
  EXPECT_DOUBLE_EQ(demand(state.owned_demands(), a.terminal), 1.e-300);
  oracle(state, points, {a, b});
}
TEST(VascularTissueDemand,
     BOTH_OrdinaryPreviewVisitsOnlyTheAffectedSampleEnvelopes)
{
  for (unsigned int count : {32u, 256u, 2048u})
    {
      const auto                points = samples(count);
      std::vector<TissueCentre> centres;
      for (unsigned int i = 0; i < count / 4; ++i)
        centres.push_back(centre(i + 1, (4. * i + 2.) / count, i % active()));
      TerritoryDemand state(points, local(centres), MPI_COMM_WORLD);
      bootstrap(state);
      const auto a       = centre(10000, .5, 0);
      const auto preview = state.preview(local({a}), 51);
      ASSERT_TRUE(preview.validation.valid());
      EXPECT_EQ(Utilities::MPI::sum(preview.work.influence_hits,
                                    MPI_COMM_WORLD),
                2u);
      EXPECT_EQ(Utilities::MPI::sum(preview.work.reassigned_samples,
                                    MPI_COMM_WORLD),
                2u);
      EXPECT_EQ(Utilities::MPI::sum(preview.work.nearest_queries,
                                    MPI_COMM_WORLD),
                0u);
      EXPECT_DOUBLE_EQ(demand(preview.demand.new_terminals, a.terminal),
                       2. / count);
      state.prepare_adopt(preview);
      state.adopt();
      centres.push_back(a);
      oracle(state, points, centres);
    }
}
TEST(VascularTissueDemand, BOTH_NearestTiesUseStableIdsAcrossNativeOwners)
{
  for (unsigned int swap = 0; swap < 2; ++swap)
    {
      const auto left  = centre(20, .25, swap % active());
      const auto right = centre(10, .75, (1 - swap) % active());
      std::map<Id, TissueDemandSample> points;
      if (rank() == 0)
        {
          points.emplace(0, TissueDemandSample{Point<3>(.5, .5, .5), .5});
          points.emplace(1, TissueDemandSample{left.position, .25});
          points.emplace(2, TissueDemandSample{right.position, .25});
        }
      TerritoryDemand state(points, local({left, right}), MPI_COMM_WORLD);
      bootstrap(state);
      EXPECT_DOUBLE_EQ(demand(state.owned_demands(), left.terminal), .25);
      EXPECT_DOUBLE_EQ(demand(state.owned_demands(), right.terminal), .75);
      oracle(state, points, {left, right});
      const auto middle  = centre(5, .5, 0);
      const auto preview = state.preview(local({middle}), 61);
      ASSERT_TRUE(preview.validation.valid());
      EXPECT_DOUBLE_EQ(demand(preview.demand.new_terminals, middle.terminal),
                       .5);
      state.prepare_adopt(preview);
      state.adopt();
      oracle(state, points, {left, right, middle});
    }
}
TEST(VascularTissueDemand,
     MPI_InvalidInputsAndSnapshotRevisionsFailCollectively)
{
  ASSERT_GT(size(), 1u);
  const auto      points  = samples(16);
  const auto      initial = initial_centres();
  TerritoryDemand state(points, local(initial), MPI_COMM_WORLD);
  bootstrap(state);
  const auto before = state.owned_assignments();
  EXPECT_THROW(state.preview({}, rank() == 0 ? 72 : 73), ExceptionBase);
  const auto duplicate = centre(10000, .5 + .01 * rank(), rank());
  EXPECT_THROW(state.preview({duplicate}, 74), ExceptionBase);
  TissueQuadratureChanges malformed;
  if (rank() == size() - 1)
    malformed.added.emplace(10000,
                            TissueDemandSample{Point<3>(.5, .5, .5), -1.});
  EXPECT_THROW(state.preview({}, 75, malformed), ExceptionBase);
  EXPECT_EQ(state.owned_assignments(), before);
  EXPECT_DOUBLE_EQ(state.total_demand(), 1.);
  EXPECT_THROW(TerritoryDemand(points, {duplicate}, MPI_COMM_WORLD),
               ExceptionBase);
}
