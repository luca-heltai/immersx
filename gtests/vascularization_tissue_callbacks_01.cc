// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>

#include "../source/vascularization/growth_geometry.h"
#include "../source/vascularization/tissue_growth_callbacks.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
using namespace ImmersX::Vascularization::internal;
namespace
{
  double
  callback_mass(const std::vector<TerminalDemandUpdate> &values, Id id)
  {
    double value = 0.;
    for (const auto &entry : values)
      if (entry.terminal == id)
        value = entry.demand;
    return Utilities::MPI::sum(value, MPI_COMM_WORLD);
  }
  NetworkData
  callback_seed(unsigned int owner)
  {
    NetworkData network;
    if (owner == Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
      {
        network.nodes[100]     = {{.25, .5, .5},
                                  owner,
                                  Port::none,
                                  Provenance::generated};
        network.terminals[101] = {100, 102, System::arterial, owner, 1.};
      }
    return network;
  }
} // namespace
TEST(VascularTissueDemand, BOTH_CallbacksBootstrapJointAndRollback)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p("/Tissue callback transaction/");
  p.tissue_demand = "1+3*x";
  auto callbacks  = tissue_growth_callbacks(geometry, p, 3, MPI_COMM_WORLD);
  const auto         rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto         size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const unsigned int seed_owner = size - 1;
  auto               network    = callback_seed(seed_owner);
  std::set<Id>       leaves;
  if (rank == seed_owner)
    leaves.insert(101);
  const auto initial = callbacks.bootstrap(network, leaves, 7);
  ASSERT_TRUE(initial.validation.valid());
  ASSERT_TRUE(
    initialize_growth_demands(network, leaves, initial, 7, MPI_COMM_WORLD)
      .valid());
  if (rank == seed_owner)
    {
      EXPECT_NEAR(network.terminals.at(101).demand, 2.5, 1.e-13);
    }
  callbacks.prepare_adopt(initial.update.quadrature_revision, 8);
  callbacks.adopt();
  EXPECT_THROW(callbacks.joint({}, 7), ExceptionBase);

  const GrowthProposalKey left{0, 201}, right{size - 1, 301};
  const auto first  = callbacks.preview(left, 201, 202, {.5, .5, .5}, 8);
  const auto second = callbacks.preview(right, 301, 302, {.75, .5, .5}, 8);
  ASSERT_TRUE(first.validation.valid());
  ASSERT_TRUE(second.validation.valid());
  const double isolated = Utilities::MPI::sum(first.new_demand, MPI_COMM_WORLD);
  std::vector<GrowthProposalKey> selected;
  if (rank == left.first)
    selected.push_back(left);
  if (rank == right.first)
    selected.push_back(right);
  const auto joint = callbacks.joint(selected, 8);
  ASSERT_TRUE(joint.validation.valid());
  const double middle = callback_mass(joint.new_terminals, 201);
  const double outer  = callback_mass(joint.new_terminals, 301);
  const double old    = callback_mass(joint.update.terminals, 101);
  EXPECT_GT(middle, 0.);
  EXPECT_GT(outer, 0.);
  EXPECT_LT(middle, isolated);
  EXPECT_NEAR(old + middle + outer, 2.5, 1.e-13);
  callbacks.prepare_adopt(joint.update.quadrature_revision, 9);
  EXPECT_THROW(callbacks.preview({0, 401}, 401, 402, {.9, .5, .5}, 8),
               ExceptionBase);
  callbacks.discard();
  const auto replay_first = callbacks.preview(left, 201, 202, {.5, .5, .5}, 8);
  ASSERT_TRUE(replay_first.validation.valid());
  EXPECT_EQ(Utilities::MPI::sum(replay_first.new_demand, MPI_COMM_WORLD),
            isolated);
  callbacks.preview(right, 301, 302, {.75, .5, .5}, 8);
  const auto replay = callbacks.joint(selected, 8);
  ASSERT_TRUE(replay.validation.valid());
  EXPECT_EQ(callback_mass(replay.new_terminals, 201), middle);
  EXPECT_EQ(callback_mass(replay.new_terminals, 301), outer);
  callbacks.prepare_adopt(replay.update.quadrature_revision, 9);
  callbacks.adopt();
  const auto next = callbacks.preview({0, 401}, 401, 402, {.9, .5, .5}, 9);
  ASSERT_TRUE(next.validation.valid());
  EXPECT_GT(Utilities::MPI::sum(next.new_demand, MPI_COMM_WORLD), 0.);
  callbacks.discard();
}
TEST(VascularTissueDemand, BOTH_CallbacksBootstrapBudgetHasNoPhysicalMutation)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p("/Tissue callback budget/");
  p.tissue_integration.maximum_points = 8;
  auto callbacks = tissue_growth_callbacks(geometry, p, 10, MPI_COMM_WORLD);
  auto network   = callback_seed(0);
  std::set<Id> leaves;
  const auto   rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  if (rank == 0)
    leaves.insert(101);
  const auto failed = callbacks.bootstrap(network, leaves, 0);
  EXPECT_FALSE(failed.validation.valid());
  EXPECT_TRUE(failed.integration_budget_exhausted);
  EXPECT_FALSE(
    initialize_growth_demands(network, leaves, failed, 0, MPI_COMM_WORLD)
      .valid());
  if (rank == 0)
    {
      EXPECT_EQ(network.terminals.at(101).demand, 1.);
    }
  callbacks.discard();
}
TEST(VascularTissueDemand, BOTH_CallbacksExhaustedPreviewCanBeDiscarded)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  Parameters         p("/Tissue callback retry/");
  p.tissue_integration.quadrature_order     = 1;
  p.tissue_integration.samples_per_terminal = 8;
  p.tissue_integration.maximum_points       = 8;
  auto callbacks = tissue_growth_callbacks(geometry, p, 1, MPI_COMM_WORLD);
  auto network   = callback_seed(0);
  std::set<Id> leaves;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    leaves.insert(101);
  const auto initial = callbacks.bootstrap(network, leaves, 0);
  ASSERT_TRUE(initial.validation.valid());
  callbacks.prepare_adopt(initial.update.quadrature_revision, 1);
  callbacks.adopt();
  const auto unresolved =
    callbacks.preview({0, 201}, 201, 202, {.25, .5, .5}, 1);
  EXPECT_FALSE(unresolved.validation.valid());
  EXPECT_TRUE(unresolved.integration_budget_exhausted);
  EXPECT_THROW(callbacks.joint({{0, 201}}, 1), ExceptionBase);
  callbacks.discard();
  const auto resolved = callbacks.preview({0, 201}, 201, 202, {.75, .5, .5}, 1);
  ASSERT_TRUE(resolved.validation.valid());
  EXPECT_FALSE(resolved.integration_budget_exhausted);
  EXPECT_NEAR(Utilities::MPI::sum(resolved.new_demand, MPI_COMM_WORLD),
              .5,
              1.e-14);
  const auto joint = callbacks.joint({{0, 201}}, 1);
  ASSERT_TRUE(joint.validation.valid());
  EXPECT_NEAR(callback_mass(joint.update.terminals, 101), .5, 1.e-14);
  callbacks.prepare_adopt(joint.update.quadrature_revision, 2);
  callbacks.adopt();
}
