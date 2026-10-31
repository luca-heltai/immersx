// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_description.h>

#include <gtest/gtest.h>

#include "../source/vascularization/growth_geometry.h"
#include "../source/vascularization/tissue_integration.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
using namespace ImmersX::Vascularization::internal;
namespace
{
  constexpr Id integration_base = (Id{1} << 58) + 71;
  unsigned int
  integration_rank()
  {
    return Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  }
  std::vector<TissueCentre>
  owned_centres(BackgroundGeometry          &geometry,
                const std::vector<Point<3>> &points,
                Id                           offset = 0)
  {
    std::vector<TissueCentre> result;
    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const auto owner = Utilities::MPI::min(geometry.containing(points[i]) ?
                                                 integration_rank() :
                                                 numbers::invalid_unsigned_int,
                                               MPI_COMM_WORLD);
        if (owner == integration_rank())
          result.push_back({integration_base + offset + i,
                            integration_base + 1000 + offset + i,
                            points[i],
                            owner});
      }
    return result;
  }
  double
  integration_demand(const std::vector<TerminalDemandUpdate> &values,
                     Id                                       terminal)
  {
    double value = 0.;
    for (const auto &d : values)
      if (d.terminal == terminal)
        value = d.demand;
    return Utilities::MPI::sum(value, MPI_COMM_WORLD);
  }
  void
  integration_bootstrap(TissueDemandIntegration &integration)
  {
    const auto preview = integration.preview({}, 1);
    ASSERT_TRUE(preview.validation.valid());
    integration.prepare_adopt(preview);
    integration.adopt();
  }
} // namespace
TEST(VascularTissueDemand, IntegrationDefaultsAndParameterParsing)
{
  Parameters p("/Tissue integration defaults/");
  EXPECT_EQ(p.tissue_integration.quadrature_order, 2u);
  EXPECT_EQ(p.tissue_integration.samples_per_terminal, 64u);
  EXPECT_EQ(p.tissue_integration.maximum_points, 4000000u);
  EXPECT_EQ(p.tissue_integration.refinement_budget, 8u);
  ParameterAcceptor::prm.parse_input_from_string(
    "subsection Tissue integration defaults\n subsection Physiology\n set Demand quadrature order = 3\n set Demand samples per requested terminal = 128\n set Demand integration point budget = 200000\n set Demand integration refinement budget = 4\n end\nend\n");
  EXPECT_EQ(p.tissue_integration.quadrature_order, 3u);
  EXPECT_EQ(p.tissue_integration.samples_per_terminal, 128u);
  EXPECT_EQ(p.tissue_integration.maximum_points, 200000u);
  EXPECT_EQ(p.tissue_integration.refinement_budget, 4u);
  p.tissue_integration.validate();
  p.tissue_integration.maximum_points = 0;
  EXPECT_THROW(p.tissue_integration.validate(), ExceptionBase);
}
TEST(VascularTissueDemand,
     BOTH_IntegrationResolutionFollowsTheRequestWithPhysicalMass)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  bg.refine_global(1);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  const auto         centres = owned_centres(geometry, {{.25, .5, .5}});
  const auto         cells   = bg.n_global_active_cells();
  for (unsigned int count : {1u, 8u, 64u})
    {
      TissueIntegrationParameters p;
      TissueDemandIntegration     integration(
        geometry, "1+3*x+2*y+z", p, count, centres, MPI_COMM_WORLD);
      ASSERT_TRUE(integration.validation().valid());
      EXPECT_EQ(integration.point_count(), 64u * count);
      EXPECT_EQ(
        Utilities::MPI::sum(integration.initialization_work().generated_points,
                            MPI_COMM_WORLD),
        integration.point_count());
      EXPECT_EQ(Utilities::MPI::sum(
                  integration.initialization_work().assignments.nearest_queries,
                  MPI_COMM_WORLD),
                0u);
      EXPECT_NEAR(integration.total_demand(), 4., 2.e-13);
      integration_bootstrap(integration);
      EXPECT_EQ(bg.n_global_active_cells(), cells);
      EXPECT_NEAR(integration_demand(integration.owned_demands(),
                                     integration_base + 1000),
                  4.,
                  2.e-13);
    }
}
TEST(VascularTissueDemand, BOTH_AdaptiveCellQuadratureResolvesZerosAndRollsBack)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  const auto centres = owned_centres(geometry, {{.25, .5, .5}, {.75, .5, .5}});
  const auto added   = owned_centres(geometry, {{.5, .5, .5}}, 10);
  TissueIntegrationParameters p;
  p.quadrature_order     = 1;
  p.samples_per_terminal = 1;
  p.maximum_points       = 1000;
  p.refinement_budget    = 3;
  TissueDemandIntegration integration(
    geometry, "1", p, 2, centres, MPI_COMM_WORLD);
  ASSERT_TRUE(integration.validation().valid());
  EXPECT_EQ(integration.point_count(), 8u);
  integration_bootstrap(integration);
  const auto preview = integration.preview(added, 2);
  ASSERT_TRUE(preview.validation.valid());
  EXPECT_FALSE(preview.integration_budget_exhausted);
  EXPECT_EQ(Utilities::MPI::sum(preview.work.refined_cells, MPI_COMM_WORLD),
            2u);
  EXPECT_EQ(Utilities::MPI::sum(preview.work.generated_points, MPI_COMM_WORLD),
            576u);
  EXPECT_EQ(integration.point_count(), 8u);
  EXPECT_DOUBLE_EQ(integration_demand(preview.territory.demand.new_terminals,
                                      integration_base + 1010),
                   .25);
  integration.prepare_adopt(preview);
  EXPECT_THROW(integration.point_count(), ExceptionBase);
  integration.discard();
  EXPECT_EQ(integration.point_count(), 8u);
  EXPECT_DOUBLE_EQ(integration.total_demand(), 1.);
  const auto replay = integration.preview(added, 2);
  ASSERT_TRUE(replay.validation.valid());
  integration.prepare_adopt(replay);
  integration.adopt();
  EXPECT_EQ(integration.point_count(), 512u);
  EXPECT_EQ(bg.n_global_active_cells(), 1u);
  EXPECT_DOUBLE_EQ(integration_demand(integration.owned_demands(),
                                      integration_base + 1000),
                   .375);
}
TEST(VascularTissueDemand,
     BOTH_IntegrationBudgetExhaustionAllocatesNoForbiddenSamples)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  const auto centres = owned_centres(geometry, {{.25, .5, .5}, {.75, .5, .5}});
  const auto added   = owned_centres(geometry, {{.5, .5, .5}}, 10);
  TissueIntegrationParameters p;
  p.quadrature_order     = 1;
  p.samples_per_terminal = 1;
  p.maximum_points       = 16;
  TissueDemandIntegration integration(
    geometry, "1", p, 2, centres, MPI_COMM_WORLD);
  integration_bootstrap(integration);
  const auto preview = integration.preview(added, 2);
  EXPECT_FALSE(preview.validation.valid());
  EXPECT_TRUE(preview.integration_budget_exhausted);
  EXPECT_EQ(Utilities::MPI::sum(preview.work.generated_points, MPI_COMM_WORLD),
            0u);
  EXPECT_EQ(integration.point_count(), 8u);
  EXPECT_THROW(integration.prepare_adopt(preview), ExceptionBase);
  p.samples_per_terminal = 64;
  TissueDemandIntegration initial_exhausted(
    geometry, "1", p, 2, centres, MPI_COMM_WORLD);
  EXPECT_FALSE(initial_exhausted.validation().valid());
  EXPECT_TRUE(initial_exhausted.integration_budget_exhausted());
  EXPECT_EQ(Utilities::MPI::sum(
              initial_exhausted.initialization_work().generated_points,
              MPI_COMM_WORLD),
            0u);
  EXPECT_EQ(initial_exhausted.point_count(), 0u);
  p.samples_per_terminal = 1;
  p.maximum_points       = 1000;
  p.refinement_budget    = 1;
  TissueDemandIntegration limited(geometry, "1", p, 2, centres, MPI_COMM_WORLD);
  integration_bootstrap(limited);
  const auto exhausted = limited.preview(added, 2);
  EXPECT_FALSE(exhausted.validation.valid());
  EXPECT_TRUE(exhausted.integration_budget_exhausted);
  EXPECT_EQ(Utilities::MPI::sum(exhausted.work.generated_points,
                                MPI_COMM_WORLD),
            64u);
  EXPECT_EQ(limited.point_count(), 8u);
}
TEST(VascularTissueDemand,
     BOTH_AdaptiveIntegrationSupportsTetrahedraAndEmptyRanks)
{
  Triangulation<3> serial;
  GridGenerator::reference_cell(serial, ReferenceCells::Tetrahedron);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<3> bg(MPI_COMM_WORLD);
  bg.create_triangulation(description);
  MappingFE<3>                mapping(FE_SimplexP<3>(1));
  BackgroundGeometry          geometry(bg, mapping);
  const auto                  centres = owned_centres(geometry, {{.1, .1, .1}});
  TissueIntegrationParameters p;
  TissueDemandIntegration     integration(
    geometry, "1+3*x+2*y+z", p, 2, centres, MPI_COMM_WORLD);
  ASSERT_TRUE(integration.validation().valid());
  EXPECT_GE(integration.point_count(), 128u);
  EXPECT_NEAR(integration.total_demand(), 5. / 12., 2.e-13);
  integration_bootstrap(integration);
  const auto added   = owned_centres(geometry, {{.2, .2, .2}}, 10);
  const auto preview = integration.preview(added, 2);
  ASSERT_TRUE(preview.validation.valid());
  integration.prepare_adopt(preview);
  integration.adopt();
  double mass = 0.;
  for (const auto &d : integration.owned_demands())
    mass += d.demand;
  EXPECT_NEAR(Utilities::MPI::sum(mass, MPI_COMM_WORLD), 5. / 12., 2.e-13);
}
TEST(VascularTissueDemand, MPI_AdaptiveIntegrationUsesTheSelectedCellOwner)
{
  ASSERT_GE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  Triangulation<3> serial;
  GridGenerator::subdivided_hyper_rectangle(serial,
                                            {2, 1, 1},
                                            Point<3>(),
                                            Point<3>(1, 1, 1));
  for (const auto &cell : serial.active_cell_iterators())
    if (cell->center()[0] < .5)
      cell->set_refine_flag();
  serial.execute_coarsening_and_refinement();
  for (const auto &cell : serial.active_cell_iterators())
    cell->set_subdomain_id(cell->center()[0] < .5 ? 1 : 0);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<3> bg(MPI_COMM_WORLD);
  bg.create_triangulation(description);
  MappingQ1<3>           mapping;
  BackgroundGeometry     geometry(bg, mapping);
  std::map<Id, Point<3>> queries;
  if (integration_rank() == 0)
    queries.emplace(0, Point<3>(.5, .25, .25));
  const auto located = geometry.locate_points(queries, MPI_COMM_WORLD);
  if (integration_rank() == 0)
    {
      EXPECT_EQ(located.size(), 1u);
      if (located.size() == 1)
        {
          EXPECT_EQ(located.at(0).point_owner, 0u);
          EXPECT_EQ(located.at(0).cell_owner, 1u);
          EXPECT_DOUBLE_EQ(located.at(0).diameter, .75);
        }
    }
  const auto centres =
    owned_centres(geometry, {{.4, .25, .25}, {.6, .25, .25}});
  const auto added = owned_centres(geometry, {{.5, .25, .25}}, 10);
  TissueIntegrationParameters p;
  p.quadrature_order     = 1;
  p.samples_per_terminal = 1;
  p.maximum_points       = 1000;
  TissueDemandIntegration integration(
    geometry, "1", p, 3, centres, MPI_COMM_WORLD);
  ASSERT_TRUE(integration.validation().valid());
  integration_bootstrap(integration);
  const auto preview = integration.preview(added, 2);
  ASSERT_TRUE(preview.validation.valid());
  EXPECT_EQ(preview.work.refined_cells, integration_rank() == 1 ? 2u : 0u);
  integration.prepare_adopt(preview);
  integration.adopt();
  EXPECT_EQ(integration.point_count(), 72u);
  EXPECT_DOUBLE_EQ(integration.total_demand(), 1.);
  EXPECT_EQ(bg.n_global_active_cells(), 9u);
}
TEST(VascularTissueDemand, BOTH_RefinedInvalidDensityNeverChangesAcceptedDemand)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  const auto centres = owned_centres(geometry, {{.25, .5, .5}, {.75, .5, .5}});
  const auto added   = owned_centres(geometry, {{.5, .5, .5}}, 10);
  TissueIntegrationParameters p;
  p.quadrature_order     = 1;
  p.samples_per_terminal = 1;
  p.maximum_points       = 1000;
  TissueDemandIntegration integration(
    geometry, "1-2*(x>0.9)", p, 2, centres, MPI_COMM_WORLD);
  ASSERT_TRUE(integration.validation().valid());
  integration_bootstrap(integration);
  const auto preview = integration.preview(added, 2);
  EXPECT_FALSE(preview.validation.valid());
  EXPECT_FALSE(preview.integration_budget_exhausted);
  EXPECT_FALSE(preview.territory.demand.validation.valid());
  EXPECT_FALSE(preview.territory.demand.integration_budget_exhausted);
  EXPECT_EQ(integration.point_count(), 8u);
  EXPECT_DOUBLE_EQ(integration.total_demand(), 1.);
  EXPECT_THROW(integration.prepare_adopt(preview), ExceptionBase);
}
TEST(VascularTissueDemand, MPI_IntegrationControlAndDensityErrorsAreCollective)
{
  ASSERT_GE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  const auto         centres = owned_centres(geometry, {{.25, .5, .5}});
  TissueIntegrationParameters p;
  p.maximum_points = integration_rank() == 0 ? 1000 : 2000;
  EXPECT_THROW(
    TissueDemandIntegration(geometry, "1", p, 2, centres, MPI_COMM_WORLD),
    ExceptionBase);
  p.maximum_points = 1000;
  EXPECT_THROW(
    TissueDemandIntegration(geometry, "-1", p, 2, centres, MPI_COMM_WORLD),
    ExceptionBase);
}
TEST(VascularTissueDemand,
     BOTH_VoronoiQuadratureConvergesToAnalyticalTerritoryIntegrals)
{
  parallel::distributed::Triangulation<3> bg(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(bg);
  MappingQ1<3>       mapping;
  BackgroundGeometry geometry(bg, mapping);
  const auto         centres =
    owned_centres(geometry, {{.15, .5, .5}, {.4, .5, .5}, {.85, .5, .5}});
  // These collinear centres partition the cube at x=.275 and x=.625.
  const auto primitive = [](double x) { return x + .5 * x * x; };
  const std::array<double, 3> exact{{primitive(.275),
                                     primitive(.625) - primitive(.275),
                                     primitive(1.) - primitive(.625)}};
  const auto                  cells        = bg.n_global_active_cells();
  double                      coarse_error = 0., fine_error = 0.;
  for (const auto &[samples, points] :
       std::array<std::pair<unsigned int, std::uint64_t>, 3>{
         {{64, 512}, {512, 4096}, {4096, 32768}}})
    {
      TissueIntegrationParameters p;
      p.samples_per_terminal = samples;
      TissueDemandIntegration integration(
        geometry, "1+x", p, 3, centres, MPI_COMM_WORLD);
      ASSERT_TRUE(integration.validation().valid());
      EXPECT_EQ(integration.point_count(), points);
      integration_bootstrap(integration);
      const auto   weights = integration.owned_demands();
      const double width   = 1. / std::cbrt(double(points) / 8.);
      double       error = 0., total = 0.;
      for (unsigned int i = 0; i < 3; ++i)
        {
          const double actual =
            integration_demand(weights, integration_base + 1000 + i);
          // Assignment error is confined to the integration cells containing
          // territory cuts. Their mass is at most max(1+x)*width per cut.
          const unsigned int cuts = i == 1 ? 2 : 1;
          EXPECT_LE(std::abs(actual - exact[i]), 2. * cuts * width + 1.e-13);
          EXPECT_GT(actual, 0.);
          error = std::max(error, std::abs(actual - exact[i]));
          total += actual;
        }
      EXPECT_NEAR(total, 1.5, 2.e-13);
      EXPECT_NEAR(integration.total_demand(), 1.5, 2.e-13);
      EXPECT_EQ(bg.n_global_active_cells(), cells);
      if (samples == 64)
        coarse_error = error;
      if (samples == 4096)
        fine_error = error;
    }
  EXPECT_GT(coarse_error, 0.);
  EXPECT_LT(fine_error, coarse_error);
}
