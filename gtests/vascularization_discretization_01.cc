// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <cmath>
#include <set>

using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  NetworkData
  line(Point<3> a, Point<3> b)
  {
    NetworkData n;
    n.nodes[10].position = a;
    n.nodes[20].position = b;
    Vessel e;
    e.nodes        = {{10, 20}};
    e.radii        = {{.01, .02}};
    e.viscosity    = .004;
    n.vessels[100] = e;
    return n;
  }
} // namespace

TEST(VascularDiscretization, TaperVolumeResistanceAndNativeRefinement)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.refine_global(2);
  const auto network = line(Point<3>(-.25, .4, .3), Point<3>(1.25, .4, .3));
  const auto result =
    discretize(background, MappingQ1<3>(), network, {}, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_GT(result.exterior_cells, 0u);
  const double r0 = .01 + .01 / 6., r1 = .01 + .01 * 5. / 6.;
  const double inside = numbers::PI * (r0 * r0 + r0 * r1 + r1 * r1) / 3.;
  EXPECT_NEAR(result.inside_volume / inside, 1., 1.e-13);
  Triangulation<1, 3> mesh;
  const auto          references = create_triangulation(network, result, mesh);
  mesh.refine_global(2);
  double v = 0., r = 0., l = 0.;
  for (const auto &cell : mesh.active_cell_iterators())
    {
      const auto ref = physical_reference(cell->id(), references);
      const auto a   = network.nodes.at(10).position,
                 b   = network.nodes.at(20).position;
      EXPECT_LT(cell->vertex(0).distance((1. - ref.begin) * a + ref.begin * b),
                1.e-13);
      EXPECT_LT(cell->vertex(1).distance((1. - ref.end) * a + ref.end * b),
                1.e-13);
      v += volume(network, ref.vessel, ref.begin, ref.end);
      r += resistance(network, ref.vessel, ref.begin, ref.end);
      l += cell->measure();
    }
  const double expected_v =
    numbers::PI * 1.5 * (.01 * .01 + .01 * .02 + .02 * .02) / 3.;
  const double expected_r = 8. * .004 / numbers::PI * 1.5 / (3. * (.02 - .01)) *
                            (std::pow(.01, -3.) - std::pow(.02, -3.));
  EXPECT_NEAR(v / expected_v, 1., 1.e-13);
  EXPECT_NEAR(r / expected_r, 1., 1.e-13);
  EXPECT_NEAR(l, 1.5, 1.e-13);
  EXPECT_EQ(network.nodes.size(), 2u);
  EXPECT_EQ(network.vessels.size(), 1u);
  EXPECT_EQ(network.vessels.at(100).radii, (std::array<double, 2>{{.01, .02}}));
}

TEST(VascularDiscretization,
     JunctionVerticesAndWidePhysicalIdsStayIndependentOfCells)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  NetworkData n;
  n.nodes[0].position = Point<3>(.2, .5, .5);
  n.nodes[1].position = Point<3>(.5, .5, .5);
  n.nodes[2].position = Point<3>(.8, .3, .5);
  n.nodes[3].position = Point<3>(.8, .7, .5);
  const Id base       = Id(1) << 54;
  for (unsigned int i = 0; i < 3; ++i)
    {
      Vessel e;
      e.nodes =
        i == 0 ? std::array<Id, 2>{{0, 1}} : std::array<Id, 2>{{1, i + 1}};
      e.radii             = {{.005, .005}};
      e.viscosity         = .004;
      n.vessels[base + i] = e;
    }
  const auto result =
    discretize(background, MappingQ1<3>(), n, {}, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  Triangulation<1, 3> mesh;
  const auto          refs = create_triangulation(n, result, mesh);
  mesh.refine_global(1);
  std::map<unsigned int, unsigned int> incidences;
  std::map<Id, unsigned int>           physical_cells;
  for (const auto &cell : mesh.active_cell_iterators())
    {
      for (const auto v : cell->vertex_indices())
        ++incidences[cell->vertex_index(v)];
      ++physical_cells[physical_reference(cell->id(), refs).vessel];
    }
  unsigned int junctions = 0;
  for (const auto &[v, count] : incidences)
    if (count == 3)
      ++junctions;
  EXPECT_EQ(junctions, 1u);
  EXPECT_EQ(physical_cells.size(), 3u);
  for (const auto &[id, count] : physical_cells)
    {
      EXPECT_GE(id, base);
      EXPECT_EQ(count, 2u);
    }
  DoFHandler<1, 3> dofs(mesh);
  dofs.distribute_dofs(FE_Q<1, 3>(1));
  EXPECT_EQ(dofs.n_dofs(), mesh.n_used_vertices());
  EXPECT_EQ(n.nodes.size(), 4u);
  EXPECT_EQ(n.vessels.size(), 3u);
}

TEST(VascularDiscretization, RefinedTetrahedralBoxFalsePositivesDoNotSelectH)
{
  Triangulation<3>            background;
  const std::vector<Point<3>> points{{0., 0., 0.},
                                     {1., 0., 0.},
                                     {0., 1., 0.},
                                     {0., 0., 1.},
                                     {.37, .37, .37},
                                     {2., .37, .37},
                                     {.37, 2., .37},
                                     {.37, .37, 2.}};
  std::vector<CellData<3>>    cells(2, CellData<3>(4));
  for (unsigned int i = 0; i < 2; ++i)
    {
      for (unsigned int v = 0; v < 4; ++v)
        cells[i].vertices[v] = 4 * i + v;
      cells[i].material_id = i + 1;
    }
  background.create_triangulation(points, cells, SubCellData());
  double coarse_h = 0.;
  for (const auto &cell : background.active_cell_iterators())
    if (cell->material_id() == 1)
      cell->set_refine_flag();
    else if (cell->material_id() == 2)
      coarse_h = cell->diameter();
  background.execute_coarsening_and_refinement();
  const auto network = line(Point<3>(.4, .4, .4), Point<3>(.45, .45, .4));
  const auto result  = discretize(
    background, MappingFE<3>(FE_SimplexP<3>(1)), network, {}, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_EQ(result.exterior_cells, 0u);
  for (const auto &cell : result.cells)
    {
      EXPECT_DOUBLE_EQ(cell.h_min, coarse_h);
      EXPECT_DOUBLE_EQ(cell.h_max, coarse_h);
    }
  EXPECT_NEAR(result.inside_volume / volume(network, 100), 1., 1.e-13);
}

TEST(VascularDiscretization, EntireSegmentControlsLocalResolution)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.refine_global(2);
  const auto network = line(Point<3>(.01, .32, .31), Point<3>(.99, .32, .31));
  const auto uniform =
    discretize(background, MappingQ1<3>(), network, {}, MPI_COMM_WORLD);
  for (const auto &cell : background.active_cell_iterators())
    if (cell->center()[0] < .25)
      cell->set_refine_flag();
  background.execute_coarsening_and_refinement();
  const auto adaptive =
    discretize(background, MappingQ1<3>(), network, {}, MPI_COMM_WORLD);
  ASSERT_TRUE(uniform.validation.valid());
  ASSERT_TRUE(adaptive.validation.valid());
  EXPECT_GT(adaptive.cells.size(), uniform.cells.size());
  double smallest = 1.e20, largest = 0.;
  for (const auto &cell : adaptive.cells)
    {
      smallest = std::min(smallest, cell.h_min);
      largest  = std::max(largest, cell.h_max);
      EXPECT_LE((cell.reference.end - cell.reference.begin) *
                  length(network, 100),
                2. * cell.h_min * (1. + 1.e-12));
    }
  EXPECT_NEAR(largest / smallest, 2., 1.e-13);
  EXPECT_NEAR(adaptive.inside_volume / uniform.inside_volume, 1., 1.e-13);
}

TEST(VascularDiscretization, WarpedBilinearBoundaryClipsAtAnalyticalCrossing)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.begin_active()->vertex(7)[2] = 1.4;
  auto network = line(Point<3>(.1, .5, 1.1), Point<3>(.9, .5, 1.1));
  network.vessels.at(100).radii = {{.01, .01}};
  const auto result =
    discretize(background, MappingQ1<3>(), network, {}, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  // z=1+0.4*x*y intersects y=0.5,z=1.1 at x=0.5.
  EXPECT_NEAR(result.inside_volume / (numbers::PI * .01 * .01 * .4),
              1.,
              1.e-12);
  EXPECT_GT(result.exterior_cells, 0u);
  for (const auto &cell : result.cells)
    if (cell.inside)
      EXPECT_GE(cell.reference.begin, .5 - 1.e-12);

  // Along y=1-x the same warped face has two intersections, at
  // x=(1+-sqrt(1/2))/2. Their physical separation is exactly one metre.
  auto diagonal = line(Point<3>(0., 1., 1.05), Point<3>(1., 0., 1.05));
  diagonal.vessels.at(100).radii = {{.01, .01}};
  const auto twice =
    discretize(background, MappingQ1<3>(), diagonal, {}, MPI_COMM_WORLD);
  ASSERT_TRUE(twice.validation.valid());
  EXPECT_NEAR(twice.inside_volume / (numbers::PI * .01 * .01), 1., 1.e-12);
  EXPECT_GE(twice.exterior_cells, 2u);
}

TEST(VascularDiscretization, NonconvexDomainHasDisjointInteriorIntervals)
{
  Triangulation<3> full, background;
  GridGenerator::subdivided_hyper_rectangle(full,
                                            {3, 3, 1},
                                            Point<3>(0., 0., 0.),
                                            Point<3>(3., 3., 1.));
  std::set<Triangulation<3>::active_cell_iterator> remove;
  for (const auto &cell : full.active_cell_iterators())
    if (cell->center()[0] > 1. && cell->center()[0] < 2. &&
        cell->center()[1] > 1.)
      remove.insert(cell);
  GridGenerator::create_triangulation_with_removed_cells(full,
                                                         remove,
                                                         background);
  auto network = line(Point<3>(.5, 2.5, .5), Point<3>(2.5, 2.5, .5));
  network.vessels.at(100).radii = {{.01, .01}};
  DiscretizationParameters p;
  p.exterior_length = .2;
  const auto result =
    discretize(background, MappingQ1<3>(), network, p, MPI_COMM_WORLD);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_NEAR(result.inside_volume / (numbers::PI * .01 * .01), 1., 1.e-13);
  EXPECT_GT(result.exterior_cells, 0u);
  for (const auto &cell : result.cells)
    {
      const double middle = .5 * (cell.reference.begin + cell.reference.end);
      EXPECT_EQ(cell.inside, middle < .25 || middle > .75);
    }
}

TEST(VascularDiscretization, ReportsShortVesselsAndBudgetExhaustion)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  const auto network = line(Point<3>(.4, .4, .4), Point<3>(.41, .4, .4));
  const auto short_vessel =
    discretize(background, MappingQ1<3>(), network, {}, MPI_COMM_WORLD);
  ASSERT_TRUE(short_vessel.validation.valid());
  EXPECT_EQ(short_vessel.short_cells, 1u);
  DiscretizationParameters p;
  p.cell_budget   = 1;
  p.target_factor = .001;
  p.lower_factor  = .0005;
  p.upper_factor  = .002;
  const auto exhausted =
    discretize(background, MappingQ1<3>(), network, p, MPI_COMM_WORLD);
  EXPECT_FALSE(exhausted.validation.valid());
  Triangulation<1, 3> mesh;
  EXPECT_THROW(create_triangulation(network, exhausted, mesh), ExceptionBase);
}

TEST(VascularDiscretization,
     NumericalResolutionCannotChangeCCOCandidatesOrPhysics)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  background.refine_global(2);
  Parameters p;
  p.seed_position  = Point<3>(.2, .5, .5);
  p.seed_direction = Point<3>(1., 0., 0.);
  p.seed_radius    = .005;
  p.seed_length    = .3;
  p.viscosity      = .004;
  p.terminal_count = 8;
  const auto coarse =
    generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  p.discretization.target_factor = .25;
  p.discretization.lower_factor  = .125;
  p.discretization.upper_factor  = .5;
  const auto fine = generate(background, MappingQ1<3>(), {}, p, MPI_COMM_WORLD);
  ASSERT_TRUE(coarse.validation.valid());
  ASSERT_TRUE(fine.validation.valid());
  EXPECT_GT(fine.discretization.cells.size(),
            coarse.discretization.cells.size());
  EXPECT_EQ(coarse.sampled_points, fine.sampled_points);
  for (const auto &[id, node] : coarse.network.nodes)
    EXPECT_EQ(node.position, fine.network.nodes.at(id).position);
  for (const auto &[id, e] : coarse.network.vessels)
    {
      EXPECT_EQ(e.nodes, fine.network.vessels.at(id).nodes);
      EXPECT_EQ(e.radii, fine.network.vessels.at(id).radii);
    }
  EXPECT_NEAR(fine.discretization.inside_volume /
                coarse.discretization.inside_volume,
              1.,
              1.e-13);
}
