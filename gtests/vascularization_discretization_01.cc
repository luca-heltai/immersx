// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_description.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <cmath>
#include <limits>
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

TEST(VascularDiscretization,
     BOTH_OwnedCrossingsMatchAnalyticalPhysicsAndResolution)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {8, 1, 1},
                                            Point<3>(0., 0., 0.),
                                            Point<3>(8., 1., 1.));
  for (const auto &cell : background.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[0] < 2.)
      cell->set_refine_flag();
  background.execute_coarsening_and_refinement();
  const Id    base = Id{1} << 60;
  NetworkData network;
  // The sole physical owner is deliberately unrelated to midpoint ownership.
  // Other ranks have neither the physical graph nor its endpoint nodes.
  if (rank == size - 1)
    {
      network = line(Point<3>(-1., .32, .31), Point<3>(9., .32, .31));
      network.vessels[base] = network.vessels.at(100);
      network.vessels.erase(100);
      network.vessels.at(base).owner = rank;
      for (auto &[id, node] : network.nodes)
        node.owner = rank;
    }
  const auto result = discretize(background, MappingQ1<3>(), network, {}, comm);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_EQ(result.cells.empty(), rank != size - 1);
  const double a = .011, b = .019;
  const double expected_inside =
    numbers::PI * 8. * (a * a + a * b + b * b) / 3.;
  EXPECT_NEAR(result.inside_volume / expected_inside, 1., 1.e-12);
  double       v = 0., r = 0., l = 0.;
  unsigned int owner_mismatches = 0, remote_numerical_cells = 0;
  // Independent global cell boxes are a test oracle only; production exchanges
  // segments sparsely and does not replicate the background or vascular graph.
  using Cell = std::pair<BoundingBox<3>, unsigned int>;
  std::vector<Cell> local;
  for (const auto &cell : background.active_cell_iterators())
    if (cell->is_locally_owned())
      local.push_back({cell->bounding_box(), rank});
  const auto boxes = Utilities::MPI::all_gather(comm, local);
  for (const auto &cell : result.cells)
    {
      EXPECT_EQ(cell.reference.vessel, base);
      v += volume(network, base, cell.reference.begin, cell.reference.end);
      r += resistance(network, base, cell.reference.begin, cell.reference.end);
      const double length = 10. * (cell.reference.end - cell.reference.begin);
      l += length;
      const double   middle = .5 * (cell.reference.begin + cell.reference.end);
      const Point<3> point(-1. + 10. * middle, .32, .31);
      unsigned int   owner = size;
      double h_min = std::numeric_limits<double>::infinity(), h_max = 0.;
      for (const auto &partition : boxes)
        for (const auto &[box, source] : partition)
          {
            if (box.point_inside(point))
              owner = std::min(owner, source);
            const auto &bounds = box.get_boundary_points();
            if (bounds.first[1] < .32 && .32 < bounds.second[1] &&
                bounds.first[2] < .31 && .31 < bounds.second[2] &&
                bounds.first[0] < -1. + 10. * cell.reference.end - 1.e-12 &&
                bounds.second[0] > -1. + 10. * cell.reference.begin + 1.e-12)
              {
                const double h = bounds.first.distance(bounds.second);
                h_min          = std::min(h_min, h);
                h_max          = std::max(h_max, h);
              }
          }
      if (owner == size)
        owner = size - 1;
      owner_mismatches += owner != cell.owner;
      remote_numerical_cells += cell.owner != rank;
      if (cell.inside)
        {
          EXPECT_NEAR(cell.h_min, h_min, 1.e-12);
          EXPECT_NEAR(cell.h_max, h_max, 1.e-12);
          EXPECT_LE(length, 2. * h_min * (1. + 1.e-12));
        }
    }
  EXPECT_EQ(Utilities::MPI::sum(owner_mismatches, comm), 0u);
  if (size > 1)
    EXPECT_GT(Utilities::MPI::sum(remote_numerical_cells, comm), 0u);
  EXPECT_NEAR(Utilities::MPI::sum(l, comm), 10., 1.e-12);
  const double expected_v = numbers::PI * 10. * (.0001 + .0002 + .0004) / 3.;
  const double expected_r = 8. * .004 / numbers::PI * 10. / .03 *
                            (std::pow(.01, -3.) - std::pow(.02, -3.));
  EXPECT_NEAR(Utilities::MPI::sum(v, comm) / expected_v, 1., 1.e-12);
  EXPECT_NEAR(Utilities::MPI::sum(r, comm) / expected_r, 1., 1.e-12);
  EXPECT_EQ(network.vessels.size(), rank == size - 1 ? 1u : 0u);
}

TEST(VascularDiscretization, BOTH_GlobalBudgetAndRemoteInvalidData)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {8, 1, 1},
                                            Point<3>(0., 0., 0.),
                                            Point<3>(8., 1., 1.));
  auto n = line(Point<3>(rank + .2, .4, .4), Point<3>(rank + .3, .4, .4));
  for (auto &[id, node] : n.nodes)
    node.owner = rank;
  n.vessels[100 + rank] = n.vessels.at(100);
  if (rank != 0)
    n.vessels.erase(100);
  n.vessels.at(100 + rank).owner = rank;
  const auto first = n.nodes.at(10), second = n.nodes.at(20);
  n.nodes.clear();
  n.nodes[10 + 100 * rank]       = first;
  n.nodes[20 + 100 * rank]       = second;
  n.vessels.at(100 + rank).nodes = {{10 + 100 * rank, 20 + 100 * rank}};
  DiscretizationParameters p;
  p.cell_budget     = size > 1 ? size - 1 : 1;
  const auto budget = discretize(background, MappingQ1<3>(), n, p, comm);
  EXPECT_EQ(budget.validation.valid(), size == 1);
  if (rank == size - 1)
    n.vessels.at(100 + rank).radii[0] = -1.;
  const auto invalid = discretize(background, MappingQ1<3>(), n, {}, comm);
  EXPECT_FALSE(invalid.validation.valid());
  if (size > 1)
    {
      p.cell_budget = 10 + rank;
      EXPECT_THROW(discretize(background, MappingQ1<3>(), n, p, comm),
                   ExceptionBase);
    }
}

TEST(VascularDiscretization, BOTH_InterfaceVolumeCountsOwnedVesselsOnce)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {8, 1, 1},
                                            Point<3>(0., 0., 0.),
                                            Point<3>(8., 1., 1.));
  // This line lies on a cell face and, for P=2/4, a partition interface.
  // Replicated ghosts must not multiply its volume or numerical cells.
  auto network = line(Point<3>(4., .1, .4), Point<3>(4., .9, .4));
  network.vessels.at(100).owner = size - 1;
  for (auto &[id, node] : network.nodes)
    node.owner = size - 1;
  const auto result = discretize(background, MappingQ1<3>(), network, {}, comm);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_EQ(result.cells.empty(), rank != size - 1);
  EXPECT_EQ(result.exterior_cells, 0u);
  EXPECT_NEAR(result.inside_volume / volume(network, 100), 1., 1.e-12);
  unsigned int   owner = size;
  const Point<3> midpoint(4., .5, .4);
  for (const auto &cell : background.active_cell_iterators())
    if (cell->is_locally_owned() && cell->bounding_box().point_inside(midpoint))
      owner = rank;
  owner = Utilities::MPI::min(owner, comm);
  EXPECT_LT(owner, size);
  for (const auto &cell : result.cells)
    EXPECT_EQ(cell.owner, owner);
}

TEST(VascularDiscretization, BOTH_EmptyBackgroundRanksStillParticipate)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::hyper_cube(background);
  NetworkData network;
  if (rank == size - 1)
    {
      network = line(Point<3>(.1, .4, .3), Point<3>(.9, .4, .3));
      network.vessels.at(100).owner = rank;
      for (auto &[id, node] : network.nodes)
        node.owner = rank;
    }
  const auto result = discretize(background, MappingQ1<3>(), network, {}, comm);
  ASSERT_TRUE(result.validation.valid());
  EXPECT_GT(result.inside_volume, 0.);
  EXPECT_EQ(result.exterior_cells, 0u);
  EXPECT_EQ(result.cells.empty(), rank != size - 1);
  unsigned int cell_owner = size;
  for (const auto &cell : background.active_cell_iterators())
    if (cell->is_locally_owned())
      cell_owner = rank;
  cell_owner = Utilities::MPI::min(cell_owner, comm);
  for (const auto &cell : result.cells)
    EXPECT_EQ(cell.owner, cell_owner);
}

TEST(VascularDiscretization,
     BOTH_TetrahedralClippingRejectsBoxOnlyIntersections)
{
  const auto       comm = MPI_COMM_WORLD;
  const auto       rank = Utilities::MPI::this_mpi_process(comm);
  const auto       size = Utilities::MPI::n_mpi_processes(comm);
  Triangulation<3> serial;
  GridGenerator::reference_cell(serial, ReferenceCells::Tetrahedron);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, comm);
  parallel::fullydistributed::Triangulation<3> background(comm);
  background.create_triangulation(description);
  NetworkData network;
  if (rank == size - 1)
    {
      // x+y+z<=1 clips the first segment at x=.6. The second lies
      // entirely in the tetrahedron's bounding box, but outside its volume.
      network = line(Point<3>(-.2, .2, .2), Point<3>(.8, .2, .2));
      network.nodes[30].position    = Point<3>(.6, .6, .6);
      network.nodes[40].position    = Point<3>(.8, .6, .6);
      network.vessels[200]          = network.vessels.at(100);
      network.vessels.at(200).nodes = {{30, 40}};
      for (auto &[id, node] : network.nodes)
        node.owner = rank;
      for (auto &[id, vessel] : network.vessels)
        vessel.owner = rank;
    }
  const auto result =
    discretize(background, MappingFE<3>(FE_SimplexP<3>(1)), network, {}, comm);
  ASSERT_TRUE(result.validation.valid());
  const double a = .012, b = .018;
  const double expected = numbers::PI * .6 * (a * a + a * b + b * b) / 3.;
  EXPECT_NEAR(result.inside_volume / expected, 1., 1.e-12);
  for (const auto &cell : result.cells)
    if (cell.reference.vessel == 200)
      {
        EXPECT_FALSE(cell.inside);
        EXPECT_DOUBLE_EQ(cell.h_min, 0.);
        EXPECT_DOUBLE_EQ(cell.h_max, 0.);
        EXPECT_EQ(cell.owner, rank);
      }
}

TEST(VascularDiscretization, BOTH_NativeDistributedBranchesShareJunctionDofs)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {8, 2, 1},
                                            Point<3>(0., 0., 0.),
                                            Point<3>(8., 2., 1.));
  const Id                      base = (Id{1} << 60) + 19;
  const std::array<Point<3>, 4> positions{{Point<3>(.2, 1., .5),
                                           Point<3>(4., 1., .5),
                                           Point<3>(7.8, .3, .5),
                                           Point<3>(7.8, 1.7, .5)}};
  NetworkData                   network;
  for (unsigned int v = 0; v < 4; ++v)
    if (v % size == rank)
      {
        network.nodes[base + v].position = positions[v];
        network.nodes[base + v].owner    = rank;
      }
  for (unsigned int e = 0; e < 3; ++e)
    if (e % size == rank)
      {
        const std::array<unsigned int, 2> nodes =
          e == 0 ? std::array<unsigned int, 2>{{0, 1}} :
                   std::array<unsigned int, 2>{{1, e + 1}};
        Vessel vessel;
        vessel.nodes              = {{base + nodes[0], base + nodes[1]}};
        vessel.radii              = {{.01, .02}};
        vessel.viscosity          = .004;
        vessel.owner              = rank;
        network.vessels[base + e] = vessel;
        for (const auto v : nodes)
          {
            network.nodes[base + v].position = positions[v];
            network.nodes[base + v].owner    = v % size;
          }
      }
  const auto result = discretize(background, MappingQ1<3>(), network, {}, comm);
  ASSERT_TRUE(result.validation.valid());
  parallel::fullydistributed::Triangulation<1, 3> mesh(comm);
  const auto refs = create_triangulation(network, result, mesh);
  const auto n_cells =
    Utilities::MPI::sum(static_cast<std::uint64_t>(result.cells.size()), comm);
  EXPECT_EQ(mesh.n_global_active_cells(), n_cells);
  DoFHandler<1, 3> dofs(mesh);
  dofs.distribute_dofs(FE_Q<1, 3>(1));
  // A connected tree has one more vertex than edges, including numerical cuts.
  EXPECT_EQ(dofs.n_dofs(), n_cells + 1);
  double                  length_sum = 0., volume_sum = 0., resistance_sum = 0.;
  unsigned int            junction_incidence = 0;
  types::global_dof_index junction_min       = numbers::invalid_dof_index,
                          junction_max       = 0;
  for (const auto &cell : dofs.active_cell_iterators())
    if (!cell->is_artificial())
      {
        const auto ref = physical_reference(cell->id(), refs);
        ASSERT_GE(ref.vessel, base);
        ASSERT_LT(ref.vessel, base + 3);
        const auto     e = ref.vessel - base;
        const Point<3> a = positions[e == 0 ? 0 : 1];
        const Point<3> b = positions[e == 0 ? 1 : e + 1];
        EXPECT_LT(cell->vertex(0).distance((1. - ref.begin) * a +
                                           ref.begin * b),
                  1.e-12);
        EXPECT_LT(cell->vertex(1).distance((1. - ref.end) * a + ref.end * b),
                  1.e-12);
        if (cell->is_locally_owned())
          {
            const double l  = a.distance(b) * (ref.end - ref.begin);
            const double r0 = .01 + .01 * ref.begin, r1 = .01 + .01 * ref.end;
            length_sum += cell->measure();
            volume_sum += numbers::PI * l * (r0 * r0 + r0 * r1 + r1 * r1) / 3.;
            resistance_sum += 8. * .004 / numbers::PI * l / (3. * (r1 - r0)) *
                              (std::pow(r0, -3.) - std::pow(r1, -3.));
            for (unsigned int v = 0; v < 2; ++v)
              if (cell->vertex(v) == positions[1])
                {
                  ++junction_incidence;
                  const auto index = cell->vertex_dof_index(v, 0);
                  junction_min     = std::min(junction_min, index);
                  junction_max     = std::max(junction_max, index);
                }
          }
      }
  double expected_length = positions[0].distance(positions[1]) +
                           positions[1].distance(positions[2]) +
                           positions[1].distance(positions[3]);
  const double expected_volume =
    numbers::PI * expected_length * (.0001 + .0002 + .0004) / 3.;
  const double expected_resistance = 8. * .004 / numbers::PI * expected_length /
                                     .03 *
                                     (std::pow(.01, -3.) - std::pow(.02, -3.));
  EXPECT_NEAR(Utilities::MPI::sum(length_sum, comm) / expected_length,
              1.,
              1.e-12);
  EXPECT_NEAR(Utilities::MPI::sum(volume_sum, comm) / expected_volume,
              1.,
              1.e-12);
  EXPECT_NEAR(Utilities::MPI::sum(resistance_sum, comm) / expected_resistance,
              1.,
              1.e-12);
  EXPECT_EQ(Utilities::MPI::sum(junction_incidence, comm), 3u);
  EXPECT_EQ(Utilities::MPI::min(junction_min, comm),
            Utilities::MPI::max(junction_max, comm));

  DiscretizationParameters fine_parameters;
  fine_parameters.target_factor = .25;
  fine_parameters.lower_factor  = .125;
  fine_parameters.upper_factor  = .5;
  const auto fine =
    discretize(background, MappingQ1<3>(), network, fine_parameters, comm);
  ASSERT_TRUE(fine.validation.valid());
  const auto fine_cells =
    Utilities::MPI::sum(static_cast<std::uint64_t>(fine.cells.size()), comm);
  EXPECT_GT(fine_cells, n_cells);
  EXPECT_NEAR(fine.inside_volume / result.inside_volume, 1., 1.e-12);
  parallel::fullydistributed::Triangulation<1, 3> fine_mesh(comm);
  const auto       fine_refs = create_triangulation(network, fine, fine_mesh);
  DoFHandler<1, 3> fine_dofs(fine_mesh);
  fine_dofs.distribute_dofs(FE_Q<1, 3>(1));
  EXPECT_EQ(fine_dofs.n_dofs(), fine_cells + 1);
  for (const auto &cell : fine_mesh.active_cell_iterators())
    if (!cell->is_artificial())
      {
        const auto ref = physical_reference(cell->id(), fine_refs);
        EXPECT_GE(ref.vessel, base);
        EXPECT_LT(ref.vessel, base + 3);
      }
  EXPECT_EQ(Utilities::MPI::sum(
              static_cast<std::uint64_t>(network.vessels.size()), comm),
            3u);
}

TEST(VascularDiscretization, BOTH_NativeDistributedMeshKeepsOnlyRelevantCells)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::subdivided_hyper_rectangle(background,
                                            {16, 1, 1},
                                            Point<3>(0., 0., 0.),
                                            Point<3>(16., 1., 1.));
  NetworkData network;
  const Id    base = Id{1} << 60;
  for (unsigned int i = 0; i < 16; ++i)
    if (i % size == rank)
      {
        const Id a = base + 2 * i, b = a + 1;
        // Distinct physical nodes at x=8 remain separate DoFs, even across
        // execution partitions. Geometric coincidence is not connectivity.
        network.nodes[a].position = Point<3>(i == 8 ? 8. : i + .2, .4, .3);
        network.nodes[b].position = Point<3>(i == 7 ? 8. : i + .8, .4, .3);
        network.nodes[a].owner = network.nodes[b].owner = rank;
        Vessel vessel;
        vessel.nodes              = {{a, b}};
        vessel.radii              = {{.01, .01}};
        vessel.viscosity          = .004;
        vessel.owner              = rank;
        network.vessels[base + i] = vessel;
      }
  const auto result = discretize(background, MappingQ1<3>(), network, {}, comm);
  ASSERT_TRUE(result.validation.valid());
  parallel::fullydistributed::Triangulation<1, 3> mesh(comm);
  const auto refs = create_triangulation(network, result, mesh);
  EXPECT_EQ(mesh.n_global_active_cells(), 16u);
  EXPECT_EQ(refs.size(), mesh.n_locally_owned_active_cells());
  if (size > 1)
    {
      EXPECT_LT(refs.size(), 16u);
    }
  DoFHandler<1, 3> dofs(mesh);
  dofs.distribute_dofs(FE_Q<1, 3>(1));
  EXPECT_EQ(dofs.n_dofs(), 32u);
  EXPECT_EQ(Utilities::MPI::sum(dofs.n_locally_owned_dofs(), comm), 32u);
}

TEST(VascularDiscretization, BOTH_NativeDistributedMeshWithEmptyRanks)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::hyper_cube(background);
  NetworkData network;
  if (rank == size - 1)
    {
      network = line(Point<3>(.1, .4, .3), Point<3>(.9, .4, .3));
      network.vessels.at(100).owner = rank;
      for (auto &[id, node] : network.nodes)
        node.owner = rank;
    }
  const auto result = discretize(background, MappingQ1<3>(), network, {}, comm);
  ASSERT_TRUE(result.validation.valid());
  parallel::fullydistributed::Triangulation<1, 3> mesh(comm);
  const auto refs = create_triangulation(network, result, mesh);
  EXPECT_EQ(mesh.n_global_active_cells(), 1u);
  EXPECT_EQ(refs.size(), mesh.n_locally_owned_active_cells());
  if (refs.empty())
    {
      for (const auto &cell : mesh.active_cell_iterators())
        EXPECT_TRUE(cell->is_artificial());
    }
  DoFHandler<1, 3> dofs(mesh);
  dofs.distribute_dofs(FE_Q<1, 3>(1));
  EXPECT_EQ(dofs.n_dofs(), 2u);
  EXPECT_EQ(Utilities::MPI::sum(dofs.n_locally_owned_dofs(), comm), 2u);
}

TEST(VascularDiscretization, BOTH_NativeMeshForEmptyPhysicalNetwork)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> background(comm);
  GridGenerator::hyper_cube(background);
  const auto empty = discretize(background, MappingQ1<3>(), {}, {}, comm);
  parallel::fullydistributed::Triangulation<1, 3> empty_mesh(comm);
  const auto empty_refs = create_triangulation({}, empty, empty_mesh);
  EXPECT_TRUE(empty_refs.empty());
  EXPECT_EQ(empty_mesh.n_global_active_cells(), 0u);
}

TEST(VascularDiscretization, BOTH_NativeMeshRejectsRemoteInvalidIntervals)
{
  const auto           comm = MPI_COMM_WORLD;
  const auto           rank = Utilities::MPI::this_mpi_process(comm);
  const auto           size = Utilities::MPI::n_mpi_processes(comm);
  NetworkData          network;
  DiscretizationResult result;
  if (rank == size - 1)
    {
      network = line(Point<3>(.1, .4, .3), Point<3>(.9, .4, .3));
      network.vessels.at(100).owner = rank;
      for (auto &[id, node] : network.nodes)
        node.owner = rank;
      DiscretizationCell cell;
      cell.reference = {100, .75, .25};
      cell.owner     = rank;
      result.cells.push_back(cell);
    }
  parallel::fullydistributed::Triangulation<1, 3> mesh(comm);
  EXPECT_THROW(create_triangulation(network, result, mesh), ExceptionBase);
  EXPECT_EQ(mesh.n_active_cells(), 0u);
  if (rank == size - 1)
    {
      network.nodes.at(10).position = Point<3>(1000., 0., 0.);
      network.nodes.at(20).position = Point<3>(1001., 0., 0.);
      result.cells[0].reference     = {100, .1, std::nextafter(.1, 1.)};
    }
  EXPECT_THROW(create_triangulation(network, result, mesh), ExceptionBase);
  EXPECT_EQ(mesh.n_active_cells(), 0u);
}

TEST(VascularDiscretization, MPI_NativeMeshRejectsInconsistentJunction)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  ASSERT_GE(size, 2u);
  NetworkData          network;
  DiscretizationResult result;
  if (rank < 2)
    {
      const Id a = rank == 0 ? 10 : 20, b = rank == 0 ? 20 : 30;
      network.nodes[a].position = Point<3>(rank, .4, .3);
      network.nodes[b].position = Point<3>(rank + 1., .4, .3);
      network.nodes[a].owner = network.nodes[b].owner = rank;
      network.nodes[20].owner                         = 0;
      if (rank == 1)
        network.nodes[a].position[1] += .01;
      Vessel vessel;
      vessel.nodes                = {{a, b}};
      vessel.radii                = {{.01, .01}};
      vessel.viscosity            = .004;
      vessel.owner                = rank;
      network.vessels[100 + rank] = vessel;
      DiscretizationCell cell;
      cell.reference = {100 + rank, 0., 1.};
      cell.owner     = rank;
      result.cells.push_back(cell);
    }
  parallel::fullydistributed::Triangulation<1, 3> mesh(comm);
  EXPECT_THROW(create_triangulation(network, result, mesh), ExceptionBase);
  EXPECT_EQ(mesh.n_active_cells(), 0u);
}
