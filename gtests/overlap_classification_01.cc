// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_description.h>

#include <gtest/gtest.h>
#include <immersx/core/domain.h>
#include <immersx/core/overlap_integration_support.h>
#include <immersx/coupling/detail/coupling_point.h>
#include <immersx/coupling/particle_coupling.h>
#include <immersx/grid/overlap_classification.h>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

using namespace dealii;
using namespace ImmersX;

namespace
{
  constexpr OverlapMaterialIds labels{17, 31, 42};

  template <int dim, int spacedim>
  void
  distribute(Triangulation<dim, spacedim>                             &coarse,
             parallel::fullydistributed::Triangulation<dim, spacedim> &mesh,
             const bool only_rank_zero = false)
  {
    const auto   ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    unsigned int c     = 0;
    for (const auto &cell : coarse.active_cell_iterators())
      cell->set_subdomain_id(only_rank_zero ? 0 : c++ % ranks);
    mesh.create_triangulation(
      TriangulationDescription::Utilities::
        create_description_from_triangulation(coarse, MPI_COMM_WORLD));
  }

  void
  make_line(Triangulation<1, 3> &mesh)
  {
    const std::vector<Point<3>> vertices{{-1., .5, .5},
                                         {-.5, .5, .5},
                                         {.25, .5, .5},
                                         {.75, .5, .5},
                                         {1.5, .5, .5},
                                         {2., .5, .5}};
    std::vector<CellData<1>>    cells(vertices.size() - 1);
    for (unsigned int c = 0; c < cells.size(); ++c)
      {
        cells[c].vertices    = {{c, c + 1}};
        cells[c].material_id = 99;
      }
    mesh.create_triangulation(vertices, cells, SubCellData());
  }

  types::material_id
  expected_line_label(const Triangulation<1, 3>::active_cell_iterator &cell,
                      const OverlapMaterialIds                        &ids)
  {
    const double a = std::min(cell->vertex(0)[0], cell->vertex(1)[0]);
    const double b = std::max(cell->vertex(0)[0], cell->vertex(1)[0]);
    if (b <= 0. || a >= 1.)
      return ids.outside;
    if (a >= 0. && b <= 1.)
      return ids.inside;
    return ids.partial;
  }
} // namespace

TEST(OverlapClassification, BOTH_MappedLineAndGhostLabels)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, 0., 1.);
  background.refine_global(1);
  for (const auto &cell : background.active_cell_iterators())
    cell->set_material_id(77);
  MappingQ1<3>        background_mapping;
  GridTools::Cache<3> cache(background, background_mapping);

  Triangulation<1, 3> coarse;
  make_line(coarse);
  parallel::fullydistributed::Triangulation<1, 3> source(MPI_COMM_WORLD);
  distribute(coarse, source);
  MappingQ1<1, 3> mapping;
  set_overlap_material_ids(source, mapping, QGauss<1>(2), cache, labels);

  std::array<unsigned int, 3> counts{};
  unsigned int                ghosts = 0;
  for (const auto &cell : source.active_cell_iterators())
    if (!cell->is_artificial())
      {
        EXPECT_EQ(cell->material_id(), expected_line_label(cell, labels));
        ghosts += cell->is_ghost();
        if (cell->is_locally_owned())
          {
            counts[0] += cell->material_id() == labels.outside;
            counts[1] += cell->material_id() == labels.partial;
            counts[2] += cell->material_id() == labels.inside;
          }
      }
  EXPECT_EQ(Utilities::MPI::sum(counts[0], MPI_COMM_WORLD), 2u);
  EXPECT_EQ(Utilities::MPI::sum(counts[1], MPI_COMM_WORLD), 2u);
  EXPECT_EQ(Utilities::MPI::sum(counts[2], MPI_COMM_WORLD), 1u);
  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
    {
      EXPECT_GT(Utilities::MPI::sum(ghosts, MPI_COMM_WORLD), 0u);
    }
  for (const auto &cell : background.active_cell_iterators())
    EXPECT_EQ(cell->material_id(), 77u);

  // Both sampled overlap classes can be selected through one existing support.
  const OverlapMaterialIds combined{17, 42, 42};
  set_overlap_material_ids(source, mapping, QGauss<1>(2), cache, combined);
  DomainParameters<1, 3> parameters("/Overlap test/Source/");
  Domain<1, 3>           domain(parameters, source);
  const auto             support = domain.support(42);
  FE_Nothing<1, 3>       geometry;
  FEValues<1, 3> values(mapping, geometry, QGauss<1>(2), update_JxW_values);
  unsigned int   selected = 0;
  double         measure  = 0.;
  for (const auto &cell : source.active_cell_iterators())
    if (cell->is_locally_owned() && support.includes_cell(cell))
      {
        ++selected;
        values.reinit(cell);
        for (const auto q : values.quadrature_point_indices())
          measure += values.JxW(q);
      }
  EXPECT_EQ(Utilities::MPI::sum(selected, MPI_COMM_WORLD), 3u);
  // Material selection integrates entire retained cells, including partials.
  EXPECT_NEAR(Utilities::MPI::sum(measure, MPI_COMM_WORLD), 2., 1.e-12);
}

TEST(OverlapClassification, BOTH_PhysicalWallSamplesAndEmptySourceRank)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, 0., 1.);
  background.refine_global(1);
  MappingQ1<3>        background_mapping;
  GridTools::Cache<3> cache(background, background_mapping);

  Triangulation<1, 3> coarse;
  GridGenerator::hyper_cube(coarse, .25, .75);
  GridTools::shift(Tensor<1, 3>({0., .9, .5}), coarse);
  parallel::fullydistributed::Triangulation<1, 3> source(MPI_COMM_WORLD);
  distribute(coarse, source, true);
  unsigned int calls = 0;
  set_overlap_material_ids(
    source,
    cache,
    [&](const auto &cell) {
      ++calls;
      const auto center = cell->center();
      // The centreline is inside; one of these wall points is outside.
      std::vector<Point<3>> points(4, center);
      points[0][1] += .2;
      points[1][1] -= .2;
      points[2][2] += .2;
      points[3][2] -= .2;
      return points;
    },
    labels);
  for (const auto &cell : source.active_cell_iterators())
    if (!cell->is_artificial())
      {
        EXPECT_EQ(cell->material_id(), labels.partial);
      }
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  EXPECT_EQ(calls, rank == 0 ? 1u : 0u);
}

TEST(OverlapClassification, MPI_CrossPartitionAndRepeatedClassification)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, 0., 1.);
  background.refine_global(1);
  MappingQ1<3>        mapping;
  GridTools::Cache<3> cache(background, mapping);
  Point<3>            local_center;
  for (const auto &cell : background.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        local_center = cell->center();
        break;
      }
  const auto centers = Utilities::MPI::all_gather(MPI_COMM_WORLD, local_center);
  Triangulation<1, 3> coarse;
  GridGenerator::hyper_cube(coarse, 0., 1.);
  parallel::fullydistributed::Triangulation<1, 3> source(MPI_COMM_WORLD);
  distribute(coarse, source, true);

  for (const auto id : {labels.inside, labels.partial, labels.outside})
    {
      set_overlap_material_ids(
        source,
        cache,
        [&](const auto &) {
          std::vector<Point<3>> points;
          if (id != labels.outside)
            points.push_back(centers[1]);
          if (id != labels.inside)
            points.push_back(Point<3>(-2., -2., -2.));
          return points;
        },
        labels);
      for (const auto &cell : source.active_cell_iterators())
        if (!cell->is_artificial())
          {
            EXPECT_EQ(cell->material_id(), id);
          }
    }
}

TEST(OverlapClassification, BOTH_BoundaryPointsAndNoOverlap)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, 0., 1.);
  background.refine_global(1);
  MappingQ1<3>        mapping;
  GridTools::Cache<3> cache(background, mapping);
  Triangulation<1, 3> coarse;
  make_line(coarse);
  parallel::fullydistributed::Triangulation<1, 3> source(MPI_COMM_WORLD);
  distribute(coarse, source);
  set_overlap_material_ids(
    source,
    cache,
    [](const auto &) {
      return std::vector<Point<3>>{{0., 0., 0.}, {.5, .5, .5}, {1., 1., 1.}};
    },
    labels);
  for (const auto &cell : source.active_cell_iterators())
    if (!cell->is_artificial())
      {
        EXPECT_EQ(cell->material_id(), labels.inside);
      }
  set_overlap_material_ids(
    source,
    cache,
    [](const auto &) {
      return std::vector<Point<3>>{{-1., -1., -1.}, {2., 2., 2.}};
    },
    labels);
  for (const auto &cell : source.active_cell_iterators())
    if (!cell->is_artificial())
      {
        EXPECT_EQ(cell->material_id(), labels.outside);
      }
}

TEST(OverlapClassification, BOTH_SimplexSourceWithAnalyticBoxOracle)
{
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, 0., 1.);
  background.refine_global(1);
  MappingQ1<2>        background_mapping;
  GridTools::Cache<2> cache(background, background_mapping);
  Triangulation<2>    hypercubes, triangles;
  GridGenerator::subdivided_hyper_cube(hypercubes, 3, -.5, 1.5);
  GridGenerator::convert_hypercube_to_simplex_mesh(hypercubes, triangles);
  parallel::fullydistributed::Triangulation<2> source(MPI_COMM_WORLD);
  distribute(triangles, source);
  FE_SimplexP<2>   geometry(1);
  MappingFE<2>     mapping(geometry);
  QGaussSimplex<2> quadrature(2);
  set_overlap_material_ids(source, mapping, quadrature, cache, labels);
  std::array<unsigned int, 3> counts{};
  for (const auto &cell : source.active_cell_iterators())
    if (!cell->is_artificial())
      {
        unsigned int inside = 0;
        for (const auto &unit : quadrature.get_points())
          {
            const auto p = cell->vertex(0) * (1. - unit[0] - unit[1]) +
                           cell->vertex(1) * unit[0] +
                           cell->vertex(2) * unit[1];
            inside += p[0] >= 0. && p[0] <= 1. && p[1] >= 0. && p[1] <= 1.;
          }
        const auto expected =
          inside == 0 ?
            labels.outside :
            (inside == quadrature.size() ? labels.inside : labels.partial);
        EXPECT_EQ(cell->material_id(), expected);
        if (cell->is_locally_owned())
          {
            counts[0] += expected == labels.outside;
            counts[1] += expected == labels.partial;
            counts[2] += expected == labels.inside;
          }
      }
  for (const auto count : counts)
    EXPECT_GT(Utilities::MPI::sum(count, MPI_COMM_WORLD), 0u);
}

TEST(OverlapClassification, BOTH_InvalidSamplesFailCollectively)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, 0., 1.);
  background.refine_global(1);
  MappingQ1<3>        mapping;
  GridTools::Cache<3> cache(background, mapping);
  Triangulation<1, 3> coarse;
  make_line(coarse);
  parallel::fullydistributed::Triangulation<1, 3> source(MPI_COMM_WORLD);
  distribute(coarse, source);
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  EXPECT_THROW(set_overlap_material_ids(
                 source,
                 cache,
                 [&](const auto &) {
                   return std::vector<Point<3>>{
                     {rank == 0 ? std::numeric_limits<double>::quiet_NaN() : .5,
                      .5,
                      .5}};
                 },
                 labels),
               ExceptionBase);
  EXPECT_THROW(set_overlap_material_ids(
                 source,
                 cache,
                 [&](const auto &) {
                   return rank == 0 ? std::vector<Point<3>>{} :
                                      std::vector<Point<3>>{{.5, .5, .5}};
                 },
                 labels),
               ExceptionBase);
  EXPECT_THROW(set_overlap_material_ids(
                 source,
                 cache,
                 [](const auto &) {
                   return std::vector<Point<3>>{{.5, .5, .5}};
                 },
                 OverlapMaterialIds{1, 1, 2}),
               ExceptionBase);
  EXPECT_THROW(set_overlap_material_ids(
                 source,
                 cache,
                 [](const auto &) {
                   return std::vector<Point<3>>{{.5, .5, .5}};
                 },
                 labels,
                 rank == 0 ? -1. : 0.),
               ExceptionBase);
  for (const auto &cell : source.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        EXPECT_EQ(cell->material_id(), 99u);
      }
}

TEST(OverlapClassification, SerialMeshes)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background, 0., 1.);
  MappingQ1<3>        background_mapping;
  GridTools::Cache<3> cache(background, background_mapping);
  Triangulation<1, 3> source;
  make_line(source);
  MappingQ1<1, 3> mapping;
  set_overlap_material_ids(source, mapping, QGauss<1>(2), cache, labels);
  for (const auto &cell : source.active_cell_iterators())
    EXPECT_EQ(cell->material_id(), expected_line_label(cell, labels));
}

TEST(OverlapClassification, MPI_EmptyBackgroundRank)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  Triangulation<3> coarse_background;
  GridGenerator::hyper_cube(coarse_background, 0., 1.);
  parallel::fullydistributed::Triangulation<3> background(MPI_COMM_WORLD);
  distribute(coarse_background, background, true);
  MappingQ1<3>        background_mapping;
  GridTools::Cache<3> cache(background, background_mapping);

  Triangulation<1, 3> coarse_source;
  GridGenerator::hyper_cube(coarse_source, .25, .75);
  GridTools::shift(Tensor<1, 3>({0., .5, .5}), coarse_source);
  coarse_source.begin_active()->set_subdomain_id(1);
  parallel::fullydistributed::Triangulation<1, 3> source(MPI_COMM_WORLD);
  source.create_triangulation(
    TriangulationDescription::Utilities::create_description_from_triangulation(
      coarse_source, MPI_COMM_WORLD));
  MappingQ1<1, 3> source_mapping;
  set_overlap_material_ids(source, source_mapping, QGauss<1>(2), cache, labels);
  for (const auto &cell : source.active_cell_iterators())
    if (!cell->is_artificial())
      {
        EXPECT_EQ(cell->material_id(), labels.inside);
      }
}

TEST(OverlapClassification, BOTH_EmptyBackground)
{
  parallel::fullydistributed::Triangulation<3> background(MPI_COMM_WORLD);
  MappingQ1<3>                                 background_mapping;
  GridTools::Cache<3> cache(background, background_mapping);
  Triangulation<1, 3> coarse;
  make_line(coarse);
  parallel::fullydistributed::Triangulation<1, 3> source(MPI_COMM_WORLD);
  distribute(coarse, source);
  MappingQ1<1, 3> mapping;
  set_overlap_material_ids(source, mapping, QGauss<1>(2), cache, labels);
  for (const auto &cell : source.active_cell_iterators())
    if (!cell->is_artificial())
      {
        EXPECT_EQ(cell->material_id(), labels.outside);
      }
}


TEST(OverlapClassification, BOTH_TranslatedHyperBallsWithVertexQuadrature)
{
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<2> source(MPI_COMM_WORLD);
  GridGenerator::hyper_ball(background);
  GridGenerator::hyper_ball(source);
  background.refine_global(2);
  source.refine_global(2);
  const Tensor<1, 2> translation({.5, .25});
  GridTools::shift(translation, source);
  EXPECT_EQ(source.n_global_active_cells(), background.n_global_active_cells());

  // The inward half-planes describe the convex, discrete MappingQ1 boundary.
  std::vector<std::array<double, 4>> local_planes;
  for (const auto &cell : background.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto face : cell->face_indices())
        if (cell->face(face)->at_boundary())
          {
            const auto   a       = cell->face(face)->vertex(0);
            const auto   b       = cell->face(face)->vertex(1);
            const auto   tangent = b - a;
            Tensor<1, 2> inward({-tangent[1], tangent[0]});
            if (inward * (cell->center() - a) < 0.)
              inward *= -1.;
            inward /= inward.norm();
            local_planes.push_back({{a[0], a[1], inward[0], inward[1]}});
          }
  std::vector<std::array<double, 4>> planes;
  for (const auto &batch :
       Utilities::MPI::all_gather(MPI_COMM_WORLD, local_planes))
    planes.insert(planes.end(), batch.begin(), batch.end());
  ASSERT_FALSE(planes.empty());
  const auto inside_background = [&](const Point<2> &point) {
    return std::all_of(planes.begin(), planes.end(), [&](const auto &plane) {
      return plane[2] * (point[0] - plane[0]) +
               plane[3] * (point[1] - plane[1]) >=
             -1.e-12;
    });
  };

  MappingQ1<2>        mapping;
  GridTools::Cache<2> cache(background, mapping);
  // This rule classifies sampled vertices; it does not certify cell
  // intersection.
  const QTrapezoid<2> quadrature;
  set_overlap_material_ids(source, mapping, quadrature, cache, labels);
  std::array<unsigned int, 3> counts{};
  unsigned int                ghosts = 0;
  for (const auto &cell : source.active_cell_iterators())
    if (!cell->is_artificial())
      {
        unsigned int inside_vertices = 0;
        for (const auto vertex : cell->vertex_indices())
          inside_vertices += inside_background(cell->vertex(vertex));
        const auto expected =
          inside_vertices == 0 ?
            labels.outside :
            (inside_vertices == quadrature.size() ? labels.inside :
                                                    labels.partial);
        EXPECT_EQ(cell->material_id(), expected)
          << "Cell " << cell->id().to_string() << " has " << inside_vertices
          << " vertices in the background";
        ghosts += cell->is_ghost();
        if (cell->is_locally_owned())
          {
            counts[0] += inside_vertices == 0;
            counts[1] +=
              inside_vertices > 0 && inside_vertices < quadrature.size();
            counts[2] += inside_vertices == quadrature.size();
          }
      }
  for (const auto count : counts)
    EXPECT_GT(Utilities::MPI::sum(count, MPI_COMM_WORLD), 0u);
  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
    {
      EXPECT_GT(Utilities::MPI::sum(ghosts, MPI_COMM_WORLD), 0u);
    }
}

TEST(OverlapClassification, BOTH_ComplementaryQuadratureSupports)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background);
  background.refine_global(1);
  Triangulation<1, 3> coarse;
  make_line(coarse);
  parallel::fullydistributed::Triangulation<1, 3> source(MPI_COMM_WORLD);
  distribute(coarse, source);
  MappingQ1<1, 3>     mapping;
  MappingQ1<3>        background_mapping;
  GridTools::Cache<3> cache(background, background_mapping);
  const QGauss<1>     quadrature(3);
  const auto supports = make_overlap_supports(IntegrationSupport<1, 3>(source),
                                              mapping,
                                              quadrature,
                                              cache);
  const FE_Nothing<1, 3> geometry;
  FEValues<1, 3>         values(mapping,
                        geometry,
                        quadrature,
                        update_quadrature_points | update_JxW_values);
  unsigned int           partial = 0;
  for (const auto &cell : source.active_cell_iterators())
    if (!cell->is_artificial())
      {
        EXPECT_EQ(cell->material_id(), 99);
        ImmersX::detail::CouplingPoint<3> material_point;
        material_point.source_cell_id = cell->id();
        EXPECT_TRUE((IntegrationSupport<1, 3>(source, 99)
                       .includes_lifted_point(material_point)));
        EXPECT_FALSE((IntegrationSupport<1, 3>(source, 7).includes_lifted_point(
          material_point)));
        values.reinit(cell);
        unsigned int inside = 0;
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            const bool located = values.quadrature_point(q)[0] >= 0. &&
                                 values.quadrature_point(q)[0] <= 1.;
            EXPECT_DOUBLE_EQ(supports.overlap.point_weight(cell, q),
                             located ? 1. : 0.);
            EXPECT_DOUBLE_EQ(supports.outside.point_weight(cell, q),
                             located ? 0. : 1.);
            EXPECT_DOUBLE_EQ(supports.overlap.full_point_weight(cell, q), 1.);
            inside += located;
          }
        EXPECT_EQ(supports.overlap.includes_cell(cell), inside > 0);
        EXPECT_EQ(supports.outside.includes_cell(cell),
                  inside < quadrature.size());
        partial +=
          cell->is_locally_owned() && inside > 0 && inside < quadrature.size();
      }
  EXPECT_GT(Utilities::MPI::sum(partial, MPI_COMM_WORLD), 0u);
}

TEST(OverlapClassification, BOTH_LiftedQuadratureSupportsKeepPhysicalWeights)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background);
  Triangulation<1, 3> coarse;
  make_line(coarse);
  parallel::fullydistributed::Triangulation<1, 3> source(MPI_COMM_WORLD);
  distribute(coarse, source);
  MappingQ1<1, 3>     mapping;
  MappingQ1<3>        background_mapping;
  GridTools::Cache<3> cache(background, background_mapping);
  const QGauss<1>     quadrature(2);
  struct Sample
  {
    Point<3>                 point;
    double                   weight;
    types::global_cell_index source_entity_id;
    unsigned int             representative_qpoint;
    unsigned int             section_qpoint;
  };
  const FE_Nothing<1, 3> geometry;
  FEValues<1, 3>         values(mapping,
                        geometry,
                        quadrature,
                        update_quadrature_points | update_JxW_values);
  std::vector<Sample>    samples;
  for (const auto &cell : source.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        values.reinit(cell);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          for (unsigned int section = 0; section < 2; ++section)
            {
              auto point = values.quadrature_point(q);
              point[1]   = section == 0 ? .5 : 1.5;
              samples.push_back({point,
                                 values.JxW(q) * (section + 1.),
                                 cell->global_active_cell_index(),
                                 q,
                                 section});
            }
      }
  const auto supports = make_overlap_supports(
    IntegrationSupport<1, 3>(source), mapping, quadrature, samples, cache);
  for (const auto &sample : samples)
    {
      const bool located =
        sample.point[0] >= 0. && sample.point[0] <= 1. && sample.point[1] <= 1.;
      EXPECT_EQ(supports.overlap.includes_lifted_point(sample), located);
      EXPECT_EQ(supports.outside.includes_lifted_point(sample), !located);
    }
  for (const auto &cell : source.active_cell_iterators())
    if (!cell->is_artificial())
      {
        values.reinit(cell);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            const bool located = values.quadrature_point(q)[0] >= 0. &&
                                 values.quadrature_point(q)[0] <= 1.;
            EXPECT_DOUBLE_EQ(supports.overlap.point_weight(cell, q),
                             located ? 1. : 0.);
            EXPECT_DOUBLE_EQ(supports.outside.point_weight(cell, q),
                             located ? 2. : 3.);
            EXPECT_DOUBLE_EQ(supports.overlap.full_point_weight(cell, q), 3.);
          }
      }
}

TEST(OverlapClassification, BOTH_RejectIncompleteOrDuplicateQuadraturePartition)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background);
  Triangulation<1, 3> coarse;
  make_line(coarse);
  parallel::fullydistributed::Triangulation<1, 3> source(MPI_COMM_WORLD);
  distribute(coarse, source, true);
  MappingQ1<1, 3>     mapping;
  MappingQ1<3>        background_mapping;
  GridTools::Cache<3> cache(background, background_mapping);
  const QGauss<1>     quadrature(2);
  struct Sample
  {
    Point<3>                 point;
    double                   weight;
    types::global_cell_index source_entity_id;
    unsigned int             representative_qpoint;
    unsigned int             section_qpoint = 0;
  };
  const FE_Nothing<1, 3> geometry;
  FEValues<1, 3>         values(mapping,
                        geometry,
                        quadrature,
                        update_quadrature_points | update_JxW_values);
  std::vector<Sample>    full;
  for (const auto &cell : source.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        values.reinit(cell);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          full.push_back({values.quadrature_point(q),
                          values.JxW(q),
                          cell->global_active_cell_index(),
                          q,
                          0});
      }
  for (unsigned int error = 0; error < 4; ++error)
    {
      auto samples = full;
      if (!samples.empty())
        {
          if (error == 0)
            samples.pop_back();
          else if (error == 1)
            samples.push_back(samples.front());
          else if (error == 2)
            samples.front().source_entity_id = numbers::invalid_unsigned_int;
          else if (error == 3)
            samples.front().section_qpoint = numbers::invalid_unsigned_int;
        }
      EXPECT_ANY_THROW(make_overlap_supports(
        IntegrationSupport<1, 3>(source), mapping, quadrature, samples, cache));
    }
}

TEST(OverlapClassification,
     BOTH_SampledMaskMatchesParticleOwnershipNearBoundary)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background);
  background.refine_global(1);
  Triangulation<1, 3> coarse;
  make_line(coarse);
  parallel::fullydistributed::Triangulation<1, 3> source(MPI_COMM_WORLD);
  distribute(coarse, source);
  MappingQ1<1, 3>             mapping;
  MappingQ1<3>                background_mapping;
  GridTools::Cache<3>         cache(background, background_mapping);
  const QGauss<1>             quadrature(1);
  const std::array<double, 9> coordinates{{-1.e-7,
                                           -1.e-11,
                                           -1.e-13,
                                           0.,
                                           .5,
                                           1.,
                                           1. + 1.e-13,
                                           1. + 1.e-11,
                                           1. + 1.e-7}};
  std::vector<ImmersX::detail::CouplingPoint<3>> samples;
  std::vector<Point<3>>                          positions;
  std::vector<types::particle_index>             ids;
  for (const auto &cell : source.active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int section = 0; section < coordinates.size(); ++section)
        {
          ImmersX::detail::CouplingPoint<3> sample;
          sample.point                 = Point<3>(coordinates[section], .5, .5);
          sample.weight                = 1.;
          sample.source_entity_id      = cell->global_active_cell_index();
          sample.representative_qpoint = 0;
          sample.section_qpoint        = section;
          sample.stable_id =
            sample.source_entity_id * coordinates.size() + section;
          positions.push_back(sample.point);
          ids.push_back(sample.stable_id);
          samples.push_back(sample);
        }
  const auto supports = make_overlap_supports(
    IntegrationSupport<1, 3>(source), mapping, quadrature, samples, cache);
  ParticleCouplingParameters<3> parameters("/Boundary ownership consistency/");
  ParticleCoupling<3>           particles(parameters);
  particles.initialize_particle_handler(background, background_mapping, 0);
  particles.insert_points(positions, {}, ids);
  std::vector<types::particle_index> local_ids;
  for (const auto &particle : particles.get_particles())
    local_ids.push_back(particle.get_id());
  std::set<types::particle_index> located_ids;
  for (const auto &rank_ids :
       Utilities::MPI::all_gather(MPI_COMM_WORLD, local_ids))
    located_ids.insert(rank_ids.begin(), rank_ids.end());
  for (const auto &sample : samples)
    EXPECT_EQ(supports.overlap.includes_lifted_point(sample),
              located_ids.count(sample.stable_id) != 0);
}
