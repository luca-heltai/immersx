// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <gtest/gtest.h>
#include <immersx/grid/local_co_refinement.h>

#include <array>
#include <fstream>
#include <random>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  // A direct all-pairs oracle, independent of the cached rtree queries.
  template <int dim>
  std::vector<std::array<double, 4>>
  boxes(const Domain<dim, 2> &domain)
  {
    std::vector<std::array<double, 4>> local;
    for (const auto &cell : domain.triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          std::array<double, 4> box{{1.e20, 1.e20, -1.e20, -1.e20}};
          for (const auto v : cell->vertex_indices())
            for (unsigned int d = 0; d < 2; ++d)
              {
                box[d]     = std::min(box[d], cell->vertex(v)[d]);
                box[d + 2] = std::max(box[d + 2], cell->vertex(v)[d]);
              }
          local.push_back(box);
        }
    std::vector<std::array<double, 4>> result;
    for (const auto &batch :
         Utilities::MPI::all_gather(domain.communicator(), local))
      result.insert(result.end(), batch.begin(), batch.end());
    return result;
  }

  template <int d1, int d2>
  LocalCoRefinementReport
  oracle(const Domain<d1, 2> &space,
         const Domain<d2, 2> &embedded,
         double               lower,
         double               upper)
  {
    LocalCoRefinementReport report;
    const auto              space_boxes    = boxes(space);
    const auto              embedded_boxes = boxes(embedded);
    for (const auto &s : space_boxes)
      for (const auto &e : embedded_boxes)
        if (s[0] <= e[2] && e[0] <= s[2] && s[1] <= e[3] && e[1] <= s[3])
          {
            const double hs = std::hypot(s[2] - s[0], s[3] - s[1]);
            const double he = std::hypot(e[2] - e[0], e[3] - e[1]);
            ++report.intersecting_pairs;
            report.lower_violations += he <= lower * hs;
            report.upper_violations += he >= upper * hs;
          }
    return report;
  }

  template <int d1, int d2>
  void
  check_report(const Domain<d1, 2>           &s,
               const Domain<d2, 2>           &e,
               const LocalCoRefinementReport &report,
               double                         lower,
               double                         upper)
  {
    const auto expected = oracle(s, e, lower, upper);
    EXPECT_EQ(report.intersecting_pairs, expected.intersecting_pairs);
    EXPECT_EQ(report.lower_violations, expected.lower_violations);
    EXPECT_EQ(report.upper_violations, expected.upper_violations);
  }

  auto
  random_tree_geometry()
  {
    std::mt19937             random(258);
    std::vector<Point<2>>    vertices;
    std::vector<CellData<1>> cells;
    std::vector<Point<2>>    parents{{0.5, 0.08}};
    for (unsigned int level = 0; level < 4; ++level)
      {
        std::vector<Point<2>> next;
        for (const auto &parent : parents)
          for (unsigned int branch = 0; branch < 2; ++branch)
            {
              const double dx =
                (branch == 0 ? -1. : 1.) * (0.18 / std::pow(2., level));
              const double dy =
                (level == 3 ? .02 : 0.21 - 0.035 * level) +
                (static_cast<int>(random() % 101) - 50) * 0.0003;
              const auto  child = parent + Point<2>(dx, dy);
              CellData<1> cell;
              cell.vertices[0] = vertices.size();
              vertices.push_back(parent);
              cell.vertices[1] = vertices.size();
              vertices.push_back(child);
              cells.push_back(cell);
              next.push_back(child);
            }
        parents = next;
      }
    return std::make_pair(vertices, cells);
  }

  std::string
  line_input(const std::string &name,
             const bool         tree    = false,
             const bool         reverse = false)
  {
    const auto path =
      TestPaths::output_path("local-co-refinement/" + name + ".vtk");
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::filesystem::create_directories(path.parent_path());
        std::vector<Point<2>> vertices{{0.15, 0.5}, {0.85, 0.5}};
        if (reverse)
          std::reverse(vertices.begin(), vertices.end());
        std::vector<std::array<unsigned int, 2>> cells{{{0, 1}}};
        if (tree)
          {
            const auto geometry = random_tree_geometry();
            vertices            = geometry.first;
            cells.clear();
            for (const auto &cell : geometry.second)
              cells.push_back({{cell.vertices[0], cell.vertices[1]}});
          }
        std::ofstream out(path);
        out.precision(17);
        out
          << "# vtk DataFile Version 3.0\nCo-refinement input\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS "
          << vertices.size() << " double\n";
        for (const auto &v : vertices)
          out << v[0] << ' ' << v[1] << " 0\n";
        out << "CELLS " << cells.size() << ' ' << 3 * cells.size() << '\n';
        for (const auto &c : cells)
          out << "2 " << c[0] << ' ' << c[1] << '\n';
        out << "CELL_TYPES " << cells.size() << '\n';
        for (unsigned int i = 0; i < cells.size(); ++i)
          out << "3\n";
        out << "CELL_DATA " << cells.size()
            << "\nSCALARS material int 1\nLOOKUP_TABLE default\n";
        for (unsigned int i = 0; i < cells.size(); ++i)
          out << "7\n";
        out << "SCALARS cell_property double 1\nLOOKUP_TABLE default\n";
        for (unsigned int i = 0; i < cells.size(); ++i)
          out << 3.5 + i << '\n';
        out << "POINT_DATA " << vertices.size()
            << "\nSCALARS affine double 1\nLOOKUP_TABLE default\n";
        for (const auto &v : vertices)
          out << 1. + v[0] + 2. * v[1] << '\n';
      }
    MPI_Barrier(MPI_COMM_WORLD);
    return path.string();
  }

  std::string
  square_input(const std::string &name)
  {
    const auto path =
      TestPaths::output_path("local-co-refinement/" + name + ".vtk");
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path);
        out
          << "# vtk DataFile Version 3.0\nCo-refinement input\nASCII\nDATASET UNSTRUCTURED_GRID\n"
             "POINTS 4 double\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n"
             "CELLS 1 5\n4 0 1 2 3\nCELL_TYPES 1\n9\n"
             "CELL_DATA 1\nSCALARS material int 1\nLOOKUP_TABLE default\n9\n"
             "SCALARS cell_property double 1\nLOOKUP_TABLE default\n6.5\n"
             "POINT_DATA 4\nSCALARS affine double 1\nLOOKUP_TABLE default\n1 2 4 3\n";
      }
    MPI_Barrier(MPI_COMM_WORLD);
    return path.string();
  }

  template <int dim>
  void
  check_fields(const Domain<dim, 2>    &domain,
               const double             cell_value,
               const types::material_id material)
  {
    const auto             point      = domain.field("affine");
    const auto             cell_field = domain.field("cell_property");
    const QGauss<dim>      quadrature(2);
    FEValues<dim, 2>       values(point.space().mapping(),
                            point.space().finite_element(),
                            quadrature,
                            update_values | update_quadrature_points);
    ImmersXLA::MPI::Vector ghosted;
    ghosted.reinit(point.field().locally_owned_dofs(),
                   point.field().locally_relevant_dofs(),
                   domain.communicator());
    ghosted = point.coefficients();
    ghosted.update_ghost_values();
    const FEValuesExtractors::Scalar point_extractor(
      point.descriptor().first_fe_component);
    const FEValuesExtractors::Scalar cell_extractor(
      cell_field.descriptor().first_fe_component);
    std::vector<double> point_values(quadrature.size()),
      cell_values(quadrature.size());
    for (const auto &cell : point.field().dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          EXPECT_EQ(cell->material_id(), material);
          values.reinit(cell);
          values[point_extractor].get_function_values(ghosted, point_values);
          values[cell_extractor].get_function_values(ghosted, cell_values);
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              const auto &p = values.quadrature_point(q);
              EXPECT_NEAR(point_values[q], 1. + p[0] + 2. * p[1], 1.e-12);
              EXPECT_NEAR(cell_values[q], cell_value, 1.e-12);
            }
        }
  }
} // namespace

TEST(LocalCoRefinement, BOTH_StrategiesAndFinalReports)
{
  for (const std::string strategy : {"space", "embedded", "both"})
    {
      DomainParameters<2> ps("/Tests/CoRefinement/Space/");
      ps.arguments_for_grid = "0: 1: false";
      ps.initial_refinement = strategy == "embedded" ? 3 : 0;
      DomainParameters<1, 2> pe("/Tests/CoRefinement/Embedded/");
      pe.name_of_grid = line_input("strategies");
      Domain<2>               space(ps);
      Domain<1, 2>            embedded(pe);
      RefinementParameters<1> parameters;
      parameters.refinement_strategy            = strategy;
      parameters.embedded_pre_refinement_cycles = strategy == "space" ? 2 : 0;
      const auto report =
        LocalCoRefinement<2, 1, 2>(parameters).run(space, embedded);
      check_report(space,
                   embedded,
                   report,
                   parameters.lower_factor,
                   parameters.upper_factor);
      EXPECT_TRUE(report.satisfied());
      EXPECT_GT(report.intersecting_pairs, 0u);
      if (strategy == "space")
        {
          EXPECT_EQ(embedded.triangulation().n_global_active_cells(), 4u);
        }
      else if (strategy == "embedded")
        {
          EXPECT_EQ(space.triangulation().n_global_active_cells(), 64u);
        }
    }
}

TEST(LocalCoRefinement, BOTH_TransfersBothDomainsAndDefersFullyDistributed)
{
  for (const std::string backend : {"distributed", "fullydistributed"})
    {
      DomainParameters<2> ps("/Tests/CoTransfer/Space/");
      ps.name_of_grid          = square_input("transfer-square");
      ps.read_vtk_fields       = true;
      ps.vtk_material_id_field = "material";
      ps.triangulation_type    = backend;
      ps.initial_refinement    = 1;
      DomainParameters<1, 2> pe("/Tests/CoTransfer/Embedded/");
      pe.name_of_grid          = line_input("transfer-line");
      pe.read_vtk_fields       = true;
      pe.vtk_material_id_field = "material";
      pe.initial_refinement    = 1;
      Domain<2>    space(ps);
      Domain<1, 2> embedded(pe);
      // Set boundary ids before partitioning either mesh. The discontinuous
      // field test separately exercises refinement of a distributed space.
      space.prepare_grid(true);
      embedded.prepare_grid(true);
      EXPECT_EQ(embedded.triangulation().n_global_active_cells(), 0u);
      EXPECT_EQ(space.triangulation().n_global_active_cells(), 0u);
      for (const auto &cell :
           space.refinement_triangulation().active_cell_iterators())
        if (!cell->is_artificial())
          for (const auto &face : cell->face_iterators())
            if (face->at_boundary())
              face->set_boundary_id(12);
      RefinementParameters<1> parameters;
      parameters.refinement_strategy             = "both";
      parameters.space_pre_refinement_cycles     = 1;
      parameters.embedded_pre_refinement_cycles  = 1;
      parameters.space_post_refinement_cycles    = 1;
      parameters.embedded_post_refinement_cycles = 1;
      const auto report =
        LocalCoRefinement<2, 1, 2>(parameters).run(space, embedded);
      EXPECT_TRUE(report.satisfied());
      check_report(space,
                   embedded,
                   report,
                   parameters.lower_factor,
                   parameters.upper_factor);
      check_fields(space, 6.5, 9);
      check_fields(embedded, 3.5, 7);
      for (const auto &cell : space.triangulation().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const auto &face : cell->face_iterators())
            if (face->at_boundary())
              {
                EXPECT_EQ(face->boundary_id(), 12u);
              }
      EXPECT_FALSE(space.grid_is_prepared());
      EXPECT_FALSE(embedded.grid_is_prepared());
      EXPECT_TRUE(embedded.uses_fully_distributed_triangulation());
    }
}

TEST(LocalCoRefinement, BOTH_ReversedIntrinsicDimensions)
{
  DomainParameters<1, 2> ps("/Tests/CoReverse/Space/");
  ps.name_of_grid       = line_input("reverse-line");
  ps.initial_refinement = 2;
  DomainParameters<2> pe("/Tests/CoReverse/Embedded/");
  pe.arguments_for_grid = "0: 1: false";
  Domain<1, 2>            space(ps);
  Domain<2>               embedded(pe);
  RefinementParameters<2> parameters;
  parameters.refinement_strategy = "both";
  const auto report =
    LocalCoRefinement<1, 2, 2>(parameters).run(space, embedded);
  EXPECT_TRUE(report.satisfied());
  check_report(
    space, embedded, report, parameters.lower_factor, parameters.upper_factor);
  EXPECT_GT(embedded.triangulation().n_global_active_cells(), 1u);
}

TEST(LocalCoRefinement, BOTH_ReportsUnresolvedViolations)
{
  DomainParameters<2> ps("/Tests/CoCap/Space/");
  ps.arguments_for_grid = "0: 1: false";
  DomainParameters<1, 2> pe("/Tests/CoCap/Embedded/");
  pe.name_of_grid       = line_input("cap-line");
  pe.initial_refinement = 3;
  for (const std::string strategy : {"embedded", "both"})
    {
      Domain<2>               space(ps);
      Domain<1, 2>            embedded(pe);
      RefinementParameters<1> parameters;
      parameters.refinement_strategy  = strategy;
      parameters.max_refinement_level = 0;
      const auto report =
        LocalCoRefinement<2, 1, 2>(parameters).run(space, embedded);
      EXPECT_FALSE(report.satisfied());
      EXPECT_GT(report.lower_violations, 0u);
      EXPECT_EQ(space.triangulation().n_global_active_cells(), 1u);
      check_report(space,
                   embedded,
                   report,
                   parameters.lower_factor,
                   parameters.upper_factor);
    }
}

TEST(LocalCoRefinement, BOTH_PostRefinementIsLocalAndIndependentOfStrategy)
{
  DomainParameters<2> ps("/Tests/CoPost/Space/");
  ps.arguments_for_grid = "0: 1: false";
  ps.triangulation_type = "fullydistributed";
  DomainParameters<1, 2> pe("/Tests/CoPost/Embedded/");
  pe.name_of_grid = line_input("post-line");
  Domain<2>               space(ps);
  Domain<1, 2>            embedded(pe);
  RefinementParameters<1> parameters;
  parameters.refinement_strategy             = "embedded";
  parameters.max_refinement_level            = 0;
  parameters.space_pre_refinement_cycles     = 2;
  parameters.embedded_pre_refinement_cycles  = 1;
  parameters.space_post_refinement_cycles    = 1;
  parameters.embedded_post_refinement_cycles = 1;
  const auto report =
    LocalCoRefinement<2, 1, 2>(parameters).run(space, embedded);
  EXPECT_EQ(space.triangulation().n_global_active_cells(), 40u);
  EXPECT_EQ(embedded.triangulation().n_global_active_cells(), 4u);
  for (const auto &cell : space.triangulation().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        EXPECT_EQ(cell->level(),
                  cell->center()[1] > .25 && cell->center()[1] < .75 ? 3 : 2);
      }
  check_report(
    space, embedded, report, parameters.lower_factor, parameters.upper_factor);
}

TEST(LocalCoRefinement, RejectsInvalidFactorsAndFinalizedDomains)
{
  DomainParameters<2>    ps("/Tests/CoInvalid/Space/");
  DomainParameters<1, 2> pe("/Tests/CoInvalid/Embedded/");
  pe.name_of_grid = line_input("invalid-line");
  for (const double lower :
       {-1., 0., 2., std::numeric_limits<double>::infinity()})
    {
      Domain<2>               space(ps);
      Domain<1, 2>            embedded(pe);
      RefinementParameters<1> parameters;
      parameters.lower_factor = lower;
      EXPECT_THROW(
        (LocalCoRefinement<2, 1, 2>(parameters).run(space, embedded)),
        ExceptionBase);
    }
  Domain<2> space(ps);
  space.make_grid();
  Domain<1, 2>            embedded(pe);
  RefinementParameters<1> parameters;
  EXPECT_THROW((LocalCoRefinement<2, 1, 2>(parameters).run(space, embedded)),
               ExceptionBase);
}

TEST(LocalCoRefinement, ParsesIndependentParameterGroups)
{
  ParameterAcceptor::clear();
  RefinementParameters<1> first("/CoGroups/First/");
  RefinementParameters<1> second("/CoGroups/Second/");
  ParameterAcceptor::initialize("", "");
  ParameterAcceptor::prm.parse_input_from_string(
    "subsection CoGroups\n"
    " subsection First\n"
    "  set Refinement strategy = both\n"
    "  set Lower factor = 0.25\n"
    "  set Embedded partitioning policy = immersed\n"
    " end\n"
    " subsection Second\n"
    "  set Refinement strategy = embedded\n"
    "  set Upper factor = 3\n"
    " end\n"
    "end\n");
  ParameterAcceptor::parse_all_parameters();
  EXPECT_EQ(first.embedded_partitioning_policy, "immersed");
  EXPECT_EQ(second.embedded_partitioning_policy, "default");
  EXPECT_EQ(first.refinement_strategy, "both");
  EXPECT_DOUBLE_EQ(first.lower_factor, 0.25);
  EXPECT_DOUBLE_EQ(first.upper_factor, 2.);
  EXPECT_EQ(second.refinement_strategy, "embedded");
  EXPECT_DOUBLE_EQ(second.lower_factor, 0.5);
  EXPECT_DOUBLE_EQ(second.upper_factor, 3.);
}

TEST(LocalCoRefinement, ParsesParametersAndChecksEquality)
{
  ParameterAcceptor::clear();
  RefinementParameters<1> parameters;
  ParameterAcceptor::initialize("", "");
  ParameterAcceptor::prm.parse_input_from_string(
    "subsection Local refinement parameters\n"
    " set Refinement strategy = both\n"
    " set Lower factor = 0.5\n"
    " set Upper factor = 2.5\n"
    "end\n");
  ParameterAcceptor::parse_all_parameters();
  EXPECT_EQ(parameters.refinement_strategy, "both");
  EXPECT_DOUBLE_EQ(parameters.upper_factor, 2.5);

  Triangulation<1, 2> space, embedded;
  GridGenerator::hyper_cube(space, 0., 1.);
  GridGenerator::hyper_cube(embedded, 0., 1.);
  parameters.refinement_strategy = "space";
  parameters.lower_factor        = 1.;
  parameters.upper_factor        = 3.;
  const auto report              = ImmersX::internal::co_refine_meshes(space,
                                                          embedded,
                                                          parameters,
                                                          MPI_COMM_WORLD);
  EXPECT_EQ(space.n_active_cells(), 2u);
  EXPECT_EQ(embedded.n_active_cells(), 1u);
  EXPECT_TRUE(report.satisfied());

  parameters.refinement_strategy = "embedded";
  parameters.lower_factor        = .25;
  parameters.upper_factor        = 2.;
  const auto second              = ImmersX::internal::co_refine_meshes(space,
                                                          embedded,
                                                          parameters,
                                                          MPI_COMM_WORLD);
  EXPECT_EQ(embedded.n_active_cells(), 2u);
  EXPECT_TRUE(second.satisfied());
}

TEST(LocalCoRefinement, BOTH_DiscontinuousFieldsOnLocallyRefinedMesh)
{
  const auto path =
    TestPaths::output_path("local-co-refinement/discontinuous.vtk");
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::filesystem::create_directories(path.parent_path());
      std::ofstream out(path);
      out
        << "# vtk DataFile Version 3.0\nTwo materials\nASCII\nDATASET UNSTRUCTURED_GRID\n"
           "POINTS 6 double\n0 0 0\n0.5 0 0\n1 0 0\n0 1 0\n0.5 1 0\n1 1 0\n"
           "CELLS 2 10\n4 0 1 4 3\n4 1 2 5 4\nCELL_TYPES 2\n9 9\n"
           "CELL_DATA 2\nSCALARS discontinuous double 1\nLOOKUP_TABLE default\n3.5 8.5\n"
           "SCALARS material int 1\nLOOKUP_TABLE default\n7 9\n"
           "POINT_DATA 6\nVECTORS position double\n0 0 0\n0.5 0 0\n1 0 0\n0 1 0\n0.5 1 0\n1 1 0\n"
           "VECTORS other_position double\n0 0 1\n1 0 1\n2 0 1\n0 3 1\n1 3 1\n2 3 1\n";
    }
  MPI_Barrier(MPI_COMM_WORLD);
  for (const std::string backend : {"distributed", "fullydistributed"})
    {
      DomainParameters<2> ps("/Tests/CoDG/Space/");
      ps.name_of_grid          = path.string();
      ps.triangulation_type    = backend;
      ps.read_vtk_fields       = true;
      ps.vtk_material_id_field = "material";
      SCOPED_TRACE(backend);
      DomainParameters<1, 2> pe("/Tests/CoDG/Embedded/");
      pe.name_of_grid       = line_input("dg-line");
      pe.initial_refinement = 3;
      Domain<2>    space(ps);
      Domain<1, 2> embedded(pe);
      space.prepare_grid();
      for (const auto &cell :
           space.refinement_triangulation().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            EXPECT_EQ(cell->material_id(), cell->center()[0] < .5 ? 7u : 9u);
          }
      RefinementParameters<1> parameters;
      parameters.refinement_strategy = "both";
      const auto report =
        LocalCoRefinement<2, 1, 2>(parameters).run(space, embedded);
      EXPECT_TRUE(report.satisfied());
      const auto scalar   = space.field("discontinuous");
      const auto position = space.field("position");
      EXPECT_EQ(position.descriptor().n_components, 3u);
      const auto other_position = space.field("other_position");
      EXPECT_EQ(other_position.descriptor().n_components, 3u);
      ImmersXLA::MPI::Vector ghosted;
      ghosted.reinit(scalar.field().locally_owned_dofs(),
                     scalar.field().locally_relevant_dofs(),
                     MPI_COMM_WORLD);
      ghosted = scalar.coefficients();
      ghosted.update_ghost_values();
      QGauss<2>           quadrature(2);
      FEValues<2>         values(scalar.space().mapping(),
                         scalar.space().finite_element(),
                         quadrature,
                         update_values | update_quadrature_points);
      std::vector<double> actual(quadrature.size());
      for (const auto &cell :
           scalar.field().dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            values.reinit(cell);
            values[FEValuesExtractors::Scalar(
                     scalar.descriptor().first_fe_component)]
              .get_function_values(ghosted, actual);
            const bool left = cell->center()[0] < .5;
            EXPECT_EQ(cell->material_id(), left ? 7u : 9u);
            for (const auto value : actual)
              EXPECT_NEAR(value, left ? 3.5 : 8.5, 1.e-12);
            for (unsigned int d = 0; d < 3; ++d)
              {
                values[FEValuesExtractors::Scalar(
                         position.descriptor().first_fe_component + d)]
                  .get_function_values(ghosted, actual);
                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  EXPECT_NEAR(actual[q],
                              d < 2 ? values.quadrature_point(q)[d] : 0.,
                              1.e-12);
                values[FEValuesExtractors::Scalar(
                         other_position.descriptor().first_fe_component + d)]
                  .get_function_values(ghosted, actual);
                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  EXPECT_NEAR(actual[q],
                              d < 2 ? (d + 2) * values.quadrature_point(q)[d] :
                                      1.,
                              1.e-12);
              }
          }
      check_report(space,
                   embedded,
                   report,
                   parameters.lower_factor,
                   parameters.upper_factor);
    }
}

TEST(LocalCoRefinement, DocumentationExamples)
{
  struct Example
  {
    std::string  name, strategy;
    double       lower, upper;
    unsigned int space_pre, embedded_pre, space_post, embedded_post;
    int          cap;
  };
  const std::vector<Example> examples = {
    {"space", "space", .5, 2., 1, 0, 0, 0, 7},
    {"embedded", "embedded", .5, 2., 4, 0, 0, 0, 7},
    {"both", "both", .5, 2., 2, 0, 0, 0, 7},
    {"both-wide", "both", .2, 3., 2, 0, 0, 0, 7},
    {"both-post", "both", .5, 2., 2, 1, 1, 1, 7},
    {"both-capped", "both", .5, 2., 0, 2, 0, 0, 1}};
  const std::filesystem::path output =
    TestPaths::output_directory("local-co-refinement/examples");
  std::filesystem::create_directories(output);
  std::ofstream summary(output / "summary.csv");
  summary
    << "example,strategy,lower,upper,space_pre,embedded_pre,space_post,embedded_post,cap,space_cells,embedded_cells,pairs,lower_violations,upper_violations\n";
  line_input("random-tree", true);
  const auto geometry = random_tree_geometry();
  for (const auto &example : examples)
    {
      DomainParameters<2> ps("/Tests/CoExamples/Space/");
      ps.arguments_for_grid = "0: 1: false";
      ps.triangulation_type = "fullydistributed";
      DomainParameters<1, 2> pe("/Tests/CoExamples/Embedded/");
      Domain<2>              space(ps);
      Domain<1, 2>           embedded(pe);
      embedded.prepare_grid(true);
      auto &tree = embedded.refinement_triangulation();
      tree.clear();
      // Codimension-one orientation checks reject this tree's triple junctions.
      // Branches touch geometrically but use distinct endpoint indices.
      tree.create_triangulation(geometry.first, geometry.second, SubCellData());
      RefinementParameters<1> parameters;
      parameters.refinement_strategy             = example.strategy;
      parameters.lower_factor                    = example.lower;
      parameters.upper_factor                    = example.upper;
      parameters.space_pre_refinement_cycles     = example.space_pre;
      parameters.embedded_pre_refinement_cycles  = example.embedded_pre;
      parameters.space_post_refinement_cycles    = example.space_post;
      parameters.embedded_post_refinement_cycles = example.embedded_post;
      parameters.max_refinement_level            = example.cap;
      const auto report =
        LocalCoRefinement<2, 1, 2>(parameters).run(space, embedded);
      check_report(space, embedded, report, example.lower, example.upper);
      EXPECT_GT(report.intersecting_pairs, 0u);
      if (example.name == "both" || example.name == "both-wide" ||
          example.name == "both-post")
        {
          EXPECT_TRUE(report.satisfied());
        }
      if (example.name == "both-capped")
        {
          EXPECT_FALSE(report.satisfied());
        }
      summary << example.name << ',' << example.strategy << ',' << example.lower
              << ',' << example.upper << ',' << example.space_pre << ','
              << example.embedded_pre << ',' << example.space_post << ','
              << example.embedded_post << ',' << example.cap << ','
              << space.triangulation().n_global_active_cells() << ','
              << embedded.triangulation().n_global_active_cells() << ','
              << report.intersecting_pairs << ',' << report.lower_violations
              << ',' << report.upper_violations << '\n';
      GridOut       grid_out;
      std::ofstream space_vtk(output / (example.name + "-space.vtk"));
      std::ofstream embedded_vtk(output / (example.name + "-embedded.vtk"));
      grid_out.write_vtk(space.triangulation(), space_vtk);
      grid_out.write_vtk(embedded.triangulation(), embedded_vtk);
      // The image is itself a test output, drawn from the same final cell
      // vertices as the VTK files. No separate refinement implementation.
      std::ofstream svg(output / (example.name + ".svg"));
      svg.precision(12);
      svg
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"-0.03 -0.03 1.06 1.06\">\n"
           "<rect x=\"-0.03\" y=\"-0.03\" width=\"1.06\" height=\"1.06\" fill=\"white\"/>\n"
           "<g transform=\"translate(0 1) scale(1 -1)\">\n";
      const auto line = [&](const Point<2> &a,
                            const Point<2> &b,
                            const char     *color,
                            double          width) {
        svg << "<line x1=\"" << a[0] << "\" y1=\"" << a[1] << "\" x2=\"" << b[0]
            << "\" y2=\"" << b[1] << "\" stroke=\"" << color
            << "\" stroke-width=\"" << width << "\"/>\n";
      };
      for (const auto &cell : space.triangulation().active_cell_iterators())
        for (const auto &face : cell->face_iterators())
          line(face->vertex(0), face->vertex(1), "#a0a8b0", .0015);
      for (const auto &cell : embedded.triangulation().active_cell_iterators())
        {
          line(cell->vertex(0), cell->vertex(1), "#006b45", .004);
          for (const auto v : cell->vertex_indices())
            svg << "<circle cx=\"" << cell->vertex(v)[0] << "\" cy=\""
                << cell->vertex(v)[1] << "\" r=\"0.005\" fill=\"#006b45\"/>\n";
        }
      svg << "</g>\n</svg>\n";
    }
}

namespace
{
  // The fixtures have axis-aligned background cells. Test ownership by
  // exhaustive geometric containment, independently of RemotePointEvaluation.
  template <int dim, int spacedim>
  unsigned int
  ownership_mismatches(const Domain<spacedim>      &space,
                       const Domain<dim, spacedim> &embedded)
  {
    std::vector<std::array<double, 2 * spacedim + 1>> local;
    const unsigned int                                rank =
      Utilities::MPI::this_mpi_process(space.communicator());
    for (const auto &cell : space.triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto box = cell->bounding_box().get_boundary_points();
          std::array<double, 2 * spacedim + 1> entry;
          for (unsigned int d = 0; d < spacedim; ++d)
            {
              entry[d]            = box.first[d];
              entry[d + spacedim] = box.second[d];
            }
          entry[2 * spacedim] = rank;
          local.push_back(entry);
        }
    const auto batches =
      Utilities::MPI::all_gather(space.communicator(), local);
    unsigned int mismatches = 0;
    for (const auto &cell : embedded.triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto   center   = cell->center();
          unsigned int expected = numbers::invalid_unsigned_int;
          for (const auto &batch : batches)
            for (const auto &box : batch)
              {
                bool contains = true;
                for (unsigned int d = 0; d < spacedim; ++d)
                  contains = contains && center[d] >= box[d] - 1.e-12 &&
                             center[d] <= box[d + spacedim] + 1.e-12;
                if (contains)
                  expected =
                    std::min(expected,
                             static_cast<unsigned int>(box[2 * spacedim]));
              }
          EXPECT_NE(expected, numbers::invalid_unsigned_int);
          mismatches += expected != rank;
        }
    return Utilities::MPI::sum(mismatches, space.communicator());
  }
} // namespace

TEST(LocalCoRefinement, BOTH_ImmersedOwnershipPreservesFields)
{
  unsigned int default_mismatches = 0;
  for (const std::string backend : {"distributed", "fullydistributed"})
    for (const bool reverse : {false, true})
      for (const std::string policy : {"default", "immersed"})
        {
          SCOPED_TRACE(backend + "/" + policy +
                       (reverse ? "/reverse" : "/forward"));
          DomainParameters<2> ps("/Tests/CoPartition/Space/");
          ps.name_of_grid          = square_input("partition-square");
          ps.triangulation_type    = backend;
          ps.read_vtk_fields       = true;
          ps.vtk_material_id_field = "material";
          ps.initial_refinement    = 2;
          DomainParameters<1, 2> pe("/Tests/CoPartition/Embedded/");
          pe.name_of_grid    = line_input("partition-line", false, reverse);
          pe.read_vtk_fields = true;
          pe.vtk_material_id_field = "material";
          pe.initial_refinement    = 1;
          Domain<2>    space(ps);
          Domain<1, 2> embedded(pe);
          embedded.prepare_grid(true);
          auto &prepared = embedded.refinement_triangulation();
          prepared.set_manifold(17, FlatManifold<1, 2>());
          for (const auto &cell : prepared.active_cell_iterators())
            {
              cell->set_all_manifold_ids(17);
              for (const auto &face : cell->face_iterators())
                if (face->at_boundary())
                  face->set_boundary_id(13);
            }
          RefinementParameters<1> parameters;
          parameters.refinement_strategy             = "both";
          parameters.embedded_partitioning_policy    = policy;
          parameters.embedded_post_refinement_cycles = 1;
          const auto report =
            LocalCoRefinement<2, 1, 2>(parameters).run(space, embedded);
          check_report(space,
                       embedded,
                       report,
                       parameters.lower_factor,
                       parameters.upper_factor);
          EXPECT_TRUE(report.satisfied());
          const auto mismatches = ownership_mismatches(space, embedded);
          if (policy == "immersed")
            EXPECT_EQ(mismatches, 0u);
          else if (policy == "default")
            default_mismatches += mismatches;
          check_fields(space, 6.5, 9);
          check_fields(embedded, 3.5, 7);
          for (const auto &cell :
               embedded.triangulation().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                EXPECT_EQ(cell->manifold_id(), 17u);
                EXPECT_NE((dynamic_cast<const FlatManifold<1, 2> *>(
                            &embedded.triangulation().get_manifold(17))),
                          nullptr);
                for (const auto &face : cell->face_iterators())
                  if (face->at_boundary())
                    {
                      EXPECT_EQ(face->boundary_id(), 13u);
                    }
              }
        }
  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 2)
    {
      EXPECT_GT(default_mismatches, 0u);
    }
}

TEST(LocalCoRefinement, RejectsUnsupportedPartitioningBeforePreparation)
{
  DomainParameters<2> ps("/Tests/CoPartitionInvalid/Space/");
  DomainParameters<2> pe("/Tests/CoPartitionInvalid/Embedded/");
  pe.triangulation_type = "distributed";
  Domain<2>               space(ps), embedded(pe);
  RefinementParameters<2> parameters;
  parameters.embedded_partitioning_policy = "immersed";
  EXPECT_THROW((LocalCoRefinement<2, 2>(parameters).run(space, embedded)),
               ExceptionBase);
  EXPECT_FALSE(space.grid_is_prepared());
  EXPECT_FALSE(embedded.grid_is_prepared());
  EXPECT_EQ(space.triangulation().n_global_active_cells(), 0u);
  EXPECT_EQ(embedded.triangulation().n_global_active_cells(), 0u);
  DomainParameters<1, 2> pl("/Tests/CoPartitionInvalid/Line/");
  Domain<1, 2>           line(pl);
  pe.triangulation_type = "fullydistributed";
  Domain<2> fully_embedded(pe);
  EXPECT_THROW(
    (LocalCoRefinement<1, 2, 2>(parameters).run(line, fully_embedded)),
    ExceptionBase);
  EXPECT_FALSE(line.grid_is_prepared());
  EXPECT_FALSE(fully_embedded.grid_is_prepared());
  parameters.embedded_partitioning_policy = "invalid";
  EXPECT_THROW(parameters.diameter_bounds(), ExceptionBase);
}

TEST(LocalCoRefinement, BOTH_ImmersedPartitionSupportsTwoDimensionalCells)
{
  DomainParameters<2> ps("/Tests/CoPartition2D/Space/");
  ps.initial_refinement = 2;
  DomainParameters<2> pe("/Tests/CoPartition2D/Embedded/");
  pe.name_of_grid          = square_input("partition-square-2d");
  pe.triangulation_type    = "fullydistributed";
  pe.initial_refinement    = 1;
  pe.read_vtk_fields       = true;
  pe.vtk_material_id_field = "material";
  Domain<2>               space(ps), embedded(pe);
  RefinementParameters<2> parameters;
  parameters.refinement_strategy          = "both";
  parameters.embedded_partitioning_policy = "immersed";
  const auto report = LocalCoRefinement<2, 2>(parameters).run(space, embedded);
  EXPECT_TRUE(report.satisfied());
  EXPECT_EQ(ownership_mismatches(space, embedded), 0u);
  check_fields(embedded, 6.5, 9);
}

TEST(LocalCoRefinement, BOTH_ImmersedPartitionHandlesEmptyRanks)
{
  DomainParameters<2> ps("/Tests/CoPartitionEmpty/Space/");
  ps.triangulation_type = "fullydistributed";
  DomainParameters<1, 2> pe("/Tests/CoPartitionEmpty/Embedded/");
  pe.name_of_grid          = line_input("partition-empty-line");
  pe.read_vtk_fields       = true;
  pe.vtk_material_id_field = "material";
  pe.initial_refinement    = 2;
  Domain<2>               space(ps);
  Domain<1, 2>            embedded(pe);
  RefinementParameters<1> parameters;
  parameters.lower_factor                 = 0.001;
  parameters.upper_factor                 = 10.;
  parameters.embedded_partitioning_policy = "immersed";
  const auto report =
    LocalCoRefinement<2, 1, 2>(parameters).run(space, embedded);
  EXPECT_TRUE(report.satisfied());
  EXPECT_EQ(ownership_mismatches(space, embedded), 0u);
  EXPECT_EQ(embedded.triangulation().n_global_active_cells(), 4u);
  check_fields(embedded, 3.5, 7);
  const unsigned int empty =
    embedded.triangulation().n_locally_owned_active_cells() == 0;
  EXPECT_EQ(Utilities::MPI::sum(empty, MPI_COMM_WORLD),
            Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1);
}

TEST(LocalCoRefinement, BOTH_ImmersedVesselInThreeDimensions)
{
  for (const std::string backend : {"distributed", "fullydistributed"})
    {
      DomainParameters<3> ps("/Tests/CoPartition3D/Space/");
      ps.arguments_for_grid = "0: 1: false";
      ps.initial_refinement = 2;
      ps.triangulation_type = backend;
      DomainParameters<1, 3> pe("/Tests/CoPartition3D/Embedded/");
      pe.arguments_for_grid = "0.15: 0.85: false";
      pe.initial_refinement = 1;
      Domain<3>    space(ps);
      Domain<1, 3> embedded(pe);
      embedded.prepare_grid(true);
      GridTools::transform(
        [](const Point<3> &p) { return Point<3>(p[0], .37, .41); },
        embedded.refinement_triangulation());
      RefinementParameters<1> parameters;
      parameters.refinement_strategy          = "both";
      parameters.embedded_partitioning_policy = "immersed";
      const auto report =
        LocalCoRefinement<3, 1>(parameters).run(space, embedded);
      EXPECT_TRUE(report.satisfied());
      EXPECT_EQ(ownership_mismatches(space, embedded), 0u);
    }
}

TEST(LocalCoRefinement, BOTH_ImmersedPartitionOfObservedDomain)
{
  DomainParameters<2> ps("/Tests/CoPartitionObserved/Space/");
  ps.initial_refinement = 2;
  DomainParameters<1, 2> pe("/Tests/CoPartitionObserved/Embedded/");
  pe.name_of_grid          = line_input("partition-observed-line");
  pe.read_vtk_fields       = true;
  pe.vtk_material_id_field = "material";
  parallel::fullydistributed::Triangulation<1, 2> owned_by_problem(
    MPI_COMM_WORLD);
  Domain<2>               space(ps);
  Domain<1, 2>            embedded(pe, owned_by_problem);
  RefinementParameters<1> parameters;
  parameters.refinement_strategy          = "both";
  parameters.embedded_partitioning_policy = "immersed";
  const auto report =
    LocalCoRefinement<2, 1, 2>(parameters).run(space, embedded);
  EXPECT_TRUE(report.satisfied());
  EXPECT_EQ(&embedded.triangulation(), &owned_by_problem);
  EXPECT_EQ(ownership_mismatches(space, embedded), 0u);
  check_fields(embedded, 3.5, 7);
}
