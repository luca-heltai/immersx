// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_parser.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/core/boundary_conditions.h>
#include <immersx/grid/local_co_refinement.h>
#include <immersx/io/utils.h>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  template <int dim>
  void
  check_boundary()
  {
    ParameterAcceptor::clear();
    DomainParameters<dim> parameters("/Simplex boundary/");
    parameters.triangulation_type = "fullydistributed";
    parameters.name_of_grid       = TestPaths::data_filename(
      dim == 2 ? "tests/simplex_square.msh" : "tests/simplex_cube.msh");
    Domain<dim> domain(parameters);
    domain.make_grid();
    FiniteElementSpace<dim> space(domain.triangulation(),
                                  std::make_unique<FE_SimplexP<dim>>(2));
    auto                    exact = std::make_shared<FunctionParser<dim>>();
    exact->initialize(dim == 2 ? "x,y" : "x,y,z", "1+x*x+2*y", {});
    const auto view = space.view();
    const auto points =
      DoFTools::map_dofs_to_support_points(view.mapping(), space.dof_handler());
    const auto check = [&]() {
      unsigned int n_lines = 0;
      for (const auto &line : space.constraints().get_lines())
        {
          EXPECT_TRUE(line.entries.empty());
          EXPECT_NEAR(line.inhomogeneity,
                      exact->value(points.at(line.index)),
                      1.e-12);
          ++n_lines;
        }
      EXPECT_GT(Utilities::MPI::sum(n_lines, MPI_COMM_WORLD), 0u);
    };
    BoundaryConditions<dim> owned_conditions(space);
    owned_conditions.add_dirichlet(0, *exact);
    owned_conditions.update(0.);
    check();
    BoundaryConditions<dim> field_conditions(view.field("u"));
    field_conditions.add_dirichlet(0, exact);
    field_conditions.update(0.);
    check();
  }


  template <int dim>
  void
  check_flux()
  {
    ParameterAcceptor::clear();
    DomainParameters<dim> parameters("/Simplex normal flux/");
    parameters.triangulation_type = "fullydistributed";
    parameters.name_of_grid       = TestPaths::data_filename(
      dim == 2 ? "tests/simplex_square.msh" : "tests/simplex_cube.msh");
    Domain<dim> domain(parameters);
    domain.make_grid();
    FiniteElementSpace<dim> space(
      domain.triangulation(),
      std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(1), dim));
    Functions::ConstantFunction<dim> exact(std::vector<double>(dim, 1.));
    BoundaryConditions<dim>          conditions(space);
    conditions.add_nonzero_normal_flux(0, exact);
    conditions.update(0.);
    ImmersXLA::MPI::Vector solution;
    solution.reinit(space.locally_owned_dofs(), MPI_COMM_WORLD);
    solution = 1.;
    space.constraints().distribute(solution);
    for (const auto i : space.locally_owned_dofs())
      EXPECT_NEAR(solution[i], 1., 1.e-12);
    EXPECT_GT(Utilities::MPI::sum(space.constraints().n_constraints(),
                                  MPI_COMM_WORLD),
              0u);
  }

#ifdef DEAL_II_WITH_VTK
  template <int dim>
  void
  check_import()
  {
    ParameterAcceptor::clear();
    DomainParameters<dim> parameters("/Simplex import/");
    parameters.triangulation_type = "fullydistributed";
    parameters.name_of_grid       = TestPaths::data_filename(
      dim == 2 ? "tests/simplex_fields_triangle.vtk" :
                 "tests/simplex_fields_tetrahedron.vtk");
    parameters.read_vtk_fields    = true;
    parameters.initial_refinement = 1;
    Domain<dim> domain(parameters);
    domain.make_grid();
    EXPECT_TRUE(domain.triangulation().all_reference_cells_are_simplex());
    for (const auto &name : {"affine", "constant"})
      {
        const auto             imported = domain.field(name);
        const auto             field    = imported.field();
        ImmersXLA::MPI::Vector ghosted;
        ghosted.reinit(field.locally_owned_dofs(),
                       field.locally_relevant_dofs(),
                       MPI_COMM_WORLD);
        ghosted = imported.coefficients();
        ghosted.update_ghost_values();
        const auto quadrature = compatibility::gauss_quadrature<dim>(
          field.finite_element().reference_cell(), 2);
        FEValues<dim>       values(field.mapping(),
                             field.finite_element(),
                             quadrature,
                             update_values | update_quadrature_points);
        std::vector<double> result(quadrature.size());
        for (const auto &cell : field.dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              values.reinit(cell);
              values[field.extractor()].get_function_values(ghosted, result);
              for (unsigned int q = 0; q < result.size(); ++q)
                {
                  double expected = 7.;
                  if (std::string(name) == "affine")
                    {
                      expected = 1.;
                      for (unsigned int d = 0; d < dim; ++d)
                        expected += (d + 1) * values.quadrature_point(q)[d];
                    }
                  EXPECT_NEAR(result[q], expected, 1.e-12);
                }
            }
      }
  }
#endif
} // namespace

TEST(BoundaryConditions, BOTH_SimplexStrongValues)
{
  check_boundary<2>();
  check_boundary<3>();
}

#ifdef DEAL_II_WITH_VTK
TEST(Domain, BOTH_SimplexImportedFieldsAfterRefinement)
{
  check_import<2>();
  check_import<3>();
}
#endif

TEST(LocalCoRefinement, BOTH_SimplexBackground)
{
  ParameterAcceptor::clear();
  DomainParameters<2> bulk_parameters("/Simplex co-refinement/bulk/");
  bulk_parameters.triangulation_type = "fullydistributed";
  bulk_parameters.name_of_grid =
    TestPaths::data_filename("tests/simplex_square.msh");
  DomainParameters<1, 2> embedded_parameters(
    "/Simplex co-refinement/embedded/");
  embedded_parameters.triangulation_type = "fullydistributed";
  embedded_parameters.arguments_for_grid = "0.1:0.9:false";
  Domain<2>               bulk(bulk_parameters);
  Domain<1, 2>            embedded(embedded_parameters);
  RefinementParameters<1> refinement("/Simplex co-refinement/policy/");
  refinement.refinement_strategy          = "both";
  refinement.lower_factor                 = 0.15;
  refinement.upper_factor                 = 2.;
  refinement.space_post_refinement_cycles = 1;
  refinement.max_refinement_level         = 4;
  const auto report =
    LocalCoRefinement<2, 1, 2>(refinement).run(bulk, embedded);
  EXPECT_TRUE(report.satisfied());
  EXPECT_GT(report.intersecting_pairs, 0u);
  EXPECT_GT(bulk.triangulation().n_global_active_cells(), 8u);
  EXPECT_TRUE(bulk.triangulation().all_reference_cells_are_simplex());
}

TEST(LocalCoRefinement, BOTH_SimplexPointCloud)
{
  ParameterAcceptor::clear();
  Triangulation<2> cubes, triangles;
  GridGenerator::hyper_cube(cubes);
  GridGenerator::convert_hypercube_to_simplex_mesh(cubes, triangles);
  const auto              initial_cells = triangles.n_active_cells();
  RefinementParameters<0> parameters;
  parameters.space_pre_refinement_cycles  = 0;
  parameters.space_post_refinement_cycles = 1;
  parameters.max_refinement_level         = 2;
  refine_space_around_points(triangles,
                             std::vector<Point<2>>{{.25, .25}},
                             std::vector<double>{.1},
                             parameters,
                             MPI_COMM_WORLD);
  EXPECT_GT(triangles.n_active_cells(), initial_cells);
  EXPECT_TRUE(triangles.all_reference_cells_are_simplex());
}

TEST(BoundaryConditions, BOTH_SimplexNormalFlux)
{
  check_flux<2>();
  check_flux<3>();
}
