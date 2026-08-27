// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <gtest/gtest.h>
#include <immersx/io/imported_finite_element_fields.h>
#include <immersx/io/vtk_utils.h>
#include <immersx/physics/elastic_static.h>

#include <cmath>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

#ifdef DEAL_II_WITH_VTK
namespace
{
  template <int dim, int spacedim>
  std::unique_ptr<parallel::fullydistributed::Triangulation<dim, spacedim>>
  make_distributed_mesh(const std::string &filename)
  {
    Triangulation<dim, spacedim> serial_tria;
    DoFHandler<dim, spacedim>    serial_dh(serial_tria);
    Vector<double>               data;
    std::vector<std::string>     names;
    VTKUtils::read_vtk(filename, serial_dh, data, names);
    GridTools::partition_triangulation(
      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), serial_tria);
    const auto description = TriangulationDescription::Utilities::
      create_description_from_triangulation(serial_tria, MPI_COMM_WORLD);
    auto result = std::make_unique<
      parallel::fullydistributed::Triangulation<dim, spacedim>>(MPI_COMM_WORLD);
    result->create_triangulation(description);
    return result;
  }

  template <int dim, int spacedim>
  void
  check_scalar_import(const std::string &filename,
                      const std::string &field_name)
  {
    auto mesh = make_distributed_mesh<dim, spacedim>(filename);
    auto fields =
      std::make_shared<ImportedFiniteElementFields<dim, spacedim>>(filename,
                                                                   *mesh);
    const auto field        = fields->field(field_name);
    const auto same_storage = fields->field(field_name);
    EXPECT_EQ(&field.coefficients(), &same_storage.coefficients());
    EXPECT_EQ(field.name(), field_name);
    EXPECT_GE(fields->catalog().size(), 1u);

    const QGauss<dim> quadrature(2);
    const auto points = field.representation().locally_owned_quadrature_points(
      quadrature, update_values | update_gradients);
    const auto plan =
      make_retained_sampling_plan(field.representation(),
                                  quadrature,
                                  update_values | update_gradients);
    auto value_operator    = plan.linearize(field.coefficients());
    auto gradient_operator = plan.gradient_linearize(field.coefficients(), 0);
    ImmersXLA::MPI::Vector values;
    ImmersXLA::MPI::Vector gradients;
    value_operator.reinit_range_vector(values, false);
    gradient_operator.reinit_range_vector(gradients, false);
    value_operator.vmult(values, field.coefficients());
    gradient_operator.vmult(gradients, field.coefficients());

    EXPECT_EQ(points.size(),
              quadrature.size() * mesh->n_locally_owned_active_cells());
    for (const auto &point : points)
      EXPECT_TRUE(std::isfinite(point.point[0]));
    for (const auto index : plan.locally_owned_points())
      {
        EXPECT_TRUE(std::isfinite(values[index]));
        EXPECT_TRUE(std::isfinite(gradients[index]));
      }
  }
} // namespace

TEST(ImportedFiniteElementFields, PointDataScalarAndGradient)
{
  check_scalar_import<1, 3>(TestPaths::data_filename(
                              "tests/one_cylinder_properties.vtk"),
                            "path_distance");
}

TEST(ImportedFiniteElementFields, CellDataScalar)
{
  check_scalar_import<1, 3>(TestPaths::data_filename("tests/mstree_10.vtk"),
                            "edge_length");
}

TEST(ImportedFiniteElementFields, ComponentLookupSharesStorage)
{
  const auto filename = TestPaths::data_filename("tests/simple_1d_grid.vtk");
  auto       mesh     = make_distributed_mesh<1, 3>(filename);
  auto       fields =
    std::make_shared<ImportedFiniteElementFields<1, 3>>(filename, *mesh);
  const auto vector_field = fields->field("xyz", 1);
  const auto scalar_field = fields->field("x");
  EXPECT_EQ(vector_field.descriptor().n_components, 3u);
  EXPECT_EQ(vector_field.component(), 1u);
  EXPECT_EQ(&vector_field.coefficients(), &scalar_field.coefficients());
  EXPECT_NE(&vector_field.representation().extractor(),
            &scalar_field.representation().extractor());
}

TEST(ImportedFiniteElementFields, MPI_SharedStorageAndCellData)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  check_scalar_import<1, 3>(TestPaths::data_filename("tests/mstree_10.vtk"),
                            "edge_length");
}

TEST(ImportedFiniteElementFields, ElasticityConsumer)
{
  ElasticStaticParameters<3> parameters;
  parameters.domain_type        = "generate";
  parameters.name_of_grid       = "hyper_cube";
  parameters.arguments_for_grid = "0: 1: false";
  parameters.initial_refinement = 0;
  parameters.triangulation_type = "fullydistributed";
  ElasticStaticProblem<3> problem(parameters);
  problem.setup();

  const auto filename =
    TestPaths::data_filename("tests/imported_lambda_3d.vtk");
  auto fields =
    std::make_shared<ImportedFiniteElementFields<3>>(filename,
                                                     problem.triangulation());
  problem.set_imported_fields(fields);
  const auto lambda = problem.imported_fields()->field("lambda");
  EXPECT_EQ(lambda.name(), "lambda");
  EXPECT_EQ(&lambda.coefficients(), &fields->coefficients());
}
#endif
