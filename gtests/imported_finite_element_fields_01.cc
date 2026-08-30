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

#include <array>
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

  void
  check_lambda_sampling(const ImportedFiniteElementFields<3>::FieldView &lambda)
  {
    const QGauss<3> quadrature(2);
    const auto      plan =
      make_retained_sampling_plan(lambda.representation(),
                                  quadrature,
                                  update_values | update_gradients);
    auto value_operator = plan.linearize(lambda.coefficients());
    std::array<ImmersXLA::MPI::Vector, 3> gradient_values;
    ImmersXLA::MPI::Vector                values;
    value_operator.reinit_range_vector(values, false);
    value_operator.vmult(values, lambda.coefficients());
    std::array<decltype(plan.linearize(lambda.coefficients())), 3>
      gradient_operators = {
        {plan.gradient_linearize(lambda.coefficients(), 0),
         plan.gradient_linearize(lambda.coefficients(), 1),
         plan.gradient_linearize(lambda.coefficients(), 2)}};
    for (unsigned int component = 0; component < 3; ++component)
      {
        gradient_operators[component].reinit_range_vector(
          gradient_values[component], false);
        gradient_operators[component].vmult(gradient_values[component],
                                            lambda.coefficients());
      }

    for (const auto &point : plan.points())
      {
        const auto index = plan.point_index(&point - plan.points().data());
        EXPECT_NEAR(values[index],
                    point.point[0] + point.point[1] + point.point[2],
                    1.e-12);
        for (unsigned int component = 0; component < 3; ++component)
          EXPECT_NEAR(gradient_values[component][index], 1., 1.e-12);
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
  check_lambda_sampling(lambda);
}

TEST(ImportedFiniteElementFields, FieldViewSurvivesParentHandle)
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
  const auto lambda = [&] {
    auto imported =
      std::make_shared<ImportedFiniteElementFields<3>>(filename,
                                                       problem.triangulation());
    return imported->field("lambda");
  }();

  check_lambda_sampling(lambda);
}

TEST(ImportedFiniteElementFields, SharedInstanceViewsUseSharedStorage)
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
  auto imported =
    std::make_shared<ImportedFiniteElementFields<3>>(filename,
                                                     problem.triangulation());
  const auto first  = imported->field("lambda");
  const auto second = imported->field("lambda");
  EXPECT_EQ(&first.coefficients(), &second.coefficients());
  EXPECT_EQ(&first.representation().dof_handler(),
            &second.representation().dof_handler());
  EXPECT_EQ(&first.representation().triangulation(),
            &second.representation().triangulation());
}

namespace
{
  void
  check_distributed_refinement()
  {
    ElasticStaticParameters<3> parameters;
    parameters.domain_type        = "generate";
    parameters.name_of_grid       = "hyper_cube";
    parameters.arguments_for_grid = "0: 1: false";
    parameters.initial_refinement = 0;
    parameters.triangulation_type = "distributed";
    ElasticStaticProblem<3> problem(parameters);
    problem.setup();

    const auto filename =
      TestPaths::data_filename("tests/imported_lambda_3d.vtk");
    auto fields =
      std::make_shared<ImportedFiniteElementFields<3>>(filename,
                                                       problem.triangulation());
    const auto  first        = fields->field("lambda");
    const auto  second       = fields->field("lambda");
    const auto *coefficients = &first.coefficients();
    const auto  n_dofs       = fields->dof_handler().n_dofs();

    problem.refine_global();

    EXPECT_GT(fields->dof_handler().n_dofs(), n_dofs);
    EXPECT_EQ(&first.coefficients(), coefficients);
    EXPECT_EQ(&second.coefficients(), coefficients);
    check_lambda_sampling(first);
    check_lambda_sampling(second);
  }

  void
  check_cell_data_refinement()
  {
    ElasticStaticParameters<3> parameters;
    parameters.domain_type        = "generate";
    parameters.name_of_grid       = "hyper_cube";
    parameters.arguments_for_grid = "0: 1: false";
    parameters.initial_refinement = 0;
    parameters.triangulation_type = "distributed";
    ElasticStaticProblem<3> problem(parameters);
    problem.setup();

    const auto filename =
      TestPaths::data_filename("tests/imported_cell_3d.vtk");
    auto fields =
      std::make_shared<ImportedFiniteElementFields<3>>(filename,
                                                       problem.triangulation());
    const auto      value = fields->field("cell_value");
    const QGauss<3> quadrature(2);
    const auto      before = make_retained_sampling_plan(value.representation(),
                                                    quadrature,
                                                    update_values);
    auto            before_values = before.linearize(value.coefficients());
    ImmersXLA::MPI::Vector before_vector;
    before_values.reinit_range_vector(before_vector, false);
    before_values.vmult(before_vector, value.coefficients());
    for (const auto index : before.locally_owned_points())
      EXPECT_DOUBLE_EQ(before_vector[index], 7.);

    problem.refine_global();

    const auto after = make_retained_sampling_plan(value.representation(),
                                                   quadrature,
                                                   update_values);
    auto       after_values = after.linearize(value.coefficients());
    ImmersXLA::MPI::Vector after_vector;
    after_values.reinit_range_vector(after_vector, false);
    after_values.vmult(after_vector, value.coefficients());
    for (const auto index : after.locally_owned_points())
      EXPECT_DOUBLE_EQ(after_vector[index], 7.);
  }
} // namespace

TEST(ImportedFiniteElementFields, DistributedRefinementTransfersPointData)
{
  check_distributed_refinement();
}

TEST(ImportedFiniteElementFields, MPI_DistributedRefinementTransfersPointData)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  check_distributed_refinement();
}

TEST(ImportedFiniteElementFields, DistributedRefinementTransfersCellData)
{
  check_cell_data_refinement();
}

TEST(ImportedFiniteElementFields, MPI_DistributedRefinementTransfersCellData)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  check_cell_data_refinement();
}
#endif
