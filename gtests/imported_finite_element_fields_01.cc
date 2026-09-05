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
#include <immersx/core/observable.h>
#include <immersx/io/imported_finite_element_fields.h>
#include <immersx/io/vtk_utils.h>
#include <immersx/physics/elastic_static.h>

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
    const auto fe_field = field.field();
    const auto frozen   = ImmersX::frozen(fe_field, field.coefficients());
    EXPECT_TRUE(frozen.is_frozen());
    EXPECT_TRUE(frozen.dependencies().empty());
    EXPECT_EQ(&fe_field.dof_handler(), &fields->dof_handler());
    EXPECT_EQ(fe_field.name(), field_name);
  }

  void
  check_lambda_sampling(const ImportedFiniteElementFields<3>::FieldView &lambda)
  {
    const auto fe_field = lambda.field();
    const auto frozen   = ImmersX::frozen(fe_field, lambda.coefficients());
    EXPECT_TRUE(frozen.is_frozen());
    EXPECT_TRUE(frozen.dependencies().empty());
    EXPECT_EQ(fe_field.extractor().component, 0u);
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
  EXPECT_NE(vector_field.field().extractor().component,
            scalar_field.field().extractor().component);
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
  const auto lambda = fields->field("lambda");
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
  EXPECT_EQ(&first.field().dof_handler(), &second.field().dof_handler());
  EXPECT_EQ(&first.field().space().dof_handler(),
            &second.field().space().dof_handler());
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
    const auto value  = fields->field("cell_value");
    const auto before = ImmersX::frozen(value.field(), value.coefficients());
    EXPECT_TRUE(before.is_frozen());
    for (const auto index : value.field().locally_owned_dofs())
      EXPECT_DOUBLE_EQ(value.coefficients()[index], 7.);

    problem.refine_global();

    const auto after = ImmersX::frozen(value.field(), value.coefficients());
    EXPECT_TRUE(after.is_frozen());
    for (const auto index : value.field().locally_owned_dofs())
      EXPECT_DOUBLE_EQ(value.coefficients()[index], 7.);
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
