#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/core/domain.h>

#include <fstream>
#include <set>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

TEST(Domain, GeneratesNamedGrid)
{
  DomainParameters<2> parameters("/Tests/Domain/");
  parameters.name_of_grid       = "hyper_cube";
  parameters.arguments_for_grid = "0: 1: false";
  parameters.initial_refinement = 1;

  Domain<2> domain(parameters);
  ASSERT_NO_THROW(domain.make_grid());
  EXPECT_EQ(domain.triangulation().n_global_active_cells(), 4u);
}


TEST(Domain, CopiesToSerialTriangulation)
{
  DomainParameters<2> parameters("/Tests/DomainTriangulation/");
  parameters.name_of_grid       = "hyper_cube";
  parameters.arguments_for_grid = "0: 1: false";
  parameters.initial_refinement = 1;

  Domain<2> domain(parameters);
  domain.make_grid();

  dealii::Triangulation<2> triangulation;
  ASSERT_NO_THROW(triangulation.copy_triangulation(domain.triangulation()));
  EXPECT_EQ(triangulation.n_active_cells(), 4u);
}


TEST(Domain, MPI_CopiesToSerialTriangulation)
{
  DomainParameters<2> parameters("/Tests/DomainMPITriangulation/");
  parameters.name_of_grid       = "hyper_cube";
  parameters.arguments_for_grid = "0: 1: false";
  parameters.initial_refinement = 1;

  Domain<2> domain(parameters);
  domain.make_grid();

  dealii::Triangulation<2> triangulation;
  ASSERT_NO_THROW(triangulation.copy_triangulation(domain.triangulation()));
  EXPECT_EQ(triangulation.n_active_cells(),
            domain.triangulation().n_active_cells());
}


TEST(Domain, ReadsLegacyVtkGrid)
{
  const auto filename = TestPaths::output_path("domain/quad.vtk");
  std::filesystem::create_directories(filename.parent_path());

  std::ofstream output(filename);
  output << "# vtk DataFile Version 3.0\n"
         << "Domain test\n"
         << "ASCII\n"
         << "DATASET UNSTRUCTURED_GRID\n"
         << "POINTS 4 double\n"
         << "0 0 0  1 0 0  1 1 0  0 1 0\n"
         << "CELLS 1 5\n"
         << "4 0 1 2 3\n"
         << "CELL_TYPES 1\n"
         << "9\n";
  output.close();

  DomainParameters<2> parameters("/Tests/DomainVtk/");
  parameters.name_of_grid = filename.string();

  Domain<2> domain(parameters);
  ASSERT_NO_THROW(domain.make_grid());
  EXPECT_EQ(domain.triangulation().n_global_active_cells(), 1u);
}


TEST(Domain, ImportsVtkFieldsAndIds)
{
  const auto filename = TestPaths::output_path("domain/quad-properties.vtk");
  std::filesystem::create_directories(filename.parent_path());

  std::ofstream output(filename);
  output << "# vtk DataFile Version 3.0\n"
         << "Domain property test\n"
         << "ASCII\n"
         << "DATASET UNSTRUCTURED_GRID\n"
         << "POINTS 4 double\n"
         << "0 0 0  1 0 0  1 1 0  0 1 0\n"
         << "CELLS 1 5\n"
         << "4 0 1 2 3\n"
         << "CELL_TYPES 1\n"
         << "9\n"
         << "CELL_DATA 1\n"
         << "SCALARS material int 1\n"
         << "LOOKUP_TABLE default\n"
         << "7\n"
         << "SCALARS cell_property double 1\n"
         << "LOOKUP_TABLE default\n"
         << "3.5\n"
         << "POINT_DATA 4\n"
         << "SCALARS vertex_property double 1\n"
         << "LOOKUP_TABLE default\n"
         << "1 2 3 4\n";
  output.close();

  DomainParameters<2> parameters("/Tests/DomainProperties/");
  parameters.name_of_grid          = filename.string();
  parameters.read_vtk_fields       = true;
  parameters.vtk_material_id_field = "material";

  Domain<2> domain(parameters);
  ASSERT_NO_THROW(domain.make_grid());
  ASSERT_TRUE(domain.has_vtk_fields());
  EXPECT_EQ(domain.triangulation().begin_active()->material_id(), 7u);
  EXPECT_EQ(domain.vtk_fields().catalog().size(), 3u);
  EXPECT_EQ(domain.field("vertex_property").descriptor().n_components, 1u);
  EXPECT_EQ(domain.field("cell_property").descriptor().association,
            FieldAssociation::cell_data);
}


TEST(Domain, TransfersImportedFieldsAcrossInitialRefinement)
{
  const auto filename = TestPaths::output_path("domain/quad-transfer.vtk");
  std::filesystem::create_directories(filename.parent_path());

  std::ofstream output(filename);
  output << "# vtk DataFile Version 3.0\n"
         << "Domain transfer test\n"
         << "ASCII\n"
         << "DATASET UNSTRUCTURED_GRID\n"
         << "POINTS 4 double\n"
         << "0 0 0  1 0 0  1 1 0  0 1 0\n"
         << "CELLS 1 5\n"
         << "4 0 1 2 3\n"
         << "CELL_TYPES 1\n"
         << "9\n"
         << "CELL_DATA 1\n"
         << "SCALARS cell_property double 1\n"
         << "LOOKUP_TABLE default\n"
         << "3.5\n";
  output.close();

  DomainParameters<2> parameters("/Tests/DomainTransfer/");
  parameters.name_of_grid       = filename.string();
  parameters.read_vtk_fields    = true;
  parameters.initial_refinement = 1;

  Domain<2> domain(parameters);
  ASSERT_NO_THROW(domain.make_grid());
  ASSERT_TRUE(domain.has_vtk_fields());

  const auto field        = domain.field("cell_property");
  const auto coefficients = &field.coefficients();
  const auto n_dofs       = field.field().dof_handler().n_dofs();
  ASSERT_EQ(domain.triangulation().n_global_active_cells(), 4u);
  EXPECT_EQ(n_dofs, 4u);
  EXPECT_EQ(&field.coefficients(), coefficients);
  for (const auto index : field.field().locally_owned_dofs())
    EXPECT_DOUBLE_EQ(field.coefficients()[index], 3.5);
}


TEST(Domain, TransfersImportedFieldsAcrossLocalRefinement)
{
  const auto filename =
    TestPaths::output_path("domain/quad-local-transfer.vtk");
  std::filesystem::create_directories(filename.parent_path());

  std::ofstream output(filename);
  output << "# vtk DataFile Version 3.0\n"
         << "Domain local transfer test\n"
         << "ASCII\n"
         << "DATASET UNSTRUCTURED_GRID\n"
         << "POINTS 4 double\n"
         << "0 0 0  1 0 0  1 1 0  0 1 0\n"
         << "CELLS 1 5\n"
         << "4 0 1 2 3\n"
         << "CELL_TYPES 1\n"
         << "9\n"
         << "CELL_DATA 1\n"
         << "SCALARS cell_property double 1\n"
         << "LOOKUP_TABLE default\n"
         << "3.5\n";
  output.close();

  DomainParameters<2> parameters("/Tests/DomainLocalTransfer/");
  parameters.name_of_grid    = filename.string();
  parameters.read_vtk_fields = true;

  Domain<2> domain(parameters);
  ASSERT_NO_THROW(domain.make_grid());
  ASSERT_TRUE(domain.has_vtk_fields());

  const auto field         = domain.field("cell_property");
  const auto before_n_dofs = field.field().dof_handler().n_dofs();
  auto      &tria = dynamic_cast<parallel::distributed::Triangulation<2> &>(
    domain.triangulation());
  tria.begin_active()->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  EXPECT_GT(field.field().dof_handler().n_dofs(), before_n_dofs);
  for (const auto index : field.field().locally_owned_dofs())
    EXPECT_DOUBLE_EQ(field.coefficients()[index], 3.5);
}


TEST(Domain, ReadsIdsWithoutRetainingVtkFields)
{
  const auto filename = TestPaths::output_path("domain/quad-ids-only.vtk");
  std::filesystem::create_directories(filename.parent_path());

  std::ofstream output(filename);
  output << "# vtk DataFile Version 3.0\n"
         << "Domain ids-only test\n"
         << "ASCII\n"
         << "DATASET UNSTRUCTURED_GRID\n"
         << "POINTS 4 double\n"
         << "0 0 0  1 0 0  1 1 0  0 1 0\n"
         << "CELLS 1 5\n"
         << "4 0 1 2 3\n"
         << "CELL_TYPES 1\n"
         << "9\n"
         << "CELL_DATA 1\n"
         << "SCALARS material double 1\n"
         << "LOOKUP_TABLE default\n"
         << "7\n";
  output.close();

  DomainParameters<2> parameters("/Tests/DomainIdsOnly/");
  parameters.name_of_grid          = filename.string();
  parameters.read_vtk_fields       = false;
  parameters.vtk_material_id_field = "material";

  Domain<2> domain(parameters);
  ASSERT_NO_THROW(domain.make_grid());
  EXPECT_FALSE(domain.has_vtk_fields());
  EXPECT_EQ(domain.triangulation().begin_active()->material_id(), 7u);
}


TEST(Domain, ReadsBoundaryIdsWithoutRetainingVtkFields)
{
  const auto filename = TestPaths::output_path("domain/quad-boundary-ids.vtk");
  std::filesystem::create_directories(filename.parent_path());

  std::ofstream output(filename);
  output << "# vtk DataFile Version 3.0\n"
         << "Domain boundary ids-only test\n"
         << "ASCII\n"
         << "DATASET UNSTRUCTURED_GRID\n"
         << "POINTS 4 double\n"
         << "0 0 0  1 0 0  1 1 0  0 1 0\n"
         << "CELLS 5 17\n"
         << "4 0 1 2 3\n"
         << "2 0 1\n"
         << "2 1 2\n"
         << "2 2 3\n"
         << "2 3 0\n"
         << "CELL_TYPES 5\n"
         << "9\n"
         << "3\n"
         << "3\n"
         << "3\n"
         << "3\n"
         << "CELL_DATA 5\n"
         << "SCALARS boundary int 1\n"
         << "LOOKUP_TABLE default\n"
         << "0 11 12 13 14\n";
  output.close();

  DomainParameters<2> parameters("/Tests/DomainBoundaryIdsOnly/");
  parameters.name_of_grid          = filename.string();
  parameters.read_vtk_fields       = false;
  parameters.vtk_boundary_id_field = "boundary";

  Domain<2> domain(parameters);
  ASSERT_NO_THROW(domain.make_grid());
  EXPECT_FALSE(domain.has_vtk_fields());

  std::set<types::boundary_id> boundary_ids;
  for (const auto &cell : domain.triangulation().active_cell_iterators())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary())
        boundary_ids.insert(face->boundary_id());

  EXPECT_EQ(boundary_ids, (std::set<types::boundary_id>{11, 12, 13, 14}));
}
