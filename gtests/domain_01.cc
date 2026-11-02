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

  const auto all_cells     = domain.support();
  const auto material_zero = domain.support(0);
  const auto material_one  = domain.support(1);
  const auto all_boundary  = domain.boundary_support();
  const auto boundary_zero = domain.boundary_support(0);
  EXPECT_EQ(&all_cells.triangulation(), &domain.triangulation());
  EXPECT_TRUE(all_cells.includes_cell(domain.triangulation().begin_active()));
  EXPECT_EQ(material_zero.material_id(), 0u);
  EXPECT_TRUE(
    material_zero.includes_cell(domain.triangulation().begin_active()));
  EXPECT_FALSE(
    material_one.includes_cell(domain.triangulation().begin_active()));
  EXPECT_EQ(&all_boundary.triangulation(), &domain.triangulation());
  EXPECT_EQ(all_boundary.boundary_ids(), std::set<types::boundary_id>{0});
  EXPECT_EQ(boundary_zero.boundary_ids(), std::set<types::boundary_id>{0});
}


TEST(Domain, BOTH_ObservesExistingTriangulation)
{
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_SELF);
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(1);
  DomainParameters<2> parameters("/Tests/DomainObserver/");
  parameters.triangulation_type = "fullydistributed";
  parameters.initial_refinement = 3;
  {
    Domain<2> domain(parameters, triangulation);
    EXPECT_EQ(&domain.triangulation(), &triangulation);
    EXPECT_FALSE(domain.uses_fully_distributed_triangulation());
    EXPECT_EQ(domain.communicator(), triangulation.get_mpi_communicator());
    EXPECT_EQ(domain.triangulation().n_global_active_cells(), 4u);
    EXPECT_EQ(&domain.support().triangulation(), &triangulation);
    EXPECT_THROW(domain.make_grid(), dealii::ExceptionBase);
  }
  triangulation.refine_global(1);
  EXPECT_EQ(triangulation.n_global_active_cells(), 16u);
}


TEST(Domain, BOTH_InitializesExternalFullyDistributedTriangulation)
{
  parallel::fullydistributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  DomainParameters<2> parameters("/Tests/DomainExternalFullyDistributed/");
  parameters.arguments_for_grid = "0: 1: false";
  parameters.initial_refinement = 1;
  {
    Domain<2> domain(parameters, triangulation);
    EXPECT_TRUE(domain.uses_fully_distributed_triangulation());
    ASSERT_NO_THROW(domain.make_grid());
    EXPECT_EQ(&domain.triangulation(), &triangulation);
    EXPECT_EQ(triangulation.n_global_active_cells(), 4u);
  }
  EXPECT_EQ(triangulation.n_global_active_cells(), 4u);
  Domain<2> observer(parameters, triangulation);
  EXPECT_TRUE(observer.uses_fully_distributed_triangulation());
  EXPECT_EQ(&observer.triangulation(), &triangulation);
}


TEST(Domain, BOTH_PreparesExistingCoarseMeshWithIds)
{
  dealii::Triangulation<1, 3> coarse_mesh;
  GridGenerator::hyper_cube(coarse_mesh, 0., 1.);
  coarse_mesh.begin_active()->set_material_id(42);
  coarse_mesh.begin_active()->face(0)->set_boundary_id(3);
  coarse_mesh.begin_active()->face(1)->set_boundary_id(4);
  parallel::fullydistributed::Triangulation<1, 3> triangulation(MPI_COMM_WORLD);
  DomainParameters<1, 3> parameters("/Tests/DomainPreparedCoarseMesh/");
  parameters.name_of_grid       = "unused.vtk";
  parameters.initial_refinement = 2;
  Domain<1, 3> domain(parameters, triangulation);
  domain.prepare_grid(coarse_mesh);
  EXPECT_EQ(domain.refinement_triangulation().n_active_cells(), 4u);
  domain.finalize_grid();
  EXPECT_EQ(domain.triangulation().n_global_active_cells(), 4u);
  unsigned int left = 0, right = 0;
  for (const auto &cell : domain.triangulation().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        EXPECT_EQ(cell->material_id(), 42u);
        for (const auto face : cell->face_indices())
          if (cell->face(face)->at_boundary())
            {
              left += cell->face(face)->boundary_id() == 3;
              right += cell->face(face)->boundary_id() == 4;
            }
      }
  EXPECT_EQ(Utilities::MPI::sum(left, MPI_COMM_WORLD), 1u);
  EXPECT_EQ(Utilities::MPI::sum(right, MPI_COMM_WORLD), 1u);
}


TEST(Domain, CopiesToSerialTriangulation)
{
  DomainParameters<2> parameters("/Tests/DomainTriangulation/");
  parameters.name_of_grid       = "hyper_cube";
  parameters.arguments_for_grid = "0: 1: false";
  parameters.initial_refinement = 2;

  Domain<2> domain(parameters);
  domain.make_grid();

  dealii::Triangulation<2> triangulation;
  ASSERT_NO_THROW(triangulation.copy_triangulation(domain.triangulation()));
  EXPECT_EQ(triangulation.n_active_cells(), 16u);
}


TEST(Domain, MPI_CopiesToSerialTriangulation)
{
  DomainParameters<2> parameters("/Tests/DomainMPITriangulation/");
  parameters.name_of_grid       = "hyper_cube";
  parameters.arguments_for_grid = "0: 1: false";
  parameters.initial_refinement = 2;

  Domain<2> domain(parameters);
  domain.make_grid();

  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  EXPECT_EQ(domain.triangulation().n_global_active_cells(), 16u);
  EXPECT_EQ(domain.triangulation().n_locally_owned_active_cells(), 8u);

  dealii::Triangulation<2> triangulation;
  ASSERT_NO_THROW(triangulation.copy_triangulation(domain.triangulation()));
  EXPECT_EQ(triangulation.n_active_cells(), 16u);
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


TEST(Domain, TreatsGridNameWithExtensionAsFile)
{
  DomainParameters<2> parameters("/Tests/DomainVtkMissing/");
  parameters.name_of_grid       = "poisson_primitives_vtk_2d.vtk";
  parameters.arguments_for_grid = "-1: 1: false";

  Domain<2> domain(parameters);
  try
    {
      domain.make_grid();
      FAIL() << "Expected the missing mesh file to fail to open";
    }
  catch (const dealii::ExceptionBase &exception)
    {
      EXPECT_EQ(std::string(exception.what()).find("not implemented"),
                std::string::npos);
    }
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


TEST(Domain, BOTH_TransfersExternalFieldsWithoutRereading)
{
  const auto filename = TestPaths::output_path("domain/external-transfer.vtk");
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::filesystem::create_directories(filename.parent_path());
      std::ofstream output(filename);
      output << "# vtk DataFile Version 3.0\n"
             << "External Domain transfer test\n"
             << "ASCII\n"
             << "DATASET UNSTRUCTURED_GRID\n"
             << "POINTS 4 double\n"
             << "0 0 0  1 0 0  1 1 0  0 1 0\n"
             << "CELLS 1 5\n"
             << "4 0 1 2 3\n"
             << "CELL_TYPES 1\n"
             << "9\n"
             << "CELL_DATA 1\n"
             << "SCALARS a0 double 1\n"
             << "LOOKUP_TABLE default\n"
             << "3.5\n";
    }
  MPI_Barrier(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  DomainParameters<2> parameters("/Tests/DomainExternalFields/");
  parameters.name_of_grid    = filename.string();
  parameters.read_vtk_fields = true;
  {
    Domain<2> domain(parameters, triangulation);
    ASSERT_TRUE(domain.has_vtk_fields());
    const auto a0       = domain.field("a0");
    const auto a0_field = a0.field();
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::filesystem::remove(filename);
    MPI_Barrier(MPI_COMM_WORLD);
    triangulation.refine_global(1);
    EXPECT_EQ(a0.field().dof_handler().n_dofs(), 4u);
    for (const auto index : a0_field.locally_owned_dofs())
      EXPECT_DOUBLE_EQ(a0.coefficients()[index], 3.5);
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_coarsen_flag();
    triangulation.execute_coarsening_and_refinement();
    EXPECT_EQ(triangulation.n_global_active_cells(), 1u);
    EXPECT_EQ(a0.field().dof_handler().n_dofs(), 1u);
    for (const auto index : a0_field.locally_owned_dofs())
      EXPECT_DOUBLE_EQ(a0.coefficients()[index], 3.5);
  }
  triangulation.refine_global(1);
  EXPECT_EQ(triangulation.n_global_active_cells(), 4u);
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
  const auto imported_field = field.field();
  for (const auto index : imported_field.locally_owned_dofs())
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
  const auto imported_field = field.field();
  for (const auto index : imported_field.locally_owned_dofs())
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
