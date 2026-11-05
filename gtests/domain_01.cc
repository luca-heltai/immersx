#include <deal.II/base/scope_exit.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <gtest/gtest.h>
#include <immersx/core/domain.h>

#include <array>
#include <fstream>
#include <set>
#include <sstream>

#if defined(DEAL_II_WITH_GMSH_API) || defined(DEAL_II_GMSH_WITH_API)
#  include <gmsh.h>
#endif

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


TEST(Domain, BOTH_CopiesToSerialTriangulation)
{
  DomainParameters<2> parameters("/Tests/DomainTriangulation/");
  parameters.name_of_grid       = "hyper_cube";
  parameters.arguments_for_grid = "0: 1: false";
  parameters.initial_refinement = 2;
  Domain<2> domain(parameters);
  domain.make_grid();

  EXPECT_EQ(domain.triangulation().n_global_active_cells(), 16u);
  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 2)
    {
      EXPECT_EQ(domain.triangulation().n_locally_owned_active_cells(), 8u);
    }
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

#if defined(DEAL_II_WITH_GMSH_API) || defined(DEAL_II_GMSH_WITH_API)
namespace
{
  std::string
  mesh_bytes(const std::string &filename)
  {
    std::ifstream      input(filename, std::ios::binary);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
  }
} // namespace

TEST(Domain, BOTH_GmshPhysicalTagsAndNamedOverrides)
{
  const bool owns_gmsh = gmsh::isInitialized() == 0;
  if (owns_gmsh)
    gmsh::initialize();
  const ScopeExit finalize([owns_gmsh]() {
    if (owns_gmsh && gmsh::isInitialized())
      gmsh::finalize();
  });
  double old_version = 0., old_save_all = 0., old_binary = 0., old_scaling = 0.,
         old_orphans = 0.;
  gmsh::option::getNumber("Mesh.MshFileVersion", old_version);
  gmsh::option::getNumber("Mesh.SaveAll", old_save_all);
  gmsh::option::getNumber("Mesh.Binary", old_binary);
  gmsh::option::getNumber("Mesh.ScalingFactor", old_scaling);
  gmsh::option::getNumber("Mesh.SaveWithoutOrphans", old_orphans);
  const ScopeExit restore(
    [old_version, old_save_all, old_binary, old_scaling, old_orphans]() {
      gmsh::option::setNumber("Mesh.MshFileVersion", old_version);
      gmsh::option::setNumber("Mesh.SaveAll", old_save_all);
      gmsh::option::setNumber("Mesh.Binary", old_binary);
      gmsh::option::setNumber("Mesh.ScalingFactor", old_scaling);
      gmsh::option::setNumber("Mesh.SaveWithoutOrphans", old_orphans);
    });
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.SaveAll", 0);
  gmsh::option::setNumber("Mesh.Binary", 1);
  gmsh::option::setNumber("Mesh.ScalingFactor", 2.5);
  gmsh::option::setNumber("Mesh.SaveWithoutOrphans", 1);

  for (const std::string backend : {"distributed", "fullydistributed"})
    for (const std::string variant : {"unnamed", "named", "mixed", "untagged"})
      {
        SCOPED_TRACE(backend + "/" + variant);
        ParameterAcceptor::clear();
        DomainParameters<2> parameters("/Gmsh physical tags/");
        parameters.name_of_grid =
          TestPaths::data_filename("tests/gmsh_physical_" + variant + ".msh");
        parameters.triangulation_type = backend;
        parameters.initial_refinement = 0;
        const auto original           = mesh_bytes(parameters.name_of_grid);
        Domain<2>  domain(parameters);
        domain.prepare_grid();
        // The reader transfers selectors; callers own their actual manifolds.
        domain.refinement_triangulation().set_manifold(2, FlatManifold<2>());
        domain.refinement_triangulation().set_manifold(3, FlatManifold<2>());
        domain.finalize_grid();
        EXPECT_EQ(mesh_bytes(parameters.name_of_grid), original);
        unsigned int cells  = 0;
        double       volume = 0., boundary_measure = 0.;
        for (const auto &cell : domain.triangulation().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              ++cells;
              volume += cell->measure();
              const bool left = cell->center()[0] < 1.;
              EXPECT_EQ(cell->material_id(),
                        left ? (variant == "mixed"    ? 7u :
                                variant == "untagged" ? 0u :
                                                        11u) :
                               42u);
              EXPECT_EQ(cell->manifold_id(),
                        left && variant == "mixed" ? 2u :
                                                     numbers::flat_manifold_id);
              for (const auto face : cell->face_indices())
                if (cell->at_boundary(face))
                  {
                    const auto &f = cell->face(face);
                    boundary_measure += f->measure();
                    const bool         left_face = f->center()[0] < 1e-10;
                    const unsigned int expected =
                      left_face ? (variant == "mixed" ? 5u : 101u) :
                      f->center()[0] > 2. - 1e-10 ? 202u :
                      f->center()[1] < .5         ? 77u :
                                                    99u;
                    EXPECT_EQ(f->boundary_id(), expected);
                    EXPECT_EQ(f->manifold_id(),
                              left_face && variant == "mixed" ?
                                3u :
                                numbers::flat_manifold_id);
                  }
            }
        EXPECT_EQ(Utilities::MPI::sum(cells, MPI_COMM_WORLD), 2u);
        EXPECT_NEAR(Utilities::MPI::sum(volume, MPI_COMM_WORLD), 2., 1e-14);
        EXPECT_NEAR(Utilities::MPI::sum(boundary_measure, MPI_COMM_WORLD),
                    6.,
                    1e-14);
        double version = 0., save_all = 0., binary = 0.;
        gmsh::option::getNumber("Mesh.MshFileVersion", version);
        gmsh::option::getNumber("Mesh.SaveAll", save_all);
        gmsh::option::getNumber("Mesh.Binary", binary);
        EXPECT_DOUBLE_EQ(version, 2.2);
        EXPECT_DOUBLE_EQ(save_all, 0.);
        EXPECT_DOUBLE_EQ(binary, 1.);
        double scaling = 0., orphans = 0.;
        gmsh::option::getNumber("Mesh.ScalingFactor", scaling);
        gmsh::option::getNumber("Mesh.SaveWithoutOrphans", orphans);
        EXPECT_DOUBLE_EQ(scaling, 2.5);
        EXPECT_DOUBLE_EQ(orphans, 1.);
      }

  ParameterAcceptor::clear();
  DomainParameters<2> precise_parameters("/Gmsh coordinates/");
  precise_parameters.name_of_grid =
    TestPaths::data_filename("tests/gmsh_physical_precise.msh");
  precise_parameters.initial_refinement = 0;
  Domain<2> precise(precise_parameters);
  precise.make_grid();
  const std::array<double, 3> xs{
    {0.12345678901234566, 1.2345678901234567, 2.3456789012345678}};
  const std::array<double, 2> ys{{0.23456789012345678, 1.3456789012345678}};
  for (const double x : xs)
    for (const double y : ys)
      {
        unsigned int found = 0;
        for (const auto &cell : precise.triangulation().active_cell_iterators())
          if (cell->is_locally_owned())
            for (const auto v : cell->vertex_indices())
              found += cell->vertex(v)[0] == x && cell->vertex(v)[1] == y;
        EXPECT_GT(Utilities::MPI::sum(found, MPI_COMM_WORLD), 0u);
      }
  double area = 0.;
  for (const auto &cell : precise.triangulation().active_cell_iterators())
    if (cell->is_locally_owned())
      area += cell->measure();
  EXPECT_NEAR(Utilities::MPI::sum(area, MPI_COMM_WORLD),
              (xs.back() - xs.front()) * (ys.back() - ys.front()),
              1e-14);

  ParameterAcceptor::clear();
  DomainParameters<2> invalid("/Gmsh failure/");
  invalid.name_of_grid =
    TestPaths::data_filename("tests/missing-gmsh-input.msh");
  invalid.initial_refinement = 0;
  Domain<2> problem(invalid);
  EXPECT_ANY_THROW(problem.make_grid());
  double version = 0., save_all = 0., binary = 0.;
  gmsh::option::getNumber("Mesh.MshFileVersion", version);
  gmsh::option::getNumber("Mesh.SaveAll", save_all);
  gmsh::option::getNumber("Mesh.Binary", binary);
  EXPECT_DOUBLE_EQ(version, 2.2);
  EXPECT_DOUBLE_EQ(save_all, 0.);
  EXPECT_DOUBLE_EQ(binary, 1.);
  double scaling = 0., orphans = 0.;
  gmsh::option::getNumber("Mesh.ScalingFactor", scaling);
  gmsh::option::getNumber("Mesh.SaveWithoutOrphans", orphans);
  EXPECT_DOUBLE_EQ(scaling, 2.5);
  EXPECT_DOUBLE_EQ(orphans, 1.);

  ParameterAcceptor::clear();
  DomainParameters<3> recovery("/Gmsh recovery/");
  recovery.name_of_grid = TestPaths::data_filename("tests/simplex_cube.msh");
  recovery.triangulation_type = "fullydistributed";
  recovery.initial_refinement = 0;
  recovery.mesh_input         = "volume";
  Domain<3> recovered(recovery);
  ASSERT_NO_THROW(recovered.make_grid());
  EXPECT_GT(recovered.triangulation().n_global_active_cells(), 0u);
}
#endif
