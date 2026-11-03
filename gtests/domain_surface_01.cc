// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/mpi.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <gtest/gtest.h>
#include <immersx/core/domain.h>

#include <fstream>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

#if defined(DEAL_II_WITH_GMSH_API) || defined(DEAL_II_GMSH_WITH_API)
namespace
{
  std::string
  surface_fixture(const std::string &extension,
                  const bool         open   = false,
                  const bool         curved = false)
  {
    const auto directory = TestPaths::output_directory("domain-surface");
    const auto filename  = directory +
                          (curved ? "/curved." :
                           open   ? "/open." :
                                    "/closed.") +
                          extension;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::filesystem::create_directories(directory);
        Triangulation<3> cube;
        if (curved)
          {
            GridGenerator::hyper_ball(cube);
            cube.refine_global(1);
          }
        else
          GridGenerator::hyper_cube(cube, 0., 1., true);
        Triangulation<2, 3> surface;
        if (curved)
          surface.set_manifold(0, SphericalManifold<2, 3>());
        GridGenerator::extract_boundary_mesh(cube,
                                             surface,
                                             open ?
                                               std::set<types::boundary_id>{0} :
                                               std::set<types::boundary_id>{});
        std::ofstream output(filename);
        if (extension == "msh")
          GridOut().write_msh(surface, output);
        else if (extension == "vtk")
          GridOut().write_vtk(surface, output);
        else if (extension == "obj")
          {
            for (const auto &vertex : surface.get_vertices())
              output << "v " << vertex << '\n';
            for (const auto &cell : surface.active_cell_iterators())
              output << "f " << cell->vertex_index(0) + 1 << ' '
                     << cell->vertex_index(1) + 1 << ' '
                     << cell->vertex_index(3) + 1 << ' '
                     << cell->vertex_index(2) + 1 << '\n';
          }
        else if (extension == "stl")
          {
            Triangulation<2, 3> triangles;
            GridGenerator::convert_hypercube_to_simplex_mesh(surface,
                                                             triangles);
            output << "solid cube\n";
            for (const auto &cell : triangles.active_cell_iterators())
              {
                output << "facet normal 0 0 0\nouter loop\n";
                for (const auto v : cell->vertex_indices())
                  output << "vertex " << cell->vertex(v) << '\n';
                output << "endloop\nendfacet\n";
              }
            output << "endsolid cube\n";
          }
      }
    MPI_Barrier(MPI_COMM_WORLD);
    return filename;
  }

  void
  check_surface(const std::string &extension, const bool hexahedra)
  {
    DomainParameters<3> parameters("/Tests/Surface/" + extension +
                                   (hexahedra ? "/Hex/" : "/Tet/"));
    parameters.name_of_grid = surface_fixture(extension);
    parameters.triangulation_type =
      hexahedra ? "distributed" : "fullydistributed";
    parameters.surface_mesh_size    = .4;
    parameters.grid_scale           = 2.;
    parameters.surface_volume_cells = hexahedra ? "hexahedra" : "tetrahedra";
    Domain<3> domain(parameters);
    ASSERT_NO_THROW(domain.make_grid());
    EXPECT_GT(domain.triangulation().n_global_active_cells(), 0u);
    EXPECT_EQ(domain.triangulation().all_reference_cells_are_hyper_cube(),
              hexahedra);
    EXPECT_NEAR(GridTools::volume(domain.triangulation()), 8., 1.e-9);
  }
} // namespace

TEST(Domain, BOTH_SurfaceMshToTetrahedra)
{
  check_surface("msh", false);
}

#  ifdef DEAL_II_WITH_ASSIMP
TEST(Domain, BOTH_SurfaceObjToTetrahedra)
{
  check_surface("obj", false);
}
#  endif

TEST(Domain, BOTH_SurfaceVtkToTetrahedra)
{
  check_surface("vtk", false);
}

TEST(Domain, BOTH_SurfaceStlToHexahedra)
{
  check_surface("stl", true);
  // The Gmsh subdivision option must not affect later tetrahedral domains.
  check_surface("stl", false);
}

TEST(Domain, BOTH_CurvedSurfaceSubdivisionPreservesVolume)
{
  const auto   input              = surface_fixture("vtk", false, true);
  double       tetrahedral_volume = 0.;
  unsigned int tetrahedra         = 0;
  for (const bool hexahedra : {false, true})
    {
      DomainParameters<3> parameters(hexahedra ? "/Tests/CurvedSurface/Hex/" :
                                                 "/Tests/CurvedSurface/Tet/");
      parameters.name_of_grid          = input;
      parameters.triangulation_type    = "fullydistributed";
      parameters.surface_mesh_size     = .4;
      parameters.surface_feature_angle = 60.;
      parameters.surface_volume_cells  = hexahedra ? "hexahedra" : "tetrahedra";
      Domain<3> domain(parameters);
      ASSERT_NO_THROW(domain.make_grid());
      const double volume = GridTools::volume(domain.triangulation());
      if (!hexahedra)
        {
          tetrahedral_volume = volume;
          tetrahedra         = domain.triangulation().n_global_active_cells();
        }
      else
        {
          EXPECT_TRUE(
            domain.triangulation().all_reference_cells_are_hyper_cube());
          EXPECT_EQ(domain.triangulation().n_global_active_cells(),
                    4 * tetrahedra);
          EXPECT_NEAR(volume, tetrahedral_volume, 1.e-9 * tetrahedral_volume);
        }
    }
}

TEST(Domain, BOTH_VolumeInputBypassesSurfaceMeshing)
{
  const auto directory = TestPaths::output_directory("domain-surface");
  const auto filename  = directory + "/volume.msh";
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::filesystem::create_directories(directory);
      std::ofstream output(filename);
      output << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
             << "$PhysicalNames\n1\n3 7 \"MaterialID:7\"\n$EndPhysicalNames\n"
             << "$Nodes\n8\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
             << "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n$EndNodes\n"
             << "$Elements\n1\n1 5 2 7 1 1 2 3 4 5 6 7 8\n$EndElements\n";
    }
  MPI_Barrier(MPI_COMM_WORLD);
  DomainParameters<3> parameters("/Tests/SurfaceVolumeBypass/");
  parameters.name_of_grid      = filename;
  parameters.surface_mesh_size = 0.;
  Domain<3> domain(parameters);
  ASSERT_NO_THROW(domain.make_grid());
  EXPECT_EQ(domain.triangulation().n_global_active_cells(), 1u);
  for (const auto &cell : domain.triangulation().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        EXPECT_EQ(cell->material_id(), 7u);
      }
}

TEST(Domain, BOTH_EmbeddedStlRemainsSurface)
{
  DomainParameters<2, 3> parameters("/Tests/EmbeddedStl/");
  parameters.name_of_grid       = surface_fixture("stl");
  parameters.triangulation_type = "fullydistributed";
  Domain<2, 3> domain(parameters);
  ASSERT_NO_THROW(domain.make_grid());
  EXPECT_EQ(domain.triangulation().n_global_active_cells(), 48u);
  EXPECT_TRUE(domain.triangulation().all_reference_cells_are_simplex());
}

TEST(Domain, BOTH_RejectsOpenSurface)
{
  DomainParameters<3> parameters("/Tests/OpenSurface/");
  parameters.name_of_grid       = surface_fixture("vtk", true);
  parameters.triangulation_type = "fullydistributed";
  Domain<3> domain(parameters);
  EXPECT_THROW(domain.make_grid(), dealii::ExceptionBase);
}

TEST(Domain, BOTH_RequiresVolumeWhenRequested)
{
  DomainParameters<3> parameters("/Tests/RequireVolume/");
  parameters.name_of_grid       = surface_fixture("msh");
  parameters.mesh_input         = "volume";
  parameters.triangulation_type = "fullydistributed";
  Domain<3> domain(parameters);
  EXPECT_THROW(domain.make_grid(), dealii::ExceptionBase);
}
#endif
