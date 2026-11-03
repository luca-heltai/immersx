// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "surface_volume_mesh.h"

#include <deal.II/base/numbers.h>
#include <deal.II/base/scope_exit.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_topology.h>

#include <array>
#include <cmath>
#include <map>
#include <queue>
#include <set>

#if defined(DEAL_II_WITH_GMSH_API) || defined(DEAL_II_GMSH_WITH_API)
#  include <gmsh.h>
#endif

namespace ImmersX::detail
{
  void
  generate_volume_from_surface(const dealii::Triangulation<2, 3> &input,
                               dealii::Triangulation<3>          &volume,
                               const DomainParameters<3, 3>      &parameters)
  {
#if defined(DEAL_II_WITH_GMSH_API) || defined(DEAL_II_GMSH_WITH_API)
    using namespace dealii;
    AssertThrow(std::isfinite(parameters.surface_mesh_size) &&
                  parameters.surface_mesh_size > 0.,
                ExcMessage("Surface target edge length must be positive."));
    AssertThrow(parameters.surface_feature_angle > 0. &&
                  parameters.surface_feature_angle <= 180.,
                ExcMessage("Surface feature angle must be in (0,180]."));
    AssertThrow(input.n_active_cells() > 0,
                ExcMessage("The input surface mesh is empty."));

    Triangulation<2, 3> surface;
    if (input.all_reference_cells_are_hyper_cube())
      GridGenerator::convert_hypercube_to_simplex_mesh(input, surface);
    else if (input.all_reference_cells_are_simplex())
      surface.copy_triangulation(input);
    AssertThrow(
      surface.n_active_cells() > 0,
      ExcMessage("Surface cells must be all triangles or all quadrilaterals."));

    // One connected closed shell is supported. Reject open/nonmanifold input
    // before Gmsh attempts to construct a volume with an ambiguous interior.
    std::map<unsigned int, std::vector<unsigned int>> edge_cells;
    for (const auto &cell : surface.active_cell_iterators())
      for (const auto line : cell->line_indices())
        edge_cells[cell->line(line)->index()].push_back(
          cell->active_cell_index());
    std::vector<std::vector<unsigned int>> neighbors(surface.n_active_cells());
    for (const auto &entry : edge_cells)
      {
        const auto &cells = entry.second;
        AssertThrow(
          cells.size() == 2,
          ExcMessage(
            "Surface must be closed and manifold: every edge must have two incident cells."));
        neighbors[cells[0]].push_back(cells[1]);
        neighbors[cells[1]].push_back(cells[0]);
      }
    std::set<unsigned int>   visited{0};
    std::queue<unsigned int> pending;
    pending.push(0);
    while (!pending.empty())
      {
        const auto cell = pending.front();
        pending.pop();
        for (const auto neighbor : neighbors[cell])
          if (visited.insert(neighbor).second)
            pending.push(neighbor);
      }
    AssertThrow(
      visited.size() == surface.n_active_cells(),
      ExcMessage(
        "Surface must consist of one connected shell; cavities and separate organs require separate domains."));

    // Older GridIn readers finalize their own Gmsh session. Keep an existing
    // session (including MPI_InitFinalize's) alive, and release only our own.
    const bool owns_gmsh = gmsh::isInitialized() == 0;
    if (owns_gmsh)
      gmsh::initialize();
    const ScopeExit finalize_gmsh([owns_gmsh]() {
      if (owns_gmsh)
        gmsh::finalize();
    });
    double          subdivision = 0., second_order_linear = 0.;
    gmsh::option::getNumber("Mesh.SubdivisionAlgorithm", subdivision);
    gmsh::option::getNumber("Mesh.SecondOrderLinear", second_order_linear);
    const ScopeExit restore_options([subdivision, second_order_linear]() {
      gmsh::option::setNumber("Mesh.SubdivisionAlgorithm", subdivision);
      gmsh::option::setNumber("Mesh.SecondOrderLinear", second_order_linear);
    });
    gmsh::clear();
    gmsh::model::add("ImmersX surface volume");
    const int                tag = gmsh::model::addDiscreteEntity(2);
    std::vector<std::size_t> node_tags;
    std::vector<double>      coordinates;
    for (unsigned int v = 0; v < surface.n_vertices(); ++v)
      if (surface.get_used_vertices()[v])
        {
          node_tags.push_back(v + 1);
          for (unsigned int d = 0; d < 3; ++d)
            coordinates.push_back(surface.get_vertices()[v][d]);
        }
    gmsh::model::mesh::addNodes(2, tag, node_tags, coordinates);
    std::vector<std::size_t> triangles;
    for (const auto &cell : surface.active_cell_iterators())
      for (const auto v : cell->vertex_indices())
        triangles.push_back(cell->vertex_index(v) + 1);
    gmsh::model::mesh::addElementsByType(tag, 2, {}, triangles);
    try
      {
        gmsh::model::mesh::classifySurfaces(
          parameters.surface_feature_angle * numbers::PI / 180., true, true);
        gmsh::model::mesh::createGeometry();
        gmsh::vectorpair patches;
        gmsh::model::getEntities(patches, 2);
        std::vector<int> tags;
        for (const auto &patch : patches)
          tags.push_back(patch.second);
        const int shell = gmsh::model::geo::addSurfaceLoop(tags);
        gmsh::model::geo::addVolume({shell});
        gmsh::model::geo::synchronize();
        gmsh::option::setNumber("Mesh.MeshSizeMin",
                                parameters.surface_mesh_size);
        gmsh::option::setNumber("Mesh.MeshSizeMax",
                                parameters.surface_mesh_size);
        gmsh::option::setNumber("Mesh.ElementOrder", 1);
        gmsh::option::setNumber("Mesh.RecombineAll", 0);
        gmsh::option::setNumber("Mesh.SubdivisionAlgorithm", 0);
        gmsh::model::mesh::generate(3);
        const bool hexahedra = parameters.surface_volume_cells == "hexahedra";
        AssertThrow(hexahedra ||
                      parameters.surface_volume_cells == "tetrahedra",
                    ExcMessage("Unknown surface volume cell type."));
        if (hexahedra)
          {
            // Gmsh's conforming subdivision is also available with deal.II 9.7,
            // which has no simplex-to-hypercube conversion helper.
            gmsh::option::setNumber("Mesh.SubdivisionAlgorithm", 2);
            // Projecting new boundary nodes onto the parametrized geometry
            // can distort cells and change the original tetrahedral volume.
            gmsh::option::setNumber("Mesh.SecondOrderLinear", 1);
            gmsh::model::mesh::refine();
            gmsh::option::setNumber("Mesh.SubdivisionAlgorithm", 0);
          }

        std::vector<double> parametric;
        gmsh::model::mesh::getNodes(node_tags, coordinates, parametric);
        std::map<std::size_t, unsigned int> vertex_index;
        std::vector<Point<3>>               vertices(node_tags.size());
        for (unsigned int v = 0; v < node_tags.size(); ++v)
          {
            vertex_index.emplace(node_tags[v], v);
            for (unsigned int d = 0; d < 3; ++d)
              vertices[v][d] = coordinates[3 * v + d];
          }
        std::vector<std::size_t> element_tags, cell_nodes;
        gmsh::model::mesh::getElementsByType(hexahedra ? 5 : 4,
                                             element_tags,
                                             cell_nodes);
        AssertThrow(!element_tags.empty(),
                    ExcMessage("Gmsh generated no volume cells."));
        const unsigned int                vertices_per_cell = hexahedra ? 8 : 4;
        const std::array<unsigned int, 8> gmsh_hex_vertices{
          {0, 1, 3, 2, 4, 5, 7, 6}};
        std::vector<CellData<3>> cells(element_tags.size(),
                                       CellData<3>(vertices_per_cell));
        for (unsigned int c = 0; c < cells.size(); ++c)
          for (unsigned int v = 0; v < vertices_per_cell; ++v)
            cells[c].vertices[v] = vertex_index.at(
              cell_nodes[vertices_per_cell * c +
                         (hexahedra ? gmsh_hex_vertices[v] : v)]);
        if (hexahedra)
          {
            GridTools::invert_all_negative_measure_cells(vertices, cells);
            GridTools::consistently_order_cells(cells);
          }
        volume.create_triangulation(vertices, cells, SubCellData());
      }
    catch (const std::exception &error)
      {
        AssertThrow(false,
                    ExcMessage(std::string("Gmsh surface meshing failed: ") +
                               error.what()));
      }
    catch (const std::string &error)
      {
        AssertThrow(false, ExcMessage("Gmsh surface meshing failed: " + error));
      }
#else
    (void)input;
    (void)volume;
    (void)parameters;
    AssertThrow(
      false,
      dealii::ExcMessage(
        "Surface volume meshing requires deal.II with the Gmsh API."));
#endif
  }
} // namespace ImmersX::detail
