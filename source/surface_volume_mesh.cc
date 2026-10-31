// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "surface_volume_mesh.h"

#include <deal.II/base/numbers.h>

#include <deal.II/grid/grid_generator.h>

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

    // deal.II initializes Gmsh through MPI_InitFinalize. Like GridIn's API
    // reader, this operation uses Gmsh's process-global current model.
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
        gmsh::model::mesh::generate(3);

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
        std::vector<std::size_t> element_tags, tetrahedra;
        gmsh::model::mesh::getElementsByType(4, element_tags, tetrahedra);
        AssertThrow(!element_tags.empty(),
                    ExcMessage("Gmsh generated no tetrahedra."));
        std::vector<CellData<3>> cells(element_tags.size(), CellData<3>(4));
        for (unsigned int c = 0; c < cells.size(); ++c)
          for (unsigned int v = 0; v < 4; ++v)
            cells[c].vertices[v] = vertex_index.at(tetrahedra[4 * c + v]);
        Triangulation<3> tetrahedral;
        tetrahedral.create_triangulation(vertices, cells, SubCellData());
        if (parameters.surface_volume_cells == "tetrahedra")
          volume.copy_triangulation(tetrahedral);
        else if (parameters.surface_volume_cells == "hexahedra")
          GridGenerator::convert_simplex_to_hypercube_mesh(tetrahedral, volume);
        else
          AssertThrow(false, ExcMessage("Unknown surface volume cell type."));
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
