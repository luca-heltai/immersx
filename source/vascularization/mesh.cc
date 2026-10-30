// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/tria_description.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/discretization.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <tuple>

namespace ImmersX::Vascularization
{
  namespace
  {
    using namespace dealii;

    struct VertexKey
    {
      Id     entity    = invalid_id;
      double parameter = 0.;
      bool   interior  = false;

      bool
      operator<(const VertexKey &other) const
      {
        return std::tie(interior, entity, parameter) <
               std::tie(other.interior, other.entity, other.parameter);
      }
      bool
      operator==(const VertexKey &other) const
      {
        return interior == other.interior && entity == other.entity &&
               parameter == other.parameter;
      }
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &entity &parameter &interior;
      }
    };

    struct MeshCell
    {
      types::coarse_cell_id    id;
      types::subdomain_id      owner;
      VesselCell               reference;
      std::array<VertexKey, 2> vertices;
      std::array<Point<3>, 2>  positions;

      bool
      operator==(const MeshCell &other) const
      {
        return id == other.id && owner == other.owner &&
               reference.vessel == other.reference.vessel &&
               reference.begin == other.reference.begin &&
               reference.end == other.reference.end &&
               vertices == other.vertices && positions == other.positions;
      }
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &id &owner;
        ar &reference.vessel &reference.begin &reference.end;
        ar &vertices                          &positions;
      }
    };

    unsigned int
    vertex_directory(const VertexKey &key, unsigned int size)
    {
      // Physical namespaces stay separate even when their integer ids coincide.
      // Mix the parameter bits to distribute a long subdivided vessel as well.
      std::uint64_t bits = 0;
      static_assert(sizeof(bits) == sizeof(key.parameter));
      std::memcpy(&bits, &key.parameter, sizeof(bits));
      std::uint64_t hash =
        key.entity ^ bits ^ static_cast<std::uint64_t>(key.interior);
      hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ULL;
      hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebULL;
      return (hash ^ (hash >> 31)) % size;
    }

    CellVessels
    create_distributed(
      const NetworkData                               &network,
      const DiscretizationResult                      &discretization,
      parallel::fullydistributed::Triangulation<1, 3> &triangulation)
    {
      const auto comm        = triangulation.get_mpi_communicator();
      const auto rank        = Utilities::MPI::this_mpi_process(comm);
      const auto size        = Utilities::MPI::n_mpi_processes(comm);
      const auto require_all = [&](bool valid, const std::string &message) {
        AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid),
                                        comm) == 1,
                    ExcMessage(message));
      };
      bool valid = triangulation.n_active_cells() == 0 &&
                   discretization.validation.valid() &&
                   validate_data(network, size).valid();
      for (const auto &cell : discretization.cells)
        {
          const auto vessel = network.vessels.find(cell.reference.vessel);
          valid             = vessel != network.vessels.end() &&
                  vessel->second.owner == rank && cell.owner < size &&
                  std::isfinite(cell.reference.begin) &&
                  std::isfinite(cell.reference.end) &&
                  cell.reference.begin >= 0. && cell.reference.end <= 1. &&
                  cell.reference.begin < cell.reference.end && valid;
        }
      require_all(
        valid,
        "A distributed vascular mesh requires an empty destination, valid owned intervals and endpoint data.");
      const std::uint64_t count = discretization.cells.size();
      const auto          total = Utilities::MPI::sum(count, comm);
      require_all(
        total < numbers::invalid_coarse_cell_id,
        "Too many vascular coarse cells for the native cell id type.");
      // Native global-cell partitioners require a positive index-space size.
      // As for the serial builder, an empty input leaves the destination empty.
      if (total == 0)
        return {};
      std::uint64_t offset = 0;
      const int     error =
        MPI_Exscan(&count, &offset, 1, MPI_UINT64_T, MPI_SUM, comm);
      AssertThrowMPI(error);
      if (rank == 0)
        offset = 0;
      using Incidence = std::pair<VertexKey, MeshCell>;
      std::map<unsigned int, std::vector<Incidence>> send;
      for (std::size_t i = 0; i < discretization.cells.size(); ++i)
        {
          const auto &reference = discretization.cells[i].reference;
          const auto &vessel    = network.vessels.at(reference.vessel);
          MeshCell    cell;
          cell.id        = offset + i;
          cell.owner     = discretization.cells[i].owner;
          cell.reference = reference;
          const auto &a  = network.nodes.at(vessel.nodes[0]).position;
          const auto &b  = network.nodes.at(vessel.nodes[1]).position;
          for (unsigned int v = 0; v < 2; ++v)
            {
              const double t = v == 0 ? reference.begin : reference.end;
              cell.vertices[v] =
                t == 0. ? VertexKey{vessel.nodes[0], 0., false} :
                t == 1. ? VertexKey{vessel.nodes[1], 0., false} :
                          VertexKey{reference.vessel, t, true};
              // Preserve exact endpoint coordinates at shared junctions even
              // when incident vessels have opposite reference orientations.
              cell.positions[v] = t == 0. ? a :
                                  t == 1. ? b :
                                            (1. - t) * a + t * b;
            }
          const double length = cell.positions[0].distance(cell.positions[1]);
          valid               = std::isfinite(length) && length > 0. && valid;
          for (const auto &key : cell.vertices)
            send[vertex_directory(key, size)].push_back({key, cell});
        }
      require_all(
        valid,
        "A vascular numerical interval collapses in physical coordinates.");
      const auto incoming = Utilities::MPI::some_to_some(comm, send);
      std::map<VertexKey, std::map<types::coarse_cell_id, MeshCell>> incidence;
      for (const auto &[source, entries] : incoming)
        for (const auto &[key, cell] : entries)
          {
            auto &cells               = incidence[key];
            const auto [it, inserted] = cells.emplace(cell.id, cell);
            valid = (inserted || it->second == cell) && valid;
          }
      std::map<unsigned int, std::vector<MeshCell>> replies;
      for (const auto &[key, cells] : incidence)
        {
          std::set<unsigned int>  recipients;
          std::optional<Point<3>> position;
          for (const auto &[id, cell] : cells)
            {
              recipients.insert(cell.owner);
              for (unsigned int v = 0; v < 2; ++v)
                if (cell.vertices[v] == key)
                  {
                    valid =
                      (!position || *position == cell.positions[v]) && valid;
                    position = cell.positions[v];
                  }
            }
          for (const auto destination : recipients)
            for (const auto &[id, cell] : cells)
              replies[destination].push_back(cell);
        }
      require_all(
        valid,
        "A shared vascular vertex has inconsistent endpoint coordinates or cell data.");
      const auto received = Utilities::MPI::some_to_some(comm, replies);
      std::map<types::coarse_cell_id, MeshCell> relevant;
      for (const auto &[source, cells] : received)
        for (const auto &cell : cells)
          {
            const auto [it, inserted] = relevant.emplace(cell.id, cell);
            valid = (inserted || it->second == cell) && valid;
          }
      require_all(valid && relevant.size() <
                             std::numeric_limits<unsigned int>::max() / 2,
                  "Inconsistent or too many locally relevant vascular cells.");
      TriangulationDescription::Description<1, 3> description;
      description.comm = comm;
      description.cell_infos.resize(1);
      std::map<VertexKey, unsigned int> vertices;
      CellVessels                       references;
      for (const auto &[id, cell] : relevant)
        {
          CellData<1> data;
          for (unsigned int v = 0; v < 2; ++v)
            {
              const auto [it, inserted] =
                vertices.emplace(cell.vertices[v], vertices.size());
              if (inserted)
                description.coarse_cell_vertices.push_back(cell.positions[v]);
              data.vertices[v] = it->second;
            }
          description.coarse_cells.push_back(data);
          description.coarse_cell_index_to_coarse_cell_id.push_back(id);
          TriangulationDescription::CellData<1> info;
          info.id                 = CellId(id, {}).to_binary<1>();
          info.subdomain_id       = cell.owner;
          info.level_subdomain_id = cell.owner;
          description.cell_infos[0].push_back(info);
          references.emplace(id, cell.reference);
        }
      triangulation.create_triangulation(description);
      return references;
    }
  } // namespace

  CellVessels
  create_triangulation(const NetworkData           &network,
                       const DiscretizationResult  &discretization,
                       dealii::Triangulation<1, 3> &triangulation)
  {
    using namespace dealii;
    if (auto *distributed =
          dynamic_cast<parallel::fullydistributed::Triangulation<1, 3> *>(
            &triangulation))
      return create_distributed(network, discretization, *distributed);
    AssertThrow(triangulation.n_active_cells() == 0,
                ExcMessage("Vascular mesh destination must be empty."));
    AssertThrow(discretization.validation.valid(),
                ExcMessage("Cannot adopt an invalid vascular discretization."));
    std::vector<Point<3>>                         vertices;
    std::vector<CellData<1>>                      cells;
    std::map<Id, unsigned int>                    endpoints;
    std::map<std::pair<Id, double>, unsigned int> interior;
    const auto vertex = [&](Id vessel, double t) {
      const auto &e        = network.vessels.at(vessel);
      const bool  endpoint = t == 0. || t == 1.;
      const Id    node     = t == 0. ? e.nodes[0] : e.nodes[1];
      auto       &index    = endpoint ? endpoints[node] : interior[{vessel, t}];
      if (index == 0)
        {
          AssertThrow(vertices.size() <
                        std::numeric_limits<unsigned int>::max(),
                      ExcMessage("Too many vascular mesh vertices."));
          const auto &a = network.nodes.at(e.nodes[0]).position;
          const auto &b = network.nodes.at(e.nodes[1]).position;
          vertices.push_back((1. - t) * a + t * b);
          index = vertices.size();
        }
      return index - 1;
    };
    CellVessels references;
    for (const auto &cell : discretization.cells)
      {
        CellData<1> data;
        data.vertices[0] = vertex(cell.reference.vessel, cell.reference.begin);
        data.vertices[1] = vertex(cell.reference.vessel, cell.reference.end);
        references[cells.size()] = cell.reference;
        cells.push_back(data);
      }
    if (!cells.empty())
      triangulation.create_triangulation(vertices, cells, SubCellData());
    return references;
  }

  VesselCell
  physical_reference(const dealii::CellId &cell,
                     const CellVessels    &coarse_cells)
  {
    auto result = coarse_cells.at(cell.get_coarse_cell_id());
    for (const unsigned int child : cell.get_child_indices())
      {
        const double middle = .5 * (result.begin + result.end);
        if (child == 0)
          result.end = middle;
        else if (child == 1)
          result.begin = middle;
        else
          AssertThrow(
            false,
            dealii::ExcMessage(
              "A vascular line cell can only have children 0 and 1."));
      }
    return result;
  }
} // namespace ImmersX::Vascularization
