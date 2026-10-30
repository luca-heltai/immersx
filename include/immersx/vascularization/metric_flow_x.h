// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_metric_flow_x_h
#define immersx_vascularization_metric_flow_x_h

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <boost/serialization/array.hpp>
#  include <boost/serialization/set.hpp>
#  include <boost/serialization/utility.hpp>

#  include <immersx/vascularization/generation.h>
#  include <metric_flow_x/blood_flow_system.h>

#  include <algorithm>
#  include <array>
#  include <cmath>
#  include <exception>
#  include <map>
#  include <set>
#  include <string>
#  include <type_traits>
#  include <utility>
#  include <vector>

namespace ImmersX::Vascularization
{
  namespace internal
  {
    template <class Flow, class = void>
    struct HasNetworkPhysics : std::false_type
    {};

    template <class Flow>
    struct HasNetworkPhysics<Flow, std::void_t<typename Flow::NetworkPhysics>>
      : std::true_type
    {};
  } // namespace internal

  /** Older installed MetricFlowX packages retain their ordinary input path. */
  template <class Flow>
  inline constexpr bool has_metric_flow_x_network_input =
    internal::HasNetworkPhysics<Flow>::value;

  /** Physical identifiers remain separate from native execution ids.
   * Coarse-cell intervals use flow coordinates: reversed_vessels contains
   * veins, whose flow coordinate is one minus the physical growth coordinate.
   * No physical geometry or interior incidence is replicated by this value.
   */
  struct MetricFlowXTransfer
  {
    CellVessels                              flow_cells;
    std::map<Id, dealii::types::material_id> vessel_ids;
    /** Keys are physical Node ids, including tissue endpoints and real ports.
     */
    std::map<Id, dealii::types::boundary_id> boundary_ids;
    /** Keys are physical Terminal ids, distinct from Node ids. */
    std::map<Id, dealii::types::boundary_id> terminal_ids;
    std::set<Id>                             reversed_vessels;

    /** Original physical interval after native binary mesh refinements.
     * For a reversed vessel, native local coordinate zero corresponds to
     * this interval's end. The physical network itself is unchanged.
     */
    VesselCell
    reference(const dealii::CellId &cell) const
    {
      auto result = physical_reference(cell, flow_cells);
      if (reversed_vessels.count(result.vessel))
        {
          const double begin = 1. - result.end;
          result.end         = 1. - result.begin;
          result.begin       = begin;
        }
      return result;
    }
  };

  /** Prepare an empty native flow Problem from a validated generation result.
   *
   * The vessel callback (physical Id, const Vessel &) returns the native
   * VesselProperties with explicit material/wall data. Geometry, reference
   * areas, length and viscosity are filled from the immutable physical vessel.
   * The boundary callback (physical Node Id, const Node &, const Terminal *)
   * returns native RCRPhysics. Its Terminal pointer is null at a real outlet;
   * arterial inlets keep native boundary zero and do not call this callback.
   * Both callbacks execute only at physical owners and may inspect local data.
   *
   * The native solver replicates coefficient and terminal tables. This setup
   * exchanges those records and identifiers, never the complete vessel graph.
   * A local mesh-only view reverses veins to orient execution cells along
   * flow. Original nodes, reference orientations, radii and ids remain intact.
   * Native microvascular links reuse the supplied positive terminal beds.
   */
  template <class Flow, class VesselPhysics, class BoundaryPhysics>
  MetricFlowXTransfer
  prepare_metric_flow_x(Flow                   &flow,
                        const GenerationResult &generation,
                        const VesselPhysics    &vessel_physics,
                        const BoundaryPhysics  &boundary_physics)
  {
    using namespace dealii;
    if constexpr (!has_metric_flow_x_network_input<Flow>)
      {
        AssertThrow(
          false,
          ExcMessage(
            "Generated networks require MetricFlowX with in-memory NetworkPhysics input."));
        return {};
      }
    else
      {
        const auto comm        = flow.mpi_communicator();
        const auto rank        = Utilities::MPI::this_mpi_process(comm);
        const auto require_all = [&](bool valid, const char *message) {
          AssertThrow(Utilities::MPI::min(valid ? 1u : 0u, comm) == 1,
                      ExcMessage(message));
        };
        require_all(
          generation.validation.valid() &&
            generation.discretization.validation.valid() &&
            flow.triangulation().n_global_active_cells() == 0,
          "Flow preparation requires a valid result and an empty native mesh.");
        using Physics          = typename Flow::NetworkPhysics;
        using VesselProperties = typename Flow::VesselProperties;
        using RCR = typename decltype(Physics{}.terminals)::mapped_type;
        std::map<Id, VesselProperties> owned_vessels;
        std::map<Id, RCR>              owned_boundaries;
        std::map<Id, Id>               owned_terminal_nodes;
        std::set<Id>                   owned_inlets, owned_reversed;
        bool                           valid = true;
        std::string                    diagnostic;
        const auto                    &network = generation.network;
        try
          {
            for (const auto &[id, vessel] : network.vessels)
              if (vessel.owner == rank)
                {
                  auto properties = vessel_physics(id, vessel);
                  auto radii      = vessel.radii;
                  if (vessel.system == System::venous)
                    {
                      std::swap(radii[0], radii[1]);
                      owned_reversed.insert(id);
                    }
                  AssertThrow(
                    numbers::PI * std::pow(std::min(radii[0], radii[1]), 2) >
                      1.e-10,
                    ExcMessage(
                      "A physical reference area reaches MetricFlowX's numerical floor."));
                  properties.r_in = properties.r_d = radii[0];
                  properties.r_out                 = radii[1];
                  properties.a0                    = properties.a_d =
                    numbers::PI * radii[0] * radii[0];
                  properties.L                 = length(network, id);
                  properties.dynamic_viscosity = vessel.viscosity;
                  owned_vessels.emplace(id, std::move(properties));
                }
            for (const auto &[id, terminal] : network.terminals)
              if (terminal.owner == rank)
                {
                  const auto &node = network.nodes.at(terminal.node);
                  valid =
                    owned_terminal_nodes.emplace(id, terminal.node).second &&
                    valid;
                  valid =
                    owned_boundaries
                      .emplace(terminal.node,
                               boundary_physics(terminal.node, node, &terminal))
                      .second &&
                    valid;
                }
            for (const auto &[id, node] : network.nodes)
              if (node.owner == rank)
                {
                  if (node.port == Port::arterial_inlet)
                    owned_inlets.insert(id);
                  else if (node.port == Port::venous_outlet ||
                           node.port == Port::arterial_outlet)
                    valid = owned_boundaries
                              .emplace(id, boundary_physics(id, node, nullptr))
                              .second &&
                            valid;
                }
          }
        catch (const std::exception &error)
          {
            valid      = false;
            diagnostic = error.what();
          }
        if (Utilities::MPI::min(valid ? 1u : 0u, comm) == 0)
          {
            const auto diagnostics =
              Utilities::MPI::all_gather(comm, diagnostic);
            std::string message =
              "A physical owner could not prepare native flow coefficients.";
            for (const auto &entry : diagnostics)
              if (!entry.empty())
                {
                  message += " " + entry;
                  break;
                }
            AssertThrow(false, ExcMessage(message));
          }
        const auto collect = [&](const auto &owned) {
          std::decay_t<decltype(owned)> result;
          for (const auto &entries : Utilities::MPI::all_gather(comm, owned))
            for (const auto &entry : entries)
              AssertThrow(
                result.insert(entry).second,
                ExcMessage(
                  "Native coefficient records have duplicate physical owners."));
          return result;
        };
        const auto vessels        = collect(owned_vessels);
        const auto boundaries     = collect(owned_boundaries);
        const auto terminal_nodes = collect(owned_terminal_nodes);
        const auto inlets         = collect(owned_inlets);
        const auto reversed       = collect(owned_reversed);
        require_all(!vessels.empty() &&
                      vessels.size() < numbers::invalid_material_id,
                    "Native flow material ids cannot represent this network.");
        Physics             input;
        MetricFlowXTransfer transfer;
        transfer.reversed_vessels = reversed;
        for (const auto &[id, properties] : vessels)
          {
            const auto native =
              static_cast<types::material_id>(input.vessels.size());
            transfer.vessel_ids.emplace(id, native);
            input.vessels.emplace(native, properties);
          }
        for (const Id id : inlets)
          transfer.boundary_ids.emplace(id, 0);
        std::uint64_t boundary = 1;
        for (const auto &[id, properties] : boundaries)
          {
            if (boundary == 255)
              ++boundary;
            AssertThrow(
              boundary < numbers::invalid_boundary_id,
              ExcMessage("Native boundary ids cannot represent this network."));
            const auto native = static_cast<types::boundary_id>(boundary++);
            AssertThrow(transfer.boundary_ids.emplace(id, native).second,
                        ExcMessage("A physical inlet also has terminal data."));
            input.terminals.emplace(native, properties);
          }
        for (const auto &[id, node] : terminal_nodes)
          transfer.terminal_ids.emplace(id, transfer.boundary_ids.at(node));
        std::vector<std::pair<std::array<Id, 3>, double>> owned_pairs;
        for (const auto &pair : generation.terminal_pairs)
          owned_pairs.push_back(
            {{{pair.territory, pair.arterial, pair.venous}}, pair.resistance});
        std::set<Id> territories;
        for (const auto &pairs : Utilities::MPI::all_gather(comm, owned_pairs))
          for (const auto &[ids, resistance] : pairs)
            {
              AssertThrow(territories.insert(ids[0]).second,
                          ExcMessage(
                            "A terminal bed has duplicate physical owners."));
              input.terminal_connections.push_back(
                {transfer.terminal_ids.at(ids[1]),
                 transfer.terminal_ids.at(ids[2]),
                 resistance});
            }

        // This temporary input is used only to construct flow-oriented cells.
        // The caller's physical result keeps its growth reference orientation.
        auto mesh_input = network;
        auto intervals  = generation.discretization;
        for (auto &[id, vessel] : mesh_input.vessels)
          if (reversed.count(id))
            {
              std::swap(vessel.nodes[0], vessel.nodes[1]);
              std::swap(vessel.radii[0], vessel.radii[1]);
            }
        for (auto &cell : intervals.cells)
          if (reversed.count(cell.reference.vessel))
            {
              const double begin   = 1. - cell.reference.end;
              cell.reference.end   = 1. - cell.reference.begin;
              cell.reference.begin = begin;
            }
        transfer.flow_cells =
          create_triangulation(mesh_input, intervals, flow.triangulation());
        for (const auto &cell : flow.triangulation().active_cell_iterators())
          if (!cell->is_artificial())
            {
              const auto reference =
                physical_reference(cell->id(), transfer.flow_cells);
              cell->set_material_id(transfer.vessel_ids.at(reference.vessel));
              for (const auto face : cell->face_indices())
                if (cell->face(face)->at_boundary())
                  cell->face(face)->set_boundary_id(255);
            }
        // Endpoint ids are assigned by physical vessel owners, then travel
        // with numerical intervals to their execution owners and ghosts.
        std::map<Id, std::array<types::boundary_id, 2>> endpoint_boundaries;
        for (const auto &[id, vessel] : mesh_input.vessels)
          if (vessel.owner == rank)
            {
              auto &ids = endpoint_boundaries[id];
              for (unsigned int v = 0; v < 2; ++v)
                {
                  const auto boundary_id =
                    transfer.boundary_ids.find(vessel.nodes[v]);
                  ids[v] = boundary_id == transfer.boundary_ids.end() ?
                             255 :
                             boundary_id->second;
                }
            }
        const auto endpoints = collect(endpoint_boundaries);
        for (const auto &cell : flow.triangulation().active_cell_iterators())
          if (!cell->is_artificial())
            {
              const auto reference =
                physical_reference(cell->id(), transfer.flow_cells);
              if (reference.begin == 0. && cell->face(0)->at_boundary())
                cell->face(0)->set_boundary_id(
                  endpoints.at(reference.vessel)[0]);
              if (reference.end == 1. && cell->face(1)->at_boundary())
                cell->face(1)->set_boundary_id(
                  endpoints.at(reference.vessel)[1]);
            }
        flow.setup(input);
        return transfer;
      }
  }
} // namespace ImmersX::Vascularization

#endif
#endif
