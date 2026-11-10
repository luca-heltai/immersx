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
    template <class Flow>
    auto
    metric_flow_x_minimum_positive_area(const Flow &flow, int)
      -> decltype(flow.get_minimum_positive_area())
    {
      return flow.get_minimum_positive_area();
    }

    // Older backends expose only their fixed area floor.
    template <class Flow>
    double
    metric_flow_x_minimum_positive_area(const Flow &, long)
    {
      return 1.e-10;
    }

    template <class Flow, class = void>
    struct HasNetworkPhysics : std::false_type
    {};

    template <class Flow>
    struct HasNetworkPhysics<Flow, std::void_t<typename Flow::NetworkPhysics>>
      : std::true_type
    {};

    template <class Physics, class = void>
    struct HasSealedTerminals : std::false_type
    {};

    template <class Physics>
    struct HasSealedTerminals<
      Physics,
      std::void_t<decltype(std::declval<Physics &>().sealed_terminals)>>
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
   * Portal inlets receive their own boundary id and call the boundary callback
   * with a null Terminal pointer, allowing independent root boundary data.
   * Both callbacks execute only at physical owners and may inspect local data.
   *
   * The native solver replicates coefficient and terminal tables. This setup
   * exchanges those records and identifiers, never the complete vessel graph.
   * A local mesh-only view reverses veins to orient execution cells along
   * flow. Original nodes, reference orientations, radii and ids remain intact.
   * Native microvascular links reuse the supplied positive terminal beds.
   * An optional set of physical Terminal ids declares sealed, zero-flow ends.
   * Its entries may be distributed or repeated on several ranks. Such ends
   * must exist and must not belong to a bed; their boundary callback is not
   * called. Older backends accept only an empty sealed set.
   */
  template <class Flow, class VesselPhysics, class BoundaryPhysics>
  MetricFlowXTransfer
  prepare_metric_flow_x(Flow                   &flow,
                        const GenerationResult &generation,
                        const VesselPhysics    &vessel_physics,
                        const BoundaryPhysics  &boundary_physics,
                        const std::set<Id>     &sealed_terminals = {})
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
        const double minimum_area =
          internal::metric_flow_x_minimum_positive_area(flow, 0);
        bool matching_area = std::isfinite(minimum_area) && minimum_area > 0.;
        for (const double other :
             Utilities::MPI::all_gather(comm, minimum_area))
          matching_area = matching_area && other == minimum_area;
        require_all(
          matching_area,
          "All ranks must use the same positive finite native minimum area.");
        using Physics = typename Flow::NetworkPhysics;
        std::set<Id> sealed;
        for (const auto &ids :
             Utilities::MPI::all_gather(comm, sealed_terminals))
          sealed.insert(ids.begin(), ids.end());
        require_all(
          sealed.empty() || internal::HasSealedTerminals<Physics>::value,
          "Sealed terminals require a compatible MetricFlowX backend.");
        bool valid_sealed_links = true;
        for (const auto &pair : generation.terminal_pairs)
          valid_sealed_links = valid_sealed_links &&
                               sealed.count(pair.arterial) == 0 &&
                               sealed.count(pair.venous) == 0;
        for (const auto &triplet : generation.terminal_triplets)
          valid_sealed_links = valid_sealed_links &&
                               sealed.count(triplet.arterial) == 0 &&
                               sealed.count(triplet.portal) == 0 &&
                               sealed.count(triplet.venous) == 0;
        require_all(valid_sealed_links,
                    "A microvascular bed terminal cannot also be sealed.");
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
                      minimum_area,
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
                  RCR boundary{};
                  if (sealed.count(id) == 0)
                    boundary = boundary_physics(terminal.node, node, &terminal);
                  valid =
                    owned_boundaries.emplace(terminal.node, boundary).second &&
                    valid;
                }
            for (const auto &[id, node] : network.nodes)
              if (node.owner == rank)
                {
                  if (node.port == Port::arterial_inlet)
                    owned_inlets.insert(id);
                  else if (node.port == Port::venous_outlet ||
                           node.port == Port::arterial_outlet ||
                           node.port == Port::portal_outlet ||
                           node.port == Port::portal_inlet)
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
        const auto   vessels        = collect(owned_vessels);
        const auto   boundaries     = collect(owned_boundaries);
        const auto   terminal_nodes = collect(owned_terminal_nodes);
        const auto   inlets         = collect(owned_inlets);
        const auto   reversed       = collect(owned_reversed);
        std::set<Id> sealed_nodes;
        for (const Id id : sealed)
          {
            const auto terminal = terminal_nodes.find(id);
            require_all(
              terminal != terminal_nodes.end(),
              "A declared sealed terminal is absent from the network.");
            sealed_nodes.insert(terminal->second);
          }
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
            if (sealed_nodes.count(id) != 0)
              {
                if constexpr (internal::HasSealedTerminals<Physics>::value)
                  input.sealed_terminals.insert(native);
              }
            else
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

        using Triplet = std::pair<std::array<Id, 4>, std::array<double, 3>>;
        std::vector<Triplet> owned_triplets;
        for (const auto &t : generation.terminal_triplets)
          owned_triplets.push_back(
            {{{t.territory, t.arterial, t.portal, t.venous}}, t.resistances});
        for (const auto &triplets :
             Utilities::MPI::all_gather(comm, owned_triplets))
          for (const auto &[ids, arms] : triplets)
            {
              AssertThrow(
                territories.insert(ids[0]).second,
                ExcMessage(
                  "A terminal compartment has duplicate physical owners."));
              std::array<long double, 3> conductance;
              long double                sum = 0.;
              for (unsigned int i = 0; i < 3; ++i)
                {
                  AssertThrow(std::isfinite(arms[i]) && arms[i] > 0.,
                              ExcMessage(
                                "Invalid microvascular arm resistance."));
                  conductance[i] = 1.L / arms[i];
                  sum += conductance[i];
                }
              // Eliminate the common pressure using the native
              // terminal-resistor graph.
              for (unsigned int i = 0; i < 3; ++i)
                for (unsigned int j = i + 1; j < 3; ++j)
                  {
                    const double resistance = static_cast<double>(
                      sum / conductance[i] / conductance[j]);
                    AssertThrow(
                      std::isfinite(resistance) && resistance > 0.,
                      ExcMessage("Unrepresentable microvascular conductance."));
                    input.terminal_connections.push_back(
                      {transfer.terminal_ids.at(ids[i + 1]),
                       transfer.terminal_ids.at(ids[j + 1]),
                       resistance});
                  }
            }

        transfer.flow_cells = create_triangulation(
          network,
          generation.discretization,
          flow.triangulation(),
          reversed,
          [&](const VesselCell &interval, const Vessel &vessel) {
            std::array<types::boundary_id, 2> ids{{255, 255}};
            for (unsigned int end = 0; end < 2; ++end)
              if ((end == 0 ? interval.begin : interval.end) == double(end))
                {
                  const auto found =
                    transfer.boundary_ids.find(vessel.nodes[end]);
                  ids[end] =
                    found == transfer.boundary_ids.end() ? 255 : found->second;
                }
            return ids;
          });
        for (const auto &cell : flow.triangulation().active_cell_iterators())
          if (!cell->is_artificial())
            {
              const auto reference =
                physical_reference(cell->id(), transfer.flow_cells);
              cell->set_material_id(transfer.vessel_ids.at(reference.vessel));
            }
        flow.setup(input);
        return transfer;
      }
  }
} // namespace ImmersX::Vascularization

#endif
#endif
