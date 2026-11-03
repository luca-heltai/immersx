// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_ownership_delta.h"

#include <deal.II/base/mpi.h>

#include <deal.II/grid/grid_tools.h>

#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include <algorithm>
#include <cmath>
#include <tuple>

#include "growth_geometry.h"
#include "network_serialization.h"
namespace ImmersX::Vascularization::internal
{
  namespace
  {
    bool
    same_node(const Node &a, const Node &b)
    {
      return a.position == b.position && a.owner == b.owner &&
             a.port == b.port && a.provenance == b.provenance &&
             a.inward_port_normal == b.inward_port_normal;
    }
  } // namespace
  GrowthOwnershipDelta
  prepare_growth_ownership(const NetworkData              &accepted,
                           const std::vector<GrowthSplit> &local_selected,
                           BackgroundGeometry             &geometry,
                           MPI_Comm                        comm,
                           GrowthWork                     *work)
  {
    using namespace dealii;
    const auto rank = Utilities::MPI::this_mpi_process(comm);
    const auto size = Utilities::MPI::n_mpi_processes(comm);
    const auto all  = [&](bool valid) {
      return Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1;
    };
    const auto failure = [](const std::string &message) {
      GrowthOwnershipDelta failed;
      failed.validation.issues.push_back(
        {"growth ownership", invalid_id, message});
      return failed;
    };
    GrowthOwnershipDelta result;
    auto                 local = local_selected;
    std::sort(local.begin(), local.end(), [](const auto &a, const auto &b) {
      return a.terminal < b.terminal;
    });
    for (const auto &part : Utilities::MPI::all_gather(comm, local))
      result.global_splits.insert(result.global_splits.end(),
                                  part.begin(),
                                  part.end());
    if (result.global_splits.empty())
      return result;
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(geometry.cache.get_triangulation().get_mpi_communicator(),
                     comm,
                     &comparison);
    bool         valid = comparison == MPI_IDENT || comparison == MPI_CONGRUENT;
    std::set<Id> selected, new_nodes, new_vessels, new_terminals;
    for (const auto &s : result.global_splits)
      {
        valid = s.owner < size && s.vessel != invalid_id &&
                s.old_nodes[0] != invalid_id && s.old_nodes[1] != invalid_id &&
                s.old_nodes[0] != s.old_nodes[1] && s.territory != invalid_id &&
                std::isfinite(s.demand) && s.demand > 0. &&
                selected.insert(s.vessel).second && valid;
        for (Id id : {s.junction, s.tip})
          valid = id != invalid_id && new_nodes.insert(id).second &&
                  !accepted.nodes.count(id) && valid;
        for (Id id : {s.continuation, s.leaf})
          valid = id != invalid_id && new_vessels.insert(id).second &&
                  !accepted.vessels.count(id) && valid;
        valid = s.terminal != invalid_id &&
                new_terminals.insert(s.terminal).second &&
                !accepted.terminals.count(s.terminal) && valid;
        for (unsigned int d = 0; d < 3; ++d)
          valid = std::isfinite(s.junction_position[d]) &&
                  std::isfinite(s.tip_position[d]) &&
                  std::isfinite(s.old_points[0][d]) &&
                  std::isfinite(s.old_points[1][d]) && valid;
        if (s.owner != rank)
          continue;
        result.written_splits.push_back(s);
        const auto vessel = accepted.vessels.find(s.vessel);
        if (vessel == accepted.vessels.end() || vessel->second.owner != rank ||
            vessel->second.nodes != s.old_nodes)
          {
            valid = false;
            continue;
          }
        result.written.vessels.emplace(s.vessel, vessel->second);
        for (unsigned int e = 0; e < 2; ++e)
          {
            const auto node = accepted.nodes.find(s.old_nodes[e]);
            if (node == accepted.nodes.end() || node->second.owner >= size ||
                node->second.position != s.old_points[e])
              valid = false;
            else
              result.written.nodes.emplace(node->first, node->second);
          }
      }
    if (!all(valid))
      return failure(
        "A selected split is stale, duplicates a write, or reuses an existing physical id.");

    // Request only the old endpoints touched by this batch, never all ghosts.
    std::map<unsigned int, std::vector<Id>> requests;
    for (const auto &[id, node] : result.written.nodes)
      requests[node.owner].push_back(id);
    const auto incoming = Utilities::MPI::some_to_some(comm, requests);
    std::map<unsigned int, std::map<Id, Node>> replies;
    for (const auto &[source, ids] : incoming)
      for (Id id : ids)
        {
          const auto node = accepted.nodes.find(id);
          if (node == accepted.nodes.end() || node->second.owner != rank)
            valid = false;
          else
            replies[source].emplace(id, node->second);
        }
    std::size_t received_nodes = 0;
    for (const auto &[source, nodes] :
         Utilities::MPI::some_to_some(comm, replies))
      for (const auto &[id, node] : nodes)
        {
          const auto current = result.written.nodes.find(id);
          valid              = current != result.written.nodes.end() &&
                  node.owner == source && same_node(current->second, node) &&
                  valid;
          ++received_nodes;
        }
    valid = received_nodes == result.written.nodes.size() && valid;
    for (const auto &s : result.written_splits)
      valid = apply_growth_split(result.written, s) && valid;
    count_work(work,
               GrowthOperation::validation_record_visits,
               result.written.nodes.size() + result.written.vessels.size() +
                 result.written.terminals.size());
    valid = validate_data(result.written, size).valid() && valid;
    if (!all(valid))
      return failure(
        "Native endpoint authority or the sparse split patch is invalid.");

    std::vector<Point<3>>     positions;
    std::map<Id, std::size_t> node_indices, vessel_indices;
    for (const auto &s : result.written_splits)
      {
        for (Id id : {s.junction, s.tip})
          {
            node_indices.emplace(id, positions.size());
            positions.push_back(result.written.nodes.at(id).position);
          }
        for (Id id : {s.vessel, s.continuation, s.leaf})
          {
            const auto &v = result.written.vessels.at(id);
            vessel_indices.emplace(id, positions.size());
            positions.push_back(
              .5 * result.written.nodes.at(v.nodes[0]).position +
              .5 * result.written.nodes.at(v.nodes[1]).position);
          }
      }
    count_work(work, GrowthOperation::node_localizations, node_indices.size());
    count_work(work,
               GrowthOperation::vessel_midpoint_localizations,
               vessel_indices.size());
    const auto &routing = geometry.routing(comm, work);
    if (!all(!routing.covering_tree.empty()))
      return failure(
        "Vascular ownership requires a nonempty global background.");
    // Same native kernel and lowest-rank tie rule as full redistribution.
    const auto matches =
      GridTools::distributed_compute_point_locations(geometry.cache,
                                                     positions,
                                                     routing.boxes);
    std::map<unsigned int, std::vector<unsigned int>> located;
    for (std::size_t cell = 0; cell < std::get<0>(matches).size(); ++cell)
      for (std::size_t i = 0; i < std::get<2>(matches)[cell].size(); ++i)
        located[std::get<4>(matches)[cell][i]].push_back(
          std::get<2>(matches)[cell][i]);
    std::vector<unsigned int> destinations(positions.size(), size);
    for (const auto &[owner, indices] :
         Utilities::MPI::some_to_some(comm, located))
      for (auto index : indices)
        destinations.at(index) = std::min(destinations.at(index), owner);
    std::map<Id, unsigned int> node_owners, vessel_owners;
    for (const auto &[id, index] : node_indices)
      {
        auto &node = result.written.nodes.at(id);
        if (destinations[index] != size)
          node.owner = destinations[index];
        node_owners.emplace(id, node.owner);
      }
    for (const auto &[id, index] : vessel_indices)
      {
        auto &vessel = result.written.vessels.at(id);
        if (destinations[index] != size)
          vessel.owner = destinations[index];
        vessel_owners.emplace(id, vessel.owner);
      }
    for (const auto &s : result.written_splits)
      result.written.terminals.at(s.terminal).owner =
        result.written.nodes.at(s.tip).owner;
    for (const auto &part : Utilities::MPI::all_gather(comm, node_owners))
      result.node_owners.insert(part.begin(), part.end());
    for (const auto &part : Utilities::MPI::all_gather(comm, vessel_owners))
      result.vessel_owners.insert(part.begin(), part.end());

    std::map<unsigned int, NetworkData> packets;
    for (const auto &[id, index] : node_indices)
      {
        const auto &node = result.written.nodes.at(id);
        packets[node.owner].nodes.emplace(id, node);
        if (node.owner != rank)
          count_work(work, GrowthOperation::nodes_migrated);
      }
    for (const auto &[id, vessel] : result.written.vessels)
      {
        auto &packet = packets[vessel.owner];
        packet.vessels.emplace(id, vessel);
        for (Id node : vessel.nodes)
          packet.nodes.emplace(node, result.written.nodes.at(node));
        if (vessel.owner != rank)
          count_work(work, GrowthOperation::vessels_migrated);
      }
    for (const auto &[id, terminal] : result.written.terminals)
      {
        auto &packet = packets[terminal.owner];
        packet.terminals.emplace(id, terminal);
        packet.nodes.emplace(terminal.node,
                             result.written.nodes.at(terminal.node));
        if (terminal.owner != rank)
          count_work(work, GrowthOperation::terminals_migrated);
      }
    for (const auto &[source, packet] :
         Utilities::MPI::some_to_some(comm, packets))
      {
        for (const auto &[id, node] : packet.nodes)
          {
            const auto [it, inserted] = result.owned.nodes.emplace(id, node);
            valid = (inserted || same_node(it->second, node)) && valid;
          }
        for (const auto &[id, vessel] : packet.vessels)
          valid = vessel.owner == rank &&
                  result.owned.vessels.emplace(id, vessel).second && valid;
        for (const auto &[id, terminal] : packet.terminals)
          valid = terminal.owner == rank &&
                  result.owned.terminals.emplace(id, terminal).second && valid;
      }
    count_work(work,
               GrowthOperation::validation_record_visits,
               result.owned.nodes.size() + result.owned.vessels.size() +
                 result.owned.terminals.size());
    valid = validate_data(result.owned, size).valid() && valid;
    if (!all(valid))
      return failure(
        "Native ownership packets have conflicting authorities or incomplete endpoints.");
    for (const auto &[id, vessel] : result.owned.vessels)
      result.changed_owned.insert(id);
    for (const auto &s : result.written_splits)
      if (result.vessel_owners.at(s.vessel) != rank)
        result.released_owned.insert(s.vessel);
    return result;
  }
} // namespace ImmersX::Vascularization::internal
