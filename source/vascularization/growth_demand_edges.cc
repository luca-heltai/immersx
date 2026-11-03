// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include "growth_demand_state.h"

namespace ImmersX::Vascularization::internal
{
  std::map<Id, GrowthDemandRootView>
  GrowthDemandState::root_controls(const std::set<Id>        &requested,
                                   const GrowthDemandPreview &preview,
                                   ValidationReport          &validation) const
  {
    using namespace dealii;
    const auto comm  = communicator;
    const auto size  = Utilities::MPI::n_mpi_processes(comm);
    bool       valid = report.valid() && preview.validation.valid() &&
                 preview.snapshot_revision == revision;
    const auto phase = [&](bool local, const std::string &message) {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(local), comm) == 1;
      if (!all)
        validation.issues.push_back({"demand roots", invalid_id, message});
      return all;
    };
    std::map<unsigned int, std::vector<Id>> send;
    for (Id root : requested)
      send[root % size].push_back(root);
    const auto root_requests = Utilities::MPI::some_to_some(comm, send);
    std::map<unsigned int, std::map<Id, unsigned int>> owner_replies;
    for (const auto &[producer, ids] : root_requests)
      for (Id root : ids)
        {
          const auto found = root_directory.find(root);
          if (found == root_directory.end())
            {
              valid = false;
              continue;
            }
          owner_replies[producer].emplace(root, found->second);
        }
    if (!phase(
          valid,
          "Mutable obstacles need a current physical root directory entry."))
      return {};
    const auto owners = Utilities::MPI::some_to_some(comm, owner_replies);
    std::map<Id, unsigned int> roots;
    send.clear();
    for (const auto &[shard, records] : owners)
      for (const auto &[root, owner] : records)
        {
          valid = requested.count(root) && root % size == shard &&
                  owner < size && roots.emplace(root, owner).second && valid;
          if (owner < size)
            send[owner].push_back(root);
        }
    valid = valid && roots.size() == requested.size();
    if (
      !phase(
        valid,
        "Every requested root must resolve to one current authoritative owner."))
      return {};
    const auto controls_requested = Utilities::MPI::some_to_some(comm, send);
    std::map<unsigned int, std::map<Id, GrowthDemandRootView>> controls;
    for (const auto &[producer, ids] : controls_requested)
      for (Id root : ids)
        {
          if (!owned_roots.count(root))
            {
              valid = false;
              continue;
            }
          controls[producer].emplace(root, root_view(root, preview));
        }
    if (
      !phase(
        valid,
        "Physical root controls must be present on their current native owner."))
      return {};
    const auto controls_received = Utilities::MPI::some_to_some(comm, controls);
    std::map<Id, GrowthDemandRootView> root_values;
    for (const auto &[owner, records] : controls_received)
      for (const auto &[root, value] : records)
        valid = roots.count(root) && roots.at(root) == owner &&
                root_values.emplace(root, value).second && valid;
    valid = valid && root_values.size() == requested.size();
    if (
      !phase(
        valid,
        "Every mutable obstacle needs its preview-updated physical root normalization."))
      return {};
    return root_values;
  }

  GrowthDemandEdges
  GrowthDemandState::physical_edges(
    const NetworkData                &network,
    const GrowthDemandPreview        &preview,
    const std::map<Id, unsigned int> &requests) const
  {
    using namespace dealii;
    const auto        comm = communicator;
    const auto        rank = Utilities::MPI::this_mpi_process(comm);
    const auto        size = Utilities::MPI::n_mpi_processes(comm);
    GrowthDemandEdges result;
    const auto        phase = [&](bool local, const std::string &message) {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(local), comm) == 1;
      if (!all)
        result.validation.issues.push_back(
          {"demand edges", invalid_id, message});
      return all;
    };
    bool valid = report.valid() && preview.validation.valid() &&
                 preview.snapshot_revision == revision;
    std::map<unsigned int, std::vector<Id>> send;
    for (const auto &[id, owner] : requests)
      {
        valid = valid && id != invalid_id && owner < size;
        if (owner < size)
          send[owner].push_back(id);
      }
    if (
      !phase(
        valid,
        "Physical edge previews require current revisions and native owners."))
      return result;
    const auto incoming = Utilities::MPI::some_to_some(comm, send);
    std::map<unsigned int, std::map<Id, GrowthDemandEdge>> replies;
    for (const auto &[producer, ids] : incoming)
      for (Id id : ids)
        {
          const auto v = network.vessels.find(id);
          if (v == network.vessels.end() || v->second.owner != rank)
            {
              valid = false;
              continue;
            }
          GrowthDemandEdge value;
          value.edge   = {id,
                          v->second.nodes,
                          {{network.nodes.at(v->second.nodes[0]).position,
                            network.nodes.at(v->second.nodes[1]).position}},
                          std::max(v->second.radii[0], v->second.radii[1])};
          const auto b = branches.find(id);
          if (b != branches.end())
            {
              value.root   = b->second.topology.root;
              value.demand = demand_value(summary(id, preview));
            }
          replies[producer].emplace(id, std::move(value));
        }
    if (
      !phase(
        valid,
        "Physical edge requests must reach their current authoritative vessels."))
      return result;
    const auto   received = Utilities::MPI::some_to_some(comm, replies);
    std::set<Id> needed_roots;
    for (const auto &[owner, records] : received)
      for (const auto &[id, value] : records)
        {
          valid = requests.count(id) && requests.at(id) == owner &&
                  result.records.emplace(id, value).second && valid;
          if (value.root != invalid_id)
            needed_roots.insert(value.root);
        }
    valid = valid && result.records.size() == requests.size();
    if (!phase(valid,
               "Each physical edge must have one authoritative preview reply."))
      return result;
    const auto root_values =
      root_controls(needed_roots, preview, result.validation);
    if (!result.validation.valid())
      return result;
    for (auto &[id, value] : result.records)
      if (value.root != invalid_id)
        {
          const auto &root  = root_values.at(value.root);
          value.root_demand = demand_value(root.summary);
          valid = valid && std::isfinite(root.gamma) && root.gamma > 0. &&
                  std::isfinite(value.demand) && value.demand > 0. &&
                  std::isfinite(value.root_demand) && value.root_demand > 0. &&
                  value.demand <= value.root_demand;
          value.edge.radius =
            id == value.root ?
              root.radius :
              static_cast<double>(
                root.radius *
                std::pow(value.demand / value.root_demand, 1.L / root.gamma));
        }
    for (const auto &[id, value] : result.records)
      valid =
        valid && std::isfinite(value.edge.radius) && value.edge.radius > 0.;
    if (
      !phase(
        valid,
        "Previewed obstacle radii must be finite and positive; no radius floor is applied."))
      result.records.clear();
    return result;
  }
} // namespace ImmersX::Vascularization::internal
