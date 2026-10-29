// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_paths.h"

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/generation.h>
namespace ImmersX::Vascularization::internal
{
  namespace
  {
    struct Record
    {
      GrowthPathBranch branch;
      ForestBranch     state;
      System           system;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &branch &state &system;
      }
    };
  } // namespace
  GrowthCandidatePaths
  fetch_growth_candidates(const NetworkData                &network,
                          const ForestState                &forest,
                          const std::map<Id, unsigned int> &requests,
                          Id                                next_node,
                          Id                                next_vessel,
                          const Parameters                 &p,
                          MPI_Comm                          comm)
  {
    using namespace dealii;
    GrowthCandidatePaths result;
    const unsigned int   rank = Utilities::MPI::this_mpi_process(comm),
                       size   = Utilities::MPI::n_mpi_processes(comm);
    const auto phase          = [&](bool valid, const std::string &message) {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1;
      if (!all)
        result.validation.issues.push_back(
          {"candidate paths", invalid_id, message});
      return all;
    };
    bool valid = forest.validation.valid() &&
                 validate_data(network, size).valid() &&
                 std::isfinite(forest.demand_scale) && forest.demand_scale > 0.;
    for (const auto &[id, owner] : requests)
      valid = valid && id != invalid_id && owner < size;
    if (
      !phase(
        valid,
        "Candidate paths require a valid forest snapshot and explicit current branch owners."))
      return result;
    const auto count =
      Utilities::MPI::sum(static_cast<std::uint64_t>(forest.owned.size()),
                          comm);
    auto                 pending = requests;
    std::map<Id, Record> cache;
    while (
      Utilities::MPI::sum(static_cast<std::uint64_t>(pending.size()), comm) > 0)
      {
        ++result.rounds;
        if (!phase(
              result.rounds <= count,
              "Candidate dependencies do not terminate at a physical root."))
          return result;
        std::map<unsigned int, std::vector<Id>> send;
        for (const auto &[id, owner] : pending)
          send[owner].push_back(id);
        const auto incoming = Utilities::MPI::some_to_some(comm, send);
        std::map<unsigned int, std::map<Id, Record>> replies;
        for (const auto &[producer, ids] : incoming)
          for (Id id : ids)
            {
              const auto v = network.vessels.find(id);
              const auto b = forest.owned.find(id);
              if (v == network.vessels.end() || b == forest.owned.end() ||
                  v->second.owner != rank)
                {
                  valid = false;
                  continue;
                }
              const auto &vessel = v->second;
              Record      record;
              record.state     = b->second;
              record.system    = vessel.system;
              auto &branch     = record.branch;
              branch.edge      = {id,
                                  vessel.nodes,
                                  {{network.nodes.at(vessel.nodes[0]).position,
                                    network.nodes.at(vessel.nodes[1]).position}},
                                  b->second.radius};
              branch.demand    = b->second.demand;
              branch.viscosity = vessel.viscosity;
              branch.owner     = rank;
              for (unsigned int end = 0; end < 2; ++end)
                {
                  const auto &node = network.nodes.at(vessel.nodes[end]);
                  branch.port_normals[end] = node.inward_port_normal;
                  branch.node_owners[end]  = node.owner;
                }
              replies[producer].emplace(id, std::move(record));
            }
        if (
          !phase(
            valid,
            "A requested candidate or ancestor is absent from its authoritative owner."))
          return result;
        const auto      received = Utilities::MPI::some_to_some(comm, replies);
        std::vector<Id> fresh;
        for (const auto &[owner, records] : received)
          for (const auto &[id, record] : records)
            {
              valid = valid && pending.count(id) && pending.at(id) == owner &&
                      record.branch.owner == owner &&
                      cache.emplace(id, record).second;
              fresh.push_back(id);
            }
        for (const auto &[id, owner] : pending)
          valid = valid && cache.count(id);
        pending.clear();
        // A shared ancestor is fetched once, even if several candidates need
        // it. Already completed paths are not scanned again at each level.
        for (Id id : fresh)
          if (cache.count(id))
            {
              const auto &record = cache.at(id);
              if (record.state.parent != invalid_id &&
                  !cache.count(record.state.parent))
                {
                  const auto [it, inserted] =
                    pending.emplace(record.state.parent,
                                    record.state.parent_owner);
                  valid = valid && record.state.parent_owner < size &&
                          (inserted || it->second == record.state.parent_owner);
                }
            }
        if (
          !phase(
            valid,
            "Candidate replies disagree with the requested physical dependencies."))
          return result;
      }
    result.retained_branches = cache.size();
    for (const auto &[selected, owner] : requests)
      {
        const auto &first   = cache.at(selected);
        auto       &x       = result.contexts[selected];
        x.root              = first.state.root;
        x.system            = first.system;
        x.next_node         = next_node;
        x.next_vessel       = next_vessel;
        x.root_radius       = first.state.root_radius;
        x.root_demand       = first.state.root_demand;
        x.minimum_demand    = first.state.root_minimum_demand;
        x.volume_sum        = first.state.root_volume_moment;
        x.dissipation_sum   = first.state.root_resistance_moment;
        x.demand_scale      = forest.demand_scale;
        unsigned int system = 0;
        if (first.system == System::venous)
          system = 1;
        else if (first.system != System::arterial)
          valid = false;
        x.murray_exponent =
          system == 0 ? p.arterial_murray_exponent : p.venous.murray_exponent;
        x.total_demand           = forest.total_demand[system];
        x.forest_volume          = forest.volume[system];
        x.forest_power_numerator = forest.power_numerator[system];
        std::set<Id> visited;
        Id           id = selected;
        while (id != invalid_id && cache.count(id) && visited.insert(id).second)
          {
            const auto &record = cache.at(id);
            valid              = valid && record.system == first.system &&
                    record.state.root == x.root;
            if (!x.path.empty())
              valid =
                valid &&
                x.path.back().edge.nodes[0] == record.branch.edge.nodes[1] &&
                x.path.back().edge.points[0] == record.branch.edge.points[1];
            if (record.state.parent != invalid_id &&
                cache.count(record.state.parent))
              valid = valid && cache.at(record.state.parent).branch.owner ==
                                 record.state.parent_owner;
            x.path.push_back(record.branch);
            id = record.state.parent;
          }
        valid = valid && id == invalid_id && !x.path.empty() &&
                x.path.back().edge.id == x.root;
      }
    if (
      !phase(
        valid,
        "Candidate paths contain a cycle, a missing root or inconsistent endpoint geometry."))
      result.contexts.clear();
    return result;
  }
} // namespace ImmersX::Vascularization::internal
