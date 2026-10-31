// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include "growth_demand_state.h"
#include "growth_paths.h"

namespace ImmersX::Vascularization::internal
{
  namespace
  {
    struct DemandPathRecord
    {
      GrowthPathBranch                  branch;
      ForestBranch                      topology;
      GrowthDemandSummary               value;
      std::map<Id, GrowthDemandSummary> children;
      GrowthDemandRootView              root;
      double                            gamma  = 3.;
      System                            system = System::arterial;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &branch &topology &value &children &root &gamma &system;
      }
    };
  } // namespace

  GrowthCandidatePaths
  GrowthDemandState::candidate_paths(const NetworkData                &network,
                                     const GrowthDemandPreview        &preview,
                                     const std::map<Id, unsigned int> &requests,
                                     Id          next_node,
                                     Id          next_vessel,
                                     double      new_demand,
                                     double      minimum_scale,
                                     GrowthWork *work) const
  {
    using namespace dealii;
    const auto           comm = communicator;
    const auto           rank = Utilities::MPI::this_mpi_process(comm);
    const auto           size = Utilities::MPI::n_mpi_processes(comm);
    GrowthCandidatePaths result;
    const auto           phase = [&](bool local, const std::string &message) {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(local), comm) == 1;
      if (!all)
        result.validation.issues.push_back(
          {"demand paths", invalid_id, message});
      return all;
    };
    bool valid =
      report.valid() && preview.validation.valid() &&
      preview.snapshot_revision == revision &&
      preview.rank_totals.size() == size && std::isfinite(minimum_scale) &&
      minimum_scale > 0. &&
      (requests.empty() || (std::isfinite(new_demand) && new_demand > 0.));
    for (const auto &[id, owner] : requests)
      valid = valid && id != invalid_id && owner < size;
    if (
      !phase(
        valid,
        "Demand paths require one current collective preview and explicit branch owners."))
      return result;
    auto                           pending = requests;
    std::map<Id, DemandPathRecord> cache;
    while (
      Utilities::MPI::sum(static_cast<std::uint64_t>(pending.size()), comm) > 0)
      {
        ++result.rounds;
        count_work(work, GrowthOperation::path_exchange_rounds);
        if (!phase(result.rounds <= branch_count,
                   "Demand paths must terminate at physical roots."))
          return result;
        std::map<unsigned int, std::vector<Id>> send;
        for (const auto &[id, owner] : pending)
          send[owner].push_back(id);
        const auto incoming = Utilities::MPI::some_to_some(comm, send);
        std::map<unsigned int, std::map<Id, DemandPathRecord>> replies;
        for (const auto &[producer, ids] : incoming)
          for (Id id : ids)
            {
              const auto v = network.vessels.find(id);
              const auto b = branches.find(id);
              if (v == network.vessels.end() || b == branches.end() ||
                  v->second.owner != rank)
                {
                  valid = false;
                  continue;
                }
              DemandPathRecord record;
              record.topology = b->second.topology;
              record.value    = summary(id, preview);
              record.children = children(id, preview);
              record.gamma    = b->second.gamma;
              record.system   = v->second.system;
              if (record.topology.parent == invalid_id)
                record.root = root_view(id, preview);
              auto &branch     = record.branch;
              branch.edge      = {id,
                                  v->second.nodes,
                                  {{network.nodes.at(v->second.nodes[0]).position,
                                    network.nodes.at(v->second.nodes[1]).position}},
                                  0.};
              branch.viscosity = v->second.viscosity;
              branch.owner     = rank;
              for (unsigned int end = 0; end < 2; ++end)
                {
                  const auto &node = network.nodes.at(v->second.nodes[end]);
                  branch.port_normals[end] = node.inward_port_normal;
                  branch.node_owners[end]  = node.owner;
                }
              replies[producer].emplace(id, std::move(record));
            }
        if (
          !phase(
            valid,
            "Requested demand paths must exist on current authoritative branch owners."))
          return result;
        const auto      received = Utilities::MPI::some_to_some(comm, replies);
        std::vector<Id> fresh;
        for (const auto &[owner, records] : received)
          for (const auto &[id, record] : records)
            {
              valid = valid && pending.count(id) && pending.at(id) == owner &&
                      record.branch.owner == owner &&
                      cache.emplace(id, record).second;
              count_work(work, GrowthOperation::path_records_received);
              fresh.push_back(id);
            }
        for (const auto &[id, owner] : pending)
          valid = valid && cache.count(id);
        pending.clear();
        for (Id id : fresh)
          if (cache.count(id))
            {
              const auto &b = cache.at(id).topology;
              if (b.parent != invalid_id && !cache.count(b.parent))
                {
                  const auto [position, inserted] =
                    pending.emplace(b.parent, b.parent_owner);
                  valid = valid && b.parent_owner < size &&
                          (inserted || position->second == b.parent_owner);
                }
            }
        if (
          !phase(
            valid,
            "Demand path replies must agree with current physical dependencies."))
          return result;
      }
    result.retained_branches = cache.size();
    const auto physical      = [&](const GrowthDemandSummary &value) {
      return GrowthSubtreePhysiology{demand_value(value),
                                     value.volume_length,
                                     value.resistance_length,
                                     value.minimum_ratio};
    };
    for (const auto &[selected, owner] : requests)
      {
        auto       &x     = result.contexts[selected];
        const auto &first = cache.at(selected);
        x.root            = first.topology.root;
        x.system          = first.system;
        x.next_node       = next_node;
        x.next_vessel     = next_vessel;
        x.murray_exponent = first.gamma;
        x.minimum_scale   = minimum_scale;
        x.physiology.emplace();
        auto        &physics = *x.physiology;
        std::set<Id> visited;
        Id           id = selected;
        while (id != invalid_id && cache.count(id) && visited.insert(id).second)
          {
            const auto &record = cache.at(id);
            valid              = valid && record.system == first.system &&
                    record.gamma == first.gamma &&
                    record.topology.root == x.root;
            if (!x.path.empty())
              valid =
                valid &&
                x.path.back().edge.nodes[0] == record.branch.edge.nodes[1] &&
                x.path.back().edge.points[0] == record.branch.edge.points[1];
            x.path.push_back(record.branch);
            physics.path.push_back(physical(record.value));
            physics.children.emplace_back();
            for (const auto &[child, value] : record.children)
              physics.children.back().emplace(child, physical(value));
            id = record.topology.parent;
            count_work(work, GrowthOperation::candidate_path_visits);
          }
        valid = valid && id == invalid_id && !x.path.empty() &&
                x.path.back().edge.id == x.root;
        if (id != invalid_id || x.path.empty() || !cache.count(x.root))
          continue;
        const auto &root   = cache.at(x.root);
        const auto  system = root.root.system;
        valid              = valid && system < 3;
        if (system >= 3)
          continue;
        auto other = root.root.other_local;
        for (unsigned int rank_index = 0; rank_index < size; ++rank_index)
          if (rank_index != root.branch.owner)
            for (unsigned int j = 0; j < 9; ++j)
              other[j] += preview.rank_totals[rank_index][j];
        physics.other_volume            = other[3 * system + 1];
        physics.other_power_coefficient = other[3 * system + 2];
        physics.reference_volume        = accepted_volume[system];
        physics.reference_resistance    = accepted_resistance[system];
        if (system < 2 && preview.canonical_total_demand)
          physics.final_system_demand =
            (*preview.canonical_total_demand)[system];
        else if (system == 2 && preview.canonical_portal_demand)
          physics.final_system_demand = *preview.canonical_portal_demand;
        else
          physics.final_system_demand = preview.totals[3 * system] + new_demand;
        x.root_radius                 = root.root.radius;
        const long double root_demand = physics.path.back().demand;
        valid = valid && root_demand > 0. && std::isfinite(root_demand) &&
                std::isfinite(physics.final_system_demand) &&
                physics.final_system_demand > 0.;
        for (std::size_t i = 0; i < x.path.size(); ++i)
          {
            x.path[i].edge.radius =
              x.path[i].edge.id == x.root ?
                x.root_radius :
                x.root_radius * std::pow(physics.path[i].demand / root_demand,
                                         1.L / x.murray_exponent);
            valid = valid && std::isfinite(x.path[i].edge.radius) &&
                    x.path[i].edge.radius > 0.;
          }
      }
    if (
      !phase(
        valid,
        "Demand paths must form complete consistent physical roots with finite normalization."))
      result.contexts.clear();
    return result;
  }
} // namespace ImmersX::Vascularization::internal
