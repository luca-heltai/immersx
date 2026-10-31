// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_demand_callbacks_h
#define immersx_internal_growth_demand_callbacks_h
#include <deal.II/base/mpi.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>

#include <cmath>
#include <functional>
#include <map>
#include <set>

#include "growth_demand.h"
namespace ImmersX::Vascularization::internal
{
  // All callbacks participate on every rank. Isolated previews are called in
  // global key order, with key=(producer rank, allocated new terminal id).
  // BatchProposal.id must use that id; the physical vessel id is separate.
  struct GrowthDemandCallbacks
  {
    std::function<GrowthDemandJointUpdate(const NetworkData &,
                                          const std::set<Id> &,
                                          std::uint64_t)>
      bootstrap;
    std::function<GrowthDemandProposal(const GrowthProposalKey &,
                                       Id,
                                       Id,
                                       const dealii::Point<3> &,
                                       std::uint64_t)>
      preview;
    std::function<
      GrowthDemandJointUpdate(const std::vector<GrowthProposalKey> &,
                              std::uint64_t)>
      joint;
    // Stage the joint tissue changes before the physical transaction vote.
    // No further preview is allowed until adopt/discard finishes that stage.
    std::function<void(std::uint64_t, std::uint64_t)> prepare_adopt;
    std::function<void()>                             adopt, discard;
  };

  struct GrowthLeafTerminals
  {
    ValidationReport validation;
    std::set<Id>     owned;
  };

  // Only topology and native node authority are used here. In particular,
  // this bootstrap does not interpret provisional positive terminal labels
  // as Di and does not run Murray, flow, resistance or objective evaluation.
  inline GrowthLeafTerminals
  collect_growth_leaf_terminals(const NetworkData  &network,
                                const std::set<Id> &mutable_owned,
                                MPI_Comm            comm)
  {
    using namespace dealii;
    const auto          rank = Utilities::MPI::this_mpi_process(comm);
    GrowthLeafTerminals result;
    const auto          phase = [&](bool valid, const std::string &message) {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1;
      if (!all)
        result.validation.issues.push_back(
          {"demand bootstrap", invalid_id, message});
      return all;
    };
    bool                                                   valid = true;
    std::map<unsigned int, std::vector<std::array<Id, 3>>> send;
    for (Id id : mutable_owned)
      {
        const auto v = network.vessels.find(id);
        if (v == network.vessels.end() || v->second.owner != rank)
          {
            valid = false;
            continue;
          }
        for (unsigned int end = 0; end < 2; ++end)
          {
            const auto node = network.nodes.find(v->second.nodes[end]);
            if (node == network.nodes.end())
              {
                valid = false;
                continue;
              }
            send[node->second.owner].push_back({node->first, id, end});
          }
      }
    if (
      !phase(
        valid,
        "Provisional mutable endpoints must be present on their vessel owner."))
      return result;
    const auto received = Utilities::MPI::some_to_some(comm, send);
    struct Links
    {
      std::set<Id> incoming, outgoing;
    };
    std::map<Id, Links> links;
    for (const auto &[source, arcs] : received)
      for (const auto &arc : arcs)
        {
          const auto node = network.nodes.find(arc[0]);
          if (node == network.nodes.end() || node->second.owner != rank)
            {
              valid = false;
              continue;
            }
          auto &ids =
            arc[2] == 0 ? links[arc[0]].outgoing : links[arc[0]].incoming;
          valid = ids.insert(arc[1]).second && valid;
        }
    std::map<Id, std::set<Id>> terminals;
    for (const auto &[id, t] : network.terminals)
      if (t.owner == rank)
        terminals[t.node].insert(id);
    for (const auto &[node, l] : links)
      {
        valid = valid && l.incoming.size() <= 1 && l.outgoing.size() <= 2;
        if (l.incoming.size() == 1 && l.outgoing.empty())
          {
            if (terminals[node].size() != 1)
              valid = false;
            else
              result.owned.insert(*terminals.at(node).begin());
          }
        else if (!l.incoming.empty() && !l.outgoing.empty())
          valid = valid && terminals[node].empty();
      }
    phase(
      valid,
      "Provisional leaves need one native terminal; internal mutable junctions need none.");
    return result;
  }

  // Bootstrap only: every already-created mutable leaf must receive exactly
  // one true Di. Inputs may be on producer ranks or directory shards. Ghost
  // terminal records are refreshed by the subsequent physical initialization.
  // No authoritative value changes unless all ranks validate the full update.
  inline ValidationReport
  initialize_growth_demands(NetworkData                   &network,
                            const std::set<Id>            &owned_leaf_terminals,
                            const GrowthDemandJointUpdate &weights,
                            std::uint64_t                  snapshot_revision,
                            MPI_Comm                       comm)
  {
    using namespace dealii;
    const auto       rank   = Utilities::MPI::this_mpi_process(comm);
    const auto       size   = Utilities::MPI::n_mpi_processes(comm);
    ValidationReport report = weights.validation;
    const auto       phase  = [&](bool valid, const std::string &message) {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1;
      if (!all)
        report.issues.push_back({"demand bootstrap", invalid_id, message});
      return all;
    };
    bool valid = report.valid() && !weights.integration_budget_exhausted &&
                 weights.update.snapshot_revision == snapshot_revision;
    std::map<unsigned int, std::vector<Id>> registrations;
    for (Id id : owned_leaf_terminals)
      {
        const auto t = network.terminals.find(id);
        if (t == network.terminals.end() || t->second.owner != rank)
          valid = false;
        else
          registrations[id % size].push_back(id);
      }
    if (
      !phase(
        valid,
        "Bootstrap weights require valid revisions and native owned mutable leaves."))
      return report;
    const auto registered = Utilities::MPI::some_to_some(comm, registrations);
    std::map<Id, unsigned int> owners;
    for (const auto &[owner, ids] : registered)
      for (Id id : ids)
        valid = owners.emplace(id, owner).second && valid;
    std::map<unsigned int, std::vector<TerminalDemandUpdate>> send;
    for (const auto &update : weights.update.terminals)
      send[update.terminal % size].push_back(update);
    for (const auto &update : weights.new_terminals)
      send[update.terminal % size].push_back(update);
    const auto           received = Utilities::MPI::some_to_some(comm, send);
    std::map<Id, double> demands;
    for (const auto &[source, updates] : received)
      for (const auto &update : updates)
        valid = std::isfinite(update.demand) && update.demand > 0. &&
                owners.count(update.terminal) &&
                demands.emplace(update.terminal, update.demand).second && valid;
    valid = valid && demands.size() == owners.size();
    if (
      !phase(
        valid,
        "Every provisional leaf needs exactly one finite positive integrated demand; no floor is applied."))
      return report;
    send.clear();
    for (const auto &[id, demand] : demands)
      send[owners.at(id)].push_back({id, demand});
    const auto           delivered = Utilities::MPI::some_to_some(comm, send);
    std::map<Id, double> local;
    for (const auto &[source, updates] : delivered)
      for (const auto &update : updates)
        valid = owned_leaf_terminals.count(update.terminal) &&
                local.emplace(update.terminal, update.demand).second && valid;
    valid = valid && local.size() == owned_leaf_terminals.size();
    if (
      !phase(
        valid,
        "Integrated demands must arrive at every native mutable terminal authority."))
      return report;
    for (const auto &[id, demand] : local)
      network.terminals.at(id).demand = demand;
    return report;
  }
} // namespace ImmersX::Vascularization::internal
#endif
