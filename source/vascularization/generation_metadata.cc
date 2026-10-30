// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "archive_internal.h"

namespace ImmersX::Vascularization::internal
{
  namespace
  {
    using Rank = dealii::types::subdomain_id;
    using Metadata =
      std::pair<std::vector<Territory>, std::vector<TerminalPair>>;
    struct Endpoint
    {
      Id               territory = invalid_id;
      System           system    = System::arterial;
      Rank             owner     = 0;
      dealii::Point<3> position;
      double           demand = 0.;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &territory &system &owner &position &demand;
      }
    };
  } // namespace
  RetainedMetadata
  retain_metadata(const NetworkData               &before,
                  const std::vector<Territory>    &territories,
                  const std::vector<TerminalPair> &pairs,
                  const NetworkData               &after,
                  MPI_Comm                         comm)
  {
    using namespace dealii;
    RetainedMetadata result;
    if (Utilities::MPI::sum(static_cast<Id>(territories.size() + pairs.size()),
                            comm) == 0)
      return result;
    const Rank rank = Utilities::MPI::this_mpi_process(comm),
               size = Utilities::MPI::n_mpi_processes(comm);
    const auto fail = [&](Id id, const std::string &message) {
      result.validation.issues.push_back({"input territory", id, message});
    };
    const auto valid = [&]() {
      const bool all = Utilities::MPI::min(static_cast<unsigned int>(
                                             result.validation.valid()),
                                           comm) == 1;
      if (!all && result.validation.valid())
        fail(invalid_id,
             "Input territory retention failed on another process.");
      return all;
    };
    std::map<Rank, std::vector<std::pair<Id, Id>>> old_send;
    for (const auto &[id, t] : before.terminals)
      if (t.owner == rank && t.system == System::arterial &&
          t.territory != invalid_id)
        old_send[t.territory % size].emplace_back(t.territory, id);
    std::map<Id, std::vector<Id>> old_arteries;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(comm, old_send))
      for (const auto &[territory, terminal] : entries)
        old_arteries[territory].push_back(terminal);
    std::map<Rank, Metadata> metadata_send;
    for (const auto &t : territories)
      metadata_send[t.id % size].first.push_back(t);
    for (const auto &p : pairs)
      metadata_send[p.territory % size].second.push_back(p);
    std::map<Id, Territory>    originals;
    std::map<Id, TerminalPair> beds;
    for (const auto &[source, metadata] :
         Utilities::MPI::some_to_some(comm, metadata_send))
      {
        for (const auto &t : metadata.first)
          if (!originals.emplace(t.id, t).second)
            fail(t.id, "Duplicate input territory.");
        for (const auto &p : metadata.second)
          if (!beds.emplace(p.territory, p).second)
            fail(p.territory, "Duplicate input bed.");
      }
    std::map<Rank, std::map<Id, Endpoint>> current_send;
    for (const auto &[id, t] : after.terminals)
      if (t.owner == rank)
        current_send[id % size].emplace(
          id,
          Endpoint{t.territory,
                   t.system,
                   rank,
                   after.nodes.at(t.node).position,
                   t.demand});
    std::map<Id, Endpoint> current;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(comm, current_send))
      for (const auto &[id, t] : entries)
        if (!current.emplace(id, t).second)
          fail(id, "Duplicate current terminal authority.");
    std::map<Rank, std::vector<Id>> requests;
    for (const auto &[id, t] : originals)
      {
        if (old_arteries[id].size() != 1)
          {
            fail(id, "An input territory must identify one arterial leaf.");
            continue;
          }
        const Id a = old_arteries.at(id)[0];
        requests[a % size].push_back(a);
        if (beds.count(id))
          {
            const auto &p = beds.at(id);
            if (p.arterial != a)
              fail(id, "Input bed disagrees with its arterial territory.");
            requests[p.venous % size].push_back(p.venous);
          }
      }
    if (!valid())
      return result;
    std::map<Rank, std::map<Id, Endpoint>> replies;
    for (const auto &[source, ids] :
         Utilities::MPI::some_to_some(comm, requests))
      for (Id id : ids)
        if (current.count(id))
          replies[source].emplace(id, current.at(id));
    std::map<Id, Endpoint> endpoints;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(comm, replies))
      endpoints.insert(entries.begin(), entries.end());
    std::map<Rank, Metadata> routes;
    for (const auto &[id, t] : originals)
      {
        const Id   arterial = old_arteries.at(id)[0];
        const bool a_alive  = endpoints.count(arterial);
        if (beds.count(id))
          {
            const auto &p       = beds.at(id);
            const bool  v_alive = endpoints.count(p.venous);
            if (a_alive != v_alive)
              {
                fail(
                  id,
                  "The two endpoints of an input microvascular bed must be extended together.");
                continue;
              }
            if (v_alive)
              {
                const auto &v = endpoints.at(p.venous);
                if (v.system != System::venous || v.territory != id ||
                    v.demand != t.demand)
                  fail(
                    id,
                    "An untouched input venous terminal changed its physical metadata.");
              }
          }
        if (!a_alive)
          continue;
        const auto &a = endpoints.at(arterial);
        if (a.system != System::arterial || a.territory != id ||
            a.demand != t.demand || a.position != t.position)
          fail(
            id,
            "An untouched input arterial terminal changed its physical metadata.");
        auto retained  = t;
        retained.owner = a.owner;
        routes[a.owner].first.push_back(retained);
        if (beds.count(id))
          routes[a.owner].second.push_back(beds.at(id));
      }
    if (!valid())
      return result;
    for (const auto &[source, metadata] :
         Utilities::MPI::some_to_some(comm, routes))
      {
        result.territories.insert(result.territories.end(),
                                  metadata.first.begin(),
                                  metadata.first.end());
        result.pairs.insert(result.pairs.end(),
                            metadata.second.begin(),
                            metadata.second.end());
      }
    return result;
  }
} // namespace ImmersX::Vascularization::internal
