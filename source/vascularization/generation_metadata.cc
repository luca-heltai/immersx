// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "archive_internal.h"

namespace ImmersX::Vascularization::internal
{
  namespace
  {
    using Rank     = dealii::types::subdomain_id;
    using Metadata = TerritoryMetadata;
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
  retain_metadata(const NetworkData                  &before,
                  const std::vector<Territory>       &territories,
                  const std::vector<TerminalPair>    &pairs,
                  const NetworkData                  &after,
                  MPI_Comm                            comm,
                  const std::vector<TerminalTriplet> &triplets)
  {
    using namespace dealii;
    RetainedMetadata result;
    if (Utilities::MPI::sum(static_cast<Id>(territories.size() + pairs.size() +
                                            triplets.size()),
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
      metadata_send[t.id % size].territories.push_back(t);
    for (const auto &p : pairs)
      metadata_send[p.territory % size].pairs.push_back(p);
    for (const auto &t : triplets)
      metadata_send[t.territory % size].triplets.push_back(t);
    std::map<Id, TerminalTriplet> compartments;
    std::map<Id, Territory>       originals;
    std::map<Id, TerminalPair>    beds;
    for (const auto &[source, metadata] :
         Utilities::MPI::some_to_some(comm, metadata_send))
      {
        for (const auto &t : metadata.territories)
          if (!originals.emplace(t.id, t).second)
            fail(t.id, "Duplicate input territory.");
        for (const auto &p : metadata.pairs)
          if (!beds.emplace(p.territory, p).second)
            fail(p.territory, "Duplicate input bed.");
        for (const auto &t : metadata.triplets)
          if (!compartments.emplace(t.territory, t).second ||
              beds.count(t.territory))
            fail(t.territory, "Duplicate input compartment.");
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
        if (compartments.count(id))
          {
            const auto &t = compartments.at(id);
            if (t.arterial != a)
              fail(id,
                   "Input compartment disagrees with its arterial territory.");
            requests[t.portal % size].push_back(t.portal);
            requests[t.venous % size].push_back(t.venous);
          }
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
        if (compartments.count(id))
          {
            const auto &compartment = compartments.at(id);
            const bool  p_alive     = endpoints.count(compartment.portal),
                       v_alive      = endpoints.count(compartment.venous);
            if (a_alive != p_alive || a_alive != v_alive)
              {
                fail(
                  id,
                  "All three endpoints of an input microvascular compartment must be extended together.");
                continue;
              }
            if (a_alive)
              for (const auto &[terminal, system] :
                   {std::make_pair(compartment.portal, System::portal),
                    std::make_pair(compartment.venous, System::venous)})
                {
                  const auto &e = endpoints.at(terminal);
                  if (e.system != system || e.territory != id ||
                      e.demand != originals.at(id).demand)
                    fail(
                      id,
                      "An untouched input compartment terminal changed its metadata.");
                }
          }
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
        routes[a.owner].territories.push_back(retained);
        if (beds.count(id))
          routes[a.owner].pairs.push_back(beds.at(id));
        if (compartments.count(id))
          routes[a.owner].triplets.push_back(compartments.at(id));
      }
    if (!valid())
      return result;
    for (const auto &[source, metadata] :
         Utilities::MPI::some_to_some(comm, routes))
      {
        result.territories.insert(result.territories.end(),
                                  metadata.territories.begin(),
                                  metadata.territories.end());
        result.pairs.insert(result.pairs.end(),
                            metadata.pairs.begin(),
                            metadata.pairs.end());
        result.triplets.insert(result.triplets.end(),
                               metadata.triplets.begin(),
                               metadata.triplets.end());
      }
    return result;
  }
} // namespace ImmersX::Vascularization::internal
