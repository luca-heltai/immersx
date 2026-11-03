// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_selection.h"

#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

namespace ImmersX::Vascularization::internal
{
  BatchSelection
  select_growth_batch(BackgroundGeometry                      &geometry,
                      const GrowthExecution::AcceptedSnapshot &accepted,
                      const std::vector<BatchProposal>        &proposals,
                      const Parameters                        &parameters,
                      MPI_Comm                                 comm)
  {
    using namespace dealii;
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(accepted.current()),
                          comm) == 1,
      ExcMessage(
        "A preliminary vote requires current accepted views on every rank."));
    const auto  rank    = Utilities::MPI::this_mpi_process(comm);
    const auto &network = accepted.network();
    using Requests =
      std::pair<std::map<Id, unsigned int>, std::map<Id, unsigned int>>;
    Requests local;
    bool     valid = true;
    for (const auto &proposal : proposals)
      {
        for (const auto &[id, owner] : proposal.owners)
          {
            const auto [entry, inserted] = local.first.emplace(id, owner);
            valid = valid && (inserted || entry->second == owner);
          }
        for (const auto &[id, owner] : proposal.port_owners)
          {
            const auto [entry, inserted] = local.second.emplace(id, owner);
            valid = valid && (inserted || entry->second == owner);
          }
      }
    NetworkData selected;
    for (const auto &requests : Utilities::MPI::all_gather(comm, local))
      {
        for (const auto &[id, owner] : requests.first)
          if (owner == rank)
            {
              const auto vessel = network.vessels.find(id);
              // Missing/stale writes are rejected by the common vote. Only
              // existing native rows can establish old endpoint provenance.
              if (vessel == network.vessels.end() ||
                  vessel->second.owner != rank)
                continue;
              count_work(geometry.counts,
                         GrowthOperation::physical_copy_attempts,
                         1 + vessel->second.nodes.size());
              selected.vessels.emplace(*vessel);
              for (Id node : vessel->second.nodes)
                selected.nodes.emplace(node, network.nodes.at(node));
            }
        for (const auto &[id, owner] : requests.second)
          if (owner == rank)
            {
              const auto node = network.nodes.find(id);
              if (node != network.nodes.end() && node->second.owner == rank)
                {
                  count_work(geometry.counts,
                             GrowthOperation::physical_copy_attempts);
                  selected.nodes.emplace(*node);
                }
            }
      }
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1,
      ExcMessage("A preliminary batch names conflicting native owners."));
    return select_growth_batch(
      geometry, selected, proposals, parameters, comm, {}, false);
  }
} // namespace ImmersX::Vascularization::internal
