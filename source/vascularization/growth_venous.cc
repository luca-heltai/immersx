// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <boost/geometry/algorithms/equals.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/distribution.h>

#include "growth_batch.h"
#include "growth_start.h"

namespace ImmersX::Vascularization::internal
{
  namespace
  {
    using namespace dealii;
    using Rank = types::subdomain_id;
    struct SharedPoint
    {
      Id       id = invalid_id;
      Point<3> position;
      Rank     owner  = 0;
      double   demand = 0.;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &id &position &owner &demand;
      }
    };
    struct RootQuery
    {
      Point<3>     position;
      std::set<Id> excluded;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &position &excluded;
      }
    };
  } // namespace

  VenousOutlet
  reserve_venous_outlet(const dealii::Triangulation<3> &bg,
                        const dealii::Mapping<3>       &mapping,
                        const NetworkData              &network,
                        const Parameters               &p,
                        MPI_Comm                        comm)
  {
    using namespace dealii;
    const Rank rank = Utilities::MPI::this_mpi_process(comm),
               size = Utilities::MPI::n_mpi_processes(comm);
    std::vector<double> controls{p.viscosity,
                                 p.minimum_radius,
                                 p.venous.seed_radius,
                                 p.venous.murray_exponent,
                                 p.venous.terminal_separation,
                                 p.venous.microvascular_resistance};
    for (const auto &point :
         {p.venous.seed_position, p.venous.seed_inward_port_normal})
      for (unsigned int d = 0; d < 3; ++d)
        controls.push_back(point[d]);
    const auto local    = std::make_pair(controls, p.venous.selected_terminals);
    bool       matching = true;
    for (const auto &other : Utilities::MPI::all_gather(comm, local))
      matching = matching && other == local;
    AssertThrow(
      matching,
      ExcMessage(
        "Venous startup requires matching outlet controls and input selections."));
    VenousOutlet result;
    result.network = network;
    if (!p.venous.selected_terminals.empty())
      return result;
    BackgroundGeometry geometry(bg, mapping);
    unsigned int       available = 0;
    for (const auto &[id, t] : network.terminals)
      if (t.owner == rank && t.system == System::venous &&
          geometry.containing(network.nodes.at(t.node).position))
        ++available;
    if (Utilities::MPI::sum(available, comm) > 0)
      return result;
    const auto fail = [&](const std::string &message) {
      result.validation.issues.push_back(
        {"venous outlet", invalid_id, message});
    };
    if (!(p.venous.seed_radius > 0.) || !(p.viscosity > 0.) ||
        p.venous.seed_radius < p.minimum_radius)
      {
        fail(
          "A new venous outlet requires an explicit admissible positive radius and dynamic viscosity.");
        return result;
      }
    const auto normal =
      normalized_port_normal(p.venous.seed_inward_port_normal);
    const bool containing = static_cast<bool>(
      geometry.containing(p.venous.seed_position,
                          normal.norm_square() > 0. ? 1.e-12 : 0.));
    const Rank source = Utilities::MPI::min(containing ? rank : size, comm);
    if (source == size)
      {
        fail("Venous outlet position is outside the background.");
        return result;
      }
    if (normal.norm_square() > 0. &&
        Utilities::MPI::max(static_cast<unsigned int>(geometry.planar_section(
                              p.venous.seed_position, normal)),
                            comm) == 0)
      {
        fail(
          "The declared venous port does not match a planar background boundary section.");
        return result;
      }
    Id   next  = 0;
    bool valid = true;
    if (!network.nodes.empty())
      {
        valid = network.nodes.rbegin()->first < invalid_id - 1;
        if (valid)
          next = network.nodes.rbegin()->first + 1;
      }
    if (Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 0)
      {
        fail("Vascular node identifier space exhausted.");
        return result;
      }
    result.port = Utilities::MPI::max(next, comm);
    if (rank == source)
      result.network.nodes[result.port] = {p.venous.seed_position,
                                           rank,
                                           Port::venous_outlet,
                                           Provenance::generated,
                                           normal};
    result.network =
      redistribute(bg, mapping, result.network, p.clearance, comm);
    return result;
  }

  AttachmentStartup
  start_venous(const dealii::Triangulation<3> &bg,
               const dealii::Mapping<3>       &mapping,
               const NetworkData              &network,
               const NetworkData              &original,
               const Parameters               &p,
               const std::vector<Territory>   &owned_territories,
               Id                              reserved_port,
               MPI_Comm                        comm)
  {
    using namespace dealii;
    const Rank rank = Utilities::MPI::this_mpi_process(comm),
               size = Utilities::MPI::n_mpi_processes(comm);
    AttachmentStartup result;
    const auto        fail = [&](Id id, const std::string &message) {
      result.accepted.validation.issues.push_back(
        {"venous startup", id, message});
    };
    const auto valid = [&]() {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(
                              result.accepted.validation.valid()),
                            comm) == 1;
      if (!all && result.accepted.validation.valid())
        fail(invalid_id, "Venous startup failed on another process.");
      if (!all)
        result.accepted.committed = false;
      return all;
    };
    auto snapshot = redistribute(bg, mapping, network, p.clearance, comm);
    BackgroundGeometry geometry(bg, mapping);
    std::set<Id>       requested(p.venous.selected_terminals.begin(),
                           p.venous.selected_terminals.end());
    if (requested.size() != p.venous.selected_terminals.size())
      fail(invalid_id, "A venous attachment may be selected only once.");
    Id found = 0;
    for (const auto &[id, t] : original.terminals)
      if (t.owner == rank && requested.count(id))
        {
          ++found;
          if (t.system != System::venous)
            fail(
              id,
              "Selected venous terminals must belong to the input venous network.");
        }
    if (Utilities::MPI::sum(found, comm) != requested.size())
      fail(invalid_id,
           "Selected venous terminal id is absent from the input registry.");
    std::map<Id, std::vector<Id>> incoming, outgoing;
    for (const auto &[id, v] : snapshot.vessels)
      {
        incoming[v.nodes[1]].push_back(id);
        outgoing[v.nodes[0]].push_back(id);
      }
    struct Pending
    {
      Id       start = invalid_id, input_terminal = invalid_id;
      Id       node = invalid_id, vessel = invalid_id, terminal = invalid_id;
      Vessel   properties;
      Point<3> position, normal;
      std::set<Id> excluded;
      Id           current   = invalid_id;
      unsigned int direction = 0;
    };
    std::vector<Pending> roots;
    if (reserved_port != invalid_id)
      {
        if (snapshot.nodes.count(reserved_port) &&
            snapshot.nodes.at(reserved_port).owner == rank)
          {
            const auto &node = snapshot.nodes.at(reserved_port);
            Pending     root;
            root.start             = reserved_port;
            root.position          = node.position;
            root.normal            = node.inward_port_normal;
            root.properties.system = System::venous;
            root.properties.radii  = {
              {p.venous.seed_radius, p.venous.seed_radius}};
            root.properties.viscosity = p.viscosity;
            roots.push_back(root);
          }
      }
    else
      for (const auto &[id, t] : snapshot.terminals)
        if (t.owner == rank && t.system == System::venous &&
            (requested.empty() || requested.count(id)))
          {
            const auto &node = snapshot.nodes.at(t.node);
            const bool  interior =
              static_cast<bool>(geometry.containing(node.position));
            if (requested.count(id) && !interior)
              fail(
                id,
                "An explicitly selected exterior venous terminal cannot be extended by interior vessels.");
            if (!interior)
              continue;
            if (incoming[t.node].size() != 1 || !outgoing[t.node].empty() ||
                node.port != Port::none || node.owner != rank)
              {
                fail(
                  id,
                  "A venous attachment must be a native non-port outward leaf.");
                continue;
              }
            Pending root;
            root.start          = t.node;
            root.input_terminal = id;
            root.position       = node.position;
            root.properties     = snapshot.vessels.at(incoming.at(t.node)[0]);
            if (root.properties.radii[1] < p.minimum_radius)
              fail(
                id,
                "Fixed attachment radius is below the minimum admissible new radius.");
            roots.push_back(root);
          }
    const Id total_roots =
      Utilities::MPI::sum(static_cast<Id>(roots.size()), comm);
    const Id total_territories =
      Utilities::MPI::sum(static_cast<Id>(owned_territories.size()), comm);
    if (total_roots == 0 || total_roots > total_territories)
      fail(
        invalid_id,
        "Common territory count must cover all real venous outlets or selected attachments.");
    // IDs are reserved once per startup, independently of geometric retries.
    const auto reserve = [&](const auto &data) {
      Id next = 0;
      if (!data.empty())
        {
          if (data.rbegin()->first >= invalid_id - 1)
            fail(invalid_id, "Vascular identifier space exhausted.");
          else
            next = data.rbegin()->first + 1;
        }
      const Id base   = Utilities::MPI::max(next, comm);
      const Id count  = roots.size();
      Id       prefix = 0;
      MPI_Exscan(&count, &prefix, 1, MPI_UINT64_T, MPI_SUM, comm);
      if (rank == 0)
        prefix = 0;
      if (total_roots > invalid_id - base)
        fail(invalid_id, "Vascular identifier space exhausted.");
      return base + prefix;
    };
    Id next_node     = reserve(snapshot.nodes),
       next_vessel   = reserve(snapshot.vessels),
       next_terminal = reserve(snapshot.terminals);
    std::map<Id, Pending> pending;
    for (auto &root : roots)
      {
        root.node     = next_node++;
        root.vessel   = next_vessel++;
        root.terminal = next_terminal++;
        pending.emplace(root.vessel, std::move(root));
      }
    std::map<Id, SharedPoint>            available;
    std::vector<std::pair<Point<3>, Id>> points;
    for (const auto &t : owned_territories)
      {
        if (t.id == invalid_id || t.owner != rank || !std::isfinite(t.demand) ||
            !(t.demand > 0.) ||
            !available
               .emplace(t.id, SharedPoint{t.id, t.position, rank, t.demand})
               .second)
          fail(
            t.id,
            "Venous startup requires unique native common territories with positive finite demand.");
        points.emplace_back(t.position, t.id);
      }
    if (!valid())
      return result;
    auto                 tree = pack_rtree(points);
    std::set<Id>         mutable_owned;
    std::map<Id, double> fixed_roots;
    while (Utilities::MPI::sum(static_cast<Id>(pending.size()), comm) > 0)
      {
        // Each producer returns one nearest available point per query. The
        // exchanges carry root queries and candidates, never a territory graph.
        const auto counts = Utilities::MPI::all_gather(comm, available.size());
        std::map<Rank, std::map<Id, RootQuery>> queries;
        for (const auto &[id, root] : pending)
          for (Rank target = 0; target < size; ++target)
            if (counts[target] > 0)
              queries[target].emplace(id,
                                      RootQuery{root.position, root.excluded});
        std::map<Rank, std::map<Id, SharedPoint>> replies;
        for (const auto &[source, entries] :
             Utilities::MPI::some_to_some(comm, queries))
          for (const auto &[id, query] : entries)
            {
              const SharedPoint *best = nullptr;
              double     distance     = std::numeric_limits<double>::infinity();
              const auto count =
                std::min(tree.size(), query.excluded.size() + 1);
              for (const auto &[point, territory] :
                   tree | bgi::adaptors::queried(
                            bgi::nearest(query.position, count)))
                if (!query.excluded.count(territory))
                  {
                    const double d = point.distance_square(query.position);
                    if (!best || d < distance ||
                        (d == distance && territory < best->id))
                      {
                        best     = &available.at(territory);
                        distance = d;
                      }
                  }
              if (best)
                replies[source].emplace(id, *best);
            }
        std::map<Id, SharedPoint> nearest;
        for (const auto &[source, entries] :
             Utilities::MPI::some_to_some(comm, replies))
          for (const auto &[id, point] : entries)
            {
              const double distance =
                pending.at(id).position.distance_square(point.position);
              const auto it = nearest.find(id);
              if (it == nearest.end() ||
                  distance < pending.at(id).position.distance_square(
                               it->second.position) ||
                  (distance == pending.at(id).position.distance_square(
                                 it->second.position) &&
                   point.id < it->second.id))
                nearest[id] = point;
            }
        for (const auto &[id, root] : pending)
          if (!nearest.count(id))
            fail(
              root.input_terminal,
              "No admissible connection from the venous outlet or attachment to a common tissue territory was found.");
        if (!valid())
          return result;
        // Temporary territory grants serialize only competing requests for the
        // same physical territory; disjoint outlets still propose together.
        std::map<Rank, std::map<Id, std::vector<Id>>> requests;
        for (const auto &[id, point] : nearest)
          requests[point.owner][point.id].push_back(id);
        std::map<Id, std::pair<Id, Rank>> winners;
        for (const auto &[source, entries] :
             Utilities::MPI::some_to_some(comm, requests))
          for (const auto &[territory, ids] : entries)
            for (Id id : ids)
              {
                const auto candidate = std::make_pair(id, source);
                if (!winners.count(territory) ||
                    candidate < winners.at(territory))
                  winners[territory] = candidate;
              }
        std::map<Rank, std::vector<Id>> grants;
        for (const auto &[territory, winner] : winners)
          grants[winner.second].push_back(winner.first);
        std::set<Id> granted;
        for (const auto &[source, ids] :
             Utilities::MPI::some_to_some(comm, grants))
          granted.insert(ids.begin(), ids.end());
        std::vector<BatchProposal> proposals;
        for (Id id : granted)
          {
            auto       &root  = pending.at(id);
            const auto &point = nearest.at(id);
            if (root.current != point.id)
              {
                root.current   = point.id;
                root.direction = 0;
              }
            Territory  territory{point.id,
                                point.position,
                                point.owner,
                                point.demand};
            const auto directions =
              initial_root_directions(root.position,
                                      {},
                                      {},
                                      {},
                                      {},
                                      &territory,
                                      p.venous.terminal_separation);
            const auto end = root.position + directions.at(root.direction++);
            if (root.direction == directions.size())
              root.excluded.insert(point.id);
            if (root.position.distance(end) <= p.minimum_length)
              continue;
            BatchProposal proposal;
            proposal.id = id;
            proposal.changed.push_back({root.vessel,
                                        {{root.start, root.node}},
                                        {{root.position, end}},
                                        root.properties.radii[1]});
            if (root.normal.norm_square() > 0.)
              {
                proposal.normals[root.start]     = root.normal;
                proposal.port_owners[root.start] = rank;
              }
            proposals.push_back(std::move(proposal));
          }
        const auto selection = select_growth_batch(
          geometry, snapshot, proposals, p, comm, [&](Id id) {
            const auto &v = snapshot.vessels.at(id);
            return std::max(v.radii[0], v.radii[1]);
          });
        auto                            trial = snapshot;
        std::vector<Id>                 accepted;
        std::map<Rank, std::vector<Id>> claims;
        for (const auto &proposal : proposals)
          if (selection.local.at(proposal.id))
            {
              const auto &root           = pending.at(proposal.id);
              const auto &point          = nearest.at(proposal.id);
              const auto &end            = proposal.changed[0].points[1];
              trial.nodes[root.node]     = {end,
                                            rank,
                                            Port::none,
                                            Provenance::generated};
              auto vessel                = root.properties;
              vessel.nodes               = {{root.start, root.node}};
              vessel.radii               = {{vessel.radii[1], vessel.radii[1]}};
              vessel.owner               = rank;
              vessel.provenance          = Provenance::generated;
              trial.vessels[root.vessel] = vessel;
              if (root.input_terminal != invalid_id)
                trial.terminals.erase(root.input_terminal);
              trial.terminals[root.terminal] = {
                root.node, point.id, System::venous, rank, point.demand};
              mutable_owned.insert(root.vessel);
              fixed_roots[root.vessel] = vessel.radii[0];
              accepted.push_back(root.vessel);
              claims[point.owner].push_back(point.id);
            }
        if (Utilities::MPI::sum(static_cast<Id>(accepted.size()), comm) == 0)
          continue;
        auto committed = commit_growth_splits(
          bg, mapping, trial, mutable_owned, fixed_roots, {}, p, comm);
        if (!committed.committed)
          {
            result.accepted = std::move(committed);
            return result;
          }
        for (const auto &[source, ids] :
             Utilities::MPI::some_to_some(comm, claims))
          for (Id id : ids)
            {
              const auto &point = available.at(id);
              tree.remove(std::make_pair(point.position, id));
              available.erase(id);
            }
        for (Id id : accepted)
          {
            if (pending.at(id).input_terminal != invalid_id)
              result.attachment_roots.emplace(pending.at(id).input_terminal,
                                              id);
            pending.erase(id);
          }
        snapshot        = committed.network;
        mutable_owned   = committed.mutable_owned;
        fixed_roots     = committed.fixed_owned_roots;
        result.accepted = std::move(committed);
      }
    return result;
  }
} // namespace ImmersX::Vascularization::internal
