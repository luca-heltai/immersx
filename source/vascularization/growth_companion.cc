// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <boost/geometry/algorithms/comparable_distance.hpp>
#include <boost/geometry/algorithms/equals.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/distribution.h>

#include <memory>

#include "distribution_internal.h"
#include "growth_batch.h"
#include "growth_start.h"
#include "growth_system.h"

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
      Rank     owner      = 0;
      double   demand     = 0.;
      double   separation = 0.;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &id &position &owner &demand &separation;
      }
    };
    struct RootQuery
    {
      Point<3> position;
      Id       rejected = invalid_id;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &position &rejected;
      }
    };
    struct TerritoryUpdates
    {
      std::vector<Id> claimed, retired;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &claimed &retired;
      }
    };
    double
    lower_distance(const BoundingBox<3> &box, const Point<3> &point)
    {
      const double distance = boost::geometry::comparable_distance(point, box);
      return std::max(
        0.,
        std::nextafter(std::min(distance, std::numeric_limits<double>::max()) *
                         (1. - 64. * std::numeric_limits<double>::epsilon()),
                       -std::numeric_limits<double>::infinity()) -
          64. * std::numeric_limits<double>::denorm_min());
    }
  } // namespace

  CompanionPort
  reserve_companion_port(const dealii::Triangulation<3> &bg,
                         const dealii::Mapping<3>       &mapping,
                         const NetworkData              &network,
                         const Parameters               &p,
                         MPI_Comm                        comm,
                         System                          system)
  {
    using namespace dealii;
    const auto &companion = companion_parameters(p, system);
    const Rank  rank      = Utilities::MPI::this_mpi_process(comm),
               size       = Utilities::MPI::n_mpi_processes(comm);
    std::vector<double> controls{p.viscosity,
                                 p.minimum_radius,
                                 companion.seed_radius,
                                 companion.murray_exponent,
                                 companion.terminal_separation,
                                 companion.microvascular_resistance};
    for (const auto &point :
         {companion.seed_position, companion.seed_inward_port_normal})
      for (unsigned int d = 0; d < 3; ++d)
        controls.push_back(point[d]);
    const auto local = std::make_pair(controls, companion.selected_terminals);
    bool       matching = true;
    for (const auto &other : Utilities::MPI::all_gather(comm, local))
      matching = matching && other == local;
    AssertThrow(
      matching,
      ExcMessage(
        "Companion startup requires matching outlet controls and input selections."));
    CompanionPort result;
    result.network = network;
    if (!companion.selected_terminals.empty())
      return result;
    std::unique_ptr<BackgroundGeometry> local_geometry;
    if (!shared_geometry)
      local_geometry = std::make_unique<BackgroundGeometry>(bg, mapping);
    auto &geometry = shared_geometry ? *shared_geometry : *local_geometry;
    geometry.configure_length_scale(p.length_scale, comm);
    const auto  &scale = *geometry.length_scale;
    const double minimum_radius =
      scale.physical_length(p.minimum_radius, scale.minimum());
    const double halo = scale.physical_length(p.clearance, scale.maximum());
    unsigned int available = 0;
    for (const auto &[id, t] : network.terminals)
      if (t.owner == rank && t.system == system &&
          geometry.containing(network.nodes.at(t.node).position))
        ++available;
    if (Utilities::MPI::sum(available, comm) > 0)
      return result;
    const auto fail = [&](const std::string &message) {
      result.validation.issues.push_back(
        {"companion outlet", invalid_id, message});
    };
    if (!(companion.seed_radius > 0.) || !(p.viscosity > 0.) ||
        companion.seed_radius < p.minimum_radius)
      {
        fail(
          "A new companion outlet requires an explicit admissible positive radius and dynamic viscosity.");
        return result;
      }
    const auto normal =
      normalized_port_normal(companion.seed_inward_port_normal);
    const bool containing = static_cast<bool>(
      geometry.containing(companion.seed_position,
                          normal.norm_square() > 0. ? 1.e-12 : 0.));
    const Rank source = Utilities::MPI::min(containing ? rank : size, comm);
    if (source == size)
      {
        fail("Companion outlet position is outside the background.");
        return result;
      }
    if (normal.norm_square() > 0. &&
        Utilities::MPI::max(static_cast<unsigned int>(geometry.planar_section(
                              companion.seed_position, normal)),
                            comm) == 0)
      {
        fail(
          "The declared companion port does not match a planar background boundary section.");
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
      result.network.nodes[result.port] = {companion.seed_position,
                                           rank,
                                           system == System::portal ?
                                             Port::portal_inlet :
                                             Port::venous_outlet,
                                           Provenance::generated,
                                           normal};
    result.network = redistribute_with_owners(
      bg, mapping, result.network, halo, comm, {}, &geometry, geometry.counts);
    return result;
  }

  AttachmentStartup
  start_companion(const dealii::Triangulation<3> &bg,
                  const dealii::Mapping<3>       &mapping,
                  const NetworkData              &network,
                  const NetworkData              &original,
                  const Parameters               &p,
                  const std::vector<Territory>   &owned_territories,
                  Id                              reserved_port,
                  MPI_Comm                        comm,
                  System                          system)
  {
    using namespace dealii;
    const auto &companion = companion_parameters(p, system);
    const Rank  rank      = Utilities::MPI::this_mpi_process(comm),
               size       = Utilities::MPI::n_mpi_processes(comm);
    AttachmentStartup result;
    const auto        fail = [&](Id id, const std::string &message) {
      result.accepted.validation.issues.push_back(
        {"companion startup", id, message});
    };
    const auto valid = [&]() {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(
                              result.accepted.validation.valid()),
                            comm) == 1;
      if (!all && result.accepted.validation.valid())
        fail(invalid_id, "Companion startup failed on another process.");
      if (!all)
        result.accepted.prepared = false;
      return all;
    };
    auto snapshot = redistribute(bg, mapping, network, p.clearance, comm);
    BackgroundGeometry geometry(bg, mapping);
    std::set<Id>       requested(companion.selected_terminals.begin(),
                           companion.selected_terminals.end());
    if (requested.size() != companion.selected_terminals.size())
      fail(invalid_id, "A companion attachment may be selected only once.");
    Id found = 0;
    for (const auto &[id, t] : original.terminals)
      if (t.owner == rank && requested.count(id))
        {
          ++found;
          if (t.system != system)
            fail(
              id,
              "Selected companion terminals must belong to the input companion network.");
        }
    if (Utilities::MPI::sum(found, comm) != requested.size())
      fail(invalid_id,
           "Selected companion terminal id is absent from the input registry.");
    std::map<Id, std::vector<Id>> incoming, outgoing;
    for (const auto &[id, v] : snapshot.vessels)
      {
        incoming[v.nodes[1]].push_back(id);
        outgoing[v.nodes[0]].push_back(id);
      }
    struct Pending
    {
      Id start = invalid_id, input_terminal = invalid_id;
      Id node = invalid_id, knee = invalid_id, vessel = invalid_id,
         continuation = invalid_id, terminal = invalid_id;
      double                seed_length = 0.;
      std::vector<Point<3>> seed_directions;
      Vessel                properties;
      Point<3>              position, normal;
      std::set<Rank>        queried_owners, exhausted_owners;
      Id                    rejected  = invalid_id;
      Id                    current   = invalid_id;
      unsigned int          direction = 0;
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
            root.properties.system = system;
            root.properties.radii  = {
              {companion.seed_radius, companion.seed_radius}};
            root.properties.viscosity = p.viscosity;
            roots.push_back(root);
          }
      }
    else
      for (const auto &[id, t] : snapshot.terminals)
        if (t.owner == rank && t.system == system &&
            (requested.empty() || requested.count(id)))
          {
            const auto &node = snapshot.nodes.at(t.node);
            const bool  interior =
              static_cast<bool>(geometry.containing(node.position));
            if (requested.count(id) && !interior)
              fail(
                id,
                "An explicitly selected exterior companion terminal cannot be extended by interior vessels.");
            if (!interior)
              continue;
            if (incoming[t.node].size() != 1 || !outgoing[t.node].empty() ||
                node.port != Port::none || node.owner != rank)
              {
                fail(
                  id,
                  "A companion attachment must be a native non-port outward leaf.");
                continue;
              }
            Pending root;
            root.start          = t.node;
            root.input_terminal = id;
            root.position       = node.position;
            root.properties     = snapshot.vessels.at(incoming.at(t.node)[0]);
            if (root.properties.radii[1] < minimum_radius)
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
        "Common territory count must cover all real companion outlets or selected attachments.");
    // IDs are reserved once per startup, independently of geometric retries.
    const auto reserve = [&](const auto &data, Id stride) {
      Id next = 0;
      if (!data.empty())
        {
          if (data.rbegin()->first >= invalid_id - 1)
            fail(invalid_id, "Vascular identifier space exhausted.");
          else
            next = data.rbegin()->first + 1;
        }
      const Id base   = Utilities::MPI::max(next, comm);
      const Id count  = stride * roots.size();
      Id       prefix = 0;
      MPI_Exscan(&count, &prefix, 1, MPI_UINT64_T, MPI_SUM, comm);
      if (rank == 0)
        prefix = 0;
      if (total_roots > (invalid_id - base) / stride)
        fail(invalid_id, "Vascular identifier space exhausted.");
      return base + prefix;
    };
    Id next_node     = reserve(snapshot.nodes, 2),
       next_vessel   = reserve(snapshot.vessels, 2),
       next_terminal = reserve(snapshot.terminals, 1);
    std::map<Id, Pending> pending;
    for (auto &root : roots)
      {
        root.knee         = next_node++;
        root.node         = next_node++;
        root.vessel       = next_vessel++;
        root.continuation = next_vessel++;
        root.terminal     = next_terminal++;
        pending.emplace(root.vessel, std::move(root));
      }
    VenousTerritorySearch territories(owned_territories, rank, geometry.counts);
    for (const auto &issue : territories.validation().issues)
      fail(issue.id, issue.message);
    if (!valid())
      return result;
    std::map<Id, Point<3>> tissue_queries;
    for (const auto &point : owned_territories)
      tissue_queries.emplace(point.id, point.position);
    const auto tissue_cells = geometry.locate_points(tissue_queries, comm);
    std::map<Id, double> separations;
    for (const auto &[id, position] : tissue_queries)
      {
        if (t.id == invalid_id || t.owner != rank || !std::isfinite(t.demand) ||
            !(t.demand > 0.) ||
            !available
               .emplace(t.id, SharedPoint{t.id, t.position, rank, t.demand})
               .second)
          fail(
            t.id,
            "Companion startup requires unique native common territories with positive finite demand.");
        points.emplace_back(t.position, t.id);
      }
    if (!valid())
      return result;
    std::vector<BoundingBox<3>> local_cover;
    if (const auto cover = territories.cover())
      local_cover.push_back(*cover);
    const auto           covers = Utilities::MPI::all_gather(comm, local_cover);
    std::set<Id>         mutable_owned;
    std::map<Id, double> fixed_roots;
    while (Utilities::MPI::sum(static_cast<Id>(pending.size()), comm) > 0)
      {
        const auto counts =
          Utilities::MPI::all_gather(comm, territories.size());
        using Queries = std::map<Rank, std::map<Id, RootQuery>>;
        std::map<Id, SharedPoint> nearest;
        const auto                exchange = [&](const Queries &queries) {
          std::map<Rank, std::map<Id, SharedPoint>> replies;
          for (const auto &[source, entries] :
               Utilities::MPI::some_to_some(comm, queries))
            for (const auto &[id, query] : entries)
              if (const auto point =
                    territories.query(id, query.position, query.rejected))
                replies[source].emplace(id,
                                        SharedPoint{point->id,
                                                    point->position,
                                                    point->owner,
                                                    point->demand,
                                                    separations.at(point->id)});
          const auto received = Utilities::MPI::some_to_some(comm, replies);
          for (const auto &[owner, entries] : queries)
            for (const auto &[id, query] : entries)
              {
                auto &root = pending.at(id);
                root.queried_owners.insert(owner);
                const auto part = received.find(owner);
                if (part == received.end() || !part->second.count(id))
                  root.exhausted_owners.insert(owner);
              }
          for (const auto &[source, entries] : received)
            for (const auto &[id, point] : entries)
              {
                const double distance =
                  pending.at(id).position.distance_square(point.position);
                const auto old = nearest.find(id);
                if (old == nearest.end() ||
                    distance < pending.at(id).position.distance_square(
                                 old->second.position) ||
                    (distance == pending.at(id).position.distance_square(
                                   old->second.position) &&
                     point.id < old->second.id))
                  nearest[id] = point;
              }
        };
        // First obtain a finite nearest bound from one nonempty owner box.
        // Only boxes that can improve it receive the second round.
        Queries            first;
        std::map<Id, Rank> first_owner;
        for (const auto &[id, root] : pending)
          {
            Rank   owner    = size;
            double distance = std::numeric_limits<double>::infinity();
            for (Rank candidate = 0; candidate < size; ++candidate)
              if (counts[candidate] > 0 &&
                  !root.exhausted_owners.count(candidate))
                for (const auto &box : covers[candidate])
                  {
                    const double bound = lower_distance(box, root.position);
                    if (owner == size || bound < distance)
                      {
                        owner    = candidate;
                        distance = bound;
                      }
                  }
            if (owner < size)
              {
                first[owner].emplace(id,
                                     RootQuery{root.position, root.rejected});
                first_owner.emplace(id, owner);
              }
          }
        exchange(first);
        Queries additional;
        for (const auto &[id, root] : pending)
          {
            const auto   best = nearest.find(id);
            const double distance =
              best == nearest.end() ?
                std::numeric_limits<double>::infinity() :
                root.position.distance_square(best->second.position);
            for (Rank owner = 0; owner < size; ++owner)
              if (counts[owner] > 0 && !root.exhausted_owners.count(owner) &&
                  (!first_owner.count(id) || first_owner.at(id) != owner))
                for (const auto &box : covers[owner])
                  if (lower_distance(box, root.position) <= distance)
                    additional[owner].emplace(id,
                                              RootQuery{root.position,
                                                        root.rejected});
          }
        exchange(additional);
        for (auto &[id, root] : pending)
          root.rejected = invalid_id;
        for (const auto &[id, root] : pending)
          if (!nearest.count(id))
            fail(
              root.input_terminal,
              "No admissible connection from the companion outlet or attachment to a common tissue territory was found.");
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
                                      companion.terminal_separation);
            const auto end = root.position + directions.at(root.direction++);
            if (root.direction == directions.size())
              root.excluded.insert(point.id);
            if (root.position.distance(end) <= p.minimum_length)
              continue;
            BatchProposal proposal;
            proposal.id = id;
            proposal.changed.push_back({root.vessel,
                                        {{root.start, root.knee}},
                                        {{root.position, knee}},
                                        root.properties.radii[1]});
            proposal.changed.push_back({root.continuation,
                                        {{root.knee, root.node}},
                                        {{knee, end}},
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
        auto                             trial = snapshot;
        std::vector<Id>                  accepted;
        std::map<Rank, TerritoryUpdates> updates;
        for (const auto &proposal : proposals)
          if (selection.local.at(proposal.id))
            {
              const auto &root           = pending.at(proposal.id);
              const auto &point          = nearest.at(proposal.id);
              const auto &end            = proposal.changed.back().points[1];
              trial.nodes[root.knee]     = {proposal.changed[0].points[1],
                                            rank,
                                            Port::none,
                                            Provenance::generated};
              trial.nodes[root.node]     = {end,
                                            rank,
                                            Port::none,
                                            Provenance::generated};
              auto vessel                = root.properties;
              vessel.nodes               = {{root.start, root.knee}};
              vessel.radii               = {{vessel.radii[1], vessel.radii[1]}};
              vessel.owner               = rank;
              vessel.provenance          = Provenance::generated;
              trial.vessels[root.vessel] = vessel;
              auto continuation          = vessel;
              continuation.nodes         = {{root.knee, root.node}};
              trial.vessels[root.continuation] = continuation;
              if (root.input_terminal != invalid_id)
                trial.terminals.erase(root.input_terminal);
              trial.terminals[root.terminal] = {
                root.node, point.id, system, rank, point.demand};
              mutable_owned.insert(root.vessel);
              mutable_owned.insert(root.continuation);
              fixed_roots[root.vessel] = vessel.radii[0];
              accepted.push_back(root.vessel);
              updates[point.owner].claimed.push_back(point.id);
              for (Rank owner : root.queried_owners)
                updates[owner].retired.push_back(root.vessel);
            }
        if (Utilities::MPI::sum(static_cast<Id>(accepted.size()), comm) == 0)
          continue;
        auto committed = prepare_growth_start(bg,
                                              mapping,
                                              trial,
                                              mutable_owned,
                                              fixed_roots,
                                              halo,
                                              comm,
                                              &geometry);
        if (!committed.prepared)
          {
            result.accepted = std::move(committed);
            return result;
          }
        for (const auto &[source, update] :
             Utilities::MPI::some_to_some(comm, updates))
          {
            for (Id id : update.claimed)
              if (!territories.claim(id))
                fail(id, "A common territory can be claimed only once.");
            for (Id root : update.retired)
              territories.forget(root);
          }
        if (!valid())
          return result;
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
