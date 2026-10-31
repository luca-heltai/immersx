// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/numerics/rtree.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/distribution.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <set>

#include "distribution_internal.h"
#include "growth_geometry.h"
#include "growth_work.h"
#include "network_serialization.h"

namespace ImmersX::Vascularization
{
  namespace
  {
    using namespace dealii;
    using Rank    = unsigned int;
    namespace bgi = boost::geometry::index;

    void
    require_all(bool local, MPI_Comm comm, const std::string &message)
    {
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(local), comm) ==
                    1,
                  ExcMessage(message));
    }

    bool
    same_node(const Node &a, const Node &b)
    {
      return a.position == b.position && a.owner == b.owner &&
             a.port == b.port && a.provenance == b.provenance &&
             a.inward_port_normal == b.inward_port_normal;
    }

    bool
    merge(NetworkData &to, const NetworkData &from, GrowthWork *work)
    {
      internal::count_copy(work, from);
      bool valid = true;
      for (const auto &[id, n] : from.nodes)
        {
          const auto [it, inserted] = to.nodes.emplace(id, n);
          valid = (inserted || same_node(it->second, n)) && valid;
        }
      for (const auto &[id, v] : from.vessels)
        valid = to.vessels.emplace(id, v).second && valid;
      for (const auto &[id, t] : from.terminals)
        valid = to.terminals.emplace(id, t).second && valid;
      return valid;
    }

    NetworkData
    authoritative_data(const NetworkData &initial,
                       Rank               rank,
                       MPI_Comm           comm,
                       GrowthWork        *work)
    {
      std::map<Rank, std::vector<Id>> requests;
      NetworkData                     result;
      for (const auto &[id, n] : initial.nodes)
        if (n.owner == rank)
          result.nodes.emplace(id, n);
        else
          requests[n.owner].push_back(id);
      const auto incoming = Utilities::MPI::some_to_some(comm, requests);
      std::map<Rank, std::map<Id, Node>> replies;
      bool                               valid = true;
      for (const auto &[source, ids] : incoming)
        for (Id id : ids)
          {
            const auto n = initial.nodes.find(id);
            if (n == initial.nodes.end() || n->second.owner != rank)
              valid = false;
            else
              replies[source].emplace(id, n->second);
          }
      const auto received = Utilities::MPI::some_to_some(comm, replies);
      for (const auto &[owner, nodes] : received)
        for (const auto &[id, n] : nodes)
          valid =
            (n.owner == owner && result.nodes.emplace(id, n).second) && valid;
      valid = valid && result.nodes.size() == initial.nodes.size();
      require_all(
        valid,
        comm,
        "A locally relevant vascular node is absent from its authoritative owner.");
      for (const auto &[id, v] : initial.vessels)
        if (v.owner == rank)
          result.vessels.emplace(id, v);
      for (const auto &[id, t] : initial.terminals)
        if (t.owner == rank)
          result.terminals.emplace(id, t);
      count_work(work,
                 GrowthOperation::ghost_vessels_removed,
                 initial.vessels.size() - result.vessels.size());
      internal::count_copy(work, result);
      for (const auto &[rank, packet] : replies)
        count_work(work,
                   GrowthOperation::physical_copy_attempts,
                   packet.size());
      return result;
    }

    void
    unique_owned_ids(const NetworkData &n,
                     Rank               rank,
                     Rank               size,
                     MPI_Comm           comm,
                     GrowthWork        *work)
    {
      // Hash-sharded id checks replicate neither the registry nor the graph.
      using Key = std::pair<unsigned int, Id>;
      std::map<Rank, std::vector<Key>> send;
      const auto add = [&](const auto &entities, unsigned int kind) {
        for (const auto &[id, e] : entities)
          if (e.owner == rank)
            send[(id % size + kind) % size].emplace_back(kind, id);
      };
      add(n.nodes, 0);
      add(n.vessels, 1);
      add(n.terminals, 2);
      for (const auto &[rank, keys] : send)
        count_work(work, GrowthOperation::metadata_copy_attempts, keys.size());
      const auto    received = Utilities::MPI::some_to_some(comm, send);
      std::set<Key> seen;
      bool          valid = true;
      for (const auto &[source, keys] : received)
        for (const auto &key : keys)
          valid = seen.insert(key).second && valid;
      require_all(
        valid,
        comm,
        "A physical vascular id has more than one authoritative owner.");
    }

    void
    add_vessel(NetworkData       &packet,
               const NetworkData &network,
               Id                 id,
               GrowthWork        *work)
    {
      count_work(work, GrowthOperation::physical_copy_attempts, 3);
      const auto &v = network.vessels.at(id);
      packet.vessels.emplace(id, v);
      for (Id node : v.nodes)
        packet.nodes.emplace(node, network.nodes.at(node));
    }
  } // namespace

  NetworkData
  internal::redistribute_with_owners(
    const dealii::Triangulation<3>              &background,
    const dealii::Mapping<3>                    &mapping,
    const NetworkData                           &initial,
    double                                       halo,
    MPI_Comm                                     communicator,
    const std::function<void(Id, unsigned int)> &notify,
    internal::BackgroundGeometry                *geometry,
    GrowthWork                                  *work)
  {
    using namespace dealii;
    const Rank rank = Utilities::MPI::this_mpi_process(communicator),
               size = Utilities::MPI::n_mpi_processes(communicator);
    int comparison  = MPI_UNEQUAL;
    MPI_Comm_compare(background.get_mpi_communicator(),
                     communicator,
                     &comparison);
    const auto *q  = dynamic_cast<const MappingQ<3> *>(&mapping);
    const auto *fe = dynamic_cast<const MappingFE<3> *>(&mapping);
    require_all(
      (comparison == MPI_IDENT || comparison == MPI_CONGRUENT) &&
        std::isfinite(halo) && halo >= 0. &&
        ((q && q->get_degree() == 1) || (fe && fe->get_degree() == 1)) &&
        internal::validate_counted_data(initial, size, work).valid(),
      communicator,
      "Vascular redistribution requires valid local data, a nonnegative finite halo, a first-order mapping and congruent communicators.");
    unique_owned_ids(initial, rank, size, communicator, work);
    auto source = authoritative_data(initial, rank, communicator, work);
    require_all(
      internal::validate_counted_data(source, size, work).valid(),
      communicator,
      "Authoritative vascular endpoints invalidate an owned vessel or terminal.");
    std::optional<GridTools::Cache<3>>         local_cache;
    std::optional<internal::BackgroundRouting> local_routing;
    if (!geometry)
      {
        count_work(work, GrowthOperation::background_cache_constructions);
        local_cache.emplace(background, mapping);
        local_routing.emplace(*local_cache, communicator, work);
      }
    const auto &cache = geometry ? geometry->cache : *local_cache;
    const auto &routing =
      geometry ? geometry->routing(communicator, work) : *local_routing;
    const auto &boxes         = routing.boxes;
    const auto &covering_tree = routing.covering_tree;
    require_all(
      !covering_tree.empty(),
      communicator,
      "Vascular redistribution requires a nonempty global background.");

    std::vector<Point<3>>     positions;
    std::map<Id, std::size_t> node_indices, vessel_indices;
    for (const auto &[id, n] : source.nodes)
      {
        node_indices[id] = positions.size();
        positions.push_back(n.position);
      }
    for (const auto &[id, v] : source.vessels)
      {
        vessel_indices[id] = positions.size();
        positions.push_back(.5 * source.nodes.at(v.nodes[0]).position +
                            .5 * source.nodes.at(v.nodes[1]).position);
      }
    count_work(work, GrowthOperation::node_localizations, source.nodes.size());
    count_work(work,
               GrowthOperation::vessel_midpoint_localizations,
               source.vessels.size());
    // This is the native ownership kernel used by insert_global_particles.
    // No temporary ParticleHandler/cache or physical-to-floating id is needed.
    const auto matches =
      GridTools::distributed_compute_point_locations(cache, positions, boxes);
    std::map<Rank, std::vector<unsigned int>> located;
    for (std::size_t cell = 0; cell < std::get<0>(matches).size(); ++cell)
      for (std::size_t point = 0; point < std::get<2>(matches)[cell].size();
           ++point)
        located[std::get<4>(matches)[cell][point]].push_back(
          std::get<2>(matches)[cell][point]);
    const auto destinations =
      Utilities::MPI::some_to_some(communicator, located);
    std::vector<Rank> point_owners(positions.size(), size);
    for (const auto &[owner, indices] : destinations)
      for (const auto i : indices)
        point_owners.at(i) = std::min(point_owners.at(i), owner);
    for (auto &[id, n] : source.nodes)
      if (point_owners[node_indices.at(id)] != size)
        n.owner = point_owners[node_indices.at(id)];
    for (auto &[id, v] : source.vessels)
      if (point_owners[vessel_indices.at(id)] != size)
        v.owner = point_owners[vessel_indices.at(id)];
    if (notify)
      for (const auto &[id, v] : source.vessels)
        notify(id, v.owner);
    for (auto &[id, t] : source.terminals)
      t.owner = source.nodes.at(t.node).owner;

    std::map<Rank, NetworkData> send;
    for (const auto &[id, n] : initial.nodes)
      if (n.owner == rank)
        {
          const auto &updated = source.nodes.at(id);
          count_work(work, GrowthOperation::physical_copy_attempts);
          if (updated.owner != rank)
            count_work(work, GrowthOperation::nodes_migrated);
          send[updated.owner].nodes.emplace(id, updated);
        }
    for (const auto &[id, v] : source.vessels)
      {
        if (v.owner != rank)
          count_work(work, GrowthOperation::vessels_migrated);
        add_vessel(send[v.owner], source, id, work);
      }
    for (const auto &[id, t] : source.terminals)
      {
        count_work(work, GrowthOperation::physical_copy_attempts, 2);
        if (t.owner != rank)
          count_work(work, GrowthOperation::terminals_migrated);
        send[t.owner].terminals.emplace(id, t);
        send[t.owner].nodes.emplace(t.node, source.nodes.at(t.node));
      }
    const auto  migrated = Utilities::MPI::some_to_some(communicator, send);
    NetworkData result;
    bool        valid = true;
    for (const auto &[sender, packet] : migrated)
      valid = merge(result, packet, work) && valid;
    require_all(valid,
                communicator,
                "Vascular migration received inconsistent authoritative data.");

    // A geometric halo is independent of mesh adjacency and may reach across
    // multiple partitions. Covering boxes only select possible recipients.
    send.clear();
    for (const auto &[id, v] : result.vessels)
      {
        const auto &a    = result.nodes.at(v.nodes[0]).position,
                   &b    = result.nodes.at(v.nodes[1]).position;
        const double pad = std::max(v.radii[0], v.radii[1]) + halo +
                           64. * std::numeric_limits<double>::epsilon() *
                             std::max({1., a.norm(), b.norm()});
        Point<3> lo, hi;
        for (unsigned int d = 0; d < 3; ++d)
          {
            lo[d] = std::min(a[d], b[d]) - pad;
            hi[d] = std::max(a[d], b[d]) + pad;
          }
        bool finite = std::isfinite(pad);
        for (unsigned int d = 0; d < 3; ++d)
          finite = finite && std::isfinite(lo[d]) && std::isfinite(hi[d]);
        valid = valid && finite;
        if (!finite)
          continue;
        const BoundingBox<3> envelope(std::make_pair(lo, hi));
        std::set<Rank>       recipients;
        // Endpoint owners need complete incidence even for imported vessels
        // outside the background, where no geometric covering box exists.
        for (Id node : v.nodes)
          if (result.nodes.at(node).owner != rank)
            recipients.insert(result.nodes.at(node).owner);
        for (const auto &[box, target] :
             covering_tree | bgi::adaptors::queried(bgi::intersects(envelope)))
          if (target != rank)
            recipients.insert(target);
        for (Rank target : recipients)
          add_vessel(send[target], result, id, work);
      }
    require_all(valid,
                communicator,
                "A vascular geometric halo is not representable.");
    for (const auto &[rank, packet] : send)
      count_work(work,
                 GrowthOperation::ghost_vessels_sent,
                 packet.vessels.size());
    const auto ghosts = Utilities::MPI::some_to_some(communicator, send);
    for (const auto &[sender, packet] : ghosts)
      {
        count_work(work,
                   GrowthOperation::ghost_vessels_updated,
                   packet.vessels.size());
        for (const auto &[id, v] : packet.vessels)
          valid = (v.owner == sender) && valid;
        valid = merge(result, packet, work) && valid;
      }
    require_all(valid,
                communicator,
                "Vascular ghost exchange received inconsistent owner data.");

    // Terminal identity follows its node, including endpoint ghosts of vessels
    // whose midpoint belongs to a distant process.
    std::map<Id, std::map<Id, Terminal>> by_node;
    for (const auto &[id, t] : result.terminals)
      by_node[t.node].emplace(id, t);
    std::map<Rank, std::vector<Id>> requests;
    for (const auto &[id, n] : result.nodes)
      if (n.owner != rank)
        requests[n.owner].push_back(id);
    const auto incoming = Utilities::MPI::some_to_some(communicator, requests);
    std::map<Rank, std::map<Id, Terminal>> terminal_packets;
    for (const auto &[requester, ids] : incoming)
      for (Id id : ids)
        {
          const auto node = result.nodes.find(id);
          if (node == result.nodes.end() || node->second.owner != rank)
            valid = false;
          else if (by_node.count(id))
            terminal_packets[requester].insert(by_node.at(id).begin(),
                                               by_node.at(id).end());
        }
    const auto terminal_ghosts =
      Utilities::MPI::some_to_some(communicator, terminal_packets);
    for (const auto &[owner, terminals] : terminal_ghosts)
      for (const auto &[id, t] : terminals)
        valid =
          (t.owner == owner && result.terminals.emplace(id, t).second) && valid;
    require_all(
      valid && internal::validate_counted_data(result, size, work).valid(),
      communicator,
      "Vascular endpoint/terminal ghosts disagree with their authoritative owner.");
    return result;
  }
  NetworkData
  redistribute(const dealii::Triangulation<3> &background,
               const dealii::Mapping<3>       &mapping,
               const NetworkData              &initial,
               double                          halo,
               MPI_Comm                        communicator)
  {
    return internal::redistribute_with_owners(
      background, mapping, initial, halo, communicator, {});
  }
} // namespace ImmersX::Vascularization
