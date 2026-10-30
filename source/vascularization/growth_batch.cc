// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_batch.h"

#include <deal.II/particles/particle_handler.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/generation.h>

#include <tuple>

namespace ImmersX::Vascularization::internal
{
  namespace
  {
    using Rank = unsigned int;
    struct Vote
    {
      Id                            id;
      bool                          valid = true;
      std::map<Id, bool>            planes;
      std::map<ProposalKey, double> conflicts;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &id &valid &planes &conflicts;
      }
    };
    BoundingBox<3>
    envelope_box(const Edge &e, double clearance)
    {
      const double pad =
        e.radius + clearance +
        inverse_mapping_tolerance *
          std::max({1., e.points[0].norm(), e.points[1].norm()});
      Point<3> lo, hi;
      for (unsigned int d = 0; d < 3; ++d)
        {
          lo[d] = std::min(e.points[0][d], e.points[1][d]) - pad;
          hi[d] = std::max(e.points[0][d], e.points[1][d]) + pad;
        }
      return BoundingBox<3>(std::make_pair(lo, hi));
    }
    auto
    priority(double cost, const ProposalKey &key)
    {
      return std::make_tuple(cost, key.first, key.second);
    }
  } // namespace

  BatchSelection
  select_growth_batch(BackgroundGeometry               &geometry,
                      const NetworkData                &network,
                      const std::vector<BatchProposal> &local,
                      const Parameters                 &parameters,
                      MPI_Comm                          comm,
                      const std::function<double(Id)>  &radius,
                      bool check_snapshot_obstacles)
  {
    const Rank rank = Utilities::MPI::this_mpi_process(comm),
               size = Utilities::MPI::n_mpi_processes(comm);
    int comparison  = MPI_UNEQUAL;
    MPI_Comm_compare(geometry.cache.get_triangulation().get_mpi_communicator(),
                     comm,
                     &comparison);
    bool valid = (comparison == MPI_IDENT || comparison == MPI_CONGRUENT) &&
                 validate_data(network, size).valid();
    std::set<Id> local_ids;
    for (const auto &p : local)
      {
        valid = valid && p.id != invalid_id && local_ids.insert(p.id).second &&
                std::isfinite(p.cost) && p.cost >= 0. && !p.changed.empty() &&
                p.owners.size() == p.old_nodes.size();
        std::set<Id>           edges;
        std::map<Id, Point<3>> node_positions;
        for (const auto &e : p.changed)
          {
            bool finite = true;
            for (const auto &point : e.points)
              for (unsigned int d = 0; d < 3; ++d)
                finite = finite && std::isfinite(point[d]);
            bool edge_valid =
              finite && e.id != invalid_id && edges.insert(e.id).second &&
              e.nodes[0] != invalid_id && e.nodes[1] != invalid_id &&
              e.nodes[0] != e.nodes[1] && std::isfinite(e.radius) &&
              e.radius > 0.;
            if (edge_valid)
              {
                const double length = e.points[0].distance(e.points[1]);
                edge_valid =
                  std::isfinite(length) && length > parameters.minimum_length;
                const auto box = envelope_box(e, parameters.clearance);
                for (const auto &point : {box.get_boundary_points().first,
                                          box.get_boundary_points().second})
                  for (unsigned int d = 0; d < 3; ++d)
                    edge_valid = edge_valid && std::isfinite(point[d]);
              }
            for (unsigned int end = 0; end < 2; ++end)
              {
                const auto [entry, inserted] =
                  node_positions.emplace(e.nodes[end], e.points[end]);
                edge_valid =
                  edge_valid && (inserted || entry->second == e.points[end]);
              }
            valid = valid && edge_valid;
          }
        for (const auto &[id, owner] : p.owners)
          valid =
            valid && owner < size && edges.count(id) && p.old_nodes.count(id);
        for (const auto &[id, normal] : p.normals)
          {
            valid =
              valid && p.port_owners.count(id) && p.port_owners.at(id) < size;
            for (unsigned int d = 0; d < 3; ++d)
              valid = valid && std::isfinite(normal[d]);
          }
      }
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1,
      ExcMessage(
        "Malformed vascular growth batch or local physical snapshot."));
    const auto &cell_tree =
      geometry.cache.get_locally_owned_cell_bounding_boxes_rtree();
    std::vector<BoundingBox<3>> local_boxes;
    if (!cell_tree.empty())
      local_boxes = extract_rtree_level(cell_tree, 1);
    const auto boxes = Utilities::MPI::all_gather(comm, local_boxes);
    std::vector<std::pair<BoundingBox<3>, Rank>> covering;
    for (Rank r = 0; r < size; ++r)
      for (const auto &box : boxes[r])
        covering.emplace_back(box, r);
    const auto covering_tree = pack_rtree(covering);

    std::vector<Point<3>>                             positions;
    std::vector<std::vector<double>>                  point_properties;
    std::map<Id, std::pair<std::size_t, std::size_t>> ranges;
    std::map<Id, bool>                                eligible;
    std::map<Id, std::map<Id, bool>>                  planes;
    std::map<Id, std::set<Rank>>                      participants;
    std::map<Rank, std::vector<BatchProposal>>        send;
    for (const auto &p : local)
      {
        eligible[p.id]   = true;
        const auto begin = positions.size();
        auto      &ranks = participants[p.id];
        ranks.insert(rank);
        for (const auto &[id, owner] : p.owners)
          ranks.insert(owner);
        for (const auto &[id, owner] : p.port_owners)
          ranks.insert(owner);
        for (const auto &e : p.changed)
          {
            for (unsigned int end = 0; end < 2; ++end)
              {
                point_properties.push_back(
                  {static_cast<double>(rank),
                   static_cast<double>(positions.size()),
                   p.normals.count(e.nodes[end]) ? 1.e-12 : 0.});
                positions.push_back(e.points[end]);
              }
            for (const auto &[box, target] :
                 covering_tree | bgi::adaptors::queried(bgi::intersects(
                                   envelope_box(e, parameters.clearance))))
              ranks.insert(target);
          }
        ranges[p.id] = {begin, positions.size()};
        for (const auto &[id, normal] : p.normals)
          planes[p.id][id] = false;
        for (Rank target : ranks)
          send[target].push_back(p);
      }
    Particles::ParticleHandler<3> points(geometry.cache.get_triangulation(),
                                         geometry.mapping,
                                         3);
    points.insert_global_particles(positions, boxes, point_properties);
    // The native search tolerance selects candidate cells. Recheck physical
    // locations without its clamped reference coordinates, allowing only
    // inverse mapping accuracy at shared faces: a very thin vessel entirely
    // outside the domain need not intersect a boundary patch. Properties
    // contain only producer/index/tolerance, never physical ids.
    std::map<Rank, std::vector<std::size_t>> strict_found;
    for (auto particle = points.begin(); particle != points.end(); ++particle)
      {
        const auto property = particle->get_properties();
        const auto cell     = particle->get_surrounding_cell();
        try
          {
            const auto unit = geometry.mapping.transform_real_to_unit_cell(
              cell, particle->get_location());
            if (reference_contains(cell, unit, property[2]))
              strict_found[static_cast<Rank>(property[0])].push_back(
                static_cast<std::size_t>(property[1]));
          }
        catch (const Mapping<3>::ExcTransformationFailed &)
          {}
      }
    const auto destinations = Utilities::MPI::some_to_some(comm, strict_found);
    std::vector<bool> located(positions.size(), false);
    for (const auto &[owner, indices] : destinations)
      for (auto i : indices)
        located.at(i) = true;
    for (const auto &[id, range] : ranges)
      for (std::size_t i = range.first; i < range.second; ++i)
        eligible[id] = eligible[id] && located[i];
    const auto received = Utilities::MPI::some_to_some(comm, send);
    std::map<ProposalKey, BatchProposal> proposals;
    for (const auto &[producer, items] : received)
      for (const auto &p : items)
        proposals.emplace(ProposalKey{producer, p.id}, p);

    using Leaf = std::pair<Segment<3>, Id>;
    std::vector<Leaf> leaves;
    double            max_radius = 0.;
    if (check_snapshot_obstacles)
      for (const auto &[id, v] : network.vessels)
        {
          leaves.emplace_back(Segment<3>(network.nodes.at(v.nodes[0]).position,
                                         network.nodes.at(v.nodes[1]).position),
                              id);
          max_radius = std::max(max_radius, radius(id));
        }
    const auto                  obstacles = pack_rtree(leaves);
    std::map<ProposalKey, Vote> votes;
    for (const auto &[key, p] : proposals)
      {
        auto &vote = votes[key];
        vote.id    = p.id;
        for (const auto &[id, owner] : p.owners)
          if (owner == rank)
            {
              const auto v = network.vessels.find(id);
              vote.valid   = vote.valid && v != network.vessels.end() &&
                           v->second.owner == rank &&
                           v->second.nodes == p.old_nodes.at(id);
              if (v != network.vessels.end() && v->second.owner == rank)
                for (const auto &e : p.changed)
                  for (unsigned int end = 0; end < 2; ++end)
                    for (Id node : v->second.nodes)
                      if (e.nodes[end] == node)
                        vote.valid =
                          vote.valid &&
                          e.points[end] == network.nodes.at(node).position;
            }
        for (const auto &[node, owner] : p.port_owners)
          if (owner == rank)
            {
              const auto n = network.nodes.find(node);
              vote.valid   = vote.valid && n != network.nodes.end() &&
                           n->second.owner == rank &&
                           n->second.port != Port::none &&
                           p.normals.count(node) &&
                           n->second.inward_port_normal == p.normals.at(node);
            }
        for (std::size_t i = 0; i < p.changed.size(); ++i)
          {
            const auto             &e = p.changed[i];
            std::array<Point<3>, 2> normals;
            for (unsigned int end = 0; end < 2; ++end)
              if (p.normals.count(e.nodes[end]))
                {
                  normals[end] = p.normals.at(e.nodes[end]);
                  vote.planes[e.nodes[end]] =
                    vote.planes[e.nodes[end]] ||
                    geometry.planar_section(
                      e.points[end], normalized_port_normal(normals[end]));
                }
            vote.valid =
              vote.valid &&
              geometry.boundary_separated({e.points[0], e.points[1]},
                                          e.radius + parameters.clearance,
                                          parameters.geometry_depth,
                                          parameters.geometry_work,
                                          normals);
            for (std::size_t j = 0; j < i; ++j)
              vote.valid = vote.valid && compatible(e,
                                                    p.changed[j],
                                                    parameters.minimum_angle,
                                                    parameters.clearance);
            for (const auto &[s, id] :
                 obstacles |
                   bgi::adaptors::queried(bgi::intersects(
                     envelope_box(e, parameters.clearance + max_radius))))
              if (check_snapshot_obstacles && !p.owners.count(id))
                {
                  const auto &v = network.vessels.at(id);
                  vote.valid =
                    vote.valid &&
                    compatible(e,
                               {id, v.nodes, {{s.first, s.second}}, radius(id)},
                               parameters.minimum_angle,
                               parameters.clearance);
                }
          }
      }
    for (auto a = proposals.begin(); a != proposals.end(); ++a)
      for (auto b = std::next(a); b != proposals.end(); ++b)
        {
          bool conflict = false;
          for (const auto &e : a->second.changed)
            for (const auto &f : b->second.changed)
              if (e.id == f.id || !compatible(e,
                                              f,
                                              parameters.minimum_angle,
                                              parameters.clearance))
                conflict = true;
          if (conflict)
            {
              votes[a->first].conflicts[b->first] = b->second.cost;
              votes[b->first].conflicts[a->first] = a->second.cost;
            }
        }
    std::map<Rank, std::vector<Vote>> vote_packets;
    for (const auto &[key, vote] : votes)
      vote_packets[key.first].push_back(vote);
    const auto replies = Utilities::MPI::some_to_some(comm, vote_packets);
    std::map<Id, std::map<ProposalKey, double>> neighbors;
    for (const auto &[participant, items] : replies)
      for (const auto &vote : items)
        {
          eligible.at(vote.id) = eligible.at(vote.id) && vote.valid;
          for (const auto &[id, match] : vote.planes)
            planes[vote.id][id] = planes[vote.id][id] || match;
          neighbors[vote.id].insert(vote.conflicts.begin(),
                                    vote.conflicts.end());
        }
    for (const auto &[id, matches] : planes)
      for (const auto &[node, match] : matches)
        eligible[id] = eligible[id] && match;
    // Exchange eligibility only along the discovered symmetric conflict graph.
    using State = std::pair<Id, bool>;
    std::map<Rank, std::vector<State>> states;
    for (const auto &[id, adjacent] : neighbors)
      {
        std::set<Rank> targets;
        for (const auto &[key, cost] : adjacent)
          targets.insert(key.first);
        for (Rank target : targets)
          states[target].emplace_back(id, eligible.at(id));
      }
    const auto remote_states = Utilities::MPI::some_to_some(comm, states);
    std::map<ProposalKey, bool> allowed;
    for (const auto &[producer, items] : remote_states)
      for (const auto &[id, state] : items)
        allowed[{producer, id}] = state;
    BatchSelection result;
    for (const auto &p : local)
      {
        bool chosen = eligible.at(p.id);
        for (const auto &[key, cost] : neighbors[p.id])
          if (allowed.at(key) &&
              priority(cost, key) < priority(p.cost, {rank, p.id}))
            chosen = false;
        result.local[p.id] = chosen;
      }
    // All geometric participants and writing owners receive the same decision.
    states.clear();
    for (const auto &[id, targets] : participants)
      for (Rank target : targets)
        states[target].emplace_back(id, result.local.at(id));
    const auto decisions = Utilities::MPI::some_to_some(comm, states);
    for (const auto &[producer, items] : decisions)
      for (const auto &[id, state] : items)
        result.relevant[{producer, id}] = state;
    return result;
  }
} // namespace ImmersX::Vascularization::internal
