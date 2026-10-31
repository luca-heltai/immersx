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
                      const std::vector<BatchProposal> &submitted,
                      const Parameters                 &parameters,
                      MPI_Comm                          comm,
                      const std::function<double(Id)>  &radius,
                      bool check_snapshot_obstacles)
  {
    auto       local = submitted;
    const Rank rank  = Utilities::MPI::this_mpi_process(comm),
               size  = Utilities::MPI::n_mpi_processes(comm);
    int comparison   = MPI_UNEQUAL;
    MPI_Comm_compare(geometry.cache.get_triangulation().get_mpi_communicator(),
                     comm,
                     &comparison);
    bool valid = (comparison == MPI_IDENT || comparison == MPI_CONGRUENT) &&
                 validate_data(network, size).valid();
    const auto controls = std::make_pair(
      std::array<double, 6>{{parameters.clearance,
                             parameters.minimum_length,
                             parameters.minimum_radius,
                             parameters.minimum_angle,
                             static_cast<double>(parameters.geometry_depth),
                             static_cast<double>(parameters.geometry_work)}},
      parameters.length_scale);
    for (const auto &other : Utilities::MPI::all_gather(comm, controls))
      valid = valid && other == controls;
    valid =
      valid && std::isfinite(parameters.clearance) &&
      parameters.clearance >= 0. && std::isfinite(parameters.minimum_length) &&
      parameters.minimum_length >= 0. &&
      std::isfinite(parameters.minimum_radius) &&
      parameters.minimum_radius >= 0. &&
      std::isfinite(parameters.minimum_angle) &&
      parameters.minimum_angle >= 0. && parameters.minimum_angle <= 180. &&
      parameters.geometry_depth <= 30 && parameters.geometry_work > 0;
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1,
      ExcMessage("Malformed growth geometry controls or snapshot ownership."));
    geometry.configure_length_scale(parameters.length_scale, comm);
    const auto  &scale = *geometry.length_scale;
    const double broad_clearance =
      scale.physical_length(parameters.clearance, scale.maximum());
    scale.physical_length(parameters.clearance, scale.minimum());
    // A zero H denotes an exterior profile, never a fabricated mesh size.
    // Such generated proposals are ineligible; frozen exterior obstacles use
    // the generated branch's defined H in the symmetric pair maximum.
    const auto physical_clearance = [&](double h) {
      return h > 0. ? scale.physical_length(parameters.clearance, h) : 0.;
    };
    geometry.retain_profiles(network);
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
                edge_valid          = std::isfinite(length) && length > 0.;
                const auto box      = envelope_box(e, broad_clearance);
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
    std::map<Id, Segment<3>>        segments;
    std::map<std::pair<Id, Id>, Id> segment_keys;
    for (const auto &proposal : local)
      for (const auto &edge : proposal.changed)
        {
          const Id key                         = segments.size();
          segment_keys[{proposal.id, edge.id}] = key;
          segments.emplace(key, Segment<3>(edge.points[0], edge.points[1]));
        }
    std::map<Id, std::vector<BackgroundInterval>> profiles;
    if (scale.policy == LengthScalePolicy::local_mesh_size)
      profiles = geometry.crossings(segments, comm);
    std::map<Id, bool> normalized_valid;
    for (auto &proposal : local)
      {
        proposal.clearance_scales.clear();
        normalized_valid[proposal.id] = true;
        for (const auto &edge : proposal.changed)
          {
            const double length = edge.points[0].distance(edge.points[1]);
            double       normalized_length = length / scale.minimum();
            double       normalized_radius = edge.radius / scale.minimum();
            double       hmax              = scale.maximum();
            bool         complete          = true;
            if (scale.policy == LengthScalePolicy::local_mesh_size)
              {
                const auto measure = measure_background_profile(
                  length,
                  {{edge.radius, edge.radius}},
                  profiles.at(segment_keys.at({proposal.id, edge.id})),
                  scale);
                normalized_length = measure.normalized_length;
                normalized_radius = measure.minimum_normalized_radius;
                hmax              = measure.maximum_scale;
                complete          = measure.complete;
              }
            proposal.clearance_scales[edge.id] = hmax;
            normalized_valid[proposal.id] =
              normalized_valid[proposal.id] && complete &&
              std::isfinite(normalized_length) &&
              normalized_length >= parameters.minimum_length &&
              std::isfinite(normalized_radius) &&
              normalized_radius >= parameters.minimum_radius;
          }
      }
    std::map<Id, double>     obstacle_scales;
    std::map<Id, Segment<3>> obstacle_segments;
    if (check_snapshot_obstacles && parameters.clearance > 0.)
      for (const auto &[id, vessel] : network.vessels)
        obstacle_segments.emplace(
          id,
          Segment<3>(network.nodes.at(vessel.nodes[0]).position,
                     network.nodes.at(vessel.nodes[1]).position));
    if (scale.policy == LengthScalePolicy::local_mesh_size)
      {
        const auto obstacle_profiles =
          geometry.crossings(obstacle_segments, comm);
        for (const auto &[id, profile] : obstacle_profiles)
          obstacle_scales[id] =
            measure_background_profile(obstacle_segments.at(id).first.distance(
                                         obstacle_segments.at(id).second),
                                       network.vessels.at(id).radii,
                                       profile,
                                       scale)
              .maximum_scale;
      }
    else if (scale.policy == LengthScalePolicy::meter ||
             scale.policy == LengthScalePolicy::min_mesh_size ||
             scale.policy == LengthScalePolicy::max_mesh_size)
      for (const auto &[id, segment] : obstacle_segments)
        obstacle_scales[id] = scale.maximum();
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
        eligible[p.id]   = normalized_valid.at(p.id);
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
                                   envelope_box(e, broad_clearance))))
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
              geometry.boundary_separated(
                {e.points[0], e.points[1]},
                e.radius + physical_clearance(p.clearance_scales.at(e.id)),
                parameters.geometry_depth,
                parameters.geometry_work,
                normals);
            for (std::size_t j = 0; j < i; ++j)
              vote.valid =
                vote.valid &&
                compatible(e,
                           p.changed[j],
                           parameters.minimum_angle,
                           physical_clearance(
                             std::max(p.clearance_scales.at(e.id),
                                      p.clearance_scales.at(p.changed[j].id))));
            for (const auto &[s, id] :
                 obstacles | bgi::adaptors::queried(bgi::intersects(
                               envelope_box(e, broad_clearance + max_radius))))
              if (check_snapshot_obstacles && !p.owners.count(id))
                {
                  const auto &v = network.vessels.at(id);
                  vote.valid =
                    vote.valid &&
                    compatible(e,
                               {id, v.nodes, {{s.first, s.second}}, radius(id)},
                               parameters.minimum_angle,
                               physical_clearance(
                                 std::max(p.clearance_scales.at(e.id),
                                          parameters.clearance > 0. ?
                                            obstacle_scales.at(id) :
                                            0.)));
                }
          }
      }
    std::vector<std::pair<BoundingBox<3>, ProposalKey>> changed_boxes;
    std::map<Id, std::set<ProposalKey>> vessel_writers, node_writers;
    for (const auto &[key, proposal] : proposals)
      for (const auto &edge : proposal.changed)
        {
          changed_boxes.emplace_back(envelope_box(edge, broad_clearance), key);
          vessel_writers[edge.id].insert(key);
          for (Id node : edge.nodes)
            node_writers[node].insert(key);
        }
    const auto changed_tree = pack_rtree(changed_boxes);
    for (const auto &[a, proposal] : proposals)
      {
        std::set<ProposalKey> nearby;
        for (const auto &edge : proposal.changed)
          {
            for (const auto &[box, key] :
                 changed_tree | bgi::adaptors::queried(bgi::intersects(
                                  envelope_box(edge, broad_clearance))))
              nearby.insert(key);
            // Identifier conflicts and shared-junction angles remain relevant
            // independently of the provisional geometric descriptions.
            const auto &writers = vessel_writers.at(edge.id);
            nearby.insert(writers.begin(), writers.end());
            for (Id node : edge.nodes)
              {
                const auto &incident = node_writers.at(node);
                nearby.insert(incident.begin(), incident.end());
              }
          }
        for (const auto &b : nearby)
          if (a < b)
            {
              const auto &other    = proposals.at(b);
              bool        conflict = false;
              for (const auto &e : proposal.changed)
                for (const auto &f : other.changed)
                  if (e.id == f.id ||
                      !compatible(e,
                                  f,
                                  parameters.minimum_angle,
                                  physical_clearance(
                                    std::max(proposal.clearance_scales.at(e.id),
                                             other.clearance_scales.at(f.id)))))
                    conflict = true;
              if (conflict)
                {
                  votes[a].conflicts[b] = other.cost;
                  votes[b].conflicts[a] = proposal.cost;
                }
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
