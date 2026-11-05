// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/array_view.h>
#include <deal.II/base/numbers.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/solver_cg.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/reference_flow.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <tuple>
#include <type_traits>

#include "reference_flow_internal.h"

namespace ImmersX::Vascularization
{
  void
  ReferenceParameters::validate() const
  {
    const auto tolerance = [](double x) {
      return std::isfinite(x) && x > 0. && x < 1.;
    };
    AssertThrow(
      maximum_iterations > 0 && tolerance(linear_relative_tolerance) &&
        tolerance(conservation_relative_tolerance) &&
        std::isfinite(absolute_flow_tolerance) && absolute_flow_tolerance >= 0.,
      dealii::ExcMessage(
        "Invalid vascular reference iteration budget or numerical tolerances."));
  }

  namespace
  {
    using Rank  = unsigned int;
    using Index = dealii::types::global_dof_index;
    struct ReferenceNode
    {
      Id               id;
      Index            index;
      Rank             owner;
      dealii::Point<3> position, normal;
      Port             port;
      Provenance       provenance;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &id &index &owner &position &normal &port &provenance;
      }
    };
    struct ReferenceTerminal
    {
      Id     id, node, territory;
      Rank   owner;
      System system;
      double demand;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &id &node &territory &owner &system &demand;
      }
    };
    struct ReferenceRegistry
    {
      std::vector<ReferenceNode>     nodes;
      std::vector<ReferenceTerminal> terminals;
      std::vector<Id>                vessels, territories, paired_terminals;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &nodes &terminals &vessels &territories &paired_terminals;
      }
    };
    struct ReferenceBoundary
    {
      enum Kind
      {
        pressure,
        inflow,
        robin
      };
      Id     node;
      Kind   kind;
      double value;
      double coefficient = 0.;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &node &kind &value &coefficient;
      }
    };
    struct ReferenceArc
    {
      Id            node;
      ReferenceNode neighbor;
      double        resistance;
      bool          vascular, end;
      System        system;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &node &neighbor &resistance &vascular &end &system;
      }
    };
    struct ReferenceLift
    {
      Id          node, anchor;
      long double distance;
      double      pressure;
      bool        robin;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &node &anchor &distance &pressure &robin;
      }
    };
  } // namespace

  static ReferenceResult
  solve_reference_impl(const NetworkData                  &network,
                       const std::vector<TerminalPair>    &pairs,
                       const std::vector<TerminalTriplet> &triplets,
                       const ReferenceConditions          &conditions,
                       const ReferenceParameters          &parameters,
                       MPI_Comm                            communicator)
  {
    using namespace dealii;
    const Rank rank       = Utilities::MPI::this_mpi_process(communicator),
               size       = Utilities::MPI::n_mpi_processes(communicator);
    bool valid_parameters = true;
    try
      {
        parameters.validate();
      }
    catch (const std::exception &)
      {
        if (size == 1)
          throw;
        valid_parameters = false;
      }
    AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid_parameters),
                                    communicator) == 1,
                ExcMessage("Invalid distributed reference solver controls."));
    const std::array<double, 4> controls{
      {static_cast<double>(parameters.maximum_iterations),
       parameters.linear_relative_tolerance,
       parameters.conservation_relative_tolerance,
       parameters.absolute_flow_tolerance}};
    for (const auto &other : Utilities::MPI::all_gather(communicator, controls))
      valid_parameters = other == controls && valid_parameters;
    AssertThrow(valid_parameters,
                ExcMessage(
                  "Reference solver controls must agree on all processes."));
    ReferenceResult result;
    result.validation = validate_data(network, size);
    const auto issue =
      [&](const std::string &entity, Id id, const std::string &message) {
        result.validation.issues.push_back({entity, id, message});
      };
    const auto phase = [&]() {
      const bool valid = Utilities::MPI::min(static_cast<unsigned int>(
                                               result.validation.valid()),
                                             communicator) == 1;
      if (!valid && result.validation.valid())
        issue("reference",
              invalid_id,
              "Reference validation failed on another process.");
      return valid;
    };
    Id local_nodes = 0, local_vessels = 0;
    for (const auto &[id, n] : network.nodes)
      if (n.owner == rank)
        ++local_nodes;
    for (const auto &[id, v] : network.vessels)
      if (v.owner == rank)
        ++local_vessels;
    const Id global_nodes = Utilities::MPI::sum(local_nodes, communicator);
    if (Utilities::MPI::sum(local_vessels, communicator) == 0 ||
        global_nodes == 0)
      issue("reference",
            invalid_id,
            "Reference flow requires a nonempty vascular graph.");
    if (global_nodes >= std::numeric_limits<Index>::max())
      issue("reference", invalid_id, "Too many reference graph nodes.");
    for (const auto &p : pairs)
      if (p.territory == invalid_id || p.arterial == invalid_id ||
          p.venous == invalid_id || !std::isfinite(p.resistance) ||
          p.resistance <= 0.)
        issue(
          "terminal pair",
          p.territory,
          "A bed requires valid identifiers and a positive finite resistance.");
    for (const auto &t : triplets)
      {
        if (t.territory == invalid_id || t.arterial == invalid_id ||
            t.portal == invalid_id || t.venous == invalid_id)
          issue("terminal triplet",
                t.territory,
                "A triplet needs valid identifiers.");
        for (const double r : t.resistances)
          if (!std::isfinite(r) || !(r > 0.))
            issue(
              "terminal triplet",
              t.territory,
              "All three microvascular star arms must be positive and finite.");
      }
    if (!phase())
      return result;
    Id offset = 0;
    MPI_Exscan(&local_nodes, &offset, 1, MPI_UINT64_T, MPI_SUM, communicator);
    if (rank == 0)
      offset = 0;
    std::map<Rank, ReferenceRegistry> registration;
    Index                             next = offset;
    std::vector<Id>                   owned_ids;
    for (const auto &[id, n] : network.nodes)
      if (n.owner == rank)
        {
          registration[id % size].nodes.push_back({id,
                                                   next++,
                                                   rank,
                                                   n.position,
                                                   n.inward_port_normal,
                                                   n.port,
                                                   n.provenance});
          owned_ids.push_back(id);
        }
    for (const auto &[id, v] : network.vessels)
      if (v.owner == rank)
        registration[id % size].vessels.push_back(id);
    for (const auto &[id, t] : network.terminals)
      if (t.owner == rank)
        registration[id % size].terminals.push_back(
          {id, t.node, t.territory, rank, t.system, t.demand});
    for (const auto &p : pairs)
      {
        registration[p.territory % size].territories.push_back(p.territory);
        registration[p.arterial % size].paired_terminals.push_back(p.arterial);
        registration[p.venous % size].paired_terminals.push_back(p.venous);
      }
    for (const auto &t : triplets)
      {
        registration[t.territory % size].territories.push_back(t.territory);
        for (Id id : {t.arterial, t.portal, t.venous})
          registration[id % size].paired_terminals.push_back(id);
      }
    std::map<Id, ReferenceNode>     node_registry;
    std::map<Id, ReferenceTerminal> terminal_registry;
    std::set<Id>                    vessel_ids, territory_ids, paired_ids;
    for (const auto &[source, packet] :
         Utilities::MPI::some_to_some(communicator, registration))
      {
        for (const auto &n : packet.nodes)
          if (!node_registry.emplace(n.id, n).second)
            issue(
              "node",
              n.id,
              "A physical reference node has multiple authoritative owners.");
        for (const auto &t : packet.terminals)
          if (!terminal_registry.emplace(t.id, t).second)
            issue(
              "terminal",
              t.id,
              "A physical reference terminal has multiple authoritative owners.");
        for (Id id : packet.vessels)
          if (!vessel_ids.insert(id).second)
            issue(
              "vessel",
              id,
              "A physical reference vessel has multiple authoritative owners.");
        for (Id id : packet.territories)
          if (!territory_ids.insert(id).second)
            issue(
              "terminal pair",
              id,
              "Each bed territory must be supplied on exactly one process.");
        for (Id id : packet.paired_terminals)
          if (!paired_ids.insert(id).second)
            issue("terminal pair",
                  id,
                  "A terminal cannot belong to multiple beds.");
      }
    if (!phase())
      return result;
    // Hash-sharded registries resolve only locally needed physical identities.
    // The graph and its compact solver numbering are never gathered.
    const auto lookup = [&](const auto &registry, const std::set<Id> &ids) {
      std::map<Rank, std::vector<Id>> queries;
      for (Id id : ids)
        queries[id % size].push_back(id);
      using Table = std::decay_t<decltype(registry)>;
      std::map<Rank, Table> replies;
      for (const auto &[source, requested] :
           Utilities::MPI::some_to_some(communicator, queries))
        for (Id id : requested)
          {
            const auto found = registry.find(id);
            if (found == registry.end())
              issue("reference",
                    id,
                    "Unknown physical reference node or terminal id.");
            else
              replies[source].emplace(id, found->second);
          }
      Table found;
      for (const auto &[source, entries] :
           Utilities::MPI::some_to_some(communicator, replies))
        found.insert(entries.begin(), entries.end());
      return found;
    };
    std::set<Id> needed_terminals;
    for (const auto &p : pairs)
      {
        needed_terminals.insert(p.arterial);
        needed_terminals.insert(p.venous);
      }
    for (const auto &t : triplets)
      for (Id id : {t.arterial, t.portal, t.venous})
        needed_terminals.insert(id);
    const auto pair_terminals = lookup(terminal_registry, needed_terminals);
    if (!phase())
      return result;
    std::set<Id> needed_nodes;
    for (const auto &[id, n] : network.nodes)
      needed_nodes.insert(id);
    for (const auto &[id, t] : pair_terminals)
      needed_nodes.insert(t.node);
    auto nodes = lookup(node_registry, needed_nodes);
    if (!phase())
      return result;
    for (const auto &[id, n] : network.nodes)
      {
        const auto &actual = nodes.at(id);
        if (actual.owner != n.owner || actual.position != n.position ||
            actual.port != n.port || actual.normal != n.inward_port_normal ||
            actual.provenance != n.provenance)
          issue(
            "node",
            id,
            "A reference endpoint ghost disagrees with its authoritative node.");
      }
    std::map<Id, Id> tissue_by_node;
    for (const auto &[id, t] : network.terminals)
      if (t.owner == rank)
        {
          if (nodes.at(t.node).owner != rank ||
              !tissue_by_node.emplace(t.node, id).second)
            issue("terminal",
                  id,
                  "Terminal authority must follow its unique physical node.");
        }
    if (!phase())
      return result;
    struct Resistor
    {
      std::array<ReferenceNode, 2> nodes;
      double                       resistance;
      bool                         vascular;
      Id                           id;
      double                       demand;
      bool                         paired;
    };
    std::vector<Resistor> resistors;
    const auto            append = [&](Id     a,
                            Id     b,
                            double r,
                            bool   vascular,
                            Id     id,
                            double demand,
                            bool   paired = true) {
      if (!std::isfinite(r) || r <= 0. || !std::isfinite(1. / r) ||
          1. / r <= 0.)
        issue(
          "resistance",
          id,
          "Resistance and conductance must be positive and representable; no radius clipping is applied.");
      else
        resistors.push_back(
          {{{nodes.at(a), nodes.at(b)}}, r, vascular, id, demand, paired});
    };
    for (const auto &[id, v] : network.vessels)
      if (v.owner == rank)
        append(v.nodes[0], v.nodes[1], resistance(network, id), true, id, 1.);
    std::map<Rank, std::vector<std::pair<Id, System>>> paired_send;
    for (const auto &p : pairs)
      {
        const auto &a = pair_terminals.at(p.arterial),
                   &v = pair_terminals.at(p.venous);
        const double separation =
          nodes.at(a.node).position.distance(nodes.at(v.node).position);
        if (a.system != System::arterial || v.system != System::venous ||
            a.territory != p.territory || v.territory != p.territory ||
            a.demand != v.demand || !std::isfinite(separation) ||
            separation <= 0.)
          issue(
            "terminal pair",
            p.territory,
            "A bed requires distinct arterial and venous endpoints with shared territory and demand.");
        append(a.node, v.node, p.resistance, false, p.territory, a.demand);
        paired_send[nodes.at(a.node).owner].emplace_back(a.node, a.system);
        paired_send[nodes.at(v.node).owner].emplace_back(v.node, v.system);
      }
    for (const auto &t : triplets)
      {
        const std::array<Id, 3>     ids{{t.arterial, t.portal, t.venous}};
        const std::array<System, 3> systems{
          {System::arterial, System::portal, System::venous}};
        const double               demand = pair_terminals.at(ids[0]).demand;
        std::array<Id, 3>          endpoints;
        std::array<long double, 3> conductance;
        long double                sum = 0.;
        for (unsigned int i = 0; i < 3; ++i)
          {
            const auto &terminal = pair_terminals.at(ids[i]);
            endpoints[i]         = terminal.node;
            if (terminal.system != systems[i] ||
                terminal.territory != t.territory || terminal.demand != demand)
              issue(
                "terminal triplet",
                t.territory,
                "A triplet needs arterial, portal and venous terminals in one territory with common demand.");
            paired_send[nodes.at(terminal.node).owner].emplace_back(
              terminal.node, terminal.system);
            conductance[i] = 1.L / t.resistances[i];
            sum += conductance[i];
          }
        for (unsigned int i = 0; i < 3; ++i)
          for (unsigned int j = i + 1; j < 3; ++j)
            {
              if (ids[i] == ids[j] || endpoints[i] == endpoints[j] ||
                  !(nodes.at(endpoints[i])
                      .position.distance(nodes.at(endpoints[j]).position) > 0.))
                issue(
                  "terminal triplet",
                  t.territory,
                  "A triplet needs three geometrically distinct endpoints.");
              // Schur complement of the common compartment pressure.
              const double r =
                static_cast<double>(sum / conductance[i] / conductance[j]);
              append(endpoints[i],
                     endpoints[j],
                     r,
                     false,
                     t.territory,
                     demand,
                     false);
            }
      }
    if (!phase())
      return result;
    std::map<Rank, std::vector<ReferenceArc>> arc_send;
    for (const auto &r : resistors)
      for (unsigned int end = 0; end < 2; ++end)
        arc_send[r.nodes[end].owner].push_back(
          {r.nodes[end].id,
           r.nodes[1 - end],
           r.resistance,
           r.vascular,
           end == 1,
           r.vascular ? network.vessels.at(r.id).system : System::arterial});
    std::vector<std::vector<ReferenceArc>> rows(local_nodes);
    for (const auto &[source, arcs] :
         Utilities::MPI::some_to_some(communicator, arc_send))
      for (const auto &arc : arcs)
        {
          const auto n = nodes.find(arc.node);
          if (n == nodes.end() || n->second.owner != rank)
            issue("node",
                  arc.node,
                  "A reference row is absent from its physical owner.");
          else
            {
              rows.at(n->second.index - offset).push_back(arc);
              nodes.emplace(arc.neighbor.id, arc.neighbor);
            }
        }
    std::set<Id> paired_nodes;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(communicator, paired_send))
      for (const auto &[id, system] : entries)
        {
          paired_nodes.insert(id);
          const auto  &row      = rows.at(nodes.at(id).index - offset);
          unsigned int degree   = 0;
          bool         oriented = true;
          for (const auto &arc : row)
            if (arc.vascular)
              {
                ++degree;
                oriented = oriented && arc.end && arc.system == system;
              }
          if (!tissue_by_node.count(id) || degree != 1 || !oriented)
            issue(
              "terminal pair",
              id,
              "Each paired terminal must end one correctly oriented vessel of its own system.");
        }
    // Boundary controls may be distributed or repeated identically. Resolve
    // them at the node registry before forwarding one value to each owner.
    std::map<Rank, std::vector<ReferenceBoundary>> boundary_send;
    for (const auto &[id, value] : conditions.pressures)
      boundary_send[id % size].push_back(
        {id, ReferenceBoundary::pressure, value});
    for (const auto &[id, value] : conditions.inflows)
      boundary_send[id % size].push_back(
        {id, ReferenceBoundary::inflow, value});
    for (const auto &[id, data] : conditions.robin)
      boundary_send[id % size].push_back(
        {id, ReferenceBoundary::robin, data.second, data.first});
    std::map<Id, ReferenceBoundary> registered;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(communicator, boundary_send))
      for (const auto &b : entries)
        {
          const auto [it, inserted] = registered.emplace(b.node, b);
          if (!std::isfinite(b.value) || !node_registry.count(b.node) ||
              !std::isfinite(b.coefficient) || b.coefficient < 0. ||
              (!inserted &&
               (it->second.kind != b.kind || it->second.value != b.value ||
                it->second.coefficient != b.coefficient)))
            issue(
              "boundary",
              b.node,
              "Boundary data require a known physical node, one condition kind, consistent finite SI values and nonnegative Robin conductance.");
        }
    boundary_send.clear();
    for (const auto &[id, data] : registered)
      if (node_registry.count(id))
        boundary_send[node_registry.at(id).owner].push_back(data);
    ReferenceConditions owned_bc;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(communicator, boundary_send))
      for (const auto &b : entries)
        {
          if ((nodes.at(b.node).port == Port::none &&
               !tissue_by_node.count(b.node)) ||
              paired_nodes.count(b.node))
            issue(
              "boundary",
              b.node,
              "External boundary data require a port or unpaired tissue terminal.");
          if (b.kind == ReferenceBoundary::robin)
            owned_bc.robin.emplace(b.node,
                                   std::make_pair(b.coefficient, b.value));
          else
            {
              auto &values = b.kind == ReferenceBoundary::pressure ?
                               owned_bc.pressures :
                               owned_bc.inflows;
              values.emplace(b.node, b.value);
            }
        }
    for (std::size_t i = 0; i < rows.size(); ++i)
      if (rows[i].empty())
        issue("node",
              owned_ids[i],
              "An isolated node has no resistive reference equation.");
    if (!phase())
      return result;

    IndexSet owned(global_nodes), ghosts(global_nodes);
    if (local_nodes > 0)
      owned.add_range(offset, offset + local_nodes);
    for (const auto &[id, n] : nodes)
      ghosts.add_index(n.index);
    ghosts.subtract_set(owned);
    using Vector = LinearAlgebra::distributed::Vector<double>;
    Vector layout, ghost_layout;
    layout.reinit(owned, communicator);
    ghost_layout.reinit(owned, ghosts, communicator);
    Vector rhs(layout), solution(layout), inverse_diagonal(layout);
    Vector fixed(ghost_layout), lifting(ghost_layout), work(ghost_layout);
    const Vector &fixed_values = fixed, &lifting_values = lifting,
                 &work_values = work;
    for (const auto &[id, value] : owned_bc.pressures)
      {
        fixed.local_element(nodes.at(id).index - offset)   = 1.;
        lifting.local_element(nodes.at(id).index - offset) = value;
      }
    fixed.update_ghost_values();
    // Solve for small excess pressures around the nearest pressure anchor in
    // resistive distance. Sparse relaxation joins local Dijkstra traversals;
    // no global topology, hop-based lift or large pressure subtraction is used.
    std::vector<long double> distances(
      local_nodes, std::numeric_limits<long double>::infinity());
    std::vector<Id>   anchors(local_nodes, invalid_id);
    std::vector<bool> robin_lift(local_nodes, false);
    using Entry = std::tuple<long double, Id, Index>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>>
      frontier;
    for (const auto &[id, value] : owned_bc.pressures)
      {
        const Index i = nodes.at(id).index - offset;
        distances[i]  = 0.;
        anchors[i]    = id;
        frontier.emplace(0., id, i);
      }
    for (const auto &[id, data] : owned_bc.robin)
      if (data.first > 0.)
        {
          const Index  i           = nodes.at(id).index - offset;
          const double equilibrium = data.second / data.first;
          // The lift is only a numerical reference: a very small beta may
          // have unrepresentable g/beta while another boundary anchors a
          // perfectly finite solution. Zero is then a valid lifting value.
          const double pressure = std::isfinite(equilibrium) ? equilibrium : 0.;
          distances[i]          = 0.;
          anchors[i]            = id;
          robin_lift[i]         = true;
          lifting.local_element(i) = pressure;
          frontier.emplace(distances[i], id, i);
        }
    if (!phase())
      return result;
    // A Robin-only component uses one common equilibrium-pressure lift.
    // Independent endpoint lifts can create artificial large interior jumps,
    // obscuring the much smaller physical boundary flows in the CG residual.
    // Prefer prescribed-pressure lifts when available; otherwise propagate the
    // lowest Robin node id, keeping the choice independent of partitioning.
    const auto better = [](long double distance,
                           Id          anchor,
                           bool        robin,
                           long double old_distance,
                           Id          old_anchor,
                           bool        old_robin) {
      if (old_anchor == invalid_id)
        return true;
      if (robin != old_robin)
        return !robin;
      if (robin && anchor != old_anchor)
        return anchor < old_anchor;
      return distance < old_distance ||
             (distance == old_distance && anchor < old_anchor);
    };
    while (Utilities::MPI::sum(static_cast<Id>(frontier.size()), communicator) >
           0)
      {
        std::map<Rank, std::map<Id, ReferenceLift>> outgoing;
        while (!frontier.empty())
          {
            const auto [distance, origin, i] = frontier.top();
            frontier.pop();
            if (distance != distances[i] || origin != anchors[i])
              continue;
            for (const auto &arc : rows[i])
              {
                const auto       &target = arc.neighbor;
                const long double candidate =
                  distance + static_cast<long double>(arc.resistance);
                if (fixed_values[target.index] != 0. ||
                    !std::isfinite(candidate))
                  continue;
                if (target.owner == rank)
                  {
                    const Index j = target.index - offset;
                    if (better(candidate,
                               origin,
                               robin_lift[i],
                               distances[j],
                               anchors[j],
                               robin_lift[j]))
                      {
                        distances[j]             = candidate;
                        anchors[j]               = origin;
                        robin_lift[j]            = robin_lift[i];
                        lifting.local_element(j) = lifting.local_element(i);
                        frontier.emplace(candidate, origin, j);
                      }
                  }
                else
                  {
                    auto      &packet = outgoing[target.owner];
                    const auto found  = packet.find(target.id);
                    if (found == packet.end() || better(candidate,
                                                        origin,
                                                        robin_lift[i],
                                                        found->second.distance,
                                                        found->second.anchor,
                                                        found->second.robin))
                      packet[target.id] = {target.id,
                                           origin,
                                           candidate,
                                           lifting.local_element(i),
                                           robin_lift[i]};
                  }
              }
          }
        for (const auto &[source, entries] :
             Utilities::MPI::some_to_some(communicator, outgoing))
          for (const auto &[id, proposal] : entries)
            {
              const Index i = nodes.at(id).index - offset;
              if (fixed.local_element(i) == 0. && better(proposal.distance,
                                                         proposal.anchor,
                                                         proposal.robin,
                                                         distances[i],
                                                         anchors[i],
                                                         robin_lift[i]))
                {
                  distances[i]             = proposal.distance;
                  anchors[i]               = proposal.anchor;
                  robin_lift[i]            = proposal.robin;
                  lifting.local_element(i) = proposal.pressure;
                  frontier.emplace(proposal.distance, proposal.anchor, i);
                }
            }
      }
    for (std::size_t i = 0; i < rows.size(); ++i)
      if (anchors[i] == invalid_id)
        issue(
          "reference",
          owned_ids[i],
          "Each reference component needs a prescribed pressure or positive Robin conductance at a representable resistive distance.");
    if (!phase())
      return result;
    lifting.update_ghost_values();
    for (std::size_t i = 0; i < rows.size(); ++i)
      {
        const Index index = offset + i;
        if (fixed_values[index] != 0.)
          {
            inverse_diagonal.local_element(i) = 1.;
            continue;
          }
        double diagonal = 0.;
        double source   = owned_bc.inflows.count(owned_ids[i]) ?
                            owned_bc.inflows.at(owned_ids[i]) :
                            0.;
        if (const auto bc = owned_bc.robin.find(owned_ids[i]);
            bc != owned_bc.robin.end())
          {
            diagonal += bc->second.first;
            source +=
              bc->second.second - bc->second.first * lifting_values[index];
          }
        for (const auto &arc : rows[i])
          {
            diagonal += 1. / arc.resistance;
            source +=
              (lifting_values[arc.neighbor.index] - lifting_values[index]) /
              arc.resistance;
          }
        if (!std::isfinite(diagonal) || diagonal <= 0. ||
            !std::isfinite(1. / diagonal) || !std::isfinite(source))
          issue("node",
                owned_ids[i],
                "Reference diagonal or lifted forcing is not representable.");
        rhs.local_element(i)              = source;
        inverse_diagonal.local_element(i) = 1. / diagonal;
      }
    if (!phase())
      return result;
    LinearOperator<Vector> op;
    op.reinit_range_vector = [&](Vector &v, bool omit_zeroing) {
      v.reinit(layout, omit_zeroing);
    };
    op.reinit_domain_vector = op.reinit_range_vector;
    op.vmult                = [&](Vector &dst, const Vector &src) {
      work.zero_out_ghost_values();
      work.copy_locally_owned_data_from(src);
      work.update_ghost_values();
      for (std::size_t i = 0; i < rows.size(); ++i)
        {
          const double value   = src.local_element(i);
          double       product = 0.;
          if (fixed.local_element(i) != 0.)
            product = value;
          else
            {
              if (const auto bc = owned_bc.robin.find(owned_ids[i]);
                  bc != owned_bc.robin.end())
                product = bc->second.first * value;
              for (const auto &arc : rows[i])
                product += (value - (fixed_values[arc.neighbor.index] != 0. ?
                                                      0. :
                                                      work_values[arc.neighbor.index])) /
                           arc.resistance;
            }
          dst.local_element(i) = product;
        }
    };
    op.Tvmult = op.vmult;
    ReductionControl             control(parameters.maximum_iterations,
                             0.,
                             parameters.linear_relative_tolerance,
                             false,
                             false);
    SolverCG<Vector>             solver(control);
    const DiagonalMatrix<Vector> jacobi(inverse_diagonal);
    try
      {
        solver.solve(op, solution, rhs, jacobi);
      }
    catch (const SolverControl::NoConvergence &)
      {
        issue("reference",
              invalid_id,
              "Reference linear solve exhausted its convergence budget.");
      }
    result.iterations      = control.last_step();
    result.linear_residual = control.last_value();
    work.zero_out_ghost_values();
    work.copy_locally_owned_data_from(solution);
    work.update_ghost_values();
    for (const auto &[id, n] : nodes)
      {
        const double pressure = work_values[n.index] + lifting_values[n.index];
        result.pressures[id]  = pressure;
        if (!std::isfinite(pressure))
          issue("pressure", id, "Reference pressure is not finite.");
      }
    const auto flow = [&](Index i, Index j, double r) {
      return ((work_values[i] - work_values[j]) +
              (lifting_values[i] - lifting_values[j])) /
             r;
    };
    double flow_scale = 0.;
    for (const auto &r : resistors)
      {
        const double q = flow(r.nodes[0].index, r.nodes[1].index, r.resistance);
        const double power = q * (q * r.resistance);
        flow_scale         = std::max(flow_scale, std::abs(q));
        result.dissipated_power += power;
        if (!std::isfinite(q) || !std::isfinite(power))
          issue("reference",
                r.id,
                "Reference flow or dissipation is not representable.");
        if (r.vascular)
          {
            const auto     &v = network.vessels.at(r.id);
            VesselReference state;
            state.resistance = r.resistance;
            state.flow       = q;
            for (unsigned int end = 0; end < 2; ++end)
              {
                state.wall_shear[end] =
                  4. * v.viscosity * std::abs(q) /
                  (numbers::PI * std::pow(v.radii[end], 3));
                if (!std::isfinite(state.wall_shear[end]))
                  issue("vessel",
                        r.id,
                        "Reference wall shear is not representable.");
              }
            result.vessels[r.id] = state;
          }
        else if (r.paired)
          {
            result.territories[r.id] = {q, q / r.demand, q, 0., q};
            if (!std::isfinite(q / r.demand) || !(q > 0.))
              issue(
                "territory",
                r.id,
                "Microvascular perfusion must be positive and representable from artery to vein.");
          }
      }
    for (const auto &t : triplets)
      {
        const std::array<Id, 3> ids{{t.arterial, t.portal, t.venous}};
        const Index origin = nodes.at(pair_terminals.at(t.venous).node).index;
        std::array<long double, 3> differences, conductance;
        long double                sum = 0., weighted = 0.;
        for (unsigned int i = 0; i < 3; ++i)
          {
            const Index index = nodes.at(pair_terminals.at(ids[i]).node).index;
            differences[i]    = static_cast<long double>(work_values[index]) -
                             work_values[origin] +
                             static_cast<long double>(lifting_values[index]) -
                             lifting_values[origin];
            conductance[i] = 1.L / t.resistances[i];
            sum += conductance[i];
            weighted += conductance[i] * differences[i];
          }
        const long double excess = weighted / sum;
        const double      qa =
          static_cast<double>((differences[0] - excess) * conductance[0]);
        const double qp =
          static_cast<double>((differences[1] - excess) * conductance[1]);
        const double qv = static_cast<double>(excess * conductance[2]);
        const double pressure =
          static_cast<double>(static_cast<long double>(work_values[origin]) +
                              lifting_values[origin] + excess);
        const double demand             = pair_terminals.at(t.arterial).demand;
        result.territories[t.territory] = {
          qv, qv / demand, qa, qp, qv, pressure};
        if (!(qa > 0.) || !(qp > 0.) || !(qv > 0.) || !std::isfinite(qa) ||
            !std::isfinite(qp) || !std::isfinite(qv) ||
            !std::isfinite(pressure) || !std::isfinite(qv / demand))
          issue(
            "terminal triplet",
            t.territory,
            "Arterial and portal perfusion and venous drainage must be positive and representable.");
        result.maximum_flow_defect =
          std::max(result.maximum_flow_defect, std::abs(qa + qp - qv));
        flow_scale =
          std::max({flow_scale, std::abs(qa), std::abs(qp), std::abs(qv)});
      }
    for (std::size_t i = 0; i < rows.size(); ++i)
      {
        double outflow = 0., vascular_outflow = 0.;
        for (const auto &arc : rows[i])
          {
            const double q =
              flow(offset + i, arc.neighbor.index, arc.resistance);
            outflow += q;
            if (arc.vascular)
              vascular_outflow += q;
            flow_scale = std::max(flow_scale, std::abs(q));
          }
        const Id id = owned_ids[i];
        if (!std::isfinite(outflow) || !std::isfinite(vascular_outflow))
          issue("node", id, "Accumulated reference flow is not representable.");
        if (tissue_by_node.count(id))
          result.terminal_outflows[tissue_by_node.at(id)] = -vascular_outflow;
        if (owned_bc.pressures.count(id))
          result.boundary_inflows[id] = outflow;
        else
          {
            double prescribed =
              owned_bc.inflows.count(id) ? owned_bc.inflows.at(id) : 0.;
            if (const auto bc = owned_bc.robin.find(id);
                bc != owned_bc.robin.end())
              prescribed =
                bc->second.second - bc->second.first * result.pressures.at(id);
            if (!std::isfinite(prescribed))
              issue("boundary",
                    id,
                    "Robin reference flux is not representable.");
            flow_scale = std::max(flow_scale, std::abs(prescribed));
            result.maximum_flow_defect =
              std::max(result.maximum_flow_defect,
                       std::abs(outflow - prescribed));
            if (owned_bc.inflows.count(id) || owned_bc.robin.count(id))
              result.boundary_inflows[id] = prescribed;
          }
      }
    for (const auto &[id, value] : result.boundary_inflows)
      {
        result.boundary_flow_balance += value;
        flow_scale = std::max(flow_scale, std::abs(value));
      }
    result.boundary_flow_balance =
      Utilities::MPI::sum(result.boundary_flow_balance, communicator);
    result.dissipated_power =
      Utilities::MPI::sum(result.dissipated_power, communicator);
    flow_scale = Utilities::MPI::max(flow_scale, communicator);
    result.maximum_flow_defect =
      std::max(Utilities::MPI::max(result.maximum_flow_defect, communicator),
               std::abs(result.boundary_flow_balance));
    result.relative_flow_defect =
      flow_scale > 0. ? result.maximum_flow_defect / flow_scale : 0.;
    const double allowance =
      parameters.absolute_flow_tolerance +
      parameters.conservation_relative_tolerance * flow_scale;
    if (result.maximum_flow_defect > allowance)
      issue(
        "reference",
        invalid_id,
        "Independent physical flow balance exceeds its requested tolerance.");
    for (const auto &[id, state] : result.vessels)
      {
        const auto system = network.vessels.at(id).system;
        if (((system == System::arterial || system == System::portal) &&
             state.flow < -allowance) ||
            (system == System::venous && state.flow > allowance))
          issue(
            "vessel",
            id,
            "Reference flow opposes the declared arterial/venous orientation.");
      }
    if (!std::isfinite(result.boundary_flow_balance) ||
        !std::isfinite(result.maximum_flow_defect) ||
        !std::isfinite(result.relative_flow_defect) ||
        !std::isfinite(result.dissipated_power) || !std::isfinite(allowance))
      issue("reference",
            invalid_id,
            "Reference flow diagnostics are not representable.");
    phase();
    return result;
  }

  ReferenceResult
  solve_reference(const NetworkData               &network,
                  const std::vector<TerminalPair> &pairs,
                  const ReferenceConditions       &conditions,
                  const ReferenceParameters       &parameters,
                  MPI_Comm                         communicator)
  {
    return solve_reference_impl(
      network, pairs, {}, conditions, parameters, communicator);
  }

  ReferenceResult
  solve_reference(const NetworkData                  &network,
                  const std::vector<TerminalPair>    &pairs,
                  const std::vector<TerminalTriplet> &triplets,
                  const ReferenceConditions          &conditions,
                  const ReferenceParameters          &parameters,
                  MPI_Comm                            communicator)
  {
    return solve_reference_impl(
      network, pairs, triplets, conditions, parameters, communicator);
  }

  static ReferenceConditions
  demand_flow_conditions_impl(const NetworkData &,
                              const std::map<Id, double> &,
                              const std::map<System, double> &,
                              const std::vector<Id> &,
                              MPI_Comm,
                              const std::set<Id> *);

  ReferenceConditions
  demand_flow_conditions(const NetworkData          &network,
                         const std::map<Id, double> &port_pressures,
                         double                      total_flow,
                         const std::vector<Id>      &selected_terminals,
                         MPI_Comm                    communicator)
  {
    return demand_flow_conditions(network,
                                  port_pressures,
                                  {{System::arterial, total_flow},
                                   {System::venous, total_flow}},
                                  selected_terminals,
                                  communicator);
  }

  ReferenceConditions
  demand_flow_conditions(const NetworkData              &network,
                         const std::map<Id, double>     &port_pressures,
                         const std::map<System, double> &total_flows,
                         const std::vector<Id>          &selected_terminals,
                         MPI_Comm                        communicator)
  {
    return demand_flow_conditions_impl(network,
                                       port_pressures,
                                       total_flows,
                                       selected_terminals,
                                       communicator,
                                       nullptr);
  }

  ReferenceConditions
  internal::owned_demand_flow_conditions(
    const NetworkData              &network,
    const std::map<System, double> &total_flows,
    const std::set<Id>             &owned_selected,
    MPI_Comm                        communicator)
  {
    return demand_flow_conditions_impl(
      network, {}, total_flows, {}, communicator, &owned_selected);
  }

  static ReferenceConditions
  demand_flow_conditions_impl(const NetworkData              &network,
                              const std::map<Id, double>     &port_pressures,
                              const std::map<System, double> &total_flows,
                              const std::vector<Id> &selected_terminals,
                              MPI_Comm               communicator,
                              const std::set<Id>    *owned_selected)
  {
    using namespace dealii;
    const Rank rank        = Utilities::MPI::this_mpi_process(communicator),
               size        = Utilities::MPI::n_mpi_processes(communicator);
    const auto require_all = [&](bool valid, const std::string &message) {
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid),
                                      communicator) == 1,
                  ExcMessage(message));
    };
    require_all(
      validate_data(network, size).valid() && !total_flows.empty(),
      "Demand-flow construction requires valid physical data and a positive finite total flow.");
    const auto controls = std::make_pair(total_flows, selected_terminals);
    bool       valid    = true;
    for (const auto &[system, rate] : total_flows)
      valid = valid &&
              (system == System::arterial || system == System::portal ||
               system == System::venous) &&
              std::isfinite(rate) && rate > 0.;
    for (const auto &other : Utilities::MPI::all_gather(communicator, controls))
      valid = other == controls && valid;
    const std::set<Id> requested(selected_terminals.begin(),
                                 selected_terminals.end());
    require_all(
      valid && requested.size() == selected_terminals.size(),
      "Demand-flow totals and terminal selections must agree on all processes; selected ids must be unique.");

    // Resolve port roles and selected identities without replicating the graph
    // or gathering the automatically selected terminal registry.
    using PortInfo = std::pair<Rank, Port>;
    std::map<Rank, std::map<Id, PortInfo>> node_send;
    std::map<Rank, std::vector<Id>>        terminal_send;
    std::set<Id>                           tissue_nodes;
    for (const auto &[id, n] : network.nodes)
      if (n.owner == rank)
        node_send[id % size].emplace(id, PortInfo{rank, n.port});
    for (const auto &[id, t] : network.terminals)
      if (t.owner == rank)
        {
          valid = network.nodes.at(t.node).owner == rank &&
                  tissue_nodes.insert(t.node).second && valid;
          terminal_send[id % size].push_back(id);
        }
    std::map<Id, PortInfo> node_registry;
    for (const auto &[source, nodes] :
         Utilities::MPI::some_to_some(communicator, node_send))
      for (const auto &[id, n] : nodes)
        valid = node_registry.emplace(id, n).second && valid;
    std::set<Id> terminal_registry;
    for (const auto &[source, ids] :
         Utilities::MPI::some_to_some(communicator, terminal_send))
      for (Id id : ids)
        valid = terminal_registry.insert(id).second && valid;
    for (Id id : requested)
      if (id % size == rank)
        valid = terminal_registry.count(id) && valid;
    if (owned_selected)
      for (Id id : *owned_selected)
        {
          const auto terminal = network.terminals.find(id);
          valid               = terminal != network.terminals.end() &&
                  terminal->second.owner == rank && valid;
        }
    require_all(
      valid,
      "Demand-flow identities require unique authoritative nodes and terminals, colocated terminal authority, and existing selected ids.");

    std::map<Rank, std::map<Id, double>> pressure_send;
    for (const auto &[id, p] : port_pressures)
      pressure_send[id % size].emplace(id, p);
    std::map<Id, double> registered_pressures;
    for (const auto &[source, pressures] :
         Utilities::MPI::some_to_some(communicator, pressure_send))
      for (const auto &[id, p] : pressures)
        {
          const auto [it, inserted] = registered_pressures.emplace(id, p);
          valid = std::isfinite(p) && node_registry.count(id) &&
                  (inserted || it->second == p) && valid;
          if (node_registry.count(id))
            valid = node_registry.at(id).second != Port::none && valid;
        }
    require_all(
      valid,
      "Demand-flow pressure data require explicit physical ports and consistent finite pressures.");
    pressure_send.clear();
    for (const auto &[id, p] : registered_pressures)
      pressure_send[node_registry.at(id).first].emplace(id, p);
    ReferenceConditions result;
    for (const auto &[source, pressures] :
         Utilities::MPI::some_to_some(communicator, pressure_send))
      result.pressures.insert(pressures.begin(), pressures.end());

    const auto system_index = [](System system) {
      if (system == System::arterial)
        return 0u;
      else if (system == System::venous)
        return 1u;
      else if (system == System::portal)
        return 2u;
      AssertThrow(false, ExcMessage("Unknown demand-flow system."));
      return 0u;
    };
    std::vector<Id>       selected;
    std::array<double, 3> scale{}, sum{};
    for (const auto &[id, t] : network.terminals)
      if (t.owner == rank &&
          (owned_selected ? owned_selected->count(id) :
                            requested.empty() || requested.count(id)))
        {
          valid = total_flows.count(t.system) && valid;
          selected.push_back(id);
          const auto i = system_index(t.system);
          scale[i]     = std::max(scale[i], t.demand);
        }
    const auto selected_count =
      Utilities::MPI::sum(static_cast<Id>(selected.size()), communicator);
    require_all(
      valid && selected_count > 0,
      "Demand-flow construction requires selected tissue terminals and an explicit positive total for every selected system.");
    Utilities::MPI::max(ArrayView<const double>(scale.data(), scale.size()),
                        communicator,
                        make_array_view(scale));
    for (Id id : selected)
      {
        const auto &t = network.terminals.at(id);
        const auto  i = system_index(t.system);
        sum[i] += t.demand / scale[i];
      }
    Utilities::MPI::sum(ArrayView<const double>(sum.data(), sum.size()),
                        communicator,
                        make_array_view(sum));
    for (Id id : selected)
      {
        const auto &t = network.terminals.at(id);
        const auto  i = system_index(t.system);
        // Separate mantissas and exponents: an intermediate demand ratio may
        // underflow even when the final SI flow is representable.
        int flow_exponent, demand_exponent, scale_exponent, sum_exponent;
        const double mantissa =
          std::frexp(total_flows.at(t.system), &flow_exponent) *
          std::frexp(t.demand, &demand_exponent) /
          (std::frexp(scale[i], &scale_exponent) *
           std::frexp(sum[i], &sum_exponent));
        const double q     = std::ldexp(mantissa,
                                    flow_exponent + demand_exponent -
                                      scale_exponent - sum_exponent);
        valid              = std::isfinite(q) && q > 0. && valid;
        double signed_flow = 0.;
        if (t.system == System::arterial || t.system == System::portal)
          signed_flow = -q;
        else if (t.system == System::venous)
          signed_flow = q;
        valid = result.inflows.emplace(t.node, signed_flow).second && valid;
      }
    require_all(
      valid,
      "Desired tissue flow must be positive and representable, with one terminal identity per node; no demand clipping is applied.");
    return result;
  }

  double
  reference_pressure(const NetworkData     &network,
                     const ReferenceResult &reference,
                     Id                     vessel,
                     double                 coordinate)
  {
    AssertThrow(reference.validation.valid(),
                dealii::ExcMessage(
                  "Cannot evaluate an invalid reference state."));
    AssertThrow(
      std::isfinite(coordinate) && coordinate >= 0. && coordinate <= 1.,
      dealii::ExcMessage("Reference pressure coordinate must lie in [0,1]."));
    const auto &edge = network.vessels.at(vessel);
    if (coordinate == 0.)
      return reference.pressures.at(edge.nodes[0]);
    if (coordinate == 1.)
      return reference.pressures.at(edge.nodes[1]);
    return reference.pressures.at(edge.nodes[0]) -
           reference.vessels.at(vessel).flow *
             resistance(network, vessel, 0., coordinate);
  }
} // namespace ImmersX::Vascularization
