// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
      Id     node;
      bool   pressure;
      double value;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &node &pressure &value;
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
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &node &anchor &distance &pressure;
      }
    };
  } // namespace

  ReferenceResult
  solve_reference(const NetworkData               &network,
                  const std::vector<TerminalPair> &pairs,
                  const ReferenceConditions       &conditions,
                  const ReferenceParameters       &parameters,
                  MPI_Comm                         communicator)
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
    };
    std::vector<Resistor> resistors;
    const auto            append = [&](Id     a,
                            Id     b,
                            double r,
                            bool   vascular,
                            Id     id,
                            double demand) {
      if (!std::isfinite(r) || r <= 0. || !std::isfinite(1. / r) ||
          1. / r <= 0.)
        issue(
          "resistance",
          id,
          "Resistance and conductance must be positive and representable; no radius clipping is applied.");
      else
        resistors.push_back(
          {{{nodes.at(a), nodes.at(b)}}, r, vascular, id, demand});
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
      boundary_send[id % size].push_back({id, true, value});
    for (const auto &[id, value] : conditions.inflows)
      boundary_send[id % size].push_back({id, false, value});
    ReferenceConditions registered;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(communicator, boundary_send))
      for (const auto &b : entries)
        {
          auto &values = b.pressure ? registered.pressures : registered.inflows;
          const auto [it, inserted] = values.emplace(b.node, b.value);
          if (!std::isfinite(b.value) || !node_registry.count(b.node) ||
              (!inserted && it->second != b.value))
            issue(
              "boundary",
              b.node,
              "Boundary data require a known physical node and consistent finite SI values.");
        }
    boundary_send.clear();
    for (const auto &[id, value] : registered.pressures)
      {
        if (registered.inflows.count(id))
          issue("boundary",
                id,
                "Prescribe either pressure or flow at a node, never both.");
        if (node_registry.count(id))
          boundary_send[node_registry.at(id).owner].push_back(
            {id, true, value});
      }
    for (const auto &[id, value] : registered.inflows)
      if (node_registry.count(id))
        boundary_send[node_registry.at(id).owner].push_back({id, false, value});
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
          auto &values = b.pressure ? owned_bc.pressures : owned_bc.inflows;
          values.emplace(b.node, b.value);
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
    std::vector<Id> anchors(local_nodes, invalid_id);
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
    const auto better = [](long double distance,
                           Id          anchor,
                           long double old_distance,
                           Id          old_anchor) {
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
                    if (better(candidate, origin, distances[j], anchors[j]))
                      {
                        distances[j]             = candidate;
                        anchors[j]               = origin;
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
                                                        found->second.distance,
                                                        found->second.anchor))
                      packet[target.id] = {target.id,
                                           origin,
                                           candidate,
                                           lifting.local_element(i)};
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
                                                         distances[i],
                                                         anchors[i]))
                {
                  distances[i]             = proposal.distance;
                  anchors[i]               = proposal.anchor;
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
          "Each reference component needs a pressure anchor at a representable resistive distance.");
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
            for (const auto &arc : rows[i])
              product += (value - (fixed_values[arc.neighbor.index] != 0. ?
                                                    0. :
                                                    work_values[arc.neighbor.index])) /
                         arc.resistance;
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
        else
          {
            result.territories[r.id] = {q, q / r.demand};
            if (!std::isfinite(q / r.demand) || !(q > 0.))
              issue(
                "territory",
                r.id,
                "Microvascular perfusion must be positive and representable from artery to vein.");
          }
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
            const double prescribed =
              owned_bc.inflows.count(id) ? owned_bc.inflows.at(id) : 0.;
            flow_scale = std::max(flow_scale, std::abs(prescribed));
            result.maximum_flow_defect =
              std::max(result.maximum_flow_defect,
                       std::abs(outflow - prescribed));
            if (owned_bc.inflows.count(id))
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
        if ((system == System::arterial && state.flow < -allowance) ||
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

  ReferenceConditions
  demand_flow_conditions(const NetworkData          &network,
                         const std::map<Id, double> &port_pressures,
                         double                      total_flow,
                         const std::vector<Id>      &selected_terminals,
                         MPI_Comm                    communicator)
  {
    using namespace dealii;
    AssertThrow(Utilities::MPI::n_mpi_processes(communicator) == 1,
                ExcMessage(
                  "The current demand-flow construction requires one rank."));
    AssertThrow(
      validate_data(network, 1).valid() && std::isfinite(total_flow) &&
        total_flow > 0.,
      ExcMessage(
        "Demand-flow construction requires valid physical data and a positive finite total flow."));
    ReferenceConditions result;
    for (const auto &[id, p] : port_pressures)
      {
        const auto node = network.nodes.find(id);
        AssertThrow(
          node != network.nodes.end() && node->second.port != Port::none &&
            std::isfinite(p),
          ExcMessage(
            "Demand-flow pressure data require explicit physical port nodes and finite pressures."));
        result.pressures[id] = p;
      }
    std::vector<Id> selected = selected_terminals;
    if (selected.empty())
      for (const auto &[id, t] : network.terminals)
        selected.push_back(id);
    AssertThrow(
      !selected.empty(),
      ExcMessage(
        "Demand-flow construction requires at least one selected tissue terminal."));
    std::set<Id>          unique;
    std::array<double, 2> scale{{0., 0.}}, sum{{0., 0.}};
    const auto            system_index = [](System system) {
      if (system == System::arterial)
        return 0u;
      else if (system == System::venous)
        return 1u;
      AssertThrow(false, ExcMessage("Unknown demand-flow system."));
      return 0u;
    };
    for (Id id : selected)
      {
        AssertThrow(
          network.terminals.count(id) && unique.insert(id).second,
          ExcMessage(
            "Selected demand-flow terminal ids must exist and be unique."));
        const auto &t = network.terminals.at(id);
        const auto  i = system_index(t.system);
        scale[i]      = std::max(scale[i], t.demand);
      }
    for (Id id : selected)
      {
        const auto &t = network.terminals.at(id);
        const auto  i = system_index(t.system);
        sum[i] += t.demand / scale[i];
      }
    for (Id id : selected)
      {
        const auto  &t = network.terminals.at(id);
        const auto   i = system_index(t.system);
        const double q = total_flow * ((t.demand / scale[i]) / sum[i]);
        AssertThrow(
          std::isfinite(q) && q > 0.,
          ExcMessage(
            "Desired tissue flow is not representable; no demand clipping is applied."));
        double signed_flow = 0.;
        if (t.system == System::arterial)
          signed_flow = -q;
        else if (t.system == System::venous)
          signed_flow = q;
        AssertThrow(
          result.inflows.emplace(t.node, signed_flow).second,
          ExcMessage(
            "A demand-flow terminal node must have one terminal identity."));
      }
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
