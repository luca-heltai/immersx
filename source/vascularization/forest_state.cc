// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "forest_state.h"

#include <deal.II/base/numbers.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/generation.h>

#include <cmath>

namespace ImmersX::Vascularization::internal
{
  namespace
  {
    using Rank = unsigned int;
    unsigned int
    index(System system)
    {
      if (system == System::arterial)
        return 0;
      else if (system == System::venous)
        return 1;
      AssertThrow(false, dealii::ExcMessage("Unknown vascular forest system."));
      return 0;
    }
    struct Arc
    {
      Id     node, vessel;
      bool   start, root;
      System system;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &node &vessel &start &root &system;
      }
    };
    struct Packet
    {
      std::vector<Arc> arcs;
      std::vector<Id>  registry;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &arcs &registry;
      }
    };
    struct EndInfo
    {
      Id                 vessel, parent = invalid_id, terminal = invalid_id;
      Rank               parent_owner = dealii::numbers::invalid_unsigned_int;
      bool               start;
      std::map<Id, Rank> children;
      double             leaf_demand = 0.;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &vessel &parent &terminal &parent_owner &start &children
          &leaf_demand;
      }
    };
    struct Up
    {
      Id            parent, child;
      double        demand, volume_moment, resistance_moment, minimum_demand;
      std::uint64_t count;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &parent &child &demand &volume_moment &resistance_moment
          &minimum_demand &count;
      }
    };
    struct Down
    {
      Id     target, root;
      double demand, radius, volume_moment, resistance_moment, minimum_demand;
      System system;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &target &root &demand &radius &volume_moment &resistance_moment
          &minimum_demand &system;
      }
    };
  } // namespace

  ForestState
  evaluate_forest(const NetworkData          &network,
                  const std::set<Id>         &mutable_owned,
                  const std::map<Id, double> &fixed_owned_roots,
                  const Parameters           &parameters,
                  MPI_Comm                    comm,
                  const MeshLengthScale      &scale)
  {
    using namespace dealii;
    const Rank rank = Utilities::MPI::this_mpi_process(comm),
               size = Utilities::MPI::n_mpi_processes(comm);
    ForestState result;
    const auto  phase = [&](bool local, const std::string &message) {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(local), comm) == 1;
      if (!all)
        result.validation.issues.push_back({"forest", invalid_id, message});
      return all;
    };
    const auto positive = [](double x) { return std::isfinite(x) && x > 0.; };
    bool       valid_scale             = true;
    double     physical_minimum_radius = 0.;
    try
      {
        scale.validate();
        valid_scale = scale.name() == parameters.length_scale;
        physical_minimum_radius =
          scale.physical_length(parameters.minimum_radius, scale.minimum());
      }
    catch (const std::exception &)
      {
        valid_scale = false;
      }
    bool valid =
      valid_scale && validate_data(network, size).valid() &&
      positive(parameters.arterial_murray_exponent) &&
      positive(parameters.venous.murray_exponent) &&
      std::isfinite(parameters.minimum_radius) &&
      parameters.minimum_radius >= 0. &&
      std::isfinite(parameters.reference_flow) &&
      parameters.reference_flow >= 0. &&
      std::isfinite(parameters.dissipation_weight) &&
      parameters.dissipation_weight >= 0. &&
      parameters.dissipation_weight <= 1. &&
      (parameters.dissipation_weight == 0. || parameters.reference_flow > 0.);
    for (Id id : mutable_owned)
      valid = valid && network.vessels.count(id) &&
              network.vessels.at(id).owner == rank;
    for (const auto &[id, r] : fixed_owned_roots)
      valid = valid && mutable_owned.count(id) && positive(r) &&
              r >= physical_minimum_radius &&
              network.vessels.at(id).radii == std::array<double, 2>{{r, r}};
    for (const auto &[id, t] : network.terminals)
      if (t.owner == rank)
        valid = valid && network.nodes.at(t.node).owner == rank;
    if (
      !phase(
        valid,
        "Forest evaluation requires valid owned mutable ids, fixed roots, colocated terminal authority and finite physiological controls."))
      return result;
    const std::array<double, 9> controls{{parameters.arterial_murray_exponent,
                                          parameters.venous.murray_exponent,
                                          parameters.minimum_radius,
                                          parameters.reference_flow,
                                          parameters.dissipation_weight,
                                          static_cast<double>(scale.policy),
                                          scale.mesh_minimum,
                                          scale.mesh_maximum,
                                          physical_minimum_radius}};
    for (const auto &other : Utilities::MPI::all_gather(comm, controls))
      valid = valid && other == controls;
    if (!phase(valid,
               "All forest owners must use the same physiological controls."))
      return result;
    const auto n =
      Utilities::MPI::sum(static_cast<std::uint64_t>(mutable_owned.size()),
                          comm);
    const auto n_roots =
      Utilities::MPI::sum(static_cast<std::uint64_t>(fixed_owned_roots.size()),
                          comm);
    if (!phase(n == 0 || n_roots > 0,
               "A nonempty generated forest requires an explicit fixed root."))
      return result;
    if (n == 0)
      return result;
    const auto gamma = [&](System s) {
      return index(s) == 0 ? parameters.arterial_murray_exponent :
                             parameters.venous.murray_exponent;
    };
    std::map<Rank, Packet> send;
    for (Id id : mutable_owned)
      {
        const auto &v = network.vessels.at(id);
        send[id % size].registry.push_back(id);
        for (unsigned int end = 0; end < 2; ++end)
          send[network.nodes.at(v.nodes[end]).owner].arcs.push_back(
            {v.nodes[end],
             id,
             end == 0,
             fixed_owned_roots.count(id) > 0,
             v.system});
        result.owned.emplace(id, ForestBranch{});
      }
    const auto received = Utilities::MPI::some_to_some(comm, send);
    struct Incident
    {
      Rank   owner;
      bool   root;
      System system;
    };
    struct NodeLinks
    {
      std::map<Id, Incident> incoming, outgoing;
    };
    std::map<Id, NodeLinks> links;
    std::set<Id>            registry;
    for (const auto &[owner, packet] : received)
      {
        for (Id id : packet.registry)
          valid = registry.insert(id).second && valid;
        for (const auto &a : packet.arcs)
          {
            const auto node = network.nodes.find(a.node);
            if (node == network.nodes.end() || node->second.owner != rank)
              {
                valid = false;
                continue;
              }
            auto &l    = links[a.node];
            auto &arcs = a.start ? l.outgoing : l.incoming;
            valid = arcs.emplace(a.vessel, Incident{owner, a.root, a.system})
                      .second &&
                    valid;
          }
      }
    std::map<Id, std::vector<std::pair<Id, Terminal>>> terminals;
    for (const auto &[id, t] : network.terminals)
      if (t.owner == rank)
        terminals[t.node].emplace_back(id, t);
    std::map<Rank, std::vector<EndInfo>> infos;
    for (const auto &[node, l] : links)
      {
        valid = valid && l.incoming.size() <= 1;
        for (const auto &[id, a] : l.outgoing)
          {
            EndInfo info;
            info.vessel = id;
            info.start  = true;
            if (a.root)
              valid = valid && l.incoming.empty();
            else if (l.incoming.size() != 1)
              valid = false;
            else
              {
                const auto &[parent, p] = *l.incoming.begin();
                info.parent             = parent;
                info.parent_owner       = p.owner;
                valid                   = valid && p.system == a.system;
              }
            infos[a.owner].push_back(info);
          }
        for (const auto &[id, a] : l.incoming)
          {
            EndInfo info;
            info.vessel = id;
            info.start  = false;
            if (l.outgoing.empty())
              {
                const auto t = terminals.find(node);
                if (t == terminals.end() || t->second.size() != 1)
                  valid = false;
                else
                  {
                    info.terminal = t->second[0].first;
                    result.owned_leaf_terminals.insert(info.terminal);
                    info.leaf_demand = t->second[0].second.demand;
                    valid = valid && t->second[0].second.system == a.system;
                  }
              }
            else
              {
                valid =
                  valid && l.outgoing.size() <= 2 && !terminals.count(node);
                for (const auto &[child, c] : l.outgoing)
                  {
                    info.children[child] = c.owner;
                    valid                = valid && c.system == a.system;
                  }
              }
            infos[a.owner].push_back(info);
          }
      }
    if (
      !phase(
        valid,
        "Mutable vessels require unique ids, rooted connectivity with at most two children, matching systems and one registered terminal per leaf."))
      return result;
    const auto   neighborhoods = Utilities::MPI::some_to_some(comm, infos);
    std::set<Id> starts, ends;
    double       leaf_scale = 0.;
    for (const auto &[node_owner, rows] : neighborhoods)
      for (const auto &info : rows)
        {
          if (!result.owned.count(info.vessel))
            {
              valid = false;
              continue;
            }
          auto &b = result.owned.at(info.vessel);
          if (info.start)
            {
              valid          = starts.insert(info.vessel).second && valid;
              b.parent       = info.parent;
              b.parent_owner = info.parent_owner;
            }
          else
            {
              valid            = ends.insert(info.vessel).second && valid;
              b.children       = info.children;
              b.terminal       = info.terminal;
              b.terminal_owner = node_owner;
              b.demand         = info.leaf_demand;
              leaf_scale       = std::max(leaf_scale, info.leaf_demand);
            }
        }
    valid = valid && starts.size() == mutable_owned.size() &&
            ends.size() == mutable_owned.size();
    result.demand_scale = Utilities::MPI::max(leaf_scale, comm);
    valid               = valid && positive(result.demand_scale);
    if (
      !phase(
        valid,
        "Forest endpoint metadata or the global tissue-demand scale is invalid."))
      return result;

    std::vector<Id> ready;
    for (auto &[id, b] : result.owned)
      if (b.children.empty())
        {
          b.demand /= result.demand_scale;
          b.terminals      = 1;
          b.minimum_demand = b.demand;
          ready.push_back(id);
          valid = valid && positive(b.demand);
        }
    if (
      !phase(
        valid,
        "A relative leaf demand is not representable; no demand clipping is applied."))
      return result;
    std::map<Id, std::map<Id, Up>> child_values;
    std::uint64_t                  processed = 0;
    while (Utilities::MPI::sum(static_cast<std::uint64_t>(ready.size()), comm) >
           0)
      {
        ++result.upward_levels;
        std::map<Rank, std::vector<Up>> packets;
        for (Id id : ready)
          {
            auto        &b = result.owned.at(id);
            const auto  &v = network.vessels.at(id);
            const double l = length(network, id), p = 2. / gamma(v.system),
                         q = 2. - 4. / gamma(v.system);
            b.volume_moment += l * std::pow(b.demand, p);
            if (parameters.dissipation_weight > 0.)
              b.resistance_moment += v.viscosity * l * std::pow(b.demand, q);
            valid = valid && positive(b.demand) && positive(b.volume_moment) &&
                    (parameters.dissipation_weight == 0. ||
                     positive(b.resistance_moment));
            ++processed;
            if (b.parent != invalid_id)
              packets[b.parent_owner].push_back({b.parent,
                                                 id,
                                                 b.demand,
                                                 b.volume_moment,
                                                 b.resistance_moment,
                                                 b.minimum_demand,
                                                 b.terminals});
          }
        const auto messages = Utilities::MPI::some_to_some(comm, packets);
        ready.clear();
        std::set<Id> touched;
        for (const auto &[owner, items] : messages)
          for (const auto &u : items)
            {
              const auto parent = result.owned.find(u.parent);
              if (parent == result.owned.end() ||
                  !parent->second.children.count(u.child) ||
                  parent->second.children.at(u.child) != owner)
                {
                  valid = false;
                  continue;
                }
              valid =
                child_values[u.parent].emplace(u.child, u).second && valid;
              touched.insert(u.parent);
            }
        for (Id id : touched)
          {
            const auto &values = child_values.at(id);
            auto       &b      = result.owned.at(id);
            if (b.terminals == 0 && values.size() == b.children.size())
              {
                double maximum_child = 0.;
                for (const auto &[child, u] : values)
                  {
                    b.demand += u.demand;
                    b.terminals += u.count;
                    b.volume_moment += u.volume_moment;
                    b.resistance_moment += u.resistance_moment;
                    b.minimum_demand =
                      std::min(b.minimum_demand, u.minimum_demand);
                    maximum_child = std::max(maximum_child, u.demand);
                  }
                if (values.size() == 1)
                  valid = valid && b.demand == maximum_child;
                else if (values.size() == 2)
                  valid = valid && b.demand > maximum_child;
                else
                  valid = false;
                ready.push_back(id);
              }
          }
        if (
          !phase(
            valid && result.upward_levels <= n,
            "Subtree demand/moment aggregation is inconsistent or not representable."))
          return result;
      }
    if (
      !phase(
        Utilities::MPI::sum(processed, comm) == n,
        "A mutable component is cyclic or does not reach a declared fixed root."))
      return result;
    std::array<double, 2> local_demands{{0., 0.}};
    std::array<double, 2> local_power_numerator{{0., 0.}};
    std::vector<Down>     down;
    for (const auto &[id, radius] : fixed_owned_roots)
      {
        const auto &b      = result.owned.at(id);
        const auto  system = network.vessels.at(id).system;
        local_demands[index(system)] += b.demand;
        if (parameters.dissipation_weight > 0.)
          {
            const double numerator = 8. / numbers::PI / std::pow(radius, 4) *
                                     b.resistance_moment *
                                     std::pow(b.demand, 4. / gamma(system));
            valid = valid && positive(numerator);
            local_power_numerator[index(system)] += numerator;
          }
        down.push_back({id,
                        id,
                        b.demand,
                        radius,
                        b.volume_moment,
                        b.resistance_moment,
                        b.minimum_demand,
                        system});
      }
    for (unsigned int i = 0; i < 2; ++i)
      {
        result.total_demand[i] = Utilities::MPI::sum(local_demands[i], comm);
        result.power_numerator[i] =
          Utilities::MPI::sum(local_power_numerator[i], comm);
        valid = valid && std::isfinite(result.power_numerator[i]);
      }
    if (!phase(valid,
               "The normalized forest power numerator is not representable."))
      return result;
    std::set<Id> normalized;
    while (Utilities::MPI::sum(static_cast<std::uint64_t>(down.size()), comm) >
           0)
      {
        ++result.downward_levels;
        std::map<Rank, std::vector<Down>> packets;
        for (const auto &d : down)
          {
            const auto found = result.owned.find(d.target);
            if (found == result.owned.end() ||
                !normalized.insert(d.target).second)
              {
                valid = false;
                continue;
              }
            auto       &b            = found->second;
            const auto &v            = network.vessels.at(d.target);
            valid                    = valid && d.system == v.system;
            b.root                   = d.root;
            b.root_demand            = d.demand;
            b.root_radius            = d.radius;
            b.root_volume_moment     = d.volume_moment;
            b.root_resistance_moment = d.resistance_moment;
            b.root_minimum_demand    = d.minimum_demand;
            b.radius =
              d.target == d.root ?
                d.radius :
                d.radius * std::pow(b.demand / d.demand, 1. / gamma(v.system));
            valid = valid && positive(b.radius) &&
                    b.radius >= physical_minimum_radius;
            for (const auto &[child, owner] : b.children)
              {
                auto message   = d;
                message.target = child;
                packets[owner].push_back(message);
              }
          }
        const auto messages = Utilities::MPI::some_to_some(comm, packets);
        down.clear();
        for (const auto &[owner, items] : messages)
          down.insert(down.end(), items.begin(), items.end());
        if (
          !phase(
            valid && result.downward_levels <= n,
            "Fixed-root normalization produced invalid radii, systems or dependencies."))
          return result;
      }
    if (!phase(normalized.size() == mutable_owned.size(),
               "Some mutable vessels lack a root normalization."))
      return result;
    std::array<double, 2> local_volume{{0., 0.}}, local_power{{0., 0.}};
    std::map<Rank, std::map<Id, double>> leaf_flows;
    for (auto &[id, b] : result.owned)
      {
        const auto  &v      = network.vessels.at(id);
        const auto   i      = index(v.system);
        const double l      = length(network, id);
        const double volume = numbers::PI * b.radius * b.radius * l;
        local_volume[i] += volume;
        valid = valid && positive(volume) && positive(result.total_demand[i]);
        if (parameters.reference_flow > 0.)
          {
            b.flow =
              parameters.reference_flow * (b.demand / result.total_demand[i]);
            b.resistance =
              8. * v.viscosity * l / (numbers::PI * std::pow(b.radius, 4));
            const double power = b.flow * (b.resistance * b.flow);
            valid = valid && positive(b.flow) && positive(b.resistance) &&
                    std::isfinite(power);
            local_power[i] += power;
            if (b.terminal != invalid_id)
              leaf_flows[b.terminal_owner].emplace(b.terminal, b.flow);
          }
      }
    const auto flows = Utilities::MPI::some_to_some(comm, leaf_flows);
    for (const auto &[owner, values] : flows)
      for (const auto &[id, q] : values)
        valid = result.terminal_flows.emplace(id, q).second && valid;
    for (unsigned int i = 0; i < 2; ++i)
      {
        result.volume[i] = Utilities::MPI::sum(local_volume[i], comm);
        result.power[i]  = Utilities::MPI::sum(local_power[i], comm);
        valid            = valid && std::isfinite(result.volume[i]) &&
                std::isfinite(result.power[i]) &&
                std::isfinite(result.total_demand[i]);
      }
    phase(
      valid,
      "Normalized vascular volume, resistance or desired flow is not representable.");
    return result;
  }
} // namespace ImmersX::Vascularization::internal
