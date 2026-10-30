// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "generation_pairs.h"

#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <cmath>
#include <set>

namespace ImmersX::Vascularization::internal
{
  namespace
  {
    using Rank = dealii::types::subdomain_id;
    struct PairEndpoint
    {
      Id     terminal = invalid_id, node = invalid_id, territory = invalid_id;
      Rank   owner  = 0;
      System system = System::arterial;
      dealii::Point<3> position;
      double           demand = 0., desired = 0.;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &terminal &node &territory &owner &system &position &demand &desired;
      }
    };
    struct PairTerritory
    {
      Rank             owner = 0;
      dealii::Point<3> position;
      double           demand = 0.;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &owner &position &demand;
      }
    };
  } // namespace

  void
  pair_territories(GenerationResult &r, const Parameters &p, MPI_Comm comm)
  {
    using namespace dealii;
    const Rank rank = Utilities::MPI::this_mpi_process(comm),
               size = Utilities::MPI::n_mpi_processes(comm);
    const auto fail = [&](Id id, const std::string &message) {
      r.validation.issues.push_back({"territory", id, message});
    };
    const auto valid = [&]() {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(r.validation.valid()),
                            comm) == 1;
      if (!all && r.validation.valid())
        fail(invalid_id, "Terminal pairing failed on another process.");
      return all;
    };
    r.terminal_pairs.clear();
    r.reference = {};
    if (!valid())
      return;
    bool       matching = true;
    const auto controls =
      std::make_pair(p.venous.microvascular_resistance, p.reference_flow);
    for (const auto &other : Utilities::MPI::all_gather(comm, controls))
      matching = matching && other == controls;
    if (!matching || !std::isfinite(controls.first) || controls.first < 0. ||
        !std::isfinite(controls.second) || controls.second < 0. ||
        (controls.first == 0. && controls.second == 0.))
      fail(
        invalid_id,
        "Pairing requires matching finite resistance and reference-flow controls, with a positive supplied resistance or flow.");
    const auto data_report = validate_data(r.network, size);
    r.validation.issues.insert(r.validation.issues.end(),
                               data_report.issues.begin(),
                               data_report.issues.end());
    if (!valid())
      return;

    std::map<Rank, std::map<Id, PairTerritory>> territory_send;
    for (const auto &t : r.territories)
      {
        if (t.id == invalid_id || t.owner != rank || !std::isfinite(t.demand) ||
            !(t.demand > 0.) || !std::isfinite(t.position.norm()))
          fail(
            t.id,
            "Common territories require a unique native writer and finite positive demand.");
        if (!territory_send[t.id % size]
               .emplace(t.id, PairTerritory{t.owner, t.position, t.demand})
               .second)
          fail(t.id, "A common territory may be supplied only once.");
      }
    std::map<Id, PairTerritory> territories;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(comm, territory_send))
      for (const auto &[id, t] : entries)
        if (!territories.emplace(id, t).second)
          fail(id,
               "A common territory may have only one authoritative writer.");
    if (Utilities::MPI::sum(static_cast<Id>(r.territories.size()), comm) == 0)
      fail(invalid_id,
           "Terminal pairing requires at least one common territory.");
    if (!valid())
      return;

    std::map<Rank, std::vector<PairEndpoint>> terminal_send;
    for (const auto &[id, t] : r.network.terminals)
      if (t.owner == rank && t.territory != invalid_id)
        {
          const auto &node = r.network.nodes.at(t.node);
          if (node.owner != rank)
            fail(id, "Terminal authority must follow its physical node.");
          double     desired = 0.;
          const auto it      = r.target_terminal_flows.find(id);
          if (it != r.target_terminal_flows.end())
            desired = it->second;
          terminal_send[t.territory % size].push_back({id,
                                                       t.node,
                                                       t.territory,
                                                       rank,
                                                       t.system,
                                                       node.position,
                                                       t.demand,
                                                       desired});
        }
    std::map<Id, std::vector<PairEndpoint>> registered;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(comm, terminal_send))
      for (const auto &t : entries)
        if (territories.count(t.territory))
          registered[t.territory].push_back(t);
    std::map<Rank, std::vector<PairEndpoint>> endpoints_send;
    std::map<Rank, std::map<Id, Rank>>        selected_send;
    for (const auto &[id, territory] : territories)
      {
        const auto         &endpoints = registered[id];
        const PairEndpoint *a = nullptr, *v = nullptr;
        for (const auto &t : endpoints)
          {
            if (t.system == System::arterial)
              {
                if (a)
                  fail(
                    id,
                    "A common territory has more than one arterial terminal.");
                a = &t;
              }
            else if (t.system == System::venous)
              {
                if (v)
                  fail(id,
                       "A common territory has more than one venous terminal.");
                v = &t;
              }
          }
        if (!a || !v || endpoints.size() != 2)
          {
            fail(
              id,
              "Every common territory requires exactly one arterial and one venous terminal.");
            continue;
          }
        if (a->owner != territory.owner || a->position != territory.position ||
            a->demand != territory.demand || v->demand != territory.demand ||
            a->position == v->position || a->node == v->node ||
            a->terminal == v->terminal)
          fail(
            id,
            "Paired endpoints must be distinct, share the common demand and retain the arterial territory position.");
        if (p.reference_flow > 0. &&
            (!(a->desired > 0.) || !std::isfinite(a->desired)))
          fail(
            id,
            "Every new arterial terminal requires a positive finite desired flow.");
        for (const auto &t : endpoints)
          {
            endpoints_send[territory.owner].push_back(t);
            if (!selected_send[t.owner]
                   .emplace(t.terminal, territory.owner)
                   .second)
              fail(
                t.terminal,
                "A tissue terminal may belong to only one common territory.");
          }
      }
    if (!valid())
      return;
    std::map<Id, std::array<PairEndpoint, 2>> paired;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(comm, endpoints_send))
      for (const auto &t : entries)
        {
          unsigned int slot = 0;
          if (t.system == System::arterial)
            slot = 0;
          else if (t.system == System::venous)
            slot = 1;
          paired[t.territory][slot] = t;
        }
    std::map<Id, Rank> selected;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(comm, selected_send))
      for (const auto &[id, writer] : entries)
        if (!selected.emplace(id, writer).second)
          fail(id,
               "A tissue terminal may belong to only one common territory.");
    if (!valid())
      return;

    std::map<Id, double> pressures;
    if (p.venous.microvascular_resistance == 0.)
      {
        std::map<Rank, std::vector<Id>> external_send, tissue_send;
        for (const auto &[id, flow] : p.reference_conditions.inflows)
          external_send[id % size].push_back(id);
        for (const auto &[id, writer] : selected)
          {
            const Id node = r.network.terminals.at(id).node;
            tissue_send[node % size].push_back(node);
          }
        std::set<Id> external_nodes;
        for (const auto &[source, ids] :
             Utilities::MPI::some_to_some(comm, external_send))
          external_nodes.insert(ids.begin(), ids.end());
        for (const auto &[source, ids] :
             Utilities::MPI::some_to_some(comm, tissue_send))
          for (Id id : ids)
            if (external_nodes.count(id))
              fail(
                id,
                "A paired tissue terminal cannot also have an independent external load.");
        if (!valid())
          return;
        // Keep original boundary data, including unpaired input terminals.
        // Only owned new tissue records enter the independent demand sums.
        auto demand_network = r.network;
        demand_network.terminals.clear();
        for (const auto &[id, writer] : selected)
          demand_network.terminals.emplace(id, r.network.terminals.at(id));
        const auto loads = demand_flow_conditions(
          demand_network, {}, p.reference_flow, {}, comm);
        auto conditions = p.reference_conditions;
        for (const auto &[id, flow] : loads.inflows)
          if (!conditions.inflows.emplace(id, flow).second)
            fail(
              id,
              "A paired tissue terminal cannot also have an independent external load.");
        if (!valid())
          return;
        const auto design = solve_reference(
          r.network, {}, conditions, p.reference_parameters, comm);
        r.validation.issues.insert(r.validation.issues.end(),
                                   design.validation.issues.begin(),
                                   design.validation.issues.end());
        if (!valid())
          return;
        std::map<Rank, std::map<Id, double>> pressure_send;
        for (const auto &[id, writer] : selected)
          pressure_send[writer].emplace(
            id, design.pressures.at(r.network.terminals.at(id).node));
        for (const auto &[source, entries] :
             Utilities::MPI::some_to_some(comm, pressure_send))
          for (const auto &[id, pressure] : entries)
            if (!pressures.emplace(id, pressure).second)
              fail(
                id,
                "A design terminal pressure must have one authoritative owner.");
      }
    for (const auto &[id, endpoints] : paired)
      {
        const auto &a = endpoints[0], &v = endpoints[1];
        double      bed = p.venous.microvascular_resistance;
        if (bed == 0.)
          bed =
            (pressures.at(a.terminal) - pressures.at(v.terminal)) / a.desired;
        if (!std::isfinite(bed) || !(bed > 0.))
          fail(
            id,
            "The requested reference flow and pressures leave no positive microvascular resistance.");
        r.terminal_pairs.push_back({id, a.terminal, v.terminal, bed});
      }
    if (!valid())
      return;
    r.reference = solve_reference(r.network,
                                  r.terminal_pairs,
                                  p.reference_conditions,
                                  p.reference_parameters,
                                  comm);
    r.validation.issues.insert(r.validation.issues.end(),
                               r.reference.validation.issues.begin(),
                               r.reference.validation.issues.end());
    if (!valid())
      return;
    if (p.reference_flow > 0.)
      for (const auto &[id, endpoints] : paired)
        {
          const double desired = endpoints[0].desired;
          const double actual  = r.reference.territories.at(id).flow;
          if (std::abs(actual - desired) >
              p.reference_parameters.absolute_flow_tolerance +
                p.reference_parameters.conservation_relative_tolerance *
                  desired)
            fail(
              id,
              "The supplied bed resistances, pressures and reference flow do not realize the requested tissue-demand split.");
        }
    valid();
  }
} // namespace ImmersX::Vascularization::internal
