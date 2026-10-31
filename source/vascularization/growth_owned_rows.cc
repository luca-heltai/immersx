// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_owned_rows.h"

#include <deal.II/base/mpi.h>

#include <cmath>
#include <limits>
#include <utility>
namespace ImmersX::Vascularization::internal
{
  namespace
  {
    template <class Map>
    void
    remember(Map &current, Map &before, std::set<Id> &changed, Id id)
    {
      if (changed.insert(id).second)
        {
          auto old = current.extract(id);
          if (!old.empty())
            before.insert(std::move(old));
        }
    }
    template <class Map>
    void
    restore(Map &current, Map &before, const std::set<Id> &changed) noexcept
    {
      for (Id id : changed)
        current.erase(id);
      // Both maps have the same allocator; transfer the original nodes intact.
      current.merge(before);
    }
    bool
    same_node(const Node &a, const Node &b)
    {
      return a.position == b.position && a.owner == b.owner &&
             a.port == b.port && a.provenance == b.provenance &&
             a.inward_port_normal == b.inward_port_normal;
    }
    ValidationReport
    failure(Id id, const std::string &message)
    {
      ValidationReport result;
      result.issues.push_back({"growth native rows", id, message});
      return result;
    }
  } // namespace
  GrowthOwnedRows::GrowthOwnedRows(const NetworkData &initial,
                                   MPI_Comm           comm,
                                   GrowthWork        *work)
    : rank(dealii::Utilities::MPI::this_mpi_process(comm))
    , size(dealii::Utilities::MPI::n_mpi_processes(comm))
    , counts(work)
  {
    count_work(counts,
               GrowthOperation::validation_record_visits,
               initial.nodes.size() + initial.vessels.size() +
                 initial.terminals.size());
    bool valid = validate_data(initial, size).valid();
    for (const auto &[id, t] : initial.terminals)
      if (t.owner == rank)
        {
          const auto node = initial.nodes.find(t.node);
          valid =
            node != initial.nodes.end() && node->second.owner == rank && valid;
        }
    AssertThrow(
      dealii::Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1,
      dealii::ExcMessage(
        "Native growth rows require a validated snapshot and terminals at their node owners."));
    for (const auto &[id, n] : initial.nodes)
      if (n.owner == rank)
        rows.nodes.emplace(id, n);
    const auto retain = [&](Id id) {
      rows.nodes.try_emplace(id, initial.nodes.at(id));
      ++references[id];
    };
    for (const auto &[id, v] : initial.vessels)
      if (v.owner == rank)
        {
          rows.vessels.emplace(id, v);
          for (Id node : v.nodes)
            retain(node);
        }
    for (const auto &[id, t] : initial.terminals)
      if (t.owner == rank)
        {
          rows.terminals.emplace(id, t);
          retain(t.node);
        }
    count_work(counts,
               GrowthOperation::physical_copy_attempts,
               rows.nodes.size() + rows.vessels.size() + rows.terminals.size());
  }
  const NetworkData &
  GrowthOwnedRows::network() const noexcept
  {
    return rows;
  }
  std::uint64_t
  GrowthOwnedRows::changed_count() const noexcept
  {
    return changed_nodes.size() + changed_vessels.size() +
           changed_terminals.size() + changed_references.size();
  }
  void
  GrowthOwnedRows::set_reference(Id id, std::uint64_t value)
  {
    remember(references, before_references, changed_references, id);
    if (value == 0)
      references.erase(id);
    else
      references.insert_or_assign(id, value);
  }
  void
  GrowthOwnedRows::add_reference(Id id)
  {
    const auto found = references.find(id);
    const auto value = found == references.end() ? 0 : found->second;
    AssertThrow(value < std::numeric_limits<std::uint64_t>::max(),
                dealii::ExcMessage(
                  "Native endpoint reference count overflows."));
    set_reference(id, value + 1);
  }
  void
  GrowthOwnedRows::remove_reference(Id id)
  {
    const auto found = references.find(id);
    AssertThrow(found != references.end() && found->second > 0,
                dealii::ExcInternalError());
    set_reference(id, found->second - 1);
  }
  void
  GrowthOwnedRows::prune_node(Id id)
  {
    const auto node = rows.nodes.find(id);
    if (node != rows.nodes.end() && node->second.owner != rank &&
        !references.count(id))
      {
        remember(rows.nodes, before.nodes, changed_nodes, id);
        rows.nodes.erase(id);
      }
  }
  ValidationReport
  GrowthOwnedRows::prepare(
    const GrowthOwnershipDelta &delta,
    const std::map<Id, double> &absolute_native_terminal_demands)
  {
    AssertThrow(!active,
                dealii::ExcMessage(
                  "Native growth rows already have a staged update."));
    if (!delta.validation.valid())
      return delta.validation;
    count_work(counts,
               GrowthOperation::validation_record_visits,
               delta.owned.nodes.size() + delta.owned.vessels.size() +
                 delta.owned.terminals.size() +
                 absolute_native_terminal_demands.size());
    auto validation = validate_data(delta.owned, size);
    if (!validation.valid())
      return validation;
    if (delta.changed_owned.size() != delta.owned.vessels.size())
      return failure(invalid_id,
                     "The native changed-vessel set does not match its patch.");
    for (const auto &[id, n] : delta.owned.nodes)
      {
        const auto old = rows.nodes.find(id);
        if (old != rows.nodes.end() && !same_node(old->second, n))
          return failure(
            id,
            "A split changes an existing endpoint's authority or physical properties.");
      }
    for (Id id : delta.released_owned)
      if (!rows.vessels.count(id) || delta.owned.vessels.count(id))
        return failure(
          id,
          "A released vessel is absent or also delivered to its old owner.");
    for (const auto &[id, v] : delta.owned.vessels)
      if (v.owner != rank || !delta.changed_owned.count(id))
        return failure(id,
                       "A vessel patch was delivered to a non-native writer.");
    for (const auto &[id, t] : delta.owned.terminals)
      if (t.owner != rank || delta.owned.nodes.at(t.node).owner != rank ||
          rows.terminals.count(id))
        return failure(
          id,
          "A new terminal lacks native authority or reuses an existing terminal id.");
    for (const auto &[id, demand] : absolute_native_terminal_demands)
      if ((!rows.terminals.count(id) && !delta.owned.terminals.count(id)) ||
          !std::isfinite(demand) || demand <= 0.)
        return failure(
          id,
          "An absolute demand requires a native terminal and finite positive mass.");

    active = true;
    try
      {
        for (const auto &[id, n] : delta.owned.nodes)
          if (!rows.nodes.count(id))
            {
              remember(rows.nodes, before.nodes, changed_nodes, id);
              rows.nodes.emplace(id, n);
            }
        for (Id id : delta.released_owned)
          {
            const auto nodes = rows.vessels.at(id).nodes;
            remember(rows.vessels, before.vessels, changed_vessels, id);
            rows.vessels.erase(id);
            for (Id node : nodes)
              remove_reference(node);
          }
        for (const auto &[id, v] : delta.owned.vessels)
          {
            const auto old = rows.vessels.find(id);
            if (old != rows.vessels.end())
              for (Id node : old->second.nodes)
                remove_reference(node);
            remember(rows.vessels, before.vessels, changed_vessels, id);
            rows.vessels.insert_or_assign(id, v);
            for (Id node : v.nodes)
              add_reference(node);
          }
        for (const auto &[id, t] : delta.owned.terminals)
          {
            remember(rows.terminals, before.terminals, changed_terminals, id);
            rows.terminals.emplace(id, t);
            add_reference(t.node);
          }
        for (const auto &[id, demand] : absolute_native_terminal_demands)
          {
            auto terminal   = rows.terminals.at(id);
            terminal.demand = demand;
            remember(rows.terminals, before.terminals, changed_terminals, id);
            rows.terminals.insert_or_assign(id, terminal);
          }
        for (Id id : changed_references)
          prune_node(id);
        for (const auto &[id, n] : delta.owned.nodes)
          prune_node(id);
        count_work(counts,
                   GrowthOperation::transaction_records,
                   changed_count());
      }
    catch (...)
      {
        discard();
        throw;
      }
    return {};
  }
  void
  GrowthOwnedRows::clear_journal() noexcept
  {
    before.nodes.clear();
    before.vessels.clear();
    before.terminals.clear();
    before_references.clear();
    changed_nodes.clear();
    changed_vessels.clear();
    changed_terminals.clear();
    changed_references.clear();
    active = false;
  }
  void
  GrowthOwnedRows::adopt() noexcept
  {
    Assert(active, dealii::ExcInternalError());
    clear_journal();
  }
  void
  GrowthOwnedRows::discard() noexcept
  {
    if (!active)
      return;
    count_work(counts, GrowthOperation::rollback_records, changed_count());
    restore(rows.nodes, before.nodes, changed_nodes);
    restore(rows.vessels, before.vessels, changed_vessels);
    restore(rows.terminals, before.terminals, changed_terminals);
    restore(references, before_references, changed_references);
    clear_journal();
  }
} // namespace ImmersX::Vascularization::internal
