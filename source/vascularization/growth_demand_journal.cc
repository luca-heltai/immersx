// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_demand_journal.h"

#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include <cmath>

#include "growth_ownership_delta.h"
namespace ImmersX::Vascularization::internal
{
  GrowthDemandJournal::GrowthDemandJournal(GrowthDemandState &value,
                                           GrowthWork        *counts)
    : state(value)
    , work(counts)
  {}
  GrowthDemandJournal::~GrowthDemandJournal()
  {
    discard();
  }
  void
  GrowthDemandJournal::save(Id id)
  {
    if (!before.count(id))
      {
        auto found = state.branches.find(id);
        count_work(work,
                   GrowthOperation::metadata_copy_attempts,
                   found == state.branches.end() ? 0 : 1);
        before.emplace(id,
                       found == state.branches.end() ?
                         std::optional<GrowthDemandState::Branch>{} :
                         found->second);
      }
  }
  const GrowthDemandPreview &
  GrowthDemandJournal::preview() const
  {
    return staged;
  }
  const GrowthDemandPreview &
  GrowthDemandJournal::prepare(const GrowthOwnershipDelta    &ownership,
                               const GrowthDemandJointUpdate &joint,
                               const GrowthGeometryMeasures  &measures)
  {
    using namespace dealii;
    discard();
    active          = true;
    batch_size      = ownership.global_splits.size();
    staged          = state.route_demands(joint.update);
    const auto comm = state.communicator;
    const auto rank = Utilities::MPI::this_mpi_process(comm);
    const auto size = Utilities::MPI::n_mpi_processes(comm);
    bool valid = staged.validation.valid() && ownership.validation.valid() &&
                 joint.validation.valid() &&
                 !joint.integration_budget_exhausted;
    auto phase = [&](const char *message) {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1;
      if (!all)
        staged.validation.issues.push_back(
          {"growth journal", invalid_id, message});
      return all;
    };
    if (
      !phase(
        "A physical stage requires validated native edits and joint demands."))
      return prepared();
    std::map<Id, GrowthSplit> splits;
    for (const auto &s : ownership.global_splits)
      valid = splits.emplace(s.vessel, s).second && valid;
    std::map<Id, double> new_demands;
    for (const auto &items :
         Utilities::MPI::all_gather(comm, joint.new_terminals))
      for (const auto &item : items)
        valid = std::isfinite(item.demand) && item.demand > 0. &&
                new_demands.emplace(item.terminal, item.demand).second && valid;
    valid = valid && new_demands.size() == splits.size() &&
            measures.size() == ownership.written.vessels.size();
    for (const auto &[id, s] : splits)
      valid = valid && new_demands.count(s.terminal) &&
              new_demands.at(s.terminal) == s.demand &&
              ownership.vessel_owners.count(s.vessel) &&
              ownership.vessel_owners.count(s.continuation) &&
              ownership.vessel_owners.count(s.leaf) &&
              ownership.node_owners.count(s.tip);
    if (
      !phase(
        "Each physical split needs its exact joint terminal demand and native destination."))
      return prepared();
    std::set<Id> changed;
    for (const auto &[id, demand] : staged.leaf_demands)
      {
        save(id);
        if (state.branches.at(id).leaf_demand != demand)
          changed.insert(id);
        state.branches.at(id).leaf_demand = demand;
      }
    staged.leaf_demands.clear();
    using Branch = GrowthDemandState::Branch;
    std::map<unsigned int, std::map<Id, Branch>> send;
    // Edits to an unselected neighbor: (branch, child) uses invalid_id for
    // a parent change; otherwise it replaces one child-authority slot.
    using Edit = std::array<Id, 4>;
    std::map<unsigned int, std::vector<Edit>> edits;
    std::map<unsigned int, std::map<Id, GrowthDemandState::DirectoryEntry>>
                                                       directories;
    std::map<unsigned int, std::map<Id, unsigned int>> root_owners;
    for (const auto &[id, s] : splits)
      if (s.owner == rank)
        {
          const auto found = state.branches.find(id);
          if (found == state.branches.end())
            {
              valid = false;
              continue;
            }
          bool complete = true;
          for (Id vessel : {s.vessel, s.continuation, s.leaf})
            {
              const auto measure = measures.find(vessel);
              if (measure == measures.end())
                complete = false;
              else
                {
                  const auto &g = measure->second;
                  complete =
                    complete && g.complete &&
                    std::isfinite(g.normalized_length) &&
                    g.normalized_length > 0. && std::isfinite(g.radius_scale) &&
                    g.radius_scale > 0. && std::isfinite(g.clearance_scale) &&
                    g.clearance_scale > 0.;
                }
            }
          valid = valid && complete;
          if (!complete)
            continue;
          count_work(work, GrowthOperation::metadata_copy_attempts, 4);
          const Branch old    = found->second;
          const auto   native = [&](Id vessel) {
            return ownership.vessel_owners.at(vessel);
          };
          Branch original = old, continuation = old, leaf = old;
          original.geometry                = measures.at(s.vessel);
          continuation.geometry            = measures.at(s.continuation);
          leaf.geometry                    = measures.at(s.leaf);
          original.topology.terminal       = invalid_id;
          original.topology.terminal_owner = numbers::invalid_unsigned_int;
          original.topology.children       = {{s.continuation,
                                               native(s.continuation)},
                                              {s.leaf, native(s.leaf)}};
          original.children    = {{s.continuation, {}}, {s.leaf, {}}};
          original.leaf_demand = 0.;
          if (splits.count(old.topology.parent))
            {
              const auto &parent             = splits.at(old.topology.parent);
              original.topology.parent       = parent.continuation;
              original.topology.parent_owner = native(parent.continuation);
            }
          continuation.topology.parent       = id;
          continuation.topology.parent_owner = native(id);
          for (auto &[child, owner] : continuation.topology.children)
            if (splits.count(child))
              owner = native(child);
          leaf.topology.parent       = id;
          leaf.topology.parent_owner = native(id);
          leaf.topology.children.clear();
          leaf.children.clear();
          leaf.topology.terminal       = s.terminal;
          leaf.topology.terminal_owner = ownership.node_owners.at(s.tip);
          leaf.leaf_demand             = s.demand;
          send[native(id)].emplace(id, std::move(original));
          send[native(s.continuation)].emplace(s.continuation,
                                               std::move(continuation));
          send[native(s.leaf)].emplace(s.leaf, std::move(leaf));
          if (old.topology.parent != invalid_id &&
              !splits.count(old.topology.parent))
            edits[old.topology.parent_owner].push_back(
              {{old.topology.parent, id, id, native(id)}});
          for (const auto &[child, owner] : old.topology.children)
            if (!splits.count(child))
              edits[owner].push_back(
                {{child, invalid_id, s.continuation, native(s.continuation)}});
          if (old.topology.children.empty())
            directories[old.topology.terminal % size].emplace(
              old.topology.terminal,
              GrowthDemandState::DirectoryEntry{s.continuation,
                                                native(s.continuation)});
          directories[s.terminal % size].emplace(
            s.terminal,
            GrowthDemandState::DirectoryEntry{s.leaf, native(s.leaf)});
          if (old.topology.parent == invalid_id)
            {
              root_owners[id % size].emplace(id, native(id));
              owned_roots_before.emplace(id, state.owned_roots.count(id));
              state.owned_roots.erase(id);
              // A migrated root no longer contributes to this rank's sum.
              if (native(id) != rank)
                staged.root_sums[{std::numeric_limits<Id>::digits, id}] = {};
            }
          save(id);
          state.branches.erase(id);
          changed.erase(id);
        }
    if (
      !phase(
        "Selected branches need current old writers and complete geometric measures."))
      return prepared();
    const auto   incoming = Utilities::MPI::some_to_some(comm, send);
    std::set<Id> arrived;
    for (const auto &[writer, records] : incoming)
      for (const auto &[id, value] : records)
        {
          const auto vessel = ownership.owned.vessels.find(id);
          valid             = arrived.insert(id).second &&
                  vessel != ownership.owned.vessels.end() && valid;
          if (vessel == ownership.owned.vessels.end())
            continue;
          save(id);
          state.branches[id] = value;
          auto &branch       = state.branches.at(id);
          branch.length      = length(ownership.owned, id);
          changed.insert(id);
          if (branch.topology.parent == invalid_id)
            {
              owned_roots_before.emplace(id, state.owned_roots.count(id));
              state.owned_roots.insert(id);
            }
        }
    valid                     = valid && arrived == ownership.changed_owned;
    const auto received_edits = Utilities::MPI::some_to_some(comm, edits);
    for (const auto &[writer, records] : received_edits)
      for (const auto &edit : records)
        {
          const auto found = state.branches.find(edit[0]);
          if (found == state.branches.end())
            {
              valid = false;
              continue;
            }
          save(edit[0]);
          auto &topology = found->second.topology;
          if (edit[1] == invalid_id)
            {
              topology.parent       = edit[2];
              topology.parent_owner = edit[3];
            }
          else if (topology.children.count(edit[1]))
            topology.children.at(edit[1]) = edit[3];
          else
            valid = false;
        }
    const auto received_directories =
      Utilities::MPI::some_to_some(comm, directories);
    for (const auto &[writer, records] : received_directories)
      for (const auto &[id, value] : records)
        {
          if (!directory_before.count(id))
            {
              const auto old = state.directory.find(id);
              directory_before.emplace(
                id,
                old == state.directory.end() ?
                  std::optional<GrowthDemandState::DirectoryEntry>{} :
                  old->second);
            }
          state.directory[id] = value;
        }
    const auto received_roots = Utilities::MPI::some_to_some(comm, root_owners);
    for (const auto &[writer, records] : received_roots)
      for (const auto &[id, owner] : records)
        {
          if (!roots_before.count(id))
            {
              const auto old = state.root_directory.find(id);
              roots_before.emplace(id,
                                   old == state.root_directory.end() ?
                                     std::optional<unsigned int>{} :
                                     old->second);
            }
          state.root_directory[id] = owner;
        }
    if (
      !phase(
        "Joint topology edits need complete geometry and current neighboring authorities."))
      return prepared();
    staged = state.preview_branches(std::move(staged), changed);
    return prepared();
  }
  const GrowthDemandPreview &
  GrowthDemandJournal::prepared()
  {
    count_work(work,
               GrowthOperation::transaction_records,
               before.size() + directory_before.size() + roots_before.size() +
                 owned_roots_before.size() + staged.owned.size() +
                 staged.root_sums.size());
    count_work(work,
               GrowthOperation::physiology_branch_visits,
               staged.recomputed_branches);
    return staged;
  }
  void
  GrowthDemandJournal::adopt()
  {
    AssertThrow(active && staged.validation.valid(),
                dealii::ExcMessage(
                  "Only a valid physical journal may be adopted."));
    for (const auto &[id, summary] : staged.owned)
      {
        auto &branch        = state.branches.at(id);
        branch.value        = summary;
        const auto children = staged.child_summaries.find(id);
        if (children != staged.child_summaries.end())
          for (const auto &[child, value] : children->second)
            branch.children.at(child) = value;
      }
    for (const auto &[key, value] : staged.root_sums)
      state.root_sums[key] = value;
    state.accepted_totals = staged.totals;
    for (unsigned int system = 0; system < 2; ++system)
      {
        state.accepted_volume[system] = staged.totals[3 * system + 1];
        const long double total       = staged.canonical_total_demand ?
                                          (*staged.canonical_total_demand)[system] :
                                          staged.totals[3 * system];
        state.accepted_resistance[system] =
          total > 0. ? staged.totals[3 * system + 2] / total / total : 0.;
      }
    state.branch_count += 2 * batch_size;
    ++state.revision;
    active = false;
    before.clear();
    directory_before.clear();
    roots_before.clear();
    owned_roots_before.clear();
  }
  void
  GrowthDemandJournal::discard() noexcept
  {
    if (!active)
      return;
    count_work(work,
               GrowthOperation::rollback_records,
               before.size() + directory_before.size() + roots_before.size() +
                 owned_roots_before.size());
    for (auto &[id, value] : before)
      if (value)
        state.branches[id] = std::move(*value);
      else
        state.branches.erase(id);
    for (const auto &[id, value] : directory_before)
      if (value)
        state.directory[id] = *value;
      else
        state.directory.erase(id);
    for (const auto &[id, value] : roots_before)
      if (value)
        state.root_directory[id] = *value;
      else
        state.root_directory.erase(id);
    for (const auto &[id, present] : owned_roots_before)
      if (present)
        state.owned_roots.insert(id);
      else
        state.owned_roots.erase(id);
    before.clear();
    directory_before.clear();
    roots_before.clear();
    owned_roots_before.clear();
    active = false;
  }
} // namespace ImmersX::Vascularization::internal
