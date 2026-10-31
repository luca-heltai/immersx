// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_demand_state.h"

#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/generation.h>

namespace ImmersX::Vascularization::internal
{
  namespace
  {
    bool
    phase(ValidationReport  &report,
          bool               valid,
          MPI_Comm           comm,
          const std::string &message)
    {
      const bool all =
        dealii::Utilities::MPI::min(static_cast<unsigned int>(valid), comm) ==
        1;
      if (!all)
        report.issues.push_back({"demand state", invalid_id, message});
      return all;
    }
    struct ChildValue
    {
      Id                  parent, child;
      GrowthDemandSummary value;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &parent &child &value;
      }
    };
    bool
    positive(long double value)
    {
      return std::isfinite(value) && value > 0.;
    }
    bool
    valid_summary(const GrowthDemandSummary &value)
    {
      return value.mass > 0 && value.leaves > 0 &&
             positive(value.volume_length) && positive(value.minimum_ratio) &&
             std::isfinite(value.resistance_length) &&
             value.resistance_length >= 0.;
    }
  } // namespace

  GrowthDemandSummary
  GrowthDemandState::aggregate(
    const Branch                            &branch,
    double                                   leaf,
    const std::map<Id, GrowthDemandSummary> &children)
  {
    GrowthDemandSummary result;
    result.volume_length     = branch.length;
    result.resistance_length = branch.viscosity * branch.length;
    if (children.empty())
      {
        result.mass   = demand_mass_units(leaf);
        result.leaves = 1;
        return result;
      }
    for (const auto &[id, child] : children)
      {
        result.mass += child.mass;
        result.leaves += child.leaves;
      }
    for (const auto &[id, child] : children)
      {
        const boost::multiprecision::cpp_rational ratio_exact =
          boost::multiprecision::cpp_rational(child.mass) / result.mass;
        const long double ratio = ratio_exact.convert_to<long double>();
        result.volume_length +=
          std::pow(ratio, 2.L / branch.gamma) * child.volume_length;
        result.resistance_length +=
          std::pow(ratio, 2.L - 4.L / branch.gamma) * child.resistance_length;
        result.minimum_ratio =
          std::min(result.minimum_ratio,
                   std::pow(ratio, 1.L / branch.gamma) * child.minimum_ratio);
      }
    return result;
  }

  GrowthDemandState::GrowthDemandState(const NetworkData &network,
                                       const ForestState &forest,
                                       const Parameters  &parameters,
                                       std::uint64_t      snapshot,
                                       MPI_Comm           comm)
    : communicator(comm)
    , revision(snapshot)
    , report(forest.validation)
  {
    using namespace dealii;
    const auto rank  = Utilities::MPI::this_mpi_process(comm);
    const auto size  = Utilities::MPI::n_mpi_processes(comm);
    bool       valid = report.valid();
    std::map<unsigned int, std::vector<std::pair<Id, Id>>> requests;
    std::map<unsigned int, std::map<Id, DirectoryEntry>>   registrations;
    for (const auto &[id, topology] : forest.owned)
      {
        const auto vessel = network.vessels.find(id);
        if (vessel == network.vessels.end() || vessel->second.owner != rank)
          {
            valid = false;
            continue;
          }
        Branch branch;
        branch.topology = topology;
        branch.length   = length(network, id);
        branch.viscosity =
          parameters.dissipation_weight > 0. ? vessel->second.viscosity : 0.;
        if (vessel->second.system == System::arterial)
          branch.gamma = parameters.arterial_murray_exponent;
        else if (vessel->second.system == System::venous)
          {
            branch.gamma  = parameters.venous.murray_exponent;
            branch.system = 1;
          }
        else
          valid = false;
        if (topology.children.empty())
          {
            valid = valid && topology.terminal != invalid_id &&
                    topology.terminal_owner < size;
            if (topology.terminal_owner < size)
              requests[topology.terminal_owner].push_back(
                {id, topology.terminal});
            registrations[topology.terminal % size].emplace(
              topology.terminal, DirectoryEntry{id, rank});
          }
        branches.emplace(id, std::move(branch));
      }
    branch_count =
      Utilities::MPI::sum(static_cast<std::uint64_t>(branches.size()), comm);
    if (!phase(report,
               valid,
               comm,
               "Demand bootstrap needs a valid authoritative forest."))
      return;
    const auto registered = Utilities::MPI::some_to_some(comm, registrations);
    for (const auto &[owner, records] : registered)
      for (const auto &[id, entry] : records)
        valid =
          entry.owner == owner && directory.emplace(id, entry).second && valid;
    const auto incoming = Utilities::MPI::some_to_some(comm, requests);
    std::map<unsigned int, std::map<Id, double>> replies;
    for (const auto &[owner, items] : incoming)
      for (const auto &[branch, terminal] : items)
        {
          const auto t = network.terminals.find(terminal);
          if (t == network.terminals.end() || t->second.owner != rank ||
              !positive(t->second.demand))
            {
              valid = false;
              continue;
            }
          replies[owner].emplace(branch, t->second.demand);
        }
    if (!phase(report,
               valid,
               comm,
               "Native leaf demands must be uniquely registered and positive."))
      return;
    const auto   received = Utilities::MPI::some_to_some(comm, replies);
    std::set<Id> ready;
    for (const auto &[owner, items] : received)
      for (const auto &[id, demand] : items)
        {
          const auto b = branches.find(id);
          if (b == branches.end() || !b->second.topology.children.empty() ||
              b->second.topology.terminal_owner != owner ||
              !ready.insert(id).second)
            {
              valid = false;
              continue;
            }
          b->second.leaf_demand = demand;
        }
    if (!phase(
          report,
          valid,
          comm,
          "Every native mutable leaf needs its authoritative absolute demand."))
      return;
    std::uint64_t processed = 0;
    while (Utilities::MPI::sum(static_cast<std::uint64_t>(ready.size()), comm) >
           0)
      {
        std::map<unsigned int, std::vector<ChildValue>> packets;
        for (Id id : ready)
          {
            auto &b = branches.at(id);
            b.value = aggregate(b, b.leaf_demand, b.children);
            valid   = valid_summary(b.value) && valid;
            ++processed;
            if (b.topology.parent != invalid_id)
              packets[b.topology.parent_owner].push_back(
                {b.topology.parent, id, b.value});
          }
        const auto messages = Utilities::MPI::some_to_some(comm, packets);
        ready.clear();
        for (const auto &[owner, items] : messages)
          for (const auto &item : items)
            {
              const auto b = branches.find(item.parent);
              if (b == branches.end() ||
                  !b->second.topology.children.count(item.child) ||
                  b->second.topology.children.at(item.child) != owner ||
                  !b->second.children.emplace(item.child, item.value).second)
                {
                  valid = false;
                  continue;
                }
              if (b->second.children.size() ==
                  b->second.topology.children.size())
                ready.insert(item.parent);
            }
        if (!phase(
              report,
              valid && processed <= branch_count,
              comm,
              "Demand bootstrap requires finite positive subtree summaries."))
          return;
      }
    if (!phase(
          report,
          Utilities::MPI::sum(processed, comm) == branch_count,
          comm,
          "Demand bootstrap must reach every accepted branch exactly once."))
      return;
    for (const auto &[id, b] : branches)
      if (b.topology.parent == invalid_id)
        root_positions.emplace(id, root_positions.size());
    while (root_capacity < root_positions.size())
      root_capacity *= 2;
    root_sums.resize(2 * root_capacity);
    for (const auto &[id, position] : root_positions)
      root_sums[root_capacity + position] =
        root_value(id, branches.at(id).value);
    for (std::size_t i = root_capacity; i-- > 1;)
      for (unsigned int j = 0; j < 6; ++j)
        root_sums[i][j] = root_sums[2 * i][j] + root_sums[2 * i + 1][j];
    for (unsigned int j = 0; j < 6; ++j)
      {
        accepted_totals[j] = Utilities::MPI::sum(root_sums[1][j], comm);
        valid = std::isfinite(accepted_totals[j]) && accepted_totals[j] >= 0. &&
                valid;
      }
    phase(report,
          valid,
          comm,
          "Accepted root aggregates must be finite and nonnegative.");
  }

  std::array<long double, 6>
  GrowthDemandState::root_value(Id id, const GrowthDemandSummary &value) const
  {
    const auto                               &branch = branches.at(id);
    const boost::multiprecision::cpp_rational physical_mass =
      boost::multiprecision::cpp_rational(value.mass) /
      (boost::multiprecision::cpp_int(1) << 1074);
    const long double          mass   = physical_mass.convert_to<long double>();
    const long double          radius = branch.topology.root_radius;
    std::array<long double, 6> result{};
    result[3 * branch.system] = mass;
    result[3 * branch.system + 1] =
      dealii::numbers::PI * radius * radius * value.volume_length;
    result[3 * branch.system + 2] = 8.L / dealii::numbers::PI /
                                    std::pow(radius, 4) *
                                    value.resistance_length * mass * mass;
    return result;
  }
  const std::array<long double, 6> &
  GrowthDemandState::totals() const
  {
    return accepted_totals;
  }
  const ValidationReport &
  GrowthDemandState::validation() const
  {
    return report;
  }
  const GrowthDemandSummary &
  GrowthDemandState::summary(Id id) const
  {
    return branches.at(id).value;
  }

  const GrowthDemandSummary &
  GrowthDemandState::summary(Id id, const GrowthDemandPreview &preview) const
  {
    const auto found = preview.owned.find(id);
    return found == preview.owned.end() ? summary(id) : found->second;
  }
  const ForestBranch &
  GrowthDemandState::topology(Id id) const
  {
    return branches.at(id).topology;
  }
  std::map<Id, GrowthDemandSummary>
  GrowthDemandState::children(Id id, const GrowthDemandPreview &preview) const
  {
    auto       result  = branches.at(id).children;
    const auto changed = preview.child_summaries.find(id);
    if (changed != preview.child_summaries.end())
      for (const auto &[child, value] : changed->second)
        result.at(child) = value;
    return result;
  }
  GrowthDemandRootView
  GrowthDemandState::root_view(Id id, const GrowthDemandPreview &preview) const
  {
    AssertThrow(
      report.valid() && preview.validation.valid() &&
        preview.snapshot_revision == revision && root_positions.count(id),
      dealii::ExcMessage(
        "A root view requires a current valid preview on its authoritative root owner."));
    GrowthDemandRootView result;
    result.summary = summary(id, preview);
    result.radius  = branches.at(id).topology.root_radius;
    result.system  = branches.at(id).system;
    for (auto i = root_capacity + root_positions.at(id); i > 1; i /= 2)
      {
        const auto  sibling = i ^ std::size_t(1);
        const auto  found   = preview.root_sums.find(sibling);
        const auto &value =
          found == preview.root_sums.end() ? root_sums[sibling] : found->second;
        for (unsigned int j = 0; j < 6; ++j)
          result.other_local[j] += value[j];
      }
    return result;
  }

  GrowthDemandPreview
  GrowthDemandState::preview(const GrowthDemandUpdate &update) const
  {
    using namespace dealii;
    const auto          comm = communicator;
    const auto          size = Utilities::MPI::n_mpi_processes(comm);
    GrowthDemandPreview result;
    result.snapshot_revision = revision;
    result.validation        = report;
    bool valid = report.valid() && update.snapshot_revision == revision;
    std::map<unsigned int, std::vector<TerminalDemandUpdate>> send;
    for (const auto &entry : update.terminals)
      {
        valid = valid && entry.terminal != invalid_id && positive(entry.demand);
        send[entry.terminal % size].push_back(entry);
      }
    if (!phase(
          result.validation,
          valid,
          comm,
          "Demand preview needs current revisions and positive absolute Di."))
      return result;
    const auto   updates = Utilities::MPI::some_to_some(comm, send);
    std::set<Id> unique;
    std::map<unsigned int, std::map<Id, double>> deliveries;
    for (const auto &[source, items] : updates)
      for (const auto &entry : items)
        {
          const auto found = directory.find(entry.terminal);
          if (found == directory.end() || !unique.insert(entry.terminal).second)
            {
              valid = false;
              continue;
            }
          deliveries[found->second.owner].emplace(found->second.branch,
                                                  entry.demand);
        }
    if (!phase(
          result.validation,
          valid,
          comm,
          "Each updated terminal must occur once in the execution directory."))
      return result;
    const auto   delivered = Utilities::MPI::some_to_some(comm, deliveries);
    std::set<Id> pending;
    std::map<Id, std::set<Id>> marked;
    for (const auto &[source, items] : delivered)
      for (const auto &[id, demand] : items)
        {
          const auto b = branches.find(id);
          if (b == branches.end() || !b->second.topology.children.empty() ||
              !result.leaf_demands.emplace(id, demand).second)
            {
              valid = false;
              continue;
            }
          if (demand != b->second.leaf_demand)
            {
              marked.emplace(id, std::set<Id>{});
              pending.insert(id);
            }
        }
    if (!phase(
          result.validation,
          valid,
          comm,
          "Demand changes must reach current authoritative leaf branches."))
      return result;
    while (
      Utilities::MPI::sum(static_cast<std::uint64_t>(pending.size()), comm) > 0)
      {
        std::map<unsigned int, std::vector<std::pair<Id, Id>>> parents;
        for (Id id : pending)
          {
            const auto &b = branches.at(id).topology;
            if (b.parent != invalid_id)
              parents[b.parent_owner].push_back({b.parent, id});
          }
        const auto messages = Utilities::MPI::some_to_some(comm, parents);
        pending.clear();
        for (const auto &[owner, items] : messages)
          for (const auto &[parent, child] : items)
            {
              const auto b = branches.find(parent);
              if (b == branches.end() ||
                  !b->second.topology.children.count(child) ||
                  b->second.topology.children.at(child) != owner)
                {
                  valid = false;
                  continue;
                }
              const auto [position, fresh] =
                marked.emplace(parent, std::set<Id>{});
              position->second.insert(child);
              if (fresh)
                pending.insert(parent);
            }
        if (!phase(result.validation,
                   valid,
                   comm,
                   "Demand dependencies must retain native parent authority."))
          return result;
      }
    result.marked_branches = marked.size();
    std::set<Id> ready;
    for (const auto &[id, children] : marked)
      if (children.empty())
        ready.insert(id);
    auto &changed_children = result.child_summaries;
    while (Utilities::MPI::sum(static_cast<std::uint64_t>(ready.size()), comm) >
           0)
      {
        std::map<unsigned int, std::vector<ChildValue>> packets;
        for (Id id : ready)
          {
            const auto &b        = branches.at(id);
            auto        children = b.children;
            for (const auto &[child, value] : changed_children[id])
              children.at(child) = value;
            auto value = aggregate(b,
                                   b.topology.children.empty() ?
                                     result.leaf_demands.at(id) :
                                     0.,
                                   children);
            valid      = valid_summary(value) &&
                    result.owned.emplace(id, value).second && valid;
            ++result.recomputed_branches;
            if (b.topology.parent != invalid_id)
              packets[b.topology.parent_owner].push_back(
                {b.topology.parent, id, std::move(value)});
          }
        const auto messages = Utilities::MPI::some_to_some(comm, packets);
        ready.clear();
        for (const auto &[owner, items] : messages)
          for (const auto &item : items)
            {
              const auto b = branches.find(item.parent);
              if (b == branches.end() || !marked.count(item.parent) ||
                  !marked.at(item.parent).count(item.child) ||
                  b->second.topology.children.at(item.child) != owner ||
                  !changed_children[item.parent]
                     .emplace(item.child, item.value)
                     .second)
                {
                  valid = false;
                  continue;
                }
              if (changed_children.at(item.parent).size() ==
                  marked.at(item.parent).size())
                ready.insert(item.parent);
            }
        if (
          !phase(
            result.validation,
            valid,
            comm,
            "Sparse demand recomputation requires finite positive child summaries."))
          return result;
      }
    if (
      !phase(
        result.validation,
        result.recomputed_branches == result.marked_branches,
        comm,
        "Every affected branch must be recomputed once after all affected children."))
      return result;
    // Patch only the paths of changed roots in the positive sum tree. A
    // disappearing dominant contribution cannot erase small retained roots.
    auto &changed_sums = result.root_sums;
    std::set<std::size_t, std::greater<std::size_t>> parents;
    for (const auto &[id, value] : result.owned)
      if (root_positions.count(id))
        {
          const auto position    = root_capacity + root_positions.at(id);
          changed_sums[position] = root_value(id, value);
          for (auto i = position / 2; i > 0; i /= 2)
            parents.insert(i);
        }
    const auto get = [&](std::size_t i) -> const std::array<long double, 6> & {
      const auto found = changed_sums.find(i);
      return found == changed_sums.end() ? root_sums[i] : found->second;
    };
    for (auto i : parents)
      for (unsigned int j = 0; j < 6; ++j)
        changed_sums[i][j] = get(2 * i)[j] + get(2 * i + 1)[j];
    result.rank_totals = Utilities::MPI::all_gather(comm, get(1));
    for (const auto &rank_sum : result.rank_totals)
      for (unsigned int j = 0; j < 6; ++j)
        result.totals[j] += rank_sum[j];
    for (long double value : result.totals)
      valid = std::isfinite(value) && value >= 0. && valid;
    phase(result.validation,
          valid,
          comm,
          "Updated root aggregates must remain finite and nonnegative.");
    return result;
  }
} // namespace ImmersX::Vascularization::internal
