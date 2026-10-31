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
             positive(value.minimum_scaled_ratio) &&
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
    result.volume_length = branch.length;
    result.resistance_length =
      branch.with_resistance ? branch.viscosity * branch.length : 0.;
    result.minimum_scaled_ratio = 1.L / branch.geometry.radius_scale;
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
        if (branch.with_resistance)
          result.resistance_length +=
            std::pow(ratio, 2.L - 4.L / branch.gamma) * child.resistance_length;
        result.minimum_scaled_ratio =
          std::min(result.minimum_scaled_ratio,
                   std::pow(ratio, 1.L / branch.gamma) *
                     child.minimum_scaled_ratio);
        result.minimum_ratio =
          std::min(result.minimum_ratio,
                   std::pow(ratio, 1.L / branch.gamma) * child.minimum_ratio);
      }
    return result;
  }

  GrowthDemandState::GrowthDemandState(const NetworkData            &network,
                                       const ForestState            &forest,
                                       const Parameters             &parameters,
                                       std::uint64_t                 snapshot,
                                       MPI_Comm                      comm,
                                       const GrowthGeometryMeasures *geometry)
    : communicator(comm)
    , revision(snapshot)
    , report(forest.validation)
  {
    using namespace dealii;
    accepted_volume = forest.volume;
    for (unsigned int system = 0; system < 2; ++system)
      if (forest.total_demand[system] > 0.)
        accepted_resistance[system] =
          forest.power_numerator[system] /
          std::pow(static_cast<long double>(forest.total_demand[system]), 2);
    const auto rank  = Utilities::MPI::this_mpi_process(comm);
    const auto size  = Utilities::MPI::n_mpi_processes(comm);
    bool       valid = report.valid() &&
                 positive(parameters.arterial_murray_exponent) &&
                 positive(parameters.venous.murray_exponent) &&
                 std::isfinite(parameters.reference_flow) &&
                 parameters.reference_flow >= 0. &&
                 std::isfinite(parameters.dissipation_weight) &&
                 parameters.dissipation_weight >= 0. &&
                 parameters.dissipation_weight <= 1.;
    if (geometry)
      valid = valid && geometry->size() == forest.owned.size();
    for (auto other : Utilities::MPI::all_gather(comm, snapshot))
      valid = valid && other == snapshot;
    const std::array<double, 4> controls{{parameters.arterial_murray_exponent,
                                          parameters.venous.murray_exponent,
                                          parameters.dissipation_weight,
                                          parameters.reference_flow}};
    for (const auto &other : Utilities::MPI::all_gather(comm, controls))
      valid = valid && other == controls;
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
        branch.topology        = topology;
        branch.length          = length(network, id);
        branch.viscosity       = vessel->second.viscosity;
        branch.with_resistance = parameters.reference_flow > 0.;
        if (geometry)
          {
            const auto found = geometry->find(id);
            if (found == geometry->end())
              valid = false;
            else
              branch.geometry = found->second;
          }
        else
          branch.geometry = {branch.length, 1., 1., true};
        valid = valid && branch.geometry.complete &&
                positive(branch.geometry.normalized_length) &&
                positive(branch.geometry.radius_scale) &&
                positive(branch.geometry.clearance_scale);
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
        owned_roots.insert(id);
    std::map<unsigned int, std::map<Id, unsigned int>> root_registrations;
    for (Id id : owned_roots)
      root_registrations[id % size].emplace(id, rank);
    const auto registered_roots =
      Utilities::MPI::some_to_some(comm, root_registrations);
    for (const auto &[source, records] : registered_roots)
      for (const auto &[id, owner] : records)
        valid =
          source == owner && root_directory.emplace(id, owner).second && valid;
    if (!phase(report,
               valid,
               comm,
               "Each physical root needs one authoritative directory entry."))
      return;
    // Prefix keys retain physical identifiers. Root migration can remove or
    // insert one leaf without renumbering or rebuilding the other roots.
    constexpr auto bits = std::numeric_limits<Id>::digits;
    std::set<GrowthRootSumKey, std::greater<GrowthRootSumKey>> parents;
    for (Id id : owned_roots)
      {
        root_sums[{bits, id}] = root_value(id, branches.at(id).value);
        for (auto key = GrowthRootSumKey{bits, id}; key.first > 0;)
          {
            --key.first;
            key.second >>= 1;
            parents.insert(key);
          }
      }
    const std::array<long double, 6> zero{};
    const auto                       get =
      [&](GrowthRootSumKey key) -> const std::array<long double, 6> & {
      const auto found = root_sums.find(key);
      return found == root_sums.end() ? zero : found->second;
    };
    for (auto key : parents)
      for (unsigned int j = 0; j < 6; ++j)
        root_sums[key][j] = get({key.first + 1, key.second << 1})[j] +
                            get({key.first + 1, (key.second << 1) | Id{1}})[j];
    for (unsigned int j = 0; j < 6; ++j)
      {
        accepted_totals[j] = Utilities::MPI::sum(get({0, 0})[j], comm);
        valid = std::isfinite(accepted_totals[j]) && accepted_totals[j] >= 0. &&
                valid;
      }
    phase(report,
          valid,
          comm,
          "Accepted root aggregates must be finite and nonnegative.");
  }

  long double
  GrowthDemandState::demand_value(const GrowthDemandSummary &value)
  {
    const boost::multiprecision::cpp_rational physical_mass =
      boost::multiprecision::cpp_rational(value.mass) /
      (boost::multiprecision::cpp_int(1) << 1074);
    return physical_mass.convert_to<long double>();
  }

  std::array<long double, 6>
  GrowthDemandState::root_value(Id id, const GrowthDemandSummary &value) const
  {
    const auto                &branch = branches.at(id);
    const long double          mass   = demand_value(value);
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
        preview.snapshot_revision == revision && owned_roots.count(id),
      dealii::ExcMessage(
        "A root view requires a current valid preview on its authoritative root owner."));
    GrowthDemandRootView result;
    result.summary      = summary(id, preview);
    result.radius       = branches.at(id).topology.root_radius;
    result.gamma        = branches.at(id).gamma;
    result.system       = branches.at(id).system;
    constexpr auto bits = std::numeric_limits<Id>::digits;
    for (auto key = GrowthRootSumKey{bits, id}; key.first > 0;)
      {
        const GrowthRootSumKey sibling{key.first, key.second ^ Id{1}};
        const auto             changed  = preview.root_sums.find(sibling);
        const auto             accepted = root_sums.find(sibling);
        for (unsigned int j = 0; j < 6; ++j)
          result.other_local[j] +=
            changed != preview.root_sums.end() ? changed->second[j] :
            accepted != root_sums.end()        ? accepted->second[j] :
                                                 0.;
        --key.first;
        key.second >>= 1;
      }
    return result;
  }

  GrowthDemandPreview
  GrowthDemandState::route_demands(const GrowthDemandUpdate &update) const
  {
    using namespace dealii;
    const auto          comm = communicator;
    const auto          size = Utilities::MPI::n_mpi_processes(comm);
    GrowthDemandPreview result;
    result.snapshot_revision      = revision;
    result.canonical_total_demand = update.canonical_total_demand;
    result.validation             = report;
    bool valid = report.valid() && update.snapshot_revision == revision;
    const std::array<std::uint64_t, 4> tokens{
      {update.snapshot_revision,
       update.base_quadrature_revision,
       update.quadrature_revision,
       static_cast<std::uint64_t>(update.canonical_total_demand.has_value())}};
    for (const auto &other : Utilities::MPI::all_gather(comm, tokens))
      valid = valid && other == tokens;
    const auto canonical =
      update.canonical_total_demand.value_or(std::array<double, 2>{{0., 0.}});
    for (double value : canonical)
      valid = valid && std::isfinite(value) && value >= 0.;
    for (const auto &other : Utilities::MPI::all_gather(comm, canonical))
      valid = valid && other == canonical;
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
    const auto delivered = Utilities::MPI::some_to_some(comm, deliveries);
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
        }
    if (!phase(
          result.validation,
          valid,
          comm,
          "Demand changes must reach current authoritative leaf branches."))
      return result;
    return result;
  }

  GrowthDemandPreview
  GrowthDemandState::preview(const GrowthDemandUpdate &update) const
  {
    auto result = route_demands(update);
    if (!result.validation.valid())
      return result;
    return preview_branches(std::move(result), {});
  }

  GrowthDemandPreview
  GrowthDemandState::preview_branches(GrowthDemandPreview result,
                                      const std::set<Id> &changed) const
  {
    using namespace dealii;
    const auto comm = communicator;
    bool       valid =
      result.validation.valid() && result.snapshot_revision == revision;
    std::set<Id>               pending;
    std::map<Id, std::set<Id>> marked;
    for (const auto &[id, demand] : result.leaf_demands)
      {
        const auto b = branches.find(id);
        if (b == branches.end() || !b->second.topology.children.empty() ||
            !positive(demand))
          {
            valid = false;
            continue;
          }
        if (demand != b->second.leaf_demand)
          pending.insert(id);
      }
    for (Id id : changed)
      if (branches.count(id))
        pending.insert(id);
      else
        valid = false;
    for (Id id : pending)
      marked.emplace(id, std::set<Id>{});
    if (
      !phase(
        result.validation,
        valid,
        comm,
        "Recomputation seeds must be authoritative branches of the current topology."))
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
            const auto   leaf = result.leaf_demands.find(id);
            const double demand =
              b.topology.children.empty() ?
                (leaf == result.leaf_demands.end() ? b.leaf_demand :
                                                     leaf->second) :
                0.;
            auto value = aggregate(b, demand, children);
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
    auto          &changed_sums = result.root_sums;
    constexpr auto bits         = std::numeric_limits<Id>::digits;
    std::set<GrowthRootSumKey, std::greater<GrowthRootSumKey>> parents;
    for (const auto &[key, value] : changed_sums)
      if (key.first == bits)
        for (auto parent = key; parent.first > 0;)
          {
            --parent.first;
            parent.second >>= 1;
            parents.insert(parent);
          }
    for (const auto &[id, value] : result.owned)
      if (owned_roots.count(id))
        {
          changed_sums[{bits, id}] = root_value(id, value);
          for (auto key = GrowthRootSumKey{bits, id}; key.first > 0;)
            {
              --key.first;
              key.second >>= 1;
              parents.insert(key);
            }
        }
    const std::array<long double, 6> zero{};
    const auto                       get =
      [&](GrowthRootSumKey key) -> const std::array<long double, 6> & {
      const auto changed = changed_sums.find(key);
      if (changed != changed_sums.end())
        return changed->second;
      const auto accepted = root_sums.find(key);
      return accepted == root_sums.end() ? zero : accepted->second;
    };
    for (auto key : parents)
      for (unsigned int j = 0; j < 6; ++j)
        changed_sums[key][j] =
          get({key.first + 1, key.second << 1})[j] +
          get({key.first + 1, (key.second << 1) | Id{1}})[j];
    result.rank_totals = Utilities::MPI::all_gather(comm, get({0, 0}));
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
