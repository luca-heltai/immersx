// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_commit.h"

#include <deal.II/base/timer.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/generation.h>

#include <cmath>
#include <memory>

#include "distribution_internal.h"
#include "growth_batch.h"
namespace ImmersX::Vascularization::internal
{
  bool
  apply_growth_split(NetworkData &network, const GrowthSplit &s)
  {
    const auto it = network.vessels.find(s.vessel);
    if (it == network.vessels.end() || it->second.owner != s.owner ||
        it->second.nodes != s.old_nodes || !std::isfinite(s.demand) ||
        s.demand <= 0. || s.junction == invalid_id || s.tip == invalid_id ||
        s.continuation == invalid_id || s.leaf == invalid_id ||
        s.terminal == invalid_id || s.territory == invalid_id ||
        s.junction == s.tip || s.continuation == s.leaf ||
        network.nodes.count(s.junction) || network.nodes.count(s.tip) ||
        network.vessels.count(s.continuation) ||
        network.vessels.count(s.leaf) || network.terminals.count(s.terminal))
      return false;
    for (unsigned int end = 0; end < 2; ++end)
      if (!network.nodes.count(s.old_nodes[end]) ||
          network.nodes.at(s.old_nodes[end]).position != s.old_points[end])
        return false;
    for (unsigned int d = 0; d < 3; ++d)
      if (!std::isfinite(s.junction_position[d]) ||
          !std::isfinite(s.tip_position[d]))
        return false;
    const auto old                  = it->second;
    network.nodes[s.junction]       = {s.junction_position,
                                       s.owner,
                                       Port::none,
                                       Provenance::generated};
    network.nodes[s.tip]            = {s.tip_position,
                                       s.owner,
                                       Port::none,
                                       Provenance::generated};
    it->second.nodes[1]             = s.junction;
    auto continuation               = old;
    continuation.nodes              = {{s.junction, old.nodes[1]}};
    continuation.provenance         = Provenance::generated;
    network.vessels[s.continuation] = continuation;
    auto leaf                       = continuation;
    leaf.nodes[1]                   = s.tip;
    network.vessels[s.leaf]         = leaf;
    network.terminals[s.terminal]   = {
      s.tip, s.territory, old.system, s.owner, s.demand};
    return true;
  }

  namespace
  {
    // Carry only private mutability and root controls to the actual physical
    // destination. Root ids never pass through floating-point properties.
    NetworkData
    migrate(const dealii::Triangulation<3> &bg,
            const dealii::Mapping<3>       &mapping,
            const NetworkData              &network,
            std::set<Id>                   &mutable_owned,
            std::map<Id, double>           &roots,
            double                          halo,
            MPI_Comm                        comm)
    {
      std::map<unsigned int, std::map<Id, double>> routes;
      auto result = redistribute_with_owners(
        bg, mapping, network, halo, comm, [&](Id id, unsigned int owner) {
          if (mutable_owned.count(id))
            routes[owner].emplace(id, roots.count(id) ? roots.at(id) : 0.);
        });
      const auto metadata = dealii::Utilities::MPI::some_to_some(comm, routes);
      mutable_owned.clear();
      roots.clear();
      for (const auto &[source, values] : metadata)
        for (const auto &[id, radius] : values)
          {
            mutable_owned.insert(id);
            if (radius > 0.)
              roots[id] = radius;
          }
      return result;
    }
  } // namespace

  GrowthCommit
  commit_growth_splits(const dealii::Triangulation<3> &bg,
                       const dealii::Mapping<3>       &mapping,
                       const NetworkData              &snapshot,
                       const std::set<Id>             &mutable_owned,
                       const std::map<Id, double>     &fixed_owned_roots,
                       const std::vector<GrowthSplit> &local,
                       const Parameters               &p,
                       MPI_Comm                        comm,
                       BackgroundGeometry             *shared_geometry)
  {
    using namespace dealii;
    GrowthCommit       result;
    const unsigned int rank = Utilities::MPI::this_mpi_process(comm),
                       size = Utilities::MPI::n_mpi_processes(comm);
    const auto phase        = [&](bool valid, const std::string &message) {
      const bool all =
        Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1;
      if (!all)
        result.validation.issues.push_back(
          {"growth commit", invalid_id, message});
      return all;
    };
    bool valid = validate_data(snapshot, size).valid() &&
                 std::isfinite(p.clearance) && p.clearance >= 0. &&
                 std::isfinite(p.minimum_length) && p.minimum_length >= 0. &&
                 std::isfinite(p.minimum_angle) && p.minimum_angle >= 0. &&
                 p.minimum_angle <= 180.;
    for (const auto &s : local)
      valid = valid && s.owner < size;
    for (Id id : mutable_owned)
      valid = valid && snapshot.vessels.count(id) &&
              snapshot.vessels.at(id).owner == rank;
    for (const auto &[id, radius] : fixed_owned_roots)
      valid = valid && mutable_owned.count(id);
    if (!phase(valid, "Invalid local ownership or split destination."))
      return result;
    const std::array<double, 5> controls{
      {p.clearance,
       p.minimum_length,
       p.minimum_angle,
       static_cast<double>(p.geometry_depth),
       static_cast<double>(p.geometry_work)}};
    for (const auto &other : Utilities::MPI::all_gather(comm, controls))
      valid = valid && other == controls;
    if (!phase(valid,
               "All batch participants must use the same geometric controls."))
      return result;
    std::unique_ptr<BackgroundGeometry> local_geometry;
    if (!shared_geometry)
      local_geometry = std::make_unique<BackgroundGeometry>(bg, mapping);
    auto &geometry = shared_geometry ? *shared_geometry : *local_geometry;
    if (!phase(
          &geometry.cache.get_triangulation() == &bg &&
            &geometry.mapping == &mapping,
          "Growth geometry must describe the prepared background and mapping."))
      return result;
    geometry.configure_length_scale(p.length_scale, comm);
    const auto  &scale = *geometry.length_scale;
    const double halo  = scale.physical_length(p.clearance, scale.maximum());
    std::map<unsigned int, std::vector<GrowthSplit>> send;
    for (const auto &s : local)
      send[s.owner].push_back(s);
    const auto   patches = Utilities::MPI::some_to_some(comm, send);
    auto         trial   = snapshot;
    auto         ids     = mutable_owned;
    auto         roots   = fixed_owned_roots;
    std::set<Id> written;
    for (const auto &[producer, splits] : patches)
      for (const auto &s : splits)
        {
          const bool admissible = s.owner == rank &&
                                  mutable_owned.count(s.vessel) &&
                                  written.insert(s.vessel).second;
          if (!admissible || !apply_growth_split(trial, s))
            valid = false;
          else
            {
              ids.insert(s.continuation);
              ids.insert(s.leaf);
            }
        }
    if (!phase(
          valid && validate_data(trial, size).valid(),
          "A split is stale, duplicates a write, or changes a frozen vessel."))
      return result;
    Timer timer;
    trial = migrate(bg, mapping, trial, ids, roots, halo, comm);
    result.redistribution_seconds = timer.wall_time();
    const double physiology_begin = timer.wall_time();
    auto         state = evaluate_forest(trial, ids, roots, p, comm, scale);
    if (!state.validation.valid())
      {
        result.physiology_seconds = timer.wall_time() - physiology_begin;
        result.validation         = state.validation;
        return result;
      }
    for (const auto &[id, b] : state.owned)
      trial.vessels.at(id).radii = {{b.radius, b.radius}};
    result.physiology_seconds = timer.wall_time() - physiology_begin;
    // Enlarged radii need refreshed multi-partition geometric ghosts.
    const double migration_begin = timer.wall_time();
    trial = migrate(bg, mapping, trial, ids, roots, halo, comm);
    result.redistribution_seconds += timer.wall_time() - migration_begin;
    const double               certification_begin = timer.wall_time();
    std::vector<BatchProposal> checks;
    for (Id id : ids)
      {
        const auto   &v = trial.vessels.at(id);
        BatchProposal check;
        check.id            = id;
        check.owners[id]    = rank;
        check.old_nodes[id] = v.nodes;
        check.changed.push_back({id,
                                 v.nodes,
                                 {{trial.nodes.at(v.nodes[0]).position,
                                   trial.nodes.at(v.nodes[1]).position}},
                                 v.radii[0]});
        for (Id node : v.nodes)
          {
            const auto &n = trial.nodes.at(node);
            if (std::max({std::abs(n.inward_port_normal[0]),
                          std::abs(n.inward_port_normal[1]),
                          std::abs(n.inward_port_normal[2])}) > 0.)
              {
                check.port_owners[node] = n.owner;
                check.normals[node]     = n.inward_port_normal;
              }
          }
        checks.push_back(std::move(check));
      }
    const auto certified =
      select_growth_batch(geometry, trial, checks, p, comm, [&](Id id) {
        return std::max(trial.vessels.at(id).radii[0],
                        trial.vessels.at(id).radii[1]);
      });
    for (const auto &[id, accepted] : certified.local)
      valid = valid && accepted;
    const bool certified_all = phase(
      valid,
      "Joint radius normalization invalidates final vessel geometry; the entire batch was discarded.");
    result.certification_seconds = timer.wall_time() - certification_begin;
    if (!certified_all)
      return result;
    result.committed         = true;
    result.network           = std::move(trial);
    result.mutable_owned     = std::move(ids);
    result.fixed_owned_roots = std::move(roots);
    result.physiology        = std::move(state);
    return result;
  }
} // namespace ImmersX::Vascularization::internal
