// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_execution.h"

#include <deal.II/base/scope_exit.h>
#include <deal.II/base/timer.h>

#include <immersx/vascularization/generation.h>

#include <cmath>

#include "distribution_internal.h"
#include "growth_system.h"
namespace ImmersX::Vascularization::internal
{
  namespace
  {
    constexpr std::array<System, 3> forest_systems{
      {System::arterial, System::venous, System::portal}};
    GrowthDemandState
    initial_state(const NetworkData            &n,
                  const ForestState            &forest,
                  const Parameters             &p,
                  std::uint64_t                 revision,
                  MPI_Comm                      comm,
                  const GrowthGeometryMeasures &all)
    {
      GrowthGeometryMeasures subset;
      for (const auto &[id, branch] : forest.owned)
        {
          const auto found = all.find(id);
          if (found != all.end())
            subset.emplace(id, found->second);
        }
      return GrowthDemandState(n, forest, p, revision, comm, &subset);
    }
    std::set<Id>
    mutable_ids(const ForestState &forest)
    {
      std::set<Id> result;
      for (const auto &[id, branch] : forest.owned)
        result.insert(id);
      return result;
    }
    std::map<Id, double>
    fixed_roots(const ForestState &forest)
    {
      std::map<Id, double> result;
      for (const auto &[id, branch] : forest.owned)
        if (branch.parent == invalid_id)
          result.emplace(id, branch.root_radius);
      return result;
    }
  } // namespace
  GrowthExecution::AcceptedSnapshot::AcceptedSnapshot(
    const GrowthExecution &value,
    std::uint64_t          revision,
    std::uint64_t          epoch)
    : owner(&value)
    , snapshot_revision(revision)
    , view_epoch(epoch)
  {}
  bool
  GrowthExecution::AcceptedSnapshot::current() const noexcept
  {
    return owner && !owner->staging && owner->report.valid() &&
           snapshot_revision == owner->demand_state.revision &&
           view_epoch == owner->view_epoch;
  }
  std::uint64_t
  GrowthExecution::AcceptedSnapshot::revision() const
  {
    AssertThrow(current(),
                dealii::ExcMessage(
                  "A growth snapshot expires when the next stage begins."));
    return snapshot_revision;
  }
  const NetworkData &
  GrowthExecution::AcceptedSnapshot::network() const
  {
    revision();
    return owner->rows.network();
  }
  const GrowthDemandState &
  GrowthExecution::AcceptedSnapshot::demand() const
  {
    revision();
    return owner->demand_state;
  }
  const GrowthIndices &
  GrowthExecution::AcceptedSnapshot::indices() const
  {
    revision();
    return owner->spatial;
  }
  const GrowthExecution::ObstacleTree &
  GrowthExecution::AcceptedSnapshot::obstacles() const
  {
    revision();
    return owner->obstacle_index;
  }
  std::uint64_t
  GrowthExecution::AcceptedSnapshot::owned_mutable_count() const
  {
    revision();
    return owner->demand_state.branches.size();
  }
  std::uint64_t
  GrowthExecution::AcceptedSnapshot::mutable_count() const
  {
    revision();
    return owner->demand_state.branch_count;
  }
  std::uint64_t
  GrowthExecution::AcceptedSnapshot::owned_terminal_count() const
  {
    revision();
    return owner->owned_leaves;
  }
  std::uint64_t
  GrowthExecution::AcceptedSnapshot::terminal_count() const
  {
    revision();
    return owner->leaves;
  }
  GrowthExecution::AcceptedSnapshot
  GrowthExecution::accepted_snapshot() const
  {
    AssertThrow(
      !staging && report.valid(),
      dealii::ExcMessage(
        "Only an accepted validated growth state has a snapshot view."));
    return AcceptedSnapshot(*this, demand_state.revision, view_epoch);
  }
  const ValidationReport &
  GrowthExecution::validation() const noexcept
  {
    return report;
  }
  bool
  GrowthExecution::phase(ValidationReport &validation,
                         bool              local,
                         const char       *message) const
  {
    const bool all =
      dealii::Utilities::MPI::min(static_cast<unsigned int>(local),
                                  communicator) == 1;
    if (!all)
      validation.issues.push_back({"growth execution", invalid_id, message});
    return all;
  }
  GrowthDemandPreview
  GrowthExecution::accepted_preview() const
  {
    GrowthDemandPreview result;
    result.validation        = demand_state.validation();
    result.snapshot_revision = demand_state.revision;
    result.totals            = demand_state.totals();
    return result;
  }
  GrowthExecution::GrowthExecution(
    const NetworkData            &initial,
    const ForestState            &forest,
    const Parameters             &p,
    std::uint64_t                 revision,
    BackgroundGeometry           &background,
    const GrowthGeometryMeasures &initial_measures,
    GrowthGeometryMeasureFunction measuring,
    GrowthGeometryReleaseFunction releasing,
    double                        halo,
    MPI_Comm                      comm,
    GrowthWork                   *counts)
    : demand_state(
        initial_state(initial, forest, p, revision, comm, initial_measures))
    , rows(initial, comm, counts)
    , journal(demand_state, counts)
    , spatial(rows.network(),
              mutable_ids(forest),
              fixed_roots(forest),
              comm,
              counts)
    , parameters(p)
    , geometry(background)
    , measure(std::move(measuring))
    , release(std::move(releasing))
    , communicator(comm)
    , work(counts)
    , ghost_halo(halo)
    , report(demand_state.validation())
  {
    bool valid = report.valid() && bool(measure) && std::isfinite(halo) &&
                 halo >= 0. && std::isfinite(p.clearance) &&
                 p.clearance >= 0. && std::isfinite(p.minimum_length) &&
                 p.minimum_length >= 0. && std::isfinite(p.minimum_radius) &&
                 p.minimum_radius >= 0. && std::isfinite(p.minimum_angle) &&
                 p.minimum_angle >= 0. && p.minimum_angle <= 180. &&
                 initial_measures.size() == rows.network().vessels.size();
    const auto geometry_mode = static_cast<unsigned int>(p.enforce_geometry);
    const auto geometry_min  = dealii::Utilities::MPI::min(geometry_mode, comm);
    const auto geometry_max  = dealii::Utilities::MPI::max(geometry_mode, comm);
    valid                    = valid && geometry_min == geometry_max;
    for (const auto &[id, v] : rows.network().vessels)
      {
        const auto found = initial_measures.find(id);
        valid            = valid && found != initial_measures.end();
        if (found != initial_measures.end())
          valid = valid && std::isfinite(found->second.clearance_scale) &&
                  found->second.clearance_scale >= 0.;
      }
    if (
      !phase(
        report,
        valid,
        "Execution bootstrap requires complete owned geometry metadata and a finite physical halo."))
      return;
    owned_leaves = forest.owned_leaf_terminals.size();
    leaves       = dealii::Utilities::MPI::sum(owned_leaves, comm);
    count_work(work,
               GrowthOperation::physiology_branch_visits,
               demand_state.branches.size());
    count_work(work,
               GrowthOperation::metadata_copy_attempts,
               demand_state.branches.size());
    count_work(work, GrowthOperation::network_index_builds);
    std::set<Id> wanted;
    for (const auto &[id, branch] : demand_state.branches)
      wanted.insert(branch.topology.root);
    const auto preview = accepted_preview();
    controls           = demand_state.root_controls(wanted, preview, report);
    if (!report.valid())
      return;
    if (p.enforce_geometry)
      for (const auto &[id, v] : rows.network().vessels)
        {
          Certificate entry;
          entry.cap             = {id,
                                   v.nodes,
                                   {{rows.network().nodes.at(v.nodes[0]).position,
                                     rows.network().nodes.at(v.nodes[1]).position}},
                                   std::max(v.radii[0], v.radii[1])};
          entry.clearance_scale = initial_measures.at(id).clearance_scale;
          clearance_bound   = std::max(clearance_bound, entry.clearance_scale);
          const auto branch = demand_state.branches.find(id);
          if (branch != demand_state.branches.end())
            {
              entry.root            = branch->second.topology.root;
              entry.mass            = branch->second.value.mass;
              entry.lower_root_mass = controls.at(entry.root).summary.mass;
              entry.upper_root_mass =
                id == entry.root ? Mass(0) : 2 * entry.lower_root_mass;
              entry.cap.radius = radius(id, preview);
              if (entry.cap.radius == controls.at(entry.root).radius)
                entry.lower_root_mass = 0;
            }
          replace_certificate(id, std::move(entry), false);
        }
    clearance_bound =
      dealii::Utilities::MPI::max(clearance_bound, communicator);
    root_members_changed.clear();
    update_bounds(demand_state.owned_roots, preview, false);
    if (!physical_bounds(preview, report))
      return;
    GrowthOwnershipDelta initial_geometry;
    for (const auto &[id, branch] : demand_state.branches)
      initial_geometry.changed_owned.insert(id);
    certify(initial_geometry, preview, initial_measures, report);
    certificates_before.clear();
    controls_before.clear();
    root_members_changed.clear();
  }
  GrowthExecution::~GrowthExecution()
  {
    discard();
  }
  bool
  GrowthExecution::refresh_controls(const GrowthOwnershipDelta &delta,
                                    const GrowthDemandPreview  &preview,
                                    ValidationReport           &validation)
  {
    std::vector<Id> changed;
    for (const auto &[id, value] : preview.owned)
      if (demand_state.branches.at(id).topology.parent == invalid_id)
        changed.push_back(id);
    std::set<Id> wanted;
    for (const auto &part :
         dealii::Utilities::MPI::all_gather(communicator, changed))
      for (Id id : part)
        if (controls.count(id))
          wanted.insert(id);
    for (Id id : delta.changed_owned)
      wanted.insert(demand_state.branches.at(id).topology.root);
    const auto current =
      demand_state.root_controls(wanted, preview, validation);
    if (!validation.valid())
      return false;
    for (const auto &[id, value] : current)
      {
        const auto found = controls.find(id);
        if (!controls_before.count(id))
          controls_before.emplace(id,
                                  found == controls.end() ?
                                    std::optional<GrowthDemandRootView>{} :
                                    found->second);
        controls[id] = value;
      }
    return true;
  }
  double
  GrowthExecution::radius_at(const Mass                 &mass,
                             const GrowthDemandRootView &root,
                             const Mass                 &root_mass) const
  {
    GrowthDemandSummary child, parent;
    child.mass  = mass;
    parent.mass = root_mass;
    return static_cast<double>(root.radius *
                               std::pow(demand_state.demand_value(child) /
                                          demand_state.demand_value(parent),
                                        1.L / root.gamma));
  }
  double
  GrowthExecution::radius(Id id, const GrowthDemandPreview &preview) const
  {
    const auto &branch = demand_state.branches.at(id);
    const auto &root   = controls.at(branch.topology.root);
    return id == branch.topology.root ?
             root.radius :
             radius_at(demand_state.summary(id, preview).mass,
                       root,
                       root.summary.mass);
  }
  bool
  GrowthExecution::physical_bounds(const GrowthDemandPreview &preview,
                                   ValidationReport          &validation) const
  {
    bool valid = invalid_bounds == 0;
    if (parameters.reference_flow > 0.)
      for (unsigned int system = 0; system < 3; ++system)
        if (!leaf_minima[system].empty())
          {
            const long double   system_demand = preview.total_demand(system);
            GrowthDemandSummary smallest;
            smallest.mass             = leaf_minima[system].begin()->first;
            const double minimum_flow = static_cast<double>(
              system_reference_flow(parameters, forest_systems[system]) *
              demand_state.demand_value(smallest) / system_demand);
            valid = valid && std::isfinite(system_demand) &&
                    system_demand > 0. && std::isfinite(minimum_flow) &&
                    minimum_flow > 0.;
          }
    for (unsigned int system = 0; system < 3; ++system)
      {
        const double flow =
          system_reference_flow(parameters, forest_systems[system]);
        const double volume =
          static_cast<double>(preview.totals[3 * system + 1]);
        const long double demand = preview.total_demand(system);
        const double      power =
          demand > 0. ? static_cast<double>(
                          static_cast<long double>(flow) * flow *
                          preview.totals[3 * system + 2] / demand / demand) :
                             0.;
        valid =
          valid && std::isfinite(volume) &&
          (preview.totals[3 * system] > 0. ? volume > 0. : volume >= 0.) &&
          std::isfinite(power) && power >= 0.;
      }
    return phase(
      validation,
      valid,
      "Joint physiology needs representable positive radii, flows and resistances and finite physical totals.");
  }
  GrowthExecutionResult
  GrowthExecution::commit(const std::vector<GrowthSplit> &local,
                          const GrowthDemandJointUpdate  &joint,
                          const GrowthDemandCallbacks    &callbacks)
  {
    using namespace dealii;
    AssertThrow(!staging,
                ExcMessage(
                  "A growth stage must finish before another commit."));
    GrowthExecutionResult result;
    bool                  adopted         = false;
    const auto            previous_counts = geometry.counts;
    ScopeExit restore_counts([&]() { geometry.counts = previous_counts; });
    if (work)
      geometry.counts = work;
    ScopeExit cleanup([&]() {
      if (!adopted)
        {
          discard();
          if (callbacks.discard)
            callbacks.discard();
        }
    });
    result.validation = report;
    if (!phase(result.validation,
               report.valid() && joint.validation.valid() &&
                 !joint.integration_budget_exhausted &&
                 joint.update.snapshot_revision == demand_state.revision,
               "A growth commit needs a valid joint tissue preview."))
      return result;
    const auto count =
      Utilities::MPI::sum(static_cast<std::uint64_t>(local.size()),
                          communicator);
    std::map<Id, double> new_demands;
    bool                 valid_new = true;
    for (const auto &part :
         Utilities::MPI::all_gather(communicator, joint.new_terminals))
      for (const auto &entry : part)
        valid_new = std::isfinite(entry.demand) && entry.demand > 0. &&
                    new_demands.emplace(entry.terminal, entry.demand).second &&
                    valid_new;
    auto selected = local;
    for (auto &split : selected)
      {
        const auto value = new_demands.find(split.terminal);
        if (value == new_demands.end())
          valid_new = false;
        else
          split.demand = value->second;
      }
    if (!phase(result.validation,
               valid_new && new_demands.size() == count,
               "Every selected terminal needs exactly one true joint demand."))
      return result;
    if (count == 0)
      return result;
    Timer      timer;
    const auto delta = prepare_growth_ownership(
      rows.network(), selected, geometry, communicator, work);
    result.redistribution_seconds = timer.wall_time();
    if (!delta.validation.valid())
      {
        result.validation = delta.validation;
        return result;
      }
    staging = true;
    ++view_epoch;
    old_clearance_bound = clearance_bound;
    for (const auto &s : delta.written_splits)
      release_old.emplace(s.vessel,
                          dealii::Segment<3>{s.old_points[0], s.old_points[1]});
    for (const auto &[id, v] : delta.written.vessels)
      release_trial.emplace(
        id,
        dealii::Segment<3>{delta.written.nodes.at(v.nodes[0]).position,
                           delta.written.nodes.at(v.nodes[1]).position});
    const auto measures = measure(release_trial);
    timer.restart();
    const auto &preview       = journal.prepare(delta, joint, measures);
    result.physiology_seconds = timer.wall_time();
    result.validation         = preview.validation;
    if (!result.validation.valid())
      {
        return result;
      }
    const auto native = rows.prepare(delta, journal.native_terminal_demands());
    if (!phase(result.validation,
               native.valid(),
               "The native physical row stage failed on a participant."))
      {
        return result;
      }
    std::set<Id> changed_bounds;
    for (Id id : delta.released_owned)
      if (root_bounds.count(id))
        changed_bounds.insert(id);
    for (Id id : delta.changed_owned)
      if (demand_state.owned_roots.count(id))
        changed_bounds.insert(id);
    for (const auto &[id, value] : preview.owned)
      if (demand_state.branches.at(id).topology.parent == invalid_id)
        changed_bounds.insert(id);
    update_bounds(changed_bounds, preview, true);
    if (!refresh_controls(delta, preview, result.validation) ||
        !physical_bounds(preview, result.validation))
      {
        return result;
      }
    index_touched = delta.changed_owned;
    index_touched.insert(delta.released_owned.begin(),
                         delta.released_owned.end());
    GrowthGeometryChanges geometry_changes;
    geometry_changes.changed_owned  = delta.changed_owned;
    geometry_changes.released_owned = delta.released_owned;
    spatial.update(rows.network(), delta.changed_owned, geometry_changes);
    timer.restart();
    bool valid                   = phase(result.validation,
                       certify(delta, preview, measures, result.validation),
                       "Joint geometry certification failed on a participant.");
    result.certification_seconds = timer.wall_time();
    if (valid && callbacks.prepare_adopt)
      {
        try
          {
            callbacks.prepare_adopt(joint.update.quadrature_revision,
                                    demand_state.revision + 1);
          }
        catch (const std::exception &error)
          {
            valid = false;
            result.validation.issues.push_back(
              {"tissue stage", invalid_id, error.what()});
          }
      }
    if (!phase(result.validation,
               valid && result.validation.valid(),
               "The joint physical/tissue transaction was rejected."))
      {
        return result;
      }
    for (unsigned int system = 0; system < 3; ++system)
      {
        const double flow =
          system_reference_flow(parameters, forest_systems[system]);
        result.volume[system]       = preview.totals[3 * system + 1];
        const long double demand    = preview.total_demand(system);
        result.total_demand[system] = demand;
        result.power[system] =
          demand > 0. ? static_cast<double>(
                          static_cast<long double>(flow) * flow *
                          preview.totals[3 * system + 2] / demand / demand) :
                        0.;
      }
    result.geometry_changes = std::move(geometry_changes);
    adopt();
    leaves += count;
    owned_leaves += delta.owned.terminals.size();
    adopted = true;
    if (callbacks.adopt)
      callbacks.adopt();
    if (release)
      {
        release(release_old);
        release(release_trial);
      }
    release_old.clear();
    release_trial.clear();
    result.committed = true;
    return result;
  }
  GrowthExecutionOutput
  GrowthExecution::materialize() const
  {
    AssertThrow(!staging && report.valid(),
                dealii::ExcMessage(
                  "Only accepted growth state can be materialized."));
    auto result = rows.network();
    count_work(work,
               GrowthOperation::physical_copy_attempts,
               result.nodes.size() + result.vessels.size() +
                 result.terminals.size());
    const auto rank = dealii::Utilities::MPI::this_mpi_process(communicator);
    std::map<Id, unsigned int> requests;
    for (const auto &[id, branch] : demand_state.branches)
      requests.emplace(id, rank);
    const auto edges =
      demand_state.physical_edges(rows.network(), accepted_preview(), requests);
    AssertThrow(edges.validation.valid(),
                dealii::ExcMessage(
                  "Accepted growth radii failed SI materialization."));
    for (const auto &[id, value] : edges.records)
      result.vessels.at(id).radii = {{value.edge.radius, value.edge.radius}};
    count_work(work,
               GrowthOperation::materialized_vessels,
               result.vessels.size());
    GrowthExecutionOutput                        output;
    std::map<unsigned int, std::map<Id, double>> tags;
    output.network = redistribute_with_owners(
      geometry.cache.get_triangulation(),
      geometry.cache.get_mapping(),
      result,
      ghost_halo,
      communicator,
      [&](Id id, unsigned int owner) {
        const auto branch = demand_state.branches.find(id);
        if (branch != demand_state.branches.end())
          tags[owner].emplace(id,
                              branch->second.topology.parent == invalid_id ?
                                branch->second.topology.root_radius :
                                0.);
      },
      &geometry,
      work);
    for (const auto &[sender, records] :
         dealii::Utilities::MPI::some_to_some(communicator, tags))
      for (const auto &[id, root_radius] : records)
        {
          output.mutable_owned.insert(id);
          if (root_radius > 0.)
            output.fixed_owned_roots.emplace(id, root_radius);
        }
    return output;
  }
} // namespace ImmersX::Vascularization::internal
