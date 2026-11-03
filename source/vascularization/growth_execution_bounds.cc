// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <immersx/vascularization/generation.h>

#include "growth_execution.h"
namespace ImmersX::Vascularization::internal
{
  GrowthExecution::RootBound
  GrowthExecution::root_bound(Id id, const GrowthDemandPreview &preview) const
  {
    count_work(work, GrowthOperation::physiology_branch_visits);
    const auto       &summary = demand_state.summary(id, preview);
    const auto       &branch  = demand_state.branches.at(id);
    const long double radius  = branch.topology.root_radius;
    const double      minimum_radius =
      static_cast<double>(radius * summary.minimum_ratio);
    RootBound result;
    result.minimum_leaf_mass = summary.minimum_leaf_mass;
    result.system            = branch.system;
    result.valid =
      std::isfinite(minimum_radius) && minimum_radius > 0. &&
      radius * summary.minimum_scaled_ratio >= parameters.minimum_radius;
    if (parameters.reference_flow > 0.)
      {
        const long double factor =
          8.L / dealii::numbers::PI / std::pow(radius, 4);
        const double minimum_resistance =
          static_cast<double>(factor * summary.minimum_resistance_length);
        const double maximum_resistance =
          static_cast<double>(factor * summary.maximum_resistance_length);
        result.valid = result.valid && std::isfinite(minimum_resistance) &&
                       minimum_resistance > 0. &&
                       std::isfinite(maximum_resistance) &&
                       maximum_resistance > 0.;
      }
    return result;
  }

  void
  GrowthExecution::update_bounds(const std::set<Id>        &changed,
                                 const GrowthDemandPreview &preview,
                                 bool                       remember)
  {
    if (remember && bounds_touched.empty())
      invalid_bounds_before = invalid_bounds;
    for (Id id : changed)
      {
        std::optional<RootBound> value;
        if (demand_state.owned_roots.count(id))
          value = root_bound(id, preview);
        if (remember)
          {
            AssertThrow(bounds_touched.insert(id).second,
                        dealii::ExcInternalError());
          }
        const auto old = root_bounds.find(id);
        if (old != root_bounds.end())
          {
            auto minimum =
              leaf_minima[old->second.system].extract(old->second.minimum);
            AssertThrow(!minimum.empty(), dealii::ExcInternalError());
            if (remember)
              {
                // extract invalidates its iterator; keep the newly inserted
                // iterator, which remains valid when merge restores the node.
                old->second.minimum = leaf_minima_before[old->second.system]
                                        .insert(std::move(minimum))
                                        .position;
              }
            invalid_bounds -= !old->second.valid;
            if (remember)
              root_bounds_before.insert(root_bounds.extract(old));
            else
              root_bounds.erase(old);
          }
        if (value)
          {
            invalid_bounds += !value->valid;
            value->minimum = leaf_minima[value->system]
                               .emplace(value->minimum_leaf_mass, id)
                               .first;
            root_bounds.emplace(id, std::move(*value));
          }
      }
  }

  void
  GrowthExecution::discard_bounds() noexcept
  {
    for (Id id : bounds_touched)
      {
        const auto staged = root_bounds.find(id);
        if (staged != root_bounds.end())
          {
            auto &minimum = leaf_minima[staged->second.system];
            minimum.erase(staged->second.minimum);
            root_bounds.erase(staged);
          }
      }
    root_bounds.merge(root_bounds_before);
    for (unsigned int system = 0; system < 3; ++system)
      leaf_minima[system].merge(leaf_minima_before[system]);
    if (!bounds_touched.empty())
      invalid_bounds = invalid_bounds_before;
    bounds_touched.clear();
  }

  void
  GrowthExecution::adopt_bounds() noexcept
  {
    root_bounds_before.clear();
    for (auto &values : leaf_minima_before)
      values.clear();
    bounds_touched.clear();
  }
} // namespace ImmersX::Vascularization::internal
