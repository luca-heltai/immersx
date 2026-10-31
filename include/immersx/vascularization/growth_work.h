// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_growth_work_h
#define immersx_vascularization_growth_work_h
#include <array>
#include <cstdint>
namespace ImmersX::Vascularization
{
  /** Local work, including unsuccessful attempts. No counter performs MPI. */
  enum class GrowthOperation : unsigned int
  {
    background_cache_constructions,
    background_cell_visits,
    boundary_face_visits,
    boundary_index_builds,
    network_index_builds,
    network_index_insertions,
    network_index_removals,
    network_index_queries,
    network_index_hits,
    physical_copy_attempts,
    metadata_copy_attempts,
    validation_record_visits,
    node_localizations,
    vessel_midpoint_localizations,
    nodes_migrated,
    vessels_migrated,
    terminals_migrated,
    ghost_vessels_sent,
    ghost_vessels_updated,
    ghost_vessels_removed,
    certified_vessels,
    physiology_branch_visits,
    path_records_received,
    path_exchange_rounds,
    candidate_path_visits,
    point_cell_queries,
    point_cell_hits,
    wall_queries,
    boundary_patch_tests,
    pair_checks,
    interval_cell_tests,
    interval_record_visits,
    interval_cut_edits,
    interval_cut_moves,
    materialized_vessels,
    transaction_records,
    rollback_records,
    count
  };
  inline constexpr std::array<const char *,
                              static_cast<unsigned int>(GrowthOperation::count)>
    growth_operation_names{{
      "background_cache_constructions",
      "background_cell_visits",
      "boundary_face_visits",
      "boundary_index_builds",
      "network_index_builds",
      "network_index_insertions",
      "network_index_removals",
      "network_index_queries",
      "network_index_hits",
      "physical_copy_attempts",
      "metadata_copy_attempts",
      "validation_record_visits",
      "node_localizations",
      "vessel_midpoint_localizations",
      "nodes_migrated",
      "vessels_migrated",
      "terminals_migrated",
      "ghost_vessels_sent",
      "ghost_vessels_updated",
      "ghost_vessels_removed",
      "certified_vessels",
      "physiology_branch_visits",
      "path_records_received",
      "path_exchange_rounds",
      "candidate_path_visits",
      "point_cell_queries",
      "point_cell_hits",
      "wall_queries",
      "boundary_patch_tests",
      "pair_checks",
      "interval_cell_tests",
      "interval_record_visits",
      "interval_cut_edits",
      "interval_cut_moves",
      "materialized_vessels",
      "transaction_records",
      "rollback_records",
    }};
  struct GrowthWork
  {
    std::array<std::uint64_t, static_cast<unsigned int>(GrowthOperation::count)>
      counts{};
    std::uint64_t
    operator[](GrowthOperation operation) const
    {
      return counts[static_cast<unsigned int>(operation)];
    }
    GrowthWork &
    operator+=(const GrowthWork &other)
    {
      for (unsigned int i = 0; i < counts.size(); ++i)
        counts[i] += other.counts[i];
      return *this;
    }
  };
  inline GrowthWork
  operator-(const GrowthWork &a, const GrowthWork &b)
  {
    GrowthWork result;
    for (unsigned int i = 0; i < a.counts.size(); ++i)
      result.counts[i] = a.counts[i] - b.counts[i];
    return result;
  }
  inline void
  count_work(GrowthWork     *work,
             GrowthOperation operation,
             std::uint64_t   count = 1)
  {
    if (work)
      work->counts[static_cast<unsigned int>(operation)] += count;
  }
} // namespace ImmersX::Vascularization
#endif
