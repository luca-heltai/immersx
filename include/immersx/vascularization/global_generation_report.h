// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_global_generation_report_h
#define immersx_vascularization_global_generation_report_h
#include <array>
#include <cstdint>
#include <vector>
namespace ImmersX::Vascularization
{
  /** A complete three-role draft, or the geometry-certified version of it.
   * Neither stage alone declares full generation/reference validation.
   */
  enum class GlobalStage
  {
    draft,
    certified
  };

  struct GlobalGenerationBatch
  {
    unsigned int  terminals       = 0;
    std::uint64_t topology_trials = 0, topology_accepted = 0,
                  geometry_calls = 0;
    double initial_objective_W = 0., best_objective_W = 0.;
    bool   geometry_certified = false;
  };

  struct GlobalGenerationReport
  {
    std::uint64_t upfront_targets = 0, sampled_points = 0;
    double        minimum_length_m = 0., maximum_length_m = 0.;
    /** System enum order: arterial, venous, portal. */
    std::array<double, 3> endpoint_radius_bounds_m{};
    /** Temporary attachment-group views, not replicated physical ownership. */
    std::vector<std::uint64_t>         peak_group_records_by_rank;
    std::vector<GlobalGenerationBatch> history;
  };
} // namespace ImmersX::Vascularization
#endif
