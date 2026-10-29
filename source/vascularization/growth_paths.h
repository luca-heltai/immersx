// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_paths_h
#define immersx_internal_growth_paths_h
#include "forest_state.h"
#include "growth_candidate.h"
namespace ImmersX::Vascularization::internal
{
  struct GrowthCandidatePaths
  {
    ValidationReport                     validation;
    std::map<Id, GrowthCandidateContext> contexts;
    std::uint64_t                        retained_branches = 0;
    unsigned int                         rounds            = 0;
  };
  // Fetch only the union of requested ancestral paths from the current owned
  // forest snapshot. Every process participates, including those making no
  // request. New ids are provisional, globally fresh ids supplied by the
  // caller; only its selected candidate may publish them.
  GrowthCandidatePaths
  fetch_growth_candidates(const NetworkData                &network,
                          const ForestState                &forest,
                          const std::map<Id, unsigned int> &requests,
                          Id                                next_node,
                          Id                                next_vessel,
                          const Parameters                 &parameters,
                          MPI_Comm                          communicator);
} // namespace ImmersX::Vascularization::internal
#endif
