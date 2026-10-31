// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_ownership_delta_h
#define immersx_internal_growth_ownership_delta_h
#include <immersx/vascularization/growth_work.h>

#include "growth_commit.h"
namespace ImmersX::Vascularization::internal
{
  struct GrowthOwnershipDelta
  {
    ValidationReport         validation;
    std::vector<GrowthSplit> global_splits, written_splits;
    // Written patches remain at the old writer; owned patches are routed to
    // native destinations with all endpoint nodes. Radii are provisional.
    NetworkData written, owned;
    // Only the two new nodes and three edited/new vessels in each split.
    std::map<Id, unsigned int> node_owners, vessel_owners;
    std::set<Id>               changed_owned, released_owned;
  };
  // Collective, nonmutating preparation. The caller checks mutability and
  // physical constraints, stages the patch, and owns collective adoption.
  GrowthOwnershipDelta
  prepare_growth_ownership(const NetworkData &,
                           const std::vector<GrowthSplit> &,
                           BackgroundGeometry &,
                           MPI_Comm,
                           GrowthWork * = nullptr);
} // namespace ImmersX::Vascularization::internal
#endif
