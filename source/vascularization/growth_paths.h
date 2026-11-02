// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_paths_h
#define immersx_internal_growth_paths_h
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
} // namespace ImmersX::Vascularization::internal
#endif
