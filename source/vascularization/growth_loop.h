// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_loop_h
#define immersx_internal_growth_loop_h
#include "growth_commit.h"
namespace ImmersX::Vascularization::internal
{
  struct GrowthRun
  {
    ValidationReport validation;
    GrowthCommit     accepted;
    std::uint64_t    sampled_points = 0, proposed_insertions = 0,
                  accepted_insertions = 0, batches = 0, rejected_batches = 0;
    unsigned int maximum_batch_insertions = 0;
  };
  // Grow an already prepared arterial forest from its REAL attachment roots.
  // The same batched execution operates with one or several processes. All
  // sampling belongs to owned background cells; no per-rank roots are added.
  // Counters for points/proposals/accepted insertions are local to producers.
  GrowthRun
  grow_forest(const dealii::Triangulation<3> &background,
              const dealii::Mapping<3>       &mapping,
              const NetworkData              &initial,
              const std::set<Id>             &mutable_owned,
              const std::map<Id, double>     &fixed_owned_roots,
              const Parameters               &parameters,
              MPI_Comm                        communicator);
} // namespace ImmersX::Vascularization::internal
#endif
