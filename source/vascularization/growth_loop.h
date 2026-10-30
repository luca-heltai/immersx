// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_loop_h
#define immersx_internal_growth_loop_h
#include <immersx/vascularization/generation.h>

#include "growth_commit.h"
namespace ImmersX::Vascularization::internal
{
  struct GrowthRun
  {
    ValidationReport                         validation;
    GrowthCommit                             accepted;
    std::map<unsigned int, GrowthStatistics> statistics;
    std::uint64_t sampled_points = 0, proposed_insertions = 0,
                  accepted_insertions = 0, batches = 0, rejected_batches = 0;
    unsigned int maximum_batch_insertions = 0;
    /** Venous terminal ids on the authoritative common-territory owner. */
    std::map<Id, Id> owned_venous_terminals;
  };
  // Grow a prepared forest from its real attachment roots with the same
  // batched execution on one or several processes. Arterial sampling belongs
  // to owned background cells. Venous proposals belong to owned common tissue
  // territories, with their shared demand. No per-rank roots are added.
  // Counters for points/proposals/accepted insertions are local to producers.
  GrowthRun
  grow_forest(const dealii::Triangulation<3> &background,
              const dealii::Mapping<3>       &mapping,
              const NetworkData              &initial,
              const std::set<Id>             &mutable_owned,
              const std::map<Id, double>     &fixed_owned_roots,
              const Parameters               &parameters,
              MPI_Comm                        communicator,
              unsigned int                    terminal_count = 0,
              System                          system         = System::arterial,
              const std::vector<Territory>   &owned_territories = {});
} // namespace ImmersX::Vascularization::internal
#endif
