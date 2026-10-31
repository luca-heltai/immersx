// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_demand_journal_h
#define immersx_internal_growth_demand_journal_h
#include "growth_demand_callbacks.h"
#include "growth_demand_state.h"
namespace ImmersX::Vascularization::internal
{
  struct GrowthOwnershipDelta;
  // Stages the accepted demand DAG in place. Only edited records and affected
  // dependencies have before-images; physical adoption remains a joint vote.
  class GrowthDemandJournal
  {
  public:
    explicit GrowthDemandJournal(GrowthDemandState &, GrowthWork * = nullptr);
    ~GrowthDemandJournal();
    GrowthDemandJournal(const GrowthDemandJournal &) = delete;
    GrowthDemandJournal &
    operator=(const GrowthDemandJournal &) = delete;
    // Measures belong to delta.written at old writers and migrate with
    // branches.
    // The returned preview is borrowed until adopt/discard or another prepare.
    const GrowthDemandPreview &
    prepare(const GrowthOwnershipDelta &,
            const GrowthDemandJointUpdate &,
            const GrowthGeometryMeasures &);
    const GrowthDemandPreview &
    preview() const;
    const std::map<Id, double> &
    native_terminal_demands() const noexcept;
    void
    adopt();
    void
    discard() noexcept;

  private:
    GrowthDemandState                              &state;
    GrowthWork                                     *work = nullptr;
    GrowthDemandPreview                             staged;
    std::map<Id, double>                            native_demands;
    bool                                            active     = false;
    std::uint64_t                                   batch_size = 0;
    std::map<Id, GrowthDemandState::Branch>         before;
    std::map<Id, GrowthDemandState::DirectoryEntry> directory_before;
    std::map<Id, unsigned int>                      roots_before;
    std::set<Id> branch_touched, directory_touched, roots_touched;
    std::set<Id> owned_roots_before, owned_roots_touched;
    const GrowthDemandPreview      &
    prepared();
    void save(Id);
    void save_root(Id);
  };
} // namespace ImmersX::Vascularization::internal
#endif
