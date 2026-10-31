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
    GrowthDemandState                                     &state;
    GrowthWork                                            *work = nullptr;
    GrowthDemandPreview                                    staged;
    std::map<Id, double>                                   native_demands;
    bool                                                   active     = false;
    std::uint64_t                                          batch_size = 0;
    std::map<Id, std::optional<GrowthDemandState::Branch>> before;
    std::map<Id, std::optional<GrowthDemandState::DirectoryEntry>>
                                              directory_before;
    std::map<Id, std::optional<unsigned int>> roots_before;
    std::map<Id, bool>                        owned_roots_before;
    const GrowthDemandPreview      &
    prepared();
    void save(Id);
  };
} // namespace ImmersX::Vascularization::internal
#endif
