// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_owned_rows_h
#define immersx_internal_growth_owned_rows_h
#include <cstdint>
#include <map>
#include <set>

#include "growth_ownership_delta.h"
namespace ImmersX::Vascularization::internal
{
  // Native physical rows for a fixed background. Vessel radii remain base
  // values; the owning execution controller supplies physical normalization.
  class GrowthOwnedRows
  {
  public:
    GrowthOwnedRows(const NetworkData &, MPI_Comm, GrowthWork * = nullptr);
    GrowthOwnedRows(const GrowthOwnedRows &) = delete;
    GrowthOwnedRows &
    operator=(const GrowthOwnedRows &) = delete;

    // Local preparation; the execution controller owns the collective vote.
    ValidationReport
    prepare(const GrowthOwnershipDelta &,
            const std::map<Id, double> &absolute_native_terminal_demands);
    void
    adopt() noexcept;
    void
    discard() noexcept;
    // Current accepted/staged base rows. The controller guards accepted views.
    const NetworkData &
    network() const noexcept;

  private:
    void add_reference(Id);
    void remove_reference(Id);
    void set_reference(Id, std::uint64_t);
    void prune_node(Id);
    void
    clear_journal() noexcept;
    std::uint64_t
    changed_count() const noexcept;

    const unsigned int          rank, size;
    GrowthWork                 *counts;
    NetworkData                 rows, before;
    std::map<Id, std::uint64_t> references, before_references;
    std::set<Id> changed_nodes, changed_vessels, changed_terminals,
      changed_references;
    bool active = false;
  };
} // namespace ImmersX::Vascularization::internal
#endif
