// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_demand_h
#define immersx_internal_growth_demand_h
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/network.h>

#include <optional>
namespace ImmersX::Vascularization::internal
{
  using GrowthProposalKey = std::pair<unsigned int, Id>;

  struct TerminalDemandUpdate
  {
    Id     terminal = invalid_id;
    double demand   = 0.;
    template <class Archive>
    void
    serialize(Archive &ar, unsigned int)
    {
      ar &terminal &demand;
    }
  };

  // Old terminal demands are ABSOLUTE values, delivered to terminalId % P.
  // The execution directory resolves current branch/node authority. No graph
  // scan or permanently cached terminal owner is required from the tissue.
  struct GrowthDemandUpdate
  {
    std::uint64_t snapshot_revision = 0, base_quadrature_revision = 0,
                  quadrature_revision = 0;
    std::vector<TerminalDemandUpdate> terminals;
    // If present, these totals ALREADY include the proposed new terminals.
    std::optional<std::array<double, 2>> canonical_total_demand;
    template <class Archive>
    void
    serialize(Archive &ar, unsigned int)
    {
      bool canonical = canonical_total_demand.has_value();
      ar &snapshot_revision &base_quadrature_revision &quadrature_revision
        &terminals &canonical;
      if constexpr (Archive::is_loading::value)
        {
          if (canonical)
            canonical_total_demand.emplace();
          else
            canonical_total_demand.reset();
        }
      if (canonical)
        ar &(*canonical_total_demand)[0] & (*canonical_total_demand)[1];
    }
  };

  // The tissue preview precedes the choice of physical vessel and junction.
  // The key is (producer rank, producer-local proposal id); terminal ids are
  // globally allocated physical ids, independent of execution numbering.
  struct GrowthDemandProposal
  {
    GrowthProposalKey  proposal_key{dealii::numbers::invalid_unsigned_int,
                                   invalid_id};
    Id                 new_terminal_id = invalid_id;
    double             new_demand      = 0.;
    GrowthDemandUpdate update;
  };

  // Territorial reassignment is joint: new Di can differ from each isolated
  // proposal's value. New values remain on their producers until the engine
  // places them in GrowthSplit and routes the physical edit to its writer.
  struct GrowthDemandJointUpdate
  {
    GrowthDemandUpdate                update;
    std::vector<TerminalDemandUpdate> new_terminals;
  };
} // namespace ImmersX::Vascularization::internal
#endif
