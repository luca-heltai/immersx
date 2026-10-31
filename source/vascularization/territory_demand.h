// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_territory_demand_h
#define immersx_internal_territory_demand_h
#include <memory>

#include "growth_demand.h"
#include "tissue_demand.h"
namespace ImmersX::Vascularization::internal
{
  struct TissueCentre
  {
    Id               territory = invalid_id, terminal = invalid_id;
    dealii::Point<3> position;
    unsigned int     owner = dealii::numbers::invalid_unsigned_int;
    template <class Archive>
    void
    serialize(Archive &ar, unsigned int)
    {
      ar &territory &terminal &position &owner;
    }
  };

  // Keys identify quadrature samples only on their background-cell owner.
  struct TissueQuadratureChanges
  {
    std::vector<Id>                  removed;
    std::map<Id, TissueDemandSample> added;
  };

  struct TerritoryDemandWork
  {
    std::uint64_t nearest_queries = 0, nearest_hits = 0, influence_queries = 0,
                  influence_hits = 0, reassigned_samples = 0, added_samples = 0,
                  removed_samples = 0;
  };

  class TerritoryDemand
  {
    struct State;
    struct Journal;

  public:
    struct Preview
    {
      ValidationReport        validation;
      GrowthDemandJointUpdate demand;
      // On terminal-directory shards. Zero mass requests more quadrature;
      // it is never replaced by an arbitrary positive floor.
      std::vector<TissueCentre> unresolved;
      TerritoryDemandWork       work;

    private:
      friend class TerritoryDemand;
      std::shared_ptr<Journal> journal;
    };

    TerritoryDemand(const std::map<Id, TissueDemandSample> &owned_samples,
                    const std::vector<TissueCentre>        &owned_centres,
                    MPI_Comm                                communicator);
    ~TerritoryDemand();
    TerritoryDemand(TerritoryDemand &&) noexcept;
    TerritoryDemand &
    operator=(TerritoryDemand &&) noexcept;

    Preview
    preview(const std::vector<TissueCentre> &new_owned_centres,
            std::uint64_t                    snapshot_revision,
            const TissueQuadratureChanges   &quadrature = {}) const;
    // Stage only changed records before the final physical transaction vote.
    // No new preview or accepted-state observation is allowed while staged.
    void
    prepare_adopt(const Preview &preview);
    void
    adopt();
    void
    discard();

    // Bootstrap/final diagnostics, not per-insertion scans. Demands belong to
    // terminalId % P, independently of physical terminal migration.
    std::vector<TerminalDemandUpdate>
    owned_demands() const;
    std::map<Id, Id>
    owned_assignments() const;
    double
    total_demand() const;
    std::uint64_t
    quadrature_revision() const;

  private:
    std::unique_ptr<State> state;
  };
} // namespace ImmersX::Vascularization::internal
#endif
