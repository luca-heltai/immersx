// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_tissue_integration_h
#define immersx_internal_tissue_integration_h
#include <immersx/vascularization/generation.h>

#include "territory_demand.h"
namespace ImmersX::Vascularization::internal
{
  struct BackgroundGeometry;
  struct TissueIntegrationWork
  {
    std::uint64_t generated_points = 0, refined_cells = 0;
    // Collective depth and peak proposed size: use max, not sum across ranks.
    std::uint64_t       refinement_rounds = 0, peak_points = 0;
    TerritoryDemandWork assignments;
  };
  class TissueDemandIntegration
  {
    struct State;
    struct Journal;

  public:
    struct Preview
    {
      ValidationReport         validation;
      bool                     integration_budget_exhausted = false;
      TerritoryDemand::Preview territory;
      TissueIntegrationWork    work;

    private:
      friend class TissueDemandIntegration;
      std::shared_ptr<Journal> journal;
    };
    TissueDemandIntegration(BackgroundGeometry                &geometry,
                            const std::string                 &density,
                            const TissueIntegrationParameters &parameters,
                            unsigned int requested_terminals,
                            const std::vector<TissueCentre> &owned_centres,
                            MPI_Comm                         communicator,
                            bool require_interior_centres = true);
    ~TissueDemandIntegration();
    const ValidationReport &
    validation() const;
    bool
    integration_budget_exhausted() const;
    const TissueIntegrationWork &
    initialization_work() const;
    Preview
    preview(const std::vector<TissueCentre> &new_owned_centres,
            std::uint64_t                    snapshot_revision);
    void
    prepare_adopt(const Preview &preview);
    void
    adopt();
    void
    discard();
    std::uint64_t
    point_count() const;
    double
    total_demand() const;
    std::vector<TerminalDemandUpdate>
    owned_demands() const;

  private:
    std::unique_ptr<State> state;
  };
} // namespace ImmersX::Vascularization::internal
#endif
