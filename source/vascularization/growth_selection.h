// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_selection_h
#define immersx_internal_growth_selection_h
#include "growth_batch.h"
#include "growth_execution.h"
namespace ImmersX::Vascularization::internal
{
  // Preliminary conflict vote against an accepted native snapshot. Read only
  // the proposed old vessels, their endpoint ghosts and declared native ports.
  // Adoption still requires the controller's joint physical certification.
  BatchSelection
  select_growth_batch(BackgroundGeometry &,
                      const GrowthExecution::AcceptedSnapshot &,
                      const std::vector<BatchProposal> &,
                      const Parameters &,
                      MPI_Comm);
} // namespace ImmersX::Vascularization::internal
#endif
