// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_tissue_growth_callbacks_h
#define immersx_internal_tissue_growth_callbacks_h
#include <immersx/vascularization/generation.h>

#include "growth_demand_callbacks.h"
namespace ImmersX::Vascularization::internal
{
  struct BackgroundGeometry;
  // Geometry and communicator must outlive the returned collective callbacks.
  GrowthDemandCallbacks
  tissue_growth_callbacks(BackgroundGeometry &geometry,
                          const Parameters   &parameters,
                          unsigned int        requested_terminals,
                          MPI_Comm            communicator);
} // namespace ImmersX::Vascularization::internal
#endif
