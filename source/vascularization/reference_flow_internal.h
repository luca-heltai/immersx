// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_reference_flow_h
#define immersx_internal_reference_flow_h

#include <immersx/vascularization/reference_flow.h>

#include <set>

namespace ImmersX::Vascularization::internal
{
  // Each selection contains only terminal ids owned by this rank. Unlike the
  // public global selection, an empty local set selects no terminal here.
  ReferenceConditions
  owned_demand_flow_conditions(const NetworkData              &network,
                               const std::map<System, double> &total_flows,
                               const std::set<Id>             &owned_selected,
                               MPI_Comm                        communicator);
} // namespace ImmersX::Vascularization::internal
#endif
