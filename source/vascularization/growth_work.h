// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_work_h
#define immersx_internal_growth_work_h
#include <immersx/vascularization/growth_work.h>
#include <immersx/vascularization/network.h>
namespace ImmersX::Vascularization::internal
{
  inline void
  count_validation(GrowthWork *work, const NetworkData &network)
  {
    count_work(work,
               GrowthOperation::validation_record_visits,
               network.nodes.size() + network.vessels.size() +
                 network.terminals.size());
  }
  inline ValidationReport
  validate_counted_data(const NetworkData &network,
                        unsigned int       size,
                        GrowthWork        *work)
  {
    count_validation(work, network);
    return validate_data(network, size);
  }
  inline void
  count_copy(GrowthWork *work, const NetworkData &network)
  {
    count_work(work,
               GrowthOperation::physical_copy_attempts,
               network.nodes.size() + network.vessels.size() +
                 network.terminals.size());
  }
} // namespace ImmersX::Vascularization::internal
#endif
