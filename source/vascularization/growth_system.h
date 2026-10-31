// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_system_h
#define immersx_internal_growth_system_h
#include <immersx/vascularization/generation.h>
namespace ImmersX::Vascularization::internal
{
  inline const CompanionParameters &
  companion_parameters(const Parameters &p, System system)
  {
    if (system == System::venous)
      return p.venous;
    else if (system == System::portal)
      return p.portal;
    AssertThrow(false,
                dealii::ExcMessage(
                  "Expected a venous or portal companion network."));
    return p.venous;
  }
  inline double
  system_reference_flow(const Parameters &p, System system)
  {
    if (system == System::venous)
      return p.reference_flow;
    else if (system == System::arterial)
      return p.reference_flow *
             (p.portal.enabled ? 1. - p.portal_flow_fraction : 1.);
    else if (system == System::portal)
      return p.reference_flow * p.portal_flow_fraction;
    AssertThrow(false, dealii::ExcMessage("Unknown vascular system."));
    return 0.;
  }
} // namespace ImmersX::Vascularization::internal
#endif
