// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_geometry_measure_h
#define immersx_internal_growth_geometry_measure_h
#include <deal.II/boost_adaptors/segment.h>

#include <immersx/vascularization/network.h>

#include <functional>
#include <map>
namespace ImmersX::Vascularization::internal
{
  struct GrowthGeometryMeasure
  {
    double normalized_length = 0., radius_scale = 0., clearance_scale = 0.;
    // Normalization is defined along the whole segment. Physical domain and
    // wall certification remain separate, including under constant policies.
    bool complete = true;
  };
  using GrowthGeometrySegments = std::map<Id, dealii::Segment<3>>;
  using GrowthGeometryMeasures = std::map<Id, GrowthGeometryMeasure>;
  // Collective, even with no local requests. Measures stay on requesters.
  using GrowthGeometryMeasureFunction =
    std::function<GrowthGeometryMeasures(const GrowthGeometrySegments &)>;
  // Local cache release for replaced or rejected segment coordinates.
  using GrowthGeometryReleaseFunction =
    std::function<void(const GrowthGeometrySegments &)>;
} // namespace ImmersX::Vascularization::internal
#endif
