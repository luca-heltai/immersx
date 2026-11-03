// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_integral_oracles_h
#define immersx_vascularization_integral_oracles_h

#include <deal.II/base/quadrature_lib.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>

#include <algorithm>
#include <limits>
#include <map>

namespace VascularTestOracles
{
  // The two unit-cube fixtures use this physical rule, independently of their
  // background refinement. Exhaustive classification avoids the production
  // territory index, ownership routing and incremental reassignment logic.
  template <typename Density>
  std::map<ImmersX::Vascularization::Id, double>
  unit_cube_demands(const ImmersX::Vascularization::GenerationResult &result,
                    const Density                                    &density)
  {
    using namespace ImmersX::Vascularization;
    std::map<Id, dealii::Point<3>> centres;
    for (const auto &[id, terminal] : result.network.terminals)
      {
        EXPECT_EQ(terminal.system, System::arterial);
        EXPECT_TRUE(centres
                      .emplace(terminal.territory,
                               result.network.nodes.at(terminal.node).position)
                      .second);
      }
    EXPECT_EQ(centres.size(), result.territories.size());
    for (const auto &territory : result.territories)
      EXPECT_EQ(territory.position, centres.at(territory.id));
    std::uint64_t cardinality = 0;
    for (const auto &statistics : result.growth_statistics)
      cardinality = std::max(cardinality, statistics.demand_quadrature_points);
    const dealii::QIterated<3> rule(dealii::QGauss<1>(2), 8);
    EXPECT_EQ(cardinality, rule.size());
    std::map<Id, long double> sums;
    for (const auto &[id, point] : centres)
      sums[id] = 0.;
    for (unsigned int q = 0; q < rule.size(); ++q)
      {
        const auto           &point = rule.point(q);
        std::pair<double, Id> closest{std::numeric_limits<double>::infinity(),
                                      invalid_id};
        for (const auto &[id, centre] : centres)
          closest = std::min(closest,
                             std::make_pair(point.distance_square(centre), id));
        sums.at(closest.second) +=
          static_cast<long double>(rule.weight(q)) * density(point);
      }
    std::map<Id, double> values;
    for (const auto &[id, mass] : sums)
      values[id] = static_cast<double>(mass);
    return values;
  }
} // namespace VascularTestOracles
#endif
