// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_global_geometry_h
#define immersx_internal_global_geometry_h
#include <array>
#include <cmath>
#include <utility>
namespace ImmersX::Vascularization::internal
{
  /** Exact auxiliary-length equality and its seven partial derivatives.
   * Coordinate order is first endpoint, second endpoint, auxiliary length.
   */
  inline std::pair<double, std::array<double, 7>>
  global_length_constraint(const std::array<double, 7> &x)
  {
    double                h = x[6] * x[6];
    std::array<double, 7> gradient{};
    for (unsigned int d = 0; d < 3; ++d)
      {
        const double delta = x[d] - x[d + 3];
        h -= delta * delta;
        gradient[d]     = -2. * delta;
        gradient[d + 3] = 2. * delta;
      }
    gradient[6] = 2. * x[6];
    return {h, gradient};
  }
} // namespace ImmersX::Vascularization::internal
#endif
