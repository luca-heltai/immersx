// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_sampling_h
#define immersx_internal_growth_sampling_h
#include <deal.II/base/exceptions.h>

#include <cmath>
namespace ImmersX::Vascularization::internal
{
  // The reference volume is the global minimum for beta>=0, maximum otherwise.
  // Its constant normalization leaves relative physical proposal density
  // intact.
  inline double
  mesh_sampling_factor(double volume, double reference_volume, double beta)
  {
    AssertThrow(std::isfinite(volume) && volume > 0. &&
                  std::isfinite(reference_volume) && reference_volume > 0. &&
                  std::isfinite(beta) &&
                  ((beta >= 0. && reference_volume <= volume) ||
                   (beta < 0. && reference_volume >= volume)),
                dealii::ExcMessage(
                  "Invalid cell-volume sampling normalization."));
    if (beta == 0.)
      return 1.;
    // A log difference avoids overflow of Vref/V for a negative exponent.
    return std::exp((beta / 3.) *
                    (std::log(reference_volume) - std::log(volume)));
  }
} // namespace ImmersX::Vascularization::internal
#endif
