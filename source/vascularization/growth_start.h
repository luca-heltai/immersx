// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_start_h
#define immersx_internal_growth_start_h
#include <immersx/vascularization/generation.h>

#include <optional>

#include "growth_commit.h"
namespace dealii
{
  template <int dim>
  class FunctionParser;
}
namespace ImmersX::Vascularization::internal
{
  // Shared deterministic direction policy for real seeds and attachments.
  std::vector<dealii::Point<3>>
  initial_root_directions(
    const dealii::Point<3>                &position,
    const dealii::Point<3>                &cell_center,
    const std::vector<dealii::Point<3>>   &cell_vertices,
    const dealii::Point<3>                &explicit_direction,
    const std::optional<dealii::Point<3>> &incoming_direction,
    const Territory                       *shared,
    double                                 separation);

  // Start exactly one arterial tree in an empty physical network. Native
  // background point ownership selects the writer; all ranks certify trials.
  GrowthCommit
  start_seed(const dealii::Triangulation<3>  &background,
             const dealii::Mapping<3>        &mapping,
             const Parameters                &parameters,
             const dealii::FunctionParser<3> &demand,
             MPI_Comm                         communicator);
} // namespace ImmersX::Vascularization::internal
#endif
