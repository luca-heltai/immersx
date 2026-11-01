// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_start_h
#define immersx_internal_growth_start_h
#include <immersx/vascularization/generation.h>

#include <optional>

#include "growth_commit.h"
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

  struct GrowthPreparation
  {
    bool                 prepared = false;
    ValidationReport     validation;
    NetworkData          network;
    std::set<Id>         mutable_owned;
    std::map<Id, double> fixed_owned_roots;
  };
  // For geometrically accepted, unbranched startup subtrees with known fixed
  // radii. This establishes native authority without interpreting leaf labels
  // as integrated demands and without any physiological evaluation.
  GrowthPreparation
  prepare_growth_start(const dealii::Triangulation<3> &background,
                       const dealii::Mapping<3>       &mapping,
                       const NetworkData              &network,
                       const std::set<Id>             &mutable_owned,
                       const std::map<Id, double>     &fixed_owned_roots,
                       double                          halo,
                       MPI_Comm                        communicator,
                       BackgroundGeometry             *geometry = nullptr);
  struct AttachmentStartup
  {
    GrowthPreparation accepted;
    // Records remain at the consumed input terminal's native node owner.
    std::map<Id, Id> attachment_roots;
  };

  AttachmentStartup
  start_attachments(const dealii::Triangulation<3> &background,
                    const dealii::Mapping<3>       &mapping,
                    const NetworkData              &initial,
                    const Parameters               &parameters,
                    unsigned int                    terminal_count,
                    MPI_Comm                        communicator,
                    BackgroundGeometry             *geometry = nullptr);

  struct VenousOutlet
  {
    NetworkData      network;
    Id               port = invalid_id;
    ValidationReport validation;
  };

  // Reserve a real outlet before arterial growth allocates further node ids.
  VenousOutlet
  reserve_venous_outlet(const dealii::Triangulation<3> &background,
                        const dealii::Mapping<3>       &mapping,
                        const NetworkData              &network,
                        const Parameters               &parameters,
                        MPI_Comm                        communicator,
                        BackgroundGeometry             *geometry = nullptr);

  AttachmentStartup
  start_venous(const dealii::Triangulation<3> &background,
               const dealii::Mapping<3>       &mapping,
               const NetworkData              &network,
               const NetworkData              &original,
               const Parameters               &parameters,
               const std::vector<Territory>   &owned_territories,
               Id                              reserved_port,
               MPI_Comm                        communicator,
               BackgroundGeometry             *geometry = nullptr);

  // Start exactly one arterial tree in an empty physical network. Native
  // background point ownership selects the writer; all ranks certify trials.
  GrowthPreparation
  start_seed(const dealii::Triangulation<3> &background,
             const dealii::Mapping<3>       &mapping,
             const Parameters               &parameters,
             MPI_Comm                        communicator,
             BackgroundGeometry             *geometry = nullptr);
} // namespace ImmersX::Vascularization::internal
#endif
