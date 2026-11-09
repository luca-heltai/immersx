// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_wall_certificates_h
#define immersx_internal_growth_wall_certificates_h
#include <array>
#include <map>

#include "growth_geometry.h"
namespace ImmersX::Vascularization
{
  struct Parameters;
  namespace internal
  {
    struct GrowthWallRequest
    {
      Edge edge{};
      // Physical metres, including any mesh-dependent clearance already.
      double                          padding = 0.;
      std::array<dealii::Point<3>, 2> port_normals{};
      // False requires a prior domain certificate for these exact coordinates
      // and port normals on the same fixed background. Walls are rechecked.
      bool check_domain = true;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &edge &padding &port_normals &check_domain;
      }
    };
    struct GrowthWallResult
    {
      ValidationReport   validation;
      std::map<Id, bool> local;
    };
    /** Passive geometry budgets, shared by generation and global NLP. */
    GrowthWallResult
    certify_growth_walls(BackgroundGeometry &,
                         const std::map<Id, GrowthWallRequest> &,
                         unsigned int geometry_depth,
                         unsigned int geometry_work,
                         MPI_Comm,
                         GrowthWork * = nullptr);
    // Collective even with no local requests. Keys are requester-local;
    // authoritative node/port metadata and length/radius checks belong to the
    // owning controller. No physical state or certification history is adopted.
    GrowthWallResult
    certify_growth_walls(BackgroundGeometry &,
                         const std::map<Id, GrowthWallRequest> &,
                         const Parameters &,
                         MPI_Comm,
                         GrowthWork * = nullptr);
  } // namespace internal
} // namespace ImmersX::Vascularization
#endif
