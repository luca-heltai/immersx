// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_surface_volume_mesh_internal_h
#define immersx_surface_volume_mesh_internal_h
#include <immersx/core/domain.h>

namespace ImmersX::detail
{
  void
  generate_volume_from_surface(const dealii::Triangulation<2, 3> &surface,
                               dealii::Triangulation<3>          &volume,
                               const DomainParameters<3, 3>      &parameters);
}
#endif
