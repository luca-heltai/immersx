// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_distribution_h
#define immersx_internal_distribution_h
#include <immersx/vascularization/distribution.h>
#include <immersx/vascularization/growth_work.h>

#include <functional>
namespace ImmersX::Vascularization::internal
{
  // Reports each previously owned vessel's destination before migration. The
  // callback only records routes; it must not perform collective operations.
  NetworkData
  redistribute_with_owners(const dealii::Triangulation<3> &background,
                           const dealii::Mapping<3>       &mapping,
                           const NetworkData              &initial,
                           double                          halo,
                           MPI_Comm                        communicator,
                           const std::function<void(Id, unsigned int)> &notify,
                           GrowthWork *work = nullptr);
} // namespace ImmersX::Vascularization::internal
#endif
