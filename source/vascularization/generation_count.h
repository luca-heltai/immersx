// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_generation_count_h
#define immersx_internal_generation_count_h
#include <deal.II/base/mpi.h>

#include <deal.II/grid/tria.h>
namespace ImmersX::Vascularization
{
  struct Parameters;
  unsigned int
  resolve_terminal_count(const dealii::Triangulation<3> &background,
                         const Parameters               &parameters,
                         MPI_Comm                        communicator);
} // namespace ImmersX::Vascularization
#endif
