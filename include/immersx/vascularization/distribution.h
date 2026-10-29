// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_distribution_h
#define immersx_vascularization_distribution_h

#include <deal.II/base/mpi.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <immersx/vascularization/network.h>

namespace ImmersX::Vascularization
{
  /**
   * Redistribute authoritative physical data against a prepared background.
   * Each input entity must be stored on its declared owner; input ghosts are
   * refreshed or discarded. Nodes and terminals follow their physical point,
   * vessels their midpoint, using native ParticleHandler point search and the
   * smallest containing rank at interfaces. Exterior entities retain their
   * previous owner (terminals follow their node). Every physical vessel keeps
   * its id, endpoints, taper and properties across process boundaries.
   *
   * Retain owned data and send vessel ghosts to background covering boxes
   * intersecting the segment box expanded by max(radius)+halo. Supply wall
   * clearance plus any extra candidate-search reach as halo. This conservative
   * broad phase can span many background cells and processes; callers still
   * perform exact geometric checks. Every retained vessel carries both endpoint
   * nodes and their terminal records, if any. No complete topology is gathered
   * or broadcast. Only the background's coarse covering boxes are replicated.
   *
   * Collective, including empty ranks. Invalid ownership, duplicate owned ids,
   * missing authoritative nodes or inconsistent payloads fail on all ranks at
   * a phase boundary. Input is unchanged. Call again after geometry/radius or
   * background ownership changes to replace obsolete ghosts. Requires a
   * first-order mapping and congruent background/caller communicators.
   */
  NetworkData
  redistribute(const dealii::Triangulation<3> &background,
               const dealii::Mapping<3>       &mapping,
               const NetworkData              &initial,
               double                          halo,
               MPI_Comm                        communicator);
} // namespace ImmersX::Vascularization
#endif
