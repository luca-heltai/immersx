// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_io_h
#define immersx_vascularization_io_h
#include <immersx/vascularization/generation.h>

#include <filesystem>

namespace ImmersX::Vascularization
{
  /** Physical data restored from disk. Numerical cells and reference-solve
   * results are recomputed against the caller's prepared background. */
  struct NetworkArchive
  {
    NetworkData                  network;
    std::vector<Territory>       territories;
    std::vector<TerminalPair>    terminal_pairs;
    std::vector<TerminalTriplet> terminal_triplets;
  };

  /**
   * Save a physical network collectively to a new directory on a shared
   * filesystem. Each part holds at most 1024 owned records of each kind.
   * Owned vessels and terminals are written once; node ghosts
   * retain endpoint information. Territories and pairs belong to their
   * arterial endpoint owner and are supplied once globally. All stable ids,
   * coordinates, taper, viscosity, extra properties, port normals, provenance,
   * demands and positive bed resistances are retained.
   *
   * The target directory must not exist. A manifest is published only after
   * every part is closed successfully. A failed write may leave parts without
   * a manifest; these are not a readable archive. No graph is gathered.
   * Invalid data and filesystem failures throw collectively. Input is
   * unchanged.
   */
  void
  save_network(const std::filesystem::path     &directory,
               const NetworkData               &network,
               const std::vector<Territory>    &territories,
               const std::vector<TerminalPair> &terminal_pairs,
               MPI_Comm                         communicator);

  /** Save both two-terminal and common three-terminal compartments. */
  void
  save_network(const std::filesystem::path        &directory,
               const NetworkData                  &network,
               const std::vector<Territory>       &territories,
               const std::vector<TerminalPair>    &terminal_pairs,
               const std::vector<TerminalTriplet> &terminal_triplets,
               MPI_Comm                            communicator);

  /**
   * Write three diagnostic PVTU datasets and owned VTU pieces to a new
   * shared directory, without validating or triangulating the physical graph.
   * Crossings, coincident points, collapsed vessels and original radius traces
   * are retained. Missing or nonfinite endpoints and unknown systems are
   * omitted and listed by ID in omitted-rank.csv; diagnostic.json reports their
   * global count.
   *
   * Every cell has DiagnosticOnly=1 and the collective ValidationFailed flag.
   * Files are for inspection, not an adoptable network archive. Empty roles and
   * ranks still produce readable datasets. Requires VTK support. Filesystem
   * failures throw collectively; no graph is gathered or changed.
   */
  void
  write_diagnostic_vtk(const std::filesystem::path &directory,
                       const NetworkData           &network,
                       const ValidationReport      &validation,
                       MPI_Comm                     communicator);

  /**
   * Restore an archive with any positive communicator size, assigning bounded
   * parts cyclically to readers before native background redistribution.
   * Interior nodes/terminals follow point ownership and vessels their midpoint;
   * exterior entities retain their assigned reader. Territories and pairs move
   * to the restored arterial endpoint owner. Empty ranks participate in every
   * phase. Coarse background boxes and file counts may be replicated; the
   * complete graph is never gathered or broadcast. The directory must be
   * visible to every rank and the mapping must have degree one.
   */
  NetworkArchive
  load_network(const std::filesystem::path    &directory,
               const dealii::Triangulation<3> &background,
               const dealii::Mapping<3>       &mapping,
               double                          halo,
               MPI_Comm                        communicator);
} // namespace ImmersX::Vascularization
#endif
