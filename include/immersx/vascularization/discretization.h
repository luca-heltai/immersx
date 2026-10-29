// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_discretization_h
#define immersx_vascularization_discretization_h

#include <deal.II/base/mpi.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/tria.h>

#include <immersx/vascularization/network.h>

namespace ImmersX::Vascularization
{
  struct DiscretizationParameters
  {
    void
    validate() const;

    double target_factor = 1.;
    double lower_factor  = .5;
    double upper_factor  = 2.;
    /** Metres; zero uses the largest background diameter for exterior parts. */
    double       exterior_length = 0.;
    unsigned int cell_budget     = 1000000;
  };

  struct DiscretizationCell
  {
    VesselCell reference;
    /** Diameters of cells crossed over positive arclength; zero outside. */
    double                      h_min  = 0.;
    double                      h_max  = 0.;
    dealii::types::subdomain_id owner  = 0;
    bool                        inside = false;
    /** An explicit exception to the desired lower length bound. */
    bool short_cell = false;
  };

  struct DiscretizationResult
  {
    std::vector<DiscretizationCell> cells;
    ValidationReport                validation;
    double                          inside_volume  = 0.;
    unsigned int                    short_cells    = 0;
    unsigned int                    exterior_cells = 0;
  };

  /**
   * Subdivide physical vessels using actual intersections with first-order
   * tetrahedral/hexahedral background cells. Preserve the complete physical
   * graph, including exterior input portions. Integrate inside volume with
   * additive vessel volumes, counting junction overlaps once per vessel.
   * Current execution requires one rank; no source geometry is changed.
   */
  DiscretizationResult
  discretize(const dealii::Triangulation<3> &background,
             const dealii::Mapping<3>       &mapping,
             const NetworkData              &network,
             const DiscretizationParameters &parameters,
             MPI_Comm                        communicator);

  using CellVessels = std::map<dealii::types::coarse_cell_id, VesselCell>;

  /**
   * Fill an empty native triangulation, sharing physical endpoint vertices at
   * junctions and retaining a separate coarse-cell-to-physical-vessel map.
   * Native neighbor(f) is not a list of all branches at a junction: use vertex
   * incidence or physical connectivity there. No physical id is narrowed into
   * a native material id or vertex index.
   */
  CellVessels
  create_triangulation(const NetworkData           &network,
                       const DiscretizationResult  &discretization,
                       dealii::Triangulation<1, 3> &triangulation);

  /** Recover the physical interval after any native binary 1D refinements. */
  VesselCell
  physical_reference(const dealii::CellId &cell,
                     const CellVessels    &coarse_cells);
} // namespace ImmersX::Vascularization
#endif
