// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_metrics_h
#define immersx_vascularization_metrics_h

#include <immersx/vascularization/discretization.h>

namespace ImmersX::Vascularization
{
  struct CoverageParameters
  {
    void
    validate() const;

    /** Metres; zero disables coverage evaluation. */
    double       service_radius   = 0.;
    unsigned int quadrature_order = 2;
  };

  struct TissueMetrics
  {
    double background_volume    = 0.;
    double inside_vessel_volume = 0.;
    double volume_fraction      = 0.;
    /** Quadrature estimates of service-ball unions, not perfusion proofs. */
    double arterial_coverage = 0.;
    double venous_coverage   = 0.;
    /** Points within service_radius of both systems, independent of pairing. */
    double dual_coverage      = 0.;
    bool   coverage_evaluated = false;
    /** Normalization volume of the coverage quadrature, zero if disabled. */
    double coverage_quadrature_volume = 0.;
  };

  /**
   * Measure additive 1D volume fraction and, independently, geometric coverage
   * by background quadrature, normalized by its own total physical weights.
   * Domain volume uses mapping-aware GridTools::volume independently of the
   * coverage quadrature order. The background must have one reference-cell
   * type (tetrahedra or hexahedra). Service balls are centred at registered
   * tissue terminal nodes (port==none). Exterior vessels and 0D beds add no
   * volume; declared final ports are not tissue terminals. Coverage is disabled
   * when service_radius is zero; its reported zero values then mean
   * unevaluated. The supplied valid discretization must describe this
   * network/background. Current execution requires one rank.
   */
  TissueMetrics
  measure_tissue(const dealii::Triangulation<3> &background,
                 const dealii::Mapping<3>       &mapping,
                 const NetworkData              &network,
                 const DiscretizationResult     &discretization,
                 const CoverageParameters       &parameters,
                 MPI_Comm                        communicator);
} // namespace ImmersX::Vascularization
#endif
