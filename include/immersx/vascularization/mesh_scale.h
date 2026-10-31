// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_mesh_scale_h
#define immersx_vascularization_mesh_scale_h
#include <deal.II/base/mpi.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <immersx/vascularization/network.h>

namespace ImmersX::Vascularization
{
  enum class LengthScalePolicy
  {
    meter,
    local_mesh_size,
    min_mesh_size,
    max_mesh_size
  };

  /** A fixed background's physical diameter bounds and one length policy.
   * This value converts configured coefficients, never physical input data.
   */
  struct MeshLengthScale
  {
    LengthScalePolicy policy       = LengthScalePolicy::meter;
    double            mesh_minimum = 1., mesh_maximum = 1.;
    void
    validate() const;
    double
    value(double local_diameter) const;
    double
    minimum() const;
    double
    maximum() const;
    double
    physical_length(double coefficient, double local_diameter) const;
    std::string
    name() const;
  };

  /** Collective on a congruent background communicator. Only owned active
   * cells contribute to the extrema; empty ranks participate. Reject an empty
   * background, invalid diameters or differing policy names collectively.
   */
  MeshLengthScale
  mesh_length_scale(const dealii::Triangulation<3> &background,
                    const std::string              &policy,
                    MPI_Comm                        communicator);

  struct ScaledVesselMeasure
  {
    double normalized_length = 0., inside_length = 0., exterior_length = 0.;
    double minimum_scale = 0., maximum_scale = 0.;
    double minimum_normalized_radius = 0., maximum_normalized_radius = 0.;
    bool   has_normalized_part = false;
    /** Local mesh size is undefined outside the background. Constant policies
     * normalize exterior portions too; local mode reports inside portions only.
     */
    bool complete = true;
  };

  /** Measure owned physical vessels with the same exact background crossings
   * used by discretize(). Input geometry, radii and topology remain unchanged.
   * Normalized length integrates ds/H over every crossed interval. Radius/H
   * bounds include both taper endpoints on each interval. Results remain on
   * vessel owners; no graph is collected. Configuration/data errors are
   * collective. Supports first-order tetrahedral and hexahedral mappings.
   */
  std::map<Id, ScaledVesselMeasure>
  measure_scaled_vessels(const dealii::Triangulation<3> &background,
                         const dealii::Mapping<3>       &mapping,
                         const NetworkData              &network,
                         const MeshLengthScale          &scale,
                         MPI_Comm                        communicator);
} // namespace ImmersX::Vascularization
#endif
