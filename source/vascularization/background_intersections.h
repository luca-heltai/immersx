// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_background_intersections_h
#define immersx_internal_background_intersections_h
#include <deal.II/base/mpi.h>

#include <deal.II/boost_adaptors/segment.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <immersx/vascularization/growth_work.h>
#include <immersx/vascularization/mesh_scale.h>
#include <immersx/vascularization/network.h>

#include <algorithm>

namespace ImmersX::Vascularization::internal
{
  // MappingQ's inverse iteration terminates at about 1e-11 in reference space.
  constexpr double inverse_mapping_tolerance = 1.e-11;

  template <typename ReferenceCell>
  auto
  boundary_reference_distance(const ReferenceCell    &reference,
                              unsigned int            face,
                              const dealii::Point<3> &unit,
                              int)
    -> decltype(reference.template vertex<3>(0), double())
  {
    return (unit - reference.template face_vertex_location<3>(face, 0)) *
           reference.template face_normal_vector<3>(face);
  }

  template <typename ReferenceCell>
  auto
  boundary_reference_distance(const ReferenceCell    &reference,
                              unsigned int            face,
                              const dealii::Point<3> &unit,
                              long) -> decltype(reference.vertex(0), double())
  {
    return (unit - reference.face_vertex_location(face, 0)) *
           reference.face_normal_vector(face);
  }

  inline bool
  reference_contains(const dealii::Triangulation<3>::active_cell_iterator &cell,
                     const dealii::Point<3>                               &unit,
                     double boundary_tolerance)
  {
    const auto reference = cell->reference_cell();
    if (!reference.contains_point(
          unit, std::max(boundary_tolerance, inverse_mapping_tolerance)))
      return false;
    // Allow inverse-map error across interior faces only. A physical boundary
    // keeps its strict half-space test, except at explicitly declared ports.
    for (const auto face : cell->face_indices())
      if (cell->face(face)->at_boundary() &&
          boundary_reference_distance(reference, face, unit, 0) >
            boundary_tolerance)
        return false;
    return true;
  }

  struct BackgroundInterval
  {
    double                      begin, end, h;
    dealii::types::subdomain_id owner;
    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int)
    {
      ar &begin &end &h &owner;
    }
  };

  ScaledVesselMeasure
  measure_background_profile(double                       physical_length,
                             const std::array<double, 2> &radii,
                             const std::vector<BackgroundInterval> &intervals,
                             const MeshLengthScale                 &scale);

  // Profile geometry without requesting radius bounds.
  ScaledVesselMeasure
  measure_background_profile(double physical_length,
                             const std::vector<BackgroundInterval> &intervals,
                             const MeshLengthScale                 &scale);

  // Keys are local to the requesting rank. Results partition [0,1], with h=0
  // outside. Shared positive-length intervals use the smallest containing h.
  // All ranks participate, including those with no segments or owned cells.
  std::map<Id, std::vector<BackgroundInterval>>
  intersect_background(const dealii::GridTools::Cache<3>      &cache,
                       const dealii::Mapping<3>               &mapping,
                       const std::map<Id, dealii::Segment<3>> &segments,
                       MPI_Comm                                communicator,
                       GrowthWork                             *work = nullptr);
} // namespace ImmersX::Vascularization::internal
#endif
