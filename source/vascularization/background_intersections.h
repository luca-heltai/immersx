// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_background_intersections_h
#define immersx_internal_background_intersections_h
#include <deal.II/base/mpi.h>

#include <deal.II/boost_adaptors/segment.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <immersx/vascularization/network.h>

namespace ImmersX::Vascularization::internal
{
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

  // Keys are local to the requesting rank. Results partition [0,1], with h=0
  // outside. Shared positive-length intervals use the smallest containing h.
  // All ranks participate, including those with no segments or owned cells.
  std::map<Id, std::vector<BackgroundInterval>>
  intersect_background(const dealii::GridTools::Cache<3>      &cache,
                       const dealii::Mapping<3>               &mapping,
                       const std::map<Id, dealii::Segment<3>> &segments,
                       MPI_Comm                                communicator);
} // namespace ImmersX::Vascularization::internal
#endif
