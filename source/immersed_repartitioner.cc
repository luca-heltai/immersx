// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

#include "immersed_repartitioner.h"

#include <deal.II/numerics/vector_tools.h>

template <int dim, int spacedim>
ImmersedRepartitioner<dim, spacedim>::ImmersedRepartitioner(
  const Triangulation<spacedim> &tria_background)
  : tria_background(tria_background)
{}

template <int dim, int spacedim>
LinearAlgebra::distributed::Vector<double>
ImmersedRepartitioner<dim, spacedim>::partition(
  const Triangulation<dim, spacedim> &tria_immersed) const
{
  // 1) collect centers of immeresed mesh
  std::vector<Point<spacedim>> points;

  for (const auto &cell : tria_immersed.active_cell_iterators())
    if (cell->is_locally_owned())
      points.push_back(cell->center());

  // 2) determine owner on background mesh
  Utilities::MPI::RemotePointEvaluation<spacedim> rpe;
  Vector<double> ranks(tria_background.n_active_cells());
  ranks = Utilities::MPI::this_mpi_process(tria_background.get_communicator());

  const auto point_ranks =
    VectorTools::point_values<1>(mapping,
                                 tria_background,
                                 ranks,
                                 points,
                                 rpe,
                                 VectorTools::EvaluationFlags::min,
                                 0);

  const auto tria =
    dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
      &tria_immersed);

  Assert(tria, ExcNotImplemented());

  // 3) set partitioning
  LinearAlgebra::distributed::Vector<double> partition(
    tria->global_active_cell_index_partitioner().lock());

  unsigned int counter = 0;
  for (const auto &cell : tria_immersed.active_cell_iterators())
    if (cell->is_locally_owned())
      partition[cell->global_active_cell_index()] = point_ranks[counter++];

  partition.update_ghost_values();

  return partition;
}


template class ImmersedRepartitioner<1, 1>;
template class ImmersedRepartitioner<1, 2>;
template class ImmersedRepartitioner<1, 3>;
template class ImmersedRepartitioner<2, 2>;
template class ImmersedRepartitioner<2, 3>;
template class ImmersedRepartitioner<3, 3>;