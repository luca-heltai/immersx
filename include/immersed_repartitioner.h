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

#ifndef rdl_immersed_repartitioner_h
#define rdl_immersed_repartitioner_h

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

/**
 * @class ImmersedRepartitioner
 * @brief A class implementing a repartitioning policy for immersed
 * triangulations.
 *
 * This class provides functionality to repartition an immersed triangulation
 * based on a background triangulation. It inherits from the
 * RepartitioningPolicyTools::Base class and overrides the partition method. The
 * rank of the immersed cell is assigned to the processor that own the
 * background cell where the center of the immersed cell is located.
 *
 * @tparam dim The dimension of the immersed triangulation.
 * @tparam spacedim The dimension of the space in which the triangulation is
 *                  embedded. Defaults to the same value as `dim`.
 */
template <int dim, int spacedim = dim>
class ImmersedRepartitioner
  : public RepartitioningPolicyTools::Base<dim, spacedim>
{
public:
  /**
   * @brief Constructor for the ImmersedRepartitioner class.
   *
   * @param tria_background A reference to the background triangulation
   *                        used for repartitioning.
   */
  ImmersedRepartitioner(const Triangulation<spacedim> &tria_background);

  /**
   * @brief Repartition the given immersed triangulation.
   *
   * This method computes a new partitioning for the immersed triangulation
   * based on the background triangulation provided during construction.
   *
   * @param tria_immersed A reference to the immersed triangulation to be repartitioned.
   * @return A distributed vector containing the partitioning information.
   */
  virtual LinearAlgebra::distributed::Vector<double>
  partition(const Triangulation<dim, spacedim> &tria_immersed) const override;

private:
  /**
   * @brief A reference to the background triangulation used for repartitioning.
   */
  const Triangulation<spacedim> &tria_background;

  /**
   * @brief A mapping object for the background triangulation.
   */
  const MappingQ1<spacedim> mapping;
};

#endif