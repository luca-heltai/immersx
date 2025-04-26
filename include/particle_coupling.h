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

#ifndef rdl_particle_coupling_h
#define rdl_particle_coupling_h

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/utilities.h>

using namespace dealii;


/**
 * @class ParticleCouplingParameters
 * @brief Stores parameters related to particle coupling in a simulation.
 *
 * This class inherits from ParameterAcceptor and provides parameters required
 * for particle coupling algorithms, such as the extraction level for the R-tree
 * structure.
 *
 * @tparam dim The spatial dimension of the problem.
 */
template <int dim>
class ParticleCouplingParameters : public ParameterAcceptor
{
public:
  /**
   * @brief Constructor that initializes the parameters.
   */
  ParticleCouplingParameters();

  /**
   * The level of the R-tree extraction.
   *
   * This parameter controls the level of detail in the R-tree structure used
   * for particle coupling. Higher values may lead to more accurate results but
   * at the cost of increased computational complexity.
   */
  unsigned int rtree_extraction_level = 1;
};


/**
 * @class ParticleCoupling
 * @brief Manages the coupling of particles with a finite element background mesh.
 *
 * This class provides functionality to initialize and manage a particle handler
 * in the context of a finite element simulation, including outputting particle
 * data and interfacing with the background triangulation and mapping.
 *
 * @tparam dim The spatial dimension.
 */
template <int dim>
class ParticleCoupling
{
public:
  /**
   * @brief Constructor.
   * @param par Reference to the parameters governing particle coupling.
   */
  ParticleCoupling(const ParticleCouplingParameters<dim> &par);

  /**
   * @brief Outputs the current state of the particles to a file.
   * @param output_name The base name of the output file.
   */
  void
  output_particles(const std::string &output_name) const;

  /**
   * Initializes the particle handler with a background triangulation.
   *
   * @param tria_background The background triangulation.
   * @param mapping The mapping associated with the triangulation.
   */
  void
  initialize_particle_handler(
    const parallel::TriangulationBase<dim> &tria_background,
    const Mapping<dim> &mapping = StaticMappingQ1<dim>::mapping);

  /**
   * Get a covering of the background triangulation indexed by processor.
   */
  std::vector<std::vector<BoundingBox<dim>>>
  get_global_bounding_boxes() const;

  /**
   * Get the particles  triangulation.
   */
  const Particles::ParticleHandler<dim> &
  get_particles() const;

  /**
   * Insert global points into the particle handler.
   * @param points The points to be inserted.
   * @return A map of processor to local indices corresponding to the processor
   * where the local qpoints ended up being locate w.r.t. the background grid.
   */
  std::map<unsigned int, IndexSet>
  insert_points(const std::vector<Point<dim>>          &points,
                const std::vector<std::vector<double>> &properties = {});

private:
  /**
   * @brief Parameters for particle coupling.
   *
   * This object contains parameters that control the behavior of the
   * particle coupling process, such as the extraction level for the R-tree.
   */
  const ParticleCouplingParameters<dim> &par;

  /**
   * @brief Get the MPI communicator associated with the triangulation.
   * @return The MPI communicator.
   */
  MPI_Comm mpi_communicator;

  /**
   * @brief Smart pointer to the background triangulation.
   */
  SmartPointer<const parallel::TriangulationBase<dim>> tria_background;

  /**
   * @brief Smart pointer to the mapping associated with the triangulation.
   */
  SmartPointer<const Mapping<dim>> mapping;

  /**
   * A covering of the background triangulation indexed by processor.
   */
  std::vector<std::vector<BoundingBox<dim>>> global_bounding_boxes;

  /**
   * @brief Handler for managing particles in the simulation.
   */
  Particles::ParticleHandler<dim> particles;
};

#endif