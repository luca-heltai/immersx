// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
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

#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/core/representation.h>

#include <cstdint>
#include <limits>
#include <map>
#include <vector>

using namespace dealii;


namespace ImmersX
{
  /**
   * @class ParticleCouplingParameters
   * @brief Stores parameters related to particle coupling in a simulation.
   *
   * This class inherits from ParameterAcceptor and provides parameters required
   * for particle coupling algorithms, such as the extraction level for the
   * R-tree structure.
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
    explicit ParticleCouplingParameters(
      const std::string &subsection = "Particle coupling");

    /**
     * The level of the R-tree extraction.
     *
     * This parameter controls the level of detail in the R-tree structure used
     * for particle coupling. Higher values may lead to more accurate results
     * but at the cost of increased computational complexity.
     */
    unsigned int rtree_extraction_level = 1;
  };


  /**
   * @class ParticleCoupling
   * @brief Manages the coupling of particles with a finite element background mesh.
   *
   * This class provides functionality to initialize and manage a particle
   * handler in the context of a finite element simulation, including outputting
   * particle data and interfacing with the background triangulation and
   * mapping.
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
     * @param n_properties Number of scalar properties stored on each particle.
     */
    void
    initialize_particle_handler(
      const parallel::TriangulationBase<dim> &tria_background,
      const Mapping<dim> &mapping      = StaticMappingQ1<dim>::mapping,
      const unsigned int  n_properties = 1);

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
     * @param properties Optional property vectors stored on inserted particles.
     * @param ids Optional particle identifiers.
     * @return A map of processor to local indices corresponding to the processor
     * where the local qpoints ended up being locate w.r.t. the background grid.
     */
    std::map<unsigned int, IndexSet>
    insert_points(const std::vector<Point<dim>>            &points,
                  const std::vector<std::vector<double>>   &properties = {},
                  const std::vector<types::particle_index> &ids        = {});

  protected:
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
    ObserverPointer<const parallel::TriangulationBase<dim>> tria_background;

    /**
     * @brief Smart pointer to the mapping associated with the triangulation.
     */
    ObserverPointer<const Mapping<dim>> mapping;

    /**
     * A covering of the background triangulation indexed by processor.
     */
    std::vector<std::vector<BoundingBox<dim>>> global_bounding_boxes;

    /**
     * @brief Handler for managing particles in the simulation.
     */
    Particles::ParticleHandler<dim> particles;
  };


  namespace detail
  {
    /** Source-side data retained for one redistributed lifted point. */
    template <int spacedim>
    struct LiftedSourceStencil
    {
      types::particle_index stable_id =
        std::numeric_limits<types::particle_index>::max();
      types::global_cell_index source_entity_id = numbers::invalid_unsigned_int;
      unsigned int    representative_qpoint     = numbers::invalid_unsigned_int;
      unsigned int    section_qpoint            = numbers::invalid_unsigned_int;
      Point<spacedim> representative_point;
      double          physical_weight    = 0.;
      unsigned int    source_rank        = numbers::invalid_unsigned_int;
      std::size_t     source_local_index = 0;
      std::vector<types::global_dof_index> source_dof_indices;
      std::vector<double>                  source_basis_values;

      template <class Archive>
      void
      serialize(Archive &archive, const unsigned int)
      {
        archive &stable_id;
        archive &source_entity_id;
        archive &representative_qpoint;
        archive &section_qpoint;
        for (unsigned int d = 0; d < spacedim; ++d)
          archive &representative_point[d];
        archive &physical_weight;
        archive &source_rank;
        archive &source_local_index;
        archive &source_dof_indices;
        archive &source_basis_values;
      }
    };
  } // namespace detail


  /**
   * Redistribute source-owned lifted quadrature to a target mesh.
   *
   * Particle ids are the stable lifted-point ids. The source stencil is sent
   * only to ranks that receive one of the source points; no source vector or
   * global stencil table is replicated.
   */
  template <int spacedim>
  class DistributedLiftedQuadrature
  {
  public:
    using Point   = RepresentationQuadraturePoint<spacedim, double>;
    using Stencil = detail::LiftedSourceStencil<spacedim>;

    explicit DistributedLiftedQuadrature(
      const ParticleCouplingParameters<spacedim> &parameters)
      : particle_coupling_(parameters)
    {}

    void
    initialize(const parallel::TriangulationBase<spacedim> &target_tria,
               const Mapping<spacedim>                     &target_mapping,
               const std::vector<Point>                    &source_points)
    {
      communicator_ = target_tria.get_mpi_communicator();
      source_ids_by_target_.clear();
      std::vector<dealii::Point<spacedim>> positions;
      std::vector<types::particle_index>   ids;
      source_stencils_.clear();
      positions.reserve(source_points.size());
      ids.reserve(source_points.size());

      for (const auto &source_point : source_points)
        {
          AssertThrow(
            source_point.stable_id != std::numeric_limits<std::uint64_t>::max(),
            ExcMessage(
              "Distributed lifted coupling requires a stable point id."));
          const auto id =
            static_cast<types::particle_index>(source_point.stable_id);
          AssertThrow(source_stencils_.find(id) == source_stencils_.end(),
                      ExcMessage("Stable lifted point ids must be unique."));

          Stencil stencil;
          stencil.stable_id             = id;
          stencil.source_entity_id      = source_point.source_entity_id;
          stencil.representative_qpoint = source_point.representative_qpoint;
          stencil.section_qpoint        = source_point.section_qpoint;
          stencil.representative_point  = source_point.representative_point;
          stencil.physical_weight       = source_point.weight;
          stencil.source_rank           = Utilities::MPI::this_mpi_process(
            target_tria.get_mpi_communicator());
          stencil.source_local_index =
            static_cast<std::size_t>(&source_point - source_points.data());
          stencil.source_dof_indices = source_point.dof_indices;
          stencil.source_basis_values.assign(source_point.basis_values.begin(),
                                             source_point.basis_values.end());
          source_stencils_.emplace(id, std::move(stencil));
          positions.push_back(source_point.point);
          ids.push_back(id);
        }

      particle_coupling_.initialize_particle_handler(target_tria,
                                                     target_mapping,
                                                     0);
      const auto receiving_ranks =
        particle_coupling_.insert_points(positions, {}, ids);

      // On a receiving rank, receiving_ranks is keyed by source rank and
      // contains source-local point indices. Exchange that request map back to
      // the source ranks, which can then send only the required stencils.
      const auto target_to_indices =
        Utilities::MPI::some_to_some(target_tria.get_mpi_communicator(),
                                     receiving_ranks);
      std::map<unsigned int, std::vector<Stencil>> stencils_by_target;
      for (const auto &[target_rank, indices] : target_to_indices)
        {
          auto &stencils       = stencils_by_target[target_rank];
          auto &ids_for_target = source_ids_by_target_[target_rank];
          stencils.reserve(indices.n_elements());
          ids_for_target.reserve(indices.n_elements());
          for (const auto index : indices)
            {
              AssertIndexRange(index, source_points.size());
              const auto id = ids[index];
              ids_for_target.push_back(id);
              stencils.push_back(source_stencils_.at(id));
            }
        }

      const auto source_to_stencils =
        Utilities::MPI::some_to_some(target_tria.get_mpi_communicator(),
                                     stencils_by_target);
      target_stencils_.clear();
      for (const auto &[unused_source_rank, stencils] : source_to_stencils)
        {
          (void)unused_source_rank;
          for (const auto &stencil : stencils)
            target_stencils_[stencil.stable_id] = stencil;
        }
    }

    std::map<types::particle_index, double>
    values_on_target(const dealii::Vector<double> &source_values) const
    {
      // This is the forward redistribution D. Geometry and source stencils
      // were exchanged by initialize(); repeated applications only exchange
      // values.
      std::map<unsigned int, std::map<types::particle_index, double>>
        values_by_target;
      for (const auto &[target_rank, ids] : source_ids_by_target_)
        for (const auto id : ids)
          values_by_target[target_rank][id] =
            source_values[source_stencils_.at(id).source_local_index];

      const auto received =
        Utilities::MPI::some_to_some(communicator_, values_by_target);
      std::map<types::particle_index, double> result;
      for (const auto &[unused_source_rank, values] : received)
        {
          (void)unused_source_rank;
          result.insert(values.begin(), values.end());
        }
      return result;
    }

    void
    add_transpose_to_source(
      const std::map<types::particle_index, double> &target_values,
      dealii::Vector<double>                        &source_values) const
    {
      // This is D^T, the transpose of the value redistribution above. It is
      // separate from the tensor-product lift transpose and does not rebuild
      // or modify the exchanged geometry stencils.
      std::map<unsigned int, std::map<types::particle_index, double>>
        values_by_source;
      for (const auto &[id, value] : target_values)
        values_by_source[target_stencils_.at(id).source_rank][id] += value;

      const auto received =
        Utilities::MPI::some_to_some(communicator_, values_by_source);
      for (const auto &[unused_target_rank, values] : received)
        {
          (void)unused_target_rank;
          for (const auto &[id, value] : values)
            source_values[source_stencils_.at(id).source_local_index] += value;
        }
    }

    const ParticleCoupling<spacedim> &
    particle_coupling() const
    {
      return particle_coupling_;
    }

    const std::map<types::particle_index, Stencil> &
    target_stencils() const
    {
      return target_stencils_;
    }

    const Stencil &
    stencil(const types::particle_index id) const
    {
      AssertThrow(target_stencils_.find(id) != target_stencils_.end(),
                  ExcMessage("No source stencil was received for particle."));
      return target_stencils_.at(id);
    }

  private:
    ParticleCoupling<spacedim>               particle_coupling_;
    MPI_Comm                                 communicator_ = MPI_COMM_WORLD;
    std::map<types::particle_index, Stencil> source_stencils_;
    std::map<types::particle_index, Stencil> target_stencils_;
    std::map<unsigned int, std::vector<types::particle_index>>
      source_ids_by_target_;
  };

} // namespace ImmersX

#endif
