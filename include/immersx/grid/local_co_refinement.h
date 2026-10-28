// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_local_co_refinement_h
#define immersx_local_co_refinement_h

#include <deal.II/base/mpi.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <boost/geometry/index/rtree.hpp>

#include <immersx/core/domain.h>
#include <immersx/coupling/immersed_repartitioner.h>
#include <immersx/grid/local_co_refinement_parameters.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace ImmersX
{
  /** Final count of intersecting bounding-box pairs outside the open interval.
   * Counts are global, including violations blocked by the strategy or level
   * cap. An empty intersection has zero violations but zero intersecting pairs.
   */
  struct LocalCoRefinementReport
  {
    unsigned long long intersecting_pairs = 0;
    unsigned long long lower_violations   = 0;
    unsigned long long upper_violations   = 0;

    bool
    satisfied() const
    {
      return lower_violations == 0 && upper_violations == 0;
    }
  };

  namespace internal
  {
    template <int d1, int d2, int spacedim>
    LocalCoRefinementReport
    co_refine_meshes(dealii::Triangulation<d1, spacedim> &space,
                     dealii::Triangulation<d2, spacedim> &embedded,
                     const RefinementParameters<d2>      &parameters,
                     const MPI_Comm                       communicator)
    {
      namespace bgi = boost::geometry::index;
      using namespace dealii;
      const bool refine_space = parameters.refinement_strategy == "space" ||
                                parameters.refinement_strategy == "both";
      const bool refine_embedded =
        parameters.refinement_strategy == "embedded" ||
        parameters.refinement_strategy == "both";
      const auto   bounds = parameters.diameter_bounds();
      const double lower  = bounds.first;
      const double upper  = bounds.second;
      AssertThrow((dynamic_cast<parallel::TriangulationBase<d2, spacedim> *>(
                     &embedded) == nullptr),
                  ExcMessage("The embedded preparation mesh must be serial."));
      space.refine_global(parameters.space_pre_refinement_cycles);
      embedded.refine_global(parameters.embedded_pre_refinement_cycles);
      detail::assert_uniform_reference_cells(space);
      detail::assert_uniform_reference_cells(embedded);
      const Mapping<d1, spacedim> &space_mapping =
        space.get_reference_cells().empty() ?
          StaticMappingQ1<d1, spacedim>::mapping :
          compatibility::linear_mapping<d1, spacedim>(
            space.get_reference_cells().front());
      const Mapping<d2, spacedim> &embedded_mapping =
        embedded.get_reference_cells().empty() ?
          StaticMappingQ1<d2, spacedim>::mapping :
          compatibility::linear_mapping<d2, spacedim>(
            embedded.get_reference_cells().front());
      GridTools::Cache<d1, spacedim> space_cache(space, space_mapping);
      GridTools::Cache<d2, spacedim> embedded_cache(embedded, embedded_mapping);
      const bool                     serial_space =
        dynamic_cast<parallel::TriangulationBase<d1, spacedim> *>(&space) ==
        nullptr;

      const auto for_pairs = [&](const auto &visit) {
        const auto &tree =
          space_cache.get_locally_owned_cell_bounding_boxes_rtree();
        const auto &embedded_tree =
          embedded_cache.get_cell_bounding_boxes_rtree();
        for (const auto &[embedded_box, embedded_cell] : embedded_tree)
          {
            const auto &[e0, e1] = embedded_box.get_boundary_points();
            const double he      = e0.distance(e1);
            for (const auto &[space_box, space_cell] :
                 tree | bgi::adaptors::queried(bgi::intersects(embedded_box)))
              {
                const auto &[s0, s1] = space_box.get_boundary_points();
                visit(space_cell, embedded_cell, s0.distance(s1), he);
              }
          }
      };

      // Every rank has the same serial mesh. Union flags, not just the stopping
      // decision: otherwise the next refinement would create different meshes.
      const auto synchronize_serial_flags = [&](auto &tria) {
        std::vector<unsigned int> flags(tria.n_active_cells(), 0);
        for (const auto &cell : tria.active_cell_iterators())
          flags[cell->active_cell_index()] = cell->refine_flag_set();
        AssertThrow(flags.size() <=
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
                    ExcMessage(
                      "Too many serial cells for MPI flag synchronization."));
        const int error = MPI_Allreduce(MPI_IN_PLACE,
                                        flags.data(),
                                        static_cast<int>(flags.size()),
                                        MPI_UNSIGNED,
                                        MPI_MAX,
                                        communicator);
        AssertThrow(error == MPI_SUCCESS,
                    ExcMessage("MPI refinement flag synchronization failed."));
        unsigned long long marked = 0;
        for (const auto &cell : tria.active_cell_iterators())
          if (flags[cell->active_cell_index()])
            {
              cell->set_refine_flag();
              ++marked;
            }
        return marked;
      };
      const auto finish_space_pass = [&]() {
        unsigned long long marked = 0;
        if (serial_space)
          marked = synchronize_serial_flags(space);
        else
          {
            for (const auto &cell : space.active_cell_iterators())
              if (cell->is_locally_owned() && cell->refine_flag_set())
                ++marked;
            marked = Utilities::MPI::sum(marked, communicator);
          }
        if (marked > 0)
          space.execute_coarsening_and_refinement();
        return marked > 0;
      };
      const auto adjust = [&]() {
        bool changed = true;
        while (changed)
          {
            changed = false;
            if (refine_space)
              {
                for_pairs(
                  [&](const auto &s, const auto &, double hs, double he) {
                    if (he <= lower * hs &&
                        s->level() < parameters.max_refinement_level)
                      s->set_refine_flag();
                  });
                changed = finish_space_pass();
              }
            if (refine_embedded)
              {
                // Query the updated space tree after its refinement, rather
                // than making both decisions against stale cell sizes.
                for_pairs(
                  [&](const auto &, const auto &e, double hs, double he) {
                    if (he >= upper * hs &&
                        e->level() < parameters.max_refinement_level)
                      e->set_refine_flag();
                  });
                if (synchronize_serial_flags(embedded) > 0)
                  {
                    embedded.execute_coarsening_and_refinement();
                    changed = true;
                  }
              }
          }
      };
      adjust();
      for (unsigned int cycle = 0;
           cycle < parameters.space_post_refinement_cycles;
           ++cycle)
        {
          for_pairs([](const auto &s, const auto &, double, double) {
            s->set_refine_flag();
          });
          finish_space_pass();
          adjust();
        }
      embedded.refine_global(parameters.embedded_post_refinement_cycles);
      adjust();

      LocalCoRefinementReport report;
      for_pairs([&](const auto &, const auto &, double hs, double he) {
        ++report.intersecting_pairs;
        report.lower_violations += he <= lower * hs;
        report.upper_violations += he >= upper * hs;
      });
      const auto global_count = [&](unsigned long long count) {
        return serial_space ? Utilities::MPI::max(count, communicator) :
                              Utilities::MPI::sum(count, communicator);
      };
      report.intersecting_pairs = global_count(report.intersecting_pairs);
      report.lower_violations   = global_count(report.lower_violations);
      report.upper_violations   = global_count(report.upper_violations);
      if (!report.satisfied() &&
          Utilities::MPI::this_mpi_process(communicator) == 0)
        std::cout
          << "Local co-refinement: remaining bounding-box pair violations: "
          << report.lower_violations << " lower, " << report.upper_violations
          << " upper (strategy or maximum level prevents further adjustment).\n";
      return report;
    }
  } // namespace internal

  /** Prepare, locally co-refine, and finalize two Domain meshes.
   *
   * The first domain is called space and the second embedded; neither role
   * depends on intrinsic dimension. Both share spacedim and congruent MPI
   * communicators. The optional immersed partitioning policy assigns embedded
   * cell centers to their background owners after finalizing space. It requires
   * a full-dimensional space and a fullydistributed embedded Domain.
   * Call run() before creating application FE spaces. Domains
   * must be empty or explicitly prepared with Domain::prepare_grid(); an
   * already finalized mesh is rejected, not replaced.
   *
   * Each domain's Initial refinement precedes these steps:
   * 1. Global pre-refinement of both meshes.
   * 2. Repeated intersection queries using two Boost rtrees supplied by
   *    GridTools::Cache. h_space and h_embedded are the diagonals of cell
   *    axis-aligned bounding boxes, not cell measures or exact intersections.
   *    The target is the open interval lower*h_space < h_embedded <
   *    upper*h_space. space refines only space at/below the lower bound;
   *    embedded refines only embedded at/above the upper bound; both executes
   *    a space pass followed by an embedded pass against the updated trees.
   *    Equality triggers refinement. No cells are coarsened.
   * 3. Each space post-refinement pass refines every space cell whose bounding
   *    box intersects an embedded cell box, then repeats step 2. Embedded
   *    post-refinement is global, followed by another complete step 2.
   * 4. Finalize both domains, partitioning and redistributing all input fields.
   *
   * The embedded preparation mesh is replicated serially on all MPI ranks;
   * its flags are unioned across ranks. fullydistributed domains also stay
   * serial until step 4. Input point fields are FE_Q(1) interpolated, input
   * cell fields are FE_DGQ(0) inherited by children. Material, boundary and
   * manifold ids follow deal.II refinement and copying. Input fields are not
   * reread from the original file after refinement.
   *
   * Max refinement level limits only the ratio-driven passes, not Initial,
   * global pre/post, or the requested local space post passes. A single-mesh
   * strategy may leave the other bound violated. A narrow interval, multiple
   * intersections with conflicting sizes, or the level cap may leave either
   * bound violated even with both. run() returns final global pair counts and
   * prints a nonfatal summary when violations remain. No intersections gives
   * a satisfied report with zero intersecting pairs, not geometric coverage.
   * Prefer upper > 2*lower for isotropic dyadic refinement; this avoids the
   * simplest size-interval incompatibility but does not guarantee feasibility.
   */
  template <int d1, int d2, int spacedim = d1>
  class LocalCoRefinement
  {
  public:
    explicit LocalCoRefinement(const RefinementParameters<d2> &parameters)
      : parameters_(parameters)
    {}

    LocalCoRefinementReport
    run(Domain<d1, spacedim> &space, Domain<d2, spacedim> &embedded) const
    {
      AssertThrow(static_cast<const void *>(&space) !=
                    static_cast<const void *>(&embedded),
                  dealii::ExcMessage(
                    "Co-refinement requires two different Domains."));
      int       comparison = MPI_UNEQUAL;
      const int error      = MPI_Comm_compare(space.communicator(),
                                         embedded.communicator(),
                                         &comparison);
      AssertThrow(error == MPI_SUCCESS &&
                    (comparison == MPI_IDENT || comparison == MPI_CONGRUENT),
                  dealii::ExcMessage(
                    "Co-refinement requires congruent MPI communicators."));
      // Reject invalid parameters before either domain is populated.
      parameters_.diameter_bounds();
      if (parameters_.embedded_partitioning_policy == "immersed")
        {
          AssertThrow(
            d1 == spacedim,
            dealii::ExcMessage(
              "Immersed partitioning requires a full-dimensional background."));
          AssertThrow(
            embedded.uses_fully_distributed_triangulation(),
            dealii::ExcMessage(
              "Immersed partitioning requires a fullydistributed embedded Domain."));
        }
      if (!space.grid_is_prepared())
        space.prepare_grid();
      if (!embedded.grid_is_prepared())
        embedded.prepare_grid(true);
      auto report =
        internal::co_refine_meshes(space.refinement_triangulation(),
                                   embedded.refinement_triangulation(),
                                   parameters_,
                                   space.communicator());
      space.finalize_grid();
      if (parameters_.embedded_partitioning_policy == "default")
        embedded.finalize_grid();
      else if (parameters_.embedded_partitioning_policy == "immersed")
        {
          if constexpr (d1 == spacedim)
            {
              const ImmersedRepartitioner<d2, spacedim> policy(
                space.triangulation());
              embedded.finalize_grid(policy);
            }
        }
      return report;
    }

  private:
    const RefinementParameters<d2> &parameters_;
  };

  /** Compatibility entry point for existing raw-triangulation coupling code.
   * Transfer hooks must already be attached to these meshes. New applications
   * should use LocalCoRefinement::run() so Domain owns the full lifecycle.
   */
  template <int reduced_dim, int spacedim>
  LocalCoRefinementReport
  adjust_grids(dealii::Triangulation<spacedim>              &space,
               dealii::Triangulation<reduced_dim, spacedim> &embedded,
               const RefinementParameters<reduced_dim>      &parameters)
  {
    AssertThrow(
      (dynamic_cast<dealii::parallel::fullydistributed::Triangulation<spacedim>
                      *>(&space) == nullptr),
      dealii::ExcMessage(
        "Use LocalCoRefinement on Domains before fullydistributed finalization."));
    return internal::co_refine_meshes(space,
                                      embedded,
                                      parameters,
                                      space.get_mpi_communicator());
  }
} // namespace ImmersX
#endif
