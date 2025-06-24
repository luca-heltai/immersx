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

#ifndef utils_h
#define utils_h

#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>


using namespace dealii;

struct RefinementParameters : public ParameterAcceptor
{
  RefinementParameters()
    : ParameterAcceptor("Local refinement parameters")
  {
    this->add_parameter("Refinement strategy",
                        refinement_strategy,
                        "",
                        this->prm,
                        Patterns::Selection("space|embedded"));
    this->add_parameter("Space post-refinement cycles",
                        space_post_refinement_cycles);
    this->add_parameter("Embedded post-refinement cycles",
                        embedded_post_refinement_cycles);
    this->add_parameter("Space pre-refinement cycles",
                        space_pre_refinement_cycles);
    this->add_parameter("Embedded pre-refinement cycles",
                        embedded_pre_refinement_cycles);
  }

  std::string  refinement_strategy             = "space";
  unsigned int space_post_refinement_cycles    = 0;
  unsigned int embedded_post_refinement_cycles = 0;
  unsigned int space_pre_refinement_cycles     = 0;
  unsigned int embedded_pre_refinement_cycles  = 0;
};



template <int reduced_dim, int spacedim>
void
adjust_grids(Triangulation<spacedim, spacedim>    &space_triangulation,
             Triangulation<reduced_dim, spacedim> &embedded_triangulation,
             const RefinementParameters &parameters = RefinementParameters())
{
  Assert(
    (dynamic_cast<parallel::TriangulationBase<reduced_dim, spacedim> *>(
       &embedded_triangulation) == nullptr),
    ExcMessage(
      "The embedded triangulation must not be distributed. It will be partitioned later."));

  namespace bgi = boost::geometry::index;

  space_triangulation.refine_global(parameters.space_pre_refinement_cycles);
  embedded_triangulation.refine_global(
    parameters.embedded_pre_refinement_cycles);

  // build caches so that we can get local trees
  GridTools::Cache<spacedim, spacedim>    space_cache{space_triangulation};
  GridTools::Cache<reduced_dim, spacedim> embedded_cache{
    embedded_triangulation};

  auto refine = [&]() {
    bool done        = false;
    bool global_done = false;

    double min_embedded = 1e10;
    double max_embedded = 0;
    double min_space    = 1e10;
    double max_space    = 0;

    while (global_done == false)
      {
        done = true;

        // Bounding boxes of the space grid
        const auto &tree =
          space_cache.get_locally_owned_cell_bounding_boxes_rtree();

        const auto &embedded_tree =
          embedded_cache.get_cell_bounding_boxes_rtree();

        // bounding box
        const bool use_space    = parameters.refinement_strategy == "space";
        const bool use_embedded = parameters.refinement_strategy == "embedded";

        for (const auto &[embedded_box, embedded_cell] : embedded_tree)
          {
            const auto &[p1, p2] = embedded_box.get_boundary_points();
            const auto diameter  = p1.distance(p2);
            min_embedded         = std::min(min_embedded, diameter);
            max_embedded         = std::max(max_embedded, diameter);

            for (const auto &[space_box, space_cell] :
                 tree | bgi::adaptors::queried(bgi::intersects(embedded_box)))
              {
                const auto &[sp1, sp2]    = space_box.get_boundary_points();
                const auto space_diameter = sp1.distance(sp2);
                min_space                 = std::min(min_space, space_diameter);
                max_space                 = std::max(max_space, space_diameter);

                if (use_embedded && space_diameter < diameter)
                  {
                    embedded_cell->set_refine_flag();
                    done = false;
                  }
                if (use_space && diameter < space_diameter)
                  {
                    space_cell->set_refine_flag();
                    done = false;
                  }
              }
          }

        // Synchronize done variable across all processes, otherwise we might
        // deadlock
        global_done =
          Utilities::MPI::min(static_cast<int>(done),
                              space_triangulation.get_communicator());

        if (global_done == false)
          {
            if (use_embedded)
              {
                embedded_triangulation.execute_coarsening_and_refinement();
              }
            if (use_space)
              {
                space_triangulation.execute_coarsening_and_refinement();
              }
          }
      }

    deallog << std::setw(20) << std::left << "Min space: " << std::setw(12)
            << std::right << min_space << std::setw(20) << std::left
            << ", max space: " << std::setw(12) << std::right << max_space
            << std::setw(25) << std::left << ", min embedded: " << std::setw(12)
            << std::right << min_embedded << std::setw(25) << std::left
            << ", max embedded: " << std::setw(12) << std::right << max_embedded
            << std::endl;

    return std::make_tuple(min_space, max_space, min_embedded, max_embedded);
  };

  // Do the refinement loop once, to make sure we satisfy our criterions
  refine();


  // Pre refine the space grid according to the delta refinement
  if (parameters.space_post_refinement_cycles > 0)
    for (unsigned int i = 0; i < parameters.space_post_refinement_cycles; ++i)
      {
        const auto &tree =
          space_cache.get_locally_owned_cell_bounding_boxes_rtree();

        const auto &embedded_tree =
          embedded_cache.get_cell_bounding_boxes_rtree();

        for (const auto &[embedded_box, embedded_cell] : embedded_tree)
          for (const auto &[space_box, space_cell] :
               tree | bgi::adaptors::queried(bgi::intersects(embedded_box)))
            space_cell->set_refine_flag();
        space_triangulation.execute_coarsening_and_refinement();

        // Make sure again we satisfy our criterion after the space
        // refinement
        refine();
      }

  embedded_triangulation.refine_global(
    parameters.embedded_post_refinement_cycles);

  // Check once again we satisfy our criterion, and record min/max
  const auto [sm, sM, em, eM] = refine();


  if (Utilities::MPI::this_mpi_process(
        space_triangulation.get_communicator()) == 0)
    std::cout << "Space local min/max diameters   : " << sm << "/" << sM
              << std::endl
              << "Embedded space min/max diameters: " << em << "/" << eM
              << std::endl;
}



#endif