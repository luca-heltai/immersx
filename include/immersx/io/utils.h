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

#ifndef utils_h
#define utils_h

#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/patterns.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <immersx/grid/local_co_refinement.h>

#include <filesystem>

#ifdef DEAL_II_WITH_OPENCASCADE
#  include <deal.II/opencascade/manifold_lib.h>
#  include <deal.II/opencascade/utilities.h>

#  include <TopoDS.hxx>
#  include <TopoDS_Shape.hxx>
#endif

#include <boost/algorithm/string.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <tuple>


namespace ImmersX
{
  using namespace dealii;


  /**
   * Runtime dimensions used to select the corresponding statically typed
   * application implementation.
   */
  struct DimensionParameters
  {
    unsigned int dimension               = 2;
    unsigned int space_dimension         = 2;
    unsigned int reduced_dimension       = 1;
    unsigned int cross_section_dimension = 2;
  };


  /**
   * Declare the dimension selectors shared by all applications.
   */
  inline void
  declare_dimension_parameters(ParameterHandler &prm)
  {
    prm.declare_entry("dimension",
                      "2",
                      Patterns::Integer(1, 3),
                      "Dimension of the unknown field.");
    prm.declare_entry("space dimension",
                      "2",
                      Patterns::Integer(1, 3),
                      "Dimension of the embedding space.");
    prm.declare_entry("reduced dimension",
                      "1",
                      Patterns::Integer(0, 2),
                      "Dimension of the reduced or embedded object.");
    prm.declare_entry("cross section dimension",
                      "2",
                      Patterns::Integer(1, 3),
                      "Intrinsic dimension of the reference cross section.");
  }


  /**
   * Read the dimension selectors from an already parsed parameter handler.
   */
  inline DimensionParameters
  get_dimension_parameters(const ParameterHandler &prm)
  {
    return {static_cast<unsigned int>(prm.get_integer("dimension")),
            static_cast<unsigned int>(prm.get_integer("space dimension")),
            static_cast<unsigned int>(prm.get_integer("reduced dimension")),
            static_cast<unsigned int>(
              prm.get_integer("cross section dimension"))};
  }


  /**
   * Return a parameter handler to its root subsection.
   */
  inline void
  reset_parameter_handler_to_root(ParameterHandler &prm)
  {
    while (!prm.get_current_path().empty())
      prm.leave_subsection();
  }


  /**
   * Read the dimension selectors without constructing a dimension-specific
   * parameter acceptor.
   */
  inline DimensionParameters
  get_dimension_parameters(const std::string &filename)
  {
    auto &prm = ParameterAcceptor::prm;
    reset_parameter_handler_to_root(prm);
    declare_dimension_parameters(prm);

    try
      {
        prm.parse_input(filename, "", true);
      }
    catch (...)
      {
        // The permissive probe can encounter subsections that are declared only
        // after the dimension-specific parameter acceptor is constructed.
      }

    reset_parameter_handler_to_root(prm);
    return get_dimension_parameters(prm);
  }


  /**
   * Report a dimension combination for which no explicit instantiation exists.
   */
  inline void
  throw_unsupported_dimension_combination(const DimensionParameters &dimensions)
  {
    AssertThrow(false,
                ExcNotImplemented(
                  "The dimension combination (" +
                  std::to_string(dimensions.dimension) + ", " +
                  std::to_string(dimensions.space_dimension) + ", " +
                  std::to_string(dimensions.reduced_dimension) + ", " +
                  std::to_string(dimensions.cross_section_dimension) +
                  ") is not supported by this application."));
  }

  /**
   * Refine a bulk triangulation around a zero-dimensional point cloud.
   *
   * The point cloud is the only foreground geometry available for a
   * zero-dimensional representative domain. The per-point thickness supplies
   * the characteristic foreground diameter, and the refinement factor is used
   * in the same bulk-to-foreground diameter criterion as in adjust_grids().
   * Space post-refinement cycles then add the requested number of local passes
   * around the point supports after that criterion has been satisfied.
   */
  template <int spacedim>
  void
  refine_space_around_points(Triangulation<spacedim> &space_triangulation,
                             const std::vector<Point<spacedim>> &local_points,
                             const std::vector<double>          &local_scales,
                             const RefinementParameters<0>      &parameters,
                             const MPI_Comm mpi_communicator)
  {
    AssertDimension(local_points.size(), local_scales.size());
    AssertThrow(parameters.refinement_factor > 0.,
                ExcMessage("The refinement factor must be positive."));
    AssertThrow(parameters.max_refinement_level >= 0,
                ExcMessage(
                  "The maximum refinement level must be non-negative."));

    space_triangulation.refine_global(parameters.space_pre_refinement_cycles);

    const auto point_batches =
      Utilities::MPI::all_gather(mpi_communicator, local_points);
    const auto scale_batches =
      Utilities::MPI::all_gather(mpi_communicator, local_scales);
    std::vector<Point<spacedim>> points;
    std::vector<double>          scales;
    for (unsigned int rank = 0; rank < point_batches.size(); ++rank)
      {
        AssertDimension(point_batches[rank].size(), scale_batches[rank].size());
        points.insert(points.end(),
                      point_batches[rank].begin(),
                      point_batches[rank].end());
        scales.insert(scales.end(),
                      scale_batches[rank].begin(),
                      scale_batches[rank].end());
      }

    const auto point_diameter = [&scales](const unsigned int point) {
      return 2. * scales[point];
    };

    const auto mark_cells = [&](const bool enforce_diameter_ratio) {
      unsigned int n_marked = 0;
      for (const auto &cell : space_triangulation.active_cell_iterators())
        if (cell->is_locally_owned() &&
            cell->level() < parameters.max_refinement_level)
          {
            const auto cell_box              = cell->bounding_box();
            const auto &[cell_min, cell_max] = cell_box.get_boundary_points();
            const double cell_diameter       = cell_min.distance(cell_max);
            for (unsigned int point = 0; point < points.size(); ++point)
              if (cell_box.create_extended(scales[point])
                    .point_inside(points[point]) &&
                  (!enforce_diameter_ratio ||
                   parameters.refinement_factor * point_diameter(point) <
                     cell_diameter))
                {
                  cell->set_refine_flag();
                  ++n_marked;
                  break;
                }
          }
      return Utilities::MPI::sum(n_marked, mpi_communicator);
    };

    // First reproduce the original adjust_grids() criterion: refine the bulk
    // until its cells intersecting a foreground support are sufficiently small.
    unsigned int criterion_cycle = 0;
    while (true)
      {
        const auto n_refs = mark_cells(true);
        deallog << "0D space refinement criterion pass " << criterion_cycle
                << ": cells marked for refinement: " << n_refs << " out of "
                << space_triangulation.n_global_active_cells()
                << " (space cells)." << std::endl;
        if (n_refs == 0)
          break;
        const auto n_space_cells = space_triangulation.n_global_active_cells();
        space_triangulation.execute_coarsening_and_refinement();
        if (n_space_cells == space_triangulation.n_global_active_cells())
          {
            deallog
              << "0D space refinement criterion made no progress; stopping."
              << std::endl;
            break;
          }
        ++criterion_cycle;
      }

    // Keep post-refinement as an additional number of local passes, rather than
    // using it as the termination criterion for the diameter-based refinement.
    for (unsigned int cycle = 0;
         cycle < parameters.space_post_refinement_cycles;
         ++cycle)
      {
        const auto n_refs = mark_cells(false);
        deallog << "0D space post-refinement pass " << cycle
                << ": cells marked for refinement: " << n_refs << " out of "
                << space_triangulation.n_global_active_cells()
                << " (space cells)." << std::endl;
        if (n_refs == 0)
          break;
        const auto n_space_cells = space_triangulation.n_global_active_cells();
        space_triangulation.execute_coarsening_and_refinement();
        if (n_space_cells == space_triangulation.n_global_active_cells())
          {
            deallog << "0D space post-refinement made no progress; stopping."
                    << std::endl;
            break;
          }
      }
  }

  template <int dim, int spacedim>
  inline void
  read_grid_and_cad_files(const std::string            &grid_file_name,
                          const std::string            &ids_and_cad_file_names,
                          Triangulation<dim, spacedim> &tria)
  {
    GridIn<dim, spacedim> grid_in;
    grid_in.attach_triangulation(tria);
    grid_in.read(grid_file_name);

#ifdef DEAL_II_WITH_OPENCASCADE
    using map_type  = std::map<types::manifold_id, std::string>;
    using Converter = Patterns::Tools::Convert<map_type>;
    for (const auto &pair : Converter::to_value(ids_and_cad_file_names))
      {
        const auto &manifold_id   = pair.first;
        const auto &cad_file_name = pair.second;
        const auto  extension     = boost::to_lower_copy(
          cad_file_name.substr(cad_file_name.find_last_of('.') + 1));
        TopoDS_Shape shape;
        if (extension == "iges" || extension == "igs")
          shape = OpenCASCADE::read_IGES(cad_file_name);
        else if (extension == "step" || extension == "stp")
          shape = OpenCASCADE::read_STEP(cad_file_name);
        else
          AssertThrow(false,
                      ExcNotImplemented("We found an extension that we "
                                        "do not recognize as a CAD file "
                                        "extension. Bailing out."));
        const auto n_elements = OpenCASCADE::count_elements(shape);
        if ((std::get<0>(n_elements) == 0))
          tria.set_manifold(
            manifold_id,
            OpenCASCADE::ArclengthProjectionLineManifold<dim, spacedim>(shape));
        else if (spacedim == 3)
          {
            const auto t = reinterpret_cast<Triangulation<dim, 3> *>(&tria);
            t->set_manifold(manifold_id,
                            OpenCASCADE::NormalToMeshProjectionManifold<dim, 3>(
                              shape));
          }
        else
          tria.set_manifold(manifold_id,
                            OpenCASCADE::NURBSPatchManifold<dim, spacedim>(
                              TopoDS::Face(shape)));
      }
#else
    (void)ids_and_cad_file_names;
#endif
  }



  inline void
  initialize_parameters(const std::string &filename                  = "",
                        const std::string &output_filename           = "",
                        const bool         skip_undefined_parameters = false)
  {
    // Two-pass initialization:
    // 1. Parse the file ignoring undeclared entries
    // 2. From file to parameters
    // 3. Declare additional acceptors
    // 4. Parse again to create additional acceptors
    // 5. From file to parameters
    auto &prm = ParameterAcceptor::prm;
    declare_dimension_parameters(prm);
    ParameterAcceptor::declare_all_parameters(prm);

    if (!filename.empty())
      {
        try
          {
            prm.parse_input(filename, "", true);
            ParameterAcceptor::parse_all_parameters(prm);

            // The second pass parses after all dynamically created acceptors
            // have been declared. Keep strict checking as the default, while
            // allowing applications to reuse parameter files with extra keys.
            ParameterAcceptor::declare_all_parameters(prm);
            prm.parse_input(filename, "", skip_undefined_parameters);
            ParameterAcceptor::parse_all_parameters(prm);
          }
        catch (const dealii::ExcFileNotOpen &)
          {
            prm.print_parameters(filename, ParameterHandler::DefaultStyle);
            AssertThrow(false,
                        ExcMessage("You specified <" + filename +
                                   "> as input " +
                                   "parameter file, but it does not exist. " +
                                   "We created it for you."));
          }
      }

    if (!output_filename.empty() &&
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        const auto parent =
          std::filesystem::path(output_filename).parent_path();
        if (!parent.empty())
          {
            std::error_code error;
            std::filesystem::create_directories(parent, error);
            AssertThrow(!error,
                        ExcMessage(
                          "Could not create parameter output directory: " +
                          parent.string()));
          }
        prm.print_parameters(output_filename, ParameterHandler::Short);
      }
  }


  inline void
  initialize_parameters_from_string(
    const std::string &prm_content,
    const std::string &output_filename           = "",
    const bool         skip_undefined_parameters = false)
  {
    // Two-pass initialization:
    // 1) Parse the prm_content ignoring undeclared entries, so we can still
    // read
    //    parameters that control the creation of additional acceptors
    //    (e.g., material tags).
    // 2) Parse once to let acceptors create additional acceptors.
    // 3) Parse again, now that the additional acceptors exist and have
    //    declared their parameters.
    auto &prm = ParameterAcceptor::prm;
    declare_dimension_parameters(prm);
    ParameterAcceptor::declare_all_parameters(prm);

    prm.parse_input_from_string(prm_content, "", true);
    ParameterAcceptor::parse_all_parameters(prm);

    // Second pass.
    ParameterAcceptor::declare_all_parameters(prm);
    prm.parse_input_from_string(prm_content, "", skip_undefined_parameters);
    ParameterAcceptor::parse_all_parameters(prm);

    if (!output_filename.empty() &&
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        const auto parent =
          std::filesystem::path(output_filename).parent_path();
        if (!parent.empty())
          {
            std::error_code error;
            std::filesystem::create_directories(parent, error);
            AssertThrow(!error,
                        ExcMessage(
                          "Could not create parameter output directory: " +
                          parent.string()));
          }
        prm.print_parameters(output_filename, ParameterHandler::Short);
      }
  }

} // namespace ImmersX

#endif
