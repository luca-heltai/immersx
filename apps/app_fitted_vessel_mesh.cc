// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of ImmersX.
//
// ---------------------------------------------------------------------

// Generate a hexahedral cube mesh whose cells fit a straight vessel of radius
// R along the x axis. The cube [-l, l]^3 with a cylindrical hole comes from
// GridGenerator::plate_with_a_hole(); the hole is filled with a
// GridGenerator::subdivided_cylinder(). The default geometry is the
// MetricFlowX elastodynamics manufactured solution (issue #379).

#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace dealii;

namespace
{
  // Manifold ids of the merged coarse mesh.
  constexpr types::manifold_id wall_manifold_id = 1;
  constexpr types::manifold_id tfi_manifold_id  = 2;

  struct Options
  {
    // Half width of the cube and vessel radius sqrt(A0/pi), A0 = 3.0605e-4.
    double       half_width         = 0.120685;
    double       radius             = std::sqrt(9.7418740666549138e-5);
    double       box_factor         = 2.0;
    unsigned int axial_cells        = 0;
    unsigned int refinement         = 0;
    unsigned int vessel_material_id = 1;
    std::string  output             = "fitted_vessel.msh";
    std::string  summary;
  };

  void
  print_usage(std::ostream &out)
  {
    out
      << "Usage: fitted_vessel_mesh [options]\n"
      << "  --refinement N        global refinements of the coarse mesh "
         "(default 0)\n"
      << "  --output FILE         mesh file; the format follows the extension "
         "(msh, vtk, vtu, ...)\n"
      << "  --summary FILE        write mesh diagnostics as JSON\n"
      << "  --half-width L        cube is [-L, L]^3 (default 0.120685)\n"
      << "  --radius R            vessel radius (default sqrt(A0/pi), "
         "A0 = 3.0605e-4)\n"
      << "  --box-factor F        half width of the square around the hole, "
         "in units of R (default 2)\n"
      << "  --axial-cells N       coarse cells along the axis, even "
         "(default: about 2L/(F R))\n"
      << "  --vessel-material-id  material id of cells inside the vessel "
         "(default 1)\n";
  }

  Options
  parse_options(int argc, char *argv[])
  {
    Options options;
    for (int i = 1; i < argc; ++i)
      {
        const std::string key = argv[i];
        if (key == "-h" || key == "--help")
          {
            print_usage(std::cout);
            std::exit(0);
          }
        if (i + 1 >= argc)
          throw std::invalid_argument("Missing value for option " + key);
        const std::string value = argv[++i];
        if (key == "--refinement")
          options.refinement = std::stoul(value);
        else if (key == "--output")
          options.output = value;
        else if (key == "--summary")
          options.summary = value;
        else if (key == "--half-width")
          options.half_width = std::stod(value);
        else if (key == "--radius")
          options.radius = std::stod(value);
        else if (key == "--box-factor")
          options.box_factor = std::stod(value);
        else if (key == "--axial-cells")
          options.axial_cells = std::stoul(value);
        else if (key == "--vessel-material-id")
          options.vessel_material_id = std::stoul(value);
        else
          throw std::invalid_argument("Unknown option " + key);
      }

    const double box_half_width = options.box_factor * options.radius;
    if (!(options.radius > 0.))
      throw std::invalid_argument("The radius must be positive.");
    if (!(options.box_factor > 1.))
      throw std::invalid_argument("The box factor must be larger than 1.");
    if (!(box_half_width < options.half_width))
      throw std::invalid_argument(
        "The square around the hole must lie inside the cube.");
    if (options.vessel_material_id >= numbers::invalid_material_id)
      throw std::invalid_argument("Invalid vessel material id.");

    if (options.axial_cells == 0)
      {
        // Coarse axial cells close to the size of the cells around the hole.
        const auto pairs = static_cast<unsigned int>(
          std::round(options.half_width / box_half_width));
        options.axial_cells = 2 * std::max(1u, pairs);
      }
    if (options.axial_cells % 2 != 0)
      throw std::invalid_argument("The number of axial cells must be even.");
    return options;
  }

  double
  distance_from_axis(const Point<3> &p)
  {
    return std::sqrt(p[1] * p[1] + p[2] * p[2]);
  }

  // Cube with a cylindrical hole along x. plate_with_a_hole() extrudes along
  // z; a rotation about y moves its axis to x. The inner circle of the hole
  // has eight vertices, at multiples of 45 degrees.
  void
  make_plate(const Options &options, Triangulation<3> &plate)
  {
    const double box_half_width = options.box_factor * options.radius;
    const double padding        = options.half_width - box_half_width;
    GridGenerator::plate_with_a_hole(plate,
                                     options.radius,
                                     box_half_width,
                                     padding,
                                     padding,
                                     padding,
                                     padding,
                                     Point<3>(),
                                     0,
                                     1,
                                     2. * options.half_width,
                                     options.axial_cells + 1,
                                     false);
    GridTools::rotate(Tensor<1, 3>({0., 1., 0.}), numbers::PI / 2., plate);
  }

  // Cylinder filling the hole. Its coarse cross-section has four boundary
  // edges, while the hole has eight; one refinement on the cylindrical
  // manifold gives eight wall vertices at the same angles. Flattening drops
  // the refinement hierarchy so that the cylinder can be merged.
  void
  make_vessel(const Options &options, Triangulation<3> &vessel)
  {
    Triangulation<3> cylinder;
    GridGenerator::subdivided_cylinder(cylinder,
                                       options.axial_cells / 2,
                                       options.radius,
                                       options.half_width);
    cylinder.refine_global(1);
    GridGenerator::flatten_triangulation(cylinder, vessel);
  }

  void
  set_ids_and_manifolds(const Options &options, Triangulation<3> &tria)
  {
    const double box_half_width = options.box_factor * options.radius;
    const double tolerance      = 1e-8 * options.radius;

    tria.reset_all_manifolds();
    tria.set_all_manifold_ids(numbers::flat_manifold_id);

    for (const auto &cell : tria.active_cell_iterators())
      {
        const Point<3> center = cell->center();
        if (distance_from_axis(center) < options.radius)
          cell->set_material_id(options.vessel_material_id);
        else
          cell->set_material_id(0);

        // Cells inside the square around the hole use transfinite
        // interpolation between the curved wall and their flat faces.
        if (std::abs(center[1]) < box_half_width &&
            std::abs(center[2]) < box_half_width)
          cell->set_all_manifold_ids(tfi_manifold_id);
      }

    // Faces whose vertices all lie on r = R form the vessel wall.
    for (const auto &cell : tria.active_cell_iterators())
      for (const auto f : cell->face_indices())
        {
          const auto face    = cell->face(f);
          bool       on_wall = true;
          for (const auto v : face->vertex_indices())
            on_wall = on_wall && std::abs(distance_from_axis(face->vertex(v)) -
                                          options.radius) < tolerance;
          if (on_wall)
            face->set_all_manifold_ids(wall_manifold_id);
        }

    for (const auto &cell : tria.active_cell_iterators())
      for (const auto f : cell->face_indices())
        if (cell->face(f)->at_boundary())
          cell->face(f)->set_boundary_id(0);

    tria.set_manifold(wall_manifold_id, CylindricalManifold<3>(0));
    // Adjacent outer cells see id 2 on their shared faces while the
    // transfinite charts are initialized. Give those faces their coarse
    // flat geometry until the initialized manifold replaces it below.
    tria.set_manifold(tfi_manifold_id, FlatManifold<3>());
    TransfiniteInterpolationManifold<3> tfi;
    tfi.initialize(tria);
    tria.set_manifold(tfi_manifold_id, tfi);
  }

  struct Diagnostics
  {
    unsigned int n_cells               = 0;
    unsigned int n_vessel_cells        = 0;
    unsigned int n_vertices            = 0;
    unsigned int n_wall_faces          = 0;
    unsigned int n_open_wall_faces     = 0;
    double       volume                = 0.;
    double       vessel_volume         = 0.;
    double       min_cell_measure      = std::numeric_limits<double>::max();
    double       min_diameter          = std::numeric_limits<double>::max();
    double       max_diameter          = 0.;
    double       max_wall_diameter     = 0.;
    double       max_wall_radius_error = 0.;
  };

  Diagnostics
  compute_diagnostics(const Options &options, const Triangulation<3> &tria)
  {
    Diagnostics d;
    d.n_cells    = tria.n_active_cells();
    d.n_vertices = tria.n_used_vertices();
    for (const auto &cell : tria.active_cell_iterators())
      {
        const double measure = cell->measure();
        d.volume += measure;
        d.min_cell_measure = std::min(d.min_cell_measure, measure);
        d.min_diameter     = std::min(d.min_diameter, cell->diameter());
        d.max_diameter     = std::max(d.max_diameter, cell->diameter());
        const bool inside  = cell->material_id() == options.vessel_material_id;
        if (inside)
          {
            ++d.n_vessel_cells;
            d.vessel_volume += measure;
          }

        // Count the wall from the vessel side: faces shared with an
        // exterior cell. A wall face at the boundary would mean that the
        // two meshes were not merged.
        if (!inside)
          continue;
        for (const auto f : cell->face_indices())
          {
            const auto face = cell->face(f);
            if (face->at_boundary())
              {
                if (std::abs(std::abs(face->center()[0]) - options.half_width) >
                    1e-8 * options.half_width)
                  ++d.n_open_wall_faces;
                continue;
              }
            if (cell->neighbor(f)->material_id() == options.vessel_material_id)
              continue;
            ++d.n_wall_faces;
            d.max_wall_diameter =
              std::max(d.max_wall_diameter, cell->diameter());
            for (const auto v : face->vertex_indices())
              d.max_wall_radius_error =
                std::max(d.max_wall_radius_error,
                         std::abs(distance_from_axis(face->vertex(v)) -
                                  options.radius));
          }
      }
    return d;
  }

  void
  write_summary(const Options &options, const Diagnostics &d, std::ostream &out)
  {
    const double length      = 2. * options.half_width;
    const double cube_volume = length * length * length;
    const double cylinder_volume =
      numbers::PI * options.radius * options.radius * length;
    out << std::setprecision(16) << "{\n"
        << "  \"output\": \"" << options.output << "\",\n"
        << "  \"refinement\": " << options.refinement << ",\n"
        << "  \"half_width\": " << options.half_width << ",\n"
        << "  \"radius\": " << options.radius << ",\n"
        << "  \"box_factor\": " << options.box_factor << ",\n"
        << "  \"coarse_axial_cells\": " << options.axial_cells << ",\n"
        << "  \"vessel_material_id\": " << options.vessel_material_id << ",\n"
        << "  \"cells\": " << d.n_cells << ",\n"
        << "  \"vessel_cells\": " << d.n_vessel_cells << ",\n"
        << "  \"vertices\": " << d.n_vertices << ",\n"
        << "  \"q1_dofs\": " << 3 * d.n_vertices << ",\n"
        << "  \"wall_faces\": " << d.n_wall_faces << ",\n"
        << "  \"open_wall_faces\": " << d.n_open_wall_faces << ",\n"
        << "  \"min_cell_diameter\": " << d.min_diameter << ",\n"
        << "  \"max_cell_diameter\": " << d.max_diameter << ",\n"
        << "  \"max_wall_cell_diameter_over_radius\": "
        << d.max_wall_diameter / options.radius << ",\n"
        << "  \"min_cell_measure\": " << d.min_cell_measure << ",\n"
        << "  \"volume_relative_error\": " << d.volume / cube_volume - 1.
        << ",\n"
        << "  \"vessel_volume_relative_error\": "
        << d.vessel_volume / cylinder_volume - 1. << ",\n"
        << "  \"max_wall_vertex_radius_error\": " << d.max_wall_radius_error
        << "\n"
        << "}\n";
  }
} // namespace



int
main(int argc, char *argv[])
{
  try
    {
      const Options options = parse_options(argc, argv);
      // MPI also initializes optional packages such as PETSc. These options
      // belong to this application and have already been consumed.
      argc    = 1;
      argv[1] = nullptr;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
                  ExcMessage("fitted_vessel_mesh runs on one process."));

      Triangulation<3> plate;
      Triangulation<3> vessel;
      make_plate(options, plate);
      make_vessel(options, vessel);

      // Matching vertices are closer than roundoff of the cylindrical
      // manifold; distinct vertices are at least a fraction of R apart.
      Triangulation<3> tria;
      GridGenerator::merge_triangulations({&plate, &vessel},
                                          tria,
                                          1e-6 * options.radius);
      set_ids_and_manifolds(options, tria);
      tria.refine_global(options.refinement);

      const Diagnostics diagnostics = compute_diagnostics(options, tria);
      AssertThrow(diagnostics.n_open_wall_faces == 0,
                  ExcMessage("The vessel and the plate were not merged "
                             "along the wall."));
      AssertThrow(diagnostics.min_cell_measure > 0.,
                  ExcMessage("The mesh contains a cell with nonpositive "
                             "measure."));

      std::ofstream out(options.output);
      AssertThrow(out.good(), ExcFileNotOpen(options.output));
      // GridOut uses the stream precision for vertex coordinates.
      out << std::setprecision(16);
      const auto dot = options.output.find_last_of('.');
      AssertThrow(dot != std::string::npos,
                  ExcMessage("The output file needs an extension."));
      GridOut grid_out;
      grid_out.write(tria,
                     out,
                     GridOut::parse_output_format(
                       options.output.substr(dot + 1)));

      write_summary(options, diagnostics, std::cout);
      if (!options.summary.empty())
        {
          std::ofstream summary(options.summary);
          AssertThrow(summary.good(), ExcFileNotOpen(options.summary));
          write_summary(options, diagnostics, summary);
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << "Error: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
