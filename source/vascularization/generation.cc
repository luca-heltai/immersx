// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/boost_adaptors/segment.h>

#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/numerics/rtree.h>

#include <immersx/vascularization/generation.h>

#include <algorithm>
#include <cmath>
#include <optional>

namespace ImmersX::Vascularization
{
  Parameters::Parameters(const std::string &section)
    : dealii::ParameterAcceptor(section)
  {
    enter_subsection("Input and ports");
    add_parameter("Seed position", seed_position, "Seed position in metres.");
    add_parameter("Seed direction",
                  seed_direction,
                  "Direction vector; zero selects an interior direction.");
    add_parameter("Seed radius",
                  seed_radius,
                  "Fixed seed radius in metres; must be supplied.");
    leave_subsection();
    enter_subsection("Growth");
    add_parameter("Seed length",
                  seed_length,
                  "Initial vessel length in metres; must be supplied.");
    add_parameter("Clearance",
                  clearance,
                  "Additional wall clearance in metres.");
    add_parameter("Geometry subdivision depth",
                  geometry_depth,
                  "Finite budget for certified boundary separation.");
    add_parameter("Geometry patch budget",
                  geometry_work,
                  "Maximum boundary patches examined for one vessel.");
    leave_subsection();
    enter_subsection("Physiology");
    add_parameter("Dynamic viscosity",
                  viscosity,
                  "Dynamic viscosity in Pa s; must be supplied.");
    leave_subsection();
  }

  void
  Parameters::validate() const
  {
    const auto positive = [](double x) { return std::isfinite(x) && x > 0.; };
    AssertThrow(
      positive(seed_radius) && positive(seed_length) && positive(viscosity),
      dealii::ExcMessage(
        "Supply positive finite seed radius, seed length and viscosity in SI units."));
    AssertThrow(
      std::isfinite(clearance) && clearance >= 0. && geometry_depth <= 30 &&
        geometry_work > 0,
      dealii::ExcMessage(
        "Clearance must be finite and nonnegative; geometry depth must be at most 30 and its patch budget positive."));
    for (unsigned int d = 0; d < 3; ++d)
      AssertThrow(
        std::isfinite(seed_position[d]) && std::isfinite(seed_direction[d]),
        dealii::ExcMessage("Seed coordinates and direction must be finite."));
  }

  namespace
  {
    using namespace dealii;
    namespace bgi = boost::geometry::index;
    using Patch   = std::vector<Point<3>>;

    double
    point_distance_square(const Point<3> &p, const Segment<3> &s)
    {
      const auto   direction = s.second - s.first;
      const double norm      = direction.norm_square();
      const double t =
        norm > 0. ? std::clamp((p - s.first) * direction / norm, 0., 1.) : 0.;
      return p.distance_square(s.first + t * direction);
    }

    // Minimize squared distance to a box on each interval on which the
    // coordinates of the closest point on the box have a fixed active set.
    double
    box_distance_square(const BoundingBox<3> &box, const Segment<3> &s)
    {
      const auto &[lo, hi]          = box.get_boundary_points();
      const auto          direction = s.second - s.first;
      std::vector<double> breaks{0., 1.};
      for (unsigned int d = 0; d < 3; ++d)
        if (direction[d] != 0.)
          for (double bound : {lo[d], hi[d]})
            {
              double t = (bound - s.first[d]) / direction[d];
              if (t > 0. && t < 1.)
                breaks.push_back(t);
            }
      std::sort(breaks.begin(), breaks.end());
      double best = std::numeric_limits<double>::infinity();
      for (std::size_t i = 1; i < breaks.size(); ++i)
        {
          const double middle = (breaks[i - 1] + breaks[i]) / 2.;
          double       a = 0., b = 0.;
          for (unsigned int d = 0; d < 3; ++d)
            {
              const double x = s.first[d] + middle * direction[d];
              if (x < lo[d] || x > hi[d])
                {
                  const double offset =
                    s.first[d] - (x < lo[d] ? lo[d] : hi[d]);
                  a += direction[d] * direction[d];
                  b += offset * direction[d];
                }
            }
          const double t =
            a > 0. ? std::clamp(-b / a, breaks[i - 1], breaks[i]) : middle;
          double square = 0.;
          for (unsigned int d = 0; d < 3; ++d)
            {
              const double x     = s.first[d] + t * direction[d];
              const double delta = x - std::clamp(x, lo[d], hi[d]);
              square += delta * delta;
            }
          best = std::min(best, square);
        }
      return best;
    }

    std::array<Patch, 4>
    children(const Patch &p)
    {
      if (p.size() == 3)
        {
          const Point<3> a = .5 * (p[0] + p[1]);
          const Point<3> b = .5 * (p[0] + p[2]);
          const Point<3> c = .5 * (p[1] + p[2]);
          return {{{p[0], a, b}, {a, p[1], c}, {b, c, p[2]}, {a, c, b}}};
        }
      AssertThrow(p.size() == 4, ExcInternalError());
      const Point<3> a = .5 * (p[0] + p[1]);
      const Point<3> b = .5 * (p[0] + p[2]);
      const Point<3> c = .5 * (p[1] + p[3]);
      const Point<3> d = .5 * (p[2] + p[3]);
      const Point<3> m = .25 * (p[0] + p[1] + p[2] + p[3]);
      return {
        {{p[0], a, b, m}, {a, p[1], m, c}, {b, m, p[2], d}, {m, c, d, p[3]}}};
    }

    bool
    separated(const Patch       &patch,
              const Segment<3>  &s,
              const double       radius,
              const unsigned int depth,
              unsigned int      &work)
    {
      if (work == 0)
        return false;
      --work;
      const BoundingBox<3> box(patch);
      // Linear triangular and bilinear quadrilateral patches are contained in
      // the convex hull of their corners. A separated box proves separation.
      const double guard = 64. * std::numeric_limits<double>::epsilon() *
                           std::max({1., s.first.norm(), s.second.norm()});
      if (box_distance_square(box, s) > (radius + guard) * (radius + guard))
        return true;
      for (const auto &p : patch)
        if (point_distance_square(p, s) <= radius * radius)
          return false;
      if (depth == 0)
        return false;
      for (const auto &child : children(patch))
        if (!separated(child, s, radius, depth - 1, work))
          return false;
      return true;
    }

    struct BackgroundGeometry
    {
      BackgroundGeometry(const Triangulation<3> &background,
                         const Mapping<3>       &mapping)
        : cache(background, mapping)
        , mapping(mapping)
      {
        const auto *q  = dynamic_cast<const MappingQ<3> *>(&mapping);
        const auto *fe = dynamic_cast<const MappingFE<3> *>(&mapping);
        AssertThrow(
          (q && q->get_degree() == 1) || (fe && fe->get_degree() == 1),
          ExcMessage("Vascular generation requires a first-order mapping."));
        std::vector<std::pair<BoundingBox<3>, std::size_t>> boxes;
        for (const auto &cell : background.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              AssertThrow(
                cell->reference_cell().is_hyper_cube() ||
                  cell->reference_cell().is_simplex(),
                ExcMessage(
                  "Only tetrahedral and hexahedral background cells are supported."));
              for (const auto face : cell->face_indices())
                if (cell->face(face)->at_boundary())
                  {
                    Patch patch;
                    for (const auto v : cell->face(face)->vertex_indices())
                      patch.push_back(cell->face(face)->vertex(v));
                    boxes.emplace_back(BoundingBox<3>(patch), patches.size());
                    patches.push_back(std::move(patch));
                  }
            }
        boundary = pack_rtree(boxes);
      }

      std::optional<Triangulation<3>::active_cell_iterator>
      containing(const Point<3> &p) const
      {
        std::optional<Triangulation<3>::active_cell_iterator> result;
        const BoundingBox<3> query(std::make_pair(p, p));
        for (const auto &[box, cell] :
             cache.get_locally_owned_cell_bounding_boxes_rtree() |
               bgi::adaptors::queried(bgi::intersects(query)))
          {
            try
              {
                const auto unit = mapping.transform_real_to_unit_cell(cell, p);
                if (cell->reference_cell().contains_point(unit, 0.) &&
                    (!result || cell->id() < (*result)->id()))
                  result = cell;
              }
            catch (const Mapping<3>::ExcTransformationFailed &)
              {}
          }
        return result;
      }

      bool
      contains(const Segment<3> &s,
               double            radius,
               unsigned int      depth,
               unsigned int      work) const
      {
        if (!containing(s.first))
          return false;
        const double guard = 64. * std::numeric_limits<double>::epsilon() *
                             std::max({1., s.first.norm(), s.second.norm()});
        Point<3> lo, hi;
        for (unsigned int d = 0; d < 3; ++d)
          {
            lo[d] = std::min(s.first[d], s.second[d]) - radius - guard;
            hi[d] = std::max(s.first[d], s.second[d]) + radius + guard;
          }
        const BoundingBox<3> query(std::make_pair(lo, hi));
        for (const auto &[box, index] :
             boundary | bgi::adaptors::queried(bgi::intersects(query)))
          if (!separated(patches[index], s, radius, depth, work))
            return false;
        // Starting inside and avoiding the complete physical boundary keeps
        // the connected swept segment inside, including in nonconvex domains.
        return true;
      }
      GridTools::Cache<3>                           cache;
      const Mapping<3>                             &mapping;
      std::vector<Patch>                            patches;
      RTree<std::pair<BoundingBox<3>, std::size_t>> boundary;
    };
  } // namespace

  GenerationResult
  generate(const dealii::Triangulation<3> &background,
           const dealii::Mapping<3>       &mapping,
           const NetworkData              &initial,
           const Parameters               &parameters,
           MPI_Comm                        communicator)
  {
    using namespace dealii;
    AssertThrow(Utilities::MPI::n_mpi_processes(communicator) == 1,
                ExcMessage(
                  "The initial seed bootstrap currently requires one rank."));
    parameters.validate();
    AssertThrow(initial.nodes.empty() && initial.vessels.empty() &&
                  initial.terminals.empty(),
                ExcMessage(
                  "The initial seed bootstrap requires an empty network."));
    GenerationResult   result;
    BackgroundGeometry geometry(background, mapping);
    const auto         cell = geometry.containing(parameters.seed_position);
    auto               fail = [&](const std::string &message) {
      result.validation.issues.push_back({"seed", 0, message});
      return result;
    };
    if (!cell)
      return fail("Seed is outside the background.");
    auto direction = parameters.seed_direction;
    if (direction.norm() == 0.)
      {
        direction = Point<3>((*cell)->center() - parameters.seed_position);
        if (direction.norm() == 0.)
          direction = Point<3>(.5 * ((*cell)->vertex(0) + (*cell)->center()) -
                               parameters.seed_position);
      }
    if (direction.norm() == 0.)
      return fail("Cannot choose an interior seed direction.");
    const auto end = parameters.seed_position +
                     parameters.seed_length / direction.norm() * direction;
    if (!geometry.contains(Segment<3>(parameters.seed_position, end),
                           parameters.seed_radius + parameters.clearance,
                           parameters.geometry_depth,
                           parameters.geometry_work))
      return fail(
        "Seed vessel wall or clearance is outside the background, or boundary separation exhausted its budget.");
    result.network.nodes[0] = {parameters.seed_position,
                               0,
                               Port::arterial_inlet,
                               Provenance::generated};
    result.network.nodes[1] = {end, 0, Port::none, Provenance::generated};
    Vessel vessel;
    vessel.nodes      = {{0, 1}};
    vessel.radii      = {{parameters.seed_radius, parameters.seed_radius}};
    vessel.viscosity  = parameters.viscosity;
    vessel.provenance = Provenance::generated;
    result.network.vessels[0]   = vessel;
    result.network.terminals[0] = {1, 0, System::arterial, 0, 1.};
    result.territories.push_back({0, end, 0, 1.});
    result.validation = validate_data(result.network, 1);
    return result;
  }
} // namespace ImmersX::Vascularization
