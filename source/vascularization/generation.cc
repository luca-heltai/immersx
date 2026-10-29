// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>

#include <deal.II/boost_adaptors/segment.h>

#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/numerics/rtree.h>

#include <immersx/vascularization/generation.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <set>

namespace ImmersX::Vascularization
{
  void
  CalibrationParameters::validate() const
  {
    AssertThrow(
      std::isfinite(volume_fraction) && volume_fraction >= 0. &&
        volume_fraction <= 1. && std::isfinite(relative_tolerance) &&
        relative_tolerance > 0. && relative_tolerance < 1. &&
        std::isfinite(minimum_coverage) && minimum_coverage >= 0. &&
        minimum_coverage <= 1. && minimum_terminals > 0 &&
        maximum_terminals >= minimum_terminals && attempt_budget > 0,
      dealii::ExcMessage(
        "Invalid vascular calibration targets, tolerance, terminal limits or attempt budget."));
  }

  Parameters::Parameters(const std::string &section)
    : dealii::ParameterAcceptor(section)
  {
    enter_subsection("Objectives");
    add_parameter("Target volume fraction",
                  calibration.volume_fraction,
                  "Zero disables the volume target.");
    add_parameter("Volume relative tolerance", calibration.relative_tolerance);
    add_parameter("Minimum tissue coverage", calibration.minimum_coverage);
    add_parameter("Minimum new terminals", calibration.minimum_terminals);
    add_parameter("Maximum new terminals", calibration.maximum_terminals);
    add_parameter("Calibration attempt budget", calibration.attempt_budget);
    add_parameter("Maximum service distance",
                  coverage.service_radius,
                  "Metres; zero disables geometric coverage measurement.");
    add_parameter("Coverage quadrature order", coverage.quadrature_order);
    leave_subsection();
    enter_subsection("Discretization");
    add_parameter("Target length factor", discretization.target_factor);
    add_parameter("Lower length factor", discretization.lower_factor);
    add_parameter("Upper length factor", discretization.upper_factor);
    add_parameter("Exterior target length",
                  discretization.exterior_length,
                  "Metres; zero uses the largest background diameter.");
    add_parameter("Numerical cell budget", discretization.cell_budget);
    leave_subsection();
    enter_subsection("Input and ports");
    add_parameter(
      "Selected terminal ids",
      selected_terminals,
      "Input tissue terminal ids to extend; empty selects all available interior arterial endpoints.");
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
    enter_subsection("Sampling");
    add_parameter(
      "Terminal count",
      terminal_count,
      "Number of newly generated tissue terminals across selected attachments; unselected input terminals are excluded.");
    add_parameter("Random seed", random_seed);
    add_parameter("Sample budget per insertion", sample_budget);
    add_parameter("Density mesh exponent",
                  density_beta,
                  "Terminal density proportional to w(x)*h(x)^(-beta).");
    add_parameter(
      "Spatial weight",
      terminal_weight,
      "Nonnegative expression in x,y,z, bounded by Spatial weight bound.");
    add_parameter("Spatial weight bound", terminal_weight_bound);
    leave_subsection();
    enter_subsection("Growth");
    add_parameter("Candidate vessels", candidate_count);
    add_parameter("Local optimization steps", optimization_steps);
    add_parameter("Minimum vessel length",
                  minimum_length,
                  "Physical length in metres.");
    add_parameter("Minimum junction angle",
                  minimum_angle,
                  "Smallest angle between incident outward rays, in degrees.");
    add_parameter(
      "Dissipation weight",
      dissipation_weight,
      "Zero minimizes volume; otherwise blend volume and dissipation, each normalized by its current tree value.");
    leave_subsection();
    enter_subsection("Physiology");
    add_parameter("Arterial Murray exponent", arterial_murray_exponent);
    add_parameter("Minimum radius",
                  minimum_radius,
                  "Smallest admissible generated radius in metres.");
    add_parameter("Dynamic viscosity",
                  viscosity,
                  "Dynamic viscosity in Pa s; must be supplied.");
    leave_subsection();
  }

  void
  Parameters::validate(const bool creating_seed) const
  {
    discretization.validate();
    coverage.validate();
    calibration.validate();
    AssertThrow(
      calibration.minimum_coverage == 0. || coverage.service_radius > 0.,
      dealii::ExcMessage(
        "A positive minimum coverage requires an explicit positive maximum service distance."));
    const auto positive = [](double x) { return std::isfinite(x) && x > 0.; };
    AssertThrow(
      positive(seed_length) &&
        (creating_seed ? (positive(seed_radius) && positive(viscosity)) :
                         (std::isfinite(seed_radius) && seed_radius >= 0. &&
                          std::isfinite(viscosity) && viscosity >= 0.)),
      dealii::ExcMessage(
        "Supply a positive initial branch length in metres and, for an empty input, a positive seed radius and viscosity in SI units."));
    AssertThrow(
      std::isfinite(clearance) && clearance >= 0. && geometry_depth <= 30 &&
        geometry_work > 0,
      dealii::ExcMessage(
        "Clearance must be finite and nonnegative; geometry depth must be at most 30 and its patch budget positive."));
    AssertThrow(
      terminal_count > 0 && sample_budget > 0 && candidate_count > 0 &&
        optimization_steps <= 100,
      dealii::ExcMessage(
        "Terminal count, sample budget and candidate count must be positive; optimization steps must not exceed 100."));
    AssertThrow(
      positive(arterial_murray_exponent) && std::isfinite(minimum_radius) &&
        minimum_radius >= 0. &&
        (!creating_seed || minimum_radius <= seed_radius) &&
        std::isfinite(minimum_length) && minimum_length >= 0. &&
        std::isfinite(minimum_angle) && minimum_angle > 0. &&
        minimum_angle < 120. && std::isfinite(dissipation_weight) &&
        dissipation_weight >= 0. && dissipation_weight <= 1. &&
        std::isfinite(density_beta) && positive(terminal_weight_bound),
      dealii::ExcMessage(
        "Invalid Murray, radius, length, angle, objective or density controls."));
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
                    (!result || cell->diameter() < (*result)->diameter() ||
                     (cell->diameter() == (*result)->diameter() &&
                      cell->id() < (*result)->id())))
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
    // Boost's installed segment distance reduces to endpoint projections after
    // an intersection check. Skew 3D segments also need an interior minimum.
    // On each interval with a fixed clamped projection onto b, the residual
    // is affine in the parameter on a, so its squared norm is quadratic.
    double
    segment_distance_square(const Segment<3> &a, const Segment<3> &b)
    {
      const auto          u  = a.second - a.first;
      const auto          v  = b.second - b.first;
      const auto          w  = a.first - b.first;
      const double        vv = v.norm_square(), uv = u * v, wv = w * v;
      std::vector<double> cuts{0., 1.};
      if (uv != 0.)
        for (double t : {-wv / uv, (vv - wv) / uv})
          if (t > 0. && t < 1.)
            cuts.push_back(t);
      std::sort(cuts.begin(), cuts.end());
      double best = std::numeric_limits<double>::infinity();
      for (std::size_t i = 1; i < cuts.size(); ++i)
        {
          const double middle     = .5 * (cuts[i - 1] + cuts[i]);
          const double projection = (wv + middle * uv) / vv;
          auto         offset     = w;
          auto         slope      = u;
          if (projection >= 1.)
            offset -= v;
          else if (projection > 0. && projection < 1.)
            {
              offset -= wv / vv * v;
              slope -= uv / vv * v;
            }
          const double norm = slope.norm_square();
          const double t =
            norm > 0. ?
              std::clamp(-(offset * slope) / norm, cuts[i - 1], cuts[i]) :
              middle;
          best = std::min(best, (offset + t * slope).norm_square());
        }
      return best;
    }

    template <typename T>
    Id
    next_identifier(const std::map<Id, T> &data)
    {
      if (data.empty())
        return 0;
      AssertThrow(data.rbegin()->first < invalid_id - 1,
                  ExcMessage("Vascular identifier space exhausted."));
      return data.rbegin()->first + 1;
    }

    Id
    take_identifiers(Id &next, const Id count = 1)
    {
      AssertThrow(next <= invalid_id - count,
                  ExcMessage("Vascular identifier space exhausted."));
      const Id first = next;
      next += count;
      return first;
    }

    // For N uniform terminal demands, r_e = R*(n_e/N)^(1/gamma).
    // Volume is proportional to sum(L_e*n_e^(2/gamma))/N^(2/gamma).
    // At fixed inlet flow, dissipation replaces that exponent by 2-4/gamma.
    // Trial insertions change counts only on one ancestral path; the global
    // radius normalization is evaluated lazily without copying the network.
    struct CCOGrowth
    {
      using Leaf = std::pair<Segment<3>, Id>;
      struct Edge
      {
        Id                      id;
        std::array<Id, 2>       nodes;
        std::array<Point<3>, 2> points;
        double                  radius;
      };
      struct Tree
      {
        double radius, viscosity;
        double volume_sum, dissipation_sum;
      };
      struct Proposal
      {
        Id              vessel = invalid_id;
        Point<3>        junction, terminal;
        std::vector<Id> ancestors;
        double          volume_sum = 0., dissipation_sum = 0.;
        double          cost = std::numeric_limits<double>::infinity();
      };

      CCOGrowth(GenerationResult      &result,
                const Parameters      &parameters,
                BackgroundGeometry    &geometry,
                const std::vector<Id> &roots,
                unsigned int           terminal_count)
        : result(result)
        , network(result.network)
        , parameters(parameters)
        , terminal_count(terminal_count)
        , geometry(geometry)
        , random(parameters.random_seed)
        , exponent(2. / parameters.arterial_murray_exponent)
        , dissipation_exponent(2. - 4. / parameters.arterial_murray_exponent)
      {
        for (Id root : roots)
          {
            const auto &e = network.vessels.at(root);
            counts[root]  = 1;
            parents[root] = invalid_id;
            tree_of[root] = root;
            trees[root]   = {e.radii[0],
                             e.viscosity,
                             length(network, root),
                             length(network, root)};
            forest_volume +=
              tree_volume(trees.at(root), trees.at(root).volume_sum, 1.);
            forest_dissipation +=
              tree_dissipation(trees.at(root),
                               trees.at(root).dissipation_sum,
                               1.);
          }
        for (const auto &[id, e] : network.vessels)
          maximum_radius = std::max({maximum_radius, e.radii[0], e.radii[1]});
        next_node     = next_identifier(network.nodes);
        next_vessel   = next_identifier(network.vessels);
        next_terminal = next_identifier(network.terminals);
        for (const auto &territory : result.territories)
          next_territory = std::max(next_territory, territory.id + 1);
        weight.initialize("x,y,z", parameters.terminal_weight, {});
        bool first = true;
        for (const auto &[box, cell] :
             geometry.cache.get_locally_owned_cell_bounding_boxes_rtree())
          {
            const auto &[a, b] = box.get_boundary_points();
            for (unsigned int d = 0; d < 3; ++d)
              {
                lower[d] = first ? a[d] : std::min(lower[d], a[d]);
                upper[d] = first ? b[d] : std::max(upper[d], b[d]);
              }
            minimum_h = std::min(minimum_h, cell->diameter());
            maximum_h = std::max(maximum_h, cell->diameter());
            first     = false;
          }
        rebuild_index();
      }

      double
      uniform()
      {
        return static_cast<double>(random() >> 11) * 0x1.0p-53;
      }

      Segment<3>
      segment(Id id) const
      {
        const auto &e = network.vessels.at(id);
        return {network.nodes.at(e.nodes[0]).position,
                network.nodes.at(e.nodes[1]).position};
      }

      double
      tree_volume(const Tree &tree, double sum, double count) const
      {
        return numbers::PI * tree.radius * tree.radius * sum /
               std::pow(count, exponent);
      }

      double
      tree_dissipation(const Tree &tree, double sum, double count) const
      {
        return 8. * tree.viscosity / numbers::PI / std::pow(tree.radius, 4) *
               sum / std::pow(count, dissipation_exponent);
      }

      double
      radius(Id id) const
      {
        const auto found = tree_of.find(id);
        if (found == tree_of.end())
          return std::max(network.vessels.at(id).radii[0],
                          network.vessels.at(id).radii[1]);
        const Id root = found->second;
        return trees.at(root).radius *
               std::pow(static_cast<double>(counts.at(id)) / counts.at(root),
                        1. / parameters.arterial_murray_exponent);
      }

      void
      rebuild_index()
      {
        std::vector<Leaf> all, growing;
        for (const auto &[id, e] : network.vessels)
          {
            all.emplace_back(segment(id), id);
            const auto found = tree_of.find(id);
            if (found != tree_of.end())
              {
                const Id root = found->second;
                if (trees.at(root).radius *
                      std::pow(1. / (counts.at(root) + 1.),
                               1. / parameters.arterial_murray_exponent) >=
                    parameters.minimum_radius)
                  growing.emplace_back(segment(id), id);
              }
          }
        vessels           = pack_rtree(all);
        candidate_vessels = pack_rtree(growing);
      }

      Edge
      edge(Id id, double r) const
      {
        const auto &e = network.vessels.at(id);
        const auto  s = segment(id);
        return {id, e.nodes, {{s.first, s.second}}, r};
      }

      static bool
      compatible(const Edge &a, const Edge &b, const Parameters &parameters)
      {
        for (unsigned int i = 0; i < 2; ++i)
          for (unsigned int j = 0; j < 2; ++j)
            if (a.nodes[i] == b.nodes[j])
              {
                // Straight incident rays meet only at this junction. Their
                // lumen overlap is the permitted junction raccord.
                const auto   u      = a.points[1 - i] - a.points[i];
                const auto   v      = b.points[1 - j] - b.points[j];
                const double cosine = (u * v) / (u.norm() * v.norm());
                return cosine <=
                       std::cos(parameters.minimum_angle * numbers::PI / 180.);
              }
        const double guard = 64. * std::numeric_limits<double>::epsilon() *
                             std::max({1.,
                                       a.points[0].norm(),
                                       a.points[1].norm(),
                                       b.points[0].norm(),
                                       b.points[1].norm()});
        const double distance =
          a.radius + b.radius + parameters.clearance + guard;
        return segment_distance_square({a.points[0], a.points[1]},
                                       {b.points[0], b.points[1]}) >
               distance * distance;
      }

      std::optional<Proposal>
      propose(Id              id,
              const Point<3> &junction,
              const Point<3> &terminal,
              double          ceiling) const
      {
        Proposal p;
        p.vessel   = id;
        p.junction = junction;
        p.terminal = terminal;
        for (Id ancestor = parents.at(id); ancestor != invalid_id;
             ancestor    = parents.at(ancestor))
          p.ancestors.push_back(ancestor);
        const auto   old  = segment(id);
        const Id     root = tree_of.at(id);
        const auto  &tree = trees.at(root);
        const double n = counts.at(id), total = counts.at(root);
        const double a     = old.first.distance(junction),
                     b     = junction.distance(old.second),
                     c     = junction.distance(terminal);
        const double guard = 64. * std::numeric_limits<double>::epsilon() *
                             std::max({1., junction.norm(), terminal.norm()});
        if (std::min({a, b, c}) <= std::max(guard, parameters.minimum_length))
          return {};
        const auto sum = [&](double power, double original) {
          double value = original - length(network, id) * std::pow(n, power) +
                         a * std::pow(n + 1., power) + b * std::pow(n, power) +
                         c;
          for (Id ancestor : p.ancestors)
            value += length(network, ancestor) *
                     (std::pow(counts.at(ancestor) + 1., power) -
                      std::pow(counts.at(ancestor), power));
          return value;
        };
        p.volume_sum      = sum(exponent, tree.volume_sum);
        p.dissipation_sum = sum(dissipation_exponent, tree.dissipation_sum);
        const double vr =
          (forest_volume - tree_volume(tree, tree.volume_sum, total) +
           tree_volume(tree, p.volume_sum, total + 1.)) /
          forest_volume;
        const double dr =
          (forest_dissipation -
           tree_dissipation(tree, tree.dissipation_sum, total) +
           tree_dissipation(tree, p.dissipation_sum, total + 1.)) /
          forest_dissipation;
        p.cost = (1. - parameters.dissipation_weight) * vr +
                 parameters.dissipation_weight * dr;
        if (!std::isfinite(p.cost) || p.cost >= ceiling)
          return {};
        std::set<Id> increased(p.ancestors.begin(), p.ancestors.end());
        increased.insert(id);
        const auto proposed_radius = [&](Id e) {
          const auto found = tree_of.find(e);
          if (found == tree_of.end() || found->second != root)
            return radius(e);
          const double count = counts.at(e) + (increased.count(e) ? 1. : 0.);
          return tree.radius *
                 std::pow(count / (total + 1.),
                          1. / parameters.arterial_murray_exponent);
        };
        const double leaf_radius =
          tree.radius *
          std::pow(1. / (total + 1.), 1. / parameters.arterial_murray_exponent);
        if (leaf_radius < parameters.minimum_radius)
          return {};
        const auto        nodes = network.vessels.at(id).nodes;
        std::vector<Edge> changed{
          {id,
           {{nodes[0], next_node}},
           {{old.first, junction}},
           proposed_radius(id)},
          {next_vessel,
           {{next_node, nodes[1]}},
           {{junction, old.second}},
           tree.radius * std::pow(n / (total + 1.),
                                  1. / parameters.arterial_murray_exponent)},
          {next_vessel + 1,
           {{next_node, next_node + 1}},
           {{junction, terminal}},
           leaf_radius}};
        for (Id ancestor : p.ancestors)
          if (proposed_radius(ancestor) > radius(ancestor))
            changed.push_back(edge(ancestor, proposed_radius(ancestor)));
        std::set<Id> replaced;
        for (const auto &e : changed)
          replaced.insert(e.id);
        for (std::size_t i = 0; i < changed.size(); ++i)
          {
            const auto &e = changed[i];
            if (!geometry.contains({e.points[0], e.points[1]},
                                   e.radius + parameters.clearance,
                                   parameters.geometry_depth,
                                   parameters.geometry_work))
              return {};
            for (std::size_t j = 0; j < i; ++j)
              if (!compatible(e, changed[j], parameters))
                return {};
            Point<3>     lo, hi;
            const double pad =
              e.radius + maximum_radius + parameters.clearance + guard;
            for (unsigned int d = 0; d < 3; ++d)
              {
                lo[d] = std::min(e.points[0][d], e.points[1][d]) - pad;
                hi[d] = std::max(e.points[0][d], e.points[1][d]) + pad;
              }
            const BoundingBox<3> box(std::make_pair(lo, hi));
            for (const auto &[s, other] :
                 vessels | bgi::adaptors::queried(bgi::intersects(box)))
              if (!replaced.count(other) &&
                  !compatible(e,
                              edge(other, proposed_radius(other)),
                              parameters))
                return {};
          }
        return p;
      }

      std::optional<Proposal>
      optimize(Id id, const Point<3> &terminal) const
      {
        const auto s    = segment(id);
        auto       best = propose(id,
                            .5 * (s.first + s.second),
                            terminal,
                            std::numeric_limits<double>::infinity());
        if (!best)
          best = propose(id,
                         (s.first + s.second + terminal) / 3.,
                         terminal,
                         std::numeric_limits<double>::infinity());
        if (!best)
          return {};
        double step = .25 * s.first.distance(s.second);
        for (unsigned int iteration = 0;
             iteration < parameters.optimization_steps;
             ++iteration)
          {
            bool improved = false;
            for (unsigned int d = 0; d < 3; ++d)
              for (double sign : {-1., 1.})
                {
                  auto trial = best->junction;
                  trial[d] += sign * step;
                  if (auto candidate =
                        propose(id, trial, terminal, best->cost - 1.e-14))
                    {
                      best     = std::move(candidate);
                      improved = true;
                    }
                }
            if (!improved)
              step *= .5;
          }
        return best;
      }

      void
      accept(const Proposal &p)
      {
        const auto old                        = network.vessels.at(p.vessel);
        network.nodes[next_node]              = {p.junction,
                                                 0,
                                                 Port::none,
                                                 Provenance::generated};
        network.nodes[next_node + 1]          = {p.terminal,
                                                 0,
                                                 Port::none,
                                                 Provenance::generated};
        network.vessels.at(p.vessel).nodes[1] = next_node;
        auto continuation                     = old;
        continuation.nodes                    = {{next_node, old.nodes[1]}};
        network.vessels[next_vessel]          = continuation;
        auto terminal                         = old;
        terminal.nodes                        = {{next_node, next_node + 1}};
        network.vessels[next_vessel + 1]      = terminal;
        const Id tid                          = take_identifiers(next_terminal);
        const Id territory     = take_identifiers(next_territory);
        network.terminals[tid] = {
          next_node + 1, territory, System::arterial, 0, 1.};
        result.territories.push_back({territory, p.terminal, 0, 1.});
        for (auto &[child, parent] : parents)
          if (parent == p.vessel)
            parent = next_vessel;
        const Id root = tree_of.at(p.vessel);
        auto    &tree = trees.at(root);
        forest_volume -= tree_volume(tree, tree.volume_sum, counts.at(root));
        forest_dissipation -=
          tree_dissipation(tree, tree.dissipation_sum, counts.at(root));
        tree_of[next_vessel] = tree_of[next_vessel + 1] = root;
        counts[next_vessel]                             = counts.at(p.vessel);
        counts[next_vessel + 1]                         = 1;
        parents[next_vessel] = parents[next_vessel + 1] = p.vessel;
        ++counts.at(p.vessel);
        for (Id ancestor : p.ancestors)
          ++counts.at(ancestor);
        tree.volume_sum      = p.volume_sum;
        tree.dissipation_sum = p.dissipation_sum;
        forest_volume += tree_volume(tree, tree.volume_sum, counts.at(root));
        forest_dissipation +=
          tree_dissipation(tree, tree.dissipation_sum, counts.at(root));
        take_identifiers(next_node, 2);
        take_identifiers(next_vessel, 2);
        ++result.accepted_insertions;
        rebuild_index();
      }

      void
      run()
      {
        double capacity = 0.;
        for (const auto &[id, tree] : trees)
          capacity +=
            parameters.minimum_radius == 0. ?
              terminal_count :
              std::min<double>(
                terminal_count,
                std::floor(
                  std::pow(tree.radius / parameters.minimum_radius,
                           parameters.arterial_murray_exponent) *
                  (1. + 32. * std::numeric_limits<double>::epsilon())));
        if (capacity < terminal_count)
          {
            result.validation.issues.push_back(
              {"growth",
               invalid_id,
               "Requested uniform-demand terminal count is incompatible with the minimum radius at the fixed attachment radii."});
            return;
          }
        while (result.territories.size() < terminal_count)
          {
            AssertThrow(next_node < invalid_id - 1 &&
                          next_vessel < invalid_id - 1,
                        ExcMessage("Vascular identifier space exhausted."));
            bool inserted = false;
            for (unsigned int attempt = 0; attempt < parameters.sample_budget;
                 ++attempt)
              {
                ++result.sampled_points;
                Point<3> target;
                for (unsigned int d = 0; d < 3; ++d)
                  target[d] = lower[d] + uniform() * (upper[d] - lower[d]);
                const auto cell = geometry.containing(target);
                if (!cell)
                  continue;
                const double w = weight.value(target);
                AssertThrow(
                  std::isfinite(w) && w >= 0. &&
                    w <= parameters.terminal_weight_bound,
                  ExcMessage(
                    "Spatial terminal weight violates its declared nonnegative bound."));
                const double h = (*cell)->diameter();
                const double scale =
                  parameters.density_beta >= 0. ? minimum_h : maximum_h;
                if (uniform() >= w / parameters.terminal_weight_bound *
                                   std::pow(scale / h, parameters.density_beta))
                  continue;
                std::vector<Leaf> candidates;
                candidate_vessels.query(
                  bgi::nearest(target, parameters.candidate_count),
                  std::back_inserter(candidates));
                std::sort(candidates.begin(),
                          candidates.end(),
                          [&](const auto &a, const auto &b) {
                            const double da = point_distance_square(target,
                                                                    a.first),
                                         db = point_distance_square(target,
                                                                    b.first);
                            return da < db || (da == db && a.second < b.second);
                          });
                std::optional<Proposal> best;
                for (const auto &[s, id] : candidates)
                  if (auto p = optimize(id, target);
                      p && (!best || p->cost < best->cost))
                    best = std::move(p);
                if (best)
                  {
                    accept(*best);
                    inserted = true;
                    break;
                  }
              }
            if (!inserted)
              {
                result.validation.issues.push_back(
                  {"growth",
                   invalid_id,
                   "No admissible insertion was found within the finite sample and optimization budgets."});
                break;
              }
          }
        for (const auto &[id, root] : tree_of)
          network.vessels.at(id).radii = {{radius(id), radius(id)}};
      }

      GenerationResult   &result;
      NetworkData        &network;
      const Parameters   &parameters;
      const unsigned int  terminal_count;
      BackgroundGeometry &geometry;
      std::mt19937_64     random;
      FunctionParser<3>   weight;
      Point<3>            lower, upper;
      double              minimum_h = std::numeric_limits<double>::infinity(),
             maximum_h              = 0.;
      const double exponent, dissipation_exponent;
      double forest_volume = 0., forest_dissipation = 0., maximum_radius = 0.;
      Id next_node = 0, next_vessel = 0, next_terminal = 0, next_territory = 0;
      std::map<Id, Tree>         trees;
      std::map<Id, Id>           tree_of;
      std::map<Id, unsigned int> counts;
      std::map<Id, Id>           parents;
      RTree<Leaf>                vessels, candidate_vessels;
    };

  } // namespace

  static GenerationResult
  generate_count(const dealii::Triangulation<3> &background,
                 const dealii::Mapping<3>       &mapping,
                 const NetworkData              &initial,
                 const Parameters               &parameters,
                 MPI_Comm                        communicator,
                 unsigned int                    terminal_count)
  {
    using namespace dealii;
    AssertThrow(Utilities::MPI::n_mpi_processes(communicator) == 1,
                ExcMessage(
                  "The current CCO execution path requires one rank."));
    const bool creating_seed = initial.nodes.empty() &&
                               initial.vessels.empty() &&
                               initial.terminals.empty();
    parameters.validate(creating_seed);
    GenerationResult result;
    result.network    = initial;
    result.validation = validate_data(initial, 1);
    if (!result.validation.valid())
      return result;
    BackgroundGeometry geometry(background, mapping);
    const auto         fail = [&](Id id, const std::string &message) {
      result.validation.issues.push_back({"attachment", id, message});
    };
    std::map<Id, std::vector<Id>> incoming, outgoing;
    for (const auto &[id, e] : initial.vessels)
      {
        outgoing[e.nodes[0]].push_back(id);
        incoming[e.nodes[1]].push_back(id);
      }
    std::set<Id> recorded_nodes;
    for (const auto &[id, t] : initial.terminals)
      {
        if (!recorded_nodes.insert(t.node).second ||
            incoming[t.node].size() != 1 || !outgoing[t.node].empty())
          fail(
            id,
            "A tissue terminal must uniquely identify an outward leaf of the input network.");
        else if (t.system != initial.vessels.at(incoming[t.node][0]).system)
          fail(id, "Terminal system disagrees with its incoming vessel.");
      }
    if (!result.validation.valid())
      return result;
    if (creating_seed && !parameters.selected_terminals.empty())
      {
        fail(invalid_id,
             "Cannot select input terminal ids from an empty network.");
        return result;
      }
    Id next_node      = next_identifier(initial.nodes);
    Id next_vessel    = next_identifier(initial.vessels);
    Id next_terminal  = next_identifier(initial.terminals);
    Id next_territory = 0;
    for (const auto &[id, t] : initial.terminals)
      if (t.territory != invalid_id)
        {
          AssertThrow(t.territory < invalid_id - 1,
                      ExcMessage(
                        "Vascular territory identifier space exhausted."));
          next_territory = std::max(next_territory, t.territory + 1);
        }
    if (!creating_seed)
      for (const auto &[node, edges] : incoming)
        if (edges.size() == 1 && outgoing[node].empty() &&
            initial.nodes.at(node).port == Port::none &&
            !recorded_nodes.count(node))
          {
            const Id tid                  = take_identifiers(next_terminal);
            result.network.terminals[tid] = {
              node,
              invalid_id,
              initial.vessels.at(edges[0]).system,
              initial.nodes.at(node).owner,
              1.};
          }
    std::vector<Id> selected = parameters.selected_terminals;
    if (selected.empty() && !creating_seed)
      for (const auto &[id, t] : result.network.terminals)
        if (t.system == System::arterial &&
            geometry.containing(result.network.nodes.at(t.node).position))
          selected.push_back(id);
    if (!parameters.selected_terminals.empty())
      {
        std::set<Id> seen;
        for (Id id : selected)
          {
            const auto t = initial.terminals.find(id);
            if (!seen.insert(id).second)
              fail(id, "An input terminal may be selected only once.");
            else if (t == initial.terminals.end())
              fail(
                id,
                "Selected terminal id is absent from the input terminal registry.");
            else if (t->second.system != System::arterial)
              fail(
                id,
                "Arterial generation requires an arterial attachment terminal.");
            else if (!geometry.containing(
                       initial.nodes.at(t->second.node).position))
              fail(
                id,
                "An explicitly selected exterior terminal cannot be extended by interior vessels.");
          }
      }
    if (!creating_seed && selected.empty())
      fail(invalid_id, "No available interior arterial terminal was found.");
    if (terminal_count < (creating_seed ? 1 : selected.size()))
      fail(
        invalid_id,
        "Terminal count is smaller than the number of selected attachments.");
    if (!result.validation.valid())
      return result;
    std::vector<CCOGrowth::Leaf> obstacles;
    double maximum_radius = creating_seed ? parameters.seed_radius : 0.;
    for (const auto &[id, e] : initial.vessels)
      {
        obstacles.emplace_back(
          Segment<3>(initial.nodes.at(e.nodes[0]).position,
                     initial.nodes.at(e.nodes[1]).position),
          id);
        maximum_radius = std::max({maximum_radius, e.radii[0], e.radii[1]});
      }
    auto            index = pack_rtree(obstacles);
    std::vector<Id> roots;
    const auto      append_root = [&](Id            start,
                                 const Vessel &properties,
                                 Id            input_terminal) {
      const Point<3> position = creating_seed ?
                                       parameters.seed_position :
                                       initial.nodes.at(start).position;
      const auto     cell     = geometry.containing(position);
      if (!cell)
        {
          fail(input_terminal, "Seed or attachment is outside the background.");
          return;
        }
      const double radius = properties.radii[1];
      if (radius < parameters.minimum_radius)
        {
          fail(
            input_terminal,
            "Fixed attachment radius is below the minimum admissible new radius.");
          return;
        }
      std::vector<Point<3>> directions;
      if (parameters.seed_direction.norm() > 0.)
        directions.push_back(parameters.seed_direction);
      else
        {
          if (!creating_seed)
            directions.emplace_back(
              position - initial.nodes.at(properties.nodes[0]).position);
          directions.emplace_back((*cell)->center() - position);
          for (const auto v : (*cell)->vertex_indices())
            directions.emplace_back(
              .5 * ((*cell)->vertex(v) + (*cell)->center()) - position);
          for (unsigned int d = 0; d < 3; ++d)
            for (double sign : {-1., 1.})
              {
                Point<3> direction;
                direction[d] = sign;
                directions.push_back(direction);
              }
        }
      AssertThrow(next_node < invalid_id - (creating_seed ? 1 : 0) &&
                    next_vessel < invalid_id,
                  ExcMessage("Vascular identifier space exhausted."));
      const Id end_node = next_node + (creating_seed ? 1 : 0);
      for (const auto &direction : directions)
        {
          const double norm = direction.norm();
          if (!(norm > 0.) || !std::isfinite(norm))
            continue;
          const Point<3> end =
            position + parameters.seed_length / norm * direction;
          if (position.distance(end) <= parameters.minimum_length)
            continue;
          if (!geometry.contains({position, end},
                                 radius + parameters.clearance,
                                 parameters.geometry_depth,
                                 parameters.geometry_work))
            continue;
          const CCOGrowth::Edge edge{next_vessel,
                                          {{start, end_node}},
                                          {{position, end}},
                                     radius};
          Point<3>              lo, hi;
          const double pad = radius + maximum_radius + parameters.clearance +
                             64. * std::numeric_limits<double>::epsilon() *
                               std::max({1., position.norm(), end.norm()});
          for (unsigned int d = 0; d < 3; ++d)
            {
              lo[d] = std::min(position[d], end[d]) - pad;
              hi[d] = std::max(position[d], end[d]) + pad;
            }
          bool compatible = true;
          for (const auto &[s, id] :
               index | bgi::adaptors::queried(bgi::intersects(
                         BoundingBox<3>(std::make_pair(lo, hi)))))
            {
              const auto           &e = result.network.vessels.at(id);
              const CCOGrowth::Edge other{id,
                                          e.nodes,
                                               {{s.first, s.second}},
                                          std::max(e.radii[0], e.radii[1])};
              if (!CCOGrowth::compatible(edge, other, parameters))
                {
                  compatible = false;
                  break;
                }
            }
          if (!compatible)
            continue;
          if (creating_seed)
            result.network.nodes[start] = {position,
                                           0,
                                           Port::arterial_inlet,
                                           Provenance::generated};
          result.network.nodes[end_node] = {end,
                                            0,
                                            Port::none,
                                            Provenance::generated};
          auto vessel                    = properties;
          vessel.nodes                   = {{start, end_node}};
          vessel.radii                   = {{radius, radius}};
          vessel.provenance              = Provenance::generated;
          vessel.owner                   = 0;
          const Id vid                   = take_identifiers(next_vessel);
          result.network.vessels[vid]    = vessel;
          const Id tid                   = take_identifiers(next_terminal),
                   territory            = take_identifiers(next_territory);
          result.network.terminals[tid] = {
            end_node, territory, System::arterial, 0, 1.};
          result.territories.push_back({territory, end, 0, 1.});
          take_identifiers(next_node, creating_seed ? 2 : 1);
          if (input_terminal != invalid_id)
            {
              result.attachment_roots[input_terminal] = vid;
              result.network.terminals.erase(input_terminal);
            }
          roots.push_back(vid);
          index.insert({{position, end}, vid});
          return;
        }
      fail(
        input_terminal,
        "No admissible initial subtree direction was found within the deterministic geometry budget.");
    };
    if (creating_seed)
      {
        Vessel properties;
        properties.radii = {{parameters.seed_radius, parameters.seed_radius}};
        properties.viscosity = parameters.viscosity;
        append_root(0, properties, invalid_id);
      }
    else
      for (Id id : selected)
        {
          const auto &terminal = result.network.terminals.at(id);
          append_root(terminal.node,
                      initial.vessels.at(incoming[terminal.node][0]),
                      id);
          if (!result.validation.valid())
            break;
        }
    if (!result.validation.valid())
      return result;
    CCOGrowth growth(result, parameters, geometry, roots, terminal_count);
    growth.run();
    const auto report = validate_data(result.network, 1);
    result.validation.issues.insert(result.validation.issues.end(),
                                    report.issues.begin(),
                                    report.issues.end());
    if (result.validation.valid())
      {
        result.discretization = discretize(background,
                                           mapping,
                                           result.network,
                                           parameters.discretization,
                                           communicator);
        result.validation.issues.insert(
          result.validation.issues.end(),
          result.discretization.validation.issues.begin(),
          result.discretization.validation.issues.end());
        if (result.validation.valid())
          result.tissue = measure_tissue(background,
                                         mapping,
                                         result.network,
                                         result.discretization,
                                         parameters.coverage,
                                         communicator);
      }
    return result;
  }

  GenerationResult
  generate(const dealii::Triangulation<3> &background,
           const dealii::Mapping<3>       &mapping,
           const NetworkData              &initial,
           const Parameters               &parameters,
           MPI_Comm                        communicator)
  {
    using namespace dealii;
    const auto &p = parameters.calibration;
    if (p.volume_fraction == 0. && p.minimum_coverage == 0.)
      return generate_count(background,
                            mapping,
                            initial,
                            parameters,
                            communicator,
                            parameters.terminal_count);
    AssertThrow(Utilities::MPI::n_mpi_processes(communicator) == 1,
                ExcMessage(
                  "The current vascular calibration requires one rank."));
    parameters.validate(initial.nodes.empty() && initial.vessels.empty() &&
                        initial.terminals.empty());
    GenerationResult baseline;
    baseline.network    = initial;
    baseline.validation = validate_data(initial, 1);
    if (!baseline.validation.valid())
      return baseline;
    baseline.discretization = discretize(
      background, mapping, initial, parameters.discretization, communicator);
    baseline.validation = baseline.discretization.validation;
    if (!baseline.validation.valid())
      return baseline;
    baseline.tissue = measure_tissue(
      background, mapping, initial, baseline.discretization, {}, communicator);
    if (p.volume_fraction > 0. &&
        baseline.tissue.volume_fraction >
          p.volume_fraction * (1. + p.relative_tolerance))
      {
        baseline.calibration.status = CalibrationStatus::incompatible;
        baseline.validation.issues.push_back(
          {"calibration",
           invalid_id,
           "The fixed imported inside volume alone exceeds the target upper tolerance; adding vessels cannot meet the target."});
        return baseline;
      }
    CalibrationReport               report;
    std::set<unsigned int>          tried;
    std::optional<GenerationResult> best;
    std::pair<double, double>       best_score{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()};
    unsigned int best_count = 0;
    unsigned int count      = std::clamp(parameters.terminal_count,
                                    p.minimum_terminals,
                                    p.maximum_terminals);
    for (unsigned int attempt = 0; attempt < p.attempt_budget; ++attempt)
      {
        auto candidate = generate_count(
          background, mapping, initial, parameters, communicator, count);
        report.sampled_points += candidate.sampled_points;
        report.attempts.push_back({count,
                                   candidate.tissue,
                                   candidate.validation,
                                   candidate.sampled_points});
        tried.insert(count);
        if (candidate.validation.valid())
          {
            const double error =
              p.volume_fraction > 0. ?
                std::abs(candidate.tissue.volume_fraction / p.volume_fraction -
                         1.) :
                0.;
            const double deficit =
              std::max(0.,
                       p.minimum_coverage - candidate.tissue.arterial_coverage);
            if (error <= p.relative_tolerance && deficit == 0.)
              {
                report.status         = CalibrationStatus::converged;
                candidate.calibration = std::move(report);
                return candidate;
              }
            const std::pair<double, double> score{deficit, error};
            if (score < best_score)
              {
                best_score = score;
                best_count = count;
                best       = std::move(candidate);
              }
          }
        else if (!best)
          baseline = std::move(candidate);
        std::optional<unsigned int> next;
        const auto                  offer = [&](long double value) {
          if (!next && std::isfinite(value))
            {
              const auto n = static_cast<unsigned int>(
                std::clamp(std::round(value),
                           static_cast<long double>(p.minimum_terminals),
                           static_cast<long double>(p.maximum_terminals)));
              if (!tried.count(n))
                next = n;
            }
        };
        // Secant interpolation and count doubling/halving are proposals only.
        // Every third step explores an untried interval regardless of volume
        // signs; no observation removes a count from the admissible domain.
        if (best && (attempt + 1) % 3 != 0)
          {
            if (p.volume_fraction > 0.)
              {
                double      closest = std::numeric_limits<double>::infinity();
                long double interpolated = 0.;
                bool        bracket      = false;
                for (const auto &a : report.attempts)
                  for (const auto &b : report.attempts)
                    if (a.generation_validation.valid() &&
                        b.generation_validation.valid() &&
                        a.tissue.volume_fraction < p.volume_fraction &&
                        b.tissue.volume_fraction > p.volume_fraction)
                      {
                        const double span =
                          b.tissue.volume_fraction - a.tissue.volume_fraction;
                        if (span < closest)
                          {
                            closest = span;
                            interpolated =
                              a.terminal_count +
                              static_cast<long double>(
                                p.volume_fraction - a.tissue.volume_fraction) /
                                span *
                                (static_cast<long double>(b.terminal_count) -
                                 a.terminal_count);
                            bracket = true;
                          }
                      }
                if (bracket)
                  offer(interpolated);
                if (!next)
                  offer(best->tissue.volume_fraction < p.volume_fraction ?
                          best_count * 2.L :
                          std::floor(best_count * .5L));
              }
            else
              offer(best_count * 2.L);
          }
        if (!next && !tried.count(p.minimum_terminals))
          offer(p.minimum_terminals);
        if (!next)
          {
            std::uint64_t begin = p.minimum_terminals, width = 0;
            const auto    interval = [&](std::uint64_t end) {
              if (end - begin > width)
                {
                  width = end - begin;
                  next  = static_cast<unsigned int>(begin + (width - 1) / 2);
                }
            };
            for (const auto n : tried)
              {
                interval(n);
                begin = static_cast<std::uint64_t>(n) + 1;
              }
            interval(static_cast<std::uint64_t>(p.maximum_terminals) + 1);
          }
        if (!next)
          break;
        count = *next;
      }
    auto result        = best ? std::move(*best) : std::move(baseline);
    report.status      = CalibrationStatus::exhausted;
    result.calibration = std::move(report);
    result.validation.issues.push_back(
      {"calibration",
       invalid_id,
       "No terminal count met all requested volume and coverage targets within the finite calibration/search budget. This does not prove incompatibility."});
    return result;
  }
} // namespace ImmersX::Vascularization
