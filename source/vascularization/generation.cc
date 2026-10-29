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
    enter_subsection("Sampling");
    add_parameter("Terminal count",
                  terminal_count,
                  "Number of tissue terminals to grow.");
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
    AssertThrow(
      terminal_count > 0 && sample_budget > 0 && candidate_count > 0 &&
        optimization_steps <= 100,
      dealii::ExcMessage(
        "Terminal count, sample budget and candidate count must be positive; optimization steps must not exceed 100."));
    AssertThrow(
      positive(arterial_murray_exponent) && std::isfinite(minimum_radius) &&
        minimum_radius >= 0. && minimum_radius <= seed_radius &&
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
    // a 2D intersection check. Skew 3D segments also need an interior minimum.
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
      struct Proposal
      {
        Id              vessel = invalid_id;
        Point<3>        junction, terminal;
        std::vector<Id> ancestors;
        double          volume_sum = 0., dissipation_sum = 0.;
        double          cost = std::numeric_limits<double>::infinity();
      };

      CCOGrowth(GenerationResult   &result,
                const Parameters   &parameters,
                BackgroundGeometry &geometry)
        : result(result)
        , network(result.network)
        , parameters(parameters)
        , geometry(geometry)
        , random(parameters.random_seed)
        , exponent(2. / parameters.arterial_murray_exponent)
        , dissipation_exponent(2. - 4. / parameters.arterial_murray_exponent)
      {
        counts[0]  = 1;
        parents[0] = invalid_id;
        volume_sum = dissipation_sum = length(network, 0);
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
      radius(Id id) const
      {
        return parameters.seed_radius *
               std::pow(static_cast<double>(counts.at(id)) / counts.at(0),
                        1. / parameters.arterial_murray_exponent);
      }

      void
      rebuild_index()
      {
        std::vector<Leaf> leaves;
        leaves.reserve(network.vessels.size());
        for (const auto &[id, e] : network.vessels)
          leaves.emplace_back(segment(id), id);
        vessels = pack_rtree(leaves);
      }

      Edge
      edge(Id id, double r) const
      {
        const auto &e = network.vessels.at(id);
        const auto  s = segment(id);
        return {id, e.nodes, {{s.first, s.second}}, r};
      }

      bool
      compatible(const Edge &a, const Edge &b) const
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
        const auto   old = segment(id);
        const double n = counts.at(id), total = counts.at(0);
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
        p.volume_sum      = sum(exponent, volume_sum);
        p.dissipation_sum = sum(dissipation_exponent, dissipation_sum);
        const double vr =
          p.volume_sum / volume_sum * std::pow(total / (total + 1.), exponent);
        const double dr = p.dissipation_sum / dissipation_sum *
                          std::pow((total + 1.) / total, -dissipation_exponent);
        p.cost = (1. - parameters.dissipation_weight) * vr +
                 parameters.dissipation_weight * dr;
        if (!std::isfinite(p.cost) || p.cost >= ceiling)
          return {};
        std::set<Id> increased(p.ancestors.begin(), p.ancestors.end());
        increased.insert(id);
        const auto proposed_radius = [&](Id e) {
          const double count = counts.at(e) + (increased.count(e) ? 1. : 0.);
          return parameters.seed_radius *
                 std::pow(count / (total + 1.),
                          1. / parameters.arterial_murray_exponent);
        };
        const double leaf_radius =
          parameters.seed_radius *
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
           parameters.seed_radius *
             std::pow(n / (total + 1.),
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
              if (!compatible(e, changed[j]))
                return {};
            Point<3>     lo, hi;
            const double pad =
              e.radius + parameters.seed_radius + parameters.clearance + guard;
            for (unsigned int d = 0; d < 3; ++d)
              {
                lo[d] = std::min(e.points[0][d], e.points[1][d]) - pad;
                hi[d] = std::max(e.points[0][d], e.points[1][d]) + pad;
              }
            const BoundingBox<3> box(std::make_pair(lo, hi));
            for (const auto &[s, other] :
                 vessels | bgi::adaptors::queried(bgi::intersects(box)))
              if (!replaced.count(other) &&
                  !compatible(e, edge(other, proposed_radius(other))))
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
        const Id tid                          = network.terminals.size();
        network.terminals[tid] = {next_node + 1, tid, System::arterial, 0, 1.};
        result.territories.push_back({tid, p.terminal, 0, 1.});
        for (auto &[child, parent] : parents)
          if (parent == p.vessel)
            parent = next_vessel;
        counts[next_vessel]     = counts.at(p.vessel);
        counts[next_vessel + 1] = 1;
        parents[next_vessel] = parents[next_vessel + 1] = p.vessel;
        ++counts.at(p.vessel);
        for (Id ancestor : p.ancestors)
          ++counts.at(ancestor);
        volume_sum      = p.volume_sum;
        dissipation_sum = p.dissipation_sum;
        next_node += 2;
        next_vessel += 2;
        ++result.accepted_insertions;
        rebuild_index();
      }

      void
      run()
      {
        if (parameters.seed_radius *
              std::pow(1. / parameters.terminal_count,
                       1. / parameters.arterial_murray_exponent) <
            parameters.minimum_radius)
          {
            result.validation.issues.push_back(
              {"growth",
               invalid_id,
               "Requested uniform-demand terminal count is incompatible with the minimum radius at the fixed seed radius."});
            return;
          }
        while (network.terminals.size() < parameters.terminal_count)
          {
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
                vessels.query(bgi::nearest(target, parameters.candidate_count),
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
        for (auto &[id, e] : network.vessels)
          e.radii = {{radius(id), radius(id)}};
      }

      GenerationResult   &result;
      NetworkData        &network;
      const Parameters   &parameters;
      BackgroundGeometry &geometry;
      std::mt19937_64     random;
      FunctionParser<3>   weight;
      Point<3>            lower, upper;
      double              minimum_h = std::numeric_limits<double>::infinity(),
             maximum_h              = 0.;
      const double               exponent, dissipation_exponent;
      double                     volume_sum = 0., dissipation_sum = 0.;
      Id                         next_node = 2, next_vessel = 1;
      std::map<Id, unsigned int> counts;
      std::map<Id, Id>           parents;
      RTree<Leaf>                vessels;
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
                  "The current CCO execution path requires one rank."));
    parameters.validate();
    AssertThrow(initial.nodes.empty() && initial.vessels.empty() &&
                  initial.terminals.empty(),
                ExcMessage(
                  "The current CCO execution path requires an empty network."));
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
    if (result.validation.valid() && parameters.terminal_count > 1)
      {
        CCOGrowth growth(result, parameters, geometry);
        growth.run();
        const auto report = validate_data(result.network, 1);
        result.validation.issues.insert(result.validation.issues.end(),
                                        report.issues.begin(),
                                        report.issues.end());
      }
    return result;
  }
} // namespace ImmersX::Vascularization
