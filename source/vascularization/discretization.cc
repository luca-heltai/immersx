// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/boost_adaptors/segment.h>

#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/numerics/rtree.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/discretization.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <set>

namespace ImmersX::Vascularization
{
  namespace
  {
    using namespace dealii;
    namespace bgi                        = boost::geometry::index;
    constexpr double reference_tolerance = 1.e-12;

    struct Interval
    {
      double              begin, end, h;
      types::subdomain_id owner;

      template <class Archive>
      void
      serialize(Archive &ar, const unsigned int)
      {
        ar &begin &end &h &owner;
      }
    };

    Point<3>
    point(const Segment<3> &s, double t)
    {
      return s.first + t * (s.second - s.first);
    }

    void
    append_intersection(const Segment<3>    &s,
                        const Point<3>      &p,
                        std::vector<double> &cuts)
    {
      const auto   d = s.second - s.first;
      const double t = (p - s.first) * d / d.norm_square();
      const double tolerance =
        128. * std::numeric_limits<double>::epsilon() *
        std::max({1., p.norm(), s.first.norm(), s.second.norm()});
      if (t >= -reference_tolerance && t <= 1. + reference_tolerance &&
          point(s, t).distance(p) <= tolerance)
        cuts.push_back(std::clamp(t, 0., 1.));
    }

    // Eliminate the line parameter by projecting along its dominant axis.
    // The two remaining face equations are bilinear in u,v; eliminating v
    // gives a quadratic in u. This handles warped Q1 faces without planar
    // triangulation of their interiors.
    void
    face_intersections(const Segment<3>            &s,
                       const std::vector<Point<3>> &p,
                       std::vector<double>         &cuts)
    {
      const auto   direction = s.second - s.first;
      unsigned int axis      = 0;
      for (unsigned int d = 1; d < 3; ++d)
        if (std::abs(direction[d]) > std::abs(direction[axis]))
          axis = d;
      const unsigned int i = (axis + 1) % 3, j = (axis + 2) % 3;
      const auto         project = [&](const Tensor<1, 3> &x) {
        return std::array<long double, 2>{
          {static_cast<long double>(x[i]) -
             static_cast<long double>(direction[i]) / direction[axis] * x[axis],
           static_cast<long double>(x[j]) -
             static_cast<long double>(direction[j]) / direction[axis] *
               x[axis]}};
      };
      const auto e = project(p[0] - s.first), u = project(p[1] - p[0]),
                 v = project(p[2] - p[0]);
      const Point<3> cross =
        p.size() == 4 ? Point<3>(p[3] - p[1] - p[2] + p[0]) : Point<3>();
      const auto w = project(cross);
      // A line lying in a face/ruling has a continuum of face intersections.
      // Its entry/exit parameters are supplied by the face's boundary edges.
      const std::vector<std::array<unsigned int, 2>> edges =
        p.size() == 3 ? std::vector<std::array<unsigned int, 2>>{{{0, 1}},
                                                                 {{1, 2}},
                                                                 {{2, 0}}} :
                        std::vector<std::array<unsigned int, 2>>{{{0, 1}},
                                                                 {{1, 3}},
                                                                 {{3, 2}},
                                                                 {{2, 0}}};
      for (const auto &edge : edges)
        {
          const auto a = project(p[edge[0]] - s.first),
                     d = project(p[edge[1]] - p[edge[0]]);
          const unsigned int component =
            std::abs(d[0]) >= std::abs(d[1]) ? 0 : 1;
          if (d[component] != 0.)
            {
              const long double t = -a[component] / d[component];
              if (t >= -reference_tolerance && t <= 1. + reference_tolerance)
                append_intersection(s,
                                    p[edge[0]] + static_cast<double>(
                                                   std::clamp(t, 0.L, 1.L)) *
                                                   (p[edge[1]] - p[edge[0]]),
                                    cuts);
            }
          else
            {
              append_intersection(s, p[edge[0]], cuts);
              append_intersection(s, p[edge[1]], cuts);
            }
        }
      const long double a = u[0] * w[1] - u[1] * w[0];
      const long double b =
        e[0] * w[1] + u[0] * v[1] - e[1] * w[0] - u[1] * v[0];
      const long double        c = e[0] * v[1] - e[1] * v[0];
      std::vector<long double> roots;
      if (a == 0.)
        {
          if (b != 0.)
            roots.push_back(-c / b);
        }
      else
        {
          long double       discriminant = b * b - 4. * a * c;
          const long double guard =
            64. * std::numeric_limits<long double>::epsilon() *
            (b * b + std::abs(4. * a * c));
          if (discriminant >= -guard)
            {
              discriminant = std::max(0.L, discriminant);
              const long double q =
                -.5L * (b + std::copysign(std::sqrt(discriminant), b));
              if (q == 0.)
                roots.push_back(-b / (2. * a));
              else
                {
                  roots.push_back(q / a);
                  roots.push_back(c / q);
                }
            }
        }
      for (long double x : roots)
        if (x >= -reference_tolerance && x <= 1. + reference_tolerance)
          {
            const std::array<long double, 2> den{
              {v[0] + w[0] * x, v[1] + w[1] * x}};
            const unsigned int component =
              std::abs(den[0]) >= std::abs(den[1]) ? 0 : 1;
            if (den[component] == 0.)
              continue;
            const long double y =
              -(e[component] + u[component] * x) / den[component];
            if (y >= -reference_tolerance && y <= 1. + reference_tolerance &&
                (p.size() == 4 || x + y <= 1. + reference_tolerance))
              {
                const double xx = std::clamp(static_cast<double>(x), 0., 1.),
                             yy = std::clamp(static_cast<double>(y), 0., 1.);
                append_intersection(s,
                                    p[0] + xx * (p[1] - p[0]) +
                                      yy * (p[2] - p[0]) + xx * yy * cross,
                                    cuts);
              }
          }
    }

    void
    sort_cuts(std::vector<double> &cuts)
    {
      std::sort(cuts.begin(), cuts.end());
      cuts.erase(std::unique(cuts.begin(),
                             cuts.end(),
                             [](double a, double b) {
                               return std::abs(a - b) < reference_tolerance;
                             }),
                 cuts.end());
      cuts.front() = 0.;
      cuts.back()  = 1.;
    }

    std::vector<Interval>
    traversed(const GridTools::Cache<3> &cache,
              const Mapping<3>          &mapping,
              const Segment<3>          &s,
              types::subdomain_id        exterior_owner)
    {
      std::vector<Triangulation<3>::active_cell_iterator> candidates;
      std::vector<double>                                 cuts{0., 1.};
      for (const auto &[box, cell] :
           cache.get_locally_owned_cell_bounding_boxes_rtree() |
             bgi::adaptors::queried(bgi::intersects(s)))
        {
          candidates.push_back(cell);
          for (const auto f : cell->face_indices())
            {
              std::vector<Point<3>> vertices;
              for (const auto v : cell->face(f)->vertex_indices())
                vertices.push_back(cell->face(f)->vertex(v));
              face_intersections(s, vertices, cuts);
            }
        }
      sort_cuts(cuts);
      std::vector<Interval> result;
      for (std::size_t i = 1; i < cuts.size(); ++i)
        {
          const Point<3> p = point(s, .5 * (cuts[i - 1] + cuts[i]));
          std::optional<Triangulation<3>::active_cell_iterator> found;
          for (const auto &cell : candidates)
            {
              try
                {
                  const auto unit =
                    mapping.transform_real_to_unit_cell(cell, p);
                  if (cell->reference_cell().contains_point(
                        unit, reference_tolerance) &&
                      (!found || cell->diameter() < (*found)->diameter() ||
                       (cell->diameter() == (*found)->diameter() &&
                        cell->id() < (*found)->id())))
                    found = cell;
                }
              catch (const Mapping<3>::ExcTransformationFailed &)
                {}
            }
          const double h   = found ? (*found)->diameter() : 0.;
          const auto owner = found ? (*found)->subdomain_id() : exterior_owner;
          if (!result.empty() && result.back().h == h &&
              result.back().owner == owner)
            result.back().end = cuts[i];
          else
            result.push_back({cuts[i - 1], cuts[i], h, owner});
        }
      return result;
    }

    // Only coarse background boxes are replicated. Physical owners obtain
    // exact positive-arclength intersections from the relevant cell owners.
    std::map<Id, std::vector<Interval>>
    distributed_intervals(const GridTools::Cache<3> &cache,
                          const Mapping<3>          &mapping,
                          const NetworkData         &network,
                          MPI_Comm                   comm)
    {
      const auto  rank = Utilities::MPI::this_mpi_process(comm);
      const auto  size = Utilities::MPI::n_mpi_processes(comm);
      const auto &tree = cache.get_locally_owned_cell_bounding_boxes_rtree();
      std::vector<BoundingBox<3>> local_boxes;
      if (!tree.empty())
        local_boxes = extract_rtree_level(tree, 1);
      const auto boxes = Utilities::MPI::all_gather(comm, local_boxes);
      std::vector<std::pair<BoundingBox<3>, unsigned int>> covering;
      for (unsigned int r = 0; r < size; ++r)
        for (const auto &box : boxes[r])
          covering.emplace_back(box, r);
      const auto covering_tree = pack_rtree(covering);
      using Query              = std::pair<Id, std::array<Point<3>, 2>>;
      std::map<unsigned int, std::vector<Query>> queries;
      std::map<Id, std::vector<Interval>>        result;
      for (const auto &[id, vessel] : network.vessels)
        if (vessel.owner == rank)
          {
            const std::array<Point<3>, 2> endpoints{
              {network.nodes.at(vessel.nodes[0]).position,
               network.nodes.at(vessel.nodes[1]).position}};
            const Segment<3> segment(endpoints[0], endpoints[1]);
            result[id];
            std::set<unsigned int> recipients;
            for (const auto &[box, source] :
                 covering_tree |
                   bgi::adaptors::queried(bgi::intersects(segment)))
              recipients.insert(source);
            for (const auto source : recipients)
              queries[source].push_back({id, endpoints});
          }
      const auto incoming = Utilities::MPI::some_to_some(comm, queries);
      std::map<unsigned int, std::map<Id, std::vector<Interval>>> replies;
      for (const auto &[source, requests] : incoming)
        for (const auto &[id, endpoints] : requests)
          for (const auto &interval : traversed(
                 cache, mapping, Segment<3>(endpoints[0], endpoints[1]), rank))
            if (interval.h > 0.)
              replies[source][id].push_back(interval);
      const auto received = Utilities::MPI::some_to_some(comm, replies);
      for (const auto &[source, vessels] : received)
        for (const auto &[id, intervals] : vessels)
          result.at(id).insert(result.at(id).end(),
                               intervals.begin(),
                               intervals.end());
      for (auto &[id, intervals] : result)
        {
          std::vector<double> cuts{0., 1.};
          for (const auto &interval : intervals)
            {
              cuts.push_back(interval.begin);
              cuts.push_back(interval.end);
            }
          sort_cuts(cuts);
          std::vector<Interval> combined;
          for (std::size_t i = 1; i < cuts.size(); ++i)
            {
              const double middle = .5 * (cuts[i - 1] + cuts[i]);
              Interval     selected{cuts[i - 1],
                                cuts[i],
                                0.,
                                network.vessels.at(id).owner};
              for (const auto &interval : intervals)
                if (interval.begin < middle && middle < interval.end)
                  {
                    // On a background interface use the finest crossed cell
                    // and the lowest containing rank for execution ownership.
                    if (selected.h == 0.)
                      {
                        selected.h     = interval.h;
                        selected.owner = interval.owner;
                      }
                    else
                      {
                        selected.h = std::min(selected.h, interval.h);
                        selected.owner =
                          std::min(selected.owner, interval.owner);
                      }
                  }
              if (!combined.empty() && combined.back().h == selected.h &&
                  combined.back().owner == selected.owner)
                combined.back().end = selected.end;
              else
                combined.push_back(selected);
            }
          intervals = std::move(combined);
        }
      return result;
    }

    DiscretizationCell
    cell_resolution(Id                           vessel,
                    double                       begin,
                    double                       end,
                    const std::vector<Interval> &intervals)
    {
      DiscretizationCell result;
      result.reference            = {vessel, begin, end};
      result.h_min                = std::numeric_limits<double>::infinity();
      const double middle         = .5 * (begin + end);
      bool         midpoint_found = false;
      for (const auto &interval : intervals)
        if (interval.begin < end && interval.end > begin)
          {
            if (interval.h > 0.)
              {
                result.inside = true;
                result.h_min  = std::min(result.h_min, interval.h);
                result.h_max  = std::max(result.h_max, interval.h);
              }
            if (interval.begin <= middle && middle <= interval.end)
              {
                result.owner   = midpoint_found ?
                                   std::min(result.owner, interval.owner) :
                                   interval.owner;
                midpoint_found = true;
              }
          }
      if (!result.inside)
        result.h_min = 0.;
      return result;
    }
  } // namespace

  void
  DiscretizationParameters::validate() const
  {
    using namespace dealii;
    AssertThrow(
      std::isfinite(lower_factor) && lower_factor > 0. &&
        std::isfinite(target_factor) && target_factor >= lower_factor &&
        std::isfinite(upper_factor) && upper_factor >= target_factor &&
        std::isfinite(exterior_length) && exterior_length >= 0. &&
        cell_budget > 0,
      ExcMessage("Invalid vascular discretization bounds or budget."));
  }

  DiscretizationResult
  discretize(const dealii::Triangulation<3> &background,
             const dealii::Mapping<3>       &mapping,
             const NetworkData              &network,
             const DiscretizationParameters &parameters,
             MPI_Comm                        communicator)
  {
    using namespace dealii;
    const auto rank        = Utilities::MPI::this_mpi_process(communicator);
    const auto size        = Utilities::MPI::n_mpi_processes(communicator);
    const auto require_all = [&](bool valid, const std::string &message) {
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid),
                                      communicator) == 1,
                  ExcMessage(message));
    };
    const std::array<double, 5> controls{
      {parameters.target_factor,
       parameters.lower_factor,
       parameters.upper_factor,
       parameters.exterior_length,
       static_cast<double>(parameters.cell_budget)}};
    const auto all_controls =
      Utilities::MPI::all_gather(communicator, controls);
    require_all(std::all_of(all_controls.begin(),
                            all_controls.end(),
                            [&](const auto &other) {
                              return other == controls;
                            }),
                "Vascular discretization parameters must agree across ranks.");
    parameters.validate();
    const auto *q          = dynamic_cast<const MappingQ<3> *>(&mapping);
    const auto *fe         = dynamic_cast<const MappingFE<3> *>(&mapping);
    int         comparison = MPI_UNEQUAL;
    MPI_Comm_compare(background.get_mpi_communicator(),
                     communicator,
                     &comparison);
    require_all(
      ((q && q->get_degree() == 1) || (fe && fe->get_degree() == 1)) &&
        (comparison == MPI_IDENT || comparison == MPI_CONGRUENT),
      "Vascular discretization requires a first-order mapping and congruent communicators.");
    DiscretizationResult result;
    result.validation = validate_data(network, size);
    if (Utilities::MPI::min(static_cast<unsigned int>(
                              result.validation.valid()),
                            communicator) == 0)
      {
        if (result.validation.valid())
          result.validation.issues.push_back(
            {"discretization",
             invalid_id,
             "Invalid vascular data on another rank."});
        return result;
      }
    std::map<unsigned int, std::vector<Id>> owned_ids;
    for (const auto &[id, vessel] : network.vessels)
      if (vessel.owner == rank)
        owned_ids[id % size].push_back(id);
    std::set<Id> seen;
    bool         unique = true;
    for (const auto &[source, ids] :
         Utilities::MPI::some_to_some(communicator, owned_ids))
      for (const auto id : ids)
        unique = seen.insert(id).second && unique;
    require_all(unique,
                "Physical vessels must have unique authoritative owners.");
    GridTools::Cache<3> cache(background, mapping);
    double              exterior_h      = parameters.exterior_length;
    bool                supported_cells = true;
    for (const auto &cell : background.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          supported_cells = (cell->reference_cell().is_simplex() ||
                             cell->reference_cell().is_hyper_cube()) &&
                            supported_cells;
          if (parameters.exterior_length == 0.)
            exterior_h = std::max(exterior_h, cell->diameter());
        }
    require_all(supported_cells,
                "Vascular discretization requires tetrahedra or hexahedra.");
    exterior_h = Utilities::MPI::max(exterior_h, communicator);
    AssertThrow(
      exterior_h > 0.,
      ExcMessage(
        "A nonempty background or an explicit exterior target length is required."));
    const auto crossings =
      distributed_intervals(cache, mapping, network, communicator);
    const auto subdivide_owned = [&]() {
      for (const auto &[id, e] : network.vessels)
        {
          if (e.owner != rank)
            continue;
          const double l         = length(network, id);
          const auto  &intervals = crossings.at(id);
          double       metric    = 0.;
          for (const auto &interval : intervals)
            {
              metric += (interval.end - interval.begin) * l /
                        (interval.h > 0. ? interval.h : exterior_h);
              if (interval.h > 0.)
                result.inside_volume +=
                  volume(network, id, interval.begin, interval.end);
            }
          const double wanted =
            std::max(1., std::round(metric / parameters.target_factor));
          if (wanted > parameters.cell_budget ||
              result.cells.size() + wanted > parameters.cell_budget)
            {
              result.validation.issues.push_back(
                {"discretization", id, "Numerical cell budget exhausted."});
              return;
            }
          const unsigned int  count = static_cast<unsigned int>(wanted);
          std::vector<double> cuts{0., 1.};
          for (unsigned int i = 1; i < count; ++i)
            {
              const double target      = metric * i / count;
              double       accumulated = 0.;
              for (const auto &interval : intervals)
                {
                  const double h = interval.h > 0. ? interval.h : exterior_h;
                  const double piece = (interval.end - interval.begin) * l / h;
                  if (target <= accumulated + piece)
                    {
                      cuts.push_back(interval.begin +
                                     (target - accumulated) * h / l);
                      break;
                    }
                  accumulated += piece;
                }
            }
          // An exterior part is preserved, but never merged across the physical
          // boundary into an interior numerical cell.
          for (std::size_t i = 1; i < intervals.size(); ++i)
            if ((intervals[i - 1].h > 0.) != (intervals[i].h > 0.))
              cuts.push_back(intervals[i].begin);
          sort_cuts(cuts);
          if (result.cells.size() + cuts.size() - 1 > parameters.cell_budget)
            {
              result.validation.issues.push_back(
                {"discretization",
                 id,
                 "Boundary-conforming numerical cell budget exhausted."});
              return;
            }
          for (std::size_t i = 1; i < cuts.size();)
            {
              const auto cell =
                cell_resolution(id, cuts[i - 1], cuts[i], intervals);
              const double h = cell.inside ? cell.h_min : exterior_h;
              if ((cuts[i] - cuts[i - 1]) * l >
                  parameters.upper_factor * h * (1. + 1.e-12))
                {
                  if (result.cells.size() + cuts.size() >
                        parameters.cell_budget ||
                      .5 * (cuts[i - 1] + cuts[i]) == cuts[i - 1])
                    {
                      result.validation.issues.push_back(
                        {"discretization",
                         id,
                         "Cannot meet the upper length bound within the numerical cell budget."});
                      return;
                    }
                  cuts.insert(cuts.begin() + i, .5 * (cuts[i - 1] + cuts[i]));
                }
              else
                ++i;
            }
          // Merge avoidable short pieces without violating any crossed cell's
          // upper bound or joining interior and exterior pieces.
          for (std::size_t i = 1; i < cuts.size();)
            {
              const auto cell =
                cell_resolution(id, cuts[i - 1], cuts[i], intervals);
              if (cell.inside && (cuts[i] - cuts[i - 1]) * l <
                                   parameters.lower_factor * cell.h_max)
                {
                  bool merged = false;
                  for (int offset : {-1, 1})
                    {
                      if ((offset == -1 && i == 1) ||
                          (offset == 1 && i + 1 == cuts.size()))
                        continue;
                      const auto neighbor =
                        cell_resolution(id,
                                        offset == -1 ? cuts[i - 2] : cuts[i],
                                        offset == -1 ? cuts[i - 1] :
                                                       cuts[i + 1],
                                        intervals);
                      const double begin =
                                     offset == -1 ? cuts[i - 2] : cuts[i - 1],
                                   end = offset == -1 ? cuts[i] : cuts[i + 1];
                      const auto combined =
                        cell_resolution(id, begin, end, intervals);
                      if (neighbor.inside == cell.inside &&
                          (end - begin) * l <=
                            parameters.upper_factor * combined.h_min &&
                          (end - begin) * l >=
                            parameters.lower_factor * combined.h_max)
                        {
                          cuts.erase(cuts.begin() + (offset == -1 ? i - 1 : i));
                          if (i > 1)
                            --i;
                          merged = true;
                          break;
                        }
                    }
                  if (merged)
                    continue;
                }
              ++i;
            }
          for (std::size_t i = 1; i < cuts.size(); ++i)
            {
              auto cell = cell_resolution(id, cuts[i - 1], cuts[i], intervals);
              cell.short_cell = cell.inside && (cuts[i] - cuts[i - 1]) * l <
                                                 parameters.lower_factor *
                                                   cell.h_max * (1. - 1.e-12);
              result.short_cells += cell.short_cell;
              result.exterior_cells += !cell.inside;
              result.cells.push_back(cell);
            }
        }
    };
    subdivide_owned();
    const auto count =
      Utilities::MPI::sum(static_cast<std::uint64_t>(result.cells.size()),
                          communicator);
    if (count > parameters.cell_budget)
      result.validation.issues.push_back(
        {"discretization",
         invalid_id,
         "Global numerical cell budget exhausted."});
    if (Utilities::MPI::min(static_cast<unsigned int>(
                              result.validation.valid()),
                            communicator) == 0 &&
        result.validation.valid())
      result.validation.issues.push_back(
        {"discretization",
         invalid_id,
         "Numerical subdivision failed on another rank."});
    result.inside_volume =
      Utilities::MPI::sum(result.inside_volume, communicator);
    result.short_cells = Utilities::MPI::sum(result.short_cells, communicator);
    result.exterior_cells =
      Utilities::MPI::sum(result.exterior_cells, communicator);
    return result;
  }
} // namespace ImmersX::Vascularization
