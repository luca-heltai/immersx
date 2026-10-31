// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_geometry_h
#define immersx_internal_growth_geometry_h
#include <deal.II/base/numbers.h>

#include <deal.II/boost_adaptors/segment.h>

#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/numerics/rtree.h>

#include <immersx/vascularization/growth_work.h>
#include <immersx/vascularization/network.h>

#include <algorithm>
#include <cmath>
#include <optional>
namespace ImmersX::Vascularization::internal
{
  using namespace dealii;
  namespace bgi = boost::geometry::index;
  using Patch   = std::vector<Point<3>>;

  // MappingQ's inverse iteration terminates at about 1e-11 in reference space.
  constexpr double inverse_mapping_tolerance = 1.e-11;

  template <typename ReferenceCell>
  auto
  boundary_reference_distance(const ReferenceCell &reference,
                              unsigned int         face,
                              const Point<3>      &unit,
                              int)
    -> decltype(reference.template vertex<3>(0), double())
  {
    return (unit - reference.template face_vertex_location<3>(face, 0)) *
           reference.template face_normal_vector<3>(face);
  }

  template <typename ReferenceCell>
  auto
  boundary_reference_distance(const ReferenceCell &reference,
                              unsigned int         face,
                              const Point<3>      &unit,
                              long) -> decltype(reference.vertex(0), double())
  {
    return (unit - reference.face_vertex_location(face, 0)) *
           reference.face_normal_vector(face);
  }

  inline bool
  reference_contains(const Triangulation<3>::active_cell_iterator &cell,
                     const Point<3>                               &unit,
                     double boundary_tolerance)
  {
    const auto reference = cell->reference_cell();
    if (!reference.contains_point(
          unit, std::max(boundary_tolerance, inverse_mapping_tolerance)))
      return false;
    // Allow inverse-map error across interior faces only. A physical boundary
    // keeps its strict half-space test, except at explicitly declared ports.
    for (const auto face : cell->face_indices())
      if (cell->face(face)->at_boundary() &&
          boundary_reference_distance(reference, face, unit, 0) >
            boundary_tolerance)
        return false;
    return true;
  }

  inline Point<3>
  normalized_port_normal(const Point<3> &value)
  {
    const double scale =
      std::max({std::abs(value[0]), std::abs(value[1]), std::abs(value[2])});
    if (scale == 0.)
      return {};
    Point<3> normal;
    // Component division avoids the overflowing reciprocal used by Tensor's
    // operator/=. Scaling also avoids overflow/underflow in its norm().
    for (unsigned int d = 0; d < 3; ++d)
      normal[d] = value[d] / scale;
    return Point<3>(normal / normal.norm());
  }

  inline double
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
  inline double
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
                const double offset = s.first[d] - (x < lo[d] ? lo[d] : hi[d]);
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

  inline std::array<Patch, 4>
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

  inline bool
  in_triangle(const Point<3> &p,
              const Point<3> &a,
              const Point<3> &b,
              const Point<3> &c,
              double          guard)
  {
    const auto   u = b - a, v = c - a, w = p - a;
    const double uu = u * u, uv = u * v, vv = v * v, wu = w * u, wv = w * v;
    const double determinant = uu * vv - uv * uv;
    if (!(determinant > 0.))
      return false;
    const double x         = (vv * wu - uv * wv) / determinant,
                 y         = (uu * wv - uv * wu) / determinant;
    const double tolerance = guard / std::sqrt(std::min(uu, vv));
    return x >= -tolerance && y >= -tolerance && x + y <= 1. + tolerance;
  }

  inline bool
  separated(const Patch                   &patch,
            const Segment<3>              &s,
            const double                   radius,
            const unsigned int             depth,
            unsigned int                  &work,
            const std::array<Point<3>, 2> &port_normals,
            GrowthWork                    *counts = nullptr)
  {
    if (work == 0)
      return false;
    count_work(counts, GrowthOperation::boundary_patch_tests);
    --work;
    const BoundingBox<3> box(patch);
    // Linear triangular and bilinear quadrilateral patches are contained in
    // the convex hull of their corners. A separated box proves separation.
    const double guard = 64. * std::numeric_limits<double>::epsilon() *
                         std::max({1., s.first.norm(), s.second.norm()});
    const std::array<Point<3>, 2> endpoints{{s.first, s.second}};
    // A cap is the intersection of the capsule with each declared inward
    // half-space. A triangle/bilinear patch stays in its corners' convex
    // hull, so an outward patch cannot meet the retained envelope interior.
    for (unsigned int i = 0; i < 2; ++i)
      if (port_normals[i].norm_square() > 0. &&
          std::all_of(patch.begin(), patch.end(), [&](const auto &p) {
            return (p - endpoints[i]) * port_normals[i] <= guard;
          }))
        return true;
    if (box_distance_square(box, s) > (radius + guard) * (radius + guard))
      return true;
    for (const auto &p : patch)
      if (point_distance_square(p, s) <= radius * radius)
        {
          bool retained = true;
          for (unsigned int i = 0; i < 2; ++i)
            if (port_normals[i].norm_square() > 0. &&
                (p - endpoints[i]) * port_normals[i] <= guard)
              retained = false;
          if (retained)
            return false;
        }
    if (depth == 0)
      return false;
    for (const auto &child : children(patch))
      if (!separated(child, s, radius, depth - 1, work, port_normals, counts))
        return false;
    return true;
  }

  struct BackgroundGeometry
  {
    BackgroundGeometry(const Triangulation<3> &background,
                       const Mapping<3>       &mapping,
                       GrowthWork             *counts = nullptr)
      : cache(background, mapping)
      , mapping(mapping)
      , counts(counts)
    {
      count_work(counts, GrowthOperation::background_cache_constructions);
      const auto *q  = dynamic_cast<const MappingQ<3> *>(&mapping);
      const auto *fe = dynamic_cast<const MappingFE<3> *>(&mapping);
      AssertThrow((q && q->get_degree() == 1) || (fe && fe->get_degree() == 1),
                  ExcMessage(
                    "Vascular generation requires a first-order mapping."));
      std::vector<std::pair<BoundingBox<3>, std::size_t>> boxes;
      for (const auto &cell : background.active_cell_iterators())
        {
          count_work(counts, GrowthOperation::background_cell_visits);
          if (cell->is_locally_owned())
            {
              AssertThrow(
                cell->reference_cell().is_hyper_cube() ||
                  cell->reference_cell().is_simplex(),
                ExcMessage(
                  "Only tetrahedral and hexahedral background cells are supported."));
              for (const auto face : cell->face_indices())
                {
                  count_work(counts, GrowthOperation::boundary_face_visits);
                  if (cell->face(face)->at_boundary())
                    {
                      Patch patch;
                      for (const auto v : cell->face(face)->vertex_indices())
                        patch.push_back(cell->face(face)->vertex(v));
                      boxes.emplace_back(BoundingBox<3>(patch), patches.size());
                      patches.push_back(std::move(patch));
                    }
                }
            }
        }
      count_work(counts, GrowthOperation::boundary_index_builds);
      boundary = pack_rtree(boxes);
    }

    std::optional<Triangulation<3>::active_cell_iterator>
    containing(const Point<3> &p, double tolerance = 0.) const
    {
      count_work(counts, GrowthOperation::point_cell_queries);
      std::optional<Triangulation<3>::active_cell_iterator> result;
      Point<3>                                              lo = p, hi = p;
      // Inverse Q1 maps on warped cells can place a shared-face point just
      // outside the reference cell. Wall separation is certified separately.
      for (unsigned int d = 0; d < 3; ++d)
        {
          const double pad = 64. * std::numeric_limits<double>::epsilon() *
                             std::max(1., p.norm());
          lo[d] -= pad;
          hi[d] += pad;
        }
      const BoundingBox<3> query(std::make_pair(lo, hi));
      for (const auto &[box, cell] :
           cache.get_locally_owned_cell_bounding_boxes_rtree() |
             bgi::adaptors::queried(bgi::intersects(query)))
        {
          count_work(counts, GrowthOperation::point_cell_hits);
          try
            {
              const auto unit = mapping.transform_real_to_unit_cell(cell, p);
              if (reference_contains(cell, unit, tolerance) &&
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

    inline bool
    planar_section(const Point<3> &p, const Point<3> &normal) const
    {
      const double guard =
        64. * std::numeric_limits<double>::epsilon() * std::max(1., p.norm());
      Point<3> lo = p, hi = p;
      for (unsigned int d = 0; d < 3; ++d)
        {
          lo[d] -= guard;
          hi[d] += guard;
        }
      for (const auto &[box, index] :
           boundary | bgi::adaptors::queried(bgi::intersects(
                        BoundingBox<3>(std::make_pair(lo, hi)))))
        {
          const auto &patch = patches[index];
          if (!std::all_of(patch.begin(), patch.end(), [&](const auto &v) {
                return std::abs((v - p) * normal) <= guard;
              }))
            continue;
          // All corners must lie in the declared plane. Only then is the
          // polygon test valid; a warped Q1 face is never flattened here.
          if (patch.size() == 3 &&
              in_triangle(p, patch[0], patch[1], patch[2], guard))
            return true;
          if (patch.size() == 4 &&
              (in_triangle(p, patch[0], patch[1], patch[3], guard) ||
               in_triangle(p, patch[0], patch[3], patch[2], guard)))
            return true;
        }
      return false;
    }

    inline bool
    contains(const Segment<3>       &s,
             double                  radius,
             unsigned int            depth,
             unsigned int            work,
             std::array<Point<3>, 2> port_normals = {}) const
    {
      const double guard = 64. * std::numeric_limits<double>::epsilon() *
                           std::max({1., s.first.norm(), s.second.norm()});
      const std::array<Point<3>, 2> endpoints{{s.first, s.second}};
      bool                          open = false;
      for (auto &normal : port_normals)
        normal = normalized_port_normal(normal);
      for (unsigned int i = 0; i < 2; ++i)
        if (port_normals[i].norm() > 0.)
          {
            open = true;
            if (!planar_section(endpoints[i], port_normals[i]) ||
                (endpoints[1 - i] - endpoints[i]) * port_normals[i] <=
                  radius + guard)
              return false;
          }
      if (!containing(s.first, port_normals[0].norm() > 0. ? 1.e-12 : 0.) ||
          (open && !containing(.5 * (s.first + s.second))))
        return false;
      return boundary_separated(s, radius, depth, work, port_normals);
    }

    bool
    boundary_separated(const Segment<3>       &s,
                       double                  radius,
                       unsigned int            depth,
                       unsigned int            work,
                       std::array<Point<3>, 2> port_normals = {}) const
    {
      count_work(counts, GrowthOperation::wall_queries);
      const double guard = 64. * std::numeric_limits<double>::epsilon() *
                           std::max({1., s.first.norm(), s.second.norm()});
      const std::array<Point<3>, 2> endpoints{{s.first, s.second}};
      for (auto &normal : port_normals)
        normal = normalized_port_normal(normal);
      for (unsigned int i = 0; i < 2; ++i)
        if (port_normals[i].norm_square() > 0. &&
            (endpoints[1 - i] - endpoints[i]) * port_normals[i] <=
              radius + guard)
          return false;
      Point<3> lo, hi;
      for (unsigned int d = 0; d < 3; ++d)
        {
          lo[d] = std::min(s.first[d], s.second[d]) - radius - guard;
          hi[d] = std::max(s.first[d], s.second[d]) + radius + guard;
        }
      const BoundingBox<3> query(std::make_pair(lo, hi));
      for (const auto &[box, index] :
           boundary | bgi::adaptors::queried(bgi::intersects(query)))
        if (!separated(
              patches[index], s, radius, depth, work, port_normals, counts))
          return false;
      // With an interior anchor, boundary separation keeps the connected
      // retained envelope inside, including in nonconvex domains; only its
      // declared planar cap may meet the boundary.
      return true;
    }
    GridTools::Cache<3>                           cache;
    const Mapping<3>                             &mapping;
    std::vector<Patch>                            patches;
    RTree<std::pair<BoundingBox<3>, std::size_t>> boundary;
    GrowthWork                                   *counts;
  };
  // Boost's installed segment distance reduces to endpoint projections after
  // an intersection check. Skew 3D segments also need an interior minimum.
  // On each interval with a fixed clamped projection onto b, the residual
  // is affine in the parameter on a, so its squared norm is quadratic.
  inline double
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

  struct Edge
  {
    Id                      id;
    std::array<Id, 2>       nodes;
    std::array<Point<3>, 2> points;
    double                  radius;
    template <class Archive>
    void
    serialize(Archive &ar, unsigned int)
    {
      ar &id &nodes &points &radius;
    }
  };
  inline bool
  compatible(const Edge &a,
             const Edge &b,
             double      minimum_angle,
             double      clearance,
             GrowthWork *counts = nullptr)
  {
    count_work(counts, GrowthOperation::pair_checks);
    for (unsigned int i = 0; i < 2; ++i)
      for (unsigned int j = 0; j < 2; ++j)
        if (a.nodes[i] == b.nodes[j])
          {
            // Straight incident rays meet only at this junction. Their
            // lumen overlap is the permitted junction raccord.
            const auto   u      = a.points[1 - i] - a.points[i];
            const auto   v      = b.points[1 - j] - b.points[j];
            const double cosine = (u * v) / (u.norm() * v.norm());
            return cosine <= std::cos(minimum_angle * numbers::PI / 180.);
          }
    const double guard = 64. * std::numeric_limits<double>::epsilon() *
                         std::max({1.,
                                   a.points[0].norm(),
                                   a.points[1].norm(),
                                   b.points[0].norm(),
                                   b.points[1].norm()});
    const double distance = a.radius + b.radius + clearance + guard;
    return segment_distance_square({a.points[0], a.points[1]},
                                   {b.points[0], b.points[1]}) >
           distance * distance;
  }

} // namespace ImmersX::Vascularization::internal
#endif
