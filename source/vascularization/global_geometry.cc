// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/config.h>

#include <deal.II/base/index_set.h>
#include <deal.II/base/numbers.h>

#include <boost/serialization/set.hpp>
#include <boost/serialization/utility.hpp>

#include <immersx/vascularization/distribution.h>
#include <immersx/vascularization/global_geometry.h>

#include <algorithm>
#include <numeric>
#include <set>
#include <tuple>

#include "global_geometry_internal.h"
#include "growth_geometry.h"
#include "growth_wall_certificates.h"
#include "network_serialization.h"
#ifdef DEAL_II_WITH_PETSC
#  include <petsctao.h>
#endif

namespace ImmersX::Vascularization
{
  using namespace dealii;

  void
  GlobalGeometryParameters::validate() const
  {
    bool valid =
      std::isfinite(minimum_length_m) && minimum_length_m >= 0. &&
      std::isfinite(maximum_length_m) && maximum_length_m >= 0. &&
      (maximum_length_m == 0. || maximum_length_m >= minimum_length_m) &&
      std::isfinite(clearance_m) && clearance_m >= 0. &&
      maximum_outer_iterations > 0 && maximum_inner_iterations > 0 &&
      geometry_work > 0 && geometry_depth <= 30 &&
      std::isfinite(constraint_tolerance) && constraint_tolerance > 0. &&
      std::isfinite(gradient_tolerance) && gradient_tolerance > 0.;
    for (const auto system : {System::arterial, System::portal, System::venous})
      valid = valid && metabolic_demand_W_m3.count(system) &&
              std::isfinite(metabolic_demand_W_m3.at(system)) &&
              metabolic_demand_W_m3.at(system) > 0.;
    AssertThrow(
      valid,
      ExcMessage(
        "Global forest geometry requires positive power densities, finite SI bounds and positive solver budgets/tolerances."));
  }

  namespace
  {
    using Point = dealii::Point<3>;
    using internal::BackgroundGeometry;
    using internal::Edge;
    unsigned int
    rank(MPI_Comm comm)
    {
      return Utilities::MPI::this_mpi_process(comm);
    }
    unsigned int
    size(MPI_Comm comm)
    {
      return Utilities::MPI::n_mpi_processes(comm);
    }
    bool
    phase(ValidationReport  &report,
          bool               valid,
          const std::string &text,
          MPI_Comm           comm)
    {
      if (Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1)
        return true;
      report.issues.push_back({"global geometry", invalid_id, text});
      return false;
    }
    template <class T>
    struct Directory
    {
      std::map<Id, T> entries;
      MPI_Comm        comm;
      Directory(const std::map<Id, T> &owned, MPI_Comm comm)
        : comm(comm)
      {
        std::map<unsigned int, std::map<Id, T>> outgoing;
        for (const auto &[id, x] : owned)
          outgoing[id % size(comm)].emplace(id, x);
        bool valid = true;
        for (const auto &[owner, part] :
             Utilities::MPI::some_to_some(comm, outgoing))
          for (const auto &[id, x] : part)
            valid = entries.emplace(id, x).second && valid;
        AssertThrow(
          Utilities::MPI::min(static_cast<unsigned int>(valid), comm),
          ExcMessage(
            "Physical identifiers must have exactly one native owner."));
      }
      std::map<Id, T>
      lookup(const std::set<Id> &ids) const
      {
        std::map<unsigned int, std::set<Id>> requests;
        for (auto id : ids)
          requests[id % size(comm)].insert(id);
        std::map<unsigned int, std::map<Id, T>> replies;
        for (const auto &[source, part] :
             Utilities::MPI::some_to_some(comm, requests))
          for (auto id : part)
            if (entries.count(id))
              replies[source].emplace(id, entries.at(id));
        std::map<Id, T> result;
        for (const auto &[source, part] :
             Utilities::MPI::some_to_some(comm, replies))
          result.insert(part.begin(), part.end());
        return result;
      }
    };
    template <class T>
    std::map<Id, T>
    owned(const std::map<Id, T> &entries, MPI_Comm comm)
    {
      std::map<Id, T> result;
      for (const auto &[id, x] : entries)
        if (x.owner == rank(comm))
          result.emplace(id, x);
      return result;
    }
    template <class T>
    std::set<Id>
    keys(const std::map<Id, T> &map)
    {
      std::set<Id> result;
      for (const auto &[id, x] : map)
        result.insert(id);
      return result;
    }
    Edge
    edge(const NetworkData &n, Id id)
    {
      const auto &v = n.vessels.at(id);
      return {id,
              v.nodes,
              {{n.nodes.at(v.nodes[0]).position,
                n.nodes.at(v.nodes[1]).position}},
              std::max(v.radii[0], v.radii[1])};
    }
    BoundingBox<3>
    box(const Edge &e, double clearance)
    {
      Point lo, hi;
      for (unsigned int d = 0; d < 3; ++d)
        {
          lo[d] =
            std::min(e.points[0][d], e.points[1][d]) - e.radius - clearance;
          hi[d] =
            std::max(e.points[0][d], e.points[1][d]) + e.radius + clearance;
        }
      return BoundingBox<3>({lo, hi});
    }
    struct Collision
    {
      Edge         first, second;
      unsigned int other_owner = 0;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &first &second &other_owner;
      }
    };
    std::vector<Collision>
    collisions(const NetworkData &network, double clearance, MPI_Comm comm)
    {
      std::vector<std::pair<BoundingBox<3>, Id>> entries;
      for (const auto &[id, v] : network.vessels)
        if (v.owner == rank(comm))
          entries.emplace_back(box(edge(network, id), clearance), id);
      const auto                  tree = pack_rtree(entries);
      std::vector<BoundingBox<3>> cover;
      if (!tree.empty())
        cover = extract_rtree_level(tree, 1);
      const auto covers = Utilities::MPI::all_gather(comm, cover);
      std::map<unsigned int,
               std::vector<std::pair<Edge, std::array<double, 2>>>>
        send;
      for (const auto &[bounds, id] : entries)
        for (unsigned int p = 0; p < covers.size(); ++p)
          if (std::any_of(covers[p].begin(),
                          covers[p].end(),
                          [&](const auto &b) {
                            return bounds.has_overlap_with(b);
                          }))
            send[p].push_back(
              {edge(network, id), network.vessels.at(id).radii});
      std::map<unsigned int, std::vector<Collision>> replies;
      for (const auto &[source, queries] :
           Utilities::MPI::some_to_some(comm, send))
        for (const auto &[a, radii_a] : queries)
          for (const auto &[bounds, id] :
               tree | internal::bgi::adaptors::queried(
                        internal::bgi::intersects(box(a, clearance))))
            if (a.id < id)
              {
                const auto b = edge(network, id);
                // Shared physical nodes are actual junctions, never a role or
                // compartment based exemption between distinct terminals.
                if (!internal::compatible(a,
                                          b,
                                          0.,
                                          clearance,
                                          radii_a,
                                          network.vessels.at(id).radii))
                  replies[source].push_back({a, b, rank(comm)});
              }
      std::vector<Collision> result;
      for (const auto &[source, part] :
           Utilities::MPI::some_to_some(comm, replies))
        result.insert(result.end(), part.begin(), part.end());
      std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
        return std::tie(a.first.id, a.second.id) <
               std::tie(b.first.id, b.second.id);
      });
      return result;
    }

    struct Ball
    {
      Id     node = invalid_id;
      Point  center;
      double radius = 0.;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &node &center &radius;
      }
    };
    struct Separation
    {
      Id     first = invalid_id, second = invalid_id;
      double distance = 0.;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &first &second &distance;
      }
    };
    struct NodeIndex
    {
      std::uint64_t index = 0;
      Point         point;
      bool          fixed = false;
      unsigned int  owner = 0;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &index &point &fixed &owner;
      }
    };
    struct LengthRow
    {
      std::array<std::uint64_t, 3> first{}, second{};
      Point                        anchor;
      bool                         constant_second = false;
      std::uint64_t                auxiliary       = 0;
      double                       lower = 0., upper = 0., weight = 0.;
    };

    struct Witness
    {
      double parameter = .5;
      Point  center;
      double radius = 0.;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &parameter &center &radius;
      }
    };
    std::pair<double, double>
    closest_parameters(const Edge &a, const Edge &b)
    {
      const auto u = a.points[1] - a.points[0], v = b.points[1] - b.points[0],
                 w           = a.points[0] - b.points[0];
      const double        vv = v.norm_square(), uv = u * v, wv = w * v;
      std::vector<double> cuts{0., 1.};
      if (uv != 0.)
        for (double t : {-wv / uv, (vv - wv) / uv})
          if (t > 0. && t < 1.)
            cuts.push_back(t);
      std::sort(cuts.begin(), cuts.end());
      double                    best = std::numeric_limits<double>::infinity();
      std::pair<double, double> result{.5, .5};
      for (std::size_t i = 1; i < cuts.size(); ++i)
        {
          const double middle     = .5 * (cuts[i - 1] + cuts[i]),
                       projection = (wv + middle * uv) / vv;
          auto offset = w, slope = u;
          if (projection >= 1.)
            offset -= v;
          else if (projection > 0.)
            {
              offset -= wv / vv * v;
              slope -= uv / vv * v;
            }
          const double t      = slope.norm_square() > 0. ?
                                  std::clamp(-(offset * slope) / slope.norm_square(),
                                        cuts[i - 1],
                                        cuts[i]) :
                                  middle;
          const double s      = std::clamp((wv + t * uv) / vv, 0., 1.);
          const double square = (w + t * u - s * v).norm_square();
          if (square < best)
            {
              best   = square;
              result = {t, s};
            }
        }
      return result;
    }
    Point
    triangle_projection(const Point &p,
                        const Point &a,
                        const Point &b,
                        const Point &c)
    {
      const auto   ab = b - a, ac = c - a, ap = p - a;
      const double d1 = ab * ap, d2 = ac * ap;
      if (d1 <= 0. && d2 <= 0.)
        return a;
      const auto   bp = p - b;
      const double d3 = ab * bp, d4 = ac * bp;
      if (d3 >= 0. && d4 <= d3)
        return b;
      const double vc = d1 * d4 - d3 * d2;
      if (vc <= 0. && d1 >= 0. && d3 <= 0.)
        return a + (d1 / (d1 - d3)) * ab;
      const auto   cp = p - c;
      const double d5 = ab * cp, d6 = ac * cp;
      if (d6 >= 0. && d5 <= d6)
        return c;
      const double vb = d5 * d2 - d1 * d6;
      if (vb <= 0. && d2 >= 0. && d6 <= 0.)
        return a + (d2 / (d2 - d6)) * ac;
      const double va = d3 * d6 - d5 * d4;
      if (va <= 0. && d4 - d3 >= 0. && d5 - d6 >= 0.)
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b);
      const double denominator = 1. / (va + vb + vc);
      return a + (vb * denominator) * ab + (vc * denominator) * ac;
    }
    Witness
    wall_witness(const Edge &e, const internal::Patch &patch)
    {
      Witness    result;
      double     best      = std::numeric_limits<double>::infinity();
      const auto direction = e.points[1] - e.points[0];
      auto       consider  = [&](const Point &p) {
        const double t =
          std::clamp(((p - e.points[0]) * direction) / direction.norm_square(),
                     0.,
                     1.);
        const double d = p.distance_square(e.points[0] + t * direction);
        if (d < best)
          {
            best             = d;
            result.center    = p;
            result.parameter = t;
          }
      };
      if (patch.size() == 3)
        {
          for (const auto &p : e.points)
            consider(triangle_projection(p, patch[0], patch[1], patch[2]));
          const auto normal =
            cross_product_3d(patch[1] - patch[0], patch[2] - patch[0]);
          const double denominator = normal * direction;
          if (denominator != 0.)
            {
              const double t =
                (normal * (patch[0] - e.points[0])) / denominator;
              if (t >= 0. && t <= 1.)
                {
                  const auto p = e.points[0] + t * direction;
                  if (internal::in_triangle(
                        p, patch[0], patch[1], patch[2], 1.e-13))
                    consider(p);
                }
            }
        }
      else
        {
          // These samples lie on the actual bilinear face. They construct
          // valid boundary-centred exclusion balls; they are not certificates.
          for (unsigned int i = 0; i <= 4; ++i)
            for (unsigned int j = 0; j <= 4; ++j)
              {
                const double u = i / 4., v = j / 4.;
                consider((1 - u) * (1 - v) * patch[0] + u * (1 - v) * patch[1] +
                         (1 - u) * v * patch[2] + u * v * patch[3]);
              }
        }
      for (unsigned int i = 0; i < patch.size(); ++i)
        {
          unsigned int j = (i + 1) % patch.size();
          if (patch.size() == 4)
            {
              constexpr unsigned int next[4] = {1, 3, 0, 2};
              j                              = next[i];
            }
          Edge       side{0, {{0, 1}}, {{patch[i], patch[j]}}, 0.};
          const auto p = closest_parameters(e, side);
          consider(side.points[0] +
                   p.second * (side.points[1] - side.points[0]));
        }
      result.radius = e.radius;
      return result;
    }
    std::map<Id, Witness>
    boundary_failures(const NetworkData              &network,
                      const std::set<Id>             &immutable,
                      BackgroundGeometry             &geometry,
                      const GlobalGeometryParameters &controls,
                      ValidationReport               &validation,
                      MPI_Comm                        comm)
    {
      std::map<Id, dealii::Segment<3>> segments;
      for (const auto &[id, v] : network.vessels)
        if (v.owner == rank(comm) && !immutable.count(id))
          {
            const auto e = edge(network, id);
            segments.emplace(id, dealii::Segment<3>{e.points[0], e.points[1]});
          }
      std::map<Id, internal::GrowthWallRequest> native_requests;
      for (const auto &[id, segment] : segments)
        {
          internal::GrowthWallRequest request;
          request.edge    = edge(network, id);
          request.padding = controls.clearance_m;
          for (unsigned int end = 0; end < 2; ++end)
            request.port_normals[end] =
              network.nodes.at(request.edge.nodes[end]).inward_port_normal;
          native_requests.emplace(id, request);
        }
      const auto native =
        internal::certify_growth_walls(geometry,
                                       native_requests,
                                       controls.geometry_depth,
                                       controls.geometry_work,
                                       comm);
      AssertThrow(
        native.validation.valid(),
        ExcMessage(
          "Native global forest wall certificate rejected its controls."));
      const auto            profiles = geometry.crossings(segments, comm);
      std::map<Id, Witness> failures;
      std::map<unsigned int, std::map<Id, Edge>> queries;
      const auto &covers = geometry.routing(comm).boxes;
      for (const auto &[id, s] : segments)
        {
          if (native.local.at(id))
            continue;
          const auto e      = edge(network, id);
          const auto bounds = box(e, controls.clearance_m);
          for (unsigned int p = 0; p < covers.size(); ++p)
            if (std::any_of(covers[p].begin(),
                            covers[p].end(),
                            [&](const auto &b) {
                              return bounds.has_overlap_with(b);
                            }))
              queries[p].emplace(id, e);
          for (const auto &interval : profiles.at(id))
            if (interval.h == 0.)
              {
                Witness w;
                w.parameter = .5 * (interval.begin + interval.end);
                w.center =
                  e.points[0] + w.parameter * (e.points[1] - e.points[0]);
                w.radius = e.radius + controls.clearance_m;
                failures.emplace(id, w);
                break;
              }
        }
      std::map<unsigned int, std::map<Id, Witness>> replies;
      for (const auto &[source, part] :
           Utilities::MPI::some_to_some(comm, queries))
        for (const auto &[id, e] : part)
          {
            double  best = std::numeric_limits<double>::infinity();
            Witness best_w;
            for (const auto &patch : geometry.patches)
              {
                const auto w = wall_witness(e, patch);
                const auto p =
                  e.points[0] + w.parameter * (e.points[1] - e.points[0]);
                const double d = p.distance_square(w.center);
                if (d < best)
                  {
                    best   = d;
                    best_w = w;
                  }
              }
            if (std::isfinite(best))
              {
                best_w.radius       = e.radius + controls.clearance_m;
                replies[source][id] = best_w;
              }
          }
      const auto           exterior = keys(failures);
      std::map<Id, double> nearest;
      for (const auto &[source, part] :
           Utilities::MPI::some_to_some(comm, replies))
        for (const auto &[id, w] : part)
          if (!exterior.count(id))
            {
              const auto   e        = edge(network, id);
              const double distance = w.center.distance_square(
                e.points[0] + w.parameter * (e.points[1] - e.points[0]));
              if (!nearest.count(id) || distance < nearest.at(id))
                {
                  nearest[id]  = distance;
                  failures[id] = w;
                }
            }
      bool complete = true;
      for (const auto &[id, safe] : native.local)
        complete = complete && (safe || failures.count(id));
      if (
        !phase(
          validation,
          complete,
          "An uncertified envelope has no admissible exterior or boundary witness; the forest cannot be adopted.",
          comm))
        return failures;
      // A centreline exterior sample defines an outside ball. Its conservative
      // distance to all physical face boxes is a valid contained-ball radius.
      std::map<unsigned int, std::map<Id, Point>> distance_queries;
      for (const auto &[id, w] : failures)
        if (std::any_of(profiles.at(id).begin(),
                        profiles.at(id).end(),
                        [](const auto &i) { return i.h == 0.; }))
          for (unsigned int p = 0; p < size(comm); ++p)
            distance_queries[p].emplace(id, w.center);
      std::map<unsigned int, std::map<Id, double>> distances;
      for (const auto &[source, part] :
           Utilities::MPI::some_to_some(comm, distance_queries))
        for (const auto &[id, p] : part)
          {
            double d = std::numeric_limits<double>::infinity();
            for (const auto &patch : geometry.patches)
              {
                const BoundingBox<3> b(patch);
                const auto &[lo, hi] = b.get_boundary_points();
                double square        = 0.;
                for (unsigned int k = 0; k < 3; ++k)
                  {
                    const double delta = p[k] - std::clamp(p[k], lo[k], hi[k]);
                    square += delta * delta;
                  }
                d = std::min(d, std::sqrt(square));
              }
            distances[source][id] = d;
          }
      std::map<Id, double> minimum;
      for (const auto &[source, part] :
           Utilities::MPI::some_to_some(comm, distances))
        for (const auto &[id, d] : part)
          {
            if (!minimum.count(id))
              minimum[id] = d;
            else
              minimum[id] = std::min(minimum[id], d);
          }
      for (const auto &[id, d] : minimum)
        if (std::isfinite(d))
          failures.at(id).radius += d;
      return failures;
    }
    Tensor<1, 3>
    perturbation(const Edge &e, Id key)
    {
      auto         tangent = e.points[1] - e.points[0];
      Tensor<1, 3> axis;
      axis[key % 3] = 1.;
      auto normal   = cross_product_3d(tangent, axis);
      if (normal.norm_square() < 1.e-24 * tangent.norm_square())
        {
          axis                = Tensor<1, 3>();
          axis[(key + 1) % 3] = 1.;
          normal              = cross_product_3d(tangent, axis);
        }
      return normal / normal.norm();
    }
    struct SplitRequest
    {
      Id           vessel    = invalid_id;
      double       parameter = .5;
      Tensor<1, 3> shift;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &vessel &parameter &shift;
      }
    };
    struct SplitResult
    {
      Id           node = invalid_id;
      Point        point;
      unsigned int owner = 0;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &node &point &owner;
      }
    };
    bool
    extend(NetworkData                  &network,
           std::map<Id, double>         &flows,
           const std::set<Id>           &immutable,
           const std::vector<Collision> &conflicts,
           const std::map<Id, Witness>  &walls,
           std::map<Id, double>         &minimum_lengths,
           std::vector<Ball>            &balls,
           std::vector<Separation>      &separations,
           double                        clearance,
           GlobalGeometryReport         &report,
           ValidationReport             &validation,
           MPI_Comm                      comm)
    {
      std::map<unsigned int, std::vector<SplitRequest>> requests;
      bool                                              valid = true;
      const auto                                        guard = [&](double r) {
        return r +
               64. * std::numeric_limits<double>::epsilon() * std::max(1., r);
      };
      // Original-vessel membership is sparse and queried for remote conflicts.
      std::map<Id, unsigned int> local_immutable;
      for (Id id : immutable)
        if (network.vessels.count(id) &&
            network.vessels.at(id).owner == rank(comm))
          local_immutable[id] = 1;
      Directory<unsigned int> frozen_directory(local_immutable, comm);
      std::set<Id>            queried;
      for (const auto &c : conflicts)
        {
          queried.insert(c.first.id);
          queried.insert(c.second.id);
        }
      const auto frozen = frozen_directory.lookup(queried);
      for (const auto &c : conflicts)
        {
          const bool fa = frozen.count(c.first.id),
                     fb = frozen.count(c.second.id);
          valid         = valid && !(fa && fb);
          const auto t  = closest_parameters(c.first, c.second);
          auto       normal =
            cross_product_3d(c.first.points[1] - c.first.points[0],
                             c.second.points[1] - c.second.points[0]);
          if (normal.norm_square() < 1.e-24)
            normal = perturbation(c.first, c.first.id);
          else
            normal /= normal.norm();
          const double amount =
            1.5 * guard(c.first.radius + c.second.radius + clearance);
          if (!fa)
            requests[rank(comm)].push_back(
              {c.first.id, t.first, amount * normal});
          if (!fb)
            requests[c.other_owner].push_back(
              {c.second.id, t.second, -amount * normal});
        }
      for (const auto &[id, w] : walls)
        requests[rank(comm)].push_back(
          {id,
           w.parameter,
           1.5 * guard(w.radius) * perturbation(edge(network, id), id)});
      if (
        !phase(
          validation,
          valid,
          "Two immutable original vessels intersect; original geometry cannot be changed by the NLP.",
          comm))
        return false;
      const auto incoming = Utilities::MPI::some_to_some(comm, requests);
      std::map<Id, SplitRequest> chosen;
      for (const auto &[source, part] : incoming)
        for (const auto &r : part)
          chosen.emplace(r.vessel, r);
      Id node_max = 0, vessel_max = 0;
      for (const auto &[id, n] : network.nodes)
        if (n.owner == rank(comm))
          node_max = std::max(node_max, id);
      for (const auto &[id, v] : network.vessels)
        if (v.owner == rank(comm))
          vessel_max = std::max(vessel_max, id);
      node_max   = Utilities::MPI::max(node_max, comm);
      vessel_max = Utilities::MPI::max(vessel_max, comm);
      const auto counts =
        Utilities::MPI::all_gather(comm, std::uint64_t(chosen.size()));
      const auto total =
        std::accumulate(counts.begin(), counts.end(), std::uint64_t(0));
      const auto prefix = std::accumulate(counts.begin(),
                                          counts.begin() + rank(comm),
                                          std::uint64_t(0));
      if (!phase(
            validation,
            node_max < invalid_id - total && vessel_max < invalid_id - total,
            "Extension identifiers exceed the persistent physical ID range.",
            comm))
        return false;
      std::map<Id, SplitResult> created;
      std::uint64_t             counter = 0;
      for (const auto &[id, r] : chosen)
        {
          const auto   old     = network.vessels.at(id);
          const auto   e       = edge(network, id);
          const double l       = e.points[0].distance(e.points[1]);
          const double t       = std::clamp(r.parameter, .05, .95);
          const Id     node_id = node_max + 1 + prefix + counter,
                   new_vessel  = vessel_max + 1 + prefix + counter;
          ++counter;
          Node       n;
          auto       normal  = perturbation(e, id);
          const auto tangent = (e.points[1] - e.points[0]) / l;
          const auto lateral = cross_product_3d(tangent, normal);
          // A stable angular phase gives distinct initial extension points
          // when several trees cross at exactly the same geometric point.
          const double angle =
            2. * numbers::PI *
            std::fmod(static_cast<double>(id % 1000003) * .6180339887498948482,
                      1.);
          const auto shift = r.shift.norm() * (std::cos(angle) * normal +
                                               std::sin(angle) * lateral);
          n.position   = e.points[0] + t * (e.points[1] - e.points[0]) + shift;
          n.owner      = rank(comm);
          n.provenance = Provenance::generated;
          network.nodes.emplace(node_id, n);
          const double radius   = (1. - t) * old.radii[0] + t * old.radii[1];
          auto         proximal = old, distal = old;
          proximal.nodes[1]      = node_id;
          proximal.radii[1]      = radius;
          distal.nodes[0]        = node_id;
          distal.radii[0]        = radius;
          network.vessels.at(id) = proximal;
          network.vessels.emplace(new_vessel, distal);
          flows[new_vessel]           = flows.at(id);
          minimum_lengths[id]         = t * l;
          minimum_lengths[new_vessel] = (1. - t) * l;
          created[id]                 = {node_id, n.position, rank(comm)};
        }
      std::map<unsigned int, std::map<Id, SplitResult>> replies;
      for (const auto &[source, part] : incoming)
        for (const auto &r : part)
          replies[source][r.vessel] = created.at(r.vessel);
      std::map<Id, SplitResult> resolved;
      for (const auto &[source, part] :
           Utilities::MPI::some_to_some(comm, replies))
        resolved.insert(part.begin(), part.end());
      for (const auto &c : conflicts)
        {
          const auto   t = closest_parameters(c.first, c.second);
          const double separation =
            guard(c.first.radius + c.second.radius + clearance);
          if (frozen.count(c.first.id))
            balls.push_back({resolved.at(c.second.id).node,
                             c.first.points[0] + t.first * (c.first.points[1] -
                                                            c.first.points[0]),
                             separation});
          else if (frozen.count(c.second.id))
            balls.push_back(
              {resolved.at(c.first.id).node,
               c.second.points[0] +
                 t.second * (c.second.points[1] - c.second.points[0]),
               separation});
          else
            separations.push_back({resolved.at(c.first.id).node,
                                   resolved.at(c.second.id).node,
                                   separation});
        }
      for (const auto &[id, w] : walls)
        balls.push_back({resolved.at(id).node, w.center, guard(w.radius)});
      report.inserted_nodes += total;
      return true;
    }

#ifdef DEAL_II_WITH_PETSC
    void
    checked(PetscErrorCode error)
    {
      AssertThrow(error == 0,
                  ExcMessage(
                    "PETSc TAO global forest geometry operation failed (code " +
                    std::to_string(error) + ")."));
    }
    struct NLP
    {
      MPI_Comm      comm;
      double        scale = 1., power_scale = 1.;
      std::uint64_t begin = 0, local_size = 0, total_size = 0, row_begin = 0,
                    total_rows = 0;
      std::map<Id, NodeIndex>           nodes;
      std::vector<LengthRow>            rows;
      std::map<Id, std::uint64_t>       vessel_lengths;
      IndexSet                          locally_owned, locally_relevant;
      std::vector<PetscInt>             ghost_indices;
      std::map<std::uint64_t, PetscInt> slots;
      Vec x = nullptr, lower = nullptr, upper = nullptr, ghost = nullptr,
          constraints = nullptr;
      Mat jacobian    = nullptr;
      Tao tao         = nullptr;
      ~NLP()
      {
        if (tao)
          TaoDestroy(&tao);
        if (jacobian)
          MatDestroy(&jacobian);
        if (constraints)
          VecDestroy(&constraints);
        if (ghost)
          VecDestroy(&ghost);
        if (upper)
          VecDestroy(&upper);
        if (lower)
          VecDestroy(&lower);
        if (x)
          VecDestroy(&x);
      }
      const PetscScalar *
      values(Vec input, Vec &local)
      {
        checked(VecCopy(input, ghost));
        checked(VecGhostUpdateBegin(ghost, INSERT_VALUES, SCATTER_FORWARD));
        checked(VecGhostUpdateEnd(ghost, INSERT_VALUES, SCATTER_FORWARD));
        checked(VecGhostGetLocalForm(ghost, &local));
        const PetscScalar *data = nullptr;
        checked(VecGetArrayRead(local, &data));
        return data;
      }
      void
      release(Vec local, const PetscScalar *&data)
      {
        checked(VecRestoreArrayRead(local, &data));
        checked(VecGhostRestoreLocalForm(ghost, &local));
      }
      double
      value(const PetscScalar *data, std::uint64_t global) const
      {
        return PetscRealPart(data[slots.at(global)]);
      }
      std::array<double, 7>
      arguments(const LengthRow &row, const PetscScalar *data) const
      {
        std::array<double, 7> a{};
        for (unsigned int d = 0; d < 3; ++d)
          {
            a[d]     = value(data, row.first[d]);
            a[d + 3] = row.constant_second ? row.anchor[d] / scale :
                                             value(data, row.second[d]);
          }
        a[6] = value(data, row.auxiliary);
        return a;
      }
      static PetscErrorCode
      objective(Tao, Vec x, PetscReal *value, Vec gradient, void *context)
      {
        auto &p = *static_cast<NLP *>(context);
        PetscFunctionBeginUser;
        PetscCall(VecSet(gradient, 0.));
        PetscReal          local = 0.;
        const PetscScalar *data  = nullptr;
        PetscCall(VecGetArrayRead(x, &data));
        for (const auto &row : p.rows)
          if (row.weight > 0.)
            {
              const double w = row.weight * p.scale / p.power_scale;
              local += w * PetscRealPart(data[row.auxiliary - p.begin]);
              PetscCall(VecSetValue(gradient, row.auxiliary, w, INSERT_VALUES));
            }
        PetscCall(VecRestoreArrayRead(x, &data));
        PetscCall(VecAssemblyBegin(gradient));
        PetscCall(VecAssemblyEnd(gradient));
        PetscCallMPI(
          MPI_Allreduce(&local, value, 1, MPIU_REAL, MPI_SUM, p.comm));
        PetscFunctionReturn(0);
      }
      static PetscErrorCode
      equality(Tao, Vec x, Vec output, void *context)
      {
        auto &p = *static_cast<NLP *>(context);
        PetscFunctionBeginUser;
        Vec                local = nullptr;
        const PetscScalar *data  = nullptr;
        try
          {
            data = p.values(x, local);
          }
        catch (...)
          {
            PetscFunctionReturn(PETSC_ERR_LIB);
          }
        PetscScalar *out = nullptr;
        PetscCall(VecGetArray(output, &out));
        for (std::size_t i = 0; i < p.rows.size(); ++i)
          out[i] =
            internal::global_length_constraint(p.arguments(p.rows[i], data))
              .first;
        PetscCall(VecRestoreArray(output, &out));
        try
          {
            p.release(local, data);
          }
        catch (...)
          {
            PetscFunctionReturn(PETSC_ERR_LIB);
          }
        PetscFunctionReturn(0);
      }
      static PetscErrorCode
      derivative(Tao, Vec x, Mat matrix, Mat preconditioner, void *context)
      {
        auto &p = *static_cast<NLP *>(context);
        PetscFunctionBeginUser;
        Vec                local = nullptr;
        const PetscScalar *data  = nullptr;
        try
          {
            data = p.values(x, local);
          }
        catch (...)
          {
            PetscFunctionReturn(PETSC_ERR_LIB);
          }
        PetscCall(MatZeroEntries(matrix));
        for (std::size_t i = 0; i < p.rows.size(); ++i)
          {
            const auto &r = p.rows[i];
            const auto  d =
              internal::global_length_constraint(p.arguments(r, data)).second;
            const PetscInt                  row = p.row_begin + i;
            std::map<PetscInt, PetscScalar> coefficients;
            for (unsigned int k = 0; k < 3; ++k)
              {
                coefficients[r.first[k]] += d[k];
                if (!r.constant_second)
                  coefficients[r.second[k]] += d[k + 3];
              }
            coefficients[r.auxiliary] += d[6];
            for (const auto &[column, v] : coefficients)
              PetscCall(MatSetValue(matrix, row, column, v, INSERT_VALUES));
          }
        try
          {
            p.release(local, data);
          }
        catch (...)
          {
            PetscFunctionReturn(PETSC_ERR_LIB);
          }
        PetscCall(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY));
        if (matrix != preconditioner)
          {
            PetscCall(MatCopy(matrix, preconditioner, SAME_NONZERO_PATTERN));
          }
        PetscFunctionReturn(0);
      }
    };

    bool
    solve(NetworkData                    &network,
          const std::set<Id>             &fixed,
          const std::set<Id>             &immutable_vessels,
          const std::map<Id, double>     &flows,
          const std::map<Id, double>     &minimum_lengths,
          const std::vector<Ball>        &balls,
          const std::vector<Separation>  &separations,
          const GlobalGeometryParameters &controls,
          double                          scale,
          GlobalGeometryIteration        &iteration,
          ValidationReport               &validation,
          MPI_Comm                        comm)
    {
      NLP p;
      p.comm                    = comm;
      p.scale                   = scale;
      const auto    own_nodes   = owned(network.nodes, comm);
      const auto    own_vessels = owned(network.vessels, comm);
      std::uint64_t local_edges = 0;
      for (const auto &[id, v] : own_vessels)
        if (!immutable_vessels.count(id))
          ++local_edges;
      const std::uint64_t local_coordinates = 3 * own_nodes.size();
      p.local_size =
        local_coordinates + local_edges + balls.size() + separations.size();
      const auto sizes = Utilities::MPI::all_gather(comm, p.local_size);
      p.begin          = std::accumulate(sizes.begin(),
                                sizes.begin() + rank(comm),
                                std::uint64_t(0));
      p.total_size =
        std::accumulate(sizes.begin(), sizes.end(), std::uint64_t(0));
      AssertThrow(p.total_size <= static_cast<std::uint64_t>(
                                    std::numeric_limits<PetscInt>::max()),
                  ExcMessage(
                    "Global geometry exceeds the PETSc integer range."));
      std::map<Id, NodeIndex> numbered;
      std::uint64_t           offset = p.begin;
      for (const auto &[id, n] : own_nodes)
        {
          numbered.emplace(
            id, NodeIndex{offset, n.position, fixed.count(id) > 0, n.owner});
          offset += 3;
        }
      Directory<NodeIndex> directory(numbered, comm);
      auto                 needed = keys(network.nodes);
      for (const auto &b : balls)
        needed.insert(b.node);
      for (const auto &s : separations)
        {
          needed.insert(s.first);
          needed.insert(s.second);
        }
      p.nodes = directory.lookup(needed);
      if (!phase(
            validation,
            p.nodes.size() == needed.size(),
            "Every geometry coordinate must resolve to its native node owner.",
            comm))
        return false;
      double local_power = 0.;
      for (const auto &[id, v] : own_vessels)
        if (!immutable_vessels.count(id))
          {
            const double l = length(network, id);
            const double weight =
              controls.metabolic_demand_W_m3.at(v.system) *
                volume(network, id) / l +
              flows.at(id) * flows.at(id) * resistance(network, id) / l;
            LengthRow row;
            for (unsigned int d = 0; d < 3; ++d)
              {
                row.first[d]  = p.nodes.at(v.nodes[0]).index + d;
                row.second[d] = p.nodes.at(v.nodes[1]).index + d;
              }
            row.auxiliary = offset++;
            row.lower =
              std::max({controls.minimum_length_m,
                        minimum_lengths.count(id) ? minimum_lengths.at(id) : 0.,
                        scale * 1.e-12}) /
              scale;
            row.upper            = controls.maximum_length_m > 0. ?
                                     controls.maximum_length_m / scale :
                                     PETSC_INFINITY;
            row.weight           = weight;
            p.vessel_lengths[id] = row.auxiliary;
            p.rows.push_back(row);
            local_power += weight * l;
          }
      for (const auto &b : balls)
        {
          LengthRow row;
          for (unsigned int d = 0; d < 3; ++d)
            row.first[d] = p.nodes.at(b.node).index + d;
          row.anchor          = b.center;
          row.constant_second = true;
          row.auxiliary       = offset++;
          row.lower           = b.radius / scale;
          row.upper           = PETSC_INFINITY;
          p.rows.push_back(row);
        }
      for (const auto &s : separations)
        {
          LengthRow row;
          for (unsigned int d = 0; d < 3; ++d)
            {
              row.first[d]  = p.nodes.at(s.first).index + d;
              row.second[d] = p.nodes.at(s.second).index + d;
            }
          row.auxiliary = offset++;
          row.lower     = s.distance / scale;
          row.upper     = PETSC_INFINITY;
          p.rows.push_back(row);
        }
      p.power_scale = Utilities::MPI::sum(local_power, comm);
      if (!(p.power_scale > 0.))
        p.power_scale = 1.;
      const auto row_sizes =
        Utilities::MPI::all_gather(comm, std::uint64_t(p.rows.size()));
      p.row_begin = std::accumulate(row_sizes.begin(),
                                    row_sizes.begin() + rank(comm),
                                    std::uint64_t(0));
      p.total_rows =
        std::accumulate(row_sizes.begin(), row_sizes.end(), std::uint64_t(0));
      if (p.total_rows == 0)
        {
          iteration.backend_reason = 1;
          return true;
        }
      std::set<std::uint64_t> ghost;
      for (const auto &row : p.rows)
        {
          for (auto i : row.first)
            if (i < p.begin || i >= p.begin + p.local_size)
              ghost.insert(i);
          if (!row.constant_second)
            for (auto i : row.second)
              if (i < p.begin || i >= p.begin + p.local_size)
                ghost.insert(i);
        }
      p.locally_owned = IndexSet(p.total_size);
      p.locally_owned.add_range(p.begin, p.begin + p.local_size);
      p.locally_relevant = p.locally_owned;
      for (auto i : ghost)
        p.locally_relevant.add_index(i);
      p.locally_relevant.compress();
      for (const auto i : p.locally_owned)
        p.slots[i] = p.locally_owned.index_within_set(i);
      for (auto i : ghost)
        {
          p.slots[i] = p.local_size + p.ghost_indices.size();
          p.ghost_indices.push_back(i);
        }
      checked(VecCreateMPI(comm, p.local_size, p.total_size, &p.x));
      checked(VecDuplicate(p.x, &p.lower));
      checked(VecDuplicate(p.x, &p.upper));
      checked(VecCreateGhost(comm,
                             p.local_size,
                             p.total_size,
                             p.ghost_indices.size(),
                             p.ghost_indices.data(),
                             &p.ghost));
      checked(VecSet(p.lower, PETSC_NINFINITY));
      checked(VecSet(p.upper, PETSC_INFINITY));
      for (const auto &[id, n] : numbered)
        for (unsigned int d = 0; d < 3; ++d)
          {
            const PetscInt i     = n.index + d;
            const double   value = n.point[d] / scale;
            checked(VecSetValue(p.x, i, value, INSERT_VALUES));
            if (n.fixed)
              {
                checked(VecSetValue(p.lower, i, value, INSERT_VALUES));
                checked(VecSetValue(p.upper, i, value, INSERT_VALUES));
              }
          }
      for (const auto &row : p.rows)
        {
          Point a, b = row.anchor;
          for (const auto &[id, n] : p.nodes)
            {
              if (n.index == row.first[0])
                a = n.point;
              if (!row.constant_second && n.index == row.second[0])
                b = n.point;
            }
          const double ell =
            std::clamp(a.distance(b) / scale, row.lower, row.upper);
          checked(VecSetValue(p.x, row.auxiliary, ell, INSERT_VALUES));
          checked(
            VecSetValue(p.lower, row.auxiliary, row.lower, INSERT_VALUES));
          checked(
            VecSetValue(p.upper, row.auxiliary, row.upper, INSERT_VALUES));
        }
      for (auto v : {p.x, p.lower, p.upper})
        {
          checked(VecAssemblyBegin(v));
          checked(VecAssemblyEnd(v));
        }
      checked(VecCreateMPI(comm, p.rows.size(), p.total_rows, &p.constraints));
      checked(
        MatCreateAIJ(comm,
                     p.rows.size(),
                     p.local_size,
                     p.total_rows,
                     p.total_size,
                     std::min<std::uint64_t>(7, p.local_size),
                     nullptr,
                     std::min<std::uint64_t>(7, p.total_size - p.local_size),
                     nullptr,
                     &p.jacobian));
      checked(
        MatSetOption(p.jacobian, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
      checked(TaoCreate(comm, &p.tao));
      checked(TaoSetType(p.tao, TAOALMM));
      checked(TaoSetSolution(p.tao, p.x));
      checked(TaoSetVariableBounds(p.tao, p.lower, p.upper));
      checked(TaoSetObjectiveAndGradient(p.tao, nullptr, NLP::objective, &p));
      checked(TaoSetEqualityConstraintsRoutine(
        p.tao, p.constraints, NLP::equality, &p));
      checked(TaoSetJacobianEqualityRoutine(
        p.tao, p.jacobian, p.jacobian, NLP::derivative, &p));
      checked(TaoSetTolerances(p.tao, controls.gradient_tolerance, 0., 0.));
      checked(
        TaoSetConstraintTolerances(p.tao, controls.constraint_tolerance, 0.));
      checked(
        TaoSetMaximumIterations(p.tao, controls.maximum_inner_iterations));
      // ALMM is first order: BQNLS applies bounds; no unused exact Hessian is
      // advertised. A positive reason is checked separately from PETSc status.
      const auto         error  = TaoSolve(p.tao);
      TaoConvergedReason reason = TAO_CONTINUE_ITERATING;
      checked(TaoGetConvergedReason(p.tao, &reason));
      PetscInt iterations = 0;
      checked(TaoGetIterationNumber(p.tao, &iterations));
      iteration.backend_reason   = reason;
      iteration.inner_iterations = iterations;
      checked(NLP::equality(p.tao, p.x, p.constraints, &p));
      PetscReal violation = 0.;
      checked(VecNorm(p.constraints, NORM_INFINITY, &violation));
      const PetscScalar *x_values = nullptr, *lower_values = nullptr,
                        *upper_values = nullptr;
      checked(VecGetArrayRead(p.x, &x_values));
      checked(VecGetArrayRead(p.lower, &lower_values));
      checked(VecGetArrayRead(p.upper, &upper_values));
      double bound_violation = 0.;
      for (std::uint64_t i = 0; i < p.local_size; ++i)
        {
          const double value = PetscRealPart(x_values[i]);
          if (!std::isfinite(value))
            bound_violation = PETSC_INFINITY;
          else
            bound_violation =
              std::max({bound_violation,
                        PetscRealPart(lower_values[i]) - value,
                        value - PetscRealPart(upper_values[i])});
        }
      checked(VecRestoreArrayRead(p.x, &x_values));
      checked(VecRestoreArrayRead(p.lower, &lower_values));
      checked(VecRestoreArrayRead(p.upper, &upper_values));
      violation =
        std::max<double>(violation, Utilities::MPI::max(bound_violation, comm));
      iteration.maximum_constraint_violation = violation;
      const PetscScalar *data                = nullptr;
      checked(VecGetArrayRead(p.x, &data));
      for (auto &[id, n] : network.nodes)
        if (n.owner == rank(comm) && !fixed.count(id))
          for (unsigned int d = 0; d < 3; ++d)
            n.position[d] =
              scale * PetscRealPart(data[numbered.at(id).index + d - p.begin]);
      checked(VecRestoreArrayRead(p.x, &data));
      PetscReal objective = 0.;
      Vec       gradient  = nullptr;
      checked(VecDuplicate(p.x, &gradient));
      checked(NLP::objective(p.tao, p.x, &objective, gradient, &p));
      checked(VecDestroy(&gradient));
      iteration.objective_W = objective * p.power_scale;
      return phase(
        validation,
        error == 0 && reason > 0 && std::isfinite(violation) &&
          violation <= controls.constraint_tolerance,
        "TAO must converge and satisfy the original auxiliary-length equalities.",
        comm);
    }
#endif
  } // namespace

  GlobalGeometryResult
  optimize_forest_geometry(const Triangulation<3>         &background,
                           const Mapping<3>               &mapping,
                           const NetworkData              &original,
                           const NetworkData              &candidate,
                           const std::map<Id, double>     &owned_design_flows,
                           const GlobalGeometryParameters &controls,
                           MPI_Comm                        comm)
  {
    GlobalGeometryResult result;
    result.network      = candidate;
    result.design_flows = owned_design_flows;
    bool valid          = true;
    try
      {
        controls.validate();
      }
    catch (const std::exception &)
      {
        valid = false;
      }
    std::vector<double> packed = {controls.minimum_length_m,
                                  controls.maximum_length_m,
                                  controls.clearance_m,
                                  double(controls.maximum_outer_iterations),
                                  double(controls.maximum_inner_iterations),
                                  double(controls.geometry_depth),
                                  double(controls.geometry_work),
                                  controls.constraint_tolerance,
                                  controls.gradient_tolerance};
    for (const auto system : {System::arterial, System::portal, System::venous})
      packed.push_back(controls.metabolic_demand_W_m3.count(system) ?
                         controls.metabolic_demand_W_m3.at(system) :
                         0.);
    for (const auto &other : Utilities::MPI::all_gather(comm, packed))
      valid = valid && other == packed;
    if (!phase(result.validation,
               valid,
               "Global geometry controls must be valid on every rank.",
               comm))
      return result;
    Directory<Node>   candidate_nodes(owned(candidate.nodes, comm), comm);
    Directory<Vessel> candidate_vessels(owned(candidate.vessels, comm), comm);
    const auto        original_owned_nodes   = owned(original.nodes, comm);
    const auto        original_owned_vessels = owned(original.vessels, comm);
    const auto        retained_nodes =
      candidate_nodes.lookup(keys(original_owned_nodes));
    const auto retained_vessels =
      candidate_vessels.lookup(keys(original_owned_vessels));
    valid = valid && retained_nodes.size() == original_owned_nodes.size() &&
            retained_vessels.size() == original_owned_vessels.size();
    Directory<Node>   original_nodes(original_owned_nodes, comm);
    Directory<Vessel> original_vessels(original_owned_vessels, comm);
    const auto        known_original_nodes =
      original_nodes.lookup(keys(original.nodes));
    const auto known_original_vessels =
      original_vessels.lookup(keys(original.vessels));
    valid = valid && known_original_nodes.size() == original.nodes.size() &&
            known_original_vessels.size() == original.vessels.size();
    const auto immutable_nodes = original_nodes.lookup(keys(candidate.nodes));
    const auto immutable_edges =
      original_vessels.lookup(keys(candidate.vessels));
    std::set<Id> frozen, frozen_edges;
    for (const auto &[id, n] : immutable_nodes)
      {
        frozen.insert(id);
        const auto &actual = candidate.nodes.at(id);
        valid              = valid && actual.position == n.position &&
                actual.port == n.port && actual.provenance == n.provenance &&
                actual.inward_port_normal == n.inward_port_normal;
      }
    for (const auto &[id, v] : immutable_edges)
      {
        frozen_edges.insert(id);
        const auto &a = candidate.vessels.at(id);
        valid         = valid && a.nodes == v.nodes && a.radii == v.radii &&
                a.viscosity == v.viscosity && a.system == v.system &&
                a.provenance == v.provenance &&
                a.physical_properties == v.physical_properties;
      }
    // Leaf identities are distributed physical nodes, not coincident tissue
    // compartments. Directory lookup communicates only requested records.
    std::map<unsigned int, std::set<Id>> leaf_send;
    for (const auto &[id, t] : candidate.terminals)
      if (t.owner == rank(comm))
        leaf_send[t.node % size(comm)].insert(t.node);
    std::set<Id> leaf_directory;
    for (const auto &[p, ids] : Utilities::MPI::some_to_some(comm, leaf_send))
      leaf_directory.insert(ids.begin(), ids.end());
    std::map<unsigned int, std::set<Id>> leaf_queries;
    for (const auto &[id, n] : candidate.nodes)
      leaf_queries[id % size(comm)].insert(id);
    std::map<unsigned int, std::set<Id>> leaf_replies;
    for (const auto &[p, ids] :
         Utilities::MPI::some_to_some(comm, leaf_queries))
      for (Id id : ids)
        if (leaf_directory.count(id))
          leaf_replies[p].insert(id);
    for (const auto &[p, ids] :
         Utilities::MPI::some_to_some(comm, leaf_replies))
      frozen.insert(ids.begin(), ids.end());
    std::map<unsigned int, std::set<Id>> outgoing_send;
    for (const auto &[id, v] : candidate.vessels)
      if (v.owner == rank(comm))
        outgoing_send[v.nodes[0] % size(comm)].insert(v.nodes[0]);
    std::set<Id> outgoing_directory;
    for (const auto &[p, ids] :
         Utilities::MPI::some_to_some(comm, outgoing_send))
      outgoing_directory.insert(ids.begin(), ids.end());
    std::map<unsigned int, std::set<Id>> final_leaf_replies;
    for (const auto &[p, ids] :
         Utilities::MPI::some_to_some(comm, leaf_queries))
      for (Id id : ids)
        if (!outgoing_directory.count(id))
          final_leaf_replies[p].insert(id);
    for (const auto &[p, ids] :
         Utilities::MPI::some_to_some(comm, final_leaf_replies))
      frozen.insert(ids.begin(), ids.end());
    for (const auto &[id, n] : candidate.nodes)
      if (n.port != Port::none)
        frozen.insert(id);
    for (const auto &[id, v] : candidate.vessels)
      if (v.owner == rank(comm) && !frozen_edges.count(id))
        valid = valid && owned_design_flows.count(id) &&
                std::isfinite(owned_design_flows.at(id)) &&
                owned_design_flows.at(id) > 0.;
    for (const auto &[id, q] : owned_design_flows)
      valid = valid && candidate.vessels.count(id) &&
              candidate.vessels.at(id).owner == rank(comm) && q > 0. &&
              std::isfinite(q);
    valid = valid && validate_data(candidate, size(comm)).valid();
    if (
      !phase(
        result.validation,
        valid,
        "Original physical data must remain unchanged; every new owned vessel needs a positive downstream SI design flow.",
        comm))
      return result;
#ifndef DEAL_II_WITH_PETSC
    phase(result.validation,
          false,
          "Global forest geometry requires deal.II with PETSc TAO ALMM.",
          comm);
    return result;
#else
    result.report.backend_available = true;
    BackgroundGeometry      geometry(background, mapping);
    std::map<Id, double>    minimum_lengths;
    std::vector<Ball>       balls;
    std::vector<Separation> separations;
    double                  local_scale = 0.;
    for (const auto &cell : background.active_cell_iterators())
      if (cell->is_locally_owned())
        local_scale = std::max(local_scale, cell->diameter());
    const double scale = Utilities::MPI::max(local_scale, comm);
    if (!phase(result.validation,
               std::isfinite(scale) && scale > 0.,
               "Global geometry needs a nonempty prepared background.",
               comm))
      return result;
    auto refresh_nodes = [&]() {
      Directory<Node> current(owned(result.network.nodes, comm), comm);
      const auto      refreshed = current.lookup(keys(result.network.nodes));
      for (const auto &[id, n] : refreshed)
        result.network.nodes.at(id) = n;
    };
    for (unsigned int outer = 0; outer < controls.maximum_outer_iterations;
         ++outer)
      {
        const auto conflicts =
          collisions(result.network, controls.clearance_m, comm);
        const auto walls = boundary_failures(result.network,
                                             frozen_edges,
                                             geometry,
                                             controls,
                                             result.validation,
                                             comm);
        if (!result.validation.valid())
          break;
        const auto conflict_count =
          Utilities::MPI::sum(std::uint64_t(conflicts.size()), comm);
        const auto wall_count =
          Utilities::MPI::sum(std::uint64_t(walls.size()), comm);
        result.report.collision_constraints += conflict_count;
        result.report.boundary_constraints += wall_count;
        if (conflict_count + wall_count > 0)
          if (!extend(result.network,
                      result.design_flows,
                      frozen_edges,
                      conflicts,
                      walls,
                      minimum_lengths,
                      balls,
                      separations,
                      controls.clearance_m,
                      result.report,
                      result.validation,
                      comm))
            break;
        GlobalGeometryIteration iteration;
        iteration.outer_iteration       = outer + 1;
        iteration.collision_constraints = conflict_count;
        iteration.boundary_constraints  = wall_count;
        const bool solved               = solve(result.network,
                                  frozen,
                                  frozen_edges,
                                  result.design_flows,
                                  minimum_lengths,
                                  balls,
                                  separations,
                                  controls,
                                  scale,
                                  iteration,
                                  result.validation,
                                  comm);
        result.report.history.push_back(iteration);
        result.report.outer_iterations = outer + 1;
        result.report.inner_iterations += iteration.inner_iterations;
        result.report.backend_reason = iteration.backend_reason;
        result.report.objective_W    = iteration.objective_W;
        result.report.maximum_constraint_violation =
          iteration.maximum_constraint_violation;
        if (!solved)
          break;
        refresh_nodes();
        const auto remaining =
          collisions(result.network, controls.clearance_m, comm);
        const auto remaining_walls = boundary_failures(result.network,
                                                       frozen_edges,
                                                       geometry,
                                                       controls,
                                                       result.validation,
                                                       comm);
        bool       physical = result.validation.valid() && remaining.empty() &&
                        remaining_walls.empty() &&
                        validate_data(result.network, size(comm)).valid();
        for (const auto &[id, v] : result.network.vessels)
          if (v.owner == rank(comm) && !frozen_edges.count(id))
            {
              const double l = length(result.network, id);
              const double minimum =
                std::max(controls.minimum_length_m,
                         minimum_lengths.count(id) ? minimum_lengths.at(id) :
                                                     0.);
              physical =
                physical &&
                l >= minimum * (1. - 10. * controls.constraint_tolerance) &&
                (controls.maximum_length_m == 0. ||
                 l <= controls.maximum_length_m *
                        (1. + 10. * controls.constraint_tolerance));
            }
        if (Utilities::MPI::min(static_cast<unsigned int>(physical), comm) == 1)
          {
            result.report.certified = true;
            break;
          }
      }
    if (!result.report.certified && result.validation.valid())
      phase(
        result.validation,
        false,
        "Joint forest geometry exhausted its outer budget without a native wall/separation/length certificate.",
        comm);
    // All candidate evaluation is on a private graph. Migration occurs only at
    // this return boundary; stored constraint rows never retain migrated IDs.
    if (Utilities::MPI::min(
          static_cast<unsigned int>(
            validate_data(result.network, size(comm)).valid()),
          comm) == 0)
      return result;
    Directory<double> flow_directory(result.design_flows, comm);
    result.network = redistribute(
      background, mapping, result.network, controls.clearance_m, comm);
    const auto frozen_returned =
      original_vessels.lookup(keys(result.network.vessels));
    std::set<Id> wanted;
    for (const auto &[id, v] : result.network.vessels)
      if (v.owner == rank(comm) && !frozen_returned.count(id))
        wanted.insert(id);
    result.design_flows = flow_directory.lookup(wanted);
    return result;
#endif
  }
} // namespace ImmersX::Vascularization
