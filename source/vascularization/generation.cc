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

#include "forest_state.h"
#include "growth_batch.h"
#include "growth_geometry.h"

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
    add_parameter(
      "Seed inward port normal",
      seed_inward_port_normal,
      "Zero retains a closed envelope; a nonzero inward normal declares a planar inlet section on the boundary.");
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
      "Zero minimizes generated-vessel volume; otherwise blend generated volume and dissipation under total new-tissue reference flow, each normalized by its current forest value.");
    leave_subsection();
    enter_subsection("Physiology");
    add_parameter(
      "Tissue demand",
      tissue_demand,
      "Positive finite relative demand at new territory positions; independent of sampling density.");
    add_parameter(
      "Total new-tissue reference flow",
      reference_flow,
      "m^3/s through newly generated terminals, split by tissue demand; required for a nonzero dissipation weight.");
    add_parameter("Arterial Murray exponent", arterial_murray_exponent);
    add_parameter(
      "Reference pressures",
      reference_conditions.pressures,
      "Physical Node id: pressure in Pa; declare every physical port used by the paired reference solve.");
    add_parameter(
      "Unpaired reference inflows",
      reference_conditions.inflows,
      "Physical Node id: external m^3/s at unpaired input terminals; never prescribe pressure and flow at the same node.");
    add_parameter("Minimum radius",
                  minimum_radius,
                  "Smallest admissible generated radius in metres.");
    add_parameter("Dynamic viscosity",
                  viscosity,
                  "Dynamic viscosity in Pa s; must be supplied.");
    leave_subsection();
    enter_subsection("Venous network");
    add_parameter("Enabled", venous.enabled);
    add_parameter(
      "Selected terminal ids",
      venous.selected_terminals,
      "Input venous tissue terminal ids to extend; empty selects available interior venous endpoints.");
    add_parameter("Seed position", venous.seed_position);
    add_parameter("Seed inward port normal", venous.seed_inward_port_normal);
    add_parameter("Seed radius",
                  venous.seed_radius,
                  "Metres; required when no venous attachment is selected.");
    add_parameter("Murray exponent", venous.murray_exponent);
    add_parameter(
      "Terminal separation",
      venous.terminal_separation,
      "Positive metres from each arterial terminal to its venous partner.");
    add_parameter(
      "Microvascular resistance",
      venous.microvascular_resistance,
      "Positive Pa s/m^3 per bed, or zero to derive each bed from the explicit reference pressures and demand flow.");
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
    if (venous.enabled)
      {
        reference_parameters.validate();
        AssertThrow(
          positive(venous.murray_exponent) &&
            positive(venous.terminal_separation) &&
            std::isfinite(venous.seed_radius) && venous.seed_radius >= 0. &&
            std::isfinite(venous.microvascular_resistance) &&
            venous.microvascular_resistance >= 0. &&
            (venous.microvascular_resistance > 0. || positive(reference_flow)),
          dealii::ExcMessage(
            "Venous growth requires a positive Murray exponent and terminal separation, and either a positive bed resistance or a positive reference flow to derive it."));
        for (unsigned int d = 0; d < 3; ++d)
          AssertThrow(
            std::isfinite(venous.seed_position[d]) &&
              std::isfinite(venous.seed_inward_port_normal[d]),
            dealii::ExcMessage(
              "Venous seed coordinates and directions must be finite."));
      }
    AssertThrow(
      std::isfinite(reference_flow) && reference_flow >= 0. &&
        (dissipation_weight == 0. || reference_flow > 0.),
      dealii::ExcMessage(
        "A blended dissipation objective requires an explicit positive total new-tissue reference flow in m^3/s."));
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
      AssertThrow(std::isfinite(seed_position[d]) &&
                    std::isfinite(seed_direction[d]) &&
                    std::isfinite(seed_inward_port_normal[d]),
                  dealii::ExcMessage(
                    "Seed coordinates and directions must be finite."));
  }

  namespace
  {
    using namespace dealii;
    using namespace internal;
    namespace bgi = boost::geometry::index;

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

    // r_e = R*(D_e/D_root)^(1/gamma), with subtree demand D_e. The physical
    // total tissue flow is split globally by demand, so power is proportional
    // to sum_roots(mu/R^4 * D_root^(4/gamma) * sum_e(L_e*D_e^(2-4/gamma)))
    // divided by total demand squared. The common total-flow squared factor
    // cancels in the normalized objective. Counts remain separate for budgets.
    // Trial demand changes touch one ancestral path; no candidate copies the
    // complete network or materializes every normalized radius.
    struct CCOGrowth
    {
      using Leaf = std::pair<Segment<3>, Id>;
      using Edge = internal::Edge;
      struct Tree
      {
        double radius, viscosity;
        double volume_sum, dissipation_sum;
        double minimum_demand;
      };
      struct Proposal
      {
        Id                vessel = invalid_id;
        Point<3>          junction, terminal;
        std::vector<Id>   ancestors;
        double            volume_sum = 0., dissipation_sum = 0.;
        double            cost = std::numeric_limits<double>::infinity();
        double            terminal_demand = 1.;
        std::vector<Edge> changed;
      };

      CCOGrowth(GenerationResult        &result,
                const Parameters        &parameters,
                BackgroundGeometry      &geometry,
                const std::vector<Id>   &roots,
                unsigned int             terminal_count,
                const FunctionParser<3> &demand_function,
                System                   active_system)
        : result(result)
        , network(result.network)
        , parameters(parameters)
        , terminal_count(terminal_count)
        , geometry(geometry)
        , demand_function(demand_function)
        , system(active_system)
        , random(parameters.random_seed)
        , murray_exponent(active_system == System::arterial ?
                            parameters.arterial_murray_exponent :
                            parameters.venous.murray_exponent)
        , exponent(2. / murray_exponent)
        , dissipation_exponent(2. - 4. / murray_exponent)
      {
        std::map<Id, double> leaf_demands;
        for (const auto &[id, t] : network.terminals)
          leaf_demands[t.node] = t.demand;
        for (Id root : roots)
          demand_scale =
            std::max(demand_scale,
                     leaf_demands.at(network.vessels.at(root).nodes[1]));
        for (Id root : roots)
          {
            const auto  &e = network.vessels.at(root);
            const double d = leaf_demands.at(e.nodes[1]) / demand_scale;
            AssertThrow(
              std::isfinite(d) && d > 0.,
              ExcMessage(
                "Initial relative tissue demand is not representable."));
            counts[root]  = 1;
            demands[root] = d;
            total_demand += d;
            parents[root] = invalid_id;
            tree_of[root] = root;
            trees[root]   = {e.radii[0],
                             e.viscosity,
                             length(network, root) * std::pow(d, exponent),
                           parameters.dissipation_weight > 0. ?
                               length(network, root) *
                               std::pow(d, dissipation_exponent) :
                               0.,
                             d};
            forest_volume +=
              tree_volume(trees.at(root), trees.at(root).volume_sum, d);
            forest_power_numerator +=
              tree_power_numerator(trees.at(root),
                                   trees.at(root).dissipation_sum,
                                   d);
          }
        AssertThrow(std::isfinite(forest_volume) && forest_volume > 0. &&
                      (parameters.dissipation_weight == 0. ||
                       (std::isfinite(forest_power_numerator) &&
                        forest_power_numerator > 0.)),
                    ExcMessage(
                      "Initial vascular objective is not representable."));
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
        active_terminal_count = roots.size();
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
      tree_volume(const Tree &tree, double sum, double root_demand) const
      {
        return numbers::PI * tree.radius * tree.radius * sum /
               std::pow(root_demand, exponent);
      }

      double
      tree_power_numerator(const Tree &tree,
                           double      sum,
                           double      root_demand) const
      {
        if (parameters.dissipation_weight == 0.)
          return 0.;
        // Numerator of generated-forest power divided by the common Q^2.
        return 8. * tree.viscosity / numbers::PI / std::pow(tree.radius, 4) *
               sum * std::pow(root_demand, 4. / murray_exponent);
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
               std::pow(demands.at(id) / demands.at(root),
                        1. / murray_exponent);
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
                               1. / murray_exponent) >=
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
        return internal::compatible(a,
                                    b,
                                    parameters.minimum_angle,
                                    parameters.clearance);
      }

      std::optional<Proposal>
      propose(Id              id,
              const Point<3> &junction,
              const Point<3> &terminal,
              double          ceiling,
              double          terminal_demand) const
      {
        Proposal p;
        p.vessel          = id;
        p.junction        = junction;
        p.terminal        = terminal;
        p.terminal_demand = terminal_demand;
        for (Id ancestor = parents.at(id); ancestor != invalid_id;
             ancestor    = parents.at(ancestor))
          p.ancestors.push_back(ancestor);
        const auto   old  = segment(id);
        const Id     root = tree_of.at(id);
        const auto  &tree = trees.at(root);
        const double n = demands.at(id), total = demands.at(root),
                     w = terminal_demand / demand_scale;
        if (!std::isfinite(w) || w <= 0. || !std::isfinite(total + w) ||
            total + w <= total || !std::isfinite(total_demand + w) ||
            total_demand + w <= total_demand)
          return {};
        const double a     = old.first.distance(junction),
                     b     = junction.distance(old.second),
                     c     = junction.distance(terminal);
        const double guard = 64. * std::numeric_limits<double>::epsilon() *
                             std::max({1., junction.norm(), terminal.norm()});
        if (std::min({a, b, c}) <= std::max(guard, parameters.minimum_length))
          return {};
        const auto sum = [&](double power, double original) {
          double value = original - length(network, id) * std::pow(n, power) +
                         a * std::pow(n + w, power) + b * std::pow(n, power) +
                         c * std::pow(w, power);
          for (Id ancestor : p.ancestors)
            value += length(network, ancestor) *
                     (std::pow(demands.at(ancestor) + w, power) -
                      std::pow(demands.at(ancestor), power));
          return value;
        };
        p.volume_sum      = sum(exponent, tree.volume_sum);
        p.dissipation_sum = parameters.dissipation_weight > 0. ?
                              sum(dissipation_exponent, tree.dissipation_sum) :
                              0.;
        if (!std::isfinite(p.volume_sum) || p.volume_sum <= 0. ||
            (parameters.dissipation_weight > 0. &&
             (!std::isfinite(p.dissipation_sum) || p.dissipation_sum <= 0.)))
          return {};
        const double vr =
          (forest_volume - tree_volume(tree, tree.volume_sum, total) +
           tree_volume(tree, p.volume_sum, total + w)) /
          forest_volume;
        p.cost = vr;
        if (parameters.dissipation_weight > 0.)
          {
            const double dr =
              (forest_power_numerator -
               tree_power_numerator(tree, tree.dissipation_sum, total) +
               tree_power_numerator(tree, p.dissipation_sum, total + w)) /
              forest_power_numerator *
              std::pow(total_demand / (total_demand + w), 2);
            p.cost = (1. - parameters.dissipation_weight) * vr +
                     parameters.dissipation_weight * dr;
          }
        if (!std::isfinite(p.cost) || p.cost >= ceiling)
          return {};
        std::set<Id> increased(p.ancestors.begin(), p.ancestors.end());
        increased.insert(id);
        const auto proposed_radius = [&](Id e) {
          const auto found = tree_of.find(e);
          if (found == tree_of.end() || found->second != root)
            return radius(e);
          const double demand = demands.at(e) + (increased.count(e) ? w : 0.);
          return tree.radius *
                 std::pow(demand / (total + w), 1. / murray_exponent);
        };
        const double leaf_radius =
          tree.radius * std::pow(w / (total + w), 1. / murray_exponent);
        const double smallest_leaf =
          tree.radius * std::pow(std::min(tree.minimum_demand, w) / (total + w),
                                 1. / murray_exponent);
        if (!(smallest_leaf > 0.) || smallest_leaf < parameters.minimum_radius)
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
           tree.radius * std::pow(n / (total + w), 1. / murray_exponent)},
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
            const auto             &e = changed[i];
            std::array<Point<3>, 2> port_normals;
            for (unsigned int end = 0; end < 2; ++end)
              if (const auto node = network.nodes.find(e.nodes[end]);
                  node != network.nodes.end())
                port_normals[end] = node->second.inward_port_normal;
            if (!geometry.contains({e.points[0], e.points[1]},
                                   e.radius + parameters.clearance,
                                   parameters.geometry_depth,
                                   parameters.geometry_work,
                                   port_normals))
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
        p.changed = std::move(changed);
        return p;
      }

      std::optional<Proposal>
      optimize(Id id, const Point<3> &terminal, double terminal_demand) const
      {
        const auto s    = segment(id);
        auto       best = propose(id,
                            .5 * (s.first + s.second),
                            terminal,
                            std::numeric_limits<double>::infinity(),
                            terminal_demand);
        if (!best)
          best = propose(id,
                         (s.first + s.second + terminal) / 3.,
                         terminal,
                         std::numeric_limits<double>::infinity(),
                         terminal_demand);
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
                  if (auto candidate = propose(id,
                                               trial,
                                               terminal,
                                               best->cost - 1.e-14,
                                               terminal_demand))
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
      accept(const Proposal &p, const Territory *shared_territory = nullptr)
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
        const Id territory                    = shared_territory ?
                                                  shared_territory->id :
                                                  take_identifiers(next_territory);
        network.terminals[tid]                = {
          next_node + 1, territory, system, 0, p.terminal_demand};
        if (!shared_territory)
          result.territories.push_back(
            {territory, p.terminal, 0, p.terminal_demand});
        ++active_terminal_count;
        for (auto &[child, parent] : parents)
          if (parent == p.vessel)
            parent = next_vessel;
        const Id root = tree_of.at(p.vessel);
        auto    &tree = trees.at(root);
        forest_volume -= tree_volume(tree, tree.volume_sum, demands.at(root));
        forest_power_numerator -=
          tree_power_numerator(tree, tree.dissipation_sum, demands.at(root));
        tree_of[next_vessel] = tree_of[next_vessel + 1] = root;
        counts[next_vessel]                             = counts.at(p.vessel);
        counts[next_vessel + 1]                         = 1;
        parents[next_vessel] = parents[next_vessel + 1] = p.vessel;
        const double w           = p.terminal_demand / demand_scale;
        demands[next_vessel]     = demands.at(p.vessel);
        demands[next_vessel + 1] = w;
        demands.at(p.vessel) += w;
        total_demand += w;
        tree.minimum_demand = std::min(tree.minimum_demand, w);
        ++counts.at(p.vessel);
        for (Id ancestor : p.ancestors)
          {
            ++counts.at(ancestor);
            demands.at(ancestor) += w;
          }
        tree.volume_sum      = p.volume_sum;
        tree.dissipation_sum = p.dissipation_sum;
        forest_volume += tree_volume(tree, tree.volume_sum, demands.at(root));
        forest_power_numerator +=
          tree_power_numerator(tree, tree.dissipation_sum, demands.at(root));
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
                           murray_exponent) *
                  (1. + 32. * std::numeric_limits<double>::epsilon())));
        if (capacity < terminal_count)
          {
            result.validation.issues.push_back(
              {"growth",
               invalid_id,
               "Requested terminal count is incompatible with the minimum radius at the fixed attachment radii."});
            return;
          }
        std::map<Id, Territory> pending;
        if (system == System::venous)
          {
            for (const auto &t : result.territories)
              pending.emplace(t.id, t);
            for (const auto &[id, t] : network.terminals)
              if (t.system == system)
                pending.erase(t.territory);
          }
        while (active_terminal_count < terminal_count)
          {
            AssertThrow(next_node < invalid_id - 1 &&
                          next_vessel < invalid_id - 1,
                        ExcMessage("Vascular identifier space exhausted."));
            bool inserted = false;
            for (unsigned int attempt = 0; attempt < parameters.sample_budget;
                 ++attempt)
              {
                ++result.sampled_points;
                Point<3>         target;
                const Territory *shared          = nullptr;
                double           terminal_demand = 0.;
                if (system == System::venous)
                  {
                    AssertThrow(!pending.empty(), ExcInternalError());
                    auto entry = pending.begin();
                    std::advance(entry, (attempt / 6) % pending.size());
                    shared                       = &entry->second;
                    target                       = shared->position;
                    const unsigned int direction = (attempt + shared->id) % 6;
                    target[direction / 2] +=
                      (direction % 2 ? 1. : -1.) *
                      parameters.venous.terminal_separation;
                    terminal_demand = shared->demand;
                    if (!geometry.containing(target))
                      continue;
                  }
                else if (system == System::arterial)
                  {
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
                    if (uniform() >=
                        w / parameters.terminal_weight_bound *
                          std::pow(scale / h, parameters.density_beta))
                      continue;
                    terminal_demand = demand_function.value(target);
                    AssertThrow(
                      std::isfinite(terminal_demand) && terminal_demand > 0.,
                      ExcMessage(
                        "Tissue demand must be positive and finite at every proposed terminal."));
                  }
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
                  if (auto p = optimize(id, target, terminal_demand);
                      p && (!best || p->cost < best->cost))
                    best = std::move(p);
                if (best)
                  {
                    BatchProposal candidate;
                    candidate.id      = 0;
                    candidate.cost    = best->cost;
                    candidate.changed = best->changed;
                    for (const auto &e : candidate.changed)
                      {
                        if (network.vessels.count(e.id))
                          {
                            candidate.owners[e.id] =
                              network.vessels.at(e.id).owner;
                            candidate.old_nodes[e.id] =
                              network.vessels.at(e.id).nodes;
                          }
                        for (Id node : e.nodes)
                          if (network.nodes.count(node) &&
                              std::max({std::abs(network.nodes.at(node)
                                                   .inward_port_normal[0]),
                                        std::abs(network.nodes.at(node)
                                                   .inward_port_normal[1]),
                                        std::abs(network.nodes.at(node)
                                                   .inward_port_normal[2])}) >
                                0.)
                            {
                              candidate.normals[node] =
                                network.nodes.at(node).inward_port_normal;
                              candidate.port_owners[node] =
                                network.nodes.at(node).owner;
                            }
                      }
                    const Id     root = tree_of.at(best->vessel);
                    const double shrink =
                      std::pow(demands.at(root) /
                                 (demands.at(root) +
                                  best->terminal_demand / demand_scale),
                               1. / murray_exponent);
                    const auto selected =
                      select_growth_batch(geometry,
                                          network,
                                          {candidate},
                                          parameters,
                                          MPI_COMM_SELF,
                                          [&](Id id) {
                                            const auto tree = tree_of.find(id);
                                            return radius(id) *
                                                   (tree != tree_of.end() &&
                                                        tree->second == root ?
                                                      shrink :
                                                      1.);
                                          });
                    if (!selected.local.at(0))
                      continue;
                    const Id shared_id = shared ? shared->id : invalid_id;
                    accept(*best, shared);
                    if (shared)
                      pending.erase(shared_id);
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
        if (result.validation.valid())
          {
            std::set<Id>         mutable_owned;
            std::map<Id, double> fixed_roots;
            for (const auto &[id, root] : tree_of)
              mutable_owned.insert(id);
            for (const auto &[id, tree] : trees)
              fixed_roots[id] = tree.radius;
            const auto state = evaluate_forest(
              network, mutable_owned, fixed_roots, parameters, MPI_COMM_SELF);
            result.validation.issues.insert(result.validation.issues.end(),
                                            state.validation.issues.begin(),
                                            state.validation.issues.end());
            if (state.validation.valid())
              {
                for (const auto &[id, b] : state.owned)
                  network.vessels.at(id).radii = {{b.radius, b.radius}};
                result.target_terminal_flows.insert(
                  state.terminal_flows.begin(), state.terminal_flows.end());
                if (parameters.reference_flow > 0.)
                  for (const auto &[id, r] : fixed_roots)
                    result.target_attachment_flows[id] =
                      state.owned.at(id).flow;
              }
          }
        else
          for (const auto &[id, root] : tree_of)
            network.vessels.at(id).radii = {{radius(id), radius(id)}};
      }

      GenerationResult        &result;
      NetworkData             &network;
      const Parameters        &parameters;
      const unsigned int       terminal_count;
      BackgroundGeometry      &geometry;
      const FunctionParser<3> &demand_function;
      const System             system;
      unsigned int             active_terminal_count = 0;
      std::mt19937_64          random;
      FunctionParser<3>        weight;
      Point<3>                 lower, upper;
      double minimum_h = std::numeric_limits<double>::infinity(),
             maximum_h = 0.;
      const double murray_exponent, exponent, dissipation_exponent;
      double       forest_volume = 0., forest_power_numerator = 0.,
             maximum_radius = 0.;
      Id next_node = 0, next_vessel = 0, next_terminal = 0, next_territory = 0;
      std::map<Id, Tree>         trees;
      std::map<Id, Id>           tree_of;
      std::map<Id, unsigned int> counts;
      std::map<Id, double>       demands;
      double                     demand_scale = 0., total_demand = 0.;
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
    FunctionParser<3> demand_function;
    demand_function.initialize("x,y,z", parameters.tissue_demand, {});
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
    const auto      append_root = [&](Id               start,
                                 const Vessel    &properties,
                                 Id               input_terminal,
                                 bool             new_port,
                                 const Node      &port,
                                 const Point<3>  &seed_direction,
                                 const Territory *shared = nullptr,
                                 bool             diagnose = true) -> bool {
      const auto reject = [&](const std::string &message) {
        if (diagnose)
          fail(input_terminal, message);
        return false;
      };
      const auto seed_normal = normalized_port_normal(port.inward_port_normal);
      const Point<3> position =
        new_port ? port.position : result.network.nodes.at(start).position;
      const auto cell = geometry.containing(
        position, new_port && seed_normal.norm_square() > 0. ? 1.e-12 : 0.);
      if (!cell)
        {
          return reject("Seed or attachment is outside the background.");
        }
      if (new_port && seed_normal.norm_square() > 0. &&
          !geometry.planar_section(position, seed_normal))
        {
          return reject(
            "The declared planar port does not match a planar background boundary section.");
        }
      const double radius = properties.radii[1];
      if (radius < parameters.minimum_radius)
        {
          return reject(
            "Fixed attachment radius is below the minimum admissible new radius.");
        }
      std::vector<Point<3>> directions;
      if (shared)
        for (unsigned int i = 0; i < 6; ++i)
          {
            auto               end  = shared->position;
            const unsigned int axis = (i + shared->id) % 6;
            end[axis / 2] +=
              (axis % 2 ? 1. : -1.) * parameters.venous.terminal_separation;
            directions.emplace_back(end - position);
          }
      else if (seed_direction.norm() > 0.)
        directions.push_back(seed_direction);
      else
        {
          if (!new_port)
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
      const bool allocate_port = new_port && !result.network.nodes.count(start);
      AssertThrow(next_node < invalid_id - (allocate_port ? 1 : 0) &&
                    next_vessel < invalid_id,
                  ExcMessage("Vascular identifier space exhausted."));
      const Id end_node = next_node + (allocate_port ? 1 : 0);
      for (const auto &direction : directions)
        {
          const double norm = direction.norm();
          if (!(norm > 0.) || !std::isfinite(norm))
            continue;
          const Point<3> end =
            shared ? position + direction :
                          position + parameters.seed_length / norm * direction;
          if (position.distance(end) <= parameters.minimum_length)
            continue;
          if (!geometry.contains({position, end},
                                 radius + parameters.clearance,
                                 parameters.geometry_depth,
                                 parameters.geometry_work,
                                      {{new_port ? seed_normal : Point<3>(),
                                   Point<3>()}}))
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
          const double terminal_demand =
            shared ? shared->demand : demand_function.value(end);
          AssertThrow(
            std::isfinite(terminal_demand) && terminal_demand > 0.,
            ExcMessage(
              "Tissue demand must be positive and finite at every initial terminal."));
          if (new_port)
            result.network.nodes[start] = {
              position, 0, port.port, Provenance::generated, seed_normal};
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
                   territory =
                     shared ? shared->id : take_identifiers(next_territory);
          result.network.terminals[tid] = {
            end_node, territory, properties.system, 0, terminal_demand};
          if (!shared)
            result.territories.push_back({territory, end, 0, terminal_demand});
          take_identifiers(next_node, allocate_port ? 2 : 1);
          if (input_terminal != invalid_id)
            {
              result.attachment_roots[input_terminal] = vid;
              result.network.terminals.erase(input_terminal);
            }
          roots.push_back(vid);
          index.insert({{position, end}, vid});
          maximum_radius = std::max(maximum_radius, radius);
          return true;
        }
      return reject(
        "No admissible initial subtree direction was found within the deterministic geometry budget.");
    };
    if (creating_seed)
      {
        Vessel properties;
        properties.radii = {{parameters.seed_radius, parameters.seed_radius}};
        properties.viscosity = parameters.viscosity;
        Node port;
        port.position           = parameters.seed_position;
        port.port               = Port::arterial_inlet;
        port.inward_port_normal = parameters.seed_inward_port_normal;
        append_root(
          0, properties, invalid_id, true, port, parameters.seed_direction);
      }
    else
      for (Id id : selected)
        {
          const auto &terminal = result.network.terminals.at(id);
          append_root(terminal.node,
                      initial.vessels.at(incoming[terminal.node][0]),
                      id,
                      false,
                      {},
                      parameters.seed_direction);
          if (!result.validation.valid())
            break;
        }
    if (!result.validation.valid())
      return result;
    Id reserved_venous_port = invalid_id;
    if (parameters.venous.enabled &&
        parameters.venous.selected_terminals.empty())
      {
        bool has_venous_attachment = false;
        for (const auto &[id, t] : result.network.terminals)
          if (t.system == System::venous &&
              geometry.containing(result.network.nodes.at(t.node).position))
            has_venous_attachment = true;
        if (!has_venous_attachment)
          {
            // Reserve the port before CCO insertions so its id is independent
            // of terminal-count calibration and the number of accepted trials.
            reserved_venous_port = next_identifier(result.network.nodes);
            result.network.nodes[reserved_venous_port] = {
              parameters.venous.seed_position,
              0,
              Port::venous_outlet,
              Provenance::generated,
              normalized_port_normal(
                parameters.venous.seed_inward_port_normal)};
          }
      }
    CCOGrowth growth(result,
                     parameters,
                     geometry,
                     roots,
                     terminal_count,
                     demand_function,
                     System::arterial);
    growth.run();
    if (parameters.venous.enabled && result.validation.valid())
      {
        // Arterial geometry remains an obstacle throughout venous growth.
        obstacles.clear();
        maximum_radius = parameters.venous.seed_radius;
        for (const auto &[id, e] : result.network.vessels)
          {
            obstacles.emplace_back(
              Segment<3>(result.network.nodes.at(e.nodes[0]).position,
                         result.network.nodes.at(e.nodes[1]).position),
              id);
            maximum_radius = std::max({maximum_radius, e.radii[0], e.radii[1]});
          }
        index = pack_rtree(obstacles);
        roots.clear();
        next_node     = next_identifier(result.network.nodes);
        next_vessel   = next_identifier(result.network.vessels);
        next_terminal = next_identifier(result.network.terminals);
        std::vector<Id> venous_selected = parameters.venous.selected_terminals;
        if (venous_selected.empty())
          for (const auto &[id, t] : result.network.terminals)
            if (t.system == System::venous &&
                geometry.containing(result.network.nodes.at(t.node).position))
              venous_selected.push_back(id);
        std::set<Id> seen;
        for (Id id : venous_selected)
          {
            const auto t = result.network.terminals.find(id);
            if (!seen.insert(id).second ||
                t == result.network.terminals.end() ||
                t->second.system != System::venous ||
                !geometry.containing(
                  result.network.nodes.at(t->second.node).position))
              fail(
                id,
                "Selected venous terminals must exist, be unique interior venous leaves, and belong to the input network.");
          }
        if (venous_selected.size() > terminal_count)
          fail(
            invalid_id,
            "Territory count is smaller than the number of selected venous attachments.");
        const bool create_venous_port = venous_selected.empty();
        if (create_venous_port && (!(parameters.venous.seed_radius > 0.) ||
                                   !(parameters.viscosity > 0.)))
          fail(
            invalid_id,
            "A new venous outlet requires an explicit positive radius and dynamic viscosity.");
        if (!result.validation.valid())
          return result;
        std::map<Id, Territory> unassigned;
        for (const auto &t : result.territories)
          unassigned.emplace(t.id, t);
        const unsigned int root_count =
          create_venous_port ? 1 : venous_selected.size();
        for (unsigned int i = 0; i < root_count; ++i)
          {
            const Id input_terminal =
              create_venous_port ? invalid_id : venous_selected[i];
            const Id start = create_venous_port ?
                               reserved_venous_port :
                               result.network.terminals.at(input_terminal).node;
            Node     port;
            port.position           = parameters.venous.seed_position;
            port.port               = Port::venous_outlet;
            port.inward_port_normal = parameters.venous.seed_inward_port_normal;
            Vessel properties;
            if (create_venous_port)
              {
                properties.system    = System::venous;
                properties.viscosity = parameters.viscosity;
                properties.radii     = {{parameters.venous.seed_radius,
                                         parameters.venous.seed_radius}};
              }
            else
              properties = initial.vessels.at(incoming.at(start).at(0));
            const Point<3>         position = create_venous_port ?
                                                port.position :
                                                result.network.nodes.at(start).position;
            std::vector<Territory> ordered;
            for (const auto &[id, t] : unassigned)
              ordered.push_back(t);
            std::sort(ordered.begin(),
                      ordered.end(),
                      [&](const auto &a, const auto &b) {
                        const auto da = position.distance_square(a.position),
                                   db = position.distance_square(b.position);
                        return da < db || (da == db && a.id < b.id);
                      });
            bool attached = false;
            for (const auto &t : ordered)
              if (append_root(start,
                              properties,
                              input_terminal,
                              create_venous_port,
                              port,
                              {},
                              &t,
                              false))
                {
                  unassigned.erase(t.id);
                  attached = true;
                  break;
                }
            if (!attached)
              {
                fail(
                  input_terminal,
                  "No admissible connection from the venous outlet or attachment to a common tissue territory was found.");
                return result;
              }
          }
        CCOGrowth venous_growth(result,
                                parameters,
                                geometry,
                                roots,
                                terminal_count,
                                demand_function,
                                System::venous);
        venous_growth.run();
      }
    if (parameters.venous.enabled && result.validation.valid())
      {
        std::map<Id, std::array<Id, 2>> paired;
        for (const auto &t : result.territories)
          paired[t.id] = {{invalid_id, invalid_id}};
        std::vector<Id> selected_tissue;
        for (const auto &[id, t] : result.network.terminals)
          if (paired.count(t.territory))
            {
              unsigned int slot = 0;
              if (t.system == System::arterial)
                slot = 0;
              else if (t.system == System::venous)
                slot = 1;
              if (paired.at(t.territory)[slot] != invalid_id)
                fail(
                  id,
                  "A common territory has more than one terminal of the same system.");
              paired.at(t.territory)[slot] = id;
              selected_tissue.push_back(id);
            }
        for (const auto &[id, terminals] : paired)
          if (terminals[0] == invalid_id || terminals[1] == invalid_id)
            fail(
              id,
              "Every new territory requires exactly one arterial and one venous terminal.");
        if (!result.validation.valid())
          return result;
        ReferenceResult design;
        if (parameters.venous.microvascular_resistance == 0.)
          {
            std::map<Id, double> port_pressures;
            for (const auto &[id, pressure] :
                 parameters.reference_conditions.pressures)
              if (result.network.nodes.count(id) &&
                  result.network.nodes.at(id).port != Port::none)
                port_pressures[id] = pressure;
            auto       conditions = parameters.reference_conditions;
            const auto demand_bc =
              demand_flow_conditions(result.network,
                                     port_pressures,
                                     parameters.reference_flow,
                                     selected_tissue,
                                     communicator);
            for (const auto &[id, flow] : demand_bc.inflows)
              if (!conditions.inflows.emplace(id, flow).second)
                fail(
                  id,
                  "A paired tissue terminal cannot also have an independent external load.");
            if (!result.validation.valid())
              return result;
            design = solve_reference(result.network,
                                     {},
                                     conditions,
                                     parameters.reference_parameters,
                                     communicator);
            result.validation.issues.insert(result.validation.issues.end(),
                                            design.validation.issues.begin(),
                                            design.validation.issues.end());
            if (!result.validation.valid())
              return result;
          }
        for (const auto &[id, terminals] : paired)
          {
            double bed_resistance = parameters.venous.microvascular_resistance;
            if (bed_resistance == 0.)
              {
                const auto a = result.network.terminals.at(terminals[0]).node;
                const auto v = result.network.terminals.at(terminals[1]).node;
                bed_resistance =
                  (design.pressures.at(a) - design.pressures.at(v)) /
                  result.target_terminal_flows.at(terminals[0]);
              }
            if (!std::isfinite(bed_resistance) || bed_resistance <= 0.)
              fail(
                id,
                "The requested reference flow and pressures leave no positive microvascular resistance.");
            result.terminal_pairs.push_back(
              {id, terminals[0], terminals[1], bed_resistance});
          }
        if (!result.validation.valid())
          return result;
        result.reference = solve_reference(result.network,
                                           result.terminal_pairs,
                                           parameters.reference_conditions,
                                           parameters.reference_parameters,
                                           communicator);
        result.validation.issues.insert(
          result.validation.issues.end(),
          result.reference.validation.issues.begin(),
          result.reference.validation.issues.end());
        if (result.reference.validation.valid() &&
            parameters.reference_flow > 0.)
          for (const auto &[id, desired] : result.target_terminal_flows)
            {
              // Compare the requested supply to the actual 0D territory flow.
              // Terminal Kirchhoff defects have their separate global-flow
              // tolerance.
              const Id territory = result.network.terminals.at(id).territory;
              const double actual =
                result.reference.territories.at(territory).flow;
              if (std::abs(actual - desired) >
                  parameters.reference_parameters.absolute_flow_tolerance +
                    parameters.reference_parameters
                        .conservation_relative_tolerance *
                      desired)
                fail(
                  id,
                  "The supplied bed resistances, pressures and reference flow do not realize the requested tissue-demand split.");
            }
      }
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
                       p.minimum_coverage -
                         (parameters.venous.enabled ?
                            candidate.tissue.dual_coverage :
                            candidate.tissue.arterial_coverage));
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
