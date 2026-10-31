// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_candidate.h"

#include <immersx/vascularization/generation.h>
namespace ImmersX::Vascularization::internal
{
  std::optional<GrowthCandidate>
  propose_growth(const GrowthCandidateContext &x,
                 const Parameters             &parameters,
                 const dealii::Point<3>       &junction,
                 const dealii::Point<3>       &terminal,
                 double                        ceiling,
                 double                        terminal_demand,
                 const CandidateGeometry      &geometry,
                 GrowthWork                   *work)
  {
    using namespace dealii;
    AssertThrow(
      !x.path.empty() && x.path.back().edge.id == x.root,
      ExcMessage(
        "A CCO candidate requires its complete physical ancestral path."));
    GrowthCandidate p;
    const auto     &selected = x.path.front();
    p.vessel                 = selected.edge.id;
    p.junction               = junction;
    p.terminal               = terminal;
    p.terminal_demand        = terminal_demand;
    count_work(work, GrowthOperation::candidate_path_visits, x.path.size() - 1);
    for (std::size_t i = 1; i < x.path.size(); ++i)
      p.ancestors.push_back(x.path[i].edge.id);
    const auto  &old = selected.edge.points;
    const double n = selected.demand, total = x.root_demand,
                 w                    = terminal_demand / x.demand_scale,
                 exponent             = 2. / x.murray_exponent,
                 dissipation_exponent = 2. - 4. / x.murray_exponent;
    if (!std::isfinite(w) || w <= 0. || !std::isfinite(total + w) ||
        total + w <= total || !std::isfinite(x.total_demand + w) ||
        x.total_demand + w <= x.total_demand)
      return {};
    const double a = old[0].distance(junction), b = junction.distance(old[1]),
                 c     = junction.distance(terminal);
    const double guard = 64. * std::numeric_limits<double>::epsilon() *
                         std::max({1., junction.norm(), terminal.norm()});
    if (std::min({a, b, c}) <= std::max(guard, parameters.minimum_length))
      return {};
    const auto sum = [&](double power, double original, bool viscous) {
      const double mu = viscous ? selected.viscosity : 1.;
      double       value =
        original + mu * (-old[0].distance(old[1]) * std::pow(n, power) +
                         a * std::pow(n + w, power) + b * std::pow(n, power) +
                         c * std::pow(w, power));
      count_work(work,
                 GrowthOperation::candidate_path_visits,
                 x.path.size() - 1);
      for (std::size_t i = 1; i < x.path.size(); ++i)
        {
          const auto &branch = x.path[i];
          value += (viscous ? branch.viscosity : 1.) *
                   branch.edge.points[0].distance(branch.edge.points[1]) *
                   (std::pow(branch.demand + w, power) -
                    std::pow(branch.demand, power));
        }
      return value;
    };
    p.volume_sum      = sum(exponent, x.volume_sum, false);
    p.dissipation_sum = parameters.dissipation_weight > 0. ?
                          sum(dissipation_exponent, x.dissipation_sum, true) :
                          0.;
    if (!std::isfinite(p.volume_sum) || p.volume_sum <= 0. ||
        (parameters.dissipation_weight > 0. &&
         (!std::isfinite(p.dissipation_sum) || p.dissipation_sum <= 0.)))
      return {};
    const auto tree_volume = [&](double moment, double demand) {
      return numbers::PI * x.root_radius * x.root_radius * moment /
             std::pow(demand, exponent);
    };
    const auto tree_power = [&](double moment, double demand) {
      return 8. / numbers::PI / std::pow(x.root_radius, 4) * moment *
             std::pow(demand, 4. / x.murray_exponent);
    };
    const double vr = (x.forest_volume - tree_volume(x.volume_sum, total) +
                       tree_volume(p.volume_sum, total + w)) /
                      x.forest_volume;
    p.cost = vr;
    if (parameters.dissipation_weight > 0.)
      {
        const double dr =
          (x.forest_power_numerator - tree_power(x.dissipation_sum, total) +
           tree_power(p.dissipation_sum, total + w)) /
          x.forest_power_numerator *
          std::pow(x.total_demand / (x.total_demand + w), 2);
        p.cost = (1. - parameters.dissipation_weight) * vr +
                 parameters.dissipation_weight * dr;
      }
    if (!std::isfinite(p.cost) || p.cost >= ceiling)
      return {};
    const auto radius = [&](double demand) {
      return x.root_radius *
             std::pow(demand / (total + w), 1. / x.murray_exponent);
    };
    const double smallest_leaf = radius(std::min(x.minimum_demand, w));
    if (!(smallest_leaf > 0.) || smallest_leaf < parameters.minimum_radius)
      return {};
    const auto &nodes = selected.edge.nodes;
    p.changed         = {{p.vessel,
                          {{nodes[0], x.next_node}},
                          {{old[0], junction}},
                          radius(n + w)},
                         {x.next_vessel,
                          {{x.next_node, nodes[1]}},
                          {{junction, old[1]}},
                          radius(n)},
                         {x.next_vessel + 1,
                          {{x.next_node, x.next_node + 1}},
                          {{junction, terminal}},
                          radius(w)}};
    count_work(work, GrowthOperation::candidate_path_visits, x.path.size() - 1);
    for (std::size_t i = 1; i < x.path.size(); ++i)
      {
        const auto  &branch   = x.path[i];
        const double proposed = radius(branch.demand + w);
        if (proposed > branch.edge.radius)
          {
            auto edge   = branch.edge;
            edge.radius = proposed;
            p.changed.push_back(edge);
          }
      }
    if (!geometry(p.changed))
      return {};
    return p;
  }

  std::optional<GrowthCandidate>
  optimize_growth(const GrowthCandidateContext &x,
                  const Parameters             &parameters,
                  const dealii::Point<3>       &terminal,
                  double                        terminal_demand,
                  const CandidateGeometry      &geometry,
                  GrowthWork                   *work)
  {
    AssertThrow(!x.path.empty(),
                dealii::ExcMessage(
                  "A CCO candidate requires a selected branch."));
    const auto &s    = x.path.front().edge.points;
    auto        best = propose_growth(x,
                               parameters,
                               .5 * (s[0] + s[1]),
                               terminal,
                               std::numeric_limits<double>::infinity(),
                               terminal_demand,
                               geometry,
                               work);
    if (!best)
      best = propose_growth(x,
                            parameters,
                            (s[0] + s[1] + terminal) / 3.,
                            terminal,
                            std::numeric_limits<double>::infinity(),
                            terminal_demand,
                            geometry,
                            work);
    if (!best)
      return {};
    double step = .25 * s[0].distance(s[1]);
    for (unsigned int iteration = 0; iteration < parameters.optimization_steps;
         ++iteration)
      {
        bool improved = false;
        for (unsigned int d = 0; d < 3; ++d)
          for (double sign : {-1., 1.})
            {
              auto trial = best->junction;
              trial[d] += sign * step;
              if (auto candidate = propose_growth(x,
                                                  parameters,
                                                  trial,
                                                  terminal,
                                                  best->cost - 1.e-14,
                                                  terminal_demand,
                                                  geometry,
                                                  work))
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
} // namespace ImmersX::Vascularization::internal
