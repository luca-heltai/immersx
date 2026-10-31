// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>

#include <immersx/vascularization/network.h>

#include <algorithm>
#include <cmath>
#include <set>

namespace ImmersX::Vascularization
{
  namespace
  {
    bool
    positive(const double value)
    {
      return std::isfinite(value) && value > 0.;
    }

    bool
    known(const System system)
    {
      return system == System::arterial || system == System::venous;
    }

    bool
    known(const Provenance provenance)
    {
      return provenance == Provenance::imported ||
             provenance == Provenance::generated;
    }

    std::array<double, 2>
    interval_radii(const Vessel &vessel, const double begin, const double end)
    {
      AssertThrow(std::isfinite(begin) && std::isfinite(end) && 0. <= begin &&
                    begin <= end && end <= 1.,
                  dealii::ExcMessage("Invalid vessel reference interval."));
      AssertThrow(positive(vessel.radii[0]) && positive(vessel.radii[1]),
                  dealii::ExcMessage(
                    "Vessel radii must be positive and finite."));
      return {{(1. - begin) * vessel.radii[0] + begin * vessel.radii[1],
               (1. - end) * vessel.radii[0] + end * vessel.radii[1]}};
    }
  } // namespace

  bool
  ValidationReport::valid() const
  {
    return issues.empty();
  }

  ValidationReport
  validate_data(const NetworkData &network,
                const unsigned int communicator_size)
  {
    AssertThrow(communicator_size > 0,
                dealii::ExcMessage("A communicator must contain a rank."));
    ValidationReport report;
    const auto       issue =
      [&](const std::string &entity, const Id id, const std::string &message) {
        report.issues.push_back({entity, id, message});
      };
    for (const auto &[id, node] : network.nodes)
      {
        if (id == invalid_id)
          issue("node", id, "Reserved identifier.");
        for (unsigned int d = 0; d < 3; ++d)
          if (!std::isfinite(node.position[d]))
            issue("node", id, "Nonfinite position.");
        for (unsigned int d = 0; d < 3; ++d)
          if (!std::isfinite(node.inward_port_normal[d]))
            issue("node", id, "Nonfinite inward port normal.");
        if (node.port == Port::none &&
            node.inward_port_normal != dealii::Point<3>())
          issue("node",
                id,
                "A surface section requires an explicit inlet or outlet role.");
        if (node.owner >= communicator_size)
          issue("node", id, "Owner is outside the communicator.");
        if (node.port != Port::none && node.port != Port::arterial_inlet &&
            node.port != Port::venous_outlet &&
            node.port != Port::arterial_outlet)
          issue("node", id, "Unknown port role.");
        if (!known(node.provenance))
          issue("node", id, "Unknown provenance.");
      }
    for (const auto &[id, vessel] : network.vessels)
      {
        if (id == invalid_id)
          issue("vessel", id, "Reserved identifier.");
        const auto a = network.nodes.find(vessel.nodes[0]);
        const auto b = network.nodes.find(vessel.nodes[1]);
        if (a == network.nodes.end() || b == network.nodes.end())
          issue("vessel", id, "Missing endpoint node or ghost.");
        else if (!positive(a->second.position.distance(b->second.position)))
          issue("vessel", id, "Length must be positive and finite.");
        if (!positive(vessel.radii[0]) || !positive(vessel.radii[1]))
          issue("vessel", id, "Radii must be positive and finite.");
        if (!positive(vessel.viscosity))
          issue("vessel", id, "Viscosity must be positive and finite.");
        if (vessel.owner >= communicator_size)
          issue("vessel", id, "Owner is outside the communicator.");
        if (!known(vessel.system))
          issue("vessel", id, "Unknown vascular system.");
        if (!known(vessel.provenance))
          issue("vessel", id, "Unknown provenance.");
        for (const auto &[name, value] : vessel.physical_properties)
          if (name.empty() || !std::isfinite(value))
            issue("vessel", id, "Invalid physical property.");
      }
    for (const auto &[id, terminal] : network.terminals)
      {
        if (id == invalid_id)
          issue("terminal", id, "Reserved identifier.");
        const auto node = network.nodes.find(terminal.node);
        if (node == network.nodes.end())
          issue("terminal", id, "Missing terminal node or ghost.");
        else if (node->second.port != Port::none)
          issue("terminal",
                id,
                "A tissue terminal cannot be an inlet or outlet port.");
        if (!positive(terminal.demand))
          issue("terminal", id, "Demand must be positive and finite.");
        if (terminal.owner >= communicator_size)
          issue("terminal", id, "Owner is outside the communicator.");
        if (!known(terminal.system))
          issue("terminal", id, "Unknown vascular system.");
      }
    return report;
  }

  ValidationReport
  validate_terminal_pairs(const NetworkData               &network,
                          const std::vector<TerminalPair> &pairs)
  {
    ValidationReport report;
    if (pairs.empty())
      return report;
    std::set<Id>                              territories, terminals;
    std::map<Id, std::vector<const Vessel *>> incident;
    for (const auto &[id, e] : network.vessels)
      {
        incident[e.nodes[0]].push_back(&e);
        incident[e.nodes[1]].push_back(&e);
      }
    for (const auto &p : pairs)
      {
        const auto issue = [&](const std::string &message) {
          report.issues.push_back({"terminal pair", p.territory, message});
        };
        if (p.territory == invalid_id ||
            !territories.insert(p.territory).second)
          issue("A supplied territory must have exactly one terminal pair.");
        if (!terminals.insert(p.arterial).second ||
            !terminals.insert(p.venous).second)
          issue("A terminal cannot belong to more than one supplied pair.");
        if (!positive(p.resistance))
          issue("Microvascular resistance must be positive and finite.");
        const auto a = network.terminals.find(p.arterial),
                   v = network.terminals.find(p.venous);
        if (a == network.terminals.end() || v == network.terminals.end())
          {
            issue("Missing terminal or terminal ghost.");
            continue;
          }
        if (a->second.system != System::arterial ||
            v->second.system != System::venous)
          issue("A pair requires an arterial terminal and a venous terminal.");
        if (a->second.territory != p.territory ||
            v->second.territory != p.territory)
          issue("Both terminals must identify the common paired territory.");
        if (a->second.demand != v->second.demand)
          issue("Paired terminals must share the same tissue demand.");
        const auto an = network.nodes.find(a->second.node),
                   vn = network.nodes.find(v->second.node);
        if (an == network.nodes.end() || vn == network.nodes.end())
          {
            issue("Missing paired endpoint node or ghost.");
            continue;
          }
        if (!positive(an->second.position.distance(vn->second.position)))
          issue("Paired terminal endpoints must be geometrically distinct.");
        for (const auto *terminal : {&a->second, &v->second})
          {
            unsigned int degree   = 0;
            bool         oriented = true;
            for (const auto *e : incident[terminal->node])
              {
                ++degree;
                oriented = oriented && e->nodes[1] == terminal->node &&
                           e->system == terminal->system;
              }
            if (degree != 1 || !oriented)
              issue(
                "Each paired terminal must end one correctly oriented vessel of its own system.");
          }
      }
    return report;
  }

  double
  length(const NetworkData &network, const Id id)
  {
    const auto  &vessel = network.vessels.at(id);
    const double result =
      network.nodes.at(vessel.nodes[0])
        .position.distance(network.nodes.at(vessel.nodes[1]).position);
    AssertThrow(positive(result),
                dealii::ExcMessage(
                  "Vessel length must be positive and finite."));
    return result;
  }

  double
  volume(const NetworkData &network,
         const Id           id,
         const double       begin,
         const double       end)
  {
    const auto r = interval_radii(network.vessels.at(id), begin, end);
    return dealii::numbers::PI * length(network, id) * (end - begin) *
           (r[0] * r[0] + r[0] * r[1] + r[1] * r[1]) / 3.;
  }

  double
  resistance(const NetworkData &network,
             const Id           id,
             const double       begin,
             const double       end)
  {
    const auto &vessel = network.vessels.at(id);
    const auto  r      = interval_radii(vessel, begin, end);
    AssertThrow(positive(vessel.viscosity),
                dealii::ExcMessage("Viscosity must be positive and finite."));
    // Factoring the difference of cubes avoids cancellation at equal radii.
    const double inverse_integral =
      (1. / (r[0] * r[0] * r[0] * r[1]) + 1. / (r[0] * r[0] * r[1] * r[1]) +
       1. / (r[0] * r[1] * r[1] * r[1])) /
      3.;
    return 8. * vessel.viscosity / dealii::numbers::PI * length(network, id) *
           (end - begin) * inverse_integral;
  }

  double
  murray_radius(const std::vector<double> &children, const double gamma)
  {
    AssertThrow(!children.empty() && positive(gamma),
                dealii::ExcMessage(
                  "Murray's law needs children and positive gamma."));
    for (const double radius : children)
      AssertThrow(positive(radius),
                  dealii::ExcMessage(
                    "Child radii must be positive and finite."));
    const double scale = *std::max_element(children.begin(), children.end());
    double       sum   = 0.;
    for (const double radius : children)
      sum += std::pow(radius / scale, gamma);
    const double result = scale * std::pow(sum, 1. / gamma);
    AssertThrow(positive(result),
                dealii::ExcMessage(
                  "Murray parent radius is not representable."));
    return result;
  }

  std::vector<VesselCell>
  subdivide(const NetworkData         &network,
            const Id                   id,
            const std::vector<double> &coordinates)
  {
    // Validate physical geometry even when the partition has only one cell.
    (void)volume(network, id);
    (void)resistance(network, id);
    AssertThrow(coordinates.size() >= 2 && coordinates.front() == 0. &&
                  coordinates.back() == 1.,
                dealii::ExcMessage(
                  "A vessel partition must start at 0 and end at 1."));
    std::vector<VesselCell> cells;
    cells.reserve(coordinates.size() - 1);
    for (std::size_t i = 1; i < coordinates.size(); ++i)
      {
        AssertThrow(std::isfinite(coordinates[i]) &&
                      coordinates[i] > coordinates[i - 1] &&
                      coordinates[i] <= 1.,
                    dealii::ExcMessage(
                      "Vessel partition coordinates must increase."));
        cells.push_back({id, coordinates[i - 1], coordinates[i]});
      }
    return cells;
  }
} // namespace ImmersX::Vascularization
