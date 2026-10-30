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
#include "growth_candidate.h"
#include "growth_commit.h"
#include "growth_geometry.h"
#include "growth_loop.h"
#include "growth_start.h"

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
    const auto size        = Utilities::MPI::n_mpi_processes(communicator);
    const bool local_empty = initial.nodes.empty() && initial.vessels.empty() &&
                             initial.terminals.empty();
    const bool creating_seed =
      Utilities::MPI::min(static_cast<unsigned int>(local_empty),
                          communicator) == 1;
    const bool any_venous =
      Utilities::MPI::max(static_cast<unsigned int>(parameters.venous.enabled),
                          communicator) > 0;
    AssertThrow(size == 1 || !any_venous,
                ExcMessage(
                  "Distributed venous startup still requires one rank."));
    bool valid_controls = true;
    try
      {
        parameters.validate(creating_seed);
      }
    catch (const std::exception &)
      {
        if (size == 1)
          throw;
        valid_controls = false;
      }
    AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid_controls),
                                    communicator) == 1,
                ExcMessage("Invalid vascular generation controls."));
    FunctionParser<3> demand_function;
    try
      {
        demand_function.initialize("x,y,z", parameters.tissue_demand, {});
      }
    catch (const std::exception &)
      {
        if (size == 1)
          throw;
        valid_controls = false;
      }
    AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid_controls),
                                    communicator) == 1,
                ExcMessage("Invalid tissue-demand expression."));
    GenerationResult result;
    result.network    = initial;
    result.validation = validate_data(initial, size);
    if (Utilities::MPI::min(static_cast<unsigned int>(
                              result.validation.valid()),
                            communicator) == 0)
      {
        if (result.validation.valid())
          result.validation.issues.push_back(
            {"input",
             invalid_id,
             "Invalid vascular input on another process."});
        return result;
      }
    bool same_selection = true;
    for (const auto &other :
         Utilities::MPI::all_gather(communicator,
                                    parameters.selected_terminals))
      same_selection = other == parameters.selected_terminals && same_selection;
    AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(same_selection),
                                    communicator) == 1,
                ExcMessage(
                  "All processes must select the same input terminal ids."));
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(background.get_mpi_communicator(),
                     communicator,
                     &comparison);
    const auto *mapping_q  = dynamic_cast<const MappingQ<3> *>(&mapping);
    const auto *mapping_fe = dynamic_cast<const MappingFE<3> *>(&mapping);
    bool        valid_mapping =
      (comparison == MPI_IDENT || comparison == MPI_CONGRUENT) &&
      ((mapping_q && mapping_q->get_degree() == 1) ||
       (mapping_fe && mapping_fe->get_degree() == 1));
    for (const auto &cell : background.active_cell_iterators())
      if (cell->is_locally_owned())
        valid_mapping = (cell->reference_cell().is_simplex() ||
                         cell->reference_cell().is_hyper_cube()) &&
                        mapping.is_compatible_with(cell->reference_cell()) &&
                        valid_mapping;
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid_mapping),
                          communicator) == 1,
      ExcMessage(
        "Vascular generation requires compatible first-order background mappings and congruent communicators."));
    BackgroundGeometry geometry(background, mapping);
    const auto         fail = [&](Id id, const std::string &message) {
      result.validation.issues.push_back({"attachment", id, message});
    };
    std::map<Id, std::vector<Id>> incoming, outgoing;
    std::vector<Id>               selected;
    Id next_node = 0, next_vessel = 0, next_terminal = 0, next_territory = 0;
    if (parameters.venous.enabled)
      {
        next_node     = next_identifier(initial.nodes);
        next_vessel   = next_identifier(initial.vessels);
        next_terminal = next_identifier(initial.terminals);
        for (const auto &[id, t] : initial.terminals)
          if (t.territory != invalid_id)
            {
              AssertThrow(t.territory < invalid_id - 1,
                          ExcMessage(
                            "Vascular territory identifier space exhausted."));
              next_territory = std::max(next_territory, t.territory + 1);
            }
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
        selected = parameters.selected_terminals;
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
          fail(invalid_id,
               "No available interior arterial terminal was found.");
        if (terminal_count < (creating_seed ? 1 : selected.size()))
          fail(
            invalid_id,
            "Terminal count is smaller than the number of selected attachments.");
        if (!result.validation.valid())
          return result;
      }
    else if (creating_seed && !parameters.selected_terminals.empty())
      {
        fail(invalid_id,
             "Cannot select input terminal ids from an empty network.");
        return result;
      }
    std::vector<std::pair<Segment<3>, Id>> obstacles;
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
      std::vector<Point<3>> vertices;
      for (const auto v : (*cell)->vertex_indices())
        vertices.push_back((*cell)->vertex(v));
      const auto directions = internal::initial_root_directions(
        position,
        (*cell)->center(),
        vertices,
        seed_direction,
        new_port ? std::optional<Point<3>>() :
                        std::optional<Point<3>>(
                     position - initial.nodes.at(properties.nodes[0]).position),
        shared,
        parameters.venous.terminal_separation);
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
          const internal::Edge edge{next_vessel,
                                         {{start, end_node}},
                                         {{position, end}},
                                    radius};
          Point<3>             lo, hi;
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
              const auto          &e = result.network.vessels.at(id);
              const internal::Edge other{id,
                                         e.nodes,
                                              {{s.first, s.second}},
                                         std::max(e.radii[0], e.radii[1])};
              if (!internal::compatible(edge,
                                        other,
                                        parameters.minimum_angle,
                                        parameters.clearance))
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
        auto startup = internal::start_seed(
          background, mapping, parameters, demand_function, communicator);
        result.validation = startup.validation;
        if (startup.committed)
          {
            result.network = std::move(startup.network);
            for (const auto &[id, radius] : startup.fixed_owned_roots)
              roots.push_back(id);
          }
      }
    else if (!parameters.venous.enabled)
      {
        auto startup      = internal::start_attachments(background,
                                                   mapping,
                                                   initial,
                                                   parameters,
                                                   demand_function,
                                                   terminal_count,
                                                   communicator);
        result.validation = startup.accepted.validation;
        if (startup.accepted.committed)
          {
            result.network          = std::move(startup.accepted.network);
            result.attachment_roots = std::move(startup.attachment_roots);
            for (const auto &[id, radius] : startup.accepted.fixed_owned_roots)
              roots.push_back(id);
          }
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
    const auto run_growth = [&](System system) {
      std::set<Id>         mutable_roots(roots.begin(), roots.end());
      std::map<Id, double> fixed_roots;
      for (Id root : roots)
        fixed_roots[root] = result.network.vessels.at(root).radii[0];
      const std::vector<Territory> empty;
      auto                         growth =
        internal::grow_forest(background,
                              mapping,
                              result.network,
                              mutable_roots,
                              fixed_roots,
                              parameters,
                              communicator,
                              terminal_count,
                              system,
                              system == System::venous ? result.territories :
                                                         empty);
      result.validation.issues.insert(result.validation.issues.end(),
                                      growth.validation.issues.begin(),
                                      growth.validation.issues.end());
      result.sampled_points +=
        Utilities::MPI::sum(growth.sampled_points, communicator);
      result.accepted_insertions += Utilities::MPI::sum(
        static_cast<unsigned int>(growth.accepted_insertions), communicator);
      if (growth.accepted.committed)
        {
          result.network = std::move(growth.accepted.network);
          if (system == System::arterial)
            {
              result.territories.clear();
              for (Id tid : growth.accepted.physiology.owned_leaf_terminals)
                {
                  const auto &t = result.network.terminals.at(tid);
                  result.territories.push_back(
                    {t.territory,
                     result.network.nodes.at(t.node).position,
                     t.owner,
                     t.demand});
                }
            }
          result.target_terminal_flows.insert(
            growth.accepted.physiology.terminal_flows.begin(),
            growth.accepted.physiology.terminal_flows.end());
          if (parameters.reference_flow > 0.)
            for (const auto &[root, radius] : growth.accepted.fixed_owned_roots)
              result.target_attachment_flows[root] =
                growth.accepted.physiology.owned.at(root).flow;
        }
    };
    run_growth(System::arterial);
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
        run_growth(System::venous);
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
    const auto report = validate_data(result.network, size);
    result.validation.issues.insert(result.validation.issues.end(),
                                    report.issues.begin(),
                                    report.issues.end());
    if (Utilities::MPI::min(static_cast<unsigned int>(
                              result.validation.valid()),
                            communicator) == 0 &&
        result.validation.valid())
      fail(invalid_id,
           "Vascular generation failed validation on another process.");
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
    const auto                 &p = parameters.calibration;
    const std::array<double, 6> modes{{p.volume_fraction,
                                       p.minimum_coverage,
                                       p.relative_tolerance,
                                       static_cast<double>(p.minimum_terminals),
                                       static_cast<double>(p.maximum_terminals),
                                       static_cast<double>(p.attempt_budget)}};
    bool                        matching_modes = true;
    for (const auto &other : Utilities::MPI::all_gather(communicator, modes))
      matching_modes = other == modes && matching_modes;
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(matching_modes),
                          communicator) == 1,
      ExcMessage(
        "All processes must use the same vascular calibration controls."));
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
