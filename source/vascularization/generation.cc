// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>

#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <immersx/vascularization/generation.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>

#include "generation_pairs.h"
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
    if (creating_seed && !parameters.selected_terminals.empty())
      {
        result.validation.issues.push_back(
          {"attachment",
           invalid_id,
           "Cannot select input terminal ids from an empty network."});
        return result;
      }
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
    const auto fail = [&](Id id, const std::string &message) {
      result.validation.issues.push_back({"attachment", id, message});
    };
    const auto collective_valid = [&]() {
      const bool valid = Utilities::MPI::min(static_cast<unsigned int>(
                                               result.validation.valid()),
                                             communicator) == 1;
      if (!valid && result.validation.valid())
        fail(invalid_id,
             "Vascular generation failed validation on another process.");
      return valid;
    };
    std::vector<Id> roots;
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
    else
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
    if (!collective_valid())
      return result;
    Id reserved_venous_port = invalid_id;
    if (parameters.venous.enabled)
      {
        auto outlet = internal::reserve_venous_outlet(
          background, mapping, result.network, parameters, communicator);
        result.validation.issues.insert(result.validation.issues.end(),
                                        outlet.validation.issues.begin(),
                                        outlet.validation.issues.end());
        if (!collective_valid())
          return result;
        result.network       = std::move(outlet.network);
        reserved_venous_port = outlet.port;
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
    if (!collective_valid())
      return result;
    if (parameters.venous.enabled)
      {
        auto startup = internal::start_venous(background,
                                              mapping,
                                              result.network,
                                              initial,
                                              parameters,
                                              result.territories,
                                              reserved_venous_port,
                                              communicator);
        result.validation.issues.insert(
          result.validation.issues.end(),
          startup.accepted.validation.issues.begin(),
          startup.accepted.validation.issues.end());
        if (!collective_valid())
          return result;
        if (!startup.accepted.committed)
          {
            fail(invalid_id, "No venous startup batch was committed.");
            return result;
          }
        result.network = std::move(startup.accepted.network);
        result.attachment_roots.insert(startup.attachment_roots.begin(),
                                       startup.attachment_roots.end());
        roots.clear();
        for (const auto &[id, radius] : startup.accepted.fixed_owned_roots)
          roots.push_back(id);
        run_growth(System::venous);
        if (!collective_valid())
          return result;
      }
    if (parameters.venous.enabled && result.validation.valid())
      internal::pair_territories(result, parameters, communicator);
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
    const std::array<double, 8> modes{
      {p.volume_fraction,
       p.minimum_coverage,
       p.relative_tolerance,
       static_cast<double>(p.minimum_terminals),
       static_cast<double>(p.maximum_terminals),
       static_cast<double>(p.attempt_budget),
       static_cast<double>(parameters.terminal_count),
       static_cast<double>(parameters.venous.enabled)}};
    bool matching_modes = true;
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
    const auto size = Utilities::MPI::n_mpi_processes(communicator);
    const bool creating_seed =
      Utilities::MPI::min(static_cast<unsigned int>(initial.nodes.empty() &&
                                                    initial.vessels.empty() &&
                                                    initial.terminals.empty()),
                          communicator) == 1;
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
                ExcMessage("Invalid vascular calibration controls."));
    // All adaptive count choices depend on the same validity and global
    // metrics. A local failure must be shared before entering those branches.
    const auto collective_valid = [&](ValidationReport &validation) {
      const bool valid =
        Utilities::MPI::min(static_cast<unsigned int>(validation.valid()),
                            communicator) == 1;
      if (!valid && validation.valid())
        validation.issues.push_back(
          {"calibration",
           invalid_id,
           "Vascular calibration failed validation on another process."});
      return valid;
    };
    GenerationResult baseline;
    baseline.network    = initial;
    baseline.validation = validate_data(initial, size);
    if (!collective_valid(baseline.validation))
      return baseline;
    baseline.discretization = discretize(
      background, mapping, initial, parameters.discretization, communicator);
    baseline.validation = baseline.discretization.validation;
    if (!collective_valid(baseline.validation))
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
        collective_valid(candidate.validation);
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
