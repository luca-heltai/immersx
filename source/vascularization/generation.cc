// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <immersx/vascularization/generation.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <set>

#include "archive_internal.h"
#include "distribution_internal.h"
#include "generation_pairs.h"
#include "growth_batch.h"
#include "growth_geometry.h"
#include "growth_loop.h"
#include "growth_start.h"
#include "network_serialization.h"
#include "tissue_growth_callbacks.h"

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
        (maximum_terminals == 0 || maximum_terminals >= minimum_terminals) &&
        attempt_budget > 0,
      dealii::ExcMessage(
        "Invalid vascular calibration targets, tolerance, terminal limits or attempt budget."));
  }

  void
  TissueIntegrationParameters::validate() const
  {
    AssertThrow(
      quadrature_order > 0 && quadrature_order <= 8 &&
        samples_per_terminal > 0 && maximum_points > 0 &&
        refinement_budget <= 20,
      dealii::ExcMessage(
        "Tissue integration requires quadrature order 1 through 8, positive sampling and point budgets, and at most 20 refinement rounds."));
  }

  Parameters::Parameters(const std::string &section)
    : dealii::ParameterAcceptor(section)
  {
    add_parameter(
      "Length scale",
      length_scale,
      "Reference for geometric length coefficients; physical input data stay in SI.",
      this->prm,
      dealii::Patterns::Selection(
        "meter|local mesh size|min mesh size|max mesh size"));
    coverage.service_radius = 1.;
    enter_subsection("Objectives");
    add_parameter("Target volume fraction",
                  calibration.volume_fraction,
                  "Zero disables the volume target.");
    add_parameter("Volume relative tolerance", calibration.relative_tolerance);
    add_parameter("Minimum tissue coverage", calibration.minimum_coverage);
    add_parameter("Minimum new terminals", calibration.minimum_terminals);
    add_parameter(
      "Maximum new terminals",
      calibration.maximum_terminals,
      "Absolute bound; zero uses the scaled initial count or the explicit minimum, whichever is larger.");
    add_parameter("Calibration attempt budget", calibration.attempt_budget);
    add_parameter(
      "Maximum service distance",
      coverage.service_radius,
      "Coefficient times H at each tissue quadrature point; zero disables coverage.");
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
    add_parameter(
      "Enforce geometry",
      enforce_geometry,
      "Enforce wall/vessel admissibility and scaled geometric growth bounds; false defers geometry repair while retaining hydraulic, topology and ownership checks.");
    add_parameter(
      "Seed length",
      seed_length,
      "Initial vessel length times H at its attachment; its radius stays physical.");
    add_parameter(
      "Clearance",
      clearance,
      "Additional wall clearance times the maximum H along the branch; a vessel pair uses the larger of its two maxima.");
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
      "Base count of newly generated tissue terminals across selected attachments, multiplied by Terminal count multiplier factor; unselected input terminals are excluded.");
    add_parameter(
      "Terminal count multiplier factor",
      terminal_count_multiplier_factor,
      "Multiply Terminal count by one, the global active background cell count or the global used background vertex count. Calibration limits remain absolute.",
      this->prm,
      dealii::Patterns::Selection(
        "none|background cell count|background vertex count"));
    add_parameter("Random seed", random_seed);
    add_parameter("Sample budget per insertion", sample_budget);
    add_parameter(
      "Density mesh exponent",
      density_beta,
      "Terminal proposal density proportional to w(x)*cell_volume^(-beta/3).");
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
                  "Minimum integral of ds/H along a complete physical branch.");
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
      "Positive finite relative demand density integrated over nearest-centre tissue territories; independent of sampling density.");
    add_parameter("Demand quadrature order",
                  tissue_integration.quadrature_order);
    add_parameter(
      "Demand samples per requested terminal",
      tissue_integration.samples_per_terminal,
      "Initial physical quadrature resolution; dyadic subdivisions reach this global target without changing the background mesh.");
    add_parameter(
      "Demand integration point budget",
      tissue_integration.maximum_points,
      "Global limit on active tissue quadrature points; exhaustion does not reduce the requested terminal count.");
    add_parameter(
      "Demand integration refinement budget",
      tissue_integration.refinement_budget,
      "Maximum additional integration subdivision rounds per territorial preview.");
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
    add_parameter(
      "Reference Robin conditions",
      reference_conditions.robin,
      "Physical Node id=beta:g for outward flow q=beta*p-g, beta>=0 in m^3/(Pa s), g in m^3/s. A positive beta anchors pressure. Do not combine with pressure or inflow at the same node.");
    add_parameter(
      "Minimum radius",
      minimum_radius,
      "Minimum generated radius/H along each complete physical branch.");
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
      "Positive coefficient times H at the shared territory centre.");
    add_parameter(
      "Microvascular resistance",
      venous.microvascular_resistance,
      "Positive Pa s/m^3 per bed, or zero to derive each bed from the explicit reference pressures and demand flow.");
    leave_subsection();
    enter_subsection("Portal network");
    add_parameter("Enabled", portal.enabled);
    add_parameter("Selected terminal ids", portal.selected_terminals);
    add_parameter("Seed position", portal.seed_position);
    add_parameter("Seed inward port normal", portal.seed_inward_port_normal);
    add_parameter("Seed radius", portal.seed_radius);
    add_parameter("Murray exponent", portal.murray_exponent);
    add_parameter("Terminal separation", portal.terminal_separation);
    add_parameter(
      "Flow fraction",
      portal_flow_fraction,
      "Fraction of total new-tissue flow supplied by portal inflow; strictly between zero and one.");
    add_parameter(
      "Compartment resistances",
      compartment_resistances,
      "Arterial, portal and venous star-arm resistances; all zero derives positive arms from demand flow.");
    leave_subsection();
  }

  void
  Parameters::validate(const bool creating_seed) const
  {
    AssertThrow(length_scale == "meter" || length_scale == "local mesh size" ||
                  length_scale == "min mesh size" ||
                  length_scale == "max mesh size",
                dealii::ExcMessage("Invalid length scale policy."));
    AssertThrow(terminal_count_multiplier_factor == "none" ||
                  terminal_count_multiplier_factor == "background cell count" ||
                  terminal_count_multiplier_factor == "background vertex count",
                dealii::ExcMessage(
                  "Invalid terminal count multiplier factor."));
    discretization.validate();
    coverage.validate();
    calibration.validate();
    tissue_integration.validate();
    AssertThrow(
      calibration.minimum_coverage == 0. || coverage.service_radius > 0.,
      dealii::ExcMessage(
        "A positive minimum coverage requires an explicit positive maximum service distance."));
    const auto positive = [](double x) { return std::isfinite(x) && x > 0.; };
    if (portal.enabled)
      {
        bool supplied = true, derived = true;
        for (const double r : compartment_resistances)
          {
            supplied = supplied && positive(r);
            derived  = derived && r == 0.;
          }
        AssertThrow(
          venous.enabled && positive(portal.murray_exponent) &&
            positive(portal.terminal_separation) &&
            std::isfinite(portal.seed_radius) && portal.seed_radius >= 0. &&
            std::isfinite(portal_flow_fraction) && portal_flow_fraction > 0. &&
            portal_flow_fraction < 1. &&
            (supplied || (derived && positive(reference_flow))) &&
            venous.microvascular_resistance == 0. &&
            portal.microvascular_resistance == 0.,
          dealii::ExcMessage(
            "Portal growth requires venous drainage, positive geometric controls, a flow fraction between zero and one, and three positive compartment resistances or a positive reference flow. The two-terminal bed resistance must remain zero."));
        for (unsigned int d = 0; d < 3; ++d)
          AssertThrow(std::isfinite(portal.seed_position[d]) &&
                        std::isfinite(portal.seed_inward_port_normal[d]),
                      dealii::ExcMessage(
                        "Portal seed coordinates and normals must be finite."));
      }
    if (venous.enabled)
      {
        reference_parameters.validate();
        AssertThrow(
          positive(venous.murray_exponent) &&
            positive(venous.terminal_separation) &&
            std::isfinite(venous.seed_radius) && venous.seed_radius >= 0. &&
            std::isfinite(venous.microvascular_resistance) &&
            venous.microvascular_resistance >= 0. &&
            (portal.enabled || venous.microvascular_resistance > 0. ||
             positive(reference_flow)),
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
        "Supply a positive initial length coefficient and, for an empty input, a positive seed radius and viscosity in SI units."));
    AssertThrow(
      std::isfinite(clearance) && clearance >= 0. &&
        (!enforce_geometry || (geometry_depth <= 30 && geometry_work > 0)),
      dealii::ExcMessage(
        "Clearance must be finite and nonnegative; geometry depth must be at most 30 and its patch budget positive."));
    AssertThrow(
      terminal_count > 0 && sample_budget > 0 && candidate_count > 0 &&
        optimization_steps <= 100,
      dealii::ExcMessage(
        "Terminal count, sample budget and candidate count must be positive; optimization steps must not exceed 100."));
    AssertThrow(
      positive(arterial_murray_exponent) && std::isfinite(minimum_radius) &&
        minimum_radius >= 0. && std::isfinite(minimum_length) &&
        minimum_length >= 0. && std::isfinite(minimum_angle) &&
        (!enforce_geometry || (minimum_angle > 0. && minimum_angle < 120.)) &&
        std::isfinite(dissipation_weight) && dissipation_weight >= 0. &&
        dissipation_weight <= 1. && std::isfinite(density_beta) &&
        positive(terminal_weight_bound),
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
    struct CompanionContinuation
    {
      NetworkData                         network;
      ValidationReport                    validation;
      std::vector<Territory>              territories;
      std::array<std::set<Id>, 3>         mutable_owned;
      std::array<std::map<Id, double>, 3> roots;
      std::map<Id, Id>                    attachments;
      std::map<Id, double>                terminal_flows, attachment_flows;
      std::array<std::uint64_t, 3>        leaves{};
      System                              active = System::portal;
    };

    struct ContinuationRootRecords
    {
      std::vector<std::pair<Id, std::pair<Terminal, bool>>> terminals;
      std::vector<std::pair<Id, Vessel>> parents, starts, ends;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &terminals &parents &starts &ends;
      }
    };

    CompanionContinuation
    prepare_companion_continuation(const NetworkData            &original,
                                   NetworkData                   checkpoint,
                                   const Parameters             &parameters,
                                   const System                  active,
                                   const MeshLengthScale        &scale,
                                   internal::BackgroundGeometry &geometry,
                                   MPI_Comm                      comm)
    {
      using namespace dealii;
      const auto            rank = Utilities::MPI::this_mpi_process(comm);
      const auto            size = Utilities::MPI::n_mpi_processes(comm);
      CompanionContinuation result;
      result.active   = active;
      result.network  = std::move(checkpoint);
      const auto fail = [&](Id id, const std::string &message) {
        result.validation.issues.push_back({"continuation", id, message});
      };
      const auto valid = [&]() {
        const bool all = Utilities::MPI::min(static_cast<unsigned int>(
                                               result.validation.valid()),
                                             comm) == 1;
        if (!all && result.validation.valid())
          fail(invalid_id,
               "Companion continuation is invalid on another process.");
        return all;
      };
      try
        {
          internal::validate_archive_data(original, {}, {}, comm);
          internal::validate_archive_data(result.network, {}, {}, comm);
        }
      catch (const std::exception &)
        {
          fail(
            invalid_id,
            "Continuation requires valid physical input and checkpoint records.");
        }
      if (!valid())
        return result;

      // Compare persistent records at id shards, independently of repartition.
      using Networks = std::pair<NetworkData, NetworkData>;
      std::map<unsigned int, Networks> send;
      const auto route = [&](const NetworkData &network, bool input) {
        for (const auto &[id, node] : network.nodes)
          if (node.owner == rank)
            (input ? send[id % size].first : send[id % size].second)
              .nodes.emplace(id, node);
        for (const auto &[id, vessel] : network.vessels)
          if (vessel.owner == rank)
            (input ? send[id % size].first : send[id % size].second)
              .vessels.emplace(id, vessel);
        for (const auto &[id, terminal] : network.terminals)
          if (terminal.owner == rank)
            (input ? send[id % size].first : send[id % size].second)
              .terminals.emplace(id, terminal);
      };
      route(original, true);
      route(result.network, false);
      NetworkData before, after;
      for (const auto &[sender, records] :
           Utilities::MPI::some_to_some(comm, send))
        {
          before.nodes.insert(records.first.nodes.begin(),
                              records.first.nodes.end());
          before.vessels.insert(records.first.vessels.begin(),
                                records.first.vessels.end());
          before.terminals.insert(records.first.terminals.begin(),
                                  records.first.terminals.end());
          after.nodes.insert(records.second.nodes.begin(),
                             records.second.nodes.end());
          after.vessels.insert(records.second.vessels.begin(),
                               records.second.vessels.end());
          after.terminals.insert(records.second.terminals.begin(),
                                 records.second.terminals.end());
        }
      for (const auto &[id, n] : before.nodes)
        {
          const auto found = after.nodes.find(id);
          if (found == after.nodes.end() ||
              found->second.position != n.position ||
              found->second.port != n.port ||
              found->second.provenance != n.provenance ||
              found->second.inward_port_normal != n.inward_port_normal)
            fail(
              id,
              "A continuation checkpoint changed or omitted an original node.");
        }
      for (const auto &[id, v] : before.vessels)
        {
          const auto found = after.vessels.find(id);
          if (found == after.vessels.end() || found->second.nodes != v.nodes ||
              found->second.radii != v.radii ||
              found->second.viscosity != v.viscosity ||
              found->second.system != v.system ||
              found->second.provenance != v.provenance ||
              found->second.physical_properties != v.physical_properties)
            fail(
              id,
              "A continuation checkpoint changed or omitted an original vessel.");
        }
      std::map<unsigned int, std::array<std::vector<Id>, 3>> replies;
      for (const auto &[id, n] : after.nodes)
        if (!before.nodes.count(id) &&
            (n.provenance != Provenance::generated || n.port != Port::none ||
             n.inward_port_normal.norm_square() != 0.))
          fail(
            id,
            "Continuation supports generated descendants of original attachments, not new seed ports.");
      for (const auto &[id, v] : after.vessels)
        if (!before.vessels.count(id))
          {
            if (v.provenance != Provenance::generated)
              fail(id,
                   "A new checkpoint vessel must retain generated provenance.");
            replies[v.owner][0].push_back(id);
          }
      for (const auto &[id, t] : before.terminals)
        {
          const auto found = after.terminals.find(id);
          if (found == after.terminals.end())
            replies[t.owner][2].push_back(id);
          else if (found->second.node != t.node ||
                   found->second.territory != t.territory ||
                   found->second.system != t.system ||
                   found->second.demand != t.demand)
            fail(
              id,
              "A retained original terminal changed physical identity or demand.");
          if (t.territory != invalid_id)
            fail(
              id,
              "Companion continuation requires an original registry without existing beds.");
        }
      for (const auto &[id, t] : after.terminals)
        if (!before.terminals.count(id))
          replies[t.owner][1].push_back(id);
      if (!valid())
        return result;
      std::set<Id> missing;
      for (const auto &[sender, ids] :
           Utilities::MPI::some_to_some(comm, replies))
        {
          for (Id id : ids[0])
            result
              .mutable_owned[static_cast<unsigned int>(
                result.network.vessels.at(id).system)]
              .insert(id);
          for (Id id : ids[1])
            {
              const auto &t = result.network.terminals.at(id);
              ++result.leaves[static_cast<unsigned int>(t.system)];
              if (t.territory == invalid_id)
                fail(
                  id,
                  "A new checkpoint terminal must identify a common territory.");
              if (t.system == System::arterial)
                result.territories.push_back(
                  {t.territory,
                   result.network.nodes.at(t.node).position,
                   rank,
                   t.demand});
            }
          missing.insert(ids[2].begin(), ids[2].end());
        }
      if (!valid())
        return result;

      // Attachment joins use node shards, not locally relevant adjacency.
      std::map<unsigned int, ContinuationRootRecords> root_send;
      for (const auto &[id, t] : original.terminals)
        if (t.owner == rank)
          root_send[t.node % size].terminals.push_back(
            {id, {t, missing.count(id) != 0}});
      for (const auto &[id, v] : original.vessels)
        if (v.owner == rank)
          root_send[v.nodes[1] % size].parents.emplace_back(id, v);
      for (const auto &ids : result.mutable_owned)
        for (Id id : ids)
          {
            const auto &v = result.network.vessels.at(id);
            root_send[v.nodes[0] % size].starts.emplace_back(id, v);
            root_send[v.nodes[1] % size].ends.emplace_back(id, v);
          }
      std::map<Id, std::pair<Id, std::pair<Terminal, bool>>> terminals;
      std::map<Id, std::vector<std::pair<Id, Vessel>>>       parents, starts;
      for (const auto &[sender, packet] :
           Utilities::MPI::some_to_some(comm, root_send))
        {
          for (const auto &t : packet.terminals)
            terminals.emplace(t.second.first.node, t);
          for (const auto &v : packet.parents)
            parents[v.second.nodes[1]].push_back(v);
          for (const auto &v : packet.starts)
            starts[v.second.nodes[0]].push_back(v);
          for (const auto &v : packet.ends)
            if (before.nodes.count(v.second.nodes[1]))
              fail(v.first,
                   "A generated descendant cannot end at an original node.");
        }
      using RootReply =
        std::pair<std::array<std::map<Id, double>, 3>, std::map<Id, Id>>;
      std::map<unsigned int, RootReply> root_replies;
      for (const auto &[node, incoming] : starts)
        if (before.nodes.count(node) && !terminals.count(node))
          fail(
            node,
            "A continuation root must start at an original tissue terminal.");
      for (const auto &[node, record] : terminals)
        {
          const auto &[id, entry]   = record;
          const auto &[t, consumed] = entry;
          const auto &children      = starts[node];
          if (children.empty())
            {
              if (consumed)
                fail(id,
                     "An original terminal disappeared without an attachment.");
              continue;
            }
          if (!consumed || children.size() != 1 || parents[node].size() != 1 ||
              before.nodes.at(node).port != Port::none)
            {
              fail(
                id,
                "An attachment must consume exactly one original non-port leaf.");
              continue;
            }
          const auto &[root, v] = children.front();
          const auto &parent    = parents[node].front().second;
          if (v.system != t.system || parent.system != t.system ||
              v.radii[0] != parent.radii[1])
            {
              fail(
                root,
                "A continuation root disagrees with its original system or fixed cap radius.");
              continue;
            }
          root_replies[v.owner]
            .first[static_cast<unsigned int>(t.system)][root] = parent.radii[1];
          root_replies[t.owner].second[id]                    = root;
        }
      if (!valid())
        return result;
      for (const auto &[sender, reply] :
           Utilities::MPI::some_to_some(comm, root_replies))
        {
          for (unsigned int s = 0; s < 3; ++s)
            result.roots[s].insert(reply.first[s].begin(),
                                   reply.first[s].end());
          result.attachments.insert(reply.second.begin(), reply.second.end());
        }
      for (auto &leaves : result.leaves)
        leaves = Utilities::MPI::sum(leaves, comm);
      const std::array<const std::vector<Id> *, 3> selections{
        {&parameters.selected_terminals,
         &parameters.venous.selected_terminals,
         &parameters.portal.selected_terminals}};
      std::array<std::uint64_t, 3> consumed_counts{};
      for (const auto &[id, root] : result.attachments)
        {
          const auto system =
            static_cast<unsigned int>(original.terminals.at(id).system);
          ++consumed_counts[system];
          if (!selections[system]->empty() &&
              std::find(selections[system]->begin(),
                        selections[system]->end(),
                        id) == selections[system]->end())
            fail(
              id,
              "A checkpoint attachment is excluded by the original input selection.");
        }
      for (unsigned int system = 0; system < 3; ++system)
        {
          consumed_counts[system] =
            Utilities::MPI::sum(consumed_counts[system], comm);
          if (result.leaves[system] > 0 && !selections[system]->empty() &&
              consumed_counts[system] != selections[system]->size())
            fail(
              invalid_id,
              "A prepared checkpoint phase must retain all selected original attachments.");
        }
      const auto n = result.leaves[static_cast<unsigned int>(System::arterial)];
      if (n == 0 || n > std::numeric_limits<unsigned int>::max() ||
          result.leaves[2] > n || result.leaves[1] > n ||
          (!parameters.portal.enabled && result.leaves[2] != 0) ||
          (active == System::portal && result.leaves[1] != 0) ||
          (active == System::venous && parameters.portal.enabled &&
           result.leaves[2] != n))
        fail(
          invalid_id,
          "Companion checkpoint counts disagree with the completed arterial territories or phase order.");
      if (!valid())
        return result;

      // Revalidate materialized Murray state without adopting recomputed radii.
      for (unsigned int s = 0; s < 3; ++s)
        {
          if (result.leaves[s] == 0)
            continue;
          const auto physical =
            internal::evaluate_forest(result.network,
                                      result.mutable_owned[s],
                                      result.roots[s],
                                      parameters,
                                      comm,
                                      scale);
          result.validation.issues.insert(result.validation.issues.end(),
                                          physical.validation.issues.begin(),
                                          physical.validation.issues.end());
          if (!valid())
            return result;
          if (s == static_cast<unsigned int>(System::arterial) ||
              (s == static_cast<unsigned int>(System::portal) &&
               active == System::venous))
            {
              if (parameters.enforce_geometry)
                {
                  // Skipped completed phases still need the native final
                  // geometry certificate, using the exact stored endpoint
                  // radius profiles.
                  std::vector<internal::BatchProposal> checks;
                  for (Id id : result.mutable_owned[s])
                    {
                      const auto             &v = result.network.vessels.at(id);
                      internal::BatchProposal check;
                      check.id            = id;
                      check.owners[id]    = rank;
                      check.old_nodes[id] = v.nodes;
                      check.changed.push_back(
                        {id,
                         v.nodes,
                         {{result.network.nodes.at(v.nodes[0]).position,
                           result.network.nodes.at(v.nodes[1]).position}},
                         v.radii[0]});
                      for (Id node : v.nodes)
                        {
                          const auto &n = result.network.nodes.at(node);
                          if (std::max({std::abs(n.inward_port_normal[0]),
                                        std::abs(n.inward_port_normal[1]),
                                        std::abs(n.inward_port_normal[2])}) >
                              0.)
                            {
                              check.port_owners[node] = n.owner;
                              check.normals[node]     = n.inward_port_normal;
                            }
                        }
                      checks.push_back(std::move(check));
                    }
                  const auto certified = internal::select_growth_batch(
                    geometry,
                    result.network,
                    checks,
                    parameters,
                    comm,
                    [&](Id id) { return result.network.vessels.at(id).radii; },
                    true);
                  for (const auto &[id, accepted] : certified.local)
                    if (!accepted)
                      fail(
                        id,
                        "Completed checkpoint geometry failed native certification.");
                  if (!valid())
                    return result;
                }
              result.terminal_flows.insert(physical.terminal_flows.begin(),
                                           physical.terminal_flows.end());
              if (parameters.reference_flow > 0.)
                for (const auto &[root, radius] : result.roots[s])
                  result.attachment_flows[root] = physical.owned.at(root).flow;
            }
          double gamma = parameters.arterial_murray_exponent;
          if (s == static_cast<unsigned int>(System::venous))
            gamma = parameters.venous.murray_exponent;
          else if (s == static_cast<unsigned int>(System::portal))
            gamma = parameters.portal.murray_exponent;
          const long double tolerance =
            64.L * std::numeric_limits<double>::epsilon() *
            (1.L + (1.L + physical.upward_levels) / gamma);
          for (const auto &[id, branch] : physical.owned)
            for (const double r : result.network.vessels.at(id).radii)
              if (std::abs(static_cast<long double>(r) - branch.radius) /
                    std::max<long double>(r, branch.radius) >
                  tolerance)
                fail(
                  id,
                  "Checkpoint radii disagree with the independently reconstructed fixed-root forest.");
          const auto leaves =
            Utilities::MPI::sum(static_cast<std::uint64_t>(
                                  physical.owned_leaf_terminals.size()),
                                comm);
          if (leaves != result.leaves[s])
            fail(
              invalid_id,
              "Checkpoint terminal registry disagrees with its generated forest.");
          if (!valid())
            return result;
        }
      return result;
    }
  } // namespace

  static unsigned int
  resolve_terminal_count(const dealii::Triangulation<3> &background,
                         const Parameters               &parameters,
                         MPI_Comm                        communicator)
  {
    using namespace dealii;
    const auto &factor = parameters.terminal_count_multiplier_factor;
    AssertThrow(factor == "none" || factor == "background cell count" ||
                  factor == "background vertex count",
                ExcMessage("Invalid terminal count multiplier factor."));
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(background.get_mpi_communicator(),
                     communicator,
                     &comparison);
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(comparison == MPI_IDENT ||
                                                    comparison ==
                                                      MPI_CONGRUENT),
                          communicator) == 1,
      ExcMessage("Background and generation communicators must be congruent."));
    std::uint64_t multiplier = 1;
    if (factor == "none")
      multiplier = 1;
    else if (factor == "background cell count")
      multiplier = background.n_global_active_cells();
    else if (factor == "background vertex count")
      {
        // One unconstrained linear DoF per used vertex gives the global
        // cardinality, including hanging nodes and shared coarse vertices.
        unsigned int families[2]{0, 0};
        bool         supported = true;
        for (const auto &cell : background.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              if (cell->reference_cell().is_simplex())
                families[0] = 1;
              else if (cell->reference_cell().is_hyper_cube())
                families[1] = 1;
              else
                supported = false;
            }
        Utilities::MPI::max(families, communicator, families);
        AssertThrow(
          Utilities::MPI::min(static_cast<unsigned int>(supported),
                              communicator) == 1 &&
            families[0] + families[1] == 1,
          ExcMessage(
            "Vertex counting requires a nonempty all-tetrahedral or all-hexahedral background."));
        const bool           simplex = families[0] != 0;
        const FE_Q<3>        hexahedral(1);
        const FE_SimplexP<3> tetrahedral(1);
        DoFHandler<3>        vertices(background);
        vertices.distribute_dofs(
          simplex ? static_cast<const FiniteElement<3> &>(tetrahedral) :
                    static_cast<const FiniteElement<3> &>(hexahedral));
        multiplier = vertices.n_dofs();
      }
    AssertThrow(
      parameters.terminal_count > 0 && multiplier > 0 &&
        multiplier <=
          std::numeric_limits<unsigned int>::max() / parameters.terminal_count,
      ExcMessage(
        "The multiplied terminal count must be positive and fit in unsigned int."));
    return static_cast<unsigned int>(parameters.terminal_count * multiplier);
  }

  static GenerationResult
  generate_count(
    const dealii::Triangulation<3>                &background,
    const dealii::Mapping<3>                      &mapping,
    const NetworkData                             &initial,
    const Parameters                              &parameters,
    MPI_Comm                                       communicator,
    unsigned int                                   terminal_count,
    unsigned int                                   initial_count,
    const std::vector<Territory>                  &initial_territories,
    const std::vector<TerminalPair>               &initial_pairs,
    const std::vector<TerminalTriplet>            &initial_triplets,
    std::unique_ptr<internal::BackgroundGeometry> &prepared_geometry,
    GrowthWork                                    &preparation_work,
    const NetworkData                             *checkpoint = nullptr,
    const System active_companion                             = System::portal)
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
    result.base_terminal_count      = parameters.terminal_count;
    result.initial_terminal_count   = initial_count;
    result.requested_terminal_count = terminal_count;
    result.network                  = initial;
    result.validation               = validate_data(initial, size);
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
      background.n_levels() > 0 &&
      (comparison == MPI_IDENT || comparison == MPI_CONGRUENT) &&
      ((mapping_q && mapping_q->get_degree() == 1) ||
       (mapping_fe && mapping_fe->get_degree() == 1));
    if (background.n_levels() > 0)
      for (const auto &cell : background.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            count_work(&preparation_work,
                       GrowthOperation::background_cell_visits);
            valid_mapping =
              (cell->reference_cell().is_simplex() ||
               cell->reference_cell().is_hyper_cube()) &&
              mapping.is_compatible_with(cell->reference_cell()) &&
              valid_mapping;
          }
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid_mapping),
                          communicator) == 1,
      ExcMessage(
        "Vascular generation requires compatible first-order background mappings and congruent communicators."));
    if (!prepared_geometry)
      prepared_geometry =
        std::make_unique<internal::BackgroundGeometry>(background,
                                                       mapping,
                                                       &preparation_work);
    auto &geometry = *prepared_geometry;
    geometry.configure_length_scale(parameters.length_scale, communicator);
    result.length_scale = *geometry.length_scale;
    const auto fail     = [&](Id id, const std::string &message) {
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
    std::set<Id>    mutable_forest;
    Id reserved_venous_port = invalid_id, reserved_portal_port = invalid_id;
    std::optional<CompanionContinuation> continuation;
    if (checkpoint)
      {
        const auto cp_validation = validate_data(*checkpoint, size);
        result.validation.issues.insert(result.validation.issues.end(),
                                        cp_validation.issues.begin(),
                                        cp_validation.issues.end());
        if (!collective_valid())
          return result;
        const double halo =
          result.length_scale.physical_length(parameters.clearance,
                                              result.length_scale.maximum());
        const auto original_owned = internal::redistribute_with_owners(
          background, mapping, initial, halo, communicator, {}, &geometry);
        auto restored = internal::redistribute_with_owners(
          background, mapping, *checkpoint, halo, communicator, {}, &geometry);
        continuation            = prepare_companion_continuation(original_owned,
                                                      std::move(restored),
                                                      parameters,
                                                      active_companion,
                                                      result.length_scale,
                                                      geometry,
                                                      communicator);
        result.validation       = continuation->validation;
        result.network          = std::move(continuation->network);
        result.territories      = continuation->territories;
        result.attachment_roots = continuation->attachments;
        result.target_terminal_flows   = continuation->terminal_flows;
        result.target_attachment_flows = continuation->attachment_flows;
        if (!collective_valid())
          return result;
        terminal_count = static_cast<unsigned int>(continuation->leaves[0]);
        result.requested_terminal_count = terminal_count;
        const auto maximum =
          parameters.calibration.maximum_terminals == 0 ?
            std::max(initial_count, parameters.calibration.minimum_terminals) :
            parameters.calibration.maximum_terminals;
        if (terminal_count < parameters.calibration.minimum_terminals ||
            terminal_count > maximum)
          fail(
            invalid_id,
            "Completed arterial count lies outside the configured terminal limits.");
        if (!collective_valid())
          return result;
      }
    else
      {
        if (creating_seed)
          {
            auto startup = internal::start_seed(
              background, mapping, parameters, communicator, &geometry);
            result.validation = startup.validation;
            if (startup.prepared)
              {
                result.network = std::move(startup.network);
                mutable_forest = std::move(startup.mutable_owned);
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
                                                       terminal_count,
                                                       communicator,
                                                       &geometry);
            result.validation = startup.accepted.validation;
            if (startup.accepted.prepared)
              {
                result.network = std::move(startup.accepted.network);
                mutable_forest = std::move(startup.accepted.mutable_owned);
                result.attachment_roots = std::move(startup.attachment_roots);
                for (const auto &[id, radius] :
                     startup.accepted.fixed_owned_roots)
                  roots.push_back(id);
              }
          }
        if (!collective_valid())
          return result;
        if (parameters.venous.enabled)
          {
            auto outlet = internal::reserve_companion_port(background,
                                                           mapping,
                                                           result.network,
                                                           parameters,
                                                           communicator,
                                                           System::venous,
                                                           &geometry);
            result.validation.issues.insert(result.validation.issues.end(),
                                            outlet.validation.issues.begin(),
                                            outlet.validation.issues.end());
            if (!collective_valid())
              return result;
            result.network       = std::move(outlet.network);
            reserved_venous_port = outlet.port;
          }
        if (parameters.portal.enabled)
          {
            auto inlet = internal::reserve_companion_port(background,
                                                          mapping,
                                                          result.network,
                                                          parameters,
                                                          communicator,
                                                          System::portal,
                                                          &geometry);
            result.validation.issues.insert(result.validation.issues.end(),
                                            inlet.validation.issues.begin(),
                                            inlet.validation.issues.end());
            if (!collective_valid())
              return result;
            result.network       = std::move(inlet.network);
            reserved_portal_port = inlet.port;
          }
      }
    const auto run_growth = [&](System system) {
      std::map<Id, double> fixed_roots;
      for (Id root : roots)
        fixed_roots[root] = result.network.vessels.at(root).radii[0];
      auto                         callbacks = system == System::arterial ?
                                                 internal::tissue_growth_callbacks(geometry,
                                                           parameters,
                                                           terminal_count,
                                                           communicator) :
                                                 internal::GrowthDemandCallbacks{};
      const std::vector<Territory> empty;
      auto                         growth =
        internal::grow_forest(background,
                              mapping,
                              result.network,
                              mutable_forest,
                              fixed_roots,
                              parameters,
                              communicator,
                              terminal_count,
                              system,
                              system != System::arterial ? result.territories :
                                                           empty,
                              std::move(callbacks),
                              &geometry);
      if (!growth.statistics.empty())
        {
          auto &first = growth.statistics.begin()->second;
          first.work += preparation_work;
          first.initialization_work += preparation_work;
          preparation_work = {};
        }
      for (const auto &[bucket, statistics] : growth.statistics)
        result.growth_statistics.push_back(statistics);
      result.validation.issues.insert(result.validation.issues.end(),
                                      growth.validation.issues.begin(),
                                      growth.validation.issues.end());
      result.sampled_points +=
        Utilities::MPI::sum(growth.sampled_points, communicator);
      result.accepted_insertions += Utilities::MPI::sum(
        static_cast<unsigned int>(growth.accepted_insertions), communicator);
      // Provisional startup labels must not escape a failed bootstrap as a
      // physical result. A later growth failure retains its last accepted
      // state.
      if (!growth.accepted.committed && system == System::arterial)
        {
          result.network = initial;
          result.attachment_roots.clear();
          mutable_forest.clear();
        }
      if (growth.accepted.committed)
        {
          result.network = std::move(growth.accepted.network);
          mutable_forest = std::move(growth.accepted.mutable_owned);
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
    if (!continuation)
      run_growth(System::arterial);
    result.effective_terminal_count =
      Utilities::MPI::sum(static_cast<unsigned int>(result.territories.size()),
                          communicator);
    if (!collective_valid())
      return result;
    for (System companion_system : {System::portal, System::venous})
      if ((companion_system == System::portal && parameters.portal.enabled) ||
          (companion_system == System::venous && parameters.venous.enabled))
        {
          if (continuation && active_companion == System::venous &&
              companion_system == System::portal)
            continue;
          const auto component = static_cast<unsigned int>(companion_system);
          if (continuation && continuation->leaves[component] > 0)
            {
              mutable_forest = continuation->mutable_owned[component];
              roots.clear();
              for (const auto &[id, radius] : continuation->roots[component])
                roots.push_back(id);
            }
          else
            {
              auto startup =
                internal::start_companion(background,
                                          mapping,
                                          result.network,
                                          initial,
                                          parameters,
                                          result.territories,
                                          companion_system == System::portal ?
                                            reserved_portal_port :
                                            reserved_venous_port,
                                          communicator,
                                          companion_system,
                                          &geometry);
              result.validation.issues.insert(
                result.validation.issues.end(),
                startup.accepted.validation.issues.begin(),
                startup.accepted.validation.issues.end());
              if (!collective_valid())
                return result;
              if (!startup.accepted.prepared)
                {
                  fail(invalid_id, "No companion startup batch was prepared.");
                  return result;
                }
              result.network = std::move(startup.accepted.network);
              mutable_forest = std::move(startup.accepted.mutable_owned);
              result.attachment_roots.insert(startup.attachment_roots.begin(),
                                             startup.attachment_roots.end());
              roots.clear();
              for (const auto &[id, radius] :
                   startup.accepted.fixed_owned_roots)
                roots.push_back(id);
            }
          run_growth(companion_system);
          if (!collective_valid())
            return result;
        }
    auto retained = internal::retain_metadata(initial,
                                              initial_territories,
                                              initial_pairs,
                                              result.network,
                                              communicator,
                                              initial_triplets);
    result.validation.issues.insert(result.validation.issues.end(),
                                    retained.validation.issues.begin(),
                                    retained.validation.issues.end());
    if (!collective_valid())
      return result;
    if (parameters.venous.enabled)
      internal::pair_territories(
        result, parameters, communicator, retained.pairs, retained.triplets);
    result.territories.insert(result.territories.end(),
                              retained.territories.begin(),
                              retained.territories.end());
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
                                         communicator,
                                         result.length_scale);
        if (result.validation.valid())
          result.normalized_geometry =
            measure_scaled_vessels(background,
                                   mapping,
                                   result.network,
                                   result.length_scale,
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
    return generate(
      background, mapping, initial, parameters, communicator, {}, {});
  }

  GenerationResult
  generate(const dealii::Triangulation<3>  &background,
           const dealii::Mapping<3>        &mapping,
           const NetworkData               &initial,
           const Parameters                &parameters,
           MPI_Comm                         communicator,
           const std::vector<Territory>    &territories,
           const std::vector<TerminalPair> &pairs)
  {
    return generate(background,
                    mapping,
                    initial,
                    parameters,
                    communicator,
                    territories,
                    pairs,
                    {});
  }

  GenerationResult
  generate(const dealii::Triangulation<3>     &background,
           const dealii::Mapping<3>           &mapping,
           const NetworkData                  &initial,
           const Parameters                   &parameters,
           MPI_Comm                            communicator,
           const std::vector<Territory>       &initial_territories,
           const std::vector<TerminalPair>    &initial_pairs,
           const std::vector<TerminalTriplet> &initial_triplets)
  {
    using namespace dealii;
    GrowthWork                                    preparation_work;
    std::unique_ptr<internal::BackgroundGeometry> prepared_geometry;

    auto p = parameters.calibration;

    const auto modes = std::make_pair(
      std::array<double, 11>{
        {p.volume_fraction,
         p.minimum_coverage,
         p.relative_tolerance,
         static_cast<double>(p.minimum_terminals),
         static_cast<double>(p.maximum_terminals),
         static_cast<double>(p.attempt_budget),
         static_cast<double>(parameters.terminal_count),
         static_cast<double>(parameters.venous.enabled),
         static_cast<double>(parameters.portal.enabled),
         parameters.portal_flow_fraction,
         static_cast<double>(parameters.enforce_geometry)}},
      std::array<std::string, 2>{{parameters.terminal_count_multiplier_factor,
                                  parameters.length_scale}});
    bool matching_modes = true;
    for (const auto &other : Utilities::MPI::all_gather(communicator, modes))
      matching_modes = other == modes && matching_modes;
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(matching_modes),
                          communicator) == 1,
      ExcMessage(
        "All processes must use the same vascular calibration controls."));
    const auto initial_count =
      resolve_terminal_count(background, parameters, communicator);
    const bool has_metadata =
      Utilities::MPI::sum(static_cast<Id>(initial_territories.size() +
                                          initial_pairs.size() +
                                          initial_triplets.size()),
                          communicator) > 0;
    const auto global_triplets =
      Utilities::MPI::sum(static_cast<Id>(initial_triplets.size()),
                          communicator);
    if (has_metadata)
      {
        AssertThrow(
          Utilities::MPI::sum(static_cast<Id>(initial_pairs.size() +
                                              initial_triplets.size()),
                              communicator) == 0 ||
            (parameters.venous.enabled &&
             (global_triplets == 0 || parameters.portal.enabled)),
          ExcMessage(
            "Completing input beds requires venous growth to be enabled."));
        internal::validate_archive_data(initial,
                                        initial_territories,
                                        initial_pairs,
                                        communicator,
                                        initial_triplets);
      }
    if (p.maximum_terminals == 0)
      p.maximum_terminals = std::max(initial_count, p.minimum_terminals);
    if (p.volume_fraction == 0. && p.minimum_coverage == 0.)
      return generate_count(background,
                            mapping,
                            initial,
                            parameters,
                            communicator,
                            initial_count,
                            initial_count,
                            initial_territories,
                            initial_pairs,
                            initial_triplets,
                            prepared_geometry,
                            preparation_work);
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
    baseline.length_scale =
      mesh_length_scale(background, parameters.length_scale, communicator);
    baseline.base_terminal_count      = parameters.terminal_count;
    baseline.initial_terminal_count   = initial_count;
    baseline.requested_terminal_count = initial_count;
    baseline.network                  = initial;
    baseline.validation               = validate_data(initial, size);
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
    std::vector<GrowthStatistics>   statistics;
    std::set<unsigned int>          tried;
    std::optional<GenerationResult> best;
    std::pair<double, double>       best_score{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()};
    unsigned int best_count = 0;
    unsigned int count =
      std::clamp(initial_count, p.minimum_terminals, p.maximum_terminals);
    for (unsigned int attempt = 0; attempt < p.attempt_budget; ++attempt)
      {
        auto candidate = generate_count(background,
                                        mapping,
                                        initial,
                                        parameters,
                                        communicator,
                                        count,
                                        initial_count,
                                        initial_territories,
                                        initial_pairs,
                                        initial_triplets,
                                        prepared_geometry,
                                        preparation_work);
        statistics.insert(statistics.end(),
                          candidate.growth_statistics.begin(),
                          candidate.growth_statistics.end());
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
                         (parameters.portal.enabled ?
                            candidate.tissue.triple_coverage :
                          parameters.venous.enabled ?
                            candidate.tissue.dual_coverage :
                            candidate.tissue.arterial_coverage));
            if (error <= p.relative_tolerance && deficit == 0.)
              {
                report.status               = CalibrationStatus::converged;
                candidate.calibration       = std::move(report);
                candidate.growth_statistics = std::move(statistics);
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
    auto result              = best ? std::move(*best) : std::move(baseline);
    report.status            = CalibrationStatus::exhausted;
    result.calibration       = std::move(report);
    result.growth_statistics = std::move(statistics);
    result.validation.issues.push_back(
      {"calibration",
       invalid_id,
       "No terminal count met all requested volume and coverage targets within the finite calibration/search budget. This does not prove incompatibility."});
    return result;
  }

  GenerationResult
  resume_companions(const dealii::Triangulation<3> &background,
                    const dealii::Mapping<3>       &mapping,
                    const NetworkData              &original,
                    const NetworkData              &checkpoint,
                    const Parameters               &parameters,
                    MPI_Comm                        communicator,
                    const System                    active_companion)
  {
    using namespace dealii;
    const auto controls = std::make_pair(
      std::make_pair(
        std::array<double, 11>{
          {static_cast<double>(active_companion),
           parameters.seed_length,
           static_cast<double>(parameters.venous.enabled),
           static_cast<double>(parameters.portal.enabled),
           parameters.calibration.volume_fraction,
           parameters.calibration.minimum_coverage,
           parameters.calibration.relative_tolerance,
           static_cast<double>(parameters.calibration.minimum_terminals),
           static_cast<double>(parameters.calibration.maximum_terminals),
           static_cast<double>(parameters.terminal_count),
           static_cast<double>(parameters.enforce_geometry)}},
        std::array<std::string, 2>{{parameters.terminal_count_multiplier_factor,
                                    parameters.length_scale}}),
      std::array<std::vector<Id>, 3>{{parameters.selected_terminals,
                                      parameters.venous.selected_terminals,
                                      parameters.portal.selected_terminals}});
    bool matching = true;
    for (const auto &other : Utilities::MPI::all_gather(communicator, controls))
      matching = matching && other == controls;
    AssertThrow(
      matching && parameters.venous.enabled &&
        ((active_companion == System::portal && parameters.portal.enabled) ||
         active_companion == System::venous),
      ExcMessage(
        "Companion continuation requires matching controls and an enabled portal or venous phase."));
    const auto initial_count =
      resolve_terminal_count(background, parameters, communicator);
    GrowthWork                                    work;
    std::unique_ptr<internal::BackgroundGeometry> geometry;
    auto        result = generate_count(background,
                                 mapping,
                                 original,
                                 parameters,
                                 communicator,
                                 initial_count,
                                 initial_count,
                                        {},
                                        {},
                                        {},
                                 geometry,
                                 work,
                                 &checkpoint,
                                 active_companion);
    const auto &p      = parameters.calibration;
    if (result.validation.valid())
      {
        const double error =
          p.volume_fraction > 0. ?
            std::abs(result.tissue.volume_fraction / p.volume_fraction - 1.) :
            0.;
        const double coverage = parameters.portal.enabled ?
                                  result.tissue.triple_coverage :
                                  result.tissue.dual_coverage;
        if (error > p.relative_tolerance || coverage < p.minimum_coverage)
          result.validation.issues.push_back(
            {"calibration",
             invalid_id,
             "The fixed arterial checkpoint count does not meet the requested volume or coverage objective; continuation never regrows arteries."});
      }
    result.calibration.status         = result.validation.valid() ?
                                          CalibrationStatus::converged :
                                          CalibrationStatus::exhausted;
    result.calibration.sampled_points = result.sampled_points;
    result.calibration.attempts.push_back({result.requested_terminal_count,
                                           result.tissue,
                                           result.validation,
                                           result.sampled_points});
    return result;
  }
} // namespace ImmersX::Vascularization
