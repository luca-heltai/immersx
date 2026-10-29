// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/matrix_tools.h>

#include <immersx/vascularization/reference_flow.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace ImmersX::Vascularization
{
  void
  ReferenceParameters::validate() const
  {
    const auto tolerance = [](double x) {
      return std::isfinite(x) && x > 0. && x < 1.;
    };
    AssertThrow(
      maximum_iterations > 0 && tolerance(linear_relative_tolerance) &&
        tolerance(conservation_relative_tolerance) &&
        std::isfinite(absolute_flow_tolerance) && absolute_flow_tolerance >= 0.,
      dealii::ExcMessage(
        "Invalid vascular reference iteration budget or numerical tolerances."));
  }

  ReferenceResult
  solve_reference(const NetworkData               &network,
                  const std::vector<TerminalPair> &pairs,
                  const ReferenceConditions       &conditions,
                  const ReferenceParameters       &parameters,
                  MPI_Comm                         communicator)
  {
    using namespace dealii;
    AssertThrow(Utilities::MPI::n_mpi_processes(communicator) == 1,
                ExcMessage(
                  "The current vascular reference solve requires one rank."));
    parameters.validate();
    ReferenceResult result;
    result.validation = validate_data(network, 1);
    const auto issue =
      [&](const std::string &entity, Id id, const std::string &message) {
        result.validation.issues.push_back({entity, id, message});
      };
    const auto paired = validate_terminal_pairs(network, pairs);
    result.validation.issues.insert(result.validation.issues.end(),
                                    paired.issues.begin(),
                                    paired.issues.end());
    if (network.vessels.empty())
      issue("reference",
            invalid_id,
            "Reference flow requires a nonempty vascular graph.");
    if (!result.validation.valid())
      return result;
    std::set<Id> paired_nodes, tissue_nodes;
    for (const auto &[id, t] : network.terminals)
      tissue_nodes.insert(t.node);
    for (const auto &pair : pairs)
      {
        paired_nodes.insert(network.terminals.at(pair.arterial).node);
        paired_nodes.insert(network.terminals.at(pair.venous).node);
      }
    for (const auto *data : {&conditions.pressures, &conditions.inflows})
      for (const auto &[id, value] : *data)
        {
          if (!network.nodes.count(id) || !std::isfinite(value))
            issue(
              "boundary",
              id,
              "Boundary data require a known physical node and a finite SI value.");
          const auto node = network.nodes.find(id);
          if (node != network.nodes.end() && node->second.port == Port::none &&
              !tissue_nodes.count(id))
            issue(
              "boundary",
              id,
              "External boundary data require an explicitly declared port or unpaired tissue terminal.");
          if (paired_nodes.count(id))
            issue(
              "boundary",
              id,
              "A paired tissue node cannot also have an external boundary condition.");
        }
    for (const auto &[id, value] : conditions.pressures)
      if (conditions.inflows.count(id))
        issue("boundary",
              id,
              "Prescribe either pressure or flow at a node, never both.");
    if (!result.validation.valid())
      return result;
    using Index = types::global_dof_index;
    AssertThrow(network.nodes.size() < std::numeric_limits<Index>::max(),
                ExcMessage("Too many reference graph nodes."));
    std::vector<Id>     node_ids;
    std::map<Id, Index> indices;
    for (const auto &[id, node] : network.nodes)
      {
        indices[id] = node_ids.size();
        node_ids.push_back(id);
      }
    const Index n = node_ids.size();
    enum class Kind
    {
      vessel,
      microvascular
    };
    struct Resistor
    {
      std::array<Index, 2> nodes;
      double               resistance;
      Kind                 kind;
      Id                   id;
      double               demand;
    };
    std::vector<Resistor>           resistors;
    std::vector<std::vector<Index>> neighbors(n);
    const auto                      append = [&](Id     a,
                            Id     b,
                            double r,
                            Kind   kind,
                            Id     id,
                            double demand = 1.) {
      if (!std::isfinite(r) || r <= 0. || !std::isfinite(1. / r) ||
          1. / r <= 0.)
        {
          issue(
            "resistance",
            id,
            "Resistance and conductance must be positive and representable; no radius clipping is applied.");
          return;
        }
      const Index i = indices.at(a), j = indices.at(b);
      resistors.push_back({{{i, j}}, r, kind, id, demand});
      neighbors[i].push_back(j);
      neighbors[j].push_back(i);
    };
    for (const auto &[id, e] : network.vessels)
      append(e.nodes[0], e.nodes[1], resistance(network, id), Kind::vessel, id);
    for (const auto &pair : pairs)
      append(network.terminals.at(pair.arterial).node,
             network.terminals.at(pair.venous).node,
             pair.resistance,
             Kind::microvascular,
             pair.territory,
             network.terminals.at(pair.arterial).demand);
    if (!result.validation.valid())
      return result;
    std::vector<bool> visited(n, false);
    for (Index i = 0; i < n; ++i)
      {
        if (neighbors[i].empty())
          issue("node",
                node_ids[i],
                "An isolated node has no resistive reference equation.");
        if (visited[i])
          continue;
        std::vector<Index> component{i};
        visited[i]    = true;
        bool anchored = false;
        for (std::size_t k = 0; k < component.size(); ++k)
          {
            const Index j = component[k];
            if (!anchored && conditions.pressures.count(node_ids[j]))
              {
                anchored = true;
              }
            for (const Index next : neighbors[j])
              if (!visited[next])
                {
                  visited[next] = true;
                  component.push_back(next);
                }
          }
        if (!anchored)
          issue(
            "reference",
            node_ids[i],
            "Each connected reference component requires an explicit pressure anchor.");
      }
    if (!result.validation.valid())
      return result;
    // A deterministic multi-source lifting uses the nearest prescribed node
    // in graph distance. Vascular drops near an inlet/outlet are then computed
    // from small corrections, rather than subtracting two large pressures.
    std::fill(visited.begin(), visited.end(), false);
    std::vector<double> lifting(n, 0.);
    std::vector<Index>  frontier;
    for (const auto &[id, pressure] : conditions.pressures)
      {
        const Index i = indices.at(id);
        visited[i]    = true;
        lifting[i]    = pressure;
        frontier.push_back(i);
      }
    for (std::size_t k = 0; k < frontier.size(); ++k)
      for (const Index next : neighbors[frontier[k]])
        if (!visited[next])
          {
            visited[next] = true;
            lifting[next] = lifting[frontier[k]];
            frontier.push_back(next);
          }
    DynamicSparsityPattern pattern(n, n);
    for (Index i = 0; i < n; ++i)
      pattern.add(i, i);
    for (const auto &edge : resistors)
      {
        pattern.add(edge.nodes[0], edge.nodes[1]);
        pattern.add(edge.nodes[1], edge.nodes[0]);
      }
    SparsityPattern sparsity;
    sparsity.copy_from(pattern);
    SparseMatrix<double> matrix(sparsity);
    for (const auto &edge : resistors)
      {
        const auto [i, j] = edge.nodes;
        const double g    = 1. / edge.resistance;
        matrix.add(i, i, g);
        matrix.add(j, j, g);
        matrix.add(i, j, -g);
        matrix.add(j, i, -g);
      }
    for (Index i = 0; i < n; ++i)
      if (!std::isfinite(matrix.diag_element(i)) ||
          matrix.diag_element(i) <= 0.)
        issue("node",
              node_ids[i],
              "Reference diagonal is not positive and finite.");
    Vector<double> rhs(n), solution(n);
    for (const auto &[id, q] : conditions.inflows)
      rhs[indices.at(id)] = q;
    for (const auto &edge : resistors)
      {
        const auto [i, j]   = edge.nodes;
        const double source = (lifting[j] - lifting[i]) / edge.resistance;
        if (!std::isfinite(source))
          issue("pressure",
                edge.id,
                "Lifted pressure forcing is not representable.");
        rhs[i] += source;
        rhs[j] -= source;
      }
    for (Index i = 0; i < n; ++i)
      if (!std::isfinite(rhs[i]))
        issue("node",
              node_ids[i],
              "Accumulated reference forcing is not representable.");
    std::map<Index, double> boundary;
    for (const auto &[id, p] : conditions.pressures)
      boundary[indices.at(id)] = 0.;
    if (!result.validation.valid())
      return result;
    MatrixTools::apply_boundary_values(boundary, matrix, solution, rhs, true);
    ReductionControl         control(parameters.maximum_iterations,
                             0.,
                             parameters.linear_relative_tolerance,
                             false,
                             false);
    SolverCG<Vector<double>> solver(control);
    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(matrix);
    try
      {
        solver.solve(matrix, solution, rhs, preconditioner);
      }
    catch (const SolverControl::NoConvergence &)
      {
        issue("reference",
              invalid_id,
              "Reference linear solve exhausted its convergence budget.");
      }
    result.iterations      = control.last_step();
    result.linear_residual = control.last_value();
    for (Index i = 0; i < n; ++i)
      {
        const double p = conditions.pressures.count(node_ids[i]) ?
                           conditions.pressures.at(node_ids[i]) :
                           solution[i] + lifting[i];
        if (!std::isfinite(p))
          issue("pressure", node_ids[i], "Reference pressure is not finite.");
        result.pressures[node_ids[i]] = p;
      }
    if (!std::isfinite(solution.l2_norm()))
      {
        issue("reference",
              invalid_id,
              "Reference excess pressures are not finite.");
        return result;
      }
    std::vector<double> outflow(n, 0.), vascular_outflow(n, 0.);
    double              flow_scale = 0.;
    for (const auto &edge : resistors)
      {
        const auto [i, j] = edge.nodes;
        const double drop =
          (solution[i] - solution[j]) + (lifting[i] - lifting[j]);
        const double q     = drop / edge.resistance;
        const double power = q * drop;
        if (!std::isfinite(q) || !std::isfinite(power))
          {
            issue("reference",
                  edge.id,
                  "Reference flow or dissipation is not representable.");
            return result;
          }
        outflow[i] += q;
        outflow[j] -= q;
        flow_scale = std::max(flow_scale, std::abs(q));
        result.dissipated_power += power;
        if (edge.kind == Kind::vessel)
          {
            const auto     &e = network.vessels.at(edge.id);
            VesselReference state;
            state.resistance = edge.resistance;
            state.flow       = q;
            for (unsigned int end = 0; end < 2; ++end)
              {
                state.wall_shear[end] =
                  4. * e.viscosity * std::abs(q) /
                  (numbers::PI * std::pow(e.radii[end], 3));
                if (!std::isfinite(state.wall_shear[end]))
                  issue("vessel",
                        edge.id,
                        "Reference wall shear is not representable.");
              }
            result.vessels[edge.id] = state;
            vascular_outflow[i] += q;
            vascular_outflow[j] -= q;
          }
        else if (edge.kind == Kind::microvascular)
          {
            result.territories[edge.id] = {q, q / edge.demand};
            if (!std::isfinite(q / edge.demand))
              issue("territory",
                    edge.id,
                    "Flow per tissue demand is not representable.");
            if (!(q > 0.))
              issue(
                "territory",
                edge.id,
                "Microvascular perfusion must be positive from artery to vein.");
          }
      }
    for (const auto &[id, t] : network.terminals)
      result.terminal_outflows[id] = -vascular_outflow[indices.at(t.node)];
    for (const auto &[id, q] : conditions.inflows)
      flow_scale = std::max(flow_scale, std::abs(q));
    for (Index i = 0; i < n; ++i)
      {
        const Id id = node_ids[i];
        if (!std::isfinite(outflow[i]))
          {
            issue("node",
                  id,
                  "Accumulated reference flow is not representable.");
            return result;
          }
        if (conditions.pressures.count(id))
          result.boundary_inflows[id] = outflow[i];
        else
          {
            const double prescribed =
              conditions.inflows.count(id) ? conditions.inflows.at(id) : 0.;
            result.maximum_flow_defect =
              std::max(result.maximum_flow_defect,
                       std::abs(outflow[i] - prescribed));
            if (conditions.inflows.count(id))
              result.boundary_inflows[id] = prescribed;
          }
      }
    for (const auto &[id, q] : result.boundary_inflows)
      {
        result.boundary_flow_balance += q;
        flow_scale = std::max(flow_scale, std::abs(q));
      }
    result.maximum_flow_defect =
      std::max(result.maximum_flow_defect,
               std::abs(result.boundary_flow_balance));
    result.relative_flow_defect =
      flow_scale > 0. ? result.maximum_flow_defect / flow_scale : 0.;
    const double allowance =
      parameters.absolute_flow_tolerance +
      parameters.conservation_relative_tolerance * flow_scale;
    if (result.maximum_flow_defect > allowance)
      issue(
        "reference",
        invalid_id,
        "Independent physical flow balance exceeds its requested tolerance.");
    for (const auto &[id, state] : result.vessels)
      {
        const auto system = network.vessels.at(id).system;
        if ((system == System::arterial && state.flow < -allowance) ||
            (system == System::venous && state.flow > allowance))
          issue(
            "vessel",
            id,
            "Reference flow opposes the declared arterial/venous orientation.");
      }
    if (!std::isfinite(result.boundary_flow_balance) ||
        !std::isfinite(result.maximum_flow_defect) ||
        !std::isfinite(result.relative_flow_defect))
      issue("reference",
            invalid_id,
            "Reference flow diagnostics are not representable.");
    if (!std::isfinite(result.dissipated_power))
      issue("reference",
            invalid_id,
            "Total reference dissipation is not representable.");
    return result;
  }

  ReferenceConditions
  demand_flow_conditions(const NetworkData          &network,
                         const std::map<Id, double> &port_pressures,
                         double                      total_flow,
                         const std::vector<Id>      &selected_terminals,
                         MPI_Comm                    communicator)
  {
    using namespace dealii;
    AssertThrow(Utilities::MPI::n_mpi_processes(communicator) == 1,
                ExcMessage(
                  "The current demand-flow construction requires one rank."));
    AssertThrow(
      validate_data(network, 1).valid() && std::isfinite(total_flow) &&
        total_flow > 0.,
      ExcMessage(
        "Demand-flow construction requires valid physical data and a positive finite total flow."));
    ReferenceConditions result;
    for (const auto &[id, p] : port_pressures)
      {
        const auto node = network.nodes.find(id);
        AssertThrow(
          node != network.nodes.end() && node->second.port != Port::none &&
            std::isfinite(p),
          ExcMessage(
            "Demand-flow pressure data require explicit physical port nodes and finite pressures."));
        result.pressures[id] = p;
      }
    std::vector<Id> selected = selected_terminals;
    if (selected.empty())
      for (const auto &[id, t] : network.terminals)
        selected.push_back(id);
    AssertThrow(
      !selected.empty(),
      ExcMessage(
        "Demand-flow construction requires at least one selected tissue terminal."));
    std::set<Id>          unique;
    std::array<double, 2> scale{{0., 0.}}, sum{{0., 0.}};
    const auto            system_index = [](System system) {
      if (system == System::arterial)
        return 0u;
      else if (system == System::venous)
        return 1u;
      AssertThrow(false, ExcMessage("Unknown demand-flow system."));
      return 0u;
    };
    for (Id id : selected)
      {
        AssertThrow(
          network.terminals.count(id) && unique.insert(id).second,
          ExcMessage(
            "Selected demand-flow terminal ids must exist and be unique."));
        const auto &t = network.terminals.at(id);
        const auto  i = system_index(t.system);
        scale[i]      = std::max(scale[i], t.demand);
      }
    for (Id id : selected)
      {
        const auto &t = network.terminals.at(id);
        const auto  i = system_index(t.system);
        sum[i] += t.demand / scale[i];
      }
    for (Id id : selected)
      {
        const auto  &t = network.terminals.at(id);
        const auto   i = system_index(t.system);
        const double q = total_flow * ((t.demand / scale[i]) / sum[i]);
        AssertThrow(
          std::isfinite(q) && q > 0.,
          ExcMessage(
            "Desired tissue flow is not representable; no demand clipping is applied."));
        double signed_flow = 0.;
        if (t.system == System::arterial)
          signed_flow = -q;
        else if (t.system == System::venous)
          signed_flow = q;
        AssertThrow(
          result.inflows.emplace(t.node, signed_flow).second,
          ExcMessage(
            "A demand-flow terminal node must have one terminal identity."));
      }
    return result;
  }

  double
  reference_pressure(const NetworkData     &network,
                     const ReferenceResult &reference,
                     Id                     vessel,
                     double                 coordinate)
  {
    AssertThrow(reference.validation.valid(),
                dealii::ExcMessage(
                  "Cannot evaluate an invalid reference state."));
    AssertThrow(
      std::isfinite(coordinate) && coordinate >= 0. && coordinate <= 1.,
      dealii::ExcMessage("Reference pressure coordinate must lie in [0,1]."));
    const auto &edge = network.vessels.at(vessel);
    if (coordinate == 0.)
      return reference.pressures.at(edge.nodes[0]);
    if (coordinate == 1.)
      return reference.pressures.at(edge.nodes[1]);
    return reference.pressures.at(edge.nodes[0]) -
           reference.vessels.at(vessel).flow *
             resistance(network, vessel, 0., coordinate);
  }
} // namespace ImmersX::Vascularization
