// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_apps_metric_flow_x_physical_input_h
#define immersx_apps_metric_flow_x_physical_input_h

#include <immersx/core/fe_space.h>
#include <immersx/vascularization/metric_flow_x.h>

#include <cmath>
#include <set>
#include <vector>

namespace ImmersX::metric_flow_x_app_detail
{
  using namespace dealii;
  template <class Adapter, class Flow>
  auto
  set_native_flow_tolerances(Adapter      &adapter,
                             Flow         &flow,
                             const FieldId field,
                             int)
    -> decltype(flow.make_absolute_tolerances(), void())
  {
    adapter.set_local_tolerances(field, [&flow](auto &values) {
      values = flow.make_absolute_tolerances();
    });
  }

  template <class Adapter, class Flow>
  void
  set_native_flow_tolerances(Adapter &, Flow &, const FieldId, long)
  {
    AssertThrow(
      false,
      ExcMessage(
        "Native flow tolerances require a compatible MetricFlowX package."));
  }

  struct ArchivedFlowParameters : ParameterAcceptor
  {
    ArchivedFlowParameters()
      : ParameterAcceptor("/MetricFlowX coupling/Archived network/")
    {
      add_parameter(
        "Directory",
        directory,
        "Native physical network archive. Empty uses the ordinary VTK or generation path.");
      add_parameter(
        "Young moduli",
        young_moduli,
        "Wall Young moduli in Pa, in arterial, venous, portal order.");
      add_parameter(
        "Wall thickness ratios",
        thickness_ratios,
        "Wall thickness divided by mean reference radius, in arterial, venous, portal order.");
      add_parameter(
        "Reference pressures",
        reference_pressures,
        "Wall reference pressures in Pa, in arterial, venous, portal order.");
      add_parameter("Portal pressure",
                    portal_pressure,
                    "Prescribed portal root reservoir pressure in Pa.");
      add_parameter("Venous pressure",
                    venous_pressure,
                    "Prescribed venous root reservoir pressure in Pa.");
      add_parameter(
        "Root resistance",
        root_resistance,
        "Positive root reservoir resistance in Pa s/m^3; saved tissue compartments supply their own resistances.");
      add_parameter(
        "Sealed input terminal ids",
        sealed_input_terminals,
        "Physical Terminal ids explicitly closed to flow; every other tissue leaf must belong to a saved microvascular bed.");
      add_parameter("Redistribution halo",
                    halo,
                    "Physical archive halo in metres.");
    }

    std::vector<Vascularization::Id> sealed_input_terminals;
    std::string                      directory;
    std::vector<double>              young_moduli{0., 0., 0.};
    std::vector<double>              thickness_ratios{0., 0., 0.};
    std::vector<double>              reference_pressures{0., 0., 0.};
    double portal_pressure = 0., venous_pressure = 0., root_resistance = 0.,
           halo = 0.;
  };

  template <class Flow>
  Vascularization::MetricFlowXTransfer
  prepare_archived_flow(Flow                                    &flow,
                        const Vascularization::GenerationResult &generation,
                        const ArchivedFlowParameters            &parameters)
  {
    namespace V = Vascularization;
    AssertThrow(
      parameters.young_moduli.size() == 3 &&
        parameters.thickness_ratios.size() == 3 &&
        parameters.reference_pressures.size() == 3,
      ExcMessage(
        "Archived wall parameters require three values in arterial, venous, portal order."));
    for (unsigned int i = 0; i < 3; ++i)
      AssertThrow(
        std::isfinite(parameters.young_moduli[i]) &&
          parameters.young_moduli[i] > 0. &&
          std::isfinite(parameters.thickness_ratios[i]) &&
          parameters.thickness_ratios[i] > 0. &&
          std::isfinite(parameters.reference_pressures[i]),
        ExcMessage(
          "Archived wall material must be positive and reference pressure finite."));
    AssertThrow(
      std::isfinite(parameters.portal_pressure) &&
        std::isfinite(parameters.venous_pressure) &&
        std::isfinite(parameters.root_resistance) &&
        parameters.root_resistance > 0.,
      ExcMessage(
        "Archived root pressures must be finite and reservoir resistance positive."));

    // Unselected anatomical ends need an explicit zero-flow declaration.
    // Independent outlet resistances would add drains beside the saved beds.
    const std::set<V::Id> sealed(parameters.sealed_input_terminals.begin(),
                                 parameters.sealed_input_terminals.end());
    AssertThrow(sealed.size() == parameters.sealed_input_terminals.size(),
                ExcMessage("A sealed input terminal id is repeated."));
    std::set<V::Id>    linked;
    const auto         comm = flow.mpi_communicator();
    std::vector<V::Id> owned_linked;
    for (const auto &pair : generation.terminal_pairs)
      owned_linked.insert(owned_linked.end(), {pair.arterial, pair.venous});
    for (const auto &triplet : generation.terminal_triplets)
      owned_linked.insert(owned_linked.end(),
                          {triplet.arterial, triplet.portal, triplet.venous});
    for (const auto &ids : Utilities::MPI::all_gather(comm, owned_linked))
      linked.insert(ids.begin(), ids.end());
    unsigned int unlinked = 0;
    const auto   rank     = Utilities::MPI::this_mpi_process(comm);
    for (const auto &[id, terminal] : generation.network.terminals)
      if (terminal.owner == rank &&
          ((linked.count(id) == 0 && sealed.count(id) == 0) ||
           (linked.count(id) != 0 && sealed.count(id) != 0)))
        ++unlinked;
    AssertThrow(
      Utilities::MPI::sum(unlinked, comm) == 0,
      ExcMessage(
        "Every archived tissue leaf must have exactly one microvascular connection or explicit sealed declaration."));

    if constexpr (V::has_metric_flow_x_network_input<Flow>)
      return V::prepare_metric_flow_x(
        flow,
        generation,
        [&](V::Id, const V::Vessel &vessel) {
          unsigned int role = 0;
          if (vessel.system == V::System::venous)
            role = 1;
          else if (vessel.system == V::System::portal)
            role = 2;
          typename Flow::VesselProperties result;
          result.E      = parameters.young_moduli[role];
          result.h_wall = parameters.thickness_ratios[role] * .5 *
                          (vessel.radii[0] + vessel.radii[1]);
          result.p_d = parameters.reference_pressures[role];
          result.p0  = 0.;
          return result;
        },
        [&](V::Id, const V::Node &node, const V::Terminal *terminal) {
          // The native input validates positive R2 even at linked leaves;
          // those leaves use the saved resistor graph instead of this RCR.
          typename Flow::RCRPhysics result;
          result.R2 = parameters.root_resistance;
          if (terminal != nullptr)
            result.label = "microvascular-compartment";
          else if (node.port == V::Port::portal_inlet)
            {
              result.P_out = parameters.portal_pressure;
              result.label = "portal-root";
            }
          else if (node.port == V::Port::venous_outlet)
            {
              result.P_out = parameters.venous_pressure;
              result.label = "venous-root";
            }
          else
            AssertThrow(
              false, ExcMessage("Unsupported real port in archived network."));
          return result;
        },
        sealed);
    else
      {
        AssertThrow(
          false,
          ExcMessage(
            "Archived flow requires MetricFlowX NetworkPhysics input."));
        return {};
      }
  }

  template <class Flow, class Vector>
  void
  interpolate_physical_reference_area(
    const Flow                                 &flow,
    const Vascularization::MetricFlowXTransfer &transfer,
    const FiniteElementSpaceView<1, 3>         &view,
    Vector                                     &reference_area)
  {
    reference_area.reinit(view.locally_owned_dofs(),
                          view.locally_relevant_dofs(),
                          flow.mpi_communicator());
    // Trilinos ghosted vectors are read-only. Assemble owned values,
    // then import them into the field used by lifted evaluation.
    Vector owned_reference_area;
    owned_reference_area.reinit(view.locally_owned_dofs(),
                                flow.mpi_communicator());
    std::vector<types::global_dof_index> indices(
      view.finite_element().n_dofs_per_cell());
    for (const auto &cell : view.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(indices);
          const auto interval =
            Vascularization::physical_reference(cell->id(),
                                                transfer.flow_cells);
          const auto &properties = flow.vessel_properties(cell->material_id());
          for (unsigned int i = 0; i < indices.size(); ++i)
            {
              const double xi = cell->get_fe().unit_support_point(i)[0];
              const double t =
                interval.begin + xi * (interval.end - interval.begin);
              const double radius =
                properties.r_in == properties.r_out ?
                  properties.r_in :
                  (1. - t) * properties.r_in + t * properties.r_out;
              owned_reference_area[indices[i]] = numbers::PI * radius * radius;
            }
        }
    owned_reference_area.compress(VectorOperation::insert);
    reference_area = owned_reference_area;
    reference_area.update_ghost_values();
  }
} // namespace ImmersX::metric_flow_x_app_detail
#endif
