// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_reference_flow_h
#define immersx_vascularization_reference_flow_h

#include <deal.II/base/mpi.h>

#include <immersx/vascularization/network.h>

namespace ImmersX::Vascularization
{
  struct ReferenceConditions
  {
    /** Node id -> pressure in Pa; one pressure anchor per component is
     * required. */
    std::map<Id, double> pressures;
    /** Node id -> externally injected m^3/s; negative means extraction. */
    std::map<Id, double> inflows;
  };

  struct ReferenceParameters
  {
    void
    validate() const;

    unsigned int maximum_iterations              = 10000;
    double       linear_relative_tolerance       = 1.e-12;
    double       conservation_relative_tolerance = 1.e-9;
    /** m^3/s; no absolute flow allowance unless explicitly provided. */
    double absolute_flow_tolerance = 0.;
  };

  struct VesselReference
  {
    /** Pa s/m^3 on the complete physical vessel, including exterior parts. */
    double resistance = 0.;
    /** m^3/s in physical endpoint order; reference venous flow is negative. */
    double flow = 0.;
    /** Pa at the two endpoints: 4*mu*abs(Q)/(pi*r^3). */
    std::array<double, 2> wall_shear{{0., 0.}};
  };

  struct TerritoryReference
  {
    /** m^3/s from artery to vein through the specified 0D resistance. */
    double flow = 0.;
    /** m^3/s divided by dimensionless tissue demand, not by tissue volume. */
    double flow_per_demand = 0.;
  };

  struct ReferenceResult
  {
    ValidationReport validation;
    /** Physical node id -> pressure in Pa. */
    std::map<Id, double>             pressures;
    std::map<Id, VesselReference>    vessels;
    std::map<Id, TerritoryReference> territories;
    /** Net 1D outflow into tissue: positive arterial, negative venous. */
    std::map<Id, double> terminal_outflows;
    /** Prescribed fluxes or pressure reactions, positive into the network. */
    std::map<Id, double> boundary_inflows;
    double               dissipated_power      = 0.;
    double               maximum_flow_defect   = 0.;
    double               relative_flow_defect  = 0.;
    double               boundary_flow_balance = 0.;
    double               linear_residual       = 0.;
    unsigned int         iterations            = 0;
  };

  /**
   * Solve the steady Poiseuille reference problem on the physical graph,
   * including supplied microvascular pairs. Each component needs an explicit
   * pressure anchor; prescribing both pressure and flow at a node is invalid.
   * Boundary nodes must be declared ports or unpaired tissue terminals;
   * paired tissue nodes cannot also receive external boundary conditions.
   * Unspecified nodes satisfy Kirchhoff balance with zero external inflow.
   *
   * Uses native deal.II sparse algebra and CG/Jacobi. An independent physical
   * balance check follows the solve. Reversed vessel reference flow and
   * nonpositive paired perfusion invalidate the result; a zero arterial-only
   * equilibrium is allowed. Candidate evaluation changes no accepted state.
   * Current execution requires one rank. Physical identifiers, geometry,
   * taper, numerical subdivisions and all input attributes remain unchanged.
   */
  ReferenceResult
  solve_reference(const NetworkData               &network,
                  const std::vector<TerminalPair> &pairs,
                  const ReferenceConditions       &conditions,
                  const ReferenceParameters       &parameters,
                  MPI_Comm                         communicator);
  /**
   * Prescribe total_flow (m^3/s) through selected tissue terminals, distributed
   * by their relative demand. Empty selection means all terminals. Arterial
   * sinks and venous sources are normalized independently to the same total.
   * Only the supplied explicit ports receive prescribed pressures; their
   * flows are reactions, so no inlet receives both pressure and flow.
   * Unselected terminals are left for the caller's boundary conditions.
   * Current execution requires one rank.
   */
  ReferenceConditions
  demand_flow_conditions(const NetworkData          &network,
                         const std::map<Id, double> &port_pressures,
                         double                      total_flow,
                         const std::vector<Id>      &selected_terminals,
                         MPI_Comm                    communicator);

  /** Exact Poiseuille pressure profile on a physical reference interval. */
  double
  reference_pressure(const NetworkData     &network,
                     const ReferenceResult &reference,
                     Id                     vessel,
                     double                 coordinate);
} // namespace ImmersX::Vascularization
#endif
