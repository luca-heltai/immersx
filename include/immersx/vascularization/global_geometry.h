// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_global_geometry_h
#define immersx_vascularization_global_geometry_h

#include <deal.II/base/mpi.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <immersx/vascularization/network.h>

namespace ImmersX::Vascularization
{
  /** Passive SI controls for the fixed-flow, joint forest geometry problem. */
  struct GlobalGeometryParameters
  {
    void
    validate() const;

    /** Metabolic power density in W/m^3; callers supply their physical values.
     */
    std::map<System, double> metabolic_demand_W_m3 = {{System::arterial, 1.},
                                                      {System::portal, 1.},
                                                      {System::venous, 1.}};
    double       minimum_length_m = 0., maximum_length_m = 0., clearance_m = 0.;
    unsigned int maximum_outer_iterations = 20, maximum_inner_iterations = 1000;
    unsigned int geometry_depth = 18, geometry_work = 100000;
    /** Dimensionless tolerances after length and objective scaling. */
    double constraint_tolerance = 1.e-8, gradient_tolerance = 1.e-8;
  };

  struct GlobalGeometryIteration
  {
    unsigned int  outer_iteration = 0, inner_iterations = 0;
    std::uint64_t collision_constraints = 0, boundary_constraints = 0;
    double        objective_W = 0., maximum_constraint_violation = 0.;
    int           backend_reason = 0;
  };

  struct GlobalGeometryReport
  {
    bool          backend_available = false, certified = false;
    int           backend_reason   = 0;
    unsigned int  outer_iterations = 0, inner_iterations = 0;
    std::uint64_t inserted_nodes = 0, collision_constraints = 0,
                  boundary_constraints = 0;
    double objective_W = 0., maximum_constraint_violation = 0.;
    std::vector<GlobalGeometryIteration> history;
  };

  struct GlobalGeometryResult
  {
    NetworkData network;
    /** Positive downstream growth-tree flux in m^3/s, on vessel owners only.
     * Venous physical flow has the opposite orientation. Splits inherit Q.
     */
    std::map<Id, double> design_flows;
    ValidationReport     validation;
    GlobalGeometryReport report;
  };

  /**
   * Minimize joint metabolic plus Poiseuille power at fixed positive flows.
   * Every original node and vessel stays unchanged, regardless of provenance.
   * Candidate terminal positions also stay fixed. Only new interior nodes move;
   * additional degree-two nodes can resolve wall and nonincident collisions.
   * Input and returned graphs use ordinary native owned/ghost semantics.
   * The graph is never gathered. Every rank participates, including empty ones.
   *
   * Acceptance requires positive TAO convergence reason, auxiliary equality and
   * bound feasibility, and independent native wall/separation certificates.
   * Unsupported backends and exhausted/infeasible solves return an invalid
   * diagnostic result; they do not mutate either supplied graph. PETSc types
   * are deliberately absent from this interface.
   */
  GlobalGeometryResult
  optimize_forest_geometry(const dealii::Triangulation<3> &background,
                           const dealii::Mapping<3>       &mapping,
                           const NetworkData              &original,
                           const NetworkData              &candidate,
                           const std::map<Id, double>     &owned_design_flows,
                           const GlobalGeometryParameters &controls,
                           MPI_Comm                        communicator);
} // namespace ImmersX::Vascularization

#endif
