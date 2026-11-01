// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at the top
// level of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#ifndef immersx_elasticity_freq_problem_parameters_h
#define immersx_elasticity_freq_problem_parameters_h

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/solver_control.h>

#include <immersx/core/domain.h>
#include <immersx/physics/material_properties.h>
#include <immersx/physics/modulated_parsed_function.h>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace ImmersX
{
  template <int dim, int spacedim = dim>
  /**
   * Parameters for the direct frequency-domain elasticity solver.
   */
  class ElasticityFreqProblemParameters : public ParameterAcceptor
  {
    std::vector<double> owned_frequencies{100.0};

  public:
    /**
     * Build and register all parameter subsections used by the elasticity
     * model. A shared frequency vector is registered by its owner and must
     * outlive this parameter object.
     */
    explicit ElasticityFreqProblemParameters(
      const std::string   &subsection         = "/Immersed Problem/",
      std::vector<double> *shared_frequencies = nullptr);

    /**
     * Return material parameters associated with a material id.
     */
    const MaterialProperties &
    get_material_properties(const types::material_id material_id) const;

    const ModulatedParsedFunction<spacedim> &
    get_dirichlet_bc(const types::boundary_id boundary_id) const;

    const ModulatedParsedFunction<spacedim> &
    get_neumann_bc(const types::boundary_id boundary_id) const;

    const ModulatedParsedFunction<spacedim> &
    get_rhs(const types::material_id material_id) const;

    void
    set_rhs_times(const double time) const;

    void
    set_boundary_condition_times(const double time) const;

    /** Validate boundary and material data after parameter parsing. */
    void
    validate_parameters();

    /** Domain settings use the same grid subsection as the legacy solver. */
    DomainParameters<dim, spacedim> domain_parameters;

    std::vector<double> &frequencies;
    double dirichlet_phase = 0.0, neumann_phase = 0.0, rhs_phase = 0.0;
    std::map<types::boundary_id, double> dirichlet_phase_by_id,
      neumann_phase_by_id;
    std::map<types::material_id, double> rhs_phase_by_id;
    unsigned int                         gmres_basis_size = 200;

    /**
     * Output, mesh, and finite-element setup parameters.
     */
    /// @{
    std::string   output_directory = ".";        ///< Output folder.
    std::string   output_name      = "solution"; ///< Output stem.
    unsigned int  fe_degree        = 1;          ///< FE degree.
    unsigned int &initial_refinement =
      domain_parameters.initial_refinement;        ///< Global refinements.
    std::set<types::boundary_id> dirichlet_ids{0}; ///< Strong Dirichlet ids.
    std::set<types::boundary_id> weak_dirichlet_ids{}; ///< Weak Dirichlet ids.
    std::set<types::boundary_id> neumann_ids{};        ///< Neumann ids.
    std::set<types::boundary_id> normal_flux_ids{};    ///< Flux-constraint ids.

    MaterialProperties default_material_properties; ///< Fallback material.
    std::map<types::material_id, std::string>
      material_tags_by_material_id; ///< Id->tag map.

    std::map<types::material_id, std::unique_ptr<MaterialProperties>>
      material_properties_by_id; ///< Runtime material table.
    std::set<types::material_id> rhs_material_ids{}; ///< Material-specific rhs.

    std::string  domain_type = "generate"; ///< Grid source mode.
    std::string &triangulation_type =
      domain_parameters.triangulation_type; ///< Parallel tria backend.
    std::string &name_of_grid =
      domain_parameters.name_of_grid; ///< Grid generator/input name.
    std::string &arguments_for_grid =
      domain_parameters.arguments_for_grid; ///< Grid generator arguments.
    /// Uniform scaling applied after grid input.
    double     &grid_scale = domain_parameters.grid_scale;
    std::string refinement_strategy =
      "fixed_fraction";                       ///< Adaptivity strategy.
    double       coarsening_fraction = 0.0;   ///< Coarsening fraction.
    double       refinement_fraction = 0.3;   ///< Refinement fraction.
    unsigned int n_refinement_cycles = 1;     ///< Number of adapt cycles.
    unsigned int max_cells           = 20000; ///< Global cell cap.
    double       penalty_term        = 1.0e4; ///< Weak-Dirichlet penalty.
    /// @}

    /**
     * Volumetric forcing and forcing modulation.
     */
    /// @{
  private:
    const std::string function_prefix;
    const std::string material_prefix;

  public:
    mutable ModulatedParsedFunction<spacedim> rhs;
    std::map<types::material_id,
             std::shared_ptr<ModulatedParsedFunction<spacedim>>>
      rhs_by_material_id;
    /// @}

    /**
     * Dirichlet and Neumann boundary data with modulation factors.
     */
    /// @{
    mutable ModulatedParsedFunction<spacedim> bc;
    std::map<types::boundary_id,
             std::shared_ptr<ModulatedParsedFunction<spacedim>>>
      dirichlet_bc_by_id;

    mutable ModulatedParsedFunction<spacedim>
      Neumann_bc; ///< Neumann boundary data function.
    std::map<types::boundary_id,
             std::shared_ptr<ModulatedParsedFunction<spacedim>>>
      neumann_bc_by_id;
    /// @}

    /** Solver control for the frequency-domain FGMRES solve. */
    mutable ParameterAcceptorProxy<ReductionControl>
      displacement_solver_control;
  };

} // namespace ImmersX

#endif
