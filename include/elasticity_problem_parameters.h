// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at the top
// level of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

#ifndef dealii_distributed_lagrange_multiplier_elasticity_problem_parameters_h
#define dealii_distributed_lagrange_multiplier_elasticity_problem_parameters_h

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/solver_control.h>

#include <map>
#include <memory>
#include <set>
#include <string>

#include "material_properties.h"
#include "modulated_parsed_function.h"

/**
 * Time integration mode inferred from user parameters and material densities.
 *
 * - `Static`: stationary solve (no time loop), selected when
 *   `initial_time == final_time`.
 * - `QuasiStatic`: time-dependent loading without inertia, selected when
 *   `initial_time != final_time` and all materials have `rho == 0`.
 * - `Dynamic`: time-dependent loading with inertia, selected when
 *   `initial_time != final_time` and all materials have `rho > 0`.
 */
enum class TimeMode
{
  Static,
  QuasiStatic,
  Dynamic
};

/**
 * Bulk constitutive model inferred from material parameters.
 *
 * This replaces the previous boolean model flags with a single, validated
 * choice.
 */
enum class ElasticityModel
{
  LinearElasticity,
  KelvinVoigt
};

template <int dim, int spacedim = dim>
/**
 * Parameter set driving mesh generation, constitutive model, and solvers.
 */
class ElasticityProblemParameters : public ParameterAcceptor
{
public:
  /**
   * Build and register all parameter subsections used by the elasticity model.
   */
  ElasticityProblemParameters();

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

  /**
   * Check model consistency and infer derived modes after parameter parsing.
   *
   * This performs the requested consistency checks and sets:
   * - `time_mode` based on `{initial_time, final_time}` and `rho` across all
   *   materials;
   * - `elasticity_model` based on `neta` across all materials;
   *
   * This is called automatically from the parse callback once all material
   * subsections have been parsed (i.e., in the second pass when material tags
   * are used).
   */
  void
  check_model_consistency();

  /**
   * Inferred time mode (set during parameter parsing).
   *
   * @note Parameters are parsed using a two-pass strategy (see
   * `initialize_parameters()` in `include/utils.h`). We infer modes only after
   * pass >= 2 so that dynamically-created `MaterialProperties` acceptors have
   * been parsed.
   */
  TimeMode time_mode = TimeMode::Static;

  /**
   * Inferred bulk constitutive model (set during parameter parsing).
   */
  ElasticityModel elasticity_model = ElasticityModel::LinearElasticity;

  /**
   * Output, mesh, and finite-element setup parameters.
   */
  /// @{
  std::string                  output_directory = "."; ///< Output folder.
  std::string                  output_name      = "solution"; ///< Output stem.
  unsigned int                 fe_degree        = 1;          ///< FE degree.
  unsigned int                 initial_refinement = 5; ///< Global refinements.
  std::set<types::boundary_id> dirichlet_ids{0};     ///< Strong Dirichlet ids.
  std::set<types::boundary_id> weak_dirichlet_ids{}; ///< Weak Dirichlet ids.
  std::set<types::boundary_id> neumann_ids{};        ///< Neumann ids.
  std::set<types::boundary_id> normal_flux_ids{};    ///< Flux-constraint ids.

  MaterialProperties default_material_properties{
    "default"}; ///< Fallback material.
  std::map<types::material_id, std::string>
    material_tags_by_material_id; ///< Id->tag map.

  std::map<types::material_id, std::unique_ptr<MaterialProperties>>
    material_properties_by_id;                     ///< Runtime material table.
  std::set<types::material_id> rhs_material_ids{}; ///< Material-specific rhs.

  std::string domain_type        = "generate";    ///< Grid source mode.
  std::string triangulation_type = "distributed"; ///< Parallel tria backend.
  std::string name_of_grid       = "hyper_cube"; ///< Grid generator/input name.
  std::string arguments_for_grid =
    "-1: 1: false";              ///< Grid generator arguments.
  double       grid_scale = 1.0; ///< Uniform scaling applied after grid input.
  std::string  refinement_strategy = "fixed_fraction"; ///< Adaptivity strategy.
  double       coarsening_fraction = 0.0;              ///< Coarsening fraction.
  double       refinement_fraction = 0.3;              ///< Refinement fraction.
  unsigned int n_refinement_cycles = 1;     ///< Number of adapt cycles.
  unsigned int max_cells           = 20000; ///< Global cell cap.
  bool         output_pressure     = false; ///< Enable pressure output.
  double       penalty_term        = 1.0e4; ///< Weak-Dirichlet penalty.
  /// @}

  /**
   * Toggle load transfer through inclusion pressure projection.
   */
  bool pressure_coupling = false;

  /**
   * Volumetric forcing and forcing modulation.
   */
  /// @{
  mutable ModulatedParsedFunction<spacedim> rhs;
  std::map<types::material_id,
           std::shared_ptr<ModulatedParsedFunction<spacedim>>>
    rhs_by_material_id;
  /// @}

  /**
   * Exact solution used for error postprocessing.
   */
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    exact_solution;

  /**
   * Initial displacement for time-dependent runs.
   */
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    initial_displacement;

  /**
   * Initial velocity for time-dependent runs.
   */
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    initial_velocity;

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

  /**
   * Expression used to define quadrature weights on inclusions.
   */
  std::string weight_expression = "1.";

  /**
   * Solver controls for different linear blocks.
   */
  /// @{
  mutable ParameterAcceptorProxy<ReductionControl> displacement_solver_control;

  mutable ParameterAcceptorProxy<ReductionControl>
    reduced_mass_solver_control; ///< Mass solve control.

  mutable ParameterAcceptorProxy<ReductionControl>
    augmented_lagrange_solver_control; ///< Augmented-Lagrangian solve control.

  mutable ParameterAcceptorProxy<ReductionControl>
    schur_complement_solver_control; ///< Schur-complement solve control.
  /// @}

  /**
   * Emit output before each solve step.
   */
  bool output_results_before_solving = false;

  /**
   * Convergence table used for run summaries.
   */
  mutable ParsedConvergenceTable convergence_table;

  /**
   * Time-integration parameters.
   */
  /// @{
  double         initial_time     = 0.0;   ///< Initial physical time.
  double         final_time       = 0.0;   ///< Final physical time.
  mutable double dt               = 5e-3;  ///< Time-step size.
  bool           refine_time_step = false; ///< Enable adaptive time step.
  double         beta             = 0.25;  ///< Newmark beta.
  double         gamma            = 0.5;   ///< Newmark gamma.
  /// @}
};

#endif
