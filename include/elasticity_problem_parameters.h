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
class ElasticityProblemParameters : public ParameterAcceptor
{
public:
  ElasticityProblemParameters();

  const MaterialProperties &
  get_material_properties(const types::material_id material_id) const;

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

  std::string                  output_directory   = ".";
  std::string                  output_name        = "solution";
  unsigned int                 fe_degree          = 1;
  unsigned int                 initial_refinement = 5;
  std::set<types::boundary_id> dirichlet_ids{0};
  std::set<types::boundary_id> weak_dirichlet_ids{};
  std::set<types::boundary_id> neumann_ids{};
  std::set<types::boundary_id> normal_flux_ids{};

  MaterialProperties default_material_properties{"default"};
  std::map<types::material_id, std::string> material_tags_by_material_id;

  std::map<types::material_id, std::unique_ptr<MaterialProperties>>
    material_properties_by_id;

  std::string  domain_type         = "generate";
  std::string  name_of_grid        = "hyper_cube";
  std::string  arguments_for_grid  = "-1: 1: false";
  std::string  refinement_strategy = "fixed_fraction";
  double       coarsening_fraction = 0.0;
  double       refinement_fraction = 0.3;
  unsigned int n_refinement_cycles = 1;
  unsigned int max_cells           = 20000;
  bool         output_pressure     = false;
  double       penalty_term        = 1.0e4;

  bool pressure_coupling = false;

  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> rhs;
  double rhs_modulation = 0.0;

  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    exact_solution;

  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    initial_displacement;

  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    initial_velocity;

  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> bc;
  double bc_modulation = 0.0;

  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
         Neumann_bc;
  double neumann_bc_modulation = 0.0;

  std::string weight_expression = "1.";

  mutable ParameterAcceptorProxy<ReductionControl> displacement_solver_control;

  mutable ParameterAcceptorProxy<ReductionControl> reduced_mass_solver_control;

  mutable ParameterAcceptorProxy<ReductionControl>
    augmented_lagrange_solver_control;

  mutable ParameterAcceptorProxy<ReductionControl>
    schur_complement_solver_control;

  bool output_results_before_solving = false;

  mutable ParsedConvergenceTable convergence_table;

  // Time dependency.
  double         initial_time     = 0.0;
  double         final_time       = 0.0;
  mutable double dt               = 5e-3;
  bool           refine_time_step = false;
  double         beta             = 0.25;
  double         gamma            = 0.5;
};

#endif
