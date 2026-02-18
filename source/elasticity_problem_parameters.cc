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

#include "elasticity_problem_parameters.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <filesystem>
#include <functional>
#include <system_error>
#include <vector>


template <int dim, int spacedim>
ElasticityProblemParameters<dim, spacedim>::ElasticityProblemParameters()
  : ParameterAcceptor("/Immersed Problem/")
  , rhs("/Functions/Right hand side", spacedim)
  , exact_solution("/Functions/Exact solution", spacedim)
  , initial_displacement("/Functions/Initial displacement", spacedim)
  , initial_velocity("/Functions/Initial velocity", spacedim)
  , bc("/Functions/Dirichlet boundary conditions", spacedim)
  , Neumann_bc("/Functions/Neumann boundary conditions", spacedim)
  , displacement_solver_control("/Solvers/Displacement")
  , reduced_mass_solver_control("/Solvers/Reduced mass")
  , augmented_lagrange_solver_control("/Solvers/Augmented Lagrange")
  , schur_complement_solver_control("/Solvers/Schur complement")
  , convergence_table(std::vector<std::string>(spacedim, "u"))
{
  add_parameter("FE degree", fe_degree, "", this->prm, Patterns::Integer(1));
  add_parameter("Output directory", output_directory);
  add_parameter("Output name", output_name);
  add_parameter("Output results also before solving",
                output_results_before_solving);
  add_parameter("Initial refinement", initial_refinement);
  add_parameter("Dirichlet boundary ids", dirichlet_ids);
  add_parameter("Weak Dirichlet boundary ids", weak_dirichlet_ids);
  add_parameter("Weak Dirichlet penalty coefficient", penalty_term);
  add_parameter("Neumann boundary ids", neumann_ids);
  add_parameter("Normal flux boundary ids", normal_flux_ids);
  add_parameter("Output pressure", output_pressure);
  add_parameter(
    "Pressure coupling",
    pressure_coupling,
    "If this is true, then we do NOT solve a saddle point problem, but we use the "
    "input data as a pressure field on the vasculature network, and we solve for "
    "the displacement field directly.");
  enter_subsection("Grid generation");
  {
    add_parameter("Domain type",
                  domain_type,
                  "",
                  this->prm,
                  Patterns::Selection("generate|file|cheese|cylinder"));
    add_parameter("Grid generator", name_of_grid);
    add_parameter("Grid generator arguments", arguments_for_grid);
  }
  leave_subsection();
  enter_subsection("Refinement and remeshing");
  {
    add_parameter("Strategy",
                  refinement_strategy,
                  "",
                  this->prm,
                  Patterns::Selection("fixed_fraction|fixed_number|global"));
    add_parameter("Coarsening fraction", coarsening_fraction);
    add_parameter("Refinement fraction", refinement_fraction);
    add_parameter("Maximum number of cells", max_cells);
    add_parameter("Number of refinement cycles", n_refinement_cycles);
  }
  leave_subsection();

  enter_subsection("Material properties");
  add_parameter("Material tags by material id", material_tags_by_material_id);
  leave_subsection();
  enter_subsection("Time dependency");
  {
    add_parameter("Initial time", initial_time);
    add_parameter("Final time", final_time);
    add_parameter("Time step", dt);
    add_parameter("Refine time step", refine_time_step);
    add_parameter("Newmark beta", beta);
    add_parameter("Newmark gamma", gamma);
  }
  leave_subsection();

  this->prm.enter_subsection("Error");
  convergence_table.add_parameters(this->prm);
  this->prm.leave_subsection();

  // Make sure all functions have reasonable defaults, and add their modulation
  // frequencies
  {
    const auto set_modulation = [this](auto &acceptor, auto &arg) {
      acceptor.declare_parameters_call_back.connect([this, &arg]() {
        this->prm.add_parameter("Modulation frequency", arg);
      });
    };

    auto reset_function = [this]() {
      this->prm.declare_entry(
        "Function expression",
        (spacedim == 2 ? "0; 0" : "0; 0; 0"),
        Patterns::List(Patterns::Anything(), spacedim, spacedim, ";"));
    };
    rhs.declare_parameters_call_back.connect(reset_function);
    set_modulation(rhs, rhs_modulation);

    exact_solution.declare_parameters_call_back.connect(reset_function);
    exact_solution.declare_parameters_call_back.connect([this]() {
      this->prm.add_parameter("Weight expression", weight_expression);
    });

    Neumann_bc.declare_parameters_call_back.connect(reset_function);
    set_modulation(Neumann_bc, neumann_bc_modulation);

    initial_displacement.declare_parameters_call_back.connect(reset_function);
    initial_velocity.declare_parameters_call_back.connect(reset_function);

    bc.declare_parameters_call_back.connect(reset_function);
    set_modulation(bc, bc_modulation);
  }
  {
    auto reduction = [&]() {
      this->prm.declare_entry("Reduction", "1.e-10", Patterns::Double(0));
    };
    displacement_solver_control.declare_parameters_call_back.connect(reduction);
    reduced_mass_solver_control.declare_parameters_call_back.connect(reduction);
    augmented_lagrange_solver_control.declare_parameters_call_back.connect(
      reduction);
    schur_complement_solver_control.declare_parameters_call_back.connect(
      reduction);
  }

  this->parse_parameters_call_back.connect([this]() {
    // Ensure output directory exists.
    {
      namespace fs = std::filesystem;
      std::error_code ec;
      fs::create_directories(output_directory, ec);
      AssertThrow(!ec,
                  ExcMessage("Could not create output directory '" +
                             output_directory + "': " + ec.message()));
    }

    // Ensure material properties acceptors exist when material tags are
    // provided. This is needed for the two-pass initialization strategy:
    // - pass 1: read tags, create acceptors
    // - pass 2: acceptors declare parameters and get parsed
    if (material_properties_by_id.empty() &&
        !material_tags_by_material_id.empty())
      {
        // Make sure we exit our subsection first, so that registration of the
        // new classes works properly.
        this->leave_my_subsection(this->prm);
        for (const auto &[material_id, tag] : material_tags_by_material_id)
          {
            material_properties_by_id[material_id] =
              std::make_unique<MaterialProperties>(tag);
          }
        // Go back to our subsection, so we can continue parsing parameters.
        this->enter_my_subsection(this->prm);
        // Do not check consistency yet: the dynamically created acceptors
        // will only be parsed in the second parameter pass.
        return;
      }
    else if (!material_tags_by_material_id.empty())
      {
        // Already initialized: check consistency.
        AssertDimension(material_properties_by_id.size(),
                        material_tags_by_material_id.size());
        // In the second pass, all material subsections have been parsed.
        check_model_consistency();
        return;
      }

    // No dynamic material tags: all information is already available.
    check_model_consistency();
  });
}


template <int dim, int spacedim>
void
ElasticityProblemParameters<dim, spacedim>::check_model_consistency()
{
  // --- Validate boundary id sets --------------------------------------------
  const auto check_disjoint = [](const char                         *name_a,
                                 const std::set<types::boundary_id> &a,
                                 const char                         *name_b,
                                 const std::set<types::boundary_id> &b) {
    for (const auto id : a)
      if (b.find(id) != b.end())
        AssertThrow(false,
                    ExcMessage(std::string("Boundary id ") +
                               std::to_string(static_cast<unsigned int>(id)) +
                               " appears in both '" + name_a + "' and '" +
                               name_b + "'. These sets must be disjoint."));
  };

  check_disjoint("dirichlet ids",
                 dirichlet_ids,
                 "weak dirichlet ids",
                 weak_dirichlet_ids);
  check_disjoint("dirichlet ids", dirichlet_ids, "neumann ids", neumann_ids);
  check_disjoint("dirichlet ids",
                 dirichlet_ids,
                 "normal flux ids",
                 normal_flux_ids);
  check_disjoint("weak dirichlet ids",
                 weak_dirichlet_ids,
                 "neumann ids",
                 neumann_ids);
  check_disjoint("weak dirichlet ids",
                 weak_dirichlet_ids,
                 "normal flux ids",
                 normal_flux_ids);
  check_disjoint("neumann ids",
                 neumann_ids,
                 "normal flux ids",
                 normal_flux_ids);

  // Collect the set of materials that may be used: always include the default
  // material, and also any explicitly configured material ids.
  std::vector<std::reference_wrapper<const MaterialProperties>> materials;
  materials.emplace_back(default_material_properties);
  for (const auto &[id, mp_ptr] : material_properties_by_id)
    {
      (void)id;
      Assert(mp_ptr != nullptr, ExcInternalError());
      materials.emplace_back(*mp_ptr);
    }

  // --- Infer TimeMode -------------------------------------------------------
  if (initial_time == final_time)
    {
      time_mode = TimeMode::Static;
    }
  else
    {
      bool any_rho_zero     = false;
      bool any_rho_positive = false;
      for (const auto &mp_ref : materials)
        {
          const auto &mp = mp_ref.get();
          AssertThrow(mp.rho >= 0.0,
                      ExcMessage("Material density (rho) must be >= 0."));
          any_rho_zero |= (mp.rho == 0.0);
          any_rho_positive |= (mp.rho > 0.0);
        }

      AssertThrow(!(any_rho_zero && any_rho_positive),
                  ExcMessage("Inconsistent densities: either all materials "
                             "must have rho == 0 (quasi-static) or all must "
                             "have rho > 0 (dynamic)."));

      time_mode = any_rho_positive ? TimeMode::Dynamic : TimeMode::QuasiStatic;
    }

  // --- Validate BCs for time-dependent problems ---------------------------
  if (time_mode != TimeMode::Static)
    {
      AssertThrow(dirichlet_ids.empty(),
                  ExcMessage("Time-dependent problems (initial_time != "
                             "final_time) do not support strong Dirichlet "
                             "boundary conditions. Use weak Dirichlet and/or "
                             "Neumann boundary conditions instead."));
      AssertThrow(normal_flux_ids.empty(),
                  ExcMessage("Time-dependent problems (initial_time != "
                             "final_time) do not support normal-flux "
                             "constraints. Use weak Dirichlet and/or "
                             "Neumann boundary conditions instead."));
    }

  // --- Infer ElasticityModel ------------------------------------------------
  bool any_neta_zero     = false;
  bool any_neta_positive = false;
  for (const auto &mp_ref : materials)
    {
      const auto &mp = mp_ref.get();
      AssertThrow(mp.neta >= 0.0,
                  ExcMessage("Material viscosity (eta/neta) must be >= 0."));
      any_neta_zero |= (mp.neta == 0.0);
      any_neta_positive |= (mp.neta > 0.0);
    }

  AssertThrow(!(any_neta_zero && any_neta_positive),
              ExcMessage("Inconsistent viscosities: either all materials must "
                         "have eta == 0 or all must have eta > 0."));

  if (any_neta_positive)
    elasticity_model = ElasticityModel::KelvinVoigt;
  else
    elasticity_model = ElasticityModel::LinearElasticity;
}


template <int dim, int spacedim>
inline const MaterialProperties &
ElasticityProblemParameters<dim, spacedim>::get_material_properties(
  const types::material_id material_id) const
{
  auto it = material_properties_by_id.find(material_id);
  if (it != material_properties_by_id.end())
    return *(it->second);
  else
    return default_material_properties;
}


template class ElasticityProblemParameters<2>;
template class ElasticityProblemParameters<2, 3>; // dim != spacedim
template class ElasticityProblemParameters<3>;
