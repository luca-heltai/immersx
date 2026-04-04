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
#include <sstream>
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
  add_parameter("Rhs material ids", rhs_material_ids);
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
                  Patterns::Selection(
                    "fixed_fraction|fixed_number|global|inclusions"));
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
    auto reset_function = [this]() {
      this->prm.declare_entry(
        "Function expression",
        (spacedim == 2 ? "0; 0" : "0; 0; 0"),
        Patterns::List(Patterns::Anything(), spacedim, spacedim, ";"));
    };

    exact_solution.declare_parameters_call_back.connect(reset_function);
    exact_solution.declare_parameters_call_back.connect([this]() {
      this->prm.add_parameter("Weight expression", weight_expression);
    });

    initial_displacement.declare_parameters_call_back.connect(reset_function);
    initial_velocity.declare_parameters_call_back.connect(reset_function);
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

    bool created_dynamic_acceptors = false;

    const auto split_subsection_path = [](const std::string &path) {
      std::vector<std::string> parts;
      std::stringstream        stream(path);
      std::string              part;
      while (std::getline(stream, part, '/'))
        if (!part.empty())
          parts.emplace_back(part);
      return parts;
    };

    struct SubsectionScope
    {
      SubsectionScope(ParameterHandler               &prm,
                      const std::vector<std::string> &subsections)
        : prm(prm)
        , n_subsections(subsections.size())
      {
        for (const auto &subsection : subsections)
          prm.enter_subsection(subsection);
      }

      ~SubsectionScope()
      {
        for (unsigned int i = 0; i < n_subsections; ++i)
          prm.leave_subsection();
      }

      ParameterHandler &prm;
      unsigned int      n_subsections;
    };

    const auto get_entry_from_subsection =
      [this, &split_subsection_path](const std::string &path,
                                     const std::string &entry) {
        const SubsectionScope scope(this->prm, split_subsection_path(path));
        return this->prm.get(entry);
      };

    const auto set_entry_in_subsection =
      [this, &split_subsection_path](const std::string &path,
                                     const std::string &entry,
                                     const std::string &value) {
        const SubsectionScope scope(this->prm, split_subsection_path(path));
        this->prm.set(entry, value);
      };

    const auto copy_modulated_function_entries =
      [&get_entry_from_subsection,
       &set_entry_in_subsection](const std::string &from_section,
                                 const std::string &to_section) {
        for (const auto &entry : {"Function constants",
                                  "Function expression",
                                  "Variable names",
                                  "Modulation frequency",
                                  "Phase shift"})
          set_entry_in_subsection(
            to_section, entry, get_entry_from_subsection(from_section, entry));
      };

    const auto ensure_function_overrides =
      [this, &created_dynamic_acceptors, &copy_modulated_function_entries](
        const auto &ids, auto &map, const std::string &prefix) {
        for (const auto id : ids)
          if (map.find(id) == map.end())
            {
              const auto override_section =
                prefix + " " + std::to_string(static_cast<unsigned int>(id));
              auto ptr = std::make_shared<ModulatedParsedFunction<spacedim>>(
                override_section, spacedim);
              ptr->enter_my_subsection(this->prm);
              ptr->declare_parameters(this->prm);
              ptr->leave_my_subsection(this->prm);
              copy_modulated_function_entries(prefix, override_section);
              map[id]                   = ptr;
              created_dynamic_acceptors = true;
            }
      };

    {
      this->leave_my_subsection(this->prm);

      ensure_function_overrides(rhs_material_ids,
                                rhs_by_material_id,
                                "/Functions/Right hand side");

      std::set<types::boundary_id> dirichlet_bc_ids = dirichlet_ids;
      dirichlet_bc_ids.insert(weak_dirichlet_ids.begin(),
                              weak_dirichlet_ids.end());
      ensure_function_overrides(dirichlet_bc_ids,
                                dirichlet_bc_by_id,
                                "/Functions/Dirichlet boundary conditions");

      std::set<types::boundary_id> neumann_bc_ids = neumann_ids;
      neumann_bc_ids.insert(normal_flux_ids.begin(), normal_flux_ids.end());
      ensure_function_overrides(neumann_bc_ids,
                                neumann_bc_by_id,
                                "/Functions/Neumann boundary conditions");

      this->enter_my_subsection(this->prm);
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

    if (created_dynamic_acceptors)
      return;

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

template <int dim, int spacedim>
const ModulatedParsedFunction<spacedim> &
ElasticityProblemParameters<dim, spacedim>::get_dirichlet_bc(
  const types::boundary_id boundary_id) const
{
  const auto it = dirichlet_bc_by_id.find(boundary_id);
  if (it != dirichlet_bc_by_id.end())
    return *(it->second);

  return bc;
}

template <int dim, int spacedim>
const ModulatedParsedFunction<spacedim> &
ElasticityProblemParameters<dim, spacedim>::get_neumann_bc(
  const types::boundary_id boundary_id) const
{
  const auto it = neumann_bc_by_id.find(boundary_id);
  if (it != neumann_bc_by_id.end())
    return *(it->second);

  return Neumann_bc;
}

template <int dim, int spacedim>
const ModulatedParsedFunction<spacedim> &
ElasticityProblemParameters<dim, spacedim>::get_rhs(
  const types::material_id material_id) const
{
  const auto it = rhs_by_material_id.find(material_id);
  if (it != rhs_by_material_id.end())
    return *(it->second);

  return rhs;
}

template <int dim, int spacedim>
void
ElasticityProblemParameters<dim, spacedim>::set_rhs_times(
  const double time) const
{
  rhs.set_time(time);
  for (const auto &[id, ptr] : rhs_by_material_id)
    {
      (void)id;
      ptr->set_time(time);
    }
}

template <int dim, int spacedim>
void
ElasticityProblemParameters<dim, spacedim>::set_boundary_condition_times(
  const double time) const
{
  bc.set_time(time);
  for (const auto &[id, ptr] : dirichlet_bc_by_id)
    {
      (void)id;
      ptr->set_time(time);
    }

  Neumann_bc.set_time(time);
  for (const auto &[id, ptr] : neumann_bc_by_id)
    {
      (void)id;
      ptr->set_time(time);
    }
}


template class ElasticityProblemParameters<2>;
template class ElasticityProblemParameters<2, 3>; // dim != spacedim
template class ElasticityProblemParameters<3>;
