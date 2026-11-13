// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>


namespace ImmersX
{
  using namespace dealii;


  namespace
  {
    std::string
    normalize_elastodynamics_subsection(const std::string &subsection)
    {
      if (subsection.empty())
        return "/Elastodynamics/";

      std::string normalized = subsection;
      if (normalized.front() != '/')
        normalized.insert(normalized.begin(), '/');
      if (normalized.back() != '/')
        normalized.push_back('/');
      return normalized;
    }

    void
    ensure_output_directory(const std::string &directory)
    {
      namespace fs = std::filesystem;
      std::error_code ec;
      fs::create_directories(directory, ec);
      AssertThrow(!ec,
                  ExcMessage("Could not create output directory '" + directory +
                             "': " + ec.message()));
    }


    bool
    expression_is_zero(const ParameterHandler &prm)
    {
      for (const auto &component :
           Utilities::split_string_list(prm.get("Function expression"), ';'))
        {
          std::size_t parsed = 0;
          try
            {
              if (std::stod(component, &parsed) != 0. ||
                  parsed != component.size())
                return false;
            }
          catch (const std::invalid_argument &)
            {
              return false;
            }
          catch (const std::out_of_range &)
            {
              return false;
            }
        }
      return true;
    }

    template <typename VectorType>
    void
    project_rhs(const VectorType                &source,
                const AffineConstraints<double> &constraints,
                VectorType                      &destination)
    {
      destination = 0.;
      destination.compress(VectorOperation::add);
      for (const auto index : source.locally_owned_elements())
        {
          const double value = source(index);
          constraints.distribute_local_to_global(&value,
                                                 &value + 1,
                                                 &index,
                                                 destination);
        }
      destination.compress(VectorOperation::add);
    }

    template <int dim, int spacedim>
    void
    read_elastodynamics_grid(const std::string &grid_file_name,
                             const std::string &ids_and_cad_file_names,
                             Triangulation<dim, spacedim> &tria)
    {
      if constexpr (dim == 1 && spacedim == 1)
        {
          GridIn<dim, spacedim> grid_in;
          grid_in.attach_triangulation(tria);
          grid_in.read(grid_file_name);
        }
      else
        read_grid_and_cad_files(grid_file_name, ids_and_cad_file_names, tria);
    }
  } // namespace

  ElastodynamicsMaterialParameters::ElastodynamicsMaterialParameters(
    const std::string &subsection)
    : ParameterAcceptor(subsection)
  {
    add_parameter("Density", coefficients.density);
    add_parameter("Lame mu", coefficients.lame_mu);
    add_parameter("Lame lambda", coefficients.lame_lambda);


    add_parameter("Rayleigh alpha", coefficients.rayleigh_alpha);
    add_parameter("Rayleigh beta", coefficients.rayleigh_beta);
    add_parameter("Viscosity eta", coefficients.viscosity_eta);

    parse_parameters_call_back.connect([this]() {
      const auto &m = coefficients;

      AssertThrow(std::isfinite(m.density) && m.density > 0.,
                  ExcMessage("Material density must be finite and positive."));

      AssertThrow(std::isfinite(m.lame_mu) && m.lame_mu > 0.,
                  ExcMessage("Material Lame mu must be finite and positive."));

      for (const double value :
           {m.lame_lambda, m.rayleigh_alpha, m.rayleigh_beta, m.viscosity_eta})
        AssertThrow(std::isfinite(value) && value >= 0.,
                    ExcMessage("Material lambda and damping coefficients "
                               "must be finite and non-negative."));
    });
  }
  template <int dim, int spacedim>
  ElastodynamicsParameters<dim, spacedim>::ElastodynamicsParameters(
    const std::string      &subsection,
    TimeIntervalParameters *shared_time_parameters,
    FixedStepParameters    *shared_fixed_step_parameters,
    IDAParameters          *shared_ida_parameters,
    const bool              expose_refinement_cycles)
    : ParameterAcceptor(normalize_elastodynamics_subsection(subsection))
    , owned_time_parameters(
        shared_time_parameters == nullptr ?
          std::make_unique<TimeIntervalParameters>(
            normalize_elastodynamics_subsection(subsection) +
            "Time interval/") :
          nullptr)
    , owned_fixed_step_parameters(
        shared_fixed_step_parameters == nullptr ?
          std::make_unique<FixedStepParameters>(
            normalize_elastodynamics_subsection(subsection) + "Fixed step/") :
          nullptr)
    , owned_ida_parameters(
        shared_ida_parameters == nullptr ?
          std::make_unique<IDAParameters>(
            normalize_elastodynamics_subsection(subsection) + "IDA/") :
          nullptr)
    , domain_parameters(
        domain_subsection(normalize_elastodynamics_subsection(subsection)))
    , default_material(normalize_elastodynamics_subsection(subsection) +
                       "Material properties/default/")
    , time_parameters(shared_time_parameters != nullptr ?
                        *shared_time_parameters :
                        *owned_time_parameters)
    , fixed_step_parameters(shared_fixed_step_parameters != nullptr ?
                              *shared_fixed_step_parameters :
                              *owned_fixed_step_parameters)
    , ida_parameters(shared_ida_parameters != nullptr ? *shared_ida_parameters :
                                                        *owned_ida_parameters)
    , body_force(normalize_elastodynamics_subsection(subsection) +
                   "Functions/Body force",
                 spacedim)
    , displacement_boundary(normalize_elastodynamics_subsection(subsection) +
                              "Functions/Displacement boundary",
                            spacedim)
    , neumann_boundary(normalize_elastodynamics_subsection(subsection) +
                         "Functions/Neumann boundary",
                       spacedim)
    , robin_coefficient(normalize_elastodynamics_subsection(subsection) +
                          "Functions/Robin coefficient",
                        1)
    , robin_rhs(normalize_elastodynamics_subsection(subsection) +
                  "Functions/Robin boundary",
                spacedim)
    , velocity_boundary(normalize_elastodynamics_subsection(subsection) +
                          "Functions/Velocity boundary",
                        spacedim)
    , initial_displacement(normalize_elastodynamics_subsection(subsection) +
                             "Functions/Initial displacement",
                           spacedim)
    , initial_velocity(normalize_elastodynamics_subsection(subsection) +
                         "Functions/Initial velocity",
                       spacedim)
    , exact_solution(normalize_elastodynamics_subsection(subsection) +
                       "Functions/Exact solution",
                     spacedim)
    , solver_control(normalize_elastodynamics_subsection(subsection) +
                     "Solver/Control")
    , convergence_table(std::vector<std::string>(spacedim, "d"))
  {
    // DomainParameters declared Initial refinement with its own default of 0;
    // redeclare it so the printed default is the value used.
    domain_parameters.initial_refinement = 2;
    domain_parameters.enter_my_subsection(this->prm);
    this->prm.declare_entry("Initial refinement", "2", Patterns::Integer(0));
    domain_parameters.leave_my_subsection(this->prm);
    // A default file must run: 10 steps match the default Time step.
    if (owned_fixed_step_parameters)
      owned_fixed_step_parameters->number_of_steps = 10;
    add_parameter("FE degree", fe_degree, "", this->prm, Patterns::Integer(1));
    add_parameter("Output directory", output_directory);
    add_parameter("Output name", output_name);
    if (expose_refinement_cycles)
      add_parameter("Number of refinement cycles", n_refinement_cycles);
    add_parameter("Dirichlet boundary ids", dirichlet_ids);
    add_parameter("Neumann boundary ids", neumann_ids);
    add_parameter("Robin boundary ids", robin_ids);



    enter_subsection("Material properties");
    {
      add_parameter("Material tags by material id",
                    material_tags_by_material_id);
    }
    leave_subsection();

    const auto declare_zero_vector_function = [this]() {
      const std::string zero_expression = spacedim == 1 ? "0" :
                                          spacedim == 2 ? "0; 0" :
                                                          "0; 0; 0";
      this->prm.declare_entry(
        "Function expression",
        zero_expression,
        Patterns::List(Patterns::Anything(), spacedim, spacedim, ";"));
    };

    body_force.declare_parameters_call_back.connect(
      declare_zero_vector_function);
    displacement_boundary.declare_parameters_call_back.connect(
      declare_zero_vector_function);
    neumann_boundary.declare_parameters_call_back.connect(
      declare_zero_vector_function);
    robin_rhs.declare_parameters_call_back.connect(
      declare_zero_vector_function);
    velocity_boundary.declare_parameters_call_back.connect(
      declare_zero_vector_function);
    initial_displacement.declare_parameters_call_back.connect(
      declare_zero_vector_function);
    initial_velocity.declare_parameters_call_back.connect(
      declare_zero_vector_function);
    exact_solution.declare_parameters_call_back.connect(
      declare_zero_vector_function);

    this->prm.enter_subsection("Error");
    convergence_table.add_parameters(this->prm);
    this->prm.leave_subsection();

    const auto track_time_dependence = [](auto &function,
                                          bool &depends_on_time) {
      function.parse_parameters_call_back.connect([&depends_on_time]() {
        depends_on_time =
          robin_detail::expression_depends_on_time(ParameterAcceptor::prm,
                                                   spacedim);
      });
    };
    body_force.parse_parameters_call_back.connect([this]() {
      body_force_is_zero = expression_is_zero(ParameterAcceptor::prm);
    });
    neumann_boundary.parse_parameters_call_back.connect([this]() {
      neumann_boundary_is_zero = expression_is_zero(ParameterAcceptor::prm);
    });
    track_time_dependence(robin_coefficient, robin_coefficient_depends_on_time);
    track_time_dependence(robin_rhs, robin_rhs_depends_on_time);
    track_time_dependence(body_force, body_force_depends_on_time);
    track_time_dependence(neumann_boundary, neumann_boundary_depends_on_time);
    track_time_dependence(displacement_boundary,
                          displacement_boundary_depends_on_time);
    track_time_dependence(velocity_boundary, velocity_boundary_depends_on_time);

    const std::string material_root =
      normalize_elastodynamics_subsection(subsection) + "Material properties/";

    parse_parameters_call_back.connect([this, material_root]() {
      ensure_output_directory(output_directory);

      // The default material already exists.
      // Create only the other named material acceptors.
      this->leave_my_subsection(this->prm);

      for (const auto &[id, tag] : material_tags_by_material_id)
        {
          (void)id;

          AssertThrow(!tag.empty() && tag.find('/') == std::string::npos,
                      ExcMessage("Material tags must be nonempty and "
                                 "must not contain '/'."));

          // Reuse the permanently registered default material.
          if (tag == "default")
            continue;

          if (material_parameters_by_tag.find(tag) ==
              material_parameters_by_tag.end())
            material_parameters_by_tag.emplace(
              tag,
              std::make_unique<ElastodynamicsMaterialParameters>(material_root +
                                                                 tag + "/"));
        }

      this->enter_my_subsection(this->prm);
    });
  }

  template <int dim, int spacedim>
  ElastodynamicsMaterialCoefficients
  ElastodynamicsParameters<dim, spacedim>::get_material_properties(
    const types::material_id material_id) const
  {
    const auto tag_it = material_tags_by_material_id.find(material_id);

    // Unmapped material ID: use Material properties/default.
    if (tag_it == material_tags_by_material_id.end())
      return default_material.coefficients;

    const std::string &tag = tag_it->second;

    // Also support an explicit mapping such as 42:default.
    if (tag == "default")
      return default_material.coefficients;

    const auto material_it = material_parameters_by_tag.find(tag);

    AssertThrow(material_it != material_parameters_by_tag.end(),
                ExcMessage("Material parameters are not initialized for tag '" +
                           tag + "'."));

    return material_it->second->coefficients;
  }


  template <int dim, int spacedim>
  ElastodynamicsSolver<dim, spacedim>::ElastodynamicsSolver(
    const ElastodynamicsParameters<dim, spacedim> &par)
    : par(par)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , domain(par.domain_parameters, mpi_communicator)
    , tria(&domain.triangulation())
    , dh()
    , current_time_storage(par.time_parameters.initial_time)
  {}


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::make_grid()
  {
    make_grid([](auto &mesh_domain) { mesh_domain.make_grid(); });
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::make_grid(
    const std::function<void(Domain<dim, spacedim> &)> &initialize)
  {
    TimerOutput::Scope t(computing_timer, "Make grid");
    initialize(domain);
    AssertThrow(!domain.grid_is_prepared() &&
                  domain.triangulation().n_global_active_cells() > 0,
                ExcMessage("The grid policy must finalize the solid Domain."));
    tria = &domain.triangulation();
    dh.reinit(*tria);
    pcout << "   Triangulation backend: "
          << (domain.uses_fully_distributed_triangulation() ?
                "fullydistributed" :
                "distributed")
          << "\n"
          << "   Number of active cells: " << tria->n_active_cells()
          << std::endl;
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::setup_fe()
  {
    TimerOutput::Scope t(computing_timer, "Initial setup");
    const auto         reference_cell = detail::uniform_reference_cell(*tria);
    AssertThrow(reference_cell.has_value(),
                ExcMessage("The elastodynamics mesh has no cells."));
    if (reference_cell->is_hyper_cube())
      {
        fe_storage = std::make_unique<FESystem<dim, spacedim>>(
          FE_Q<dim, spacedim>(par.fe_degree), spacedim);
        quadrature      = std::make_unique<QGauss<dim>>(par.fe_degree + 1);
        face_quadrature = std::make_unique<QGauss<dim - 1>>(par.fe_degree + 1);
        mapping_storage = std::make_unique<MappingQ<dim, spacedim>>(1);
      }
    // Line meshes are hypercube meshes, so only dim > 1 reaches this branch.
    else if constexpr (dim > 1)
      {
        AssertThrow(reference_cell->is_simplex(),
                    ExcMessage("ElastodynamicsSolver supports meshes of only "
                               "hypercube or only simplex cells."));
        fe_storage = std::make_unique<FESystem<dim, spacedim>>(
          FE_SimplexP<dim, spacedim>(par.fe_degree), spacedim);
        quadrature = std::make_unique<QGaussSimplex<dim>>(par.fe_degree + 1);
        face_quadrature =
          std::make_unique<QGaussSimplex<dim - 1>>(par.fe_degree + 1);
        mapping_storage = std::make_unique<MappingFE<dim, spacedim>>(
          FE_SimplexP<dim, spacedim>(1));
      }
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::copy_constraints(
    const AffineConstraints<double> &source,
    AffineConstraints<double>       &target,
    const types::global_dof_index    shift)
  {
    for (const auto &line : source.get_lines())
      {
        const auto shifted_index = line.index + shift;
        target.add_line(shifted_index);
        for (const auto &entry : line.entries)
          target.add_entry(shifted_index, entry.first + shift, entry.second);
        target.set_inhomogeneity(shifted_index, line.inhomogeneity);
      }
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::rebuild_combined_constraints() const
  {
    if (!combined_constraints_initialized)
      {
        combined_constraints_storage.clear();
        combined_constraints_storage.reinit(combined_owned_dofs,
                                            combined_relevant_dofs);
        copy_constraints(displacement_constraints_storage,
                         combined_constraints_storage,
                         0);
        copy_constraints(velocity_constraints_storage,
                         combined_constraints_storage,
                         dh.n_dofs());
        combined_constraints_storage.close();
        combined_constraints_initialized = true;
      }
    else
      {
        for (const auto &line : displacement_constraints_storage.get_lines())
          combined_constraints_storage.set_inhomogeneity(line.index,
                                                         line.inhomogeneity);
        for (const auto &line : velocity_constraints_storage.get_lines())
          combined_constraints_storage.set_inhomogeneity(dh.n_dofs() +
                                                           line.index,
                                                         line.inhomogeneity);
      }
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::update_discrete_velocity_constraints(
    const AffineConstraints<double> &previous_displacement,
    const double                     previous_time,
    const double                     current_time) const
  {
    const double dt = current_time - previous_time;
    AssertThrow(dt > 0., ExcMessage("Time must increase between steps."));

    for (const auto &line : displacement_constraints_storage.get_lines())
      {
        const double previous_value =
          previous_displacement.is_constrained(line.index) ?
            previous_displacement.get_inhomogeneity(line.index) :
            0.;
        velocity_constraints_storage.set_inhomogeneity(
          line.index, (line.inhomogeneity - previous_value) / dt);
      }
    velocity_constraints_are_boundary_data = false;
    rebuild_combined_constraints();
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::update_trapezoidal_velocity_constraints(
    const AffineConstraints<double> &previous_displacement,
    const AffineConstraints<double> &previous_velocity,
    const double                     previous_time,
    const double                     current_time) const
  {
    const double dt = current_time - previous_time;
    AssertThrow(dt > 0., ExcMessage("Time must increase between steps."));

    for (const auto &line : displacement_constraints_storage.get_lines())
      {
        const double previous_displacement_value =
          previous_displacement.is_constrained(line.index) ?
            previous_displacement.get_inhomogeneity(line.index) :
            0.;
        const double previous_velocity_value =
          previous_velocity.is_constrained(line.index) ?
            previous_velocity.get_inhomogeneity(line.index) :
            0.;
        velocity_constraints_storage.set_inhomogeneity(
          line.index,
          2. * (line.inhomogeneity - previous_displacement_value) / dt -
            previous_velocity_value);
      }
    velocity_constraints_are_boundary_data = false;
    rebuild_combined_constraints();
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::update_constraints(
    const double time) const
  {
    update_constraints(time, true);
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::update_constraints(
    const double time,
    const bool   update_velocity) const
  {
    const bool update_displacement =
      !displacement_constraints_time.has_value() ||
      (!par.dirichlet_ids.empty() &&
       par.displacement_boundary_depends_on_time &&
       *displacement_constraints_time != time);
    const bool refresh_velocity =
      update_velocity &&
      (!velocity_constraints_time.has_value() ||
       !velocity_constraints_are_boundary_data ||
       (!par.dirichlet_ids.empty() && par.velocity_boundary_depends_on_time &&
        *velocity_constraints_time != time));
    if (!update_displacement && !refresh_velocity)
      return;

    if (update_displacement)
      {
        displacement_constraints_storage.clear();
        displacement_constraints_storage.reinit(owned_dofs, relevant_dofs);
        copy_constraints(hanging_node_constraints_storage,
                         displacement_constraints_storage,
                         0);
        par.displacement_boundary.set_time(time);
        for (const auto id : par.dirichlet_ids)
          VectorTools::interpolate_boundary_values(
            *mapping_storage,
            dh,
            id,
            par.displacement_boundary,
            displacement_constraints_storage);
        displacement_constraints_storage.close();
        displacement_constraints_time = time;
      }

    if (refresh_velocity)
      {
        velocity_constraints_storage.clear();
        velocity_constraints_storage.reinit(owned_dofs, relevant_dofs);
        copy_constraints(hanging_node_constraints_storage,
                         velocity_constraints_storage,
                         0);
        par.velocity_boundary.set_time(time);
        for (const auto id : par.dirichlet_ids)
          VectorTools::interpolate_boundary_values(
            *mapping_storage,
            dh,
            id,
            par.velocity_boundary,
            velocity_constraints_storage);
        velocity_constraints_storage.close();
        velocity_constraints_time              = time;
        velocity_constraints_are_boundary_data = true;
      }

    ++assembly_statistics_storage.constraints;
    rebuild_combined_constraints();
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "Setup system");
    AssertThrow(fe_storage != nullptr,
                ExcMessage("setup_fe() must be called first."));

    dh.distribute_dofs(*fe_storage);
    owned_dofs    = dh.locally_owned_dofs();
    relevant_dofs = DoFTools::extract_locally_relevant_dofs(dh);

    robin_detail::assert_disjoint_boundary_ids(par.robin_ids,
                                               par.dirichlet_ids,
                                               par.neumann_ids,
                                               mpi_communicator);
    robin_operator_time.reset();
    robin_force_time.reset();
    spatial_operators_ready = false;
    time_preconditioner.reset();
    system_matrix_time_step.reset();
    volume_force_cache_time.reset();
    traction_force_cache_time.reset();
    displacement_constraints_time.reset();
    velocity_constraints_time.reset();
    combined_constraints_initialized = false;
    hanging_node_constraints_storage.clear();
    hanging_node_constraints_storage.reinit(owned_dofs, relevant_dofs);
    DoFTools::make_hanging_node_constraints(dh,
                                            hanging_node_constraints_storage);
    hanging_node_constraints_storage.close();

    const auto n_spatial_dofs = dh.n_dofs();
    combined_owned_dofs       = IndexSet(2 * n_spatial_dofs);
    combined_relevant_dofs    = IndexSet(2 * n_spatial_dofs);
    for (const auto index : owned_dofs)
      {
        combined_owned_dofs.add_index(index);
        combined_owned_dofs.add_index(n_spatial_dofs + index);
      }
    for (const auto index : relevant_dofs)
      {
        combined_relevant_dofs.add_index(index);
        combined_relevant_dofs.add_index(n_spatial_dofs + index);
      }

    update_constraints(par.time_parameters.initial_time);

    DynamicSparsityPattern spatial_dsp(relevant_dofs);
    // Retain prescribed-value columns while expanding mesh-slave couplings.
    DoFTools::make_sparsity_pattern(dh,
                                    spatial_dsp,
                                    hanging_node_constraints_storage,
                                    true);
    SparsityTools::distribute_sparsity_pattern(spatial_dsp,
                                               owned_dofs,
                                               mpi_communicator,
                                               relevant_dofs);

    DynamicSparsityPattern combined_dsp(combined_relevant_dofs);
    for (const auto row : relevant_dofs)
      {
        combined_dsp.add(row, row);
        combined_dsp.add(row, n_spatial_dofs + row);
        for (unsigned int k = 0; k < spatial_dsp.row_length(row); ++k)
          {
            const auto column = spatial_dsp.column_number(row, k);
            combined_dsp.add(n_spatial_dofs + row, column);
            combined_dsp.add(n_spatial_dofs + row, n_spatial_dofs + column);
          }
      }
    SparsityTools::distribute_sparsity_pattern(combined_dsp,
                                               combined_owned_dofs,
                                               mpi_communicator,
                                               combined_relevant_dofs);

    mass_matrix_storage.reinit(owned_dofs,
                               owned_dofs,
                               spatial_dsp,
                               mpi_communicator);
    {
      DynamicSparsityPattern diagonal_dsp(dh.n_dofs(), dh.n_dofs(), owned_dofs);
      for (const auto row : owned_dofs)
        diagonal_dsp.add(row, row);
      diagonal_mass_matrix_storage.reinit(owned_dofs,
                                          owned_dofs,
                                          diagonal_dsp,
                                          mpi_communicator);
    }
    stiffness_matrix_storage.reinit(owned_dofs,
                                    owned_dofs,
                                    spatial_dsp,
                                    mpi_communicator);
    damping_matrix_storage.reinit(owned_dofs,
                                  owned_dofs,
                                  spatial_dsp,
                                  mpi_communicator);
    homogeneous_mass_matrix_storage.reinit(owned_dofs,
                                           owned_dofs,
                                           spatial_dsp,
                                           mpi_communicator);
    homogeneous_stiffness_matrix_storage.reinit(owned_dofs,
                                                owned_dofs,
                                                spatial_dsp,
                                                mpi_communicator);
    homogeneous_damping_matrix_storage.reinit(owned_dofs,
                                              owned_dofs,
                                              spatial_dsp,
                                              mpi_communicator);
    system_matrix_storage.reinit(combined_owned_dofs,
                                 combined_owned_dofs,
                                 combined_dsp,
                                 mpi_communicator);

    body_force_storage.reinit(owned_dofs, mpi_communicator);
    cached_body_force.reinit(owned_dofs, mpi_communicator);
    cached_robin_force.reinit(owned_dofs, mpi_communicator);
    previous_stiffness_action.reinit(owned_dofs, mpi_communicator);
    cached_volume_force.reinit(owned_dofs, mpi_communicator);
    cached_traction_force.reinit(owned_dofs, mpi_communicator);
    displacement_storage.reinit(owned_dofs, mpi_communicator);
    velocity_storage.reinit(owned_dofs, mpi_communicator);
    locally_relevant_displacement.reinit(owned_dofs,
                                         relevant_dofs,
                                         mpi_communicator);
    locally_relevant_velocity.reinit(owned_dofs,
                                     relevant_dofs,
                                     mpi_communicator);
    system_rhs_storage.reinit(combined_owned_dofs, mpi_communicator);

    body_force_storage            = 0.;
    displacement_storage          = 0.;
    velocity_storage              = 0.;
    locally_relevant_displacement = 0.;
    locally_relevant_velocity     = 0.;
    system_rhs_storage            = 0.;

    pcout << "   Number of spatial degrees of freedom: " << dh.n_dofs()
          << " (locally owned: " << owned_dofs.n_elements() << ")" << std::endl;
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::assemble_operators()
  {
    if (spatial_operators_ready)
      return;
    TimerOutput::Scope t(computing_timer, "Assemble spatial operators");
    ++assembly_statistics_storage.spatial_operators;
    AssertThrow(fe_storage != nullptr && quadrature != nullptr,
                ExcMessage("setup_fe() must be called before assembly."));

    mass_matrix_storage                  = 0.;
    stiffness_matrix_storage             = 0.;
    damping_matrix_storage               = 0.;
    homogeneous_mass_matrix_storage      = 0.;
    homogeneous_stiffness_matrix_storage = 0.;
    homogeneous_damping_matrix_storage   = 0.;

    const auto &mesh_constraints = hanging_node_constraints_storage;

    FEValues<dim, spacedim>          fe_values(*mapping_storage,
                                      *fe_storage,
                                      *quadrature,
                                      update_values | update_gradients |
                                        update_quadrature_points |
                                        update_JxW_values);
    const FEValuesExtractors::Vector vector_field(0);
    const unsigned int dofs_per_cell = fe_storage->n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature->size();

    FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiffness(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_damping(dofs_per_cell, dofs_per_cell);
    std::vector<Tensor<2, spacedim>>     symmetric_gradients(dofs_per_cell);
    std::vector<double>                  divergences(dofs_per_cell);
    std::vector<Tensor<1, spacedim>>     values(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto mp  = par.get_material_properties(cell->material_id());
          cell_mass      = 0.;
          cell_stiffness = 0.;
          cell_damping   = 0.;
          fe_values.reinit(cell);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  symmetric_gradients[k] =
                    fe_values[vector_field].symmetric_gradient(k, q);
                  divergences[k] = fe_values[vector_field].divergence(k, q);
                  values[k]      = fe_values[vector_field].value(k, q);
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const auto jxw = fe_values.JxW(q);

                    cell_mass(i, j) +=
                      mp.density * (values[i] * values[j]) * jxw;

                    cell_stiffness(i, j) +=
                      (2. * mp.lame_mu *
                         scalar_product(symmetric_gradients[i],
                                        symmetric_gradients[j]) +
                       mp.lame_lambda * divergences[i] * divergences[j]) *
                      jxw;

                    cell_damping(i, j) +=
                      (mp.rayleigh_alpha * mp.density *
                         (values[i] * values[j]) +

                       (mp.viscosity_eta + 2. * mp.rayleigh_beta * mp.lame_mu) *
                         scalar_product(symmetric_gradients[i],
                                        symmetric_gradients[j]) +

                       (mp.rayleigh_beta * mp.lame_lambda) * divergences[i] *
                         divergences[j]) *
                      jxw;
                  }
            }

          cell->get_dof_indices(local_dof_indices);
          mesh_constraints.distribute_local_to_global(cell_mass,
                                                      local_dof_indices,
                                                      local_dof_indices,
                                                      mass_matrix_storage);
          mesh_constraints.distribute_local_to_global(cell_stiffness,
                                                      local_dof_indices,
                                                      local_dof_indices,
                                                      stiffness_matrix_storage);
          mesh_constraints.distribute_local_to_global(cell_damping,
                                                      local_dof_indices,
                                                      local_dof_indices,
                                                      damping_matrix_storage);
          displacement_constraints_storage.distribute_local_to_global(
            cell_mass,
            local_dof_indices,
            local_dof_indices,
            homogeneous_mass_matrix_storage);
          displacement_constraints_storage.distribute_local_to_global(
            cell_stiffness,
            local_dof_indices,
            local_dof_indices,
            homogeneous_stiffness_matrix_storage);
          displacement_constraints_storage.distribute_local_to_global(
            cell_damping,
            local_dof_indices,
            local_dof_indices,
            homogeneous_damping_matrix_storage);
        }

    mass_matrix_storage.compress(VectorOperation::add);
    for (const auto row : owned_dofs)
      diagonal_mass_matrix_storage.set(row,
                                       row,
                                       mass_matrix_storage.diag_element(row));
    diagonal_mass_matrix_storage.compress(VectorOperation::insert);
    stiffness_matrix_storage.compress(VectorOperation::add);
    damping_matrix_storage.compress(VectorOperation::add);
    homogeneous_mass_matrix_storage.compress(VectorOperation::add);
    homogeneous_stiffness_matrix_storage.compress(VectorOperation::add);
    homogeneous_damping_matrix_storage.compress(VectorOperation::add);

    const auto set_constraint_diagonal =
      [&](MatrixType &destination, const double constrained_diagonal) {
        for (const auto &line : displacement_constraints_storage.get_lines())
          if (owned_dofs.is_element(line.index))
            destination.set(line.index, line.index, constrained_diagonal);
        destination.compress(VectorOperation::insert);
      };
    set_constraint_diagonal(homogeneous_mass_matrix_storage, 1.);
    set_constraint_diagonal(homogeneous_stiffness_matrix_storage, 0.);
    set_constraint_diagonal(homogeneous_damping_matrix_storage, 0.);
    spatial_operators_ready = true;
    update_boundary_operators(par.time_parameters.initial_time);
  }

  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::update_boundary_operators(
    const double time) const
  {
    if (par.robin_ids.empty() ||
        (robin_operator_time && (!par.robin_coefficient_depends_on_time ||
                                 *robin_operator_time == time)))
      return;
    AssertThrow(spatial_operators_ready,
                ExcMessage("Assemble volume operators before Robin terms."));
    FEFaceValues<dim, spacedim>          values(*mapping_storage,
                                       *fe_storage,
                                       *face_quadrature,
                                       update_values |
                                         update_quadrature_points |
                                         update_JxW_values);
    const FEValuesExtractors::Vector     extractor(0);
    const auto                           n = fe_storage->n_dofs_per_cell();
    FullMatrix<double>                   matrix(n, n);
    Vector<double>                       rhs(n);
    std::vector<types::global_dof_index> indices(n);
    Functions::ZeroFunction<spacedim>    zero(spacedim);
    const auto &mesh_constraints = hanging_node_constraints_storage;
    par.robin_coefficient.set_time(time);
    unsigned int invalid = 0;
    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          matrix = 0.;
          rhs    = 0.;
          for (const auto face : cell->face_indices())
            if (cell->face(face)->at_boundary() &&
                par.robin_ids.count(cell->face(face)->boundary_id()))
              {
                values.reinit(cell, face);
                // Update only face integrals: never duplicate the volume K.
                if (robin_operator_time)
                  {
                    par.robin_coefficient.set_time(*robin_operator_time);
                    invalid |=
                      robin_detail::add_face_terms(values,
                                                   extractor,
                                                   par.robin_coefficient,
                                                   zero,
                                                   matrix,
                                                   rhs,
                                                   -1.);
                  }
                par.robin_coefficient.set_time(time);
                invalid |= robin_detail::add_face_terms(
                  values, extractor, par.robin_coefficient, zero, matrix, rhs);
              }
          cell->get_dof_indices(indices);
          mesh_constraints.distribute_local_to_global(matrix,
                                                      indices,
                                                      indices,
                                                      stiffness_matrix_storage);
          displacement_constraints_storage.distribute_local_to_global(
            matrix, indices, indices, homogeneous_stiffness_matrix_storage);
        }
    robin_detail::assert_valid_boundary_data(invalid, mpi_communicator);
    stiffness_matrix_storage.compress(VectorOperation::add);
    homogeneous_stiffness_matrix_storage.compress(VectorOperation::add);
    for (const auto &line : displacement_constraints_storage.get_lines())
      if (owned_dofs.is_element(line.index))
        homogeneous_stiffness_matrix_storage.set(line.index, line.index, 0.);
    homogeneous_stiffness_matrix_storage.compress(VectorOperation::insert);
    robin_operator_time = time;
    ++assembly_statistics_storage.robin_operators;
    system_matrix_time_step.reset();
    time_preconditioner.reset();
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::assemble_body_force(const double time)
  {
    body_force_at_time(time, body_force_storage);
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::body_force_at_time(
    const double time,
    VectorType  &destination) const
  {
    const bool refresh_volume =
      !volume_force_cache_time.has_value() ||
      (par.body_force_depends_on_time && *volume_force_cache_time != time);
    const bool refresh_traction =
      !traction_force_cache_time.has_value() ||
      (!par.neumann_ids.empty() && par.neumann_boundary_depends_on_time &&
       *traction_force_cache_time != time);
    const bool refresh_robin =
      !par.robin_ids.empty() &&
      (!robin_force_time ||
       (par.robin_rhs_depends_on_time && *robin_force_time != time));
    if (!refresh_volume && !refresh_traction && !refresh_robin)
      {
        destination = cached_body_force;
        return;
      }
    TimerOutput::Scope t(computing_timer, "Assemble body force");
    ++assembly_statistics_storage.body_force;

    const auto             &mesh_constraints = hanging_node_constraints_storage;
    FEValues<dim, spacedim> fe_values(*mapping_storage,
                                      *fe_storage,
                                      *quadrature,
                                      update_values | update_quadrature_points |
                                        update_JxW_values);
    FEFaceValues<dim, spacedim>      fe_face_values(*mapping_storage,
                                               *fe_storage,
                                               *face_quadrature,
                                               update_values |
                                                 update_quadrature_points |
                                                 update_JxW_values);
    const FEValuesExtractors::Vector vector_field(0);
    const unsigned int          dofs_per_cell = fe_storage->n_dofs_per_cell();
    const unsigned int          n_q_points    = quadrature->size();
    Vector<double>              cell_rhs(dofs_per_cell);
    std::vector<Vector<double>> force_values(n_q_points,
                                             Vector<double>(spacedim));
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    if (refresh_volume)
      {
        cached_volume_force = 0.;
        if (!par.body_force_is_zero)
          {
            par.body_force.set_time(time);
            for (const auto &cell : dh.active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  cell_rhs = 0.;
                  fe_values.reinit(cell);
                  par.body_force.vector_value_list(
                    fe_values.get_quadrature_points(), force_values);
                  for (unsigned int q = 0; q < n_q_points; ++q)
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        const auto component =
                          fe_storage->system_to_component_index(i).first;
                        cell_rhs(i) +=
                          fe_values[vector_field].value(i, q)[component] *
                          force_values[q](component) * fe_values.JxW(q);
                      }

                  cell->get_dof_indices(local_dof_indices);
                  mesh_constraints.distribute_local_to_global(
                    cell_rhs, local_dof_indices, cached_volume_force);
                }
          }
        cached_volume_force.compress(VectorOperation::add);
        volume_force_cache_time = time;
      }

    if (refresh_traction)
      {
        cached_traction_force = 0.;
        if (!par.neumann_boundary_is_zero && !par.neumann_ids.empty())
          {
            par.neumann_boundary.set_time(time);
            for (const auto &cell : dh.active_cell_iterators())
              if (cell->is_locally_owned())
                for (const auto face : cell->face_indices())
                  if (cell->face(face)->at_boundary() &&
                      par.neumann_ids.find(cell->face(face)->boundary_id()) !=
                        par.neumann_ids.end())
                    {
                      fe_face_values.reinit(cell, face);
                      std::vector<Vector<double>> values(
                        fe_face_values.n_quadrature_points,
                        Vector<double>(spacedim));
                      par.neumann_boundary.vector_value_list(
                        fe_face_values.get_quadrature_points(), values);
                      cell_rhs = 0.;
                      for (unsigned int q = 0;
                           q < fe_face_values.n_quadrature_points;
                           ++q)
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            const auto component =
                              fe_storage->system_to_component_index(i).first;
                            cell_rhs(i) += fe_face_values.shape_value(i, q) *
                                           values[q](component) *
                                           fe_face_values.JxW(q);
                          }
                      cell->get_dof_indices(local_dof_indices);
                      mesh_constraints.distribute_local_to_global(
                        cell_rhs, local_dof_indices, cached_traction_force);
                    }
          }
        cached_traction_force.compress(VectorOperation::add);
        traction_force_cache_time = time;
      }
    if (refresh_robin)
      {
        cached_robin_force = 0.;
        par.robin_rhs.set_time(time);
        Functions::ConstantFunction<spacedim> zero(0.);
        FullMatrix<double> ignored(dofs_per_cell, dofs_per_cell);
        unsigned int       invalid = 0;
        for (const auto &cell : dh.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell_rhs = 0.;
              ignored  = 0.;
              for (const auto face : cell->face_indices())
                if (cell->face(face)->at_boundary() &&
                    par.robin_ids.count(cell->face(face)->boundary_id()))
                  {
                    fe_face_values.reinit(cell, face);
                    invalid |= robin_detail::add_face_terms(fe_face_values,
                                                            vector_field,
                                                            zero,
                                                            par.robin_rhs,
                                                            ignored,
                                                            cell_rhs);
                  }
              cell->get_dof_indices(local_dof_indices);
              mesh_constraints.distribute_local_to_global(cell_rhs,
                                                          local_dof_indices,
                                                          cached_robin_force);
            }
        robin_detail::assert_valid_boundary_data(invalid, mpi_communicator);
        cached_robin_force.compress(VectorOperation::add);
        robin_force_time = time;
      }
    cached_body_force = cached_volume_force;
    cached_body_force += cached_traction_force;
    if (!par.robin_ids.empty())
      cached_body_force += cached_robin_force;
    destination = cached_body_force;
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::set_initial_conditions()
  {
    TimerOutput::Scope t(computing_timer, "Set initial conditions");
    current_time_storage     = par.time_parameters.initial_time;
    time_step_number_storage = 0;
    current_time_step        = 0.;
    update_constraints(current_time_storage);
    update_boundary_operators(current_time_storage);

    par.initial_displacement.set_time(current_time_storage);
    par.initial_velocity.set_time(current_time_storage);
    VectorTools::interpolate(*mapping_storage,
                             dh,
                             par.initial_displacement,
                             displacement_storage);
    VectorTools::interpolate(*mapping_storage,
                             dh,
                             par.initial_velocity,
                             velocity_storage);

    // The initial velocity is the value of the initial condition, not an
    // independently prescribed boundary datum.  In particular, do not impose
    // the default zero velocity boundary on a moving displacement boundary.
    velocity_constraints_are_boundary_data = false;
    velocity_constraints_storage.clear();
    velocity_constraints_storage.reinit(owned_dofs, relevant_dofs);
    DoFTools::make_hanging_node_constraints(dh, velocity_constraints_storage);
    for (const auto id : par.dirichlet_ids)
      VectorTools::interpolate_boundary_values(*mapping_storage,
                                               dh,
                                               id,
                                               par.initial_velocity,
                                               velocity_constraints_storage);
    velocity_constraints_storage.close();
    rebuild_combined_constraints();

    displacement_constraints_storage.distribute(displacement_storage);
    velocity_constraints_storage.distribute(velocity_storage);
    update_locally_relevant_state();
    assemble_body_force(current_time_storage);
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::refine_global()
  {
    AssertThrow(!domain.uses_fully_distributed_triangulation(),
                ExcMessage(
                  "ElastodynamicsSolver::refine_global() is unavailable for "
                  "parallel::fullydistributed::Triangulation."));
    AssertThrow(dh.n_dofs() != 0,
                ExcMessage("Call setup_system() before refining the mesh."));

    dh.clear();
    dynamic_cast<parallel::distributed::Triangulation<dim, spacedim> &>(
      domain.triangulation())
      .refine_global(1);
    cycles_and_solutions.clear();
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::initial_acceleration(
    VectorType &acceleration) const
  {
    update_boundary_operators(current_time_storage);
    VectorType rhs;
    VectorType work;
    body_force_at_time(current_time_storage, rhs);
    work.reinit(rhs);
    stiffness_matrix_storage.vmult(work, displacement_storage);
    rhs -= work;
    damping_matrix_storage.vmult(work, velocity_storage);
    rhs -= work;

    VectorType lifting;
    lifting.reinit(owned_dofs, mpi_communicator);
    lifting = 0.;
    velocity_constraints_storage.distribute(lifting);
    mass_matrix_storage.vmult(work, lifting);
    rhs -= work;
    VectorType projected_rhs;
    projected_rhs.reinit(rhs);
    project_rhs(rhs, velocity_constraints_storage, projected_rhs);

    acceleration.reinit(owned_dofs, mpi_communicator);
    acceleration = 0.;
    SolverControl            control(par.solver_control.max_steps(),
                          par.solver_control.tolerance());
    LA::MPI::PreconditionAMG preconditioner;
    preconditioner.initialize(homogeneous_mass_matrix_storage);
    SolverCG<VectorType> solver(control);
    solver.solve(homogeneous_mass_matrix_storage,
                 acceleration,
                 projected_rhs,
                 preconditioner);
    velocity_constraints_storage.distribute(acceleration);
  }



  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::assemble_backward_euler_system(
    const VectorType &previous_displacement,
    const VectorType &previous_velocity,
    const double      dt)
  {
    prepare_time_integration_matrix(dt, 1.);
    assemble_time_integration_rhs(
      previous_displacement, previous_velocity, body_force_storage, dt, 1.);
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::assemble_trapezoidal_system(
    const VectorType &previous_displacement,
    const VectorType &previous_velocity,
    const VectorType &previous_body_force,
    const double      dt)
  {
    prepare_time_integration_matrix(dt, .5);
    assemble_time_integration_rhs(
      previous_displacement, previous_velocity, previous_body_force, dt, .5);
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::prepare_time_integration_matrix(
    const double dt,
    const double theta)
  {
    const bool same_step =
      system_matrix_time_step.has_value() &&
      std::abs(*system_matrix_time_step - dt) <=
        32. * std::numeric_limits<double>::epsilon() *
          std::max(std::abs(dt), std::abs(*system_matrix_time_step));
    if (same_step && system_matrix_theta == theta)
      return;
    AssertThrow(spatial_operators_ready,
                ExcMessage("Assemble spatial operators before advancing."));
    TimerOutput::Scope timer(computing_timer, "Build time-integration matrix");
    ++assembly_statistics_storage.time_system_matrix;
    const auto n_spatial_dofs = dh.n_dofs();
    system_matrix_storage     = 0.;
    system_matrix_storage.compress(VectorOperation::add);
    for (const auto row : owned_dofs)
      if (!displacement_constraints_storage.is_constrained(row) &&
          homogeneous_mass_matrix_storage.row_length(row) != 0)
        for (auto entry = homogeneous_mass_matrix_storage.begin(row),
                  end   = homogeneous_mass_matrix_storage.end(row);
             entry != end;
             ++entry)
          {
            const auto   column = entry->column();
            const double mass   = entry->value();
            const double stiffness =
              homogeneous_stiffness_matrix_storage.el(row, column);
            const double damping =
              homogeneous_damping_matrix_storage.el(row, column);
            system_matrix_storage.add(n_spatial_dofs + row,
                                      column,
                                      theta * stiffness);
            system_matrix_storage.add(n_spatial_dofs + row,
                                      n_spatial_dofs + column,
                                      mass / dt + theta * damping);
          }
    // On independent DoFs the kinematic equation only enforces u' = v.
    // Keep its two blocks diagonal, including on locally refined meshes.
    for (const auto row : owned_dofs)
      if (!displacement_constraints_storage.is_constrained(row))
        {
          const double diagonal =
            diagonal_mass_matrix_storage.diag_element(row);
          system_matrix_storage.add(row, row, diagonal / dt);
          system_matrix_storage.add(row,
                                    n_spatial_dofs + row,
                                    -theta * diagonal);
        }
    system_matrix_storage.compress(VectorOperation::add);
    for (const auto &line : combined_constraints_storage.get_lines())
      if (combined_owned_dofs.is_element(line.index))
        system_matrix_storage.set(line.index, line.index, 1.);
    system_matrix_storage.compress(VectorOperation::insert);
    system_matrix_time_step = dt;
    system_matrix_theta     = theta;
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::assemble_time_integration_rhs(
    const VectorType &previous_displacement,
    const VectorType &previous_velocity,
    const VectorType &previous_body_force,
    const double,
    const double theta)
  {
    TimerOutput::Scope timer(computing_timer, "Assemble time-integration rhs");
    // Use the cached step after normalizing roundoff in the final time step.
    const double dt = *system_matrix_time_step;
    VectorType   displacement_rhs, velocity_rhs, work;
    displacement_rhs.reinit(owned_dofs, mpi_communicator);
    velocity_rhs.reinit(displacement_rhs);
    work.reinit(displacement_rhs);
    diagonal_mass_matrix_storage.vmult(displacement_rhs, previous_displacement);
    displacement_rhs /= dt;
    diagonal_mass_matrix_storage.vmult(work, previous_velocity);
    displacement_rhs.add(1. - theta, work);
    mass_matrix_storage.vmult(velocity_rhs, previous_velocity);
    velocity_rhs /= dt;
    velocity_rhs.add(-(1. - theta), previous_stiffness_action);
    damping_matrix_storage.vmult(work, previous_velocity);
    velocity_rhs.add(-(1. - theta), work);
    velocity_rhs.add(theta, body_force_storage);
    velocity_rhs.add(1. - theta, previous_body_force);

    VectorType displacement_lifting, velocity_lifting;
    displacement_lifting.reinit(displacement_rhs);
    velocity_lifting.reinit(displacement_rhs);
    displacement_lifting = 0.;
    velocity_lifting     = 0.;
    displacement_constraints_storage.distribute(displacement_lifting);
    velocity_constraints_storage.distribute(velocity_lifting);
    diagonal_mass_matrix_storage.vmult(work, displacement_lifting);
    displacement_rhs.add(-1. / dt, work);
    stiffness_matrix_storage.vmult(work, displacement_lifting);
    velocity_rhs.add(-theta, work);
    diagonal_mass_matrix_storage.vmult(work, velocity_lifting);
    displacement_rhs.add(theta, work);
    mass_matrix_storage.vmult(work, velocity_lifting);
    velocity_rhs.add(-1. / dt, work);
    damping_matrix_storage.vmult(work, velocity_lifting);
    velocity_rhs.add(-theta, work);

    VectorType raw_rhs;
    raw_rhs.reinit(combined_owned_dofs, mpi_communicator);
    for (const auto index : owned_dofs)
      {
        // Kinematic rows act only on independent DoFs; slave values must
        // not contribute to their right-hand sides.
        raw_rhs(index) =
          displacement_constraints_storage.is_constrained(index) ?
            0. :
            displacement_rhs(index);
        raw_rhs(dh.n_dofs() + index) = velocity_rhs(index);
      }
    raw_rhs.compress(VectorOperation::insert);
    project_rhs(raw_rhs, combined_constraints_storage, system_rhs_storage);
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::solve_time_integration_system()
  {
    TimerOutput::Scope t(computing_timer, "Solve time-integration system");

    const auto n_spatial_dofs = dh.n_dofs();
    VectorType combined_solution;
    combined_solution.reinit(combined_owned_dofs, mpi_communicator);
    for (const auto index : owned_dofs)
      {
        combined_solution(index)                  = displacement_storage(index);
        combined_solution(n_spatial_dofs + index) = velocity_storage(index);
      }
    // The condensed solve uses homogeneous slave entries; physical boundary
    // values and hanging-node relations are restored after the solve.
    for (const auto &line : combined_constraints_storage.get_lines())
      if (combined_owned_dofs.is_element(line.index))
        combined_solution(line.index) = 0.;

    if (!time_preconditioner)
      {
        TimerOutput::Scope setup_timer(computing_timer,
                                       "Initialize block preconditioner");
        IndexSet           constrained(dh.n_dofs());
        for (const auto &line : displacement_constraints_storage.get_lines())
          constrained.add_index(line.index);
        constrained.compress();
        LA::MPI::PreconditionAMG::AdditionalData amg_data;
        if constexpr (dim == spacedim)
          amg_data.constant_modes_values =
            DoFTools::extract_rigid_body_modes(*mapping_storage, dh);
        else
          {
            // deal.II does not instantiate extract_constant_modes() for
            // embedded spaces; extract the component masks directly.
            amg_data.constant_modes.assign(
              spacedim, std::vector<bool>(owned_dofs.n_elements(), false));
            for (unsigned int component = 0; component < spacedim; ++component)
              {
                const auto selected = DoFTools::extract_dofs(
                  dh,
                  fe_storage->component_mask(
                    FEValuesExtractors::Scalar(component)));
                std::size_t local_index = 0;
                for (const auto index : owned_dofs)
                  amg_data.constant_modes[component][local_index++] =
                    selected.is_element(index);
              }
          }
        amg_data.higher_order_elements = fe_storage->degree > 1;
        time_preconditioner            = std::make_unique<
          SecondOrderBlockPreconditioner<VectorType, MatrixType>>();
        time_preconditioner->initialize(
          diagonal_mass_matrix_storage,
          homogeneous_mass_matrix_storage,
          homogeneous_stiffness_matrix_storage,
          homogeneous_damping_matrix_storage,
          constrained,
          [this](VectorType &vector, const bool omit) {
            vector.reinit(owned_dofs, mpi_communicator, omit);
          },
          amg_data);
      }
    {
      TimerOutput::Scope setup_timer(computing_timer,
                                     "Update block Schur preconditioner");
      time_preconditioner->set_coefficients(1. / *system_matrix_time_step,
                                            system_matrix_theta,
                                            system_matrix_theta,
                                            system_matrix_theta);
    }

    VectorType source_displacement, source_velocity;
    VectorType destination_displacement, destination_velocity;
    source_displacement.reinit(owned_dofs, mpi_communicator);
    source_velocity.reinit(source_displacement);
    destination_displacement.reinit(source_displacement);
    destination_velocity.reinit(source_displacement);
    LinearOperator<VectorType> preconditioner;
    preconditioner.vmult = [&](VectorType       &destination,
                               const VectorType &source) {
      for (const auto index : owned_dofs)
        {
          source_displacement(index) = source(index);
          source_velocity(index)     = source(n_spatial_dofs + index);
        }
      time_preconditioner->vmult(destination_displacement,
                                 destination_velocity,
                                 source_displacement,
                                 source_velocity);
      for (const auto index : owned_dofs)
        {
          destination(index)                  = destination_displacement(index);
          destination(n_spatial_dofs + index) = destination_velocity(index);
        }
    };
    typename SolverFGMRES<VectorType>::AdditionalData solver_data;
    solver_data.orthogonalization_strategy =
      LinearAlgebra::OrthogonalizationStrategy::modified_gram_schmidt;
    SolverFGMRES<VectorType> solver(par.solver_control, solver_data);
    solver.solve(system_matrix_storage,
                 combined_solution,
                 system_rhs_storage,
                 preconditioner);
    combined_constraints_storage.distribute(combined_solution);

    for (const auto index : owned_dofs)
      {
        displacement_storage(index) = combined_solution(index);
        velocity_storage(index)     = combined_solution(n_spatial_dofs + index);
      }
    displacement_constraints_storage.distribute(displacement_storage);
    velocity_constraints_storage.distribute(velocity_storage);
    update_locally_relevant_state();

    pcout << "   FGMRES iterations: " << par.solver_control.last_step()
          << std::endl;
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::advance_one_timestep()
  {
    AssertThrow(par.fixed_step_parameters.time_step > 0.,
                ExcMessage("Cannot advance with a non-positive time step."));

    advance_one_timestep(par.fixed_step_parameters.time_step);
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::advance_one_timestep(const double dt)
  {
    AssertThrow(dt > 0.,
                ExcMessage("Cannot advance with a non-positive time step."));

    const auto previous_displacement = displacement_storage;
    const auto previous_velocity     = velocity_storage;
    const auto previous_body_force   = body_force_storage;
    const AffineConstraints<double> previous_constraints(
      displacement_constraints_storage);
    const AffineConstraints<double> previous_velocity_constraints(
      velocity_constraints_storage);
    const auto next_time = current_time_storage + dt;
    update_boundary_operators(current_time_storage);
    stiffness_matrix_storage.vmult(previous_stiffness_action,
                                   previous_displacement);
    update_boundary_operators(next_time);

    update_constraints(next_time, false);
    if (par.fixed_step_parameters.time_step_strategy == "trapezoidal")
      update_trapezoidal_velocity_constraints(previous_constraints,
                                              previous_velocity_constraints,
                                              current_time_storage,
                                              next_time);
    else
      update_discrete_velocity_constraints(previous_constraints,
                                           current_time_storage,
                                           next_time);
    assemble_body_force(next_time);
    if (par.fixed_step_parameters.time_step_strategy == "trapezoidal")
      assemble_trapezoidal_system(previous_displacement,
                                  previous_velocity,
                                  previous_body_force,
                                  dt);
    else
      assemble_backward_euler_system(previous_displacement,
                                     previous_velocity,
                                     dt);
    solve_time_integration_system();

    current_time_storage += dt;
    current_time_step = dt;
    ++time_step_number_storage;
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::solve()
  {
    advance_one_timestep();
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::update_locally_relevant_state()
  {
    locally_relevant_displacement = displacement_storage;
    locally_relevant_displacement.update_ghost_values();
    locally_relevant_velocity = velocity_storage;
    locally_relevant_velocity.update_ghost_values();
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::compute_error() const
  {
    // ParsedConvergenceTable::error_from_exact() integrates with QGauss.
    AssertThrow(fe_storage != nullptr &&
                  fe_storage->reference_cell().is_hyper_cube(),
                ExcMessage("compute_error() supports hypercube meshes only."));
    par.exact_solution.set_time(current_time_storage);
    par.convergence_table.error_from_exact(*mapping_storage,
                                           dh,
                                           locally_relevant_displacement,
                                           par.exact_solution);
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::output_results() const
  {
    TimerOutput::Scope t(computing_timer, "Output results");
    ensure_output_directory(par.output_directory);

    DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(dh);

    const std::vector<std::string> displacement_names(spacedim, "displacement");
    const std::vector<std::string> velocity_names(spacedim, "velocity");
    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
      vector_interpretation(
        spacedim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(locally_relevant_displacement,
                             displacement_names,
                             DataOut<dim, spacedim>::type_dof_data,
                             vector_interpretation);
    data_out.add_data_vector(locally_relevant_velocity,
                             velocity_names,
                             DataOut<dim, spacedim>::type_dof_data,
                             vector_interpretation);

    Vector<float> subdomain(tria->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria->locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches(*mapping_storage);

    const std::string cycle_suffix =
      par.n_refinement_cycles > 1 ?
        "_cycle_" + std::to_string(refinement_cycle_storage) :
        "";
    const std::string filename = par.output_name + cycle_suffix + "_" +
                                 std::to_string(time_step_number_storage) +
                                 ".vtu";
    data_out.write_vtu_in_parallel(par.output_directory + "/" + filename,
                                   mpi_communicator);
    cycles_and_solutions.push_back({current_time_storage, filename});

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::ofstream pvd_solutions(par.output_directory + "/" +
                                    par.output_name + cycle_suffix + ".pvd");
        DataOutBase::write_pvd_record(pvd_solutions, cycles_and_solutions);
      }
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::run_time_integration()
  {
    setup_system();
    assemble_operators();
    set_initial_conditions();

    if (par.time_parameters.output_time_interval > 0)
      output_results();

    const double initial_time     = par.time_parameters.initial_time;
    const double final_time       = par.time_parameters.final_time;
    const double time_interval    = final_time - initial_time;
    const bool   refine_time_step = par.fixed_step_parameters.refine_time_step;

    unsigned int n_steps           = 0;
    double       nominal_time_step = 0.;
    if (par.fixed_step_parameters.time_step_policy == "number_of_steps")
      {
        n_steps = par.fixed_step_parameters.number_of_steps;
        AssertThrow(time_interval <= 0. || n_steps > 0,
                    ExcMessage("Number of time steps must be positive."));
        for (unsigned int refinement = 0;
             refinement < (refine_time_step ? refinement_cycle_storage : 0);
             ++refinement)
          {
            AssertThrow(n_steps <= std::numeric_limits<unsigned int>::max() / 2,
                        ExcMessage(
                          "The refined number of time steps overflows."));
            n_steps *= 2;
          }
        nominal_time_step = n_steps > 0 ? time_interval / n_steps : 0.;
      }
    else
      {
        nominal_time_step = par.fixed_step_parameters.time_step;
        for (unsigned int refinement = 0;
             refinement < (refine_time_step ? refinement_cycle_storage : 0);
             ++refinement)
          nominal_time_step *= 0.5;

        AssertThrow(time_interval <= 0. || nominal_time_step > 0.,
                    ExcMessage("The fixed time step must be positive."));
        if (time_interval > 0.)
          n_steps = static_cast<unsigned int>(
            std::ceil(time_interval / nominal_time_step));
      }

    pcout << "   Time integration cycle " << refinement_cycle_storage + 1 << "/"
          << par.n_refinement_cycles << ": "
          << par.fixed_step_parameters.time_step_strategy
          << ", policy = " << par.fixed_step_parameters.time_step_policy
          << ", steps = " << n_steps << ", nominal dt = " << nominal_time_step
          << ", interval = [" << initial_time << ", " << final_time << "]"
          << std::endl;

    double next_output_time =
      initial_time + par.time_parameters.output_time_interval;

    for (unsigned int step = 0; step < n_steps; ++step)
      {
        const double dt =
          std::min(nominal_time_step, final_time - current_time_storage);
        pcout << "      step " << step + 1 << "/" << n_steps
              << ": t = " << current_time_storage << " -> "
              << current_time_storage + dt << ", dt = " << dt << std::endl;
        advance_one_timestep(dt);
        if (par.time_parameters.output_time_interval > 0 &&
            (current_time_storage >= next_output_time || step + 1 == n_steps))
          {
            output_results();
            while (next_output_time <= current_time_storage)
              next_output_time += par.time_parameters.output_time_interval;
          }
      }
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::run()
  {
    ensure_output_directory(par.output_directory);
    pcout << "Running ElastodynamicsSolver<"
          << Utilities::dim_string(dim, spacedim) << ">." << std::endl;
    pcout << "   Time integration: "
          << par.fixed_step_parameters.time_step_strategy << std::endl
          << "      policy: " << par.fixed_step_parameters.time_step_policy
          << std::endl
          << "      configured time step: "
          << par.fixed_step_parameters.time_step << std::endl
          << "      configured number of steps: "
          << par.fixed_step_parameters.number_of_steps << std::endl
          << "      refine time step per cycle: "
          << (par.fixed_step_parameters.refine_time_step ? "true" : "false")
          << std::endl
          << "      time interval: [" << par.time_parameters.initial_time
          << ", " << par.time_parameters.final_time << "]" << std::endl
          << "      output interval: "
          << par.time_parameters.output_time_interval << std::endl
          << "      linear solver: FGMRES/block Schur with CG/AMG, max steps = "
          << par.solver_control.max_steps()
          << ", tolerance = " << par.solver_control.tolerance() << std::endl;
    par.prm.print_parameters(par.output_directory + "/used_parameters_" +
                               std::to_string(dim) + std::to_string(spacedim) +
                               ".prm",
                             ParameterHandler::Short);

    make_grid();
    AssertThrow(
      !domain.uses_fully_distributed_triangulation() ||
        par.n_refinement_cycles <= 1,
      ExcMessage(
        "Refinement cycles are not available with the immutable "
        "parallel::fullydistributed::Triangulation in ElastodynamicsSolver."));
    setup_fe();

    for (refinement_cycle_storage = 0;
         refinement_cycle_storage < par.n_refinement_cycles;
         ++refinement_cycle_storage)
      {
        cycles_and_solutions.clear();
        run_time_integration();

        if (fe_storage != nullptr &&
            fe_storage->reference_cell().is_hyper_cube())
          {
            compute_error();
            if (pcout.is_active())
              par.convergence_table.output_table(pcout.get_stream());
          }
        else if (fe_storage != nullptr &&
                 fe_storage->reference_cell().is_simplex())
          pcout << "   Error table skipped: not available on simplex meshes."
                << std::endl;

        if (refinement_cycle_storage + 1 < par.n_refinement_cycles)
          refine_global();
      }
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::set_displacement(
    const VectorType &new_displacement)
  {
    AssertThrow(displacement_storage.size() == new_displacement.size(),
                ExcDimensionMismatch(displacement_storage.size(),
                                     new_displacement.size()));
    displacement_storage = new_displacement;
    displacement_constraints_storage.distribute(displacement_storage);
    update_locally_relevant_state();
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::set_velocity(
    const VectorType &new_velocity)
  {
    AssertThrow(velocity_storage.size() == new_velocity.size(),
                ExcDimensionMismatch(velocity_storage.size(),
                                     new_velocity.size()));
    velocity_storage = new_velocity;
    velocity_constraints_storage.distribute(velocity_storage);
    update_locally_relevant_state();
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::accept_state(
    const VectorType  &new_displacement,
    const VectorType  &new_velocity,
    const double       time,
    const unsigned int step_number)
  {
    const double previous_time = current_time_storage;
    AssertThrow(displacement_storage.size() == new_displacement.size(),
                ExcDimensionMismatch(displacement_storage.size(),
                                     new_displacement.size()));
    AssertThrow(velocity_storage.size() == new_velocity.size(),
                ExcDimensionMismatch(velocity_storage.size(),
                                     new_velocity.size()));
    AssertThrow(std::isfinite(time),
                ExcMessage("An accepted elastodynamics time must be finite."));

    update_constraints(time);
    update_boundary_operators(time);
    displacement_storage = new_displacement;
    velocity_storage     = new_velocity;
    displacement_constraints_storage.distribute(displacement_storage);
    velocity_constraints_storage.distribute(velocity_storage);
    current_time_step        = time - previous_time;
    current_time_storage     = time;
    time_step_number_storage = step_number;
    update_locally_relevant_state();
  }


  template <int dim, int spacedim>
  types::global_dof_index
  ElastodynamicsSolver<dim, spacedim>::n_dofs() const
  {
    return dh.n_dofs();
  }


  template <int dim, int spacedim>
  bool
  ElastodynamicsSolver<dim, spacedim>::state_is_finite() const
  {
    return displacement_storage.size() != 0 && velocity_storage.size() != 0 &&
           std::isfinite(displacement_storage.l2_norm()) &&
           std::isfinite(velocity_storage.l2_norm());
  }


  template <int dim, int spacedim>
  const AffineConstraints<double> &
  ElastodynamicsSolver<dim, spacedim>::hanging_node_constraints() const
  {
    return hanging_node_constraints_storage;
  }

  template <int dim, int spacedim>
  const parallel::TriangulationBase<dim, spacedim> &
  ElastodynamicsSolver<dim, spacedim>::triangulation() const
  {
    return *tria;
  }


  template <int dim, int spacedim>
  const FiniteElement<dim, spacedim> &
  ElastodynamicsSolver<dim, spacedim>::fe() const
  {
    AssertThrow(fe_storage != nullptr, ExcMessage("FE is not initialized."));
    return *fe_storage;
  }


  template <int dim, int spacedim>
  const Mapping<dim, spacedim> &
  ElastodynamicsSolver<dim, spacedim>::mapping() const
  {
    AssertThrow(mapping_storage != nullptr,
                ExcMessage("Mapping is not initialized."));
    return *mapping_storage;
  }


  template <int dim, int spacedim>
  const DoFHandler<dim, spacedim> &
  ElastodynamicsSolver<dim, spacedim>::dof_handler() const
  {
    return dh;
  }


  template <int dim, int spacedim>
  const AffineConstraints<double> &
  ElastodynamicsSolver<dim, spacedim>::constraints() const
  {
    return displacement_constraints_storage;
  }


  template <int dim, int spacedim>
  const AffineConstraints<double> &
  ElastodynamicsSolver<dim, spacedim>::velocity_constraints() const
  {
    return velocity_constraints_storage;
  }


  template <int dim, int spacedim>
  const IndexSet &
  ElastodynamicsSolver<dim, spacedim>::locally_owned_dofs() const
  {
    return owned_dofs;
  }


  template <int dim, int spacedim>
  const IndexSet &
  ElastodynamicsSolver<dim, spacedim>::locally_relevant_dofs() const
  {
    return relevant_dofs;
  }


  template <int dim, int spacedim>
  const typename ElastodynamicsSolver<dim, spacedim>::MatrixType &
  ElastodynamicsSolver<dim, spacedim>::homogeneous_mass_matrix() const
  {
    return homogeneous_mass_matrix_storage;
  }


  template <int dim, int spacedim>
  const typename ElastodynamicsSolver<dim, spacedim>::MatrixType &
  ElastodynamicsSolver<dim, spacedim>::homogeneous_stiffness_matrix() const
  {
    return homogeneous_stiffness_matrix_storage;
  }


  template <int dim, int spacedim>
  const typename ElastodynamicsSolver<dim, spacedim>::MatrixType &
  ElastodynamicsSolver<dim, spacedim>::homogeneous_damping_matrix() const
  {
    return homogeneous_damping_matrix_storage;
  }


  template <int dim, int spacedim>
  const typename ElastodynamicsSolver<dim, spacedim>::AssemblyStatistics &
  ElastodynamicsSolver<dim, spacedim>::assembly_statistics() const
  {
    return assembly_statistics_storage;
  }


  template <int dim, int spacedim>
  const typename ElastodynamicsSolver<dim, spacedim>::MatrixType &
  ElastodynamicsSolver<dim, spacedim>::mass_matrix() const
  {
    return mass_matrix_storage;
  }


  template <int dim, int spacedim>
  const typename ElastodynamicsSolver<dim, spacedim>::MatrixType &
  ElastodynamicsSolver<dim, spacedim>::diagonal_mass_matrix() const
  {
    return diagonal_mass_matrix_storage;
  }


  template <int dim, int spacedim>
  const typename ElastodynamicsSolver<dim, spacedim>::MatrixType &
  ElastodynamicsSolver<dim, spacedim>::stiffness_matrix() const
  {
    return stiffness_matrix_storage;
  }


  template <int dim, int spacedim>
  const typename ElastodynamicsSolver<dim, spacedim>::MatrixType &
  ElastodynamicsSolver<dim, spacedim>::damping_matrix() const
  {
    return damping_matrix_storage;
  }


  template <int dim, int spacedim>
  const typename ElastodynamicsSolver<dim, spacedim>::VectorType &
  ElastodynamicsSolver<dim, spacedim>::body_force_vector() const
  {
    return body_force_storage;
  }


  template <int dim, int spacedim>
  const typename ElastodynamicsSolver<dim, spacedim>::MatrixType &
  ElastodynamicsSolver<dim, spacedim>::system_matrix() const
  {
    return system_matrix_storage;
  }


  template <int dim, int spacedim>
  const typename ElastodynamicsSolver<dim, spacedim>::VectorType &
  ElastodynamicsSolver<dim, spacedim>::system_rhs() const
  {
    return system_rhs_storage;
  }


  template <int dim, int spacedim>
  const typename ElastodynamicsSolver<dim, spacedim>::VectorType &
  ElastodynamicsSolver<dim, spacedim>::displacement() const
  {
    return displacement_storage;
  }


  template <int dim, int spacedim>
  const typename ElastodynamicsSolver<dim, spacedim>::VectorType &
  ElastodynamicsSolver<dim, spacedim>::velocity() const
  {
    return velocity_storage;
  }


  template <int dim, int spacedim>
  double
  ElastodynamicsSolver<dim, spacedim>::current_time() const
  {
    return current_time_storage;
  }


  template <int dim, int spacedim>
  double
  ElastodynamicsSolver<dim, spacedim>::time_step() const
  {
    return current_time_step;
  }


  template <int dim, int spacedim>
  unsigned int
  ElastodynamicsSolver<dim, spacedim>::time_step_number() const
  {
    return time_step_number_storage;
  }


  template class ElastodynamicsParameters<1, 1>;
  template class ElastodynamicsParameters<1, 2>;
  template class ElastodynamicsParameters<1, 3>;
  template class ElastodynamicsParameters<2, 2>;
  template class ElastodynamicsParameters<2, 3>;
  template class ElastodynamicsParameters<3, 3>;

  template class ElastodynamicsSolver<1, 1>;
  template class ElastodynamicsSolver<1, 2>;
  template class ElastodynamicsSolver<1, 3>;
  template class ElastodynamicsSolver<2, 2>;
  template class ElastodynamicsSolver<2, 3>;
  template class ElastodynamicsSolver<3, 3>;
} // namespace ImmersX
