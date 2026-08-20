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

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>


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

    template <int dim, int spacedim>
    void
    read_elastodynamics_grid(const std::string &grid_file_name,
                             const std::string &ids_and_cad_file_names,
                             Triangulation<dim, spacedim> &tria)
    {
      read_grid_and_cad_files(grid_file_name, ids_and_cad_file_names, tria);
    }
  } // namespace


  template <int dim, int spacedim>
  ElastodynamicsParameters<dim, spacedim>::ElastodynamicsParameters(
    const std::string &subsection)
    : ParameterAcceptor(normalize_elastodynamics_subsection(subsection))
    , body_force(normalize_elastodynamics_subsection(subsection) +
                   "Functions/Body force",
                 spacedim)
    , displacement_boundary(normalize_elastodynamics_subsection(subsection) +
                              "Functions/Displacement boundary",
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
    add_parameter("FE degree", fe_degree, "", this->prm, Patterns::Integer(1));
    add_parameter("Output directory", output_directory);
    add_parameter("Output name", output_name);
    add_parameter("Output frequency", output_frequency);
    add_parameter("Initial refinement", initial_refinement);
    add_parameter("Number of refinement cycles", n_refinement_cycles);
    add_parameter("Dirichlet boundary ids", dirichlet_ids);

    enter_subsection("Grid generation");
    {
      add_parameter("Grid generator", name_of_grid);
      add_parameter("Grid generator arguments", arguments_for_grid);
      add_parameter("Triangulation type",
                    triangulation_type,
                    "",
                    this->prm,
                    Patterns::Selection("distributed|fullydistributed"));
    }
    leave_subsection();

    enter_subsection("Material");
    {
      add_parameter("Density", density, "", this->prm, Patterns::Double(0));
      add_parameter("Lame mu", lame_mu, "", this->prm, Patterns::Double(0));
      add_parameter(
        "Lame lambda", lame_lambda, "", this->prm, Patterns::Double(0));
      add_parameter(
        "Damping shear", damping_shear, "", this->prm, Patterns::Double(0));
      add_parameter(
        "Damping bulk", damping_bulk, "", this->prm, Patterns::Double(0));
    }
    leave_subsection();

    enter_subsection("Time integration");
    {
      add_parameter("Initial time", initial_time);
      add_parameter("Final time", final_time);
      add_parameter("Time step", time_step, "", this->prm, Patterns::Double(0));
      add_parameter("Number of time steps", number_of_steps);
    }
    leave_subsection();

    const auto declare_zero_vector_function = [this]() {
      const std::string zero_expression = spacedim == 2 ? "0; 0" : "0; 0; 0";
      this->prm.declare_entry(
        "Function expression",
        zero_expression,
        Patterns::List(Patterns::Anything(), spacedim, spacedim, ";"));
    };

    body_force.declare_parameters_call_back.connect(
      declare_zero_vector_function);
    displacement_boundary.declare_parameters_call_back.connect(
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

    parse_parameters_call_back.connect([this]() {
      ensure_output_directory(output_directory);
      AssertThrow(density > 0., ExcMessage("Density must be positive."));
      AssertThrow(lame_mu > 0., ExcMessage("Lame mu must be positive."));
      AssertThrow(lame_lambda >= 0.,
                  ExcMessage("Lame lambda must be non-negative."));
      AssertThrow(damping_shear >= 0. && damping_bulk >= 0.,
                  ExcMessage("Damping coefficients must be non-negative."));
      AssertThrow(final_time >= initial_time,
                  ExcMessage("Final time must not precede initial time."));
      if (final_time > initial_time || number_of_steps > 0)
        AssertThrow(time_step > 0.,
                    ExcMessage("Time step must be positive for a transient "
                               "run."));
    });
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
    , triangulation_storage(make_triangulation_storage(mpi_communicator))
    , tria(&std::visit(
        [](
          auto &selected_tria) -> parallel::TriangulationBase<dim, spacedim> & {
          return selected_tria;
        },
        triangulation_storage))
    , dh()
    , current_time_storage(par.initial_time)
  {}


  template <int dim, int spacedim>
  typename ElastodynamicsSolver<dim, spacedim>::TriangulationVariant
  ElastodynamicsSolver<dim, spacedim>::make_triangulation_storage(
    MPI_Comm mpi_communicator)
  {
    return TriangulationVariant(
      std::in_place_type<DistributedTriangulation>,
      mpi_communicator,
      typename Triangulation<dim, spacedim>::MeshSmoothing(
        Triangulation<dim, spacedim>::smoothing_on_refinement |
        Triangulation<dim, spacedim>::smoothing_on_coarsening),
      parallel::distributed::Triangulation<dim, spacedim>::
        construct_multigrid_hierarchy);
  }


  template <int dim, int spacedim>
  bool
  ElastodynamicsSolver<dim, spacedim>::uses_fully_distributed_triangulation()
    const
  {
    return std::holds_alternative<FullyDistributedTriangulation>(
      triangulation_storage);
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::make_grid()
  {
    TimerOutput::Scope t(computing_timer, "Make grid");

    const bool need_fully_distributed =
      par.triangulation_type == "fullydistributed";

    if (need_fully_distributed && !uses_fully_distributed_triangulation())
      triangulation_storage.template emplace<FullyDistributedTriangulation>(
        mpi_communicator);
    else if (!need_fully_distributed && uses_fully_distributed_triangulation())
      triangulation_storage.template emplace<DistributedTriangulation>(
        mpi_communicator,
        typename Triangulation<dim, spacedim>::MeshSmoothing(
          Triangulation<dim, spacedim>::smoothing_on_refinement |
          Triangulation<dim, spacedim>::smoothing_on_coarsening),
        parallel::distributed::Triangulation<dim, spacedim>::
          construct_multigrid_hierarchy);

    tria = &std::visit(
      [](auto &selected_tria) -> parallel::TriangulationBase<dim, spacedim> & {
        return selected_tria;
      },
      triangulation_storage);
    dh.reinit(*tria);

    if (!uses_fully_distributed_triangulation())
      {
        auto &distributed_tria =
          std::get<DistributedTriangulation>(triangulation_storage);
        try
          {
            GridGenerator::generate_from_name_and_arguments(
              distributed_tria, par.name_of_grid, par.arguments_for_grid);
          }
        catch (...)
          {
            pcout << "Generating from name and arguments failed.\n"
                  << "Trying to read the grid from a file." << std::endl;
            read_elastodynamics_grid(par.name_of_grid,
                                     par.arguments_for_grid,
                                     distributed_tria);
          }

        distributed_tria.refine_global(par.initial_refinement);
        pcout << "   Triangulation backend: distributed\n"
              << "   Number of active cells: " << tria->n_active_cells()
              << std::endl;
        return;
      }

    Triangulation<dim, spacedim> serial_tria(
      typename Triangulation<dim, spacedim>::MeshSmoothing(
        Triangulation<dim, spacedim>::smoothing_on_refinement |
        Triangulation<dim, spacedim>::smoothing_on_coarsening));
    try
      {
        GridGenerator::generate_from_name_and_arguments(serial_tria,
                                                        par.name_of_grid,
                                                        par.arguments_for_grid);
      }
    catch (...)
      {
        pcout << "Generating from name and arguments failed.\n"
              << "Trying to read the grid from a file." << std::endl;
        read_elastodynamics_grid(par.name_of_grid,
                                 par.arguments_for_grid,
                                 serial_tria);
      }

    serial_tria.refine_global(par.initial_refinement);
    auto &fully_distributed_tria =
      std::get<FullyDistributedTriangulation>(triangulation_storage);
    for (const auto manifold_id : serial_tria.get_manifold_ids())
      if (manifold_id != numbers::flat_manifold_id)
        fully_distributed_tria.set_manifold(
          manifold_id, serial_tria.get_manifold(manifold_id));
    fully_distributed_tria.copy_triangulation(serial_tria);

    pcout << "   Triangulation backend: fullydistributed\n"
          << "   Number of active cells: " << tria->n_active_cells()
          << std::endl;
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::setup_fe()
  {
    TimerOutput::Scope t(computing_timer, "Initial setup");
    fe_storage = std::make_unique<FESystem<dim, spacedim>>(
      FE_Q<dim, spacedim>(par.fe_degree), spacedim);
    quadrature      = std::make_unique<QGauss<dim>>(par.fe_degree + 1);
    mapping_storage = std::make_unique<MappingQ<dim, spacedim>>(1);
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
  ElastodynamicsSolver<dim, spacedim>::update_constraints(const double time)
  {
    displacement_constraints_storage.clear();
    displacement_constraints_storage.reinit(owned_dofs, relevant_dofs);
    DoFTools::make_hanging_node_constraints(dh,
                                            displacement_constraints_storage);
    par.displacement_boundary.set_time(time);
    for (const auto id : par.dirichlet_ids)
      VectorTools::interpolate_boundary_values(
        dh, id, par.displacement_boundary, displacement_constraints_storage);
    displacement_constraints_storage.close();

    velocity_constraints_storage.clear();
    velocity_constraints_storage.reinit(owned_dofs, relevant_dofs);
    DoFTools::make_hanging_node_constraints(dh, velocity_constraints_storage);
    par.velocity_boundary.set_time(time);
    for (const auto id : par.dirichlet_ids)
      VectorTools::interpolate_boundary_values(dh,
                                               id,
                                               par.velocity_boundary,
                                               velocity_constraints_storage);
    velocity_constraints_storage.close();

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

    update_constraints(par.initial_time);

    DynamicSparsityPattern spatial_dsp(relevant_dofs);
    DoFTools::make_sparsity_pattern(dh,
                                    spatial_dsp,
                                    displacement_constraints_storage,
                                    true);
    SparsityTools::distribute_sparsity_pattern(spatial_dsp,
                                               owned_dofs,
                                               mpi_communicator,
                                               relevant_dofs);

    DynamicSparsityPattern               combined_dsp(combined_relevant_dofs);
    std::vector<types::global_dof_index> cell_dof_indices(
      fe_storage->n_dofs_per_cell());
    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(cell_dof_indices);
          for (const auto i : cell_dof_indices)
            for (const auto j : cell_dof_indices)
              {
                combined_dsp.add(i, j);
                combined_dsp.add(i, n_spatial_dofs + j);
                combined_dsp.add(n_spatial_dofs + i, j);
                combined_dsp.add(n_spatial_dofs + i, n_spatial_dofs + j);
              }

          for (const auto i : cell_dof_indices)
            if (displacement_constraints_storage.is_constrained(i))
              {
                const auto *entries =
                  displacement_constraints_storage.get_constraint_entries(i);
                if (entries != nullptr)
                  for (const auto &entry : *entries)
                    for (const auto j : cell_dof_indices)
                      {
                        combined_dsp.add(entry.first, j);
                        combined_dsp.add(j, entry.first);
                        combined_dsp.add(n_spatial_dofs + entry.first,
                                         n_spatial_dofs + j);
                        combined_dsp.add(n_spatial_dofs + j,
                                         n_spatial_dofs + entry.first);
                      }
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
    stiffness_matrix_storage.reinit(owned_dofs,
                                    owned_dofs,
                                    spatial_dsp,
                                    mpi_communicator);
    damping_matrix_storage.reinit(owned_dofs,
                                  owned_dofs,
                                  spatial_dsp,
                                  mpi_communicator);
    system_matrix_storage.reinit(combined_owned_dofs,
                                 combined_owned_dofs,
                                 combined_dsp,
                                 mpi_communicator);

    body_force_storage.reinit(owned_dofs, mpi_communicator);
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
    TimerOutput::Scope t(computing_timer, "Assemble spatial operators");
    AssertThrow(fe_storage != nullptr && quadrature != nullptr,
                ExcMessage("setup_fe() must be called before assembly."));

    mass_matrix_storage      = 0.;
    stiffness_matrix_storage = 0.;
    damping_matrix_storage   = 0.;

    AffineConstraints<double> no_constraints;
    no_constraints.close();

    FEValues<dim, spacedim>          fe_values(*fe_storage,
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
                      par.density * (values[i] * values[j]) * jxw;
                    cell_stiffness(i, j) +=
                      (2. * par.lame_mu *
                         scalar_product(symmetric_gradients[i],
                                        symmetric_gradients[j]) +
                       par.lame_lambda * divergences[i] * divergences[j]) *
                      jxw;
                    cell_damping(i, j) +=
                      (2. * par.damping_shear *
                         scalar_product(symmetric_gradients[i],
                                        symmetric_gradients[j]) +
                       par.damping_bulk * divergences[i] * divergences[j]) *
                      jxw;
                  }
            }

          cell->get_dof_indices(local_dof_indices);
          no_constraints.distribute_local_to_global(cell_mass,
                                                    local_dof_indices,
                                                    mass_matrix_storage);
          no_constraints.distribute_local_to_global(cell_stiffness,
                                                    local_dof_indices,
                                                    stiffness_matrix_storage);
          no_constraints.distribute_local_to_global(cell_damping,
                                                    local_dof_indices,
                                                    damping_matrix_storage);
        }

    mass_matrix_storage.compress(VectorOperation::add);
    stiffness_matrix_storage.compress(VectorOperation::add);
    damping_matrix_storage.compress(VectorOperation::add);
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
    TimerOutput::Scope t(computing_timer, "Assemble body force");
    par.body_force.set_time(time);
    destination.reinit(owned_dofs, mpi_communicator);
    destination = 0.;

    AffineConstraints<double> no_constraints;
    no_constraints.close();
    FEValues<dim, spacedim>          fe_values(*fe_storage,
                                      *quadrature,
                                      update_values | update_quadrature_points |
                                        update_JxW_values);
    const FEValuesExtractors::Vector vector_field(0);
    const unsigned int          dofs_per_cell = fe_storage->n_dofs_per_cell();
    const unsigned int          n_q_points    = quadrature->size();
    Vector<double>              cell_rhs(dofs_per_cell);
    std::vector<Vector<double>> force_values(n_q_points,
                                             Vector<double>(spacedim));
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_rhs = 0.;
          fe_values.reinit(cell);
          par.body_force.vector_value_list(fe_values.get_quadrature_points(),
                                           force_values);
          for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const auto component =
                  fe_storage->system_to_component_index(i).first;
                cell_rhs(i) += fe_values[vector_field].value(i, q)[component] *
                               force_values[q](component) * fe_values.JxW(q);
              }

          cell->get_dof_indices(local_dof_indices);
          no_constraints.distribute_local_to_global(cell_rhs,
                                                    local_dof_indices,
                                                    destination);
        }

    destination.compress(VectorOperation::add);
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::set_initial_conditions()
  {
    TimerOutput::Scope t(computing_timer, "Set initial conditions");
    current_time_storage     = par.initial_time;
    time_step_number_storage = 0;
    current_time_step        = 0.;
    update_constraints(current_time_storage);

    par.initial_displacement.set_time(current_time_storage);
    par.initial_velocity.set_time(current_time_storage);
    VectorTools::interpolate(dh,
                             par.initial_displacement,
                             displacement_storage);
    VectorTools::interpolate(dh, par.initial_velocity, velocity_storage);
    displacement_constraints_storage.distribute(displacement_storage);
    velocity_constraints_storage.distribute(velocity_storage);
    update_locally_relevant_state();
    assemble_body_force(current_time_storage);
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::assemble_backward_euler_system(
    const VectorType &previous_displacement,
    const VectorType &previous_velocity,
    const double      dt)
  {
    TimerOutput::Scope t(computing_timer, "Assemble backward-Euler system");
    system_matrix_storage = 0.;
    system_rhs_storage    = 0.;

    locally_relevant_displacement = previous_displacement;
    locally_relevant_displacement.update_ghost_values();
    locally_relevant_velocity = previous_velocity;
    locally_relevant_velocity.update_ghost_values();

    FEValues<dim, spacedim>          fe_values(*fe_storage,
                                      *quadrature,
                                      update_values | update_gradients |
                                        update_quadrature_points |
                                        update_JxW_values);
    const FEValuesExtractors::Vector vector_field(0);
    const unsigned int dofs_per_cell  = fe_storage->n_dofs_per_cell();
    const unsigned int n_q_points     = quadrature->size();
    const auto         n_spatial_dofs = dh.n_dofs();

    FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiffness(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_damping(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix(2 * dofs_per_cell, 2 * dofs_per_cell);
    Vector<double>     cell_rhs(2 * dofs_per_cell);
    std::vector<Tensor<2, spacedim>>     symmetric_gradients(dofs_per_cell);
    std::vector<double>                  divergences(dofs_per_cell);
    std::vector<Tensor<1, spacedim>>     values(dofs_per_cell);
    std::vector<Vector<double>>          force_values(n_q_points,
                                             Vector<double>(spacedim));
    std::vector<types::global_dof_index> spatial_indices(dofs_per_cell);
    std::vector<types::global_dof_index> combined_indices(2 * dofs_per_cell);

    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_mass      = 0.;
          cell_stiffness = 0.;
          cell_damping   = 0.;
          cell_matrix    = 0.;
          cell_rhs       = 0.;
          fe_values.reinit(cell);
          par.body_force.vector_value_list(fe_values.get_quadrature_points(),
                                           force_values);

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
                      par.density * (values[i] * values[j]) * jxw;
                    cell_stiffness(i, j) +=
                      (2. * par.lame_mu *
                         scalar_product(symmetric_gradients[i],
                                        symmetric_gradients[j]) +
                       par.lame_lambda * divergences[i] * divergences[j]) *
                      jxw;
                    cell_damping(i, j) +=
                      (2. * par.damping_shear *
                         scalar_product(symmetric_gradients[i],
                                        symmetric_gradients[j]) +
                       par.damping_bulk * divergences[i] * divergences[j]) *
                      jxw;
                  }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const auto component =
                    fe_storage->system_to_component_index(i).first;
                  cell_rhs(dofs_per_cell + i) +=
                    fe_values[vector_field].value(i, q)[component] *
                    force_values[q](component) * fe_values.JxW(q);
                }
            }

          cell->get_dof_indices(spatial_indices);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              combined_indices[i] = spatial_indices[i];
              combined_indices[dofs_per_cell + i] =
                n_spatial_dofs + spatial_indices[i];
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += cell_mass(i, j) / dt;
                  cell_matrix(i, dofs_per_cell + j) -= cell_mass(i, j);
                  cell_matrix(dofs_per_cell + i, j) += cell_stiffness(i, j);
                  cell_matrix(dofs_per_cell + i, dofs_per_cell + j) +=
                    cell_mass(i, j) / dt + cell_damping(i, j);
                  cell_rhs(i) +=
                    cell_mass(i, j) / dt *
                    locally_relevant_displacement(spatial_indices[j]);
                  cell_rhs(dofs_per_cell + i) +=
                    cell_mass(i, j) / dt *
                    locally_relevant_velocity(spatial_indices[j]);
                }
            }

          combined_constraints_storage.distribute_local_to_global(
            cell_matrix,
            cell_rhs,
            combined_indices,
            system_matrix_storage,
            system_rhs_storage,
            true);
        }

    system_matrix_storage.compress(VectorOperation::add);
    system_rhs_storage.compress(VectorOperation::add);
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::solve_backward_euler_system()
  {
    TimerOutput::Scope t(computing_timer, "Solve backward-Euler system");

    const auto n_spatial_dofs = dh.n_dofs();
    VectorType combined_solution;
    combined_solution.reinit(combined_owned_dofs, mpi_communicator);
    for (const auto index : owned_dofs)
      {
        combined_solution(index)                  = displacement_storage(index);
        combined_solution(n_spatial_dofs + index) = velocity_storage(index);
      }
    combined_constraints_storage.distribute(combined_solution);

    LA::MPI::PreconditionJacobi preconditioner;
    preconditioner.initialize(system_matrix_storage);
    SolverGMRES<VectorType> solver(par.solver_control);
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

    pcout << "   GMRES iterations: " << par.solver_control.last_step()
          << std::endl;
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::advance_one_timestep()
  {
    AssertThrow(par.time_step > 0.,
                ExcMessage("Cannot advance with a non-positive time step."));

    advance_one_timestep(par.time_step);
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::advance_one_timestep(const double dt)
  {
    AssertThrow(dt > 0.,
                ExcMessage("Cannot advance with a non-positive time step."));

    const auto previous_displacement = displacement_storage;
    const auto previous_velocity     = velocity_storage;
    const auto next_time             = current_time_storage + dt;

    update_constraints(next_time);
    assemble_body_force(next_time);
    assemble_backward_euler_system(previous_displacement,
                                   previous_velocity,
                                   dt);
    solve_backward_euler_system();

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
    data_out.build_patches();

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

    if (par.output_frequency > 0)
      output_results();

    unsigned int n_steps = par.number_of_steps;
    if (n_steps == 0 && par.final_time > par.initial_time)
      n_steps = static_cast<unsigned int>(
        std::ceil((par.final_time - par.initial_time) / par.time_step));

    for (unsigned int step = 0; step < n_steps; ++step)
      {
        double dt = par.time_step;
        if (par.number_of_steps == 0)
          dt = std::min(dt, par.final_time - current_time_storage);
        advance_one_timestep(dt);
        if (par.output_frequency > 0 &&
            (time_step_number_storage % par.output_frequency == 0 ||
             step + 1 == n_steps))
          output_results();
      }
  }


  template <int dim, int spacedim>
  void
  ElastodynamicsSolver<dim, spacedim>::run()
  {
    ensure_output_directory(par.output_directory);
    pcout << "Running ElastodynamicsSolver<"
          << Utilities::dim_string(dim, spacedim) << ">." << std::endl;
    par.prm.print_parameters(par.output_directory + "/used_parameters_" +
                               std::to_string(dim) + std::to_string(spacedim) +
                               ".prm",
                             ParameterHandler::Short);

    make_grid();
    setup_fe();

    for (refinement_cycle_storage = 0;
         refinement_cycle_storage < par.n_refinement_cycles;
         ++refinement_cycle_storage)
      {
        cycles_and_solutions.clear();
        run_time_integration();

        par.exact_solution.set_time(current_time_storage);
        par.convergence_table.error_from_exact(dh,
                                               locally_relevant_displacement,
                                               par.exact_solution);

        if (pcout.is_active())
          par.convergence_table.output_table(pcout.get_stream());

        if (refinement_cycle_storage + 1 < par.n_refinement_cycles)
          {
            tria->refine_global(1);
            dh.clear();
          }
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
  ElastodynamicsSolver<dim, spacedim>::mass_matrix() const
  {
    return mass_matrix_storage;
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


  template class ElastodynamicsParameters<2>;
  template class ElastodynamicsParameters<3>;

  template class ElastodynamicsSolver<2>;
  template class ElastodynamicsSolver<3>;
} // namespace ImmersX
