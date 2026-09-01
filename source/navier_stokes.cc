// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <immersx/io/utils.h>
#include <immersx/physics/navier_stokes.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

namespace ImmersX
{
  using namespace dealii;


  namespace
  {
    template <int dim>
    std::vector<std::string>
    navier_stokes_error_component_names()
    {
      std::vector<std::string> names(dim, "u");
      names.emplace_back("p");
      return names;
    }


    template <int dim>
    std::vector<std::set<VectorTools::NormType>>
    navier_stokes_error_norms()
    {
      return {{VectorTools::L2_norm}, {}};
    }


    std::string
    normalize_navier_stokes_subsection(const std::string &subsection)
    {
      if (subsection.empty())
        return "/Navier-Stokes/";

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
  } // namespace


  template <int dim, int spacedim>
  NavierStokesParameters<dim, spacedim>::NavierStokesParameters(
    const std::string &subsection)
    : ParameterAcceptor(normalize_navier_stokes_subsection(subsection))
    , convergence_table(navier_stokes_error_component_names<dim>(),
                        navier_stokes_error_norms<dim>())
    , rhs(normalize_navier_stokes_subsection(subsection) + "Right hand side",
          spacedim + 1)
    , bc(normalize_navier_stokes_subsection(subsection) +
           "Dirichlet boundary conditions",
         spacedim + 1)
    , initial_condition(normalize_navier_stokes_subsection(subsection) +
                          "Initial condition",
                        spacedim + 1)
    , solver_control(normalize_navier_stokes_subsection(subsection) +
                     "Solver/Control")
  {
    add_parameter("Output directory", output_directory);
    add_parameter("Output name", output_name);
    add_parameter("Output frequency", output_frequency);

    enter_subsection("Finite element spaces");
    {
      add_parameter("Velocity degree",
                    velocity_degree,
                    "Degree of the scalar finite element used for velocity",
                    this->prm,
                    Patterns::Integer(1));
      add_parameter("Pressure degree",
                    pressure_degree,
                    "Degree of the scalar finite element used for pressure",
                    this->prm,
                    Patterns::Integer(1));
    }
    leave_subsection();

    add_parameter("Initial refinement", initial_refinement);
    add_parameter("Dirichlet boundary ids", dirichlet_ids);

    enter_subsection("Grid generation");
    {
      add_parameter("Grid generator", name_of_grid);
      add_parameter("Grid generator arguments", arguments_for_grid);
      add_parameter("Triangulation type",
                    triangulation_type,
                    "Distributed backend used for the fluid mesh",
                    this->prm,
                    Patterns::Selection("distributed|fullydistributed"));
    }
    leave_subsection();

    enter_subsection("Physical properties");
    {
      add_parameter("Density", density, "Constant fluid density");
      add_parameter("Viscosity", viscosity, "Constant kinematic viscosity");
      add_parameter(
        "Include convective term",
        include_convective_term,
        "Evaluate rho (u^n . grad) u^n explicitly on the right-hand side. "
        "Set false for unsteady Stokes.");
    }
    leave_subsection();

    enter_subsection("Time stepping");
    {
      add_parameter("Policy",
                    time_step_policy,
                    "Use number_of_steps to divide the time interval, or "
                    "fixed to use the configured time step",
                    this->prm,
                    Patterns::Selection("number_of_steps|fixed"));
      add_parameter("Initial time", initial_time);
      add_parameter("Final time", final_time);
      add_parameter("Time step", time_step);
      add_parameter("Number of time steps", number_of_time_steps);
    }
    leave_subsection();

    add_parameter(
      "Analytical solution expression",
      analytical_solution_expression,
      "Optional exact velocity/pressure expression for error "
      "postprocessing. The solver does not use it during assembly.");

    this->prm.enter_subsection("Error");
    convergence_table.add_parameters(this->prm);
    this->prm.leave_subsection();

    enter_subsection("Solver");
    {
      add_parameter("Inner maximum steps", inner_solver_max_steps);
      add_parameter("Inner tolerance", inner_solver_tolerance);
      add_parameter("Log iterations", log_solver_iterations);
    }
    leave_subsection();

    const auto reset_function = [this](const std::string &expression) {
      const unsigned int n_components =
        Utilities::split_string_list(expression, ";").size();
      this->prm.declare_entry(
        "Function expression",
        expression,
        Patterns::List(Patterns::Anything(), n_components, n_components, ";"));
    };

    const auto helper = [reset_function](auto              &function,
                                         const std::string &expression) {
      function.declare_parameters_call_back.connect(
        [reset_function, expression]() { reset_function(expression); });
    };

    const std::string zero_velocity = dim == 2 ? "0; 0; 0" : "0; 0; 0; 0";
    helper(rhs, zero_velocity);
    helper(bc, zero_velocity);
    helper(initial_condition, zero_velocity);

    parse_parameters_call_back.connect(
      [this]() { ensure_output_directory(output_directory); });
  }


  template <int dim, int spacedim>
  void
  NavierStokesParameters<dim, spacedim>::set_time(const double time) const
  {
    rhs.set_time(time);
    bc.set_time(time);
    initial_condition.set_time(time);
  }


  template <int dim, int spacedim>
  NavierStokesSolver<dim, spacedim>::NavierStokesSolver(
    const NavierStokesParameters<dim, spacedim> &par)
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
    , velocity(0)
    , pressure(dim)
    , current_time_storage(par.initial_time)
  {}


  template <int dim, int spacedim>
  typename NavierStokesSolver<dim, spacedim>::TriangulationVariant
  NavierStokesSolver<dim, spacedim>::make_triangulation_storage(
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
  NavierStokesSolver<dim, spacedim>::uses_fully_distributed_triangulation()
    const
  {
    return std::holds_alternative<FullyDistributedTriangulation>(
      triangulation_storage);
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::make_grid()
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
            AssertThrow(false,
                        ExcMessage(
                          "Could not generate the Navier-Stokes grid from '" +
                          par.name_of_grid + "' and its arguments."));
          }
        distributed_tria.refine_global(par.initial_refinement);
      }
    else
      {
        Triangulation<dim, spacedim> serial_tria(
          typename Triangulation<dim, spacedim>::MeshSmoothing(
            Triangulation<dim, spacedim>::smoothing_on_refinement |
            Triangulation<dim, spacedim>::smoothing_on_coarsening));
        try
          {
            GridGenerator::generate_from_name_and_arguments(
              serial_tria, par.name_of_grid, par.arguments_for_grid);
          }
        catch (...)
          {
            AssertThrow(false,
                        ExcMessage(
                          "Could not generate the Navier-Stokes grid from '" +
                          par.name_of_grid + "' and its arguments."));
          }
        serial_tria.refine_global(par.initial_refinement);
        std::get<FullyDistributedTriangulation>(triangulation_storage)
          .copy_triangulation(serial_tria);
      }

    pcout << "   Number of active cells: " << tria->n_active_cells() << " ("
          << (uses_fully_distributed_triangulation() ? "fullydistributed" :
                                                       "distributed")
          << ")" << std::endl;
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::setup_fe()
  {
    TimerOutput::Scope t(computing_timer, "Initial setup");

    AssertThrow(par.pressure_degree < par.velocity_degree,
                ExcMessage("The pressure degree must be smaller than the "
                           "velocity degree for the Taylor-Hood pair."));

    fe = std::make_unique<FESystem<dim, spacedim>>(
      FE_Q<dim, spacedim>(par.velocity_degree),
      dim,
      FE_Q<dim, spacedim>(par.pressure_degree),
      1);
    quadrature    = std::make_unique<QGauss<dim>>(par.velocity_degree + 1);
    velocity_mask = fe->component_mask(velocity);
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::initialize_time_control()
  {
    current_time_storage    = par.initial_time;
    timestep_number_storage = 0;

    AssertThrow(par.final_time > par.initial_time,
                ExcMessage("Final time must be larger than initial time."));

    if (par.time_step_policy == "number_of_steps")
      {
        AssertThrow(par.number_of_time_steps > 0,
                    ExcMessage("Number of time steps must be positive."));
        n_time_steps_storage = par.number_of_time_steps;
        time_step_storage =
          (par.final_time - par.initial_time) / n_time_steps_storage;
      }
    else
      {
        AssertThrow(par.time_step > 0.,
                    ExcMessage("The fixed time step must be positive."));
        n_time_steps_storage = static_cast<unsigned int>(
          std::ceil((par.final_time - par.initial_time) / par.time_step));
        time_step_storage = par.time_step;
      }
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::update_constraints()
  {
    TimerOutput::Scope t(computing_timer, "Update constraints");

    par.set_time(current_time_storage);
    constraints_storage.clear();
    constraints_storage.reinit(owned_dofs, relevant_dofs);
    DoFTools::make_hanging_node_constraints(dh, constraints_storage);

    for (const auto boundary_id : par.dirichlet_ids)
      VectorTools::interpolate_boundary_values(
        dh, boundary_id, par.bc, constraints_storage, velocity_mask);

    // Component-wise renumbering places the pressure block after all velocity
    // DoFs, so this global index is the same on every MPI rank.
    const types::global_dof_index pinned_pressure_dof = dofs_per_block[0];

    AssertThrow(dofs_per_block[1] > 0,
                ExcMessage("The pressure finite-element space has no degrees "
                           "of freedom."));

    // Add the pressure pin on every rank on which the global DoF is relevant.
    // This is the standard distributed AffineConstraints ownership pattern.
    if (relevant_dofs.is_element(pinned_pressure_dof))
      constraints_storage.constrain_dof_to_zero(pinned_pressure_dof);

    constraints_storage.close();
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "Setup system");
    AssertThrow(fe != nullptr && quadrature != nullptr,
                ExcMessage("setup_fe() must be called first."));

    initialize_time_control();
    dh.distribute_dofs(*fe);

    std::vector<unsigned int> sub_blocks(dim + 1, 0);
    sub_blocks[dim] = 1;
    DoFRenumbering::component_wise(dh, sub_blocks);

    dofs_per_block = DoFTools::count_dofs_per_fe_block(dh, sub_blocks);
    AssertThrow(dofs_per_block.size() == 2 && dofs_per_block[0] > 0 &&
                  dofs_per_block[1] > 0,
                ExcMessage(
                  "The velocity/pressure DoF blocks must be nonempty."));

    owned_dofs    = dh.locally_owned_dofs();
    relevant_dofs = DoFTools::extract_locally_relevant_dofs(dh);
    owned_dofs_by_block.resize(2);
    relevant_dofs_by_block.resize(2);
    owned_dofs_by_block[0] = owned_dofs.get_view(0, dofs_per_block[0]);
    owned_dofs_by_block[1] =
      owned_dofs.get_view(dofs_per_block[0], dh.n_dofs());
    relevant_dofs_by_block[0] = relevant_dofs.get_view(0, dofs_per_block[0]);
    relevant_dofs_by_block[1] =
      relevant_dofs.get_view(dofs_per_block[0], dh.n_dofs());

    update_constraints();

    const auto owned_dofs_per_processor =
      Utilities::MPI::all_gather(mpi_communicator, owned_dofs);

    Table<2, DoFTools::Coupling> system_coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        system_coupling[c][d] =
          c == dim && d == dim ? DoFTools::none : DoFTools::always;

    BlockDynamicSparsityPattern system_dsp(dofs_per_block, dofs_per_block);
    DoFTools::make_sparsity_pattern(
      dh, system_coupling, system_dsp, constraints_storage, false);
    SparsityTools::distribute_sparsity_pattern(system_dsp,
                                               owned_dofs_per_processor,
                                               mpi_communicator,
                                               relevant_dofs);
    system_matrix_storage.reinit(owned_dofs_by_block,
                                 system_dsp,
                                 mpi_communicator);
    continuous_operator_storage.reinit(owned_dofs_by_block,
                                       system_dsp,
                                       mpi_communicator);

    Table<2, DoFTools::Coupling> mass_coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        mass_coupling[c][d] = (c < dim && d < dim) || (c == dim && d == dim) ?
                                DoFTools::always :
                                DoFTools::none;

    BlockDynamicSparsityPattern mass_dsp(dofs_per_block, dofs_per_block);
    DoFTools::make_sparsity_pattern(
      dh, mass_coupling, mass_dsp, constraints_storage, false);
    SparsityTools::distribute_sparsity_pattern(mass_dsp,
                                               owned_dofs_per_processor,
                                               mpi_communicator,
                                               relevant_dofs);
    mass_matrix_storage.reinit(owned_dofs_by_block, mass_dsp, mpi_communicator);

    solution_storage.reinit(owned_dofs_by_block, mpi_communicator);
    previous_solution_storage.reinit(owned_dofs_by_block, mpi_communicator);
    system_rhs_storage.reinit(owned_dofs_by_block, mpi_communicator);
    locally_relevant_solution_storage.reinit(owned_dofs_by_block,
                                             relevant_dofs_by_block,
                                             mpi_communicator);

    solution_storage          = 0.;
    previous_solution_storage = 0.;
    system_rhs_storage        = 0.;
    interpolate_initial_condition();

    pcout << "   Number of degrees of freedom: " << dh.n_dofs() << " ("
          << dofs_per_block[0] << '+' << dofs_per_block[1] << ")" << std::endl;
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::interpolate_initial_condition()
  {
    par.set_time(current_time_storage);
    VectorTools::interpolate(dh,
                             par.initial_condition,
                             solution_storage,
                             velocity_mask);
    solution_storage.block(1) = 0.;
    constraints_storage.distribute(solution_storage);
    previous_solution_storage = solution_storage;
    update_locally_relevant_solution();
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "Assemble Navier-Stokes system");
    AssertThrow(fe != nullptr && quadrature != nullptr,
                ExcMessage("setup_fe() must be called before assembly."));

    par.set_time(current_time_storage);
    system_matrix_storage       = 0.;
    mass_matrix_storage         = 0.;
    continuous_operator_storage = 0.;
    system_rhs_storage          = 0.;

    FEValues<dim, spacedim> fe_values(mapping_storage,
                                      *fe,
                                      *quadrature,
                                      update_values | update_gradients |
                                        update_quadrature_points |
                                        update_JxW_values);

    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature->size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_continuous_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<Vector<double>>               rhs_values(n_q_points,
                                           Vector<double>(spacedim + 1));
    std::vector<Tensor<1, spacedim>>          phi_u(dofs_per_cell);
    std::vector<SymmetricTensor<2, spacedim>> sym_grad_phi_u(dofs_per_cell);
    std::vector<double>                       div_phi_u(dofs_per_cell);
    std::vector<double>                       phi_p(dofs_per_cell);
    Tensor<1, spacedim>                       rhs_u;
    std::vector<Tensor<1, spacedim>>          old_u(n_q_points);
    std::vector<Tensor<2, spacedim>>          old_grad_u(n_q_points);
    std::vector<types::global_dof_index>      local_dof_indices(dofs_per_cell);

    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix            = 0.;
          cell_continuous_matrix = 0.;
          cell_mass_matrix       = 0.;
          cell_rhs               = 0.;

          fe_values.reinit(cell);
          par.rhs.vector_value_list(fe_values.get_quadrature_points(),
                                    rhs_values);
          fe_values[velocity].get_function_values(
            locally_relevant_solution_storage, old_u);
          fe_values[velocity].get_function_gradients(
            locally_relevant_solution_storage, old_grad_u);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  phi_u[k] = fe_values[velocity].value(k, q);
                  sym_grad_phi_u[k] =
                    fe_values[velocity].symmetric_gradient(k, q);
                  div_phi_u[k] = fe_values[velocity].divergence(k, q);
                  phi_p[k]     = fe_values[pressure].value(k, q);
                }

              const Tensor<1, spacedim> convective =
                par.include_convective_term ? old_grad_u[q] * old_u[q] :
                                              Tensor<1, spacedim>();
              rhs_u = Tensor<1, spacedim>();
              for (unsigned int d = 0; d < dim; ++d)
                rhs_u[d] = rhs_values[q][d];

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const unsigned int component_i =
                    fe->system_to_component_index(i).first;
                  if (component_i < dim)
                    cell_rhs(i) +=
                      par.density *
                      (rhs_u * phi_u[i] +
                       (1. / time_step_storage) * (old_u[q] * phi_u[i]) -
                       convective * phi_u[i]) *
                      fe_values.JxW(q);

                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      const auto spatial_term =
                        (2. * par.viscosity *
                           scalar_product(sym_grad_phi_u[i],
                                          sym_grad_phi_u[j]) -
                         div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j]) *
                        fe_values.JxW(q);
                      cell_continuous_matrix(i, j) += spatial_term;
                      cell_matrix(i, j) += ((par.density / time_step_storage) *
                                            phi_u[i] * phi_u[j]) *
                                             fe_values.JxW(q) +
                                           spatial_term;

                      cell_mass_matrix(i, j) +=
                        (phi_u[i] * phi_u[j] + phi_p[i] * phi_p[j]) *
                        fe_values.JxW(q);
                    }
                }
            }

          cell->get_dof_indices(local_dof_indices);
          constraints_storage.distribute_local_to_global(cell_matrix,
                                                         cell_rhs,
                                                         local_dof_indices,
                                                         system_matrix_storage,
                                                         system_rhs_storage);
          constraints_storage.distribute_local_to_global(
            cell_continuous_matrix,
            local_dof_indices,
            continuous_operator_storage);
          constraints_storage.distribute_local_to_global(cell_mass_matrix,
                                                         local_dof_indices,
                                                         mass_matrix_storage);
        }

    system_matrix_storage.compress(VectorOperation::add);
    continuous_operator_storage.compress(VectorOperation::add);
    mass_matrix_storage.compress(VectorOperation::add);
    system_rhs_storage.compress(VectorOperation::add);

    LA::MPI::PreconditionAMG::AdditionalData velocity_data;
#ifdef IMMERSX_USE_PETSC_LA
    velocity_data.symmetric_operator = true;
#endif
    velocity_preconditioner.initialize(system_matrix_storage.block(0, 0),
                                       velocity_data);

    LA::MPI::PreconditionAMG::AdditionalData pressure_data;
#ifdef IMMERSX_USE_PETSC_LA
    pressure_data.symmetric_operator = true;
#endif
    pressure_preconditioner.initialize(mass_matrix_storage.block(1, 1),
                                       pressure_data);
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "Solve");

    using Vector = LA::MPI::Vector;

    const auto A  = linear_operator<Vector>(system_matrix_storage.block(0, 0));
    const auto Bt = linear_operator<Vector>(system_matrix_storage.block(0, 1));
    const auto B  = linear_operator<Vector>(system_matrix_storage.block(1, 0));
    const auto Mp = linear_operator<Vector>(mass_matrix_storage.block(1, 1));
    const auto Z  = 0.0 * Mp;

    const auto system =
      block_operator<2, 2, BlockVectorType>({{{{A, Bt}}, {{B, Z}}}});

    SolverControl    velocity_control(par.inner_solver_max_steps,
                                   par.inner_solver_tolerance,
                                   false,
                                   par.log_solver_iterations);
    SolverControl    pressure_control(par.inner_solver_max_steps,
                                   par.inner_solver_tolerance,
                                   false,
                                   par.log_solver_iterations);
    SolverCG<Vector> velocity_solver(velocity_control);
    SolverCG<Vector> pressure_solver(pressure_control);

    const auto inv_velocity =
      inverse_operator(A, velocity_solver, velocity_preconditioner);
    const auto inv_pressure =
      inverse_operator(Mp, pressure_solver, pressure_preconditioner);
    const auto block_preconditioner = block_operator<2, 2, BlockVectorType>(
      {{{{inv_velocity, 0.0 * Bt}}, {{0.0 * B, inv_pressure}}}});

    constraints_storage.distribute(solution_storage);
    SolverFGMRES<BlockVectorType> solver(par.solver_control);
    solver.solve(system,
                 solution_storage,
                 system_rhs_storage,
                 block_preconditioner);
    constraints_storage.distribute(solution_storage);
    update_locally_relevant_solution();

    pcout << "   FGMRES iterations: " << par.solver_control.last_step()
          << std::endl;
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::update_locally_relevant_solution()
  {
    locally_relevant_solution_storage = solution_storage;
    locally_relevant_solution_storage.update_ghost_values();
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::advance_one_timestep()
  {
    AssertThrow(solution_storage.size() > 0,
                ExcMessage(
                  "setup_system() must be called before time stepping."));
    AssertThrow(timestep_number_storage < n_time_steps_storage,
                ExcMessage(
                  "The configured final time has already been reached."));

    const double next_time =
      std::min(par.final_time,
               current_time_storage + (par.time_step_policy == "fixed" ?
                                         par.time_step :
                                         time_step_storage));
    time_step_storage    = next_time - current_time_storage;
    current_time_storage = next_time;
    par.set_time(current_time_storage);
    update_constraints();
    assemble_system();
    solve();

    previous_solution_storage = solution_storage;
    update_locally_relevant_solution();
    ++timestep_number_storage;
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::output_results() const
  {
    TimerOutput::Scope t(computing_timer, "Output results");
    ensure_output_directory(par.output_directory);

    std::vector<std::string> names(dim, "velocity");
    names.emplace_back("pressure");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(dh);
    data_out.add_data_vector(locally_relevant_solution_storage,
                             names,
                             DataOut<dim, spacedim>::type_dof_data,
                             interpretation);

    Vector<float> subdomain(tria->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria->locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches(mapping_storage);

    const std::string filename =
      par.output_name + "_" + std::to_string(output_cycle) + ".vtu";
    data_out.write_vtu_in_parallel(par.output_directory + "/" + filename,
                                   mpi_communicator);
    times_and_names.emplace_back(current_time_storage, filename);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::ofstream pvd(par.output_directory + "/" + par.output_name +
                          ".pvd");
        DataOutBase::write_pvd_record(pvd, times_and_names);
      }
    ++output_cycle;
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::accept_state(
    const VectorType  &velocity,
    const VectorType  &pressure,
    const double       time,
    const unsigned int step_number)
  {
    AssertThrow(velocity.size() == dofs_per_block[0],
                ExcDimensionMismatch(velocity.size(), dofs_per_block[0]));
    AssertThrow(pressure.size() == dofs_per_block[1],
                ExcDimensionMismatch(pressure.size(), dofs_per_block[1]));
    AssertThrow(std::isfinite(time),
                ExcMessage("An accepted Navier-Stokes time must be finite."));

    current_time_storage    = time;
    timestep_number_storage = step_number;
    par.set_time(current_time_storage);
    update_constraints();
    solution_storage.block(0) = velocity;
    solution_storage.block(1) = pressure;
    constraints_storage.distribute(solution_storage);
    previous_solution_storage = solution_storage;
    update_locally_relevant_solution();
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::run()
  {
    ensure_output_directory(par.output_directory);
    pcout << "Running NavierStokesSolver<"
          << Utilities::dim_string(dim, spacedim) << ">." << std::endl;
    par.prm.print_parameters(par.output_directory + "/used_parameters_" +
                               std::to_string(dim) + std::to_string(spacedim) +
                               ".prm",
                             ParameterHandler::Short);

    make_grid();
    setup_fe();
    setup_system();

    if (par.output_frequency > 0)
      output_results();

    while (timestep_number_storage < n_time_steps_storage)
      {
        advance_one_timestep();
        if (par.output_frequency > 0 &&
            (timestep_number_storage % par.output_frequency == 0 ||
             timestep_number_storage == n_time_steps_storage))
          output_results();
      }
  }


  template <int dim, int spacedim>
  types::global_dof_index
  NavierStokesSolver<dim, spacedim>::n_dofs() const
  {
    return dh.n_dofs();
  }


  template <int dim, int spacedim>
  unsigned int
  NavierStokesSolver<dim, spacedim>::n_time_steps() const
  {
    return n_time_steps_storage;
  }


  template <int dim, int spacedim>
  unsigned int
  NavierStokesSolver<dim, spacedim>::timestep_number() const
  {
    return timestep_number_storage;
  }


  template <int dim, int spacedim>
  double
  NavierStokesSolver<dim, spacedim>::solution_l2_norm() const
  {
    return solution_storage.size() == 0 ? 0. : solution_storage.l2_norm();
  }


  template <int dim, int spacedim>
  bool
  NavierStokesSolver<dim, spacedim>::solution_is_finite() const
  {
    return solution_storage.size() != 0 && std::isfinite(solution_l2_norm());
  }


  template <int dim, int spacedim>
  double
  NavierStokesSolver<dim, spacedim>::system_residual_l2_norm() const
  {
    if (solution_storage.size() == 0)
      return 0.;

    BlockVectorType residual;
    residual.reinit(solution_storage);
    system_matrix_storage.vmult(residual, solution_storage);
    residual -= system_rhs_storage;
    return residual.l2_norm();
  }


  template <int dim, int spacedim>
  double
  NavierStokesSolver<dim, spacedim>::divergence_l2_norm() const
  {
    if (solution_storage.size() == 0)
      return 0.;

    FEValues<dim, spacedim> fe_values(mapping_storage,
                                      *fe,
                                      *quadrature,
                                      update_gradients | update_JxW_values);
    std::vector<double>     divergences(quadrature->size());
    double                  local_integral = 0.;

    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[velocity].get_function_divergences(
            locally_relevant_solution_storage, divergences);
          for (unsigned int q = 0; q < quadrature->size(); ++q)
            local_integral +=
              divergences[q] * divergences[q] * fe_values.JxW(q);
        }

    return std::sqrt(Utilities::MPI::sum(local_integral, mpi_communicator));
  }


  template <int dim, int spacedim>
  double
  NavierStokesSolver<dim, spacedim>::current_time() const
  {
    return current_time_storage;
  }


  template <int dim, int spacedim>
  double
  NavierStokesSolver<dim, spacedim>::time_step() const
  {
    return time_step_storage;
  }


  template <int dim, int spacedim>
  const parallel::TriangulationBase<dim, spacedim> &
  NavierStokesSolver<dim, spacedim>::triangulation() const
  {
    return *tria;
  }


  template <int dim, int spacedim>
  const DoFHandler<dim, spacedim> &
  NavierStokesSolver<dim, spacedim>::dof_handler() const
  {
    return dh;
  }


  template <int dim, int spacedim>
  const FiniteElement<dim, spacedim> &
  NavierStokesSolver<dim, spacedim>::finite_element() const
  {
    return *fe;
  }


  template <int dim, int spacedim>
  const Mapping<dim, spacedim> &
  NavierStokesSolver<dim, spacedim>::mapping() const
  {
    return mapping_storage;
  }


  template <int dim, int spacedim>
  const AffineConstraints<double> &
  NavierStokesSolver<dim, spacedim>::constraints() const
  {
    return constraints_storage;
  }


  template <int dim, int spacedim>
  const LA::MPI::BlockSparseMatrix &
  NavierStokesSolver<dim, spacedim>::system_matrix() const
  {
    return system_matrix_storage;
  }


  template <int dim, int spacedim>
  const LA::MPI::BlockSparseMatrix &
  NavierStokesSolver<dim, spacedim>::mass_matrix() const
  {
    return mass_matrix_storage;
  }


  template <int dim, int spacedim>
  const LA::MPI::BlockSparseMatrix &
  NavierStokesSolver<dim, spacedim>::continuous_operator() const
  {
    return continuous_operator_storage;
  }


  template <int dim, int spacedim>
  const LA::MPI::SparseMatrix &
  NavierStokesSolver<dim, spacedim>::velocity_mass_matrix() const
  {
    return mass_matrix_storage.block(0, 0);
  }


  template <int dim, int spacedim>
  const LA::MPI::SparseMatrix &
  NavierStokesSolver<dim, spacedim>::pressure_metric_matrix() const
  {
    return mass_matrix_storage.block(1, 1);
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::velocity_forcing_at_time(
    const double     time,
    LA::MPI::Vector &destination) const
  {
    AssertThrow(fe != nullptr && quadrature != nullptr,
                ExcMessage(
                  "setup_fe() must be called before forcing assembly."));

    par.rhs.set_time(time);
    destination.reinit(owned_dofs_by_block[0], mpi_communicator);
    destination = 0.;

    FEValues<dim, spacedim>              fe_values(mapping_storage,
                                      *fe,
                                      *quadrature,
                                      update_values | update_quadrature_points |
                                        update_JxW_values);
    const unsigned int                   dofs_per_cell = fe->n_dofs_per_cell();
    const unsigned int                   n_q_points    = quadrature->size();
    Vector<double>                       cell_rhs(dofs_per_cell);
    std::vector<Vector<double>>          rhs_values(n_q_points,
                                           Vector<double>(spacedim + 1));
    std::vector<Tensor<1, spacedim>>     phi_u(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> velocity_indices;
    std::vector<double>                  velocity_values;

    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_rhs = 0.;
          fe_values.reinit(cell);
          par.rhs.vector_value_list(fe_values.get_quadrature_points(),
                                    rhs_values);
          cell->get_dof_indices(local_dof_indices);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              Tensor<1, spacedim> rhs_u;
              for (unsigned int d = 0; d < dim; ++d)
                rhs_u[d] = rhs_values[q][d];

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                if (fe->system_to_component_index(i).first < dim)
                  {
                    phi_u[i] = fe_values[velocity].value(i, q);
                    cell_rhs(i) += (rhs_u * phi_u[i]) * fe_values.JxW(q);
                  }
            }

          velocity_indices.clear();
          velocity_values.clear();
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            if (fe->system_to_component_index(i).first < dim)
              {
                velocity_indices.push_back(local_dof_indices[i]);
                velocity_values.push_back(cell_rhs(i));
              }
          destination.add(velocity_indices, velocity_values);
        }

    destination.compress(VectorOperation::add);
    for (const auto index : destination.locally_owned_elements())
      if (constraints_storage.is_constrained(index))
        destination(index) = 0.;
  }


  template <int dim, int spacedim>
  const typename NavierStokesSolver<dim, spacedim>::BlockVectorType &
  NavierStokesSolver<dim, spacedim>::system_rhs() const
  {
    return system_rhs_storage;
  }


  template <int dim, int spacedim>
  const typename NavierStokesSolver<dim, spacedim>::BlockVectorType &
  NavierStokesSolver<dim, spacedim>::solution() const
  {
    return solution_storage;
  }


  template <int dim, int spacedim>
  const typename NavierStokesSolver<dim, spacedim>::BlockVectorType &
  NavierStokesSolver<dim, spacedim>::previous_solution() const
  {
    return previous_solution_storage;
  }


  template <int dim, int spacedim>
  const typename NavierStokesSolver<dim, spacedim>::BlockVectorType &
  NavierStokesSolver<dim, spacedim>::locally_relevant_solution() const
  {
    return locally_relevant_solution_storage;
  }


  template <int dim, int spacedim>
  const IndexSet &
  NavierStokesSolver<dim, spacedim>::locally_owned_dofs() const
  {
    return owned_dofs;
  }


  template <int dim, int spacedim>
  const IndexSet &
  NavierStokesSolver<dim, spacedim>::locally_relevant_dofs() const
  {
    return relevant_dofs;
  }


  template <int dim, int spacedim>
  const std::vector<IndexSet> &
  NavierStokesSolver<dim, spacedim>::locally_owned_dofs_by_block() const
  {
    return owned_dofs_by_block;
  }


  template <int dim, int spacedim>
  const std::vector<IndexSet> &
  NavierStokesSolver<dim, spacedim>::locally_relevant_dofs_by_block() const
  {
    return relevant_dofs_by_block;
  }


  template <int dim, int spacedim>
  const FEValuesExtractors::Vector &
  NavierStokesSolver<dim, spacedim>::velocity_extractor() const
  {
    return velocity;
  }


  template <int dim, int spacedim>
  const FEValuesExtractors::Scalar &
  NavierStokesSolver<dim, spacedim>::pressure_extractor() const
  {
    return pressure;
  }


  template <int dim, int spacedim>
  const ComponentMask &
  NavierStokesSolver<dim, spacedim>::velocity_component_mask() const
  {
    return velocity_mask;
  }


  template <int dim, int spacedim>
  double
  NavierStokesSolver<dim, spacedim>::density() const
  {
    return par.density;
  }


  template <int dim, int spacedim>
  types::global_dof_index
  NavierStokesSolver<dim, spacedim>::velocity_block_size() const
  {
    AssertThrow(dofs_per_block.size() == 2,
                ExcMessage("Navier-Stokes DoF blocks are not initialized."));
    return dofs_per_block[0];
  }


  template <int dim, int spacedim>
  void
  NavierStokesSolver<dim, spacedim>::set_solution(
    const BlockVectorType &new_solution)
  {
    AssertThrow(solution_storage.size() == new_solution.size(),
                ExcDimensionMismatch(solution_storage.size(),
                                     new_solution.size()));
    solution_storage = new_solution;
    constraints_storage.distribute(solution_storage);
    previous_solution_storage = solution_storage;
    update_locally_relevant_solution();
  }


  template class NavierStokesParameters<2>;
  template class NavierStokesParameters<3>;

  template class NavierStokesSolver<2>;
  template class NavierStokesSolver<3>;
} // namespace ImmersX
