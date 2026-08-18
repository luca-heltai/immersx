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
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the LICENSE.md file at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include "poisson.h"

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/vector.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "utils.h"


using namespace dealii;


namespace
{
  template <int dim, int spacedim>
  void
  read_poisson_grid(const std::string            &grid_file_name,
                    const std::string            &ids_and_cad_file_names,
                    Triangulation<dim, spacedim> &tria)
  {
    if constexpr (dim == 1)
      {
        // The distributed 1D path is intentionally independent of the CAD
        // manifold helper, whose deal.II link-time instantiations are only
        // available for the volume/surface cases used by the main solvers.
        GridIn<dim, spacedim> grid_in;
        grid_in.attach_triangulation(tria);
        grid_in.read(grid_file_name);
        (void)ids_and_cad_file_names;
      }
    else
      read_grid_and_cad_files(grid_file_name, ids_and_cad_file_names, tria);
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
PoissonParameters<dim, spacedim>::PoissonParameters()
  : ParameterAcceptor("/Poisson/")
  , rhs("/Poisson/Right hand side")
  , bc("/Poisson/Dirichlet boundary conditions")
  , solver_control("/Poisson/Solver/Control")
{
  add_parameter("FE degree", fe_degree, "", this->prm, Patterns::Integer(1));
  add_parameter("Output directory", output_directory);
  add_parameter("Output name", output_name);
  add_parameter("Output results also before solving",
                output_results_before_solving);
  add_parameter("Estimate condition number", estimate_condition_number);
  add_parameter("Initial refinement", initial_refinement);
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

  this->prm.enter_subsection("Error");
  convergence_table.add_parameters(this->prm);
  this->prm.leave_subsection();

  parse_parameters_call_back.connect(
    [this]() { ensure_output_directory(output_directory); });
}


template <int dim, int spacedim>
PoissonSolver<dim, spacedim>::PoissonSolver(
  const PoissonParameters<dim, spacedim> &par)
  : par(par)
  , mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , triangulation_storage(make_triangulation_storage(mpi_communicator))
  , tria(&std::visit(
      [](auto &selected_tria) -> parallel::TriangulationBase<dim, spacedim> & {
        return selected_tria;
      },
      triangulation_storage))
  , dh()
{}


template <int dim, int spacedim>
typename PoissonSolver<dim, spacedim>::TriangulationVariant
PoissonSolver<dim, spacedim>::make_triangulation_storage(
  MPI_Comm mpi_communicator)
{
  if constexpr (dim == 1)
    return TriangulationVariant(
      std::in_place_type<FullyDistributedTriangulation>, mpi_communicator);
  else
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
PoissonSolver<dim, spacedim>::uses_fully_distributed_triangulation() const
{
  return std::holds_alternative<FullyDistributedTriangulation>(
    triangulation_storage);
}


template <int dim, int spacedim>
void
PoissonSolver<dim, spacedim>::make_grid()
{
  TimerOutput::Scope t(computing_timer, "Make grid");

  const bool need_fully_distributed =
    (dim == 1 || par.triangulation_type == "fullydistributed");

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
          read_poisson_grid(par.name_of_grid,
                            par.arguments_for_grid,
                            distributed_tria);
        }

      distributed_tria.refine_global(par.initial_refinement);
      pcout << "   Triangulation backend: distributed" << std::endl
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
      read_poisson_grid(par.name_of_grid, par.arguments_for_grid, serial_tria);
    }

  serial_tria.refine_global(par.initial_refinement);
  auto &fully_distributed_tria =
    std::get<FullyDistributedTriangulation>(triangulation_storage);

  if constexpr (dim == 1)
    {
      // The default fully distributed partitioner keeps children of one
      // coarse cell together. For a once-refined interval this can leave a
      // rank without cells and invalid local DoF indices at the boundary.
      fully_distributed_tria.set_partitioner(
        [](Triangulation<dim, spacedim> &serial_tria,
           const unsigned int            n_partitions) {
          GridTools::partition_triangulation_zorder(n_partitions,
                                                    serial_tria,
                                                    false);
        },
        TriangulationDescription::Settings::default_setting);
    }

  for (const auto manifold_id : serial_tria.get_manifold_ids())
    if (manifold_id != numbers::flat_manifold_id)
      fully_distributed_tria.set_manifold(
        manifold_id, serial_tria.get_manifold(manifold_id));
  fully_distributed_tria.copy_triangulation(serial_tria);

  pcout << "   Triangulation backend: fullydistributed"
        << (dim == 1 ? " (forced for dim=1)" : "") << std::endl;
  pcout << "   Number of active cells: " << tria->n_active_cells() << std::endl;
}


template <int dim, int spacedim>
void
PoissonSolver<dim, spacedim>::setup_fe()
{
  TimerOutput::Scope t(computing_timer, "Initial setup");

  fe = std::make_unique<FESystem<dim, spacedim>>(
    FE_Q<dim, spacedim>(par.fe_degree), 1);
  quadrature = std::make_unique<QGauss<dim>>(par.fe_degree + 1);
}


template <int dim, int spacedim>
void
PoissonSolver<dim, spacedim>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "Setup system");
  AssertThrow(fe != nullptr, ExcMessage("setup_fe() must be called first."));

  dh.distribute_dofs(*fe);
  owned_dofs    = dh.locally_owned_dofs();
  relevant_dofs = DoFTools::extract_locally_relevant_dofs(dh);

  constraints.reinit(owned_dofs, relevant_dofs);
  DoFTools::make_hanging_node_constraints(dh, constraints);
  for (const auto boundary_id : par.dirichlet_ids)
    VectorTools::interpolate_boundary_values(dh,
                                             boundary_id,
                                             par.bc,
                                             constraints);
  constraints.close();

  DynamicSparsityPattern dsp(relevant_dofs);
  DoFTools::make_sparsity_pattern(dh, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             owned_dofs,
                                             mpi_communicator,
                                             relevant_dofs);

  stiffness_matrix.clear();
  stiffness_matrix.reinit(owned_dofs, owned_dofs, dsp, mpi_communicator);

  solution.reinit(owned_dofs, mpi_communicator);
  system_rhs.reinit(owned_dofs, mpi_communicator);
  locally_relevant_solution.reinit(owned_dofs, relevant_dofs, mpi_communicator);

  solution   = 0.;
  system_rhs = 0.;
  update_locally_relevant_solution();

  pcout << "   Number of degrees of freedom: " << dh.n_dofs()
        << " (locally owned: " << owned_dofs.n_elements() << ")" << std::endl;
}


template <int dim, int spacedim>
void
PoissonSolver<dim, spacedim>::assemble_system()
{
  TimerOutput::Scope t(computing_timer, "Assemble system");
  AssertThrow(fe != nullptr && quadrature != nullptr,
              ExcMessage("setup_fe() must be called before assembly."));

  stiffness_matrix = 0.;
  system_rhs       = 0.;

  FEValues<dim, spacedim> fe_values(*fe,
                                    *quadrature,
                                    update_values | update_gradients |
                                      update_quadrature_points |
                                      update_JxW_values);
  const unsigned int      dofs_per_cell = fe->n_dofs_per_cell();
  const unsigned int      n_q_points    = quadrature->size();

  FullMatrix<double>               cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>                   cell_rhs(dofs_per_cell);
  std::vector<double>              rhs_values(n_q_points);
  std::vector<Tensor<1, spacedim>> grad_phi(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const FEValuesExtractors::Scalar     scalar(0);

  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0.;
        cell_rhs    = 0.;
        fe_values.reinit(cell);
        par.rhs.value_list(fe_values.get_quadrature_points(), rhs_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              grad_phi[k] = fe_values[scalar].gradient(k, q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) +=
                    grad_phi[i] * grad_phi[j] * fe_values.JxW(q);

                cell_rhs(i) += fe_values[scalar].value(i, q) * rhs_values[q] *
                               fe_values.JxW(q);
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               stiffness_matrix,
                                               system_rhs);
      }

  stiffness_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}


template <int dim, int spacedim>
void
PoissonSolver<dim, spacedim>::solve()
{
  TimerOutput::Scope t(computing_timer, "Solve");
  pcout << "Solving Poisson system." << std::endl;

  LA::MPI::PreconditionAMG preconditioner;
  {
    LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef IMMERSX_POISSON_USE_PETSC_LA
    data.symmetric_operator = true;
#endif
    preconditioner.initialize(stiffness_matrix, data);
  }

  SolverCG<LA::MPI::Vector> solver(par.solver_control);
  if (par.estimate_condition_number)
    solver.connect_condition_number_slot([this](const double condition_number) {
      pcout << "   Condition number estimate: " << condition_number
            << std::endl;
    });

  constraints.distribute(solution);
  solver.solve(stiffness_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);
  update_locally_relevant_solution();

  pcout << "   CG iterations: " << par.solver_control.last_step() << std::endl;
}


template <int dim, int spacedim>
void
PoissonSolver<dim, spacedim>::update_locally_relevant_solution()
{
  locally_relevant_solution = solution;
  locally_relevant_solution.update_ghost_values();
}


template <int dim, int spacedim>
void
PoissonSolver<dim, spacedim>::output_results() const
{
  TimerOutput::Scope t(computing_timer, "Output results");
  ensure_output_directory(par.output_directory);

  // Keep one record per cycle, matching the existing application output
  // convention when output before solving is enabled.
  if (cycles_and_solutions.size() == cycle)
    {
      DataOut<dim, spacedim> data_out;
      data_out.attach_dof_handler(dh);
      data_out.add_data_vector(locally_relevant_solution, "solution");

      Vector<float> subdomain(tria->n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = tria->locally_owned_subdomain();
      data_out.add_data_vector(subdomain, "subdomain");
      data_out.build_patches();

      const std::string filename =
        par.output_name + "_" + std::to_string(cycle) + ".vtu";
      data_out.write_vtu_in_parallel(par.output_directory + "/" + filename,
                                     mpi_communicator);
      cycles_and_solutions.push_back({static_cast<double>(cycle), filename});

      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          std::ofstream pvd_solutions(par.output_directory + "/" +
                                      par.output_name + ".pvd");
          DataOutBase::write_pvd_record(pvd_solutions, cycles_and_solutions);
        }
    }
}


template <int dim, int spacedim>
void
PoissonSolver<dim, spacedim>::refine_grid()
{
  TimerOutput::Scope t(computing_timer, "Refine and transfer");

  if constexpr (dim == 1)
    AssertThrow(
      false,
      ExcMessage(
        "Adaptive refinement is not implemented for the fully distributed "
        "1D Poisson triangulation. Use one initial mesh."));
  else
    {
      AssertThrow(
        !uses_fully_distributed_triangulation(),
        ExcMessage(
          "Adaptive refinement is not implemented with "
          "parallel::fullydistributed::Triangulation in PoissonSolver. "
          "Use one initial mesh with that backend."));

      Vector<float> error_per_cell(tria->n_active_cells());
      KellyErrorEstimator<dim, spacedim>::estimate(dh,
                                                   QGauss<dim - 1>(
                                                     par.fe_degree + 1),
                                                   {},
                                                   locally_relevant_solution,
                                                   error_per_cell);

      if (par.refinement_strategy == "fixed_fraction")
        parallel::distributed::GridRefinement::
          refine_and_coarsen_fixed_fraction(*tria,
                                            error_per_cell,
                                            par.refinement_fraction,
                                            par.coarsening_fraction);
      else if (par.refinement_strategy == "fixed_number")
        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
          *tria,
          error_per_cell,
          par.refinement_fraction,
          par.coarsening_fraction,
          par.max_cells);
      else if (par.refinement_strategy == "global")
        for (const auto &cell : tria->active_cell_iterators())
          cell->set_refine_flag();
      else
        AssertThrow(false,
                    ExcMessage("Unknown Poisson refinement strategy '" +
                               par.refinement_strategy + "'."));

      SolutionTransfer<dim, VectorType, spacedim> transfer(dh);
      tria->prepare_coarsening_and_refinement();
      transfer.prepare_for_coarsening_and_refinement(locally_relevant_solution);
      tria->execute_coarsening_and_refinement();

      setup_system();
      transfer.interpolate(solution);
      constraints.distribute(solution);
      update_locally_relevant_solution();
    }
}


template <int dim, int spacedim>
void
PoissonSolver<dim, spacedim>::run()
{
  ensure_output_directory(par.output_directory);

  pcout << "Running PoissonSolver<" << Utilities::dim_string(dim, spacedim)
        << ">." << std::endl;
  par.prm.print_parameters(par.output_directory + "/used_parameters_" +
                             std::to_string(dim) + std::to_string(spacedim) +
                             ".prm",
                           ParameterHandler::Short);

  make_grid();
  setup_fe();

  for (cycle = 0; cycle < par.n_refinement_cycles; ++cycle)
    {
      setup_system();
      if (par.output_results_before_solving)
        output_results();

      assemble_system();
      solve();
      output_results();

      par.convergence_table.error_from_exact(dh,
                                             locally_relevant_solution,
                                             par.bc);

      if (cycle + 1 < par.n_refinement_cycles)
        refine_grid();

      if (pcout.is_active())
        par.convergence_table.output_table(pcout.get_stream());
    }
}


template <int dim, int spacedim>
types::global_dof_index
PoissonSolver<dim, spacedim>::n_dofs() const
{
  return dh.n_dofs();
}


template <int dim, int spacedim>
double
PoissonSolver<dim, spacedim>::solution_l2_norm() const
{
  return solution.size() == 0 ? 0. : solution.l2_norm();
}


template <int dim, int spacedim>
bool
PoissonSolver<dim, spacedim>::solution_is_finite() const
{
  return solution.size() != 0 && std::isfinite(solution_l2_norm());
}


template class PoissonParameters<1>;
template class PoissonParameters<1, 2>;
template class PoissonParameters<1, 3>;
template class PoissonParameters<2>;
template class PoissonParameters<2, 3>;
template class PoissonParameters<3>;

template class PoissonSolver<1>;
template class PoissonSolver<1, 2>;
template class PoissonSolver<1, 3>;
template class PoissonSolver<2>;
template class PoissonSolver<2, 3>;
template class PoissonSolver<3>;
