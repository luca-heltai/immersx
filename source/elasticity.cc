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
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------


#include "elasticity.h"

#include <iomanip>
#include <limits>
#include <sstream>

#include "augmented_lagrangian.h"
#include "augmented_lagrangian_preconditioner.h"
#include "utils.h"

template <int dim>
RigidBodyMotion<dim>::RigidBodyMotion(const unsigned int _type)
  : Function<dim>(dim)
  , type(_type)
{
  Assert(dim == 2 || dim == 3, ExcNotImplemented());
  Assert((dim == 2 && type <= 2) || (dim == 3 && type <= 5),
         ExcNotImplemented());
}



template <int dim>
double
RigidBodyMotion<dim>::value(const Point<dim>  &p,
                            const unsigned int component) const
{
  if constexpr (dim == 2)
    {
      // 2D rigid body modes: 2 translations and 1 rotation
      const std::array<double, 3> modes{{static_cast<double>(component == 0),
                                         static_cast<double>(component == 1),
                                         (component == 0) ? -p[1] :
                                         (component == 1) ? p[0] :
                                                            0.}};

      return modes[type];
    }
  else // dim == 3
    {
      // 3D rigid body modes: 3 translations and 3 rotations
      const std::array<double, 6> modes{{static_cast<double>(component == 0),
                                         static_cast<double>(component == 1),
                                         static_cast<double>(component == 2),
                                         (component == 0) ? 0. :
                                         (component == 1) ? p[2] :
                                                            -p[1],
                                         (component == 0) ? -p[2] :
                                         (component == 1) ? 0. :
                                                            p[0],
                                         (component == 0) ? p[1] :
                                         (component == 1) ? -p[0] :
                                                            0.}};

      return modes[type];
    }
}



template <int dim, int spacedim>
ElasticityProblem<dim, spacedim>::ElasticityProblem(
  const ElasticityProblemParameters<dim, spacedim> &par)
  : par(par)
  , mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , tria(mpi_communicator,
         typename Triangulation<spacedim>::MeshSmoothing(
           Triangulation<spacedim>::smoothing_on_refinement |
           Triangulation<spacedim>::smoothing_on_coarsening))
  , inclusions(spacedim)
  , dh(tria)
  , displacement(0)
{}



template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::make_grid()
{
  if (par.domain_type == "generate")
    {
      try
        {
          GridGenerator::generate_from_name_and_arguments(
            tria, par.name_of_grid, par.arguments_for_grid);
        }
      catch (...)
        {
          pcout << "Generating from name and argument failed." << std::endl
                << "Trying to read from file name." << std::endl;
          read_grid_and_cad_files(par.name_of_grid,
                                  par.arguments_for_grid,
                                  tria);
        }
    }
  else if (par.domain_type == "cylinder")
    {
      Assert(spacedim == 2, ExcInternalError());
      GridGenerator::hyper_ball(tria, Point<spacedim>(), 1.);
      std::cout << " ATTENTION: GRID: cirle of radius 1." << std::endl;
    }
  else if (par.domain_type == "cheese")
    {
      Assert(spacedim == 2, ExcInternalError());
      GridGenerator::cheese(tria, std::vector<unsigned int>(2, 2));
    }
  else if (par.domain_type == "file")
    {
      GridIn<spacedim> gi;
      gi.attach_triangulation(tria);
#ifdef DEAL_II_WITH_GMSH_API
      std::string infile(par.name_of_grid);
#else
      std::ifstream infile(par.name_of_grid);
      Assert(infile.good(), ExcIO());
#endif
      try
        {
          gi.read_msh(infile);
          // gi.read_vtk(infile);
        }
      catch (...)
        {
          Assert(false, ExcInternalError());
        }
    }

  tria.refine_global(par.initial_refinement);
}



template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::setup_fe()
{
  TimerOutput::Scope t(computing_timer, "Initial setup");
  fe = std::make_unique<FESystem<spacedim>>(FE_Q<spacedim>(par.fe_degree),
                                            spacedim);
  quadrature = std::make_unique<QGauss<spacedim>>(par.fe_degree + 1);
  face_quadrature_formula =
    std::make_unique<QGauss<spacedim - 1>>(par.fe_degree + 1);
}



template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::setup_dofs()
{
  TimerOutput::Scope t(computing_timer, "Setup dofs");
  dh.distribute_dofs(*fe);

  owned_dofs.resize(2);
  owned_dofs[0] = dh.locally_owned_dofs();
  relevant_dofs.resize(2);
  relevant_dofs[0] = DoFTools::extract_locally_relevant_dofs(dh);

  FEFaceValues<spacedim> fe_face_values(*fe,
                                        *face_quadrature_formula,
                                        update_values | update_JxW_values |
                                          update_quadrature_points |
                                          update_normal_vectors);

  {
    constraints.reinit(owned_dofs[0], relevant_dofs[0]);
    DoFTools::make_hanging_node_constraints(dh, constraints);

    for (const auto id : par.dirichlet_ids)
      VectorTools::interpolate_boundary_values(dh, id, par.bc, constraints);

    std::map<types::boundary_id, const Function<spacedim, double> *>
      function_map;
    for (const auto id : par.normal_flux_ids)
      {
        function_map.insert(
          std::pair<types::boundary_id, const Function<spacedim, double> *>(
            id, &par.Neumann_bc));
      }
    VectorTools::compute_nonzero_normal_flux_constraints(
      dh, 0, par.normal_flux_ids, function_map, constraints);
    constraints.close();


    /*{
      mean_value_constraints.clear();
      mean_value_constraints.reinit(relevant_dofs[0]);

      for (const auto id : par.normal_flux_ids)
      {
        const std::set<types::boundary_id > &boundary_ids={id};
        const ComponentMask &component_mask=ComponentMask();
        const IndexSet boundary_dofs = DoFTools::extract_boundary_dofs(dh,
    component_mask, boundary_ids);

        const types::global_dof_index first_boundary_dof =
          boundary_dofs.nth_index_in_set(0);

        mean_value_constraints.add_line(first_boundary_dof);
        for (types::global_dof_index i : boundary_dofs)
          if (i != first_boundary_dof)
            mean_value_constraints.add_entry(first_boundary_dof, i, -1);
      }
        mean_value_constraints.close();

        constraints.merge(mean_value_constraints);
    }*/
  }
  {
    DynamicSparsityPattern dsp(relevant_dofs[0]);
    DoFTools::make_sparsity_pattern(dh, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               owned_dofs[0],
                                               mpi_communicator,
                                               relevant_dofs[0]);
    stiffness_matrix.clear();

    stiffness_matrix.reinit(owned_dofs[0],
                            owned_dofs[0],
                            dsp,
                            mpi_communicator);
    if (par.time_mode != TimeMode::Static)
      {
        mass_matrix.clear();
        newmark_matrix.clear();
        damping_matrix.clear();
        newmark_matrix.reinit(owned_dofs[0],
                              owned_dofs[0],
                              dsp,
                              mpi_communicator);
        mass_matrix.reinit(owned_dofs[0], owned_dofs[0], dsp, mpi_communicator);
        damping_matrix.reinit(owned_dofs[0],
                              owned_dofs[0],
                              dsp,
                              mpi_communicator);
      }
  }

  inclusion_constraints.close();

  if (inclusions.n_dofs() > 0)
    {
      auto inclusions_set =
        Utilities::MPI::create_evenly_distributed_partitioning(
          mpi_communicator, inclusions.n_inclusions());

      owned_dofs[1] = inclusions_set.tensor_product(
        complete_index_set(inclusions.n_dofs_per_inclusion()));

      DynamicSparsityPattern dsp(dh.n_dofs(),
                                 inclusions.n_dofs(),
                                 relevant_dofs[0]);

      relevant_dofs[1] = assemble_coupling_sparsity(dsp);
      relevant_dofs[1].add_indices(owned_dofs[1]);
      SparsityTools::distribute_sparsity_pattern(dsp,
                                                 owned_dofs[0],
                                                 mpi_communicator,
                                                 relevant_dofs[0]);
      coupling_matrix.clear();
      coupling_matrix.reinit(owned_dofs[0],
                             owned_dofs[1],
                             dsp,
                             mpi_communicator);

      DynamicSparsityPattern idsp(relevant_dofs[1]);
      for (const auto i : relevant_dofs[1])
        idsp.add(i, i);

      SparsityTools::distribute_sparsity_pattern(idsp,
                                                 owned_dofs[1],
                                                 mpi_communicator,
                                                 relevant_dofs[1]);
      inclusion_matrix.clear();
      inclusion_matrix.reinit(owned_dofs[1],
                              owned_dofs[1],
                              idsp,
                              mpi_communicator);
    }

  locally_relevant_solution.reinit(owned_dofs, relevant_dofs, mpi_communicator);
  system_rhs.reinit(owned_dofs, mpi_communicator);
  solution.reinit(owned_dofs, mpi_communicator);
  force_rhs.reinit(owned_dofs, mpi_communicator);
  bc_rhs.reinit(owned_dofs, mpi_communicator);
  neumann_bc_rhs.reinit(owned_dofs, mpi_communicator);

  if (par.time_mode != TimeMode::Static)
    {
      velocity.reinit(owned_dofs, mpi_communicator);
      acceleration.reinit(owned_dofs, mpi_communicator);
      predictor.reinit(owned_dofs, mpi_communicator);
      corrector.reinit(owned_dofs, mpi_communicator);
    }

  pcout << "   Number of degrees of freedom: " << owned_dofs[0].size() << " + "
        << owned_dofs[1].size()
        << " (locally owned: " << owned_dofs[0].n_elements() << " + "
        << owned_dofs[1].n_elements() << ")" << std::endl;
}



template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::assemble_elasticity_system()
{
  TimerOutput::Scope t(computing_timer, "Assemble matrices and rhs");
  stiffness_matrix = 0;
  coupling_matrix  = 0;
  system_rhs       = 0;
  if (par.time_mode != TimeMode::Static)
    {
      newmark_matrix = 0;
      damping_matrix = 0;
      mass_matrix    = 0;
    }

  par.rhs.set_time(current_time);
  par.bc.set_time(current_time);
  par.Neumann_bc.set_time(current_time);

  FEValues<spacedim>     fe_values(*fe,
                               *quadrature,
                               update_values | update_gradients |
                                 update_quadrature_points | update_JxW_values);
  FEFaceValues<spacedim> fe_face_values(*fe,
                                        *face_quadrature_formula,
                                        update_values |
                                          update_quadrature_points |
                                          update_JxW_values | update_gradients |
                                          update_normal_vectors);

  const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature->size();

  // Constant-less matrices
  FullMatrix<double> cell_value(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_grad(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_div(dofs_per_cell, dofs_per_cell);

  // Penalty matrices for weak Dirichlet conditions
  FullMatrix<double> cell_penalty_grad(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_penalty_div(dofs_per_cell, dofs_per_cell);

  // Parameter dependent matrices
  FullMatrix<double> cell_penalty_value(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_damping(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_newmark(dofs_per_cell, dofs_per_cell);

  Vector<double>              cell_rhs(dofs_per_cell);
  Vector<double>              cell_penalty_value_rhs(dofs_per_cell);
  Vector<double>              cell_penalty_grad_rhs(dofs_per_cell);
  Vector<double>              cell_penalty_div_rhs(dofs_per_cell);
  std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(spacedim));

  std::vector<Tensor<2, spacedim>> grad_phi_u(dofs_per_cell);
  std::vector<double>              div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, spacedim>> phi_u(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        // Get material properties
        const auto &mp = par.get_material_properties(cell->material_id());

        cell_grad          = 0;
        cell_value         = 0;
        cell_div           = 0;
        cell_penalty_value = 0;
        cell_penalty_grad  = 0;
        cell_penalty_div   = 0;
        cell_mass          = 0;
        cell_damping       = 0;
        cell_newmark       = 0;
        cell_stiffness     = 0;

        cell_rhs               = 0;
        cell_penalty_grad_rhs  = 0;
        cell_penalty_div_rhs   = 0;
        cell_penalty_value_rhs = 0;

        fe_values.reinit(cell);
        par.rhs.vector_value_list(fe_values.get_quadrature_points(),
                                  rhs_values);

        // Assemble bulk contributions, no constants yet
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                grad_phi_u[k] =
                  fe_values[displacement].symmetric_gradient(k, q);
                div_phi_u[k] = fe_values[displacement].divergence(k, q);
                phi_u[k]     = fe_values[displacement].value(k, q);
              }
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_grad(i, j) +=
                      scalar_product(grad_phi_u[i], grad_phi_u[j]) *
                      fe_values.JxW(q);
                    cell_div(i, j) +=
                      div_phi_u[i] * div_phi_u[j] * fe_values.JxW(q);
                    cell_value(i, j) += phi_u[i] * phi_u[j] * fe_values.JxW(q);
                  }
                const auto comp_i = fe->system_to_component_index(i).first;
                cell_rhs(i) += fe_values.shape_value(i, q) *
                               rhs_values[q](comp_i) * fe_values.JxW(q);
              }
          }

        // Boundary conditions
        for (const auto &f : cell->face_indices())
          if (cell->face(f)->at_boundary())
            {
              // Weak Dirichlet conditions
              if (par.weak_dirichlet_ids.find(cell->face(f)->boundary_id()) !=
                  par.weak_dirichlet_ids.end())
                {
                  fe_face_values.reinit(cell, f);
                  const auto cell_diameter = cell->diameter();

                  for (unsigned int q = 0;
                       q < fe_face_values.n_quadrature_points;
                       ++q)
                    {
                      const auto n = fe_face_values.normal_vector(q);
                      for (unsigned int k = 0; k < dofs_per_cell; ++k)
                        {
                          phi_u[k] = fe_face_values[displacement].value(k, q);
                          grad_phi_u[k] =
                            fe_face_values[displacement].symmetric_gradient(k,
                                                                            q);
                          div_phi_u[k] =
                            fe_face_values[displacement].divergence(k, q);
                        }
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              cell_penalty_value(i, j) +=
                                par.penalty_term * (1.0 / cell_diameter) *
                                phi_u[i] * phi_u[j] * fe_face_values.JxW(q);
                              cell_penalty_grad(i, j) +=
                                (-grad_phi_u[j] * n * phi_u[i] -
                                 grad_phi_u[i] * n * phi_u[j]) *
                                fe_face_values.JxW(q);
                              cell_penalty_div(i, j) +=
                                (-div_phi_u[j] * (n * phi_u[i]) -
                                 div_phi_u[i] * (n * phi_u[j])) *
                                fe_face_values.JxW(q);
                            }
                          const auto comp_i =
                            fe->system_to_component_index(i).first;
                          Tensor<1, spacedim> g;
                          g[comp_i] =
                            par.bc.value(fe_face_values.quadrature_point(q),
                                         comp_i);

                          cell_penalty_value_rhs(i) +=
                            par.penalty_term * (1.0 / cell_diameter) * g *
                            phi_u[i] * fe_face_values.JxW(q);

                          cell_penalty_grad_rhs(i) +=
                            -grad_phi_u[i] * n * g * fe_face_values.JxW(q);

                          cell_penalty_div_rhs(i) +=
                            -div_phi_u[i] * (n * g) * fe_face_values.JxW(q);
                        }
                    }
                }
              // Neumann Boundary conditions
              else if (par.neumann_ids.find(cell->face(f)->boundary_id()) !=
                       par.neumann_ids.end())
                {
                  fe_face_values.reinit(cell, f);
                  for (unsigned int q = 0;
                       q < fe_face_values.n_quadrature_points;
                       ++q)
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          const auto comp_i =
                            fe->system_to_component_index(i).first;
                          const auto un = par.Neumann_bc.value(
                            fe_face_values.quadrature_point(q), comp_i);
                          cell_rhs(i) += un * fe_face_values.shape_value(i, q) *
                                         fe_face_values.JxW(q);
                        }
                    }
                }
            }

        cell->get_dof_indices(local_dof_indices);

        cell_stiffness.equ(2 * mp.Lame_mu,
                           cell_grad,
                           mp.Lame_lambda,
                           cell_div,
                           1.0,
                           cell_penalty_value);

        if (par.time_mode == TimeMode::Dynamic)
          {
            cell_mass.equ(mp.rho, cell_value);

            cell_damping.equ(mp.rayleigh_beta,
                             cell_stiffness,
                             mp.rayleigh_alpha,
                             cell_mass,
                             mp.neta,
                             cell_grad);

            constraints.distribute_local_to_global(cell_mass,
                                                   // cell_rhs,
                                                   local_dof_indices,
                                                   mass_matrix);

            constraints.distribute_local_to_global(cell_damping,
                                                   // cell_rhs,
                                                   local_dof_indices,
                                                   damping_matrix);
          }
        // We don't want penalty terms in the mass and damping matrices, but we
        // do want them in the stiffness and newmark matrices
        cell_stiffness.add(2 * mp.Lame_mu,
                           cell_penalty_grad,
                           mp.Lame_lambda,
                           cell_penalty_div);

        cell_rhs.add(2 * mp.Lame_mu,
                     cell_penalty_grad_rhs,
                     mp.Lame_lambda,
                     cell_penalty_div_rhs);

        cell_rhs += cell_penalty_value_rhs;

        constraints.distribute_local_to_global(cell_stiffness,
                                               cell_rhs,
                                               local_dof_indices,
                                               stiffness_matrix,
                                               system_rhs.block(0));

        if (par.time_mode == TimeMode::Dynamic)
          {
            cell_newmark.equ(mp.rho,
                             cell_value,
                             par.beta * par.dt * par.dt,
                             cell_stiffness,
                             par.gamma * par.dt,
                             cell_damping);

            constraints.distribute_local_to_global(cell_newmark,
                                                   // cell_rhs,
                                                   local_dof_indices,
                                                   newmark_matrix);
          }
      }
  stiffness_matrix.compress(VectorOperation::add);
  mass_matrix.compress(VectorOperation::add);
  damping_matrix.compress(VectorOperation::add);
  newmark_matrix.compress(VectorOperation::add);

  system_rhs.compress(VectorOperation::add);

  {
    Teuchos::ParameterList amg_parameter_list;
    amg_parameter_list.set("smoother: type", "Chebyshev");
    amg_parameter_list.set("smoother: sweeps", 2);
    amg_parameter_list.set("smoother: pre or post", "both");
    amg_parameter_list.set("coarse: type", "Amesos-KLU");
    amg_parameter_list.set("coarse: max size", 2000);
    amg_parameter_list.set("aggregation: threshold", 0.02);

#if DEAL_II_VERSION_GTE(9, 7, 0)
    using VectorType = std::vector<double>;
    MappingQ1<spacedim>              mapping;
    std::vector<std::vector<double>> rigid_body_modes =
      DoFTools::extract_rigid_body_modes(mapping, dh);
#else
    using VectorType = LinearAlgebra::distributed::Vector<double>;
    std::vector<LinearAlgebra::distributed::Vector<double>> rigid_body_modes(
      spacedim == 3 ? 6 : 3);
    for (unsigned int i = 0; i < rigid_body_modes.size(); ++i)
      {
        rigid_body_modes[i].reinit(dh.locally_owned_dofs(), mpi_communicator);
        RigidBodyMotion<spacedim> rbm(i);
        VectorTools::interpolate(dh, rbm, rigid_body_modes[i]);
      }
#endif

    {
      auto                                parameter_list_A = amg_parameter_list;
      std::unique_ptr<Epetra_MultiVector> ptr_operator_modes;
      UtilitiesAL::set_null_space<spacedim, VectorType>(
        parameter_list_A,
        ptr_operator_modes,
        stiffness_matrix.trilinos_matrix(),
        rigid_body_modes);
      prec_A.initialize(stiffness_matrix, parameter_list_A);
    }

    if (par.time_mode == TimeMode::Dynamic)
      {
        auto parameter_list_newmark = amg_parameter_list;
        std::unique_ptr<Epetra_MultiVector> ptr_operator_modes;
        UtilitiesAL::set_null_space<spacedim, VectorType>(
          parameter_list_newmark,
          ptr_operator_modes,
          stiffness_matrix.trilinos_matrix(),
          rigid_body_modes);
        prec_newmark.initialize(newmark_matrix, parameter_list_newmark);

        auto parameter_list_C = amg_parameter_list;
        prec_C.initialize(mass_matrix, parameter_list_C);
      }
  }
}


template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::assemble_forcing_terms()
{
  const auto evaluation_time = current_time + par.dt;
  par.rhs.set_time(evaluation_time);
  par.bc.set_time(evaluation_time);
  par.Neumann_bc.set_time(evaluation_time);

  force_rhs      = 0;
  bc_rhs         = 0;
  neumann_bc_rhs = 0;

  TimerOutput::Scope     t(computing_timer, "Assemble rhs");
  FEValues<spacedim>     fe_values(*fe,
                               *quadrature,
                               update_values | update_quadrature_points |
                                 update_JxW_values);
  FEFaceValues<spacedim> fe_face_values(*fe,
                                        *face_quadrature_formula,
                                        update_values | update_JxW_values |
                                          update_quadrature_points |
                                          update_gradients |
                                          update_normal_vectors);

  const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature->size();

  Vector<double>              cell_bc_rhs(dofs_per_cell);
  Vector<double>              cell_neumann_bc_rhs(dofs_per_cell);
  Vector<double>              cell_force_rhs(dofs_per_cell);
  std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(spacedim));

  std::vector<Tensor<2, spacedim>> grad_phi_u(dofs_per_cell);
  std::vector<double>              div_phi_u(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const auto &mp      = par.get_material_properties(cell->material_id());
        cell_bc_rhs         = 0;
        cell_neumann_bc_rhs = 0;
        cell_force_rhs      = 0;

        fe_values.reinit(cell);
        par.rhs.vector_value_list(fe_values.get_quadrature_points(),
                                  rhs_values);

        // Assemble bulk contributions, no constants yet
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const auto comp_i = fe->system_to_component_index(i).first;
                cell_force_rhs(i) += fe_values.shape_value(i, q) *
                                     rhs_values[q][comp_i] * fe_values.JxW(q);
              }
          }

        // Boundary conditions
        for (const auto &f : cell->face_indices())
          if (cell->face(f)->at_boundary())
            {
              // Weak Dirichlet conditions
              if (par.weak_dirichlet_ids.find(cell->face(f)->boundary_id()) !=
                  par.weak_dirichlet_ids.end())
                {
                  fe_face_values.reinit(cell, f);
                  const auto cell_diameter = cell->diameter();

                  for (unsigned int q = 0;
                       q < fe_face_values.n_quadrature_points;
                       ++q)
                    {
                      const auto n = fe_face_values.normal_vector(q);
                      for (unsigned int k = 0; k < dofs_per_cell; ++k)
                        {
                          grad_phi_u[k] =
                            fe_face_values[displacement].symmetric_gradient(k,
                                                                            q);
                          div_phi_u[k] =
                            fe_face_values[displacement].divergence(k, q);
                        }
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          const auto comp_i =
                            fe->system_to_component_index(i).first;
                          const auto g_value =
                            par.bc.value(fe_face_values.quadrature_point(q),
                                         comp_i);
                          Tensor<1, spacedim> g;
                          g[comp_i] = g_value;
                          cell_bc_rhs(i) += par.penalty_term *
                                            (1.0 / cell_diameter) * g_value *
                                            fe_face_values.shape_value(i, q) *
                                            fe_face_values.JxW(q);
                          cell_bc_rhs(i) += -2.0 * mp.Lame_mu *
                                            (grad_phi_u[i] * n * g) *
                                            fe_face_values.JxW(q);
                          cell_bc_rhs(i) += -mp.Lame_lambda *
                                            (div_phi_u[i] * (n * g)) *
                                            fe_face_values.JxW(q);
                        }
                    }
                }
              // Neumann Boundary conditions
              else if (par.neumann_ids.find(cell->face(f)->boundary_id()) !=
                       par.neumann_ids.end())
                {
                  fe_face_values.reinit(cell, f);
                  for (unsigned int q = 0;
                       q < fe_face_values.n_quadrature_points;
                       ++q)
                    {
                      double neumann_value = 0;
                      for (int d = 0; d < spacedim; ++d)
                        neumann_value +=
                          par.Neumann_bc.value(
                            fe_face_values.quadrature_point(q), d) *
                          fe_face_values.normal_vector(q)[d];
                      neumann_value /= spacedim;
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          cell_neumann_bc_rhs(i) +=
                            -neumann_value * fe_face_values.shape_value(i, q) *
                            fe_face_values.JxW(q);
                        }
                    }
                }
            }

        cell->get_dof_indices(local_dof_indices);

        constraints.distribute_local_to_global(cell_force_rhs,
                                               local_dof_indices,
                                               force_rhs.block(0));

        constraints.distribute_local_to_global(cell_bc_rhs,
                                               local_dof_indices,
                                               bc_rhs.block(0));

        constraints.distribute_local_to_global(cell_neumann_bc_rhs,
                                               local_dof_indices,
                                               neumann_bc_rhs.block(0));
      }
  force_rhs.compress(VectorOperation::add);
  bc_rhs.compress(VectorOperation::add);
  neumann_bc_rhs.compress(VectorOperation::add);

  system_rhs          = 0;
  system_rhs.block(0) = force_rhs.block(0);
  system_rhs.block(0) += bc_rhs.block(0);
  system_rhs.block(0) += neumann_bc_rhs.block(0);
}


template <int dim, int spacedim>
IndexSet
ElasticityProblem<dim, spacedim>::assemble_coupling_sparsity(
  DynamicSparsityPattern &dsp)
{
  TimerOutput::Scope t(computing_timer,
                       "Setup dofs: Assemble Coupling sparsity");

  IndexSet relevant(inclusions.n_dofs());

  std::vector<types::global_dof_index> dof_indices(fe->n_dofs_per_cell());
  std::vector<types::global_dof_index> inclusion_dof_indices;

  auto particle = inclusions.inclusions_as_particles.begin();
  while (particle != inclusions.inclusions_as_particles.end())
    {
      const auto &cell = particle->get_surrounding_cell();
      const auto  dh_cell =
        typename DoFHandler<spacedim>::cell_iterator(*cell, &dh);
      dh_cell->get_dof_indices(dof_indices);
      const auto pic =
        inclusions.inclusions_as_particles.particles_in_cell(cell);
      Assert(pic.begin() == particle, ExcInternalError());
      std::set<types::global_dof_index> inclusion_dof_indices_set;
      for (const auto &p : pic)
        {
          const auto ids = inclusions.get_dof_indices(p.get_id());
          inclusion_dof_indices_set.insert(ids.begin(), ids.end());
        }
      inclusion_dof_indices.resize(0);
      inclusion_dof_indices.insert(inclusion_dof_indices.begin(),
                                   inclusion_dof_indices_set.begin(),
                                   inclusion_dof_indices_set.end());
      constraints.add_entries_local_to_global(dof_indices,
                                              inclusion_constraints,
                                              inclusion_dof_indices,
                                              dsp);
      relevant.add_indices(inclusion_dof_indices.begin(),
                           inclusion_dof_indices.end());
      particle = pic.end();
    }
  return relevant;
}



template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::assemble_coupling()
{
  TimerOutput::Scope t(computing_timer, "Assemble Coupling matrix");

  // const FEValuesExtractors::Scalar     scalar(0);
  std::vector<types::global_dof_index> fe_dof_indices(fe->n_dofs_per_cell());
  std::vector<types::global_dof_index> inclusion_dof_indices(
    inclusions.n_dofs_per_inclusion());

  FullMatrix<double> local_coupling_matrix(fe->n_dofs_per_cell(),
                                           inclusions.n_dofs_per_inclusion());

  FullMatrix<double> local_inclusion_matrix(inclusions.n_dofs_per_inclusion(),
                                            inclusions.n_dofs_per_inclusion());

  Vector<double> local_rhs(inclusions.n_dofs_per_inclusion());

  auto particle = inclusions.inclusions_as_particles.begin();
  while (particle != inclusions.inclusions_as_particles.end())
    {
      const auto &cell = particle->get_surrounding_cell();
      const auto  dh_cell =
        typename DoFHandler<spacedim>::cell_iterator(*cell, &dh);
      dh_cell->get_dof_indices(fe_dof_indices);
      const auto pic =
        inclusions.inclusions_as_particles.particles_in_cell(cell);
      Assert(pic.begin() == particle, ExcInternalError());

      auto p      = pic.begin();
      auto next_p = pic.begin();
      while (p != pic.end())
        {
          const auto inclusion_id = inclusions.get_inclusion_id(p->get_id());
          inclusion_dof_indices   = inclusions.get_dof_indices(p->get_id());
          local_coupling_matrix   = 0;
          local_inclusion_matrix  = 0;
          local_rhs               = 0;

          const auto section_measure =
            inclusions.get_section_measure(inclusion_id);

          auto Rotation = inclusions.get_rotation(inclusion_id);

          // Extract all points that refer to the same inclusion
          std::vector<Point<spacedim>> ref_q_points;
          for (; next_p != pic.end() &&
                 inclusions.get_inclusion_id(next_p->get_id()) == inclusion_id;
               ++next_p)
            ref_q_points.push_back(next_p->get_reference_location());
          FEValues<spacedim, spacedim> fev(*fe,
                                           ref_q_points,
                                           update_values | update_gradients);
          fev.reinit(dh_cell);
          // double temp = 0;
          for (unsigned int q = 0; q < ref_q_points.size(); ++q)
            {
              const auto  id                  = p->get_id();
              const auto &inclusion_fe_values = inclusions.get_fe_values(id);
              const auto &real_q              = p->get_location();
              const auto  ds                  = inclusions.get_JxW(id);


              // Coupling and inclusions matrix
              for (unsigned int j = 0; j < inclusions.n_dofs_per_inclusion();
                   ++j)
                {
                  for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
                    {
                      const auto comp_i =
                        fe->system_to_component_index(i).first;
                      //  if (comp_i == inclusions.get_component(j))
                      {
                        local_coupling_matrix(i, j) +=
                          ((fev.shape_value(i, q)) * inclusion_fe_values[j] /
                           section_measure * ds) *
                          (Rotation[comp_i][inclusions.get_component(j)]);
                      }
                    }
                  if (inclusions.inclusions_data[inclusion_id].size() > 0)
                    {
                      if (inclusions.inclusions_data[inclusion_id].size() + 1 >
                          inclusions.get_fourier_component(j))
                        {
                          auto temp =
                            inclusion_fe_values[j] * ds * // /
                            // inclusions.get_section_measure(inclusion_id) *
                            // phi_i ds
                            // now we need to build g from the data.
                            // this is sum E^i g_i where g_i are coefficients
                            // of the modes, but only the j one survives
                            inclusion_fe_values[j] *
                            inclusions.get_inclusion_data(inclusion_id, j);

                          if (par.initial_time != par.final_time)
                            {
                              // temp=temp;

                              temp *= inclusions.inclusions_rhs.value(
                                real_q, inclusions.get_component(j));
                            }
                          temp /= section_measure;
                          local_rhs(j) += temp;
                        }
                    }
                  else
                    {
                      local_rhs(j) +=
                        inclusion_fe_values[j] * // /
                        // inclusions.get_section_measure(inclusion_id) *
                        inclusions.inclusions_rhs.value(
                          real_q, inclusions.get_component(j)) // /
                        // inclusions.get_radius(inclusion_id)
                        * ds / section_measure;
                    }
                  local_inclusion_matrix(j, j) +=
                    (inclusion_fe_values[j] * inclusion_fe_values[j] * ds); // /
                  //  inclusions.get_section_measure(inclusion_id));
                }
              ++p;
            }
          // I expect p and next_p to be the same now.
          Assert(p == next_p, ExcInternalError());
          // Add local matrices to global ones
          constraints.distribute_local_to_global(local_coupling_matrix,
                                                 fe_dof_indices,
                                                 inclusion_constraints,
                                                 inclusion_dof_indices,
                                                 coupling_matrix);
          inclusion_constraints.distribute_local_to_global(
            local_rhs, inclusion_dof_indices, force_rhs.block(1));
          inclusion_constraints.distribute_local_to_global(
            local_rhs, inclusion_dof_indices, system_rhs.block(1));

          inclusion_constraints.distribute_local_to_global(
            local_inclusion_matrix, inclusion_dof_indices, inclusion_matrix);
        }
      particle = pic.end();
    }
  coupling_matrix.compress(VectorOperation::add);
  inclusion_matrix.compress(VectorOperation::add);
  force_rhs.compress(VectorOperation::add);
  system_rhs.block(1) = force_rhs.block(1);

  if (inclusions.n_dofs() > 0)
    {
      Teuchos::ParameterList amg_parameter_list;
      amg_parameter_list.set("smoother: type", "Chebyshev");
      amg_parameter_list.set("smoother: sweeps", 2);
      amg_parameter_list.set("smoother: pre or post", "both");
      amg_parameter_list.set("coarse: type", "Amesos-KLU");
      amg_parameter_list.set("coarse: max size", 2000);
      amg_parameter_list.set("aggregation: threshold", 0.02);
      prec_M.initialize(inclusion_matrix, amg_parameter_list);
    }
}



inline void
output_double_number(double input, const std::string &text)
{
  std::cout << text << input << std::endl;
}



template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::solve()
{
  switch (par.time_mode)
    {
      case TimeMode::Static:
        solve_static();
        break;
      case TimeMode::QuasiStatic:
        solve_quasistatic();
        break;
      case TimeMode::Dynamic:
        solve_newmark();
        break;
      default:
        AssertThrow(false, ExcInternalError());
    }
}

template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::solve_static()
{
  TimerOutput::Scope t(computing_timer, "Solve (static)");
  pcout << "Solving static system..." << std::endl;

  const auto A    = linear_operator<LA::MPI::Vector>(stiffness_matrix);
  const auto amgA = linear_operator(A, prec_A);
  SolverCG<LA::MPI::Vector> cg_stiffness(par.displacement_solver_control);
  const auto                invA = inverse_operator(A, cg_stiffness, amgA);

  auto &u      = solution.block(0);
  auto &lambda = solution.block(1);
  auto &f      = system_rhs.block(0);
  auto &g      = system_rhs.block(1);

  pcout << "   f norm: " << f.l2_norm() << ", g norm: " << g.l2_norm()
        << std::endl;

  if (inclusions.n_dofs() == 0)
    {
      // no inclusions, no coupling
      u = invA * f;

      pcout << "   Solved for u in "
            << par.displacement_solver_control.last_step() << " iterations."
            << std::endl;

      pcout << "   u max: " << u.max() << std::endl;
    }
  else
    {
      if (par.pressure_coupling == false)
        {
          // Solve a saddle point problem
          const auto Bt = linear_operator<LA::MPI::Vector>(coupling_matrix);
          const auto B  = transpose_operator(Bt);
          const auto M  = linear_operator<LA::MPI::Vector>(inclusion_matrix);

          {
            // Estimate condition number:
            pcout << "- - - - - - - - - - - - - - - - - - - - - - - -"
                  << std::endl;
            std::cout << "Estimate condition number of CCt using CG"
                      << std::endl;
            SolverControl             solver_control(2000, 1e-12);
            SolverCG<LA::MPI::Vector> solver_cg(solver_control);

            solver_cg.connect_condition_number_slot(
              std::bind(output_double_number,
                        std::placeholders::_1,
                        "Condition number estimate: "));

            auto CCt = B * Bt;

            LA::MPI::Vector u;
            u.reinit(system_rhs.block(1));
            u = 0.;

            LA::MPI::Vector f;
            f.reinit(system_rhs.block(1));
            f = 1.;
            PreconditionIdentity prec_no;
            try
              {
                solver_cg.solve(CCt, u, f, prec_no);
              }
            catch (...)
              {
                std::cerr
                  << "***CCt solve not successfull (see condition number above)***"
                  << std::endl;
              }
          }

#ifdef FALSE
          { // auto interp_g = g;
            // interp_g      = 0.1;
            // g             = C * interp_g;

            // Schur complement
            const auto S = B * invA * Bt;

            // Schur complement preconditioner
            // VERSION 1
            // auto                          invS = S;
            // SolverFGMRES<LA::MPI::Vector> cg_schur(par.outer_control);
            SolverMinRes<LA::MPI::Vector> cg_schur(par.schur_control);
            // invS = inverse_operator(S, cg_schur);
            // VERSION2
            auto invS       = S;
            auto S_inv_prec = B * invA * Bt + M;
            // auto S_inv_prec = B * invA * Bt;
            // SolverCG<Vector<double>> cg_schur(par.outer_control);
            // PrimitiveVectorMemory<Vector<double>> mem;
            // SolverGMRES<Vector<double>> solver_gmres(
            //                     par.outer_control, mem,
            //                     SolverGMRES<Vector<double>>::AdditionalData(20));
            invS = inverse_operator(S, cg_schur, S_inv_prec);

            pcout << "   f norm: " << f.l2_norm() << ", g norm: " << g.l2_norm()
                  << std::endl;
            // pcout << "   g: ";
            // g.print(std::cout);

            // Compute Lambda first
            lambda = invS * (B * invA * f - g);
            pcout << "   Solved for lambda in " << par.schur_control.last_step()
                  << " iterations." << std::endl;

            // Then compute u
            u = invA * (f - Bt * lambda);
            pcout << "   u norm: " << u.l2_norm()
                  << ", lambda norm: " << lambda.l2_norm() << std::endl;
            // std::cout << "   lambda: ";
            // lambda.print(std::cout);
          }
#endif
          {
            const auto M = linear_operator<LA::MPI::Vector>(inclusion_matrix);
            const auto amgM = linear_operator(M, prec_M);
            SolverCG<TrilinosWrappers::MPI::Vector> solver_CG_M(
              par.reduced_mass_solver_control);
            auto invM = inverse_operator(M, solver_CG_M, amgM);
            auto invW = invM * invM;

            // Try augmented lagrangian preconditioner
            const double gamma = 10;
            auto         Aug   = A + gamma * Bt * invW * B;


            auto Zero = M * 0.0;
            auto AA   = block_operator<2, 2, LA::MPI::BlockVector>(
              {{{{Aug, Bt}}, {{B, Zero}}}}); //! Augmented the (1,1) block

            LA::MPI::BlockVector solution_block;
            LA::MPI::BlockVector system_rhs_block;
            AA.reinit_domain_vector(solution_block, false);
            AA.reinit_range_vector(system_rhs_block, false);


            // lagrangian term
            LA::MPI::Vector tmp;
            tmp.reinit(system_rhs.block(0));
            tmp                       = gamma * Bt * invW * system_rhs.block(1);
            system_rhs_block.block(0) = system_rhs.block(0);
            system_rhs_block.block(0).add(1., tmp); // ! augmented
            system_rhs_block.block(1) = system_rhs.block(1);

            SolverCG<LA::MPI::Vector> solver_lagrangian(
              par.displacement_solver_control);

            auto Aug_inv =
              inverse_operator(Aug, solver_lagrangian); //! augmented

            SolverFGMRES<LA::MPI::BlockVector> solver_fgmres(
              par.augmented_lagrange_solver_control);

            BlockPreconditionerAugmentedLagrangian<LA::MPI::Vector>
              augmented_lagrangian_preconditioner{Aug_inv, B, Bt, invW, gamma};

            solver_fgmres.solve(AA,
                                solution_block,
                                system_rhs_block,
                                augmented_lagrangian_preconditioner);

            solution.block(0) = solution_block.block(0);
            solution.block(1) = solution_block.block(1);
            pcout << "Solver with FGMRES in "
                  << par.augmented_lagrange_solver_control.last_step()
                  << " iterations." << std::endl;
            constraints.distribute(solution_block.block(0));
            inclusion_constraints.distribute(solution_block.block(1));
          }

          return;
        }
      else
        {
          // pressure_coupling == true
          const auto Bt   = linear_operator<LA::MPI::Vector>(coupling_matrix);
          const auto B    = transpose_operator(Bt);
          const auto M    = linear_operator<LA::MPI::Vector>(inclusion_matrix);
          const auto amgM = linear_operator(M, prec_M);

          // Solver for M
          auto                      invM = M;
          SolverCG<LA::MPI::Vector> cg_M(par.reduced_mass_solver_control);
          invM = inverse_operator(M, cg_M, amgM);

          // Compute the rhs, given the data file as if it was a pressure
          // condition on the vessels.
          lambda = invM * g;

          pcout << "   Solved for lambda "
                << par.reduced_mass_solver_control.last_step() << " iterations."
                << std::endl;

          u = invA * (f + Bt * lambda);

          pcout << "   Solved for u "
                << par.displacement_solver_control.last_step() << " iterations."
                << std::endl;
        }
    }


  constraints.distribute(u);
  inclusion_constraints.distribute(lambda);
  locally_relevant_solution = solution;
}

template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::solve_quasistatic()
{
  TimerOutput::Scope t(computing_timer, "Solve (quasistatic)");

  AssertThrow(par.pressure_coupling == false || inclusions.n_dofs() == 0,
              ExcNotImplemented("Quasi-static solve not implemented for "
                                "pressure_coupling == true."));
  AssertThrow(inclusions.n_dofs() == 0,
              ExcNotImplemented("Quasi-static solve not implemented for "
                                "inclusions."));

  const auto A    = linear_operator<LA::MPI::Vector>(stiffness_matrix);
  const auto amgA = linear_operator(A, prec_A);
  SolverCG<LA::MPI::Vector> cg_stiffness(par.displacement_solver_control);
  const auto                invA = inverse_operator(A, cg_stiffness, amgA);

  auto &u = solution.block(0);
  auto &f = system_rhs.block(0);

  u = invA * f;

  pcout << "   Solved for u " << par.displacement_solver_control.last_step()
        << " iterations." << std::endl;

  constraints.distribute(u);
  locally_relevant_solution = solution;
}

template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::solve_newmark()
{
  TimerOutput::Scope t(computing_timer, "Solve (newmark)");

  AssertThrow(par.pressure_coupling == true || inclusions.n_dofs() == 0,
              ExcNotImplemented("Dynamic solve not implemented for "
                                "pressure_coupling == false."));

  const auto A = linear_operator<LA::MPI::Vector>(stiffness_matrix);
  const auto D = linear_operator<LA::MPI::Vector>(damping_matrix);

  auto &u      = solution.block(0);
  auto &lambda = solution.block(1);
  auto &v      = velocity.block(0);
  auto &v_pred = predictor.block(0);
  auto &a      = acceleration.block(0);
  auto &u_pred = corrector.block(0);
  auto &f      = system_rhs.block(0);
  auto &g      = system_rhs.block(1);

  const auto N    = linear_operator<LA::MPI::Vector>(newmark_matrix);
  const auto amgN = linear_operator(N, prec_newmark);
  SolverCG<LA::MPI::Vector> cg_stiffness(par.displacement_solver_control);
  const auto                invN = inverse_operator(N, cg_stiffness, amgN);

  const double beta  = par.beta;
  const double gamma = par.gamma;

  // predictor step
  u_pred = u + par.dt * v + (par.dt * par.dt / 2) * (1 - 2 * beta) * a;
  v_pred = v + par.dt * (1 - gamma) * a;

  if (par.elasticity_model == ElasticityModel::LinearElasticity ||
      par.elasticity_model == ElasticityModel::KelvinVoigt)
    {
      if (inclusions.n_dofs() == 0)
        {
          a = invN * (f - D * v_pred - A * u_pred);
        }
      else
        {
          // Solve the problem with input data coming from the file.
          const auto Bt   = linear_operator<LA::MPI::Vector>(coupling_matrix);
          const auto B    = transpose_operator(Bt);
          const auto M    = linear_operator<LA::MPI::Vector>(inclusion_matrix);
          const auto amgM = linear_operator(M, prec_M);

          // Solver for M
          auto                      invM = M;
          SolverCG<LA::MPI::Vector> cg_M(par.reduced_mass_solver_control);
          invM = inverse_operator(M, cg_M, amgM);

          // Compute the rhs, given the data file as if it was a pressure
          // condition on the vessels.
          lambda = invM * g;

          pcout << "   Solved for lambda "
                << par.reduced_mass_solver_control.last_step() << " iterations."
                << std::endl;

          const auto f_inclusions = Bt * lambda;
          a = invN * (f_inclusions + f - D * v_pred - A * u_pred);
        }
    }
  else
    {
      AssertThrow(false, ExcInternalError());
    }

  // corrector step
  u = u_pred + par.dt * par.dt * beta * a;
  v = v_pred + par.dt * gamma * a;

  pcout << "   Solved for u " << par.displacement_solver_control.last_step()
        << " iterations." << std::endl;

  pcout << "   u max: " << u.max() << std::endl;

  constraints.distribute(u);
  inclusion_constraints.distribute(lambda);
  locally_relevant_solution = solution;
}



template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::refine_and_transfer()
{
  TimerOutput::Scope t(computing_timer, "Refine");
  Vector<float>      error_per_cell(tria.n_active_cells());
  KellyErrorEstimator<spacedim>::estimate(dh,
                                          QGauss<spacedim - 1>(par.fe_degree +
                                                               1),
                                          {},
                                          locally_relevant_solution.block(0),
                                          error_per_cell);
  if (par.refinement_strategy == "fixed_fraction")
    {
      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
        tria, error_per_cell, par.refinement_fraction, par.coarsening_fraction);
    }
  else if (par.refinement_strategy == "fixed_number")
    {
      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        error_per_cell,
        par.refinement_fraction,
        par.coarsening_fraction,
        par.max_cells);
    }
  else if (par.refinement_strategy == "global")
    for (const auto &cell : tria.active_cell_iterators())
      cell->set_refine_flag();

  execute_actual_refine_and_transfer();
}
template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::execute_actual_refine_and_transfer()
{
  parallel::distributed::SolutionTransfer<spacedim, LA::MPI::Vector> transfer(
    dh);
  tria.prepare_coarsening_and_refinement();
  inclusions.inclusions_as_particles.prepare_for_coarsening_and_refinement();
  transfer.prepare_for_coarsening_and_refinement(
    locally_relevant_solution.block(0));
  // transfer.prepare_for_coarsening_and_refinement(
  //  locally_relevant_solution.block(1));
  tria.execute_coarsening_and_refinement();
  inclusions.inclusions_as_particles.unpack_after_coarsening_and_refinement();
  setup_dofs();
  transfer.interpolate(solution.block(0));
  constraints.distribute(solution.block(0));
  locally_relevant_solution.block(0) = solution.block(0);
  locally_relevant_solution.block(1) = solution.block(1);
}



template <int dim, int spacedim>
std::string
ElasticityProblem<dim, spacedim>::output_solution() const
{
  std::vector<std::string> solution_names(spacedim, "displacement");
  std::vector<std::string> exact_solution_names(spacedim, "exact_displacement");


  auto exact_vec(solution.block(0));
  exact_vec = 0.0;

  // Ensure the exact solution is evaluated at the current time.
  par.exact_solution.set_time(current_time);

  VectorTools::interpolate(dh, par.exact_solution, exact_vec);
  constraints.distribute(exact_vec);
  auto exact_vec_locally_relevant(locally_relevant_solution.block(0));

  exact_vec_locally_relevant = 0.0;
  exact_vec_locally_relevant = exact_vec;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      spacedim, DataComponentInterpretation::component_is_part_of_vector);

  DataOut<spacedim> data_out;
  data_out.attach_dof_handler(dh);
  data_out.add_data_vector(locally_relevant_solution.block(0),
                           solution_names,
                           DataOut<spacedim>::type_dof_data,
                           data_component_interpretation);

  data_out.add_data_vector(exact_vec_locally_relevant,
                           exact_solution_names,
                           DataOut<spacedim>::type_dof_data,
                           data_component_interpretation);

  Vector<float> subdomain(tria.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  Vector<double> material_ids(tria.n_active_cells());
  {
    auto cell = tria.begin_active();
    auto endc = tria.end();
    for (unsigned int i = 0; cell != endc; ++cell, ++i)
      {
        material_ids[i] = cell->material_id();
      }
  }

  data_out.add_data_vector(material_ids, "material_id");

  data_out.build_patches();
  std::ostringstream filename;
  if (par.time_mode == TimeMode::Static)
    {
      filename << par.output_name << "_" << cycle << ".vtu";
    }
  else
    {
      filename << par.output_name << "_" << cycle << "_" << std::setw(3)
               << std::setfill('0') << time_step << ".vtu";
    }
  const std::string filename_str = filename.str();
  data_out.write_vtu_in_parallel(par.output_directory + "/" + filename_str,
                                 mpi_communicator);
  return filename_str;
}

// template <int dim, int spacedim>
// std::string
// ElasticityProblem<dim, spacedim>::output_stresses() const
// {
//   std::vector<std::string> solution_names(spacedim*spacedim,
//   "face_stress");

//   std::vector<DataComponentInterpretation::DataComponentInterpretation>
//     face_component_type(
//       spacedim, DataComponentInterpretation::component_is_part_of_vector);

//   DataOutFaces<spacedim> data_out_faces(true);
//   data_out_faces.add_data_vector(dh,
//                            sigma_n,
//                            solution_names,
//                            face_component_type);

//   data_out_faces.build_patches(fe->degree);
//   const std::string filename =
//     par.output_name + "_stress_" + std::to_string(cycle) + ".vtu";
//   data_out_faces.write_vtu_in_parallel(par.output_directory + "/" +
//   filename,
//                                  mpi_communicator);
//   return filename;
// }



// template <int dim, int spacedim>
// std::string
// ElasticityProblem<dim, spacedim>::output_particles() const
// {
//   Particles::DataOut<spacedim> particles_out;
//   particles_out.build_patches(inclusions.inclusions_as_particles);
//   const std::string filename =
//     par.output_name + "_particles_" + std::to_string(cycle) + ".vtu";
//   particles_out.write_vtu_in_parallel(par.output_directory + "/" +
//   filename,
//                                       mpi_communicator);
//   return filename;
// }


template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::output_results() const
{
  TimerOutput::Scope t(computing_timer, "Postprocessing: Output results");
  static std::vector<std::pair<double, std::string>> cycles_and_solutions;
  static std::vector<std::vector<std::pair<double, std::string>>>
    cycles_and_solutions_by_cycle;
  static std::vector<std::pair<double, std::string>> cycles_and_particles;
  static std::vector<std::vector<std::pair<double, std::string>>>
    cycles_and_particles_by_cycle;
  // static std::vector<std::pair<double, std::string>> cycles_and_stresses;

  static unsigned int last_output_cycle =
    std::numeric_limits<unsigned int>::max();
  static unsigned int last_output_step =
    std::numeric_limits<unsigned int>::max();

  if (last_output_cycle != cycle || last_output_step != time_step)
    {
      last_output_cycle        = cycle;
      last_output_step         = time_step;
      const double output_time = (par.time_mode == TimeMode::Static) ?
                                   static_cast<double>(cycle) :
                                   current_time;
      if (par.time_mode == TimeMode::Static)
        {
          cycles_and_solutions.push_back({output_time, output_solution()});
          std::ofstream pvd_solutions(par.output_directory + "/" +
                                      par.output_name + ".pvd");
          DataOutBase::write_pvd_record(pvd_solutions, cycles_and_solutions);
        }
      else
        {
          if (cycles_and_solutions_by_cycle.size() <= cycle)
            cycles_and_solutions_by_cycle.resize(cycle + 1);
          auto &records = cycles_and_solutions_by_cycle[cycle];
          records.push_back({output_time, output_solution()});
          const std::string pvd_name = par.output_directory + "/" +
                                       par.output_name + "_cycle_" +
                                       std::to_string(cycle) + ".pvd";
          std::ofstream pvd_solutions(pvd_name);
          DataOutBase::write_pvd_record(pvd_solutions, records);
        }

      if (par.time_mode == TimeMode::Static && cycle == 0 && time_step == 0)
        {
          const std::string particles_filename =
            par.output_name + "_particles.vtu";

          inclusions.output_particles(par.output_directory + "/" +
                                      particles_filename);
          cycles_and_particles.push_back({output_time, particles_filename});

          std::ofstream pvd_particles(par.output_directory + "/" +
                                      par.output_name + "_particles.pvd");
          DataOutBase::write_pvd_record(pvd_particles, cycles_and_particles);
        }
      else if (par.time_mode != TimeMode::Static)
        {
          if (cycles_and_particles_by_cycle.size() <= cycle)
            cycles_and_particles_by_cycle.resize(cycle + 1);
          auto &particle_records = cycles_and_particles_by_cycle[cycle];
          std::ostringstream particles_filename;
          particles_filename << par.output_name << "_particles_" << cycle << "_"
                             << std::setw(3) << std::setfill('0') << time_step
                             << ".vtu";

          const std::string particles_filename_str = particles_filename.str();
          inclusions.output_particles(par.output_directory + "/" +
                                      particles_filename_str);
          particle_records.push_back({output_time, particles_filename_str});

          const std::string pvd_particles_name =
            par.output_directory + "/" + par.output_name + "_particles_cycle_" +
            std::to_string(cycle) + ".pvd";
          std::ofstream pvd_particles(pvd_particles_name);
          DataOutBase::write_pvd_record(pvd_particles, particle_records);
        }
    }
}

template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::print_parameters() const
{
#ifdef USE_PETSC_LA
  pcout << "Running ElasticityProblem<" << Utilities::dim_string(dim, spacedim)
        << "> using PETSc." << std::endl;
#else
  pcout << "Running ElasticityProblem<" << Utilities::dim_string(dim, spacedim)
        << "> using Trilinos." << std::endl;
#endif
  par.prm.print_parameters(par.output_directory + "/" + par.output_name + "_" +
                             std::to_string(dim) + std::to_string(spacedim) +
                             ".prm",
                           ParameterHandler::Short);
  par.prm.print_parameters(par.output_directory + "/" + par.output_name +
                             "_full_" + std::to_string(dim) +
                             std::to_string(spacedim) + ".prm",
                           ParameterHandler::PRM);
#if DEAL_II_VERSION_GTE(9, 7, 0)
  par.prm.print_parameters(par.output_directory + "/" + par.output_name +
                             "_reduced_" + std::to_string(dim) +
                             std::to_string(spacedim) + ".prm",
                           ParameterHandler::KeepOnlyChanged |
                             ParameterHandler::Short);
#endif
}

/**
 * @brief compute stresses on boundaries (2D and 3D) and internal (2D)
 * this function makes use of boundary id, so make sure that the ifd starts
 * from 0 and are sequential when importing a mesh, this is automatically
 * taken care of for meshes generated with GridTools output is a txt file
 * containing stresses at each cycle/time
 */
template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::compute_internal_and_boundary_stress(
  bool openfilefirsttime) const
{
  TimerOutput::Scope t(computing_timer, "Postprocessing: Computing stresses");

  std::map<types::boundary_id, Tensor<1, spacedim>> boundary_stress;
  std::map<types::boundary_id, double>              u_dot_n;

  auto                                 all_ids = tria.get_boundary_ids();
  std::map<types::boundary_id, double> perimeter;
  for (auto id : all_ids)
    // for (const auto id : par.dirichlet_ids)
    {
      boundary_stress[id] = 0.0;
      perimeter[id]       = 0.0;
      u_dot_n[id]         = 0.0;
    }
  double internal_area = 0.;
  //   // FEValues<spacedim>               fe_values(*fe,
  //   //                              *quadrature,
  //   //                              update_values | update_gradients |
  //   //                                update_quadrature_points | update_JxW_values);
  FEFaceValues<spacedim>           fe_face_values(*fe,
                                        *face_quadrature_formula,
                                        update_values | update_gradients |
                                          update_JxW_values |
                                          update_quadrature_points |
                                          update_normal_vectors);
  const FEValuesExtractors::Vector displacement(0);

  //   // const unsigned int                   dofs_per_cell =
  //   fe->n_dofs_per_cell();
  //   // const unsigned int                   n_q_points    =
  //   quadrature->size();
  //   // std::vector<types::global_dof_index>
  //   local_dof_indices(dofs_per_cell);
  //   // Tensor<2, spacedim>                  grad_phi_u;
  //   // double                               div_phi_u;
  Tensor<2, spacedim> identity;
  for (unsigned int ix = 0; ix < spacedim; ++ix)
    identity[ix][ix] = 1;

  //   // std::vector<std::vector<Tensor<1,spacedim>>>
  //   // solution_gradient(face_quadrature_formula->size(),
  //   // std::vector<Tensor<1,spacedim> >(spacedim+1));
  std::vector<Tensor<2, spacedim>> displacement_gradient(
    face_quadrature_formula->size());
  std::vector<double> displacement_divergence(face_quadrature_formula->size());
  std::vector<Tensor<1, spacedim>> displacement_values(
    face_quadrature_formula->size());

  for (const auto &cell : dh.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          const auto &mp = par.get_material_properties(cell->material_id());
          //           // if constexpr (spacedim == 2)
          //           //   {
          //           //     cell->get_dof_indices(local_dof_indices);
          //           //     fe_values.reinit(cell);
          //           //     for (unsigned int q = 0; q < n_q_points; ++q)
          //           //       {
          //           //         internal_area += fe_values.JxW(q);
          //           //         for (unsigned int k = 0; k < dofs_per_cell;
          //           ++k)
          //           //           {
          //           //             grad_phi_u =
          //           // fe_values[displacement].symmetric_gradient(k, q);
          //           //             div_phi_u =
          //           fe_values[displacement].divergence(k, q);
          //           //             internal_stress +=
          //           //               (2 * par.Lame_mu * grad_phi_u +
          //           //                par.Lame_lambda * div_phi_u *
          //           identity)
          //           *
          //           //               locally_relevant_solution.block(
          //           //                 0)[local_dof_indices[k]] *
          //           //               fe_values.JxW(q);
          //           //             average_displacement +=
          //           //               fe_values[displacement].value(k, q) *
          //           //               locally_relevant_solution.block(
          //           //                 0)[local_dof_indices[k]] *
          //           //               fe_values.JxW(q);
          //           //           }
          //           //       }
          //           //   }

          for (unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell;
               ++f)
            // for (const auto &f : cell->face_iterators())
            // for (const auto f : GeometryInfo<spacedim>::face_indices())
            if (cell->face(f)->at_boundary())
              {
                auto boundary_index = cell->face(f)->boundary_id();
                fe_face_values.reinit(cell, f);

                fe_face_values[displacement].get_function_gradients(
                  locally_relevant_solution.block(0), displacement_gradient);
                fe_face_values[displacement].get_function_values(
                  locally_relevant_solution.block(0), displacement_values);
                fe_face_values[displacement].get_function_divergences(
                  locally_relevant_solution.block(0), displacement_divergence);

                for (unsigned int q = 0; q < fe_face_values.n_quadrature_points;
                     ++q)
                  {
                    perimeter[boundary_index] += fe_face_values.JxW(q);
                    boundary_stress[boundary_index] +=
                      (2 * mp.Lame_mu * displacement_gradient[q] +
                       mp.Lame_lambda * displacement_divergence[q] * identity) *
                      fe_face_values.JxW(q) * fe_face_values.normal_vector(q);
                    u_dot_n[boundary_index] +=
                      (displacement_values[q] *
                       fe_face_values.normal_vector(q)) *
                      fe_face_values.JxW(q);
                  }
              }
        }
    }

  // if constexpr (spacedim == 2)
  //   {
  //     internal_stress = Utilities::MPI::sum(internal_stress,
  //     mpi_communicator); average_displacement =
  //       Utilities::MPI::sum(average_displacement, mpi_communicator);
  //     internal_area = Utilities::MPI::sum(internal_area, mpi_communicator);

  //     internal_stress /= internal_area;
  //     average_displacement /= internal_area;
  //   }
  for (auto id : all_ids) // par.dirichlet_ids) // all_ids)
    {
      boundary_stress[id] =
        Utilities::MPI::sum(boundary_stress[id], mpi_communicator);
      perimeter[id] = Utilities::MPI::sum(perimeter[id], mpi_communicator);
      Assert(perimeter[id] > 0, ExcInternalError());
      boundary_stress[id] /= perimeter[id];
    }

  const unsigned int output_index =
    (par.time_mode == TimeMode::Static) ? cycle : time_step;
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      const std::string filename(par.output_directory + "/" + par.output_name +
                                 "_forces.txt");
      std::ofstream     forces_file;
      if (openfilefirsttime)
        {
          forces_file.open(filename);
          if constexpr (spacedim == 2)
            {
              forces_file << "cycle area";
              // forces_file
              //   << " meanInternalStressxx meanInternalStressxy
              //   meanInternalStressyx meanInternalStressyy avg_u_x avg_u_y";
              for (auto id : all_ids)
                forces_file // << " perimeter" << id
                  << " boundaryStressX_" << id << " boundaryStressY_" << id
                  << " uDotN_" << id;
              forces_file << std::endl;
            }
          else
            {
              forces_file << "cycle";
              for (auto id : all_ids)
                forces_file // << " perimeter" << id
                  << " sigmanX_" << id << " sigmanY_" << id << " sigmanZ_" << id
                  << " uDotN_" << id;
              forces_file << std::endl;
            }
        }
      else
        forces_file.open(filename, std::ios_base::app);

      if constexpr (spacedim == 2)
        {
          forces_file << output_index << " " << internal_area << " ";
          for (auto id : all_ids)
            forces_file // << perimeter[id] << " "
              << boundary_stress[id] << " " << u_dot_n[id] << " ";
          forces_file << std::endl;
        }
      else // spacedim = 3
        {
          forces_file << output_index << " ";
          for (auto id : all_ids)
            forces_file // << perimeter[id] << " "
              << boundary_stress[id] << " " << u_dot_n[id] << " ";
          forces_file << std::endl;
        }
      forces_file.close();
    }

  return;
}


template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::output_lambda() const
{
  auto      &lambda            = locally_relevant_solution.block(1);
  const auto used_number_modes = inclusions.get_n_coefficients();

  if (lambda.size() > 0)
    {
      const std::string filename(par.output_directory + "/lambda.txt");
      std::ofstream     file;
      file.open(filename);

      unsigned int tot = inclusions.reference_inclusion_data.size() - 4;
      Assert(tot > 0, ExcNotImplemented());

      for (unsigned int i = 0; i < inclusions.n_inclusions(); ++i)
        {
          for (unsigned int j = 0; j < tot; ++j)
            file << "lambda" << (i * tot + j) << " ";
          file << "lambda" << i << "norm ";
        }
      file << std::endl;

      for (unsigned int i = 0; i < inclusions.n_inclusions(); ++i)
        {
          unsigned int mode_of_inclusion = 0;
          double       lambda_norm       = 0;
          for (mode_of_inclusion = 0; mode_of_inclusion < used_number_modes;
               mode_of_inclusion++)
            {
              auto elem_of_lambda = i * used_number_modes + mode_of_inclusion;
              file << lambda[elem_of_lambda] << " ";
              lambda_norm += lambda[elem_of_lambda] * lambda[elem_of_lambda];
            }
          while (mode_of_inclusion < tot)
            {
              file << "0 ";
              mode_of_inclusion++;
            }
          file << lambda_norm << " ";
        }
      file << std::endl;
      file.close();
    }
}

/**
 * @brief set up, assemble and run the problem
 */
template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::run()
{
  switch (par.time_mode)
    {
      case TimeMode::Static:
        run_static();
        break;
      case TimeMode::QuasiStatic:
        run_quasistatic();
        break;
      case TimeMode::Dynamic:
        run_newmark();
        break;
      default:
        AssertThrow(false, ExcInternalError());
    }
}

template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::run_static()
{
  AssertThrow(par.time_mode == TimeMode::Static, ExcInternalError());

  print_parameters();
  make_grid();
  setup_fe();

  current_time = par.initial_time;

  {
    TimerOutput::Scope t(computing_timer, "Setup inclusion");
    inclusions.setup_inclusions_particles(tria);
  }

  setup_dofs(); // called inside refine_and_transfer
  for (cycle = 0; cycle < par.n_refinement_cycles; ++cycle)
    {
      setup_dofs();
      time_step = 0;
      par.dt    = par.dt;
      if (par.output_results_before_solving)
        output_results();
      assemble_elasticity_system();
      assemble_coupling();
      solve_static();
      output_results();
      if (spacedim == 2)
        {
          FunctionParser<spacedim> weight(par.weight_expression);
          par.convergence_table.error_from_exact(
            dh,
            locally_relevant_solution.block(0),
            par.exact_solution,
            &weight);
        }
      else
        par.convergence_table.error_from_exact(
          dh, locally_relevant_solution.block(0), par.bc);
      if (cycle != par.n_refinement_cycles - 1)
        refine_and_transfer();
    }

  compute_internal_and_boundary_stress(true);

  if (pcout.is_active())
    par.convergence_table.output_table(pcout.get_stream());

  if (false)
    output_lambda();
}

template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::run_quasistatic()
{
  AssertThrow(par.time_mode == TimeMode::QuasiStatic, ExcInternalError());
  pcout << "time dependent simulation" << std::endl;

  print_parameters();
  make_grid();
  setup_fe();

  cycle = 0;
  {
    TimerOutput::Scope t(computing_timer, "Setup inclusion");
    inclusions.setup_inclusions_particles(tria);
  }

  setup_dofs();
  for (cycle = 0; cycle < par.n_refinement_cycles; ++cycle)
    {
      setup_dofs();
      assemble_elasticity_system();
      current_time = par.initial_time;
      time_step    = 0;
      assemble_forcing_terms();
      inclusions.inclusions_rhs.set_time(current_time);
      assemble_coupling();

      for (time_step = 0, current_time = par.initial_time;
           current_time < par.final_time;
           current_time += par.dt, ++time_step)
        {
          compute_system_rhs();
          solve_quasistatic();
          output_results();
          const bool openfilefirsttime = (cycle == 0 && time_step == 0);

          if (par.domain_type == "generate")
            compute_internal_and_boundary_stress(openfilefirsttime);
        }

      if (cycle != par.n_refinement_cycles - 1)
        {
          refine_and_transfer();
          if (par.refine_time_step)
            par.dt *= 0.5;
        }
    }
}

template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::run_newmark()
{
  AssertThrow(par.time_mode == TimeMode::Dynamic, ExcInternalError());
  pcout << "time dependent simulation" << std::endl;

  print_parameters();
  make_grid();
  setup_fe();

  cycle = 0;
  {
    TimerOutput::Scope t(computing_timer, "Setup inclusion");
    inclusions.setup_inclusions_particles(tria);
  }

  setup_dofs();
  for (cycle = 0; cycle < par.n_refinement_cycles; ++cycle)
    {
      assemble_elasticity_system();
      current_time = par.initial_time;
      time_step    = 0;
      assemble_forcing_terms();
      inclusions.inclusions_rhs.set_time(current_time);
      assemble_coupling();

      par.initial_displacement.set_time(current_time);
      par.initial_velocity.set_time(current_time);
      VectorTools::interpolate(dh, par.initial_displacement, solution.block(0));
      constraints.distribute(solution.block(0));
      VectorTools::interpolate(dh, par.initial_velocity, velocity.block(0));
      constraints.distribute(velocity.block(0));

      // Initialize acceleration consistently at t0:
      // C a0 = f0 - D v0 - A u0 (plus inclusion forcing if present).
      compute_system_rhs();
      {
        const auto A = linear_operator<LA::MPI::Vector>(stiffness_matrix);
        const auto D = linear_operator<LA::MPI::Vector>(damping_matrix);
        const auto C = linear_operator<LA::MPI::Vector>(mass_matrix);

        auto &u      = solution.block(0);
        auto &v      = velocity.block(0);
        auto &a      = acceleration.block(0);
        auto &lambda = solution.block(1);

        auto rhs = system_rhs.block(0);

        if (inclusions.n_dofs() == 0)
          {
            rhs -= D * v;
            rhs -= A * u;
          }
        else
          {
            const auto Bt = linear_operator<LA::MPI::Vector>(coupling_matrix);
            const auto M  = linear_operator<LA::MPI::Vector>(inclusion_matrix);
            const auto amgM = linear_operator(M, prec_M);

            auto                      invM = M;
            SolverCG<LA::MPI::Vector> cg_M(par.reduced_mass_solver_control);
            invM = inverse_operator(M, cg_M, amgM);

            lambda                  = invM * system_rhs.block(1);
            const auto f_inclusions = Bt * lambda;

            rhs += f_inclusions;
            rhs -= D * v;
            rhs -= A * u;
          }

        const auto                amgC = linear_operator(C, prec_C);
        SolverCG<LA::MPI::Vector> cg_mass(par.displacement_solver_control);
        const auto                invC = inverse_operator(C, cg_mass, amgC);
        a                              = invC * rhs;
      }

      locally_relevant_solution = solution;

      for (time_step = 0, current_time = par.initial_time;
           current_time < par.final_time;
           current_time += par.dt, ++time_step)
        {
          output_results();
          const bool openfilefirsttime = (cycle == 0 && time_step == 0);
          compute_system_rhs();
          solve_newmark();

          if (par.domain_type == "generate")
            compute_internal_and_boundary_stress(openfilefirsttime);
        }
      output_results();

      par.exact_solution.set_time(current_time);
      par.convergence_table.error_from_exact(dh,
                                             locally_relevant_solution.block(0),
                                             par.exact_solution);

      if (cycle != par.n_refinement_cycles - 1)
        {
          refine_and_transfer();
          if (par.refine_time_step)
            par.dt *= 0.5;
        }
      if (pcout.is_active())
        par.convergence_table.output_table(pcout.get_stream());
    }
}


template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::compute_system_rhs()
{
  const auto get_scale = [](const double modulation, const double time) {
    return (modulation == 0.0) ? 1.0 :
                                 std::sin(numbers::PI * 2 * modulation * time);
  };

  const auto is_zero = [](const double x) { return std::abs(x) == 0.0; };

  const bool rhs_has_any_zero_modulation = is_zero(par.rhs_modulation) ||
                                           is_zero(par.bc_modulation) ||
                                           is_zero(par.neumann_bc_modulation);
  const bool inclusion_has_zero_modulation =
    is_zero(inclusions.modulation_frequency);

  pcout << "Time: " << current_time << std::endl;

  if (rhs_has_any_zero_modulation)
    assemble_forcing_terms();

  if (inclusion_has_zero_modulation)
    {
      inclusions.inclusions_rhs.set_time(current_time);
      assemble_coupling();
    }


  const auto rhs_scale     = get_scale(par.rhs_modulation, current_time);
  const auto bc_scale      = get_scale(par.bc_modulation, current_time);
  const auto neumann_scale = get_scale(par.neumann_bc_modulation, current_time);
  const auto inclusion_scale =
    get_scale(inclusions.modulation_frequency, current_time);

  system_rhs.block(0) = 0.0;
  system_rhs.block(0).add(rhs_scale, force_rhs.block(0));
  system_rhs.block(0).add(bc_scale, bc_rhs.block(0));
  system_rhs.block(0).add(neumann_scale, neumann_bc_rhs.block(0));

  system_rhs.block(1) = force_rhs.block(1);
  system_rhs.block(1) *= inclusion_scale;
}



// Template instantiations
template class ElasticityProblem<2>;
template class ElasticityProblem<2, 3>; // dim != spacedim
template class ElasticityProblem<3>;

template class RigidBodyMotion<2>;
template class RigidBodyMotion<3>;
