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

/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 * Modified by: Luca Heltai, 2020
 */



#include "elasticity.h"

#include <boost/algorithm/string.hpp>

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
read_grid_and_cad_files(const std::string            &grid_file_name,
                        const std::string            &ids_and_cad_file_names,
                        Triangulation<dim, spacedim> &tria)
{
  GridIn<dim, spacedim> grid_in;
  grid_in.attach_triangulation(tria);
  grid_in.read(grid_file_name);
#ifdef DEAL_II_WITH_OPENCASCADE
  using map_type  = std::map<types::manifold_id, std::string>;
  using Converter = Patterns::Tools::Convert<map_type>;
  for (const auto &pair : Converter::to_value(ids_and_cad_file_names))
    {
      const auto &manifold_id   = pair.first;
      const auto &cad_file_name = pair.second;
      const auto  extension     = boost::to_lower_copy(
        cad_file_name.substr(cad_file_name.find_last_of('.') + 1));
      TopoDS_Shape shape;
      if (extension == "iges" || extension == "igs")
        shape = OpenCASCADE::read_IGES(cad_file_name);
      else if (extension == "step" || extension == "stp")
        shape = OpenCASCADE::read_STEP(cad_file_name);
      else
        AssertThrow(false,
                    ExcNotImplemented("We found an extension that we "
                                      "do not recognize as a CAD file "
                                      "extension. Bailing out."));
      const auto n_elements = OpenCASCADE::count_elements(shape);
      if ((std::get<0>(n_elements) == 0))
        tria.set_manifold(
          manifold_id,
          OpenCASCADE::ArclengthProjectionLineManifold<dim, spacedim>(shape));
      else if (spacedim == 3)
        {
          const auto t = reinterpret_cast<Triangulation<dim, 3> *>(&tria);
          t->set_manifold(manifold_id,
                          OpenCASCADE::NormalToMeshProjectionManifold<dim, 3>(
                            shape));
        }
      else
        tria.set_manifold(manifold_id,
                          OpenCASCADE::NURBSPatchManifold<dim, spacedim>(
                            TopoDS::Face(shape)));
    }
#else
  (void)ids_and_cad_file_names;
  AssertThrow(false, ExcNotImplemented("Generation of the grid failed."));
#endif
}



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
  DoFTools::extract_locally_relevant_dofs(dh, relevant_dofs[0]);

  FEFaceValues<spacedim> fe_face_values(*fe,
                                        *face_quadrature_formula,
                                        update_values | update_JxW_values |
                                          update_quadrature_points |
                                          update_normal_vectors);

  {
    constraints.reinit(relevant_dofs[0]);
    DoFTools::make_hanging_node_constraints(dh, constraints);
    for (const auto id : par.dirichlet_ids)
      {
        VectorTools::interpolate_boundary_values(dh, id, par.bc, constraints);
      }
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

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      pcout << "   Number of degrees of freedom: " << owned_dofs[0].size()
            << " + " << owned_dofs[1].size()
            << " (locally owned: " << owned_dofs[0].n_elements() << " + "
            << owned_dofs[1].n_elements() << ")" << std::endl;
    }
}

template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::assemble_elasticity_system()
{
  stiffness_matrix = 0;
  coupling_matrix  = 0;
  system_rhs       = 0;
  TimerOutput::Scope     t(computing_timer, "Assemble Stiffness and rhs");
  FEValues<spacedim>     fe_values(*fe,
                               *quadrature,
                               update_values | update_gradients |
                                 update_quadrature_points | update_JxW_values);
  FEFaceValues<spacedim> fe_face_values(*fe,
                                        *face_quadrature_formula,
                                        update_values | update_JxW_values |
                                          update_quadrature_points |
                                          update_normal_vectors);

  const unsigned int          dofs_per_cell = fe->n_dofs_per_cell();
  const unsigned int          n_q_points    = quadrature->size();
  FullMatrix<double>          cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>              cell_rhs(dofs_per_cell);
  std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(spacedim));
  std::vector<Tensor<2, spacedim>>     grad_phi_u(dofs_per_cell);
  std::vector<double>                  div_phi_u(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);
        par.rhs.vector_value_list(fe_values.get_quadrature_points(),
                                  rhs_values);
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                grad_phi_u[k] =
                  fe_values[displacement].symmetric_gradient(k, q);
                div_phi_u[k] = fe_values[displacement].divergence(k, q);
              }
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_matrix(i, j) +=
                      (2 * par.Lame_mu *
                         scalar_product(grad_phi_u[i], grad_phi_u[j]) +
                       par.Lame_lambda * div_phi_u[i] * div_phi_u[j]) *
                      fe_values.JxW(q);
                  }
                const auto comp_i = fe->system_to_component_index(i).first;
                cell_rhs(i) += fe_values.shape_value(i, q) *
                               rhs_values[q][comp_i] * fe_values.JxW(q);
              }
          }


        // Neumann boundary conditions
        // for (const auto &f : cell->face_iterators()) ////
        for (unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell;
             ++f)
          if (cell->face(f)->at_boundary())
            {
              // auto it = par.neumann_ids.find(cell->face(f)->boundary_id());
              // if (it != par.neumann_ids.end())
              if (std::find(par.neumann_ids.begin(),
                            par.neumann_ids.end(),
                            cell->face(f)->boundary_id()) !=
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
                          cell_rhs(i) += -neumann_value *
                                         fe_face_values.shape_value(i, q) *
                                         fe_face_values.JxW(q);
                        }
                    }
                }
            }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               stiffness_matrix,
                                               system_rhs.block(0));
      }
  stiffness_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
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
                      if (comp_i == inclusions.get_component(j))
                        {
                          local_coupling_matrix(i, j) +=
                            (fev.shape_value(i, q)) * inclusion_fe_values[j] /
                            inclusions.get_section_measure(inclusion_id) * ds;
                        }
                    }
                  if (inclusions.inclusions_data[inclusion_id].size() > 0)
                    {
                      if (inclusions.inclusions_data[inclusion_id].size() + 1 >
                          inclusions.get_fourier_component(j))
                        {
                          auto temp =
                            inclusion_fe_values[j] * ds /
                            inclusions.get_section_measure(inclusion_id) *
                            // phi_i ds
                            // now we need to build g from the data.
                            // this is sum E^i g_i where g_i are coefficients of
                            // the modes, but only the j one survives
                            inclusion_fe_values[j] *
                            inclusions.get_inclusion_data(inclusion_id, j);

                          if (par.initial_time != par.final_time)
                            temp *= inclusions.inclusions_rhs.value(
                              real_q, inclusions.get_component(j));
                          local_rhs(j) += temp;
                        }
                    }
                  else
                    {
                      local_rhs(j) +=
                        inclusion_fe_values[j] /
                        inclusions.get_section_measure(inclusion_id) *
                        inclusions.inclusions_rhs.value(
                          real_q, inclusions.get_component(j)) // /
                        // inclusions.get_radius(inclusion_id)
                        * ds;
                    }
                  local_inclusion_matrix(j, j) +=
                    (inclusion_fe_values[j] * inclusion_fe_values[j] * ds);
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
            local_rhs, inclusion_dof_indices, system_rhs.block(1));

          inclusion_constraints.distribute_local_to_global(
            local_inclusion_matrix, inclusion_dof_indices, inclusion_matrix);
        }
      particle = pic.end();
    }
  coupling_matrix.compress(VectorOperation::add);
  inclusion_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}



template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::solve()
{
  TimerOutput::Scope       t(computing_timer, "Solve");
  LA::MPI::PreconditionAMG prec_A;
  {
    // LA::MPI::PreconditionAMG::AdditionalData data;
    TrilinosWrappers::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#endif
    // informo il precondizionatore dei modi costanti del problema elastico
    std::vector<std::vector<bool>>   constant_modes;
    const FEValuesExtractors::Vector displacement_components(0); // gia in .h
    DoFTools::extract_constant_modes(
      dh, fe->component_mask(displacement_components), constant_modes);
    data.constant_modes = constant_modes;

    prec_A.initialize(stiffness_matrix, data);
  }

  const auto A    = linear_operator<LA::MPI::Vector>(stiffness_matrix);
  auto       invA = A;

  const auto amgA = linear_operator(A, prec_A);

  // for small radius you might need SolverFGMRES<LA::MPI::Vector>
  SolverCG<LA::MPI::Vector> cg_stiffness(par.inner_control);
  // SolverFGMRES<LA::MPI::Vector> cg_stiffness(par.inner_control);
  invA = inverse_operator(A, cg_stiffness, amgA);

  // Some aliases
  auto &u      = solution.block(0);
  auto &lambda = solution.block(1);

  const auto &f = system_rhs.block(0);
  auto       &g = system_rhs.block(1);

  if (inclusions.n_dofs() == 0)
    {
      u = invA * f;
    }
  else
    {
      const auto Bt = linear_operator<LA::MPI::Vector>(coupling_matrix);
      const auto B  = transpose_operator(Bt);
      const auto M  = linear_operator<LA::MPI::Vector>(inclusion_matrix);

      // auto interp_g = g;
      // interp_g      = 0.1;
      // g             = C * interp_g;

      // Schur complement
      const auto S = B * invA * Bt;

      // Schur complement preconditioner
      // VERSION 1
      // auto                          invS = S;
      // SolverFGMRES<LA::MPI::Vector> cg_schur(par.outer_control);
      SolverMinRes<LA::MPI::Vector> cg_schur(par.outer_control);
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

      // Compute Lambda first
      lambda = invS * (B * invA * f - g);
      pcout << "   Solved for lambda in " << par.outer_control.last_step()
            << " iterations." << std::endl;

      // Then compute u
      u = invA * (f - Bt * lambda);
      pcout << "   u norm: " << u.l2_norm()
            << ", lambda norm: " << lambda.l2_norm() << std::endl;
    }

  pcout << "   Solved for u in " << par.inner_control.last_step()
        << " iterations." << std::endl;
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
  // VectorTools::interpolate(dh, par.bc, exact_vec);
  VectorTools::interpolate(dh, par.exact_solution, exact_vec);
  auto exact_vec_locally_relevant(locally_relevant_solution.block(0));
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
  data_out.build_patches();
  const std::string filename =
    par.output_name + "_" + std::to_string(cycle) + ".vtu";
  data_out.write_vtu_in_parallel(par.output_directory + "/" + filename,
                                 mpi_communicator);
  return filename;
}

// template <int dim, int spacedim>
// std::string
// ElasticityProblem<dim, spacedim>::output_stresses() const
// {
//   std::vector<std::string> solution_names(spacedim*spacedim, "face_stress");

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
//   data_out_faces.write_vtu_in_parallel(par.output_directory + "/" + filename,
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
  static std::vector<std::pair<double, std::string>> cycles_and_particles;
  // static std::vector<std::pair<double, std::string>> cycles_and_stresses;

  if (cycles_and_solutions.size() == cycle)
    {
      cycles_and_solutions.push_back({(double)cycle, output_solution()});
      std::ofstream pvd_solutions(par.output_directory + "/" + par.output_name +
                                  ".pvd");
      DataOutBase::write_pvd_record(pvd_solutions, cycles_and_solutions);

      if (cycle == 0)
        {
          const std::string particles_filename =
            par.output_name + "_particles.vtu";

          inclusions.output_particles(par.output_directory + "/" +
                                      particles_filename);
          cycles_and_particles.push_back({(double)cycle, particles_filename});

          std::ofstream pvd_particles(par.output_directory + "/" +
                                      par.output_name + "_particles.pvd");
          DataOutBase::write_pvd_record(pvd_particles, cycles_and_particles);
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
  par.prm.print_parameters(par.output_directory + "/" + "used_parameters_" +
                             std::to_string(dim) + std::to_string(spacedim) +
                             ".prm",
                           ParameterHandler::Short);
}

/**
 * @brief check on the boundary id that no boundary conditions are in disagreement
 */
template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::check_boundary_ids()
{
  for (const auto id : par.dirichlet_ids)
    {
      for (const auto Nid : par.neumann_ids)
        if (id == Nid)
          AssertThrow(false,
                      ExcNotImplemented("incoherent boundary conditions."));
      for (const auto noid : par.normal_flux_ids)
        if (id == noid)
          AssertThrow(false,
                      ExcNotImplemented("incoherent boundary conditions."));
    }
}


/**
 * @brief compute stresses on boundaries (2D and 3D) and internal (2D)
 * this function makes use of boundary id, so make sure that the ifd starts from
 * 0 and are sequential when importing a mesh, this is automatically taken care
 * of for meshes generated with GridTools output is a txt file containing
 * stresses at each cycle/time
 */
template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::compute_internal_and_boundary_stress(
  bool openfilefirsttime) const
{
  TimerOutput::Scope t(
    computing_timer,
    "Postprocessing: Computing internal and boundary stresses");

  std::map<types::boundary_id, Tensor<1, spacedim>> boundary_stress;
  std::map<types::boundary_id, double>              u_dot_n;
  Tensor<2, spacedim>                               internal_stress;
  Tensor<1, spacedim>                               average_displacement;

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
  //   // std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
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
          //           //                par.Lame_lambda * div_phi_u * identity)
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
                      (2 * par.Lame_mu * displacement_gradient[q] +
                       par.Lame_lambda * displacement_divergence[q] *
                         identity) *
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

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      const std::string filename(par.output_directory + "/forces.txt");
      std::ofstream     forces_file;
      if (openfilefirsttime)
        {
          forces_file.open(filename);
          if constexpr (spacedim == 2)
            {
              forces_file << "cycle area";
              for (auto id : all_ids) // par.dirichlet_ids) // all_ids)
                forces_file << " perimeter" << id;
              forces_file
                << " meanInternalStressxx meanInternalStressxy meanInternalStressyx meanInternalStressyy avg_u_x avg_u_y";
              for (auto id : all_ids) // par.dirichlet_ids) // all_ids)
                forces_file << " boundaryStressX_" << id << " boundaryStressY_"
                            << id << " uDotN_" << id;
              forces_file << std::endl;
            }
          else
            {
              forces_file << "cycle";
              for (auto id : all_ids)
                forces_file << "perimeter" << id << " sigmanX_" << id
                            << " sigmanY_" << id << " sigmanZ_" << id
                            << " uDotN_" << id;
              forces_file << std::endl;
            }
        }
      else
        forces_file.open(filename, std::ios_base::app);

      if constexpr (spacedim == 2)
        {
          forces_file << cycle << " " << internal_area << " ";
          for (auto id : all_ids)
            forces_file << perimeter[id] << " ";
          forces_file << internal_stress << " " << average_displacement << " ";
          for (auto id : all_ids)
            forces_file << boundary_stress[id] << " " << u_dot_n[id] << " ";
          forces_file << std::endl;
        }
      else // spacedim = 3
        {
          forces_file << cycle << " ";
          for (auto id : all_ids)
            forces_file << perimeter[id] << " ";
          for (auto id : all_ids)
            forces_file << boundary_stress[id] << " " << u_dot_n[id] << " ";
          forces_file << std::endl;
        }
      forces_file.close();
    }

  return;
}

/**
 * @brief compute tissue pressure ($\Lambda$) over the vessels and output to a .txt file (sequential) or .h5 file (mpi)
 *
 * @param openfilefisrttime internal variable to open file
 */

template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::output_pressure(bool openfilefirsttime) const
{
  if (par.output_pressure == false)
    return;
  TimerOutput::Scope t(computing_timer, "Postprocessing: Output Pressure");

  if (inclusions.n_inclusions() > 0
      // &&
      // inclusions.get_offset_coefficients() == 1 &&
      // inclusions.n_coefficients >= 2
  )
    {
      const auto locally_owned_vessels =
        Utilities::MPI::create_evenly_distributed_partitioning(
          mpi_communicator, inclusions.get_n_vessels());
      const auto locally_owned_inclusions =
        Utilities::MPI::create_evenly_distributed_partitioning(
          mpi_communicator, inclusions.n_inclusions());

      TrilinosWrappers::MPI::Vector pressure(locally_owned_vessels,
                                             mpi_communicator);
      pressure = 0;
      TrilinosWrappers::MPI::Vector pressure_at_inc(locally_owned_inclusions,
                                                    mpi_communicator);
      pressure_at_inc = 0;

      auto &lambda_to_pressure = locally_relevant_solution.block(1);

      const auto used_number_modes = inclusions.get_n_coefficients();

      const auto local_lambda = lambda_to_pressure.locally_owned_elements();

      if constexpr (spacedim == 3)
        {
          for (const auto &element_of_local_lambda : local_lambda)
            {
              const unsigned inclusion_number = (unsigned int)floor(
                element_of_local_lambda / (used_number_modes));

              AssertIndexRange(inclusion_number, inclusions.n_inclusions());
              pressure[inclusions.get_vesselID(inclusion_number)] +=
                lambda_to_pressure[element_of_local_lambda];

              pressure_at_inc[inclusion_number] +=
                lambda_to_pressure[element_of_local_lambda];
            }
          pressure.compress(VectorOperation::add);
          pressure_at_inc.compress(VectorOperation::add);
        }
      else // spacedim = 2
        {
          for (auto element_of_local_lambda : local_lambda)
            {
              const unsigned inclusion_number = (unsigned int)floor(
                element_of_local_lambda / (used_number_modes));

              AssertIndexRange(inclusion_number, inclusions.n_inclusions());
              pressure_at_inc[inclusion_number] +=
                lambda_to_pressure[element_of_local_lambda];
            }
          pressure = pressure_at_inc;
          // pressure.compress(VectorOperation::add);
          // pressure_at_inc.compress(VectorOperation::add);
        }

      // print .txt only sequential
      if (Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
        {
          const std::string filename(par.output_directory +
                                     "/externalPressure.txt");
          std::ofstream     pressure_file;
          if (openfilefirsttime)
            {
              pressure_file.open(filename);
              pressure_file << "cycle ";
              for (unsigned int num = 0; num < pressure.size(); ++num)
                pressure_file << "vessel" << num << " ";
              pressure_file << std::endl;
            }
          else
            pressure_file.open(filename, std::ios_base::app);

          pressure_file << cycle << " ";
          pressure.print(pressure_file);
          pressure_file.close();
        }
      else
        // print .h5
        if (par.initial_time == par.final_time)
          {
            const std::string FILE_NAME(par.output_directory +
                                        "/externalPressure.h5");

            auto accessMode = HDF5::File::FileAccessMode::create;
            if (!openfilefirsttime)
              accessMode = HDF5::File::FileAccessMode::open;

            HDF5::File        file_h5(FILE_NAME, accessMode, mpi_communicator);
            const std::string DATASET_NAME("externalPressure_" +
                                           std::to_string(cycle));

            HDF5::DataSet dataset =
              file_h5.create_dataset<double>(DATASET_NAME,
                                             {inclusions.get_n_vessels()});

            std::vector<double> data_to_write;
            // std::vector<hsize_t> coordinates;
            data_to_write.reserve(pressure.locally_owned_size());
            // coordinates.reserve(pressure.locally_owned_size());
            for (const auto &el : locally_owned_vessels)
              {
                // coordinates.emplace_back(el);
                data_to_write.emplace_back(pressure[el]);
              }
            if (pressure.locally_owned_size() > 0)
              {
                hsize_t prefix = 0;
                hsize_t los    = pressure.locally_owned_size();
                int     ierr   = MPI_Exscan(&los,
                                      &prefix,
                                      1,
                                      MPI_UNSIGNED_LONG_LONG,
                                      MPI_SUM,
                                      mpi_communicator);
                AssertThrowMPI(ierr);

                std::vector<hsize_t> offset = {prefix, 1};
                std::vector<hsize_t> count = {pressure.locally_owned_size(), 1};
                // data.write_selection(data_to_write, coordinates);
                dataset.write_hyperslab(data_to_write, offset, count);
              }
            else
              dataset.write_none<int>();
          }
        else
          {
            pcout
              << "output_pressure file for time dependent simulation not implemented"
              << std::endl;
          }
    }
  else
    {
      pcout << "Inclusions number = 0, pressure file not created" << std::endl;
    }
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
  if (par.initial_time == par.final_time) // time stationary
    {
      print_parameters();
      make_grid();
      setup_fe();
      check_boundary_ids();
      {
        TimerOutput::Scope t(computing_timer, "Setup inclusion");
        inclusions.setup_inclusions_particles(tria);
      }
      // setup_dofs(); // called inside refine_and_transfer
      for (cycle = 0; cycle < par.n_refinement_cycles; ++cycle)
        {
          setup_dofs();
          if (par.output_results_before_solving)
            output_results();
          assemble_elasticity_system();

          assemble_coupling();
          solve();
          output_results();
          output_pressure(cycle == 0 ? true : false);
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

      // if (par.domain_type == "generate")
      {
        compute_internal_and_boundary_stress(true);
      }
      if (pcout.is_active())
        {
          par.convergence_table.output_table(pcout.get_stream());
        }
      if (false)
        output_lambda();
    }
  else // Time dependent simulation
    {
      // TODO: add refinement as the first cycle,
      pcout << "time dependent simulation, refinement not implemented"
            << std::endl;
      print_parameters();
      make_grid();
      setup_fe();
      check_boundary_ids();
      cycle = 0;
      {
        TimerOutput::Scope t(computing_timer, "Setup inclusion");
        inclusions.setup_inclusions_particles(tria);
      }
      setup_dofs();
      assemble_elasticity_system();
      for (current_time = par.initial_time; current_time < par.final_time;
           current_time += par.dt, ++cycle)
        {
          pcout << "Time: " << current_time << std::endl;
          // assemble_elasticity_system();
          inclusions.inclusions_rhs.set_time(current_time);
          par.Neumann_bc.set_time(current_time);
          assemble_coupling();
          solve();
          output_results();
          output_pressure(cycle == 0 ? true : false);

          if (par.domain_type == "generate")
            compute_internal_and_boundary_stress(cycle == 0 ? true : false);
        }
    }
}



// Template instantiations
template class ElasticityProblem<2>;
template class ElasticityProblem<2, 3>; // dim != spacedim
template class ElasticityProblem<3>;
