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
read_grid_and_cad_files(const std::string &           grid_file_name,
                        const std::string &           ids_and_cad_file_names,
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
      const auto  extension     = boost::algorithm::to_lower_copy(
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
      GridGenerator::generate_from_name_and_arguments(tria,
                                                      par.name_of_grid,
                                                      par.arguments_for_grid);
    }
  catch (...)
    {
      pcout << "Generating from name and argument failed." << std::endl
            << "Trying to read from file name." << std::endl;
      read_grid_and_cad_files(par.name_of_grid, par.arguments_for_grid, tria);
    }
  }
  else if (par.domain_type == "cylinder")
  {
    Assert (spacedim == 2, ExcInternalError());
      GridGenerator::hyper_ball(tria, Point<spacedim>(), 1.);
      std::cout << " ATTENTION: GRID: cirle of radius 1." << std::endl;
  }
  else if (par.domain_type == "cheese")
  {
    Assert (spacedim == 2, ExcInternalError());
    GridGenerator::cheese(tria, std::vector<unsigned int> (2, 2));
  }
  else if (par.domain_type == "file")
  {
    GridIn<spacedim> gi;
    gi.attach_triangulation(tria);
    //std::ifstream infile(par.name_of_grid);
    const std::string infile(par.name_of_grid);
    Assert(!infile.empty(), ExcIO());
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
  face_quadrature_formula = std::make_unique<QGauss<spacedim-1>>(par.fe_degree + 1);
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

  FEFaceValues<spacedim>      fe_face_values(*fe,
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
    std::map< types::boundary_id, const Function< spacedim, double > * > function_map;
    for (const auto id : par.normal_flux_ids)
    {
      function_map.insert(std::pair< types::boundary_id, const Function< spacedim, double > * > (id, & par.Neumann_bc));
    }
    VectorTools::compute_nonzero_normal_flux_constraints(dh, 0, par.normal_flux_ids, 
                                                          function_map, constraints);
    constraints.close();
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

      DynamicSparsityPattern dsp(dh.n_dofs(), inclusions.n_dofs());

      relevant_dofs[1] = assemble_coupling_sparsity(dsp);
      SparsityTools::distribute_sparsity_pattern(dsp,
                                                 owned_dofs[0],
                                                 mpi_communicator,
                                                 relevant_dofs[0]);
      coupling_matrix.clear();
      coupling_matrix.reinit(owned_dofs[0],
                             owned_dofs[1],
                             dsp,
                             mpi_communicator);

      DynamicSparsityPattern idsp(inclusions.n_dofs(), inclusions.n_dofs());
      for (const auto i : owned_dofs[1])
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

  pcout << "   Number of degrees of freedom: " << owned_dofs[0].size() << " + "
        << owned_dofs[1].size()
        << " (locally owned: " << owned_dofs[0].n_elements() << " + "
        << owned_dofs[1].n_elements() << ")" << std::endl;
}

template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::assemble_elasticity_system()
{
  stiffness_matrix = 0;
  coupling_matrix  = 0;
  system_rhs       = 0;
  TimerOutput::Scope          t(computing_timer, "Assemble Stiffness and Neumann rhs");
  FEValues<spacedim>          fe_values(*fe,
                               *quadrature,
                               update_values | update_gradients |
                               update_quadrature_points | update_JxW_values);
  FEFaceValues<spacedim>      fe_face_values(*fe,
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
                grad_phi_u[k] = fe_values[displacement].symmetric_gradient(k, q);
                div_phi_u[k]  = fe_values[displacement].divergence(k, q);
              }
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_matrix(i, j) +=
                      (2* par.Lame_mu * scalar_product(grad_phi_u[i], grad_phi_u[j]) +
                       par.Lame_lambda * div_phi_u[i] * div_phi_u[j]) *
                      fe_values.JxW(q);
                  }
                const auto comp_i = fe->system_to_component_index(i).first;
                cell_rhs(i) += fe_values.shape_value(i, q) *
                               rhs_values[q][comp_i] * fe_values.JxW(q);
              }
          }


        // Neumann boundary conditions
        //for (const auto &f : cell->face_iterators()) //// 
        for (unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell; ++f)
          if (cell->face(f)->at_boundary())
          {
            // auto it = par.neumann_ids.find(cell->face(f)->boundary_id());
            // if (it != par.neumann_ids.end())
            if (std::find(par.neumann_ids.begin(), par.neumann_ids.end(), cell->face(f)->boundary_id()) != par.neumann_ids.end())
              {
                fe_face_values.reinit(cell, f);
                for (unsigned int q = 0; q < fe_face_values.n_quadrature_points;++q)
                  {
                    double neumann_value = 0;
                    for (int d = 0; d < spacedim; ++d)
                      neumann_value += par.Neumann_bc.value(fe_face_values.quadrature_point(q), d)
                                      * fe_face_values.normal_vector(q)[d];
                    neumann_value /= spacedim;
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        cell_rhs(i) +=
                          -neumann_value *
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
  DynamicSparsityPattern &dsp) const
{
  TimerOutput::Scope t(computing_timer, "Assemble Coupling sparsity");
  IndexSet           relevant(inclusions.n_dofs());

  std::vector<types::global_dof_index> dof_indices(fe->n_dofs_per_cell());
  std::vector<types::global_dof_index> inclusion_dof_indices;

  if(! par.treat_as_hypersingular)
  {
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
                                                inclusion_dof_indices,
                                                dsp);
        relevant.add_indices(inclusion_dof_indices.begin(),
                             inclusion_dof_indices.end());
        particle = pic.end();
      }
  }
  else // if treat_as_hypersingular
  {

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

  FullMatrix<double> local_bulk_matrix(fe->n_dofs_per_cell(),
                                       fe->n_dofs_per_cell());

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
          local_bulk_matrix       = 0;
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
              const auto  ds                  = inclusions.get_JxW(id);// /inclusions.get_radius(inclusion_id);

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
                            (fev.shape_value(i, q)) * inclusion_fe_values[j] *
                            ds;

                        }
                    }
                  if (inclusions.data_file != "")
                    {
                      if (inclusions.inclusions_data[inclusion_id].size() > j)
                        {
                          auto temp =
                            inclusion_fe_values[j] * inclusion_fe_values[j] *
                            inclusions.inclusions_data[inclusion_id][j] 
                            /inclusions.get_radius(inclusion_id) * 
                            ds;
                          if (par.initial_time != par.final_time)
                            temp *= inclusions.inclusions_rhs.value(
                                        real_q, inclusions.get_component(j));
                          local_rhs(j) += temp;
                        }
                    }
                  else
                    {
                      local_rhs(j) += inclusion_fe_values[j] *
                                      inclusions.inclusions_rhs.value(
                                        real_q, inclusions.get_component(j)) 
                                        /inclusions.get_radius(inclusion_id) * 
                                      ds;
                    }
                  local_inclusion_matrix(j, j) +=
                    (inclusion_fe_values[j] * inclusion_fe_values[j] * ds);
                }
              ++p;
            }
          // I expect p and next_p to be the same now.
          Assert(p == next_p, ExcInternalError());
          // Add local matrices to global ones
          constraints.distribute_local_to_global(local_bulk_matrix,
                                                 fe_dof_indices,
                                                 stiffness_matrix);

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
  stiffness_matrix.compress(VectorOperation::add);
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
    DoFTools::extract_constant_modes(dh,
                                     fe->component_mask(
                                      displacement_components),
                                     constant_modes);
    data.constant_modes = constant_modes;

    prec_A.initialize(stiffness_matrix, data);
  }

  const auto A    = linear_operator<LA::MPI::Vector>(stiffness_matrix);
  auto       invA = A;

  const auto amgA = linear_operator(A, prec_A);

  // for small radius you might need SolverFGMRES<LA::MPI::Vector>
  SolverCG<LA::MPI::Vector> cg_stiffness(par.inner_control);
  invA = inverse_operator(A, cg_stiffness, amgA);

  // Some aliases
  auto &u      = solution.block(0);
  auto &lambda = solution.block(1);

  const auto &f = system_rhs.block(0);
  auto &      g = system_rhs.block(1);

  if (inclusions.n_dofs() == 0)
    {
      u = invA * f;
    }
  else
    {
      // std::cout << "matrix B" << std::endl;
      // coupling_matrix.print(std::cout);
      // std::cout << std::endl << " end matrix B" << std::endl;
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
      SolverFGMRES<LA::MPI::Vector> cg_schur(par.outer_control);
      // invS = inverse_operator(S, cg_schur);
      // VERSION2
      auto                          invS = S;
      auto                     S_inv_prec = B * invA * Bt + M;
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
      pcout << "   Solved for lambda in " << par.outer_control.last_step()
            << " iterations." << std::endl;

      // Then compute u
      u = invA * (f - Bt * lambda);
      pcout << "   u norm: " << u.l2_norm()
            << ", lambda norm: " << lambda.l2_norm() << std::endl;
      std::cout << "   lambda: ";
      lambda.print(std::cout);
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
  tria.execute_coarsening_and_refinement();
  inclusions.inclusions_as_particles.unpack_after_coarsening_and_refinement();
  setup_dofs();
  transfer.interpolate(solution.block(0));
  constraints.distribute(solution.block(0));
  locally_relevant_solution.block(0) = solution.block(0);
}



template <int dim, int spacedim>
std::string
ElasticityProblem<dim, spacedim>::output_solution() const
{
  TimerOutput::Scope       t(computing_timer, "Output results");
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
// ElasticityProblem<dim, spacedim>::output_particles() const
// {
//   Particles::DataOut<spacedim> particles_out;
//   particles_out.build_patches(inclusions.inclusions_as_particles);
//   const std::string filename =
//     par.output_name + "_particles_" + std::to_string(cycle) + ".vtu";
//   particles_out.write_vtu_in_parallel(par.output_directory + "/" + filename,
//                                       mpi_communicator);
//   return filename;
// }


template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::output_results() const
{
    static std::vector<std::pair<double, std::string>> cycles_and_solutions;
    static std::vector<std::pair<double, std::string>> cycles_and_particles;
  
    if (cycles_and_solutions.size() == cycle)
      {
        cycles_and_solutions.push_back({(double)cycle, output_solution()});
  
        const std::string particles_filename =
          par.output_name + "_particles_" + std::to_string(cycle) + ".vtu";
        inclusions.output_particles(par.output_directory + "/" +
                                    particles_filename);
  
        cycles_and_particles.push_back({(double)cycle, particles_filename});
  
        std::ofstream pvd_solutions(par.output_directory + "/" + par.output_name +
                                    ".pvd");
        std::ofstream pvd_particles(par.output_directory + "/" + par.output_name +
                                    "_particles.pvd");
        DataOutBase::write_pvd_record(pvd_solutions, cycles_and_solutions);
        DataOutBase::write_pvd_record(pvd_particles, cycles_and_particles);
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

template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::check_boundary_ids()
{
  std::vector< types::boundary_id > temp = tria.get_boundary_ids();
  std::cout << "all boundary : ";
  for (auto i : temp)
    std::cout << i << ", ";
  std::cout << std::endl;
  std::cout << "Dir boundary : ";
  for (const auto id : par.dirichlet_ids)
    std::cout << id << ", ";
  std::cout << std::endl;
  std::cout << "Neu boundary : ";
  for (const auto Nid : par.neumann_ids)
    std::cout << Nid << ", ";
  std::cout << std::endl;
  std::cout << "flux boundary : ";
  for (const auto noid : par.normal_flux_ids)
    std::cout << noid << ", ";
  std::cout << std::endl;
  for (const auto id : par.dirichlet_ids)
      for (const auto Nid : par.neumann_ids)
          if (id == Nid)
            AssertThrow(false, ExcNotImplemented("incoherent boundary conditions."));
}

template <int dim, int spacedim>
void
ElasticityProblem<dim, spacedim>::compute_boundary_stress() const
{
  TimerOutput::Scope       t(computing_timer, "computing stresses");
  Tensor<1, spacedim> b_stress;
  Tensor<2, spacedim> i_stress;
  double u_dot_n;
  double i_area = 0.;
  double perimeter = 0.;

  FEValues<spacedim>          fe_values(*fe,
                               *quadrature,
                               update_values | update_gradients |
                               update_quadrature_points | update_JxW_values);
  FEFaceValues<spacedim>      fe_face_values(*fe,
                               *face_quadrature_formula,
                               update_values | update_gradients | update_JxW_values |
                               update_quadrature_points |
                               update_normal_vectors);

  const FEValuesExtractors::Vector displacement(0);

  const unsigned int          dofs_per_cell = fe->n_dofs_per_cell();
  const unsigned int          n_q_points    = quadrature->size();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  Tensor<2, spacedim>         grad_phi_u;
  double                      div_phi_u;
  Tensor<2, spacedim>         identity;
  for (unsigned int ix = 0; ix < spacedim; ++ix)
    identity[ix][ix] = 1;

  // std::vector<std::vector<Tensor<1,spacedim>>> solution_gradient(face_quadrature_formula->size(), std::vector<Tensor<1,spacedim> >(spacedim+1));
  std::vector<Tensor<2,spacedim>> displacement_gradient(face_quadrature_formula->size());
  std::vector<Vector<double>> displacement_values(face_quadrature_formula->size());

  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell->get_dof_indices(local_dof_indices);
        fe_values.reinit(cell);
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            i_area += fe_values.JxW(q);
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                grad_phi_u = fe_values[displacement].symmetric_gradient(k, q);
                div_phi_u  = fe_values[displacement].divergence(k, q);
                i_stress += (2* par.Lame_mu * grad_phi_u + par.Lame_lambda * div_phi_u * identity) 
                            * solution.block(0)[local_dof_indices[k]]*fe_values.JxW(q);
              }
          }
        // /*
        for (unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell; ++f)
        // for (const auto &f : cell->face_iterators())
        // for (const auto f : GeometryInfo<spacedim>::face_indices())
          if (cell->face(f)->at_boundary())
          {
            fe_face_values.reinit(cell, f);
            // fe_face_values.get_function_gradients(solution, solution_gradient);
            fe_face_values[displacement].get_function_gradients(solution, displacement_gradient);
            fe_face_values.get_function_values(solution, displacement_values);
            for (unsigned int q = 0; q < fe_face_values.n_quadrature_points;++q)
            {
              perimeter += fe_face_values.JxW(q);
              std::cout << "871" << std::endl;
              // for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  u_dot_n += //solution.block(0)[local_dof_indices[i]] // 
                            (displacement_values[q][0] * fe_face_values.normal_vector(q)[0]
                            + displacement_values[q][1] * fe_face_values.normal_vector(q)[1])
                              * fe_face_values.JxW(q);
              const Tensor<1,spacedim> disp_grad_x = displacement_gradient[q][0];
              const Tensor<1,spacedim> disp_grad_y = displacement_gradient[q][1];
              double div = disp_grad_x[0] + disp_grad_y[1];
              // std::cout << div << std::endl;
              // b_stress[0] += (2* par.Lame_mu * disp_grad_x[0] + par.Lame_lambda * div)
              //             * fe_face_values.JxW(q)* fe_face_values.normal_vector(q)[0]
              //             + (2* par.Lame_mu * disp_grad_x[1])
              //             * fe_face_values.JxW(q)* fe_face_values.normal_vector(q)[1];
              // b_stress[1] += (2* par.Lame_mu * disp_grad_y[0])
              //             * fe_face_values.JxW(q)* fe_face_values.normal_vector(q)[0]
              //             + (2* par.Lame_mu * disp_grad_y[1] + par.Lame_lambda * div)
              //             * fe_face_values.JxW(q)* fe_face_values.normal_vector(q)[1]; 
              b_stress += (2* par.Lame_mu * displacement_gradient[q] + par.Lame_lambda * div * identity) 
                            * fe_face_values.JxW(q)* fe_face_values.normal_vector(q);
              u_dot_n += std::abs((displacement_values[q][0]* fe_face_values.normal_vector(q)[0]
                                  + displacement_values[q][1]* fe_face_values.normal_vector(q)[1])* fe_face_values.JxW(q));
            }
          }
        // */
      }
  i_stress = Utilities::MPI::sum(i_stress, mpi_communicator);
  b_stress = Utilities::MPI::sum(b_stress, mpi_communicator);
  i_area = Utilities::MPI::sum(i_area, mpi_communicator);
  perimeter = Utilities::MPI::sum(perimeter, mpi_communicator);
  i_stress/= i_area;
  //b_stress /= perimeter;
  pcout << "perimeter: " << perimeter << ", Boundary stress: " << b_stress << std::endl;
  pcout << "area: " << i_area << ", Mean internal stress: " << i_stress << std::endl;
  // std::cout << "u dot n" << u_dot_n << std::endl;
  // pcout << "Mean internal solution: " << u_avg << std::endl;
  // i_stress.print(std::cout);
}

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
    inclusions.setup_inclusions_particles(tria);
    for (cycle = 0; cycle < par.n_refinement_cycles; ++cycle)
      {
        setup_dofs();
        if (par.output_results_before_solving)
          output_results();
        assemble_elasticity_system();
        assemble_coupling();
        solve();
        output_results();
        if (spacedim == 2)
        {
          FunctionParser<spacedim> weight(par.weight_expression);
          par.convergence_table.error_from_exact(dh, solution.block(0), par.exact_solution, &weight);
        }
        else
          par.convergence_table.error_from_exact(dh, solution.block(0), par.bc);
        //compute_boundary_stress();
        if (cycle != par.n_refinement_cycles - 1)
          refine_and_transfer();
        if (pcout.is_active())
          par.convergence_table.output_table(pcout.get_stream());
      }
    }
  else // Time dependent simulation
    {
      pcout << "time dependent simulation, refinement is not possible" << std::endl;
      print_parameters();
      make_grid();
      setup_fe();
      check_boundary_ids();
      cycle = 0;
      inclusions.setup_inclusions_particles(tria);
      for (current_time = par.initial_time; current_time < par.final_time; current_time += par.dt, ++cycle)
        {
          pcout << "Time: " << current_time << std::endl;
          setup_dofs();
          // if (par.output_results_before_solving)
          //   output_results();
          assemble_elasticity_system();
          inclusions.inclusions_rhs.set_time(current_time);
          assemble_coupling();
          solve();
          output_results();
        }
    }
}


// Template instantiations
template class ElasticityProblem<2>;
template class ElasticityProblem<2, 3>; // dim != spacedim
template class ElasticityProblem<3>;