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



#include "laplacian.h"

template <int dim, int spacedim>
PoissonProblem<dim, spacedim>::PoissonProblem(
  const ProblemParameters<dim, spacedim> &par)
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
  , dh(tria)
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
PoissonProblem<dim, spacedim>::make_grid()
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
  tria.refine_global(par.initial_refinement);
}



template <int dim, int spacedim>
void
PoissonProblem<dim, spacedim>::setup_inclusions_particles()
{
  inclusions_as_particles.clear();
  inclusions_as_particles.initialize(tria, StaticMappingQ1<spacedim>::mapping);

  if (par.inclusions.empty())
    return;

  std::vector<Point<spacedim>> particles_positions;
  particles_positions.reserve(inclusion->n_q_points * par.inclusions.size());
  for (unsigned int i = 0; i < par.inclusions.size(); ++i)
    {
      const auto &p = inclusion->get_current_support_points(par.inclusions[i]);
      particles_positions.insert(particles_positions.end(), p.begin(), p.end());
    }

  std::vector<BoundingBox<spacedim>> all_boxes;
  all_boxes.reserve(tria.n_locally_owned_active_cells());
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      all_boxes.emplace_back(cell->bounding_box());
  const auto tree = pack_rtree(all_boxes);
  const auto local_boxes =
    extract_rtree_level(tree, par.rtree_extraction_level);

  global_bounding_boxes =
    Utilities::MPI::all_gather(mpi_communicator, local_boxes);

  Assert(!global_bounding_boxes.empty(),
         ExcInternalError(
           "I was expecting the "
           "global_bounding_boxes to be filled at this stage. "
           "Make sure you fill this vector before trying to use it "
           "here. Bailing out."));
  inclusions_as_particles.insert_global_particles(particles_positions,
                                                  global_bounding_boxes);
  tria.signals.pre_distributed_refinement.connect(
    [&]() { inclusions_as_particles.register_store_callback_function(); });
  tria.signals.post_distributed_refinement.connect(
    [&]() { inclusions_as_particles.register_load_callback_function(false); });
  pcout << "Inclusions particles: "
        << inclusions_as_particles.n_global_particles() << std::endl;
}



template <int dim, int spacedim>
void
PoissonProblem<dim, spacedim>::setup_fe()
{
  TimerOutput::Scope t(computing_timer, "Initial setup");
  fe = std::make_unique<FESystem<spacedim>>(FE_Q<spacedim>(par.fe_degree), 1);
  quadrature        = std::make_unique<QGauss<spacedim>>(par.fe_degree + 1);
  const auto factor = std::pow(2, cycle);
  inclusion         = std::make_unique<ReferenceInclusion<spacedim>>(
    par.inclusions_refinement * factor, par.n_fourier_coefficients);
}


template <int dim, int spacedim>
types::global_dof_index
PoissonProblem<dim, spacedim>::n_inclusions_dofs() const
{
  if (!par.inclusions.empty())
    return par.inclusions.size() * par.n_fourier_coefficients;
  else
    return 0;
}



template <int dim, int spacedim>
void
PoissonProblem<dim, spacedim>::setup_dofs()
{
  TimerOutput::Scope t(computing_timer, "Setup dofs");
  dh.distribute_dofs(*fe);

  owned_dofs.resize(2);
  owned_dofs[0] = dh.locally_owned_dofs();
  relevant_dofs.resize(2);
  DoFTools::extract_locally_relevant_dofs(dh, relevant_dofs[0]);
  {
    constraints.reinit(relevant_dofs[0]);
    DoFTools::make_hanging_node_constraints(dh, constraints);
    for (const auto id : par.dirichlet_ids)
      VectorTools::interpolate_boundary_values(dh, id, par.bc, constraints);
    constraints.close();
  }
  {
    stiffness_matrix.clear();
    DynamicSparsityPattern dsp(relevant_dofs[0]);
    DoFTools::make_sparsity_pattern(dh, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               owned_dofs[0],
                                               mpi_communicator,
                                               relevant_dofs[0]);
    stiffness_matrix.reinit(owned_dofs[0],
                            owned_dofs[0],
                            dsp,
                            mpi_communicator);
  }
  {
    auto inclusions_set =
      Utilities::MPI::create_evenly_distributed_partitioning(
        mpi_communicator, par.inclusions.size());

    owned_dofs[1] = inclusions_set.tensor_product(
      complete_index_set(par.n_fourier_coefficients));

    coupling_matrix.clear();
    DynamicSparsityPattern dsp(dh.n_dofs(), n_inclusions_dofs());

    relevant_dofs[1] = assemble_coupling_sparsity(dsp);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               owned_dofs[0],
                                               mpi_communicator,
                                               relevant_dofs[0]);
    coupling_matrix.reinit(owned_dofs[0], owned_dofs[1], dsp, mpi_communicator);
    inclusion_constraints.close();
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
PoissonProblem<dim, spacedim>::assemble_poisson_system()
{
  stiffness_matrix = 0;
  coupling_matrix  = 0;
  system_rhs       = 0;
  TimerOutput::Scope               t(computing_timer, "Assemble Stiffness");
  FEValues<spacedim>               fe_values(*fe,
                               *quadrature,
                               update_values | update_gradients |
                                 update_quadrature_points | update_JxW_values);
  const unsigned int               dofs_per_cell = fe->n_dofs_per_cell();
  const unsigned int               n_q_points    = quadrature->size();
  FullMatrix<double>               cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>                   cell_rhs(dofs_per_cell);
  std::vector<double>              rhs_values(n_q_points);
  std::vector<Tensor<1, spacedim>> grad_phi_u(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const FEValuesExtractors::Scalar     scalar(0);
  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);
        par.rhs.value_list(fe_values.get_quadrature_points(), rhs_values);
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              grad_phi_u[k] = fe_values[scalar].gradient(k, q);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_matrix(i, j) +=
                      grad_phi_u[i] * grad_phi_u[j] * fe_values.JxW(q);
                  }
                cell_rhs(i) += fe_values.shape_value(i, q) * rhs_values[q] *
                               fe_values.JxW(q);
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
PoissonProblem<dim, spacedim>::assemble_coupling_sparsity(
  DynamicSparsityPattern &dsp) const
{
  TimerOutput::Scope t(computing_timer, "Assemble Coupling sparsity");
  IndexSet           relevant(n_inclusions_dofs());

  const FEValuesExtractors::Scalar scalar(0);

  std::vector<types::global_dof_index> dof_indices(fe->n_dofs_per_cell());
  std::vector<types::global_dof_index> inclusion_dof_indices;

  auto particle = inclusions_as_particles.begin();
  while (particle != inclusions_as_particles.end())
    {
      const auto &cell = particle->get_surrounding_cell(tria);
      const auto &dh_cell =
        typename DoFHandler<spacedim>::cell_iterator(*cell, &dh);
      dh_cell->get_dof_indices(dof_indices);

      const auto pic = inclusions_as_particles.particles_in_cell(cell);
      Assert(pic.begin() == particle, ExcInternalError());
      std::set<types::global_dof_index> inclusion_dof_indices_set;
      for (const auto &p : pic)
        {
          const auto ids = inclusion->get_dof_indices(p.get_id());
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
  return relevant;
}



template <int dim, int spacedim>
void
PoissonProblem<dim, spacedim>::assemble_coupling()
{
  TimerOutput::Scope t(computing_timer, "Assemble Coupling matrix");
  const FEValuesExtractors::Scalar     scalar(0);
  std::vector<types::global_dof_index> fe_dof_indices(fe->n_dofs_per_cell());
  std::vector<types::global_dof_index> inclusion_dof_indices(
    inclusion->n_coefficients);

  FullMatrix<double> local_matrix(fe->n_dofs_per_cell(),
                                  inclusion->n_coefficients);
  Vector<double>     local_rhs(inclusion->n_coefficients);

  auto particle = inclusions_as_particles.begin();
  while (particle != inclusions_as_particles.end())
    {
      const auto &cell = particle->get_surrounding_cell(tria);
      const auto &dh_cell =
        typename DoFHandler<spacedim>::cell_iterator(*cell, &dh);
      dh_cell->get_dof_indices(fe_dof_indices);
      const auto pic = inclusions_as_particles.particles_in_cell(cell);
      Assert(pic.begin() == particle, ExcInternalError());

      auto p      = pic.begin();
      auto next_p = pic.begin();
      while (p != pic.end())
        {
          const auto inclusion_id = inclusion->get_inclusion_id(p->get_id());
          inclusion_dof_indices   = inclusion->get_dof_indices(p->get_id());
          local_matrix            = 0;
          local_rhs               = 0;
          const auto alpha1       = par.inclusions[inclusion_id][spacedim + 1];
          const auto alpha2       = par.inclusions[inclusion_id][spacedim + 2];

          std::vector<Point<spacedim>> ref_q_points;
          for (; next_p != pic.end() &&
                 inclusion->get_inclusion_id(next_p->get_id()) == inclusion_id;
               ++next_p)
            ref_q_points.push_back(next_p->get_reference_location());
          FEValues<spacedim, spacedim> fev(*fe,
                                           ref_q_points,
                                           update_values | update_gradients);
          fev.reinit(dh_cell);
          for (unsigned int q = 0; q < ref_q_points.size(); ++q)
            {
              const auto  id = p->get_id();
              const auto &inclusion_fe_values =
                inclusion->reinit(id, par.inclusions);
              const auto &real_q = p->get_location();

              for (unsigned int j = 0; j < inclusion->n_coefficients; ++j)
                {
                  for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
                    local_matrix(i, j) += (alpha1 * fev.shape_value(i, q) +
                                           alpha2 * fev.shape_grad(i, q) *
                                             inclusion->get_normal(id)) *
                                          inclusion_fe_values[j];
                  local_rhs(j) +=
                    inclusion_fe_values[j] * par.inclusions_rhs.value(real_q);
                }
              ++p;
            }
          // I expect p and next_p to be the same now.
          Assert(p == next_p, ExcInternalError());
          constraints.distribute_local_to_global(local_matrix,
                                                 fe_dof_indices,
                                                 inclusion_constraints,
                                                 inclusion_dof_indices,
                                                 coupling_matrix);
          inclusion_constraints.distribute_local_to_global(
            local_rhs, inclusion_dof_indices, system_rhs.block(1));
        }
      particle = pic.end();
    }
  coupling_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}



template <int dim, int spacedim>
void
PoissonProblem<dim, spacedim>::solve()
{
  TimerOutput::Scope       t(computing_timer, "Solve");
  LA::MPI::PreconditionAMG prec_A;
  {
    LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#endif
    prec_A.initialize(stiffness_matrix, data);
  }

  const auto A    = linear_operator<LA::MPI::Vector>(stiffness_matrix);
  const auto amgA = linear_operator(A, prec_A);

  // LA::SolverCG cg_stiffness(par.inner_control);
  SolverCG<LA::MPI::Vector> cg_stiffness(par.inner_control);
  const auto                invA = inverse_operator(A, cg_stiffness, amgA);

  // Some aliases
  auto &u      = solution.block(0);
  auto &lambda = solution.block(1);

  const auto &f = system_rhs.block(0);
  const auto &g = system_rhs.block(1);

  if (par.inclusions.empty())
    {
      u = invA * f;
    }
  else
    {
      const auto Bt = linear_operator<LA::MPI::Vector>(coupling_matrix);
      const auto B  = transpose_operator(Bt);


      // Schur complement
      const auto S = B * invA * Bt;

      LA::SolverCG cg_schur(par.outer_control);
      // LA::SolverGMRES              gmres_schur(solver_control);
      SolverGMRES<LA::MPI::Vector> gmres_schur(par.outer_control);
      const auto                   invS = inverse_operator(S, gmres_schur);


      pcout << "   f norm: " << f.l2_norm() << ", g norm: " << g.l2_norm()
            << std::endl;

      // Compute Lambda first
      lambda = invS * (B * invA * f - g);
      pcout << "   Solved for lambda in " << par.outer_control.last_step()
            << " iterations." << std::endl;

      // Then compute u
      u = invA * (f - Bt * lambda);
    }

  pcout << "   Solved for u in " << par.inner_control.last_step()
        << " iterations." << std::endl;
  constraints.distribute(u);
  inclusion_constraints.distribute(lambda);
  locally_relevant_solution = solution;
}



template <int dim, int spacedim>
void
PoissonProblem<dim, spacedim>::refine_and_transfer()
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
    {
      for (const auto &cell : tria.active_cell_iterators())
        cell->set_refine_flag();
    }
  // for (const auto &cell : tria.active_cell_iterators())
  //   {
  //     if (cell->refine_flag_set() && cell->level() ==
  //     par.max_level_refinement)
  //       cell->clear_refine_flag();
  //     if (cell->coarsen_flag_set() && cell->level() ==
  //     par.min_level_refinement)
  //       cell->clear_coarsen_flag();
  //   }
  parallel::distributed::SolutionTransfer<spacedim, LA::MPI::Vector> transfer(
    dh);
  tria.prepare_coarsening_and_refinement();
  transfer.prepare_for_coarsening_and_refinement(
    locally_relevant_solution.block(0));
  tria.execute_coarsening_and_refinement();
  setup_dofs();
  transfer.interpolate(solution.block(0));
  constraints.distribute(solution.block(0));
  locally_relevant_solution.block(0) = solution.block(0);
}



template <int dim, int spacedim>
std::string
PoissonProblem<dim, spacedim>::output_solution() const
{
  TimerOutput::Scope t(computing_timer, "Output results");
  std::string        solution_name = "solution";
  DataOut<spacedim>  data_out;
  data_out.attach_dof_handler(dh);
  data_out.add_data_vector(locally_relevant_solution.block(0), solution_name);
  Vector<float> subdomain(tria.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");
  data_out.build_patches();
  const std::string filename = "solution_" + std::to_string(cycle) + ".vtu";
  data_out.write_vtu_in_parallel(par.output_directory + "/" + filename,
                                 mpi_communicator);
  return filename;
}



template <int dim, int spacedim>
std::string
PoissonProblem<dim, spacedim>::output_particles() const
{
  Particles::DataOut<spacedim> particles_out;
  particles_out.build_patches(inclusions_as_particles);
  const std::string filename = "particles_" + std::to_string(cycle) + ".vtu";
  particles_out.write_vtu_in_parallel(par.output_directory + "/" + filename,
                                      mpi_communicator);
  return filename;
}


template <int dim, int spacedim>
void
PoissonProblem<dim, spacedim>::output_results() const
{
  static std::vector<std::pair<double, std::string>> cycles_and_solutions;
  static std::vector<std::pair<double, std::string>> cycles_and_particles;

  if (cycles_and_solutions.size() == cycle)
    {
      cycles_and_solutions.push_back({(double)cycle, output_solution()});
      cycles_and_particles.push_back({(double)cycle, output_particles()});

      std::ofstream pvd_solutions(par.output_directory + "/solutions.pvd");
      std::ofstream pvd_particles(par.output_directory + "/particles.pvd");
      DataOutBase::write_pvd_record(pvd_solutions, cycles_and_solutions);
      DataOutBase::write_pvd_record(pvd_particles, cycles_and_particles);
    }
}

template <int dim, int spacedim>
void
PoissonProblem<dim, spacedim>::print_parameters() const
{
#ifdef USE_PETSC_LA
  pcout << "Running PoissonProblem<" << Utilities::dim_string(dim, spacedim)
        << "> using PETSc." << std::endl;
#else
  pcout << "Running PoissonProblem<" << Utilities::dim_string(dim, spacedim)
        << "> using Trilinos." << std::endl;
#endif
  par.prm.print_parameters(par.output_directory + "/" + "used_parameters_" +
                             std::to_string(dim) + std::to_string(spacedim) +
                             ".prm",
                           ParameterHandler::Short);
}

template <int dim, int spacedim>
void
PoissonProblem<dim, spacedim>::run()
{
  print_parameters();
  make_grid();
  setup_fe();
  setup_inclusions_particles();
  for (cycle = 0; cycle < par.n_refinement_cycles; ++cycle)
    {
      setup_dofs();
      if (par.output_results_before_solving)
        output_results();
      assemble_poisson_system();
      assemble_coupling();
      solve();
      output_results();
      if (cycle != par.n_refinement_cycles - 1)
        refine_and_transfer();
    }
}


template class PoissonProblem<2>;
template class PoissonProblem<2, 3>;
template class PoissonProblem<3>;