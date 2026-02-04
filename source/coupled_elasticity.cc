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

#include "coupled_elasticity.h"

#include <deal.II/grid/grid_tools.h>


template <int dim, int spacedim>
CoupledElasticityProblemParameters<dim, spacedim>::
  CoupledElasticityProblemParameters()
  : ParameterAcceptor("/Immersed Problem/")
  , rhs("/Immersed Problem/Right hand side", spacedim)
  , exact_solution("/Immersed Problem/Exact solution", spacedim)
  , bc("/Immersed Problem/Dirichlet boundary conditions", spacedim)
  , Neumann_bc("/Immersed Problem/Neumann boundary conditions", spacedim)
  , inner_control("/Immersed Problem/Solver/Inner control")
  , outer_control("/Immersed Problem/Solver/Outer control")
  , convergence_table(std::vector<std::string>(spacedim, "u"))
{
  add_parameter("FE degree", fe_degree, "", this->prm, Patterns::Integer(1));
  add_parameter("Output directory", output_directory);
  add_parameter("Output name", output_name);
  add_parameter("Output results", output_results);
  add_parameter("Initial refinement", initial_refinement);
  add_parameter("Dirichlet boundary ids", dirichlet_ids);
  add_parameter("Neumann boundary ids", neumann_ids);
  add_parameter("Normal flux boundary ids", normal_flux_ids);
  enter_subsection("Grid generation");
  {
    add_parameter("Domain type", domain_type);
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
  enter_subsection("Physical constants");
  {
    add_parameter("Lame mu", Lame_mu);
    add_parameter("Lame lambda", Lame_lambda);
  }
  leave_subsection();
  enter_subsection("Exact solution");
  {
    add_parameter("Weight expression", weight_expression);
  }
  leave_subsection();
  enter_subsection("Time dependency");
  {
    add_parameter("Initial time", initial_time);
    add_parameter("Final time", final_time);
    add_parameter("Time step", dt);
  }
  leave_subsection();

  this->prm.enter_subsection("Error");
  convergence_table.add_parameters(this->prm);
  this->prm.leave_subsection();
}


template <int dim, int spacedim>
CoupledElasticityProblem<dim, spacedim>::CoupledElasticityProblem(
  const CoupledElasticityProblemParameters<dim, spacedim> &par)
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
CoupledElasticityProblem<dim, spacedim>::make_grid()
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
      // std::ifstream infile(par.name_of_grid);
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
CoupledElasticityProblem<dim, spacedim>::setup_fe()
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
CoupledElasticityProblem<dim, spacedim>::setup_dofs()
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
CoupledElasticityProblem<dim, spacedim>::assemble_elasticity_system()
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
CoupledElasticityProblem<dim, spacedim>::assemble_coupling_sparsity(
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
CoupledElasticityProblem<dim, spacedim>::assemble_coupling()
{
  TimerOutput::Scope t(computing_timer, "Assemble Coupling matrix");

  // system_rhs.block(1) = 0;

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
              const auto  ds =
                inclusions.get_JxW(id); // /inclusions.get_radius(inclusion_id);

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
                            inclusions.get_rotated_inclusion_data(
                              inclusion_id)[j] /
                            // inclusions.inclusions_data[inclusion_id][j] / //
                            // data is always prescribed in relative coordinates
                            inclusions.get_radius(inclusion_id) * ds;
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
                                        real_q, inclusions.get_component(j)) /
                                      inclusions.get_radius(inclusion_id) * ds;
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

          // inclusion_constraints.distribute_local_to_global(
          //   local_inclusion_matrix, inclusion_dof_indices, inclusion_matrix);
        }
      particle = pic.end();
    }
  coupling_matrix.compress(VectorOperation::add);
  inclusion_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}


template <int dim, int spacedim>
void
CoupledElasticityProblem<dim, spacedim>::reassemble_coupling_rhs()
{
  TimerOutput::Scope t(computing_timer, "updating coupling rhs");

  system_rhs.block(1) = 0;

  std::vector<types::global_dof_index> fe_dof_indices(fe->n_dofs_per_cell());
  std::vector<types::global_dof_index> inclusion_dof_indices(
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
              const auto  ds =
                inclusions.get_JxW(id); // /inclusions.get_radius(inclusion_id);

              // Coupling and inclusions matrix
              for (unsigned int j = 0; j < inclusions.n_dofs_per_inclusion();
                   ++j)
                {
                  if (inclusions.data_file != "")
                    {
                      if (inclusions.inclusions_data[inclusion_id].size() > j)
                        {
                          auto temp =
                            inclusion_fe_values[j] * inclusion_fe_values[j] *
                            inclusions.get_rotated_inclusion_data(
                              inclusion_id)[j] /
                            // inclusions.inclusions_data[inclusion_id][j] / //
                            // data is always prescribed in relative coordinates
                            inclusions.get_radius(inclusion_id) * ds;
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
                                        real_q, inclusions.get_component(j)) /
                                      inclusions.get_radius(inclusion_id) * ds;
                    }
                }
              ++p;
            }
          // I expect p and next_p to be the same now.
          Assert(p == next_p, ExcInternalError());
          // Add local matrices to global ones
          inclusion_constraints.distribute_local_to_global(
            local_rhs, inclusion_dof_indices, system_rhs.block(1));
        }
      particle = pic.end();
    }
  system_rhs.compress(VectorOperation::add);
}



template <int dim, int spacedim>
void
CoupledElasticityProblem<dim, spacedim>::solve()
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

  // for small radius you might need
  // SolverFGMRES<LA::MPI::Vector>
  SolverCG<LA::MPI::Vector> cg_stiffness(par.inner_control);
  invA = inverse_operator(A, cg_stiffness, amgA);

  // Some aliases
  auto &u      = solution.block(0);
  auto &lambda = solution.block(1);

  const auto &f = system_rhs.block(0);
  auto       &g = system_rhs.block(1);
  //   MPI_Barrier(mpi_communicator);
  //   g.print(std::cout);
  // MPI_Barrier(mpi_communicator);
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
      // SolverFGMRES<LA::MPI::Vector> cg_schur(par.outer_control);
      SolverMinRes<LA::MPI::Vector> cg_schur(par.outer_control);
      // invS = inverse_operator(S, cg_schur);
      // VERSION2
      auto invS       = S;
      auto S_inv_prec = B * invA * Bt + M;
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
      // std::cout << "   lambda: ";
      // lambda.print(std::cout);
    }

  pcout << "   Solved for u in " << par.inner_control.last_step()
        << " iterations." << std::endl;
  constraints.distribute(u);
  inclusion_constraints.distribute(lambda);
  locally_relevant_solution = solution;
}

template <int dim, int spacedim>
void
CoupledElasticityProblem<dim, spacedim>::refine_and_transfer_around_inclusions()
{
  TimerOutput::Scope t(computing_timer, "Refine");
  Vector<float>      error_per_cell(tria.n_active_cells());
  KellyErrorEstimator<spacedim>::estimate(dh,
                                          QGauss<spacedim - 1>(par.fe_degree +
                                                               1),
                                          {},
                                          locally_relevant_solution.block(0),
                                          error_per_cell);

  const int material_id = 1;

  auto particle = inclusions.inclusions_as_particles.begin();
  while (particle != inclusions.inclusions_as_particles.end())
    {
      const auto &cell = particle->get_surrounding_cell();
      const auto  pic =
        inclusions.inclusions_as_particles.particles_in_cell(cell);
      Assert(pic.begin() == particle, ExcInternalError());

      cell->set_refine_flag();
      cell->set_material_id(material_id);

      // for (auto f : cell->face_indices())
      // for (unsigned int face_no = 0; face_no
      // <GeometryInfo<spacedim>::faces_per_cell; ++face_no)
      for (auto vertex : cell->vertex_indices())
        {
          for (const auto &neigh_i :
               GridTools::find_cells_adjacent_to_vertex(dh, vertex))
            // for (auto neigh_i = cell->neighbor(face_no)->begin_face();
            // neigh_i < cell->neighbor(face_no)->end_face(); ++neigh_i)
            {
              if (!neigh_i->refine_flag_set())
                {
                  neigh_i->set_refine_flag();
                  neigh_i->set_material_id(material_id);
                  for (auto vertey : neigh_i->vertex_indices())
                    for (const auto &neigh_j :
                         GridTools::find_cells_adjacent_to_vertex(dh, vertey))
                      // for (auto neigh_j =
                      // cell->neighbor(neigh_i)->begin_face(); neigh_j <
                      // cell->neighbor(neigh_i)->end_face(); ++neigh_j)
                      {
                        neigh_j->set_refine_flag();
                        neigh_j->set_material_id(material_id);
                      }
                }
            }
        }

      particle = pic.end();
    }
  execute_actual_refine_and_transfer();

  for (unsigned int ref_cycle = 0; ref_cycle < par.n_refinement_cycles - 1;
       ++ref_cycle)
    {
      for (const auto &cell : tria.active_cell_iterators())
        {
          if (cell->material_id() == material_id)
            cell->set_refine_flag();
        }
      execute_actual_refine_and_transfer();
    }
}

template <int dim, int spacedim>
void
CoupledElasticityProblem<dim, spacedim>::refine_and_transfer()
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
CoupledElasticityProblem<dim, spacedim>::execute_actual_refine_and_transfer()
{
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
CoupledElasticityProblem<dim, spacedim>::output_solution() const
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
// CoupledElasticityProblem<dim, spacedim>::output_particles() const
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
CoupledElasticityProblem<dim, spacedim>::output_results() const
{
  TimerOutput::Scope t(computing_timer, "Postprocessing: Output results");
  static std::vector<std::pair<double, std::string>> cycles_and_solutions;
  static std::vector<std::pair<double, std::string>> cycles_and_particles;

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
CoupledElasticityProblem<dim, spacedim>::print_parameters() const
{
#ifdef USE_PETSC_LA
  pcout << "Running CoupledElasticityProblem<"
        << Utilities::dim_string(dim, spacedim) << "> using PETSc."
        << std::endl;
#else
  pcout << "Running CoupledElasticityProblem<"
        << Utilities::dim_string(dim, spacedim) << "> using Trilinos."
        << std::endl;
#endif
  par.prm.print_parameters(par.output_directory + "/" + "used_parameters_" +
                             std::to_string(dim) + std::to_string(spacedim) +
                             ".prm",
                           ParameterHandler::Short);
}

template <int dim, int spacedim>
void
CoupledElasticityProblem<dim, spacedim>::check_boundary_ids()
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
/*
template <int dim, int spacedim>
void
CoupledElasticityProblem<dim, spacedim>::compute_face_stress(bool
openfilefirsttime) const
{
  TimerOutput::Scope t(computing_timer,
                       "Postprocessing: Computing face stresses");


  const std::string full_filename(par.output_directory +
"/full_face_stress.csv"); std::ofstream     full_face_stress_file; const
std::string filename(par.output_directory + "/face_stress.csv"); std::ofstream
face_stress_file;

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  {
    full_face_stress_file.open(full_filename);
    full_face_stress_file
            << "cellx celly cellz area normalx normaly normalz "
            //<< "face_stress_x face_stress_y face_stress_z"
            << "face_stress_11 face_stress_12 face_stress_13 face_stress_21
face_stress_22 face_stress_23 face_stress_31 face_stress_32 face_stress_33"
            << std::endl;
    face_stress_file.open(filename);
    face_stress_file
            << "cellx celly cellz area normalx normaly normalz "
            << "face_stress_x face_stress_y face_stress_z"
            << std::endl;
  }
  else
  {
    face_stress_file.open(filename, std::ios_base::app);
    full_face_stress_file.open(full_filename, std::ios_base::app);
  }

  // unsigned int number_active_boundary_faces = 0;
  // for (const auto &cell : dh.active_cell_iterators())
  //     if (cell->is_locally_owned())
  //         for (unsigned int f = 0; f <
GeometryInfo<spacedim>::faces_per_cell;
  //              ++f)
  //           if (cell->face(f)->at_boundary())
  //             number_active_boundary_faces ++;

  // Utilities::MPI::sum(count, mpi_communicator);

  // Vector<double> sigma(number_active_boundary_faces);
  // std::vector<Tensor<1, spacedim>> face_stresses_vector;
  // std::vector<Tensor<1, spacedim>> face_normals_vector;
  // std::vector<Point<spacedim>> baricenters_vector;

  std::vector<std::vector<double>> cells_normals_full_stresses;
  std::vector<std::vector<double>> cells_normals_stresses;


  FEFaceValues<spacedim>           fe_face_values(*fe,
                                        *face_quadrature_formula,
                                        update_values | update_gradients |
                                          update_JxW_values |
                                          update_quadrature_points |
                                          update_normal_vectors);
  const FEValuesExtractors::Vector displacement(0);

  Tensor<2, spacedim>                  identity;
  for (unsigned int ix = 0; ix < spacedim; ++ix)
    identity[ix][ix] = 1;

  std::vector<Tensor<2, spacedim>> displacement_gradient(
    face_quadrature_formula->size());
  std::vector<double> displacement_divergence(
    face_quadrature_formula->size());
  std::vector<Tensor<1, spacedim>> displacement_values(
    face_quadrature_formula->size());

  for (const auto &cell : dh.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          for (unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell;
               ++f)
            {
              Tensor<1, spacedim, double> face_stress;
              Tensor<2, spacedim, double> full_face_stress;
              Tensor<1, spacedim, double> face_normal;
              Point<spacedim> baricenter(0.0, 0.0, 0.0);
              face_stress = 0.0;
              full_face_stress = 0.0;
              face_normal = 0.0;
              double cell_area = 0.0;

            if (cell->face(f)->at_boundary())
              {
                fe_face_values.reinit(cell, f);

                fe_face_values[displacement].get_function_gradients(
                  locally_relevant_solution.block(0), displacement_gradient);
                fe_face_values[displacement].get_function_divergences(
                  locally_relevant_solution.block(0), displacement_divergence);

                for (unsigned int q = 0; q < fe_face_values.n_quadrature_points;
                     ++q)
                  {
                    face_stress +=
                      (2 * par.Lame_mu * displacement_gradient[q] +
                       par.Lame_lambda *
                       displacement_divergence[q] * identity
                       ) *
                      fe_face_values.JxW(q)* fe_face_values.normal_vector(q);

                    full_face_stress +=
                      (2 * par.Lame_mu * displacement_gradient[q] +
                       par.Lame_lambda *
                       displacement_divergence[q] * identity
                       ) *
                      fe_face_values.JxW(q);

                    face_normal += fe_face_values.normal_vector(q);
                    baricenter += fe_face_values.quadrature_point(q);
                    cell_area += fe_face_values.JxW(q);
                  }

                face_normal /= fe_face_values.get_normal_vectors().size();
                baricenter /= fe_face_values.get_quadrature_points().size();
                //print on file
                // face_stress_file << baricenter << " " << cell_area << " " <<
face_normal << " " << face_stress << std::endl; std::vector<double>
temp(5*spacedim+1); baricenter.unroll(temp.begin(), temp.begin()+spacedim);
                temp[spacedim] = cell_area;
                face_normal.unroll(temp.begin()+spacedim+1,
temp.begin()+2*spacedim+1); full_face_stress.unroll(temp.begin()+2*spacedim+1,
temp.begin()+(3+2)*spacedim+1); cells_normals_full_stresses.push_back(temp);

                std::vector<double> tump(3*spacedim+1);
                baricenter.unroll(tump.begin(), tump.begin()+spacedim);
                tump[spacedim] = cell_area;
                face_normal.unroll(tump.begin()+spacedim+1,
tump.begin()+2*spacedim+1); face_stress.unroll(tump.begin()+2*spacedim+1,
tump.begin()+3*spacedim+1); cells_normals_stresses.push_back(tump);
              }
            // else
            //   face_stress = 0.0;
            // face_stresses_vector.push_back(face_stress);
            // face_normals_vector.push_back(face_normal);
            // baricenters_vector.push_back(baricenter);
            //sigma.add(cell->active_cell_index(), face_stress);
            }
        }
    }
    //print on file

    for (unsigned int proc_id = 0; proc_id <
Utilities::MPI::n_mpi_processes(mpi_communicator); ++proc_id)
    {
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == proc_id)
      {
        for (const auto & cell_data : cells_normals_stresses)
        {
          for (const auto & data_i : cell_data)
          {
            face_stress_file << data_i << " ";
          }
          face_stress_file << std::endl;
        }
        for (const auto & full_cell_data : cells_normals_full_stresses)
        {
          for (const auto & full_data_i : full_cell_data)
          {
            full_face_stress_file << full_data_i << " ";
          }
          full_face_stress_file << std::endl;
        }
      }
      MPI_Barrier(mpi_communicator);
    }
    face_stress_file.close();
    full_face_stress_file.close();
    // sigma_n.compress(VectorOperation::add);

  return;
}
*/
template <int dim, int spacedim>
void
CoupledElasticityProblem<dim, spacedim>::compute_internal_and_boundary_stress(
  bool openfilefirsttime) const
{
  TimerOutput::Scope t(
    computing_timer,
    "Postprocessing: Computing internal and boundary stresses");

  std::map<types::boundary_id, Tensor<1, spacedim>> boundary_stress;
  Tensor<2, spacedim>                               internal_stress;
  Tensor<1, spacedim>                               average_displacement;
  std::vector<double> u_dot_n(spacedim * spacedim);

  auto                all_ids = tria.get_boundary_ids();
  std::vector<double> perimeter;
  for (auto id : all_ids)
    // for (const auto id : par.dirichlet_ids)
    {
      // boundary_stress[id] = Tensor<1, spacedim>();
      boundary_stress[id] = 0.0;
      perimeter.push_back(0.0);
    }
  double                           internal_area = 0.;
  FEValues<spacedim>               fe_values(*fe,
                               *quadrature,
                               update_values | update_gradients |
                                 update_quadrature_points | update_JxW_values);
  FEFaceValues<spacedim>           fe_face_values(*fe,
                                        *face_quadrature_formula,
                                        update_values | update_gradients |
                                          update_JxW_values |
                                          update_quadrature_points |
                                          update_normal_vectors);
  const FEValuesExtractors::Vector displacement(0);

  const unsigned int                   dofs_per_cell = fe->n_dofs_per_cell();
  const unsigned int                   n_q_points    = quadrature->size();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  Tensor<2, spacedim>                  grad_phi_u;
  double                               div_phi_u;
  Tensor<2, spacedim>                  identity;
  for (unsigned int ix = 0; ix < spacedim; ++ix)
    identity[ix][ix] = 1;
  // std::vector<std::vector<Tensor<1,spacedim>>>
  // solution_gradient(face_quadrature_formula->size(),
  // std::vector<Tensor<1,spacedim> >(spacedim+1));
  std::vector<Tensor<2, spacedim>> displacement_gradient(
    face_quadrature_formula->size());
  std::vector<double> displacement_divergence(face_quadrature_formula->size());
  std::vector<Tensor<1, spacedim>> displacement_values(
    face_quadrature_formula->size());

  for (const auto &cell : dh.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          if constexpr (spacedim == 2)
            {
              cell->get_dof_indices(local_dof_indices);
              fe_values.reinit(cell);
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  internal_area += fe_values.JxW(q);
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                      grad_phi_u =
                        fe_values[displacement].symmetric_gradient(k, q);
                      div_phi_u = fe_values[displacement].divergence(k, q);
                      internal_stress +=
                        (2 * par.Lame_mu * grad_phi_u +
                         par.Lame_lambda * div_phi_u * identity) *
                        locally_relevant_solution.block(
                          0)[local_dof_indices[k]] *
                        fe_values.JxW(q);
                      average_displacement +=
                        fe_values[displacement].value(k, q) *
                        locally_relevant_solution.block(
                          0)[local_dof_indices[k]] *
                        fe_values.JxW(q);
                    }
                }
            }

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

  if constexpr (spacedim == 2)
    {
      internal_stress = Utilities::MPI::sum(internal_stress, mpi_communicator);
      average_displacement =
        Utilities::MPI::sum(average_displacement, mpi_communicator);
      internal_area = Utilities::MPI::sum(internal_area, mpi_communicator);

      internal_stress /= internal_area;
      average_displacement /= internal_area;
    }
  for (auto id : all_ids)
    {
      boundary_stress[id] =
        Utilities::MPI::sum(boundary_stress[id], mpi_communicator);
      perimeter[id] = Utilities::MPI::sum(perimeter[id], mpi_communicator);
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
              forces_file
                << "cycle area perimeter meanInternalStressxx meanInternalStressxy meanInternalStressyx meanInternalStressyy avg_u_x avg_u_y";
              for (auto id : all_ids)
                forces_file << " boundaryStressX_" << id << " boundaryStressY_"
                            << id << " uDotN_" << id;
              forces_file << std::endl;
            }
          else
            {
              forces_file << "cycle perimeter";
              for (auto id : all_ids)
                forces_file << " sigmanX_" << id << " sigmanY_"
                            << " sigmanZ_" << id << " uDotN_" << id;
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

template <int dim, int spacedim>
// TrilinosWrappers::MPI::Vector
void
CoupledElasticityProblem<dim, spacedim>::compute_coupling_pressure() /*const*/
{
  TimerOutput::Scope t(computing_timer, "Postprocessing: Computing Pressure");
  if (inclusions.n_inclusions() > 0 &&
      inclusions.get_offset_coefficients() == 1 &&
      inclusions.get_n_coefficients() >= 2)
    {
      const auto locally_owned_vessels =
        Utilities::MPI::create_evenly_distributed_partitioning(
          mpi_communicator, inclusions.get_n_vessels());
      //  mpi_communicator,
      //  std::min(Utilities::MPI::this_n_processes(mpi_communicator),
      //  inclusions.get_n_vessels()), inclusions.get_n_vessels());
      const auto locally_owned_inclusions =
        Utilities::MPI::create_evenly_distributed_partitioning(
          mpi_communicator, inclusions.n_inclusions());

      // TrilinosWrappers::MPI::Vector pressure(locally_owned_vessels,
      //                                        mpi_communicator);
      coupling_pressure.reinit(locally_owned_vessels, mpi_communicator);
      auto &pressure = coupling_pressure;
      pressure       = 0;
      TrilinosWrappers::MPI::Vector pressure_at_inc(locally_owned_inclusions,
                                                    mpi_communicator);
      // coupling_pressure_at_inclusions.reinit(locally_owned_inclusions,mpi_communicator);
      // auto & pressure_at_inc = coupling_pressure_at_inclusions;
      pressure_at_inc = 0;

      const auto &lambda_to_pressure = locally_relevant_solution.block(1);

      // TODO: set the weight in a smarter way
      // std::vector<double> weights(inclusions.n_inclusions(),
      //                             inclusions.get_h3D1D());
      TrilinosWrappers::MPI::Vector inclusions_to_divide_by(
        locally_owned_vessels, mpi_communicator);
      inclusions_to_divide_by = 0;

      const auto used_number_modes = inclusions.get_n_coefficients();

      const auto local_lambda = lambda_to_pressure.locally_owned_elements();
      if constexpr (spacedim == 3)
        {
          unsigned int previous_inclusion_number =
            numbers::invalid_unsigned_int;
          auto tensorR = inclusions.get_rotation(0);
          for (const auto &ll : local_lambda)
            {
              const unsigned inclusion_number = (unsigned int)floor(
                ll / (inclusions.get_n_coefficients() * spacedim));

              auto lii = ll - inclusion_number *
                                inclusions.get_n_coefficients() * spacedim;
              const unsigned mode_number = (unsigned int)floor(lii / spacedim);
              const unsigned coor_number = lii % spacedim;

              if (previous_inclusion_number != inclusion_number)
                tensorR = inclusions.get_rotation(inclusion_number);

              if (mode_number == 0 || mode_number == 1)
                {
                  AssertIndexRange(inclusion_number, inclusions.n_inclusions());
                  pressure[inclusions.get_vesselID(inclusion_number)] +=
                    lambda_to_pressure[ll] * tensorR[coor_number][mode_number] /
                    used_number_modes;
                  // * weights[inclusion_number];
                  //                    inclusions_to_divide_by[inclusions.get_vesselID(inclusion_number)]
                  //                    += 1;
                  pressure_at_inc[inclusion_number] +=
                    lambda_to_pressure[ll] * tensorR[coor_number][mode_number] /
                    used_number_modes;
                }
              previous_inclusion_number = inclusion_number;
            }
          pressure.compress(VectorOperation::add);
          pressure_at_inc.compress(VectorOperation::add);
        }
      else // spacedim = 2
        {
          for (auto ll : local_lambda)
            {
              const unsigned inclusion_number = (unsigned int)floor(
                ll / (inclusions.get_n_coefficients() * spacedim));

              auto lii = ll - inclusion_number *
                                (inclusions.get_n_coefficients() * spacedim);
              if (lii == 0 || lii == 3)
                {
                  AssertIndexRange(inclusion_number, inclusions.n_inclusions());
                  pressure[inclusions.get_vesselID(inclusion_number)] +=
                    lambda_to_pressure[ll] / used_number_modes;
                }
            }
          pressure_at_inc = pressure;
          pressure.print(std::cout);
          local_lambda.print(std::cout);
        }
      Utilities::MPI::gather(mpi_communicator, pressure_at_inc, 0);
      coupling_pressure_at_inclusions = pressure_at_inc;

      output_coupling_pressure(cycle == 1 ? true : false);
    }
}

template <int dim, int spacedim>
// TrilinosWrappers::MPI::Vector
void
CoupledElasticityProblem<dim, spacedim>::output_coupling_pressure(
  bool openfilefirsttime) const
{
  TimerOutput::Scope t(computing_timer, "Postprocessing: output Pressure");
  if (inclusions.n_inclusions() > 0 &&
      inclusions.get_offset_coefficients() == 1 &&
      inclusions.get_n_coefficients() >= 2)
    {
      const auto &pressure = coupling_pressure;
      // print .txt only sequential
      if (Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
        {
          const std::string filename(par.output_directory +
                                     "/externalPressure.txt");
          std::ofstream     pressure_file;
          if (openfilefirsttime)
            pressure_file.open(filename);
          else
            pressure_file.open(filename, std::ios_base::app);
          // pressure_file << cycle << " ";
          for (unsigned int in = 0; in < pressure.size(); ++in)
            pressure_file << in << " " << pressure[in] << std::endl;
          // pressure.print(pressure_file);
          pressure_file.close();
        }
      else
        // print .h5
        if (par.initial_time == par.final_time)
          {
#ifdef DEAL_II_WITH_HDF5
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
            const auto locally_owned_vessels =
              Utilities::MPI::create_evenly_distributed_partitioning(
                mpi_communicator, inclusions.get_n_vessels());

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
#else

            AssertThrow(false, ExcNeedsHDF5());
#endif
          }
        else
          {
            pcout
              << "output_pressure file for time dependent simulation not implemented"
              << std::endl;
          }
      //       // coupling_pressure = pressure;
      //       return pressure;
    }
  // else
  //   {
  //     pcout
  //       << "inclusions parameters ('Start index of Fourier coefficients' or
  //       'Number of fourier coefficients') not compatible with the computation
  //       of the pressure as intended, pressure.hdf5 not generated"
  //       << std::endl;
  //   }
  // TrilinosWrappers::MPI::Vector temp;
  // return temp;
}

template <int dim, int spacedim>
std::vector<std::vector<double>>
CoupledElasticityProblem<dim, spacedim>::split_pressure_over_inclusions(
  std::vector<int> number_of_cells_per_vessel,
  Vector<double> /* full_press */) const
{
  Assert(number_of_cells_per_vessel.size() == inclusions.get_n_vessels(),
         ExcInternalError());

  std::vector<std::vector<double>> split_pressure;
  unsigned                         starting_inclusion = 0;

  for (unsigned int vessel = 0; vessel < number_of_cells_per_vessel.size();
       ++vessel)
    {
      auto N2 = number_of_cells_per_vessel[vessel];
      auto N1 = inclusions.get_inclusions_in_vessel(vessel);

      std::vector<double> new_vector;

      // const std::vector<double> pressure_of_inc_in_vessel(pressure);//
      // (*pressure[starting_inclusion], *pressure[starting_inclusion+N1]);
      const Vector<double> pressure_of_inc_in_vessel(
        coupling_pressure_at_inclusions);

      new_vector.push_back(pressure_of_inc_in_vessel[starting_inclusion]);

      for (auto cell = 1; cell < N2 - 1; ++cell)
        {
          auto X = cell / (N2 - 1) * (N1 - 1);
          auto j = floor(X);
          auto w = X - j;

          new_vector.push_back(
            (1 - w) * pressure_of_inc_in_vessel[starting_inclusion + j] +
            (w)*pressure_of_inc_in_vessel[starting_inclusion + j + 1]);
        }
      new_vector.push_back(
        pressure_of_inc_in_vessel[starting_inclusion + N1 - 1]);
      starting_inclusion += N1;
      split_pressure.push_back(new_vector);
    }

  return split_pressure;
}


template <int dim, int spacedim>
void
CoupledElasticityProblem<dim, spacedim>::run()
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

          assemble_elasticity_system();

          assemble_coupling();
          solve();
          if (par.output_results)
            output_results();

          {
            if constexpr (spacedim == 2)
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
            par.convergence_table.output_table(pcout.get_stream());
          }
          if constexpr (spacedim == 2)
            {
              // output_pressure(cycle == 0 ? true : false);
              compute_coupling_pressure();
              output_coupling_pressure(cycle == 0 ? true : false);
            }

          if (cycle != par.n_refinement_cycles - 1)
            refine_and_transfer();
        }
      // output_pressure(true);
      compute_coupling_pressure();
      output_coupling_pressure(true);

      if (par.domain_type == "generate")
        compute_internal_and_boundary_stress(true);
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

          if (par.output_results)
            output_results();
          // output_pressure(cycle == 0 ? true : false);
          compute_coupling_pressure();
          output_coupling_pressure(cycle == 0 ? true : false);

          if (par.domain_type == "generate")
            compute_internal_and_boundary_stress(cycle == 0 ? true : false);
        }
    }
}

template <int dim, int spacedim>
void
CoupledElasticityProblem<dim, spacedim>::run_timestep0()
{
  print_parameters();
  make_grid();
  setup_fe();
  check_boundary_ids();
  {
    TimerOutput::Scope t(computing_timer, "Setup inclusion");
    inclusions.setup_inclusions_particles(tria);
  }
  cycle = 0;
  setup_dofs();
  // assemble_elasticity_system();
}

template <int dim, int spacedim>
void
CoupledElasticityProblem<dim, spacedim>::run_timestep()
{
  if (cycle == 0) // at first timestep we refine
    {
      if (par.refinement_strategy == "inclusions")
        {
          refine_and_transfer_around_inclusions();
          std::cout << "refining around inclusions" << std::endl;

          assemble_elasticity_system(); // questo mi serve perche sto raffinando
          assemble_coupling();
          solve();
        }
      else
        {
          for (unsigned int ref_cycle = 0; ref_cycle < par.n_refinement_cycles;
               ++ref_cycle)
            {
              assemble_elasticity_system(); // questo mi serve perche sto
                                            // raffinando
              assemble_coupling();
              solve();
              if (ref_cycle != par.n_refinement_cycles - 1)
                refine_and_transfer();
            }
        }
    }
  else
    {
      reassemble_coupling_rhs();
      solve();
    }


  if (par.output_results)
    output_results();

  coupling_pressure.clear();
  coupling_pressure_at_inclusions.clear();
  // coupling_pressure = output_pressure(cycle == 0 ? true : false);
  //   compute_coupling_pressure();
  //   output_coupling_pressure(cycle == 0 ? true : false);

  compute_internal_and_boundary_stress(cycle == 0 ? true : false);
  cycle++;
}

template <int dim, int spacedim>
void
CoupledElasticityProblem<dim, spacedim>::update_inclusions_data(
  std::vector<double> new_data)
{
  inclusions.update_inclusions_data(new_data);
}

template <int dim, int spacedim>
void
CoupledElasticityProblem<dim, spacedim>::update_inclusions_data(
  std::vector<double> new_data,
  std::vector<int>    cells_per_vessel)
{
  Assert(cells_per_vessel.size() == inclusions.get_n_vessels(),
         ExcInternalError());
  std::vector<std::vector<double>> full_vector;
  unsigned int                     starting_point = 0;
  for (auto &N1 : cells_per_vessel)
    {
      std::vector<double> vessel_vector;
      for (auto j = 0; j < N1; ++j)
        {
          AssertIndexRange(starting_point + j, new_data.size());
          vessel_vector.push_back(new_data[starting_point + j]);
        }
      starting_point += N1;

      full_vector.push_back(vessel_vector);
    }
  inclusions.update_inclusions_data(full_vector);
}

// Template instantiations
template class CoupledElasticityProblemParameters<2>;
template class CoupledElasticityProblemParameters<2, 3>;
template class CoupledElasticityProblemParameters<3>;

template class CoupledElasticityProblem<2>;
template class CoupledElasticityProblem<2, 3>; // dim != spacedim
template class CoupledElasticityProblem<3>;