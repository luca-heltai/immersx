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
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------


#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <immersx/io/utils.h>
#include <immersx/physics/elasticity_explicit.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>

namespace ImmersX
{
  template <int dim, int spacedim>
  ElasticityExplicit<dim, spacedim>::ElasticityExplicit(
    const ElasticityProblemParameters<dim, spacedim> &par)
    : par(par)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , triangulation_storage(std::in_place_type<DistributedTriangulation>,
                            mpi_communicator,
                            typename Triangulation<spacedim>::MeshSmoothing(
                              Triangulation<spacedim>::smoothing_on_refinement |
                              Triangulation<spacedim>::smoothing_on_coarsening),
                            parallel::distributed::Triangulation<
                              spacedim>::construct_multigrid_hierarchy)
    , tria(&std::get<DistributedTriangulation>(triangulation_storage))
    , dh()
    , displacement(0)
  {}

  template <int dim, int spacedim>
  bool
  ElasticityExplicit<dim, spacedim>::uses_fully_distributed_triangulation()
    const
  {
    return std::holds_alternative<FullyDistributedTriangulation>(
      triangulation_storage);
  }

  template <int dim, int spacedim>
  void
  ElasticityExplicit<dim, spacedim>::make_grid()
  {
    const bool need_fully_distributed =
      (par.triangulation_type == "fullydistributed");

    if (need_fully_distributed && !uses_fully_distributed_triangulation())
      triangulation_storage.template emplace<FullyDistributedTriangulation>(
        mpi_communicator);
    else if (!need_fully_distributed && uses_fully_distributed_triangulation())
      triangulation_storage.template emplace<DistributedTriangulation>(
        mpi_communicator,
        typename Triangulation<spacedim>::MeshSmoothing(
          Triangulation<spacedim>::smoothing_on_refinement |
          Triangulation<spacedim>::smoothing_on_coarsening),
        parallel::distributed::Triangulation<
          spacedim>::construct_multigrid_hierarchy);

    tria = &std::visit(
      [](auto &selected_tria) -> parallel::TriangulationBase<spacedim> & {
        return selected_tria;
      },
      triangulation_storage);

    dh.reinit(*tria);

    if (!uses_fully_distributed_triangulation())
      {
        auto &distributed_tria =
          std::get<DistributedTriangulation>(triangulation_storage);

        if (par.domain_type == "generate")
          {
            try
              {
                GridGenerator::generate_from_name_and_arguments(
                  distributed_tria, par.name_of_grid, par.arguments_for_grid);
              }
            catch (...)
              {
                pcout << "Generating from name and argument failed."
                      << std::endl
                      << "Trying to read from file name." << std::endl;
                read_grid_and_cad_files(par.name_of_grid,
                                        par.arguments_for_grid,
                                        distributed_tria);
              }
          }
        else if (par.domain_type == "cylinder")
          {
            Assert(spacedim == 2, ExcInternalError());
            GridGenerator::hyper_ball(distributed_tria, Point<spacedim>(), 1.);
            std::cout << " ATTENTION: GRID: cirle of radius 1." << std::endl;
          }
        else if (par.domain_type == "cheese")
          {
            Assert(spacedim == 2, ExcInternalError());
            GridGenerator::cheese(distributed_tria,
                                  std::vector<unsigned int>(2, 2));
          }
        else if (par.domain_type == "file")
          {
            GridIn<spacedim> gi;
            gi.attach_triangulation(distributed_tria);
#ifdef DEAL_II_WITH_GMSH_API
            std::string infile(par.name_of_grid);
#else
            std::ifstream infile(par.name_of_grid);
            Assert(infile.good(), ExcIO());
#endif
            try
              {
                gi.read_msh(infile);
              }
            catch (...)
              {
                // Try standard readers if msh reader fails, in case the file is
                // not actually a file that msh understands
                gi.read(par.name_of_grid);
              }
          }

        if (par.grid_scale != 1.0)
          GridTools::scale(par.grid_scale, distributed_tria);

        distributed_tria.refine_global(par.initial_refinement);
        return;
      }

    Triangulation<spacedim> serial_tria(
      typename Triangulation<spacedim>::MeshSmoothing(
        Triangulation<spacedim>::smoothing_on_refinement |
        Triangulation<spacedim>::smoothing_on_coarsening));

    if (par.domain_type == "generate")
      {
        try
          {
            GridGenerator::generate_from_name_and_arguments(
              serial_tria, par.name_of_grid, par.arguments_for_grid);
          }
        catch (...)
          {
            pcout << "Generating from name and argument failed." << std::endl
                  << "Trying to read from file name." << std::endl;
            read_grid_and_cad_files(par.name_of_grid,
                                    par.arguments_for_grid,
                                    serial_tria);
          }
      }
    else if (par.domain_type == "cylinder")
      {
        Assert(spacedim == 2, ExcInternalError());
        GridGenerator::hyper_ball(serial_tria, Point<spacedim>(), 1.);
        std::cout << " ATTENTION: GRID: cirle of radius 1." << std::endl;
      }
    else if (par.domain_type == "cheese")
      {
        Assert(spacedim == 2, ExcInternalError());
        GridGenerator::cheese(serial_tria, std::vector<unsigned int>(2, 2));
      }
    else if (par.domain_type == "file")
      {
        GridIn<spacedim> gi;
        gi.attach_triangulation(serial_tria);
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
            // Try standard readers if msh reader fails, in case the file is not
            // actually a file that msh understands
            gi.read(par.name_of_grid);
          }
      }

    if (par.grid_scale != 1.0)
      GridTools::scale(par.grid_scale, serial_tria);

    const auto reference_cell = detail::uniform_reference_cell(serial_tria);
    AssertThrow(reference_cell.has_value(),
                ExcMessage("The explicit elasticity mesh has no cells."));
    if (reference_cell->is_simplex())
      serial_tria.set_mesh_smoothing(Triangulation<spacedim>::none);
    serial_tria.refine_global(par.initial_refinement);
    auto &fully_distributed_tria =
      std::get<FullyDistributedTriangulation>(triangulation_storage);
    for (const auto manifold_id : serial_tria.get_manifold_ids())
      if (manifold_id != numbers::flat_manifold_id)
        fully_distributed_tria.set_manifold(
          manifold_id, serial_tria.get_manifold(manifold_id));

    fully_distributed_tria.copy_triangulation(serial_tria);

    pcout << "Number of active cells: " << tria->n_active_cells() << std::endl
          << "   Boundary ids: "
          << Patterns::Tools::to_string(tria->get_boundary_ids()) << std::endl;

    std::set<types::material_id> material_ids;
    for (const auto &cell : tria->active_cell_iterators())
      material_ids.insert(cell->material_id());
    pcout << "   Material ids: " << Patterns::Tools::to_string(material_ids)
          << std::endl;

    pcout << "   Grid volume: "
          << GridTools::volume(
               *tria, compatibility::linear_mapping<spacedim>(*reference_cell))
          << std::endl
          << "   Dirichlet boundary ids: "
          << Patterns::Tools::to_string(par.dirichlet_ids) << std::endl
          << "   Neumann boundary ids: "
          << Patterns::Tools::to_string(par.neumann_ids) << std::endl;
  }

  template <int dim, int spacedim>
  void
  ElasticityExplicit<dim, spacedim>::setup_fe()
  {
    TimerOutput::Scope t(computing_timer, "Initial setup");
    const auto         reference_cell = detail::uniform_reference_cell(*tria);
    AssertThrow(reference_cell.has_value(),
                ExcMessage("The explicit elasticity mesh has no cells."));
    if (reference_cell->is_hyper_cube())
      fe = std::make_unique<FESystem<spacedim>>(FE_Q<spacedim>(par.fe_degree),
                                                spacedim);
    else if (reference_cell->is_simplex())
      {
        AssertThrow(par.fe_degree == 1,
                    ExcMessage(
                      "Simplex row-sum lumped mass requires FE degree 1."));
        fe = std::make_unique<FESystem<spacedim>>(FE_SimplexP<spacedim>(1),
                                                  spacedim);
      }
    quadrature = std::make_unique<Quadrature<spacedim>>(
      compatibility::gauss_quadrature<spacedim>(*reference_cell,
                                                par.fe_degree + 1));
    face_quadrature_formula = std::make_unique<Quadrature<spacedim - 1>>(
      compatibility::gauss_quadrature<spacedim - 1>(
        reference_cell->face_reference_cell(0), par.fe_degree + 1));
  }

  template <int dim, int spacedim>
  void
  ElasticityExplicit<dim, spacedim>::setup_constraints()
  {
    constraints.clear();
    constraints.reinit(owned_dofs[0], relevant_dofs[0]);
    DoFTools::make_hanging_node_constraints(dh, constraints);
    std::map<types::global_dof_index, double> boundary_values;
    // At intersecting faces, higher boundary ID wins (zero side-wall corners).
    for (const auto id : par.dirichlet_ids)
      {
        std::map<types::global_dof_index, double> face_values;
        VectorTools::interpolate_boundary_values(
          geometry_mapping(), dh, id, par.get_dirichlet_bc(id), face_values);
        for (const auto &entry : face_values)
          boundary_values[entry.first] = entry.second;
      }
    for (const auto &entry : boundary_values)
      if (relevant_dofs[0].is_element(entry.first) &&
          !constraints.is_constrained(entry.first))
        {
          constraints.add_line(entry.first);
          constraints.set_inhomogeneity(entry.first, entry.second);
        }
    constraints.close();
  }

  template <int dim, int spacedim>
  void
  ElasticityExplicit<dim, spacedim>::assemble_elasticity_system()
  {
    TimerOutput::Scope t(computing_timer, "Assemble matrices");
    const unsigned int local_has_hanging_nodes =
      operator_constraints.get_lines().empty() ? 0u : 1u;
    AssertThrow(
      Utilities::MPI::max(local_has_hanging_nodes, mpi_communicator) == 0,
      ExcMessage("Lumped mass requires a mesh without hanging nodes."));

    stiffness_matrix = 0;
    damping_matrix   = 0;
    lumped_mass      = 0.0;

    FEValues<spacedim> fe_values(geometry_mapping(),
                                 *fe,
                                 *quadrature,
                                 update_values | update_gradients |
                                   update_JxW_values);

    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature->size();
    FullMatrix<double> cell_value(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_grad(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_div(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_damping(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiffness(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_lumped_mass(dofs_per_cell);
    std::vector<Tensor<2, spacedim>>     grad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, spacedim>>     phi_u(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto &mp = par.get_material_properties(cell->material_id());
          cell_grad      = 0;
          cell_value     = 0;
          cell_div       = 0;
          cell_mass      = 0;
          cell_damping   = 0;
          cell_stiffness = 0;

          fe_values.reinit(cell);
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
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_grad(i, j) +=
                      scalar_product(grad_phi_u[i], grad_phi_u[j]) *
                      fe_values.JxW(q);
                    cell_div(i, j) +=
                      div_phi_u[i] * div_phi_u[j] * fe_values.JxW(q);
                    cell_value(i, j) += phi_u[i] * phi_u[j] * fe_values.JxW(q);
                  }
            }

          cell->get_dof_indices(local_dof_indices);
          cell_mass.equ(mp.rho, cell_value);
          cell_stiffness.equ(2 * mp.Lame_mu,
                             cell_grad,
                             mp.Lame_lambda,
                             cell_div);
          cell_damping.equ(mp.rayleigh_beta,
                           cell_stiffness,
                           mp.rayleigh_alpha,
                           cell_mass,
                           mp.neta,
                           cell_grad);

          cell_lumped_mass = 0.0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_lumped_mass(i) += cell_mass(i, j);
          lumped_mass.add(local_dof_indices, cell_lumped_mass);

          operator_constraints.distribute_local_to_global(cell_damping,
                                                          local_dof_indices,
                                                          damping_matrix);
          operator_constraints.distribute_local_to_global(cell_stiffness,
                                                          local_dof_indices,
                                                          stiffness_matrix);
        }

    stiffness_matrix.compress(VectorOperation::add);
    damping_matrix.compress(VectorOperation::add);
    lumped_mass.compress(VectorOperation::add);

    unsigned int local_bad_mass = 0;
    for (const auto i : owned_dofs[0])
      if (!std::isfinite(lumped_mass[i]) || lumped_mass[i] <= 0.0)
        ++local_bad_mass;
    AssertThrow(Utilities::MPI::sum(local_bad_mass, mpi_communicator) == 0,
                ExcMessage("Nonpositive or nonfinite lumped nodal mass."));
  }

  template <int dim, int spacedim>
  std::string
  ElasticityExplicit<dim, spacedim>::output_solution() const
  {
    std::vector<std::string> solution_names(spacedim, "displacement");
    std::vector<std::string> exact_solution_names(spacedim,
                                                  "exact_displacement");


    auto exact_vec(solution.block(0));
    exact_vec = 0.0;

    // Ensure the exact solution is evaluated at the current time.
    par.exact_solution.set_time(current_time);

    VectorTools::interpolate(geometry_mapping(),
                             dh,
                             par.exact_solution,
                             exact_vec);
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

    Vector<float> subdomain(tria->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria->locally_owned_subdomain();

    Vector<double> material_ids(tria->n_active_cells());
    {
      auto cell = tria->begin_active();
      auto endc = tria->end();
      for (unsigned int i = 0; cell != endc; ++cell, ++i)
        {
          material_ids[i] = cell->material_id();
        }
    }

    // Material and partition identifiers are constant across time steps.
    if (time_step == 0)
      {
        data_out.add_data_vector(subdomain, "subdomain");
        data_out.add_data_vector(material_ids, "material_id");
      }

    data_out.build_patches(geometry_mapping());
    std::ostringstream filename;
    filename << par.output_name << "_" << std::setw(3) << std::setfill('0')
             << time_step << ".vtu";
    const std::string filename_str = filename.str();
    data_out.write_vtu_in_parallel(par.output_directory + "/" + filename_str,
                                   mpi_communicator);
    return filename_str;
  }

  template <int dim, int spacedim>
  void
  ElasticityExplicit<dim, spacedim>::output_results() const
  {
    TimerOutput::Scope timer(computing_timer, "Output explicit solution");

    const std::string filename = output_solution();

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        static std::vector<std::pair<double, std::string>> records;

        records.emplace_back(current_time, filename);

        std::ofstream pvd(par.output_directory + "/" + par.output_name +
                          "_explicit.pvd");

        AssertThrow(pvd.is_open(),
                    ExcMessage("Could not open the PVD output file."));

        DataOutBase::write_pvd_record(pvd, records);
      }
  }
  template <int dim, int spacedim>
  void
  ElasticityExplicit<dim, spacedim>::assemble_forcing_terms(
    const double evaluation_time)
  {
    par.set_rhs_times(evaluation_time);
    par.set_boundary_condition_times(evaluation_time);
    system_rhs = 0;

    TimerOutput::Scope     t(computing_timer, "Assemble rhs");
    FEValues<spacedim>     fe_values(geometry_mapping(),
                                 *fe,
                                 *quadrature,
                                 update_values | update_quadrature_points |
                                   update_JxW_values);
    FEFaceValues<spacedim> fe_face_values(geometry_mapping(),
                                          *fe,
                                          *face_quadrature_formula,
                                          update_values | update_JxW_values |
                                            update_quadrature_points |
                                            update_normal_vectors);

    const unsigned int                   dofs_per_cell = fe->n_dofs_per_cell();
    const unsigned int                   n_q_points    = quadrature->size();
    Vector<double>                       cell_rhs(dofs_per_cell);
    std::vector<Vector<double>>          rhs_values(n_q_points,
                                           Vector<double>(spacedim));
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_rhs = 0;
          fe_values.reinit(cell);

          const auto &rhs_function = par.get_rhs(cell->material_id());
          const auto  rhs_scale    = rhs_function.scale(evaluation_time);
          rhs_function.vector_value_list(fe_values.get_quadrature_points(),
                                         rhs_values);
          for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const auto component = fe->system_to_component_index(i).first;
                cell_rhs(i) += rhs_scale * fe_values.shape_value(i, q) *
                               rhs_values[q][component] * fe_values.JxW(q);
              }

          for (const auto &face : cell->face_indices())
            if (cell->face(face)->at_boundary() &&
                par.neumann_ids.find(cell->face(face)->boundary_id()) !=
                  par.neumann_ids.end())
              {
                const auto &neumann_bc =
                  par.get_neumann_bc(cell->face(face)->boundary_id());
                const auto scale = neumann_bc.scale(evaluation_time);
                fe_face_values.reinit(cell, face);
                for (unsigned int q = 0; q < fe_face_values.n_quadrature_points;
                     ++q)
                  {
                    double normal_value = 0.0;
                    for (unsigned int d = 0; d < spacedim; ++d)
                      normal_value +=
                        scale *
                        neumann_bc.value(fe_face_values.quadrature_point(q),
                                         d) *
                        fe_face_values.normal_vector(q)[d];
                    normal_value /= spacedim;
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      cell_rhs(i) -= normal_value *
                                     fe_face_values.shape_value(i, q) *
                                     fe_face_values.JxW(q);
                  }
              }

          cell->get_dof_indices(local_dof_indices);
          operator_constraints.distribute_local_to_global(cell_rhs,
                                                          local_dof_indices,
                                                          system_rhs.block(0));
        }

    system_rhs.compress(VectorOperation::add);
  }

  template <int dim, int spacedim>
  void
  ElasticityExplicit<dim, spacedim>::setup_dofs()
  {
    dh.distribute_dofs(*fe);
    owned_dofs    = {dh.locally_owned_dofs()};
    relevant_dofs = {DoFTools::extract_locally_relevant_dofs(dh)};
    par.set_boundary_condition_times(current_time);
    operator_constraints.clear();
    operator_constraints.reinit(owned_dofs[0], relevant_dofs[0]);
    DoFTools::make_hanging_node_constraints(dh, operator_constraints);
    operator_constraints.close();
    setup_boundary_kinematics(current_time);

    DynamicSparsityPattern dsp(relevant_dofs[0]);
    DoFTools::make_sparsity_pattern(dh, dsp, operator_constraints, true);
    std::vector<types::global_dof_index> dofs(fe->n_dofs_per_cell());
    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(dofs);
          operator_constraints.add_entries_local_to_global(dofs,
                                                           operator_constraints,
                                                           dofs,
                                                           dsp);
        }
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               owned_dofs[0],
                                               mpi_communicator,
                                               relevant_dofs[0]);
    stiffness_matrix.reinit(owned_dofs[0],
                            owned_dofs[0],
                            dsp,
                            mpi_communicator);
    lumped_mass.reinit(owned_dofs[0], mpi_communicator);
    damping_matrix.reinit(stiffness_matrix);
    solution.reinit(owned_dofs, mpi_communicator);
    velocity.reinit(owned_dofs, mpi_communicator);
    acceleration.reinit(owned_dofs, mpi_communicator);
    system_rhs.reinit(owned_dofs, mpi_communicator);
    locally_relevant_solution.reinit(owned_dofs,
                                     relevant_dofs,
                                     mpi_communicator);
    pcout << "Explicit displacement DoFs: " << dh.n_dofs() << std::endl;
  }

  template <int dim, int spacedim>
  void
  ElasticityExplicit<dim, spacedim>::setup_boundary_kinematics(
    const double time)
  {
    const double epsilon = 1.e-6;
    par.set_boundary_condition_times(time - epsilon);
    setup_constraints();
    AffineConstraints<double> minus;
    minus.copy_from(constraints);
    par.set_boundary_condition_times(time + epsilon);
    setup_constraints();
    AffineConstraints<double> plus;
    plus.copy_from(constraints);
    par.set_boundary_condition_times(time);
    setup_constraints();
    velocity_constraints.copy_from(constraints);
    acceleration_constraints.copy_from(constraints);
    homogeneous_constraints.copy_from(constraints);
    for (const auto &line : constraints.get_lines())
      {
        const double gm = minus.get_inhomogeneity(line.index);
        const double gp = plus.get_inhomogeneity(line.index);
        velocity_constraints.set_inhomogeneity(line.index,
                                               (gp - gm) / (2 * epsilon));
        acceleration_constraints.set_inhomogeneity(
          line.index, (gp - 2 * line.inhomogeneity + gm) / (epsilon * epsilon));
        homogeneous_constraints.set_inhomogeneity(line.index, 0.0);
      }
  }


  template <int dim, int spacedim>
  void
  ElasticityExplicit<dim, spacedim>::solve_forward_euler(const double dt)
  {
    TimerOutput::Scope timer(computing_timer, "Lumped-mass Euler step");

    // Boundary displacement, velocity, and acceleration at t_n.
    setup_boundary_kinematics(current_time);

    auto &u = solution.block(0);
    auto &v = velocity.block(0);
    auto &a = acceleration.block(0);

    constraints.distribute(u);
    velocity_constraints.distribute(v);

    assemble_forcing_terms(current_time);

    // Strong-boundary columns
    const auto K = linear_operator<LA::MPI::Vector>(stiffness_matrix);

    const auto D = linear_operator<LA::MPI::Vector>(damping_matrix);

    LA::MPI::Vector rhs;
    rhs.reinit(system_rhs.block(0));
    rhs = system_rhs.block(0);

    rhs -= K * u;
    rhs -= D * v;

    // Prescribed boundary equations are not solved.
    homogeneous_constraints.set_zero(rhs);

    const double rhs_norm = rhs.l2_norm();

    AssertThrow(std::isfinite(rhs_norm),
                ExcMessage("Lumped Euler RHS became nonfinite."));

    a = 0.0;

    // Divide by the diagonal lumped mass; no matrix solve is needed.
    for (const auto i : owned_dofs[0])
      if (!homogeneous_constraints.is_constrained(i))
        a[i] = rhs[i] / lumped_mass[i];

    a.compress(VectorOperation::insert);
    acceleration_constraints.distribute(a);

    // Update displacement using the current velocity.
    u.add(dt, v);
    v.add(dt, a);

    setup_boundary_kinematics(current_time + dt);
    constraints.distribute(u);
    velocity_constraints.distribute(v);

    locally_relevant_solution = solution;

    if (time_step % par.output_every_n_steps == 0)
      pcout << "Time step=" << time_step << " time=" << current_time
            << " |u|=" << u.l2_norm() << std::endl;
  }

  template <int dim, int spacedim>
  void
  ElasticityExplicit<dim, spacedim>::run()
  {
    AssertThrow(par.time_mode == TimeMode::Dynamic,
                ExcMessage("elasticity_explicit requires dynamic time mode."));
    AssertThrow(par.time_parameters.time_step > 0.0,
                ExcMessage("Time step must be positive."));
    AssertThrow(par.time_parameters.final_time >=
                  par.time_parameters.initial_time,
                ExcMessage("Final time must not precede initial time."));
    AssertThrow(par.output_every_n_steps > 0,
                ExcMessage("Output frequency must be positive."));
    AssertThrow(
      par.weak_dirichlet_ids.empty(),
      ExcMessage(
        "Only use Strong Dirichlet boundary IDs in elasticity_explicit."));
    pcout << "Elasticity solver with Forward Euler" << std::endl;
    std::filesystem::create_directories(par.output_directory);
    make_grid();
    setup_fe();
    current_time = par.time_parameters.initial_time;
    time_step    = 0;
    setup_dofs();
    assemble_elasticity_system();
    par.initial_displacement.set_time(current_time);
    par.initial_velocity.set_time(current_time);
    VectorTools::interpolate(geometry_mapping(),
                             dh,
                             par.initial_displacement,
                             solution.block(0));
    VectorTools::interpolate(geometry_mapping(),
                             dh,
                             par.initial_velocity,
                             velocity.block(0));
    constraints.distribute(solution.block(0));
    velocity_constraints.distribute(velocity.block(0));
    acceleration              = 0.0;
    locally_relevant_solution = solution;
    output_results();
    while (current_time < par.time_parameters.final_time)
      {
        const double dt =
          std::min(par.time_parameters.time_step,
                   par.time_parameters.final_time - current_time);
        AssertThrow(current_time + dt > current_time,
                    ExcMessage("Time step size is too large"));
        solve_forward_euler(dt);
        current_time =
          std::min(current_time + dt, par.time_parameters.final_time);
        ++time_step;
        if (time_step % par.output_every_n_steps == 0 ||
            current_time == par.time_parameters.final_time)
          {
            locally_relevant_solution = solution;
            output_results();
          }
        const double remaining = par.time_parameters.final_time - current_time;
        if (remaining <=
            16 * std::numeric_limits<double>::epsilon() *
              std::max(1.0, std::abs(par.time_parameters.final_time)))
          current_time = par.time_parameters.final_time;
      }
    par.exact_solution.set_time(current_time);
    if (fe->reference_cell().is_hyper_cube())
      par.convergence_table.error_from_exact(dh,
                                             locally_relevant_solution.block(0),
                                             par.exact_solution);
    if (pcout.is_active() && fe->reference_cell().is_hyper_cube())
      par.convergence_table.output_table(pcout.get_stream());
  }

  template class ElasticityExplicit<2>;
  template class ElasticityExplicit<2, 3>;
  template class ElasticityExplicit<3>;
} // namespace ImmersX
