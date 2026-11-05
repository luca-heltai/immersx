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

#include <immersx/core/reduced_field_values.h>
#include <immersx/physics/coupled_elasticity.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ImmersX
{
  template <int dim, int spacedim>
  CoupledElasticityProblem<dim, spacedim>::CoupledElasticityProblem(
    const ElasticityProblemParameters<dim, spacedim> &par)
    : ElasticityProblem<dim, spacedim>(par)
  {}


  /**
   * @brief Compute the mean normal stress (the "pressure") that the tissue
   * exerts on each vessel wall, from the reduced Lagrange multiplier.
   *
   * The class documentation in coupled_elasticity.h defines the sampled
   * nominal-wall average and its scalar projection.
   */
  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim,
                           spacedim>::compute_pressure_at_immersed_surface()
  {
    TimerOutput::Scope t(this->computing_timer,
                         "Postprocessing: Computing Pressure");

    if (!this->uses_tensor_product_coupling() &&
        this->inclusions.n_inclusions() == 0)
      return;

    const auto     &lambda = this->locally_relevant_solution.block(1);
    LA::MPI::Vector owned_lambda(lambda.locally_owned_elements(),
                                 this->mpi_communicator);
    owned_lambda = lambda;

    if (!this->uses_tensor_product_coupling() &&
        this->inclusions.n_inclusions() > 0)
      {
        const auto &inclusions = this->inclusions;
        const auto &source_inclusions =
          inclusions.locally_owned_source_inclusions();
        const auto source_points = inclusions.locally_owned_source_points();
        const auto n_q = inclusions.n_particles() / inclusions.n_inclusions();
        IndexSet   relevant = owned_lambda.locally_owned_elements();
        for (const auto inclusion : source_inclusions)
          {
            const auto &indices = inclusions.get_dof_indices(inclusion * n_q);
            relevant.add_indices(indices.begin(), indices.end());
          }
        relevant.compress();
        LA::MPI::Vector source_lambda(owned_lambda.locally_owned_elements(),
                                      relevant,
                                      this->mpi_communicator);
        source_lambda = owned_lambda;

        LA::MPI::Vector stiffness;
        if (this->par.remove_interior_stress_from_pressure)
          {
            stiffness.reinit(source_points, this->mpi_communicator);
            for (const auto &particle : inclusions.inclusions_as_particles)
              if (particle.get_surrounding_cell()->is_locally_owned())
                {
                  const auto &material = this->par.get_material_properties(
                    particle.get_surrounding_cell()->material_id());
                  stiffness[particle.get_id()] +=
                    2. * (material.Lame_lambda + material.Lame_mu);
                }
            stiffness.compress(VectorOperation::add);
          }

        const auto         &selection = inclusions.get_selected_coefficients();
        std::vector<double> nominal_displacement(source_inclusions.n_elements(),
                                                 0.);
        if (this->par.remove_interior_stress_from_pressure)
          {
            LA::MPI::Vector nominal_rhs(owned_lambda.locally_owned_elements(),
                                        this->mpi_communicator);
            this->assemble_point_coupling_rhs(nominal_rhs, false);
            LA::MPI::Vector source_datum(owned_lambda.locally_owned_elements(),
                                         relevant,
                                         this->mpi_communicator);
            source_datum = nominal_rhs;
            std::vector<unsigned int> segment_sizes;
            if (inclusions.cluster_with_segments)
              {
                segment_sizes.assign(inclusions.n_global_segments(), 0);
                for (unsigned int inclusion = 0;
                     inclusion < inclusions.n_inclusions();
                     ++inclusion)
                  ++segment_sizes[inclusions.get_segment_index(inclusion *
                                                               n_q)];
              }
            for (const auto inclusion : source_inclusions)
              {
                const auto local =
                  source_inclusions.index_within_set(inclusion);
                const auto &indices =
                  inclusions.get_dof_indices(inclusion * n_q);
                for (unsigned int j = 0; j < indices.size(); ++j)
                  if (selection[j] == spacedim ||
                      selection[j] == 2 * spacedim + 1)
                    nominal_displacement[local] += source_datum[indices[j]];
                if (inclusions.cluster_with_segments)
                  nominal_displacement[local] /=
                    segment_sizes[inclusions.get_segment_index(inclusion *
                                                               n_q)];
              }
          }

        std::vector<double> normal_force(source_inclusions.n_elements(), 0.);
        std::vector<double> interior_force(source_inclusions.n_elements(), 0.);
        std::vector<unsigned int> found(source_inclusions.n_elements(), 0);
        std::vector<double> mean_stiffness(source_inclusions.n_elements(), 0.);
        inclusions.for_each_source_point(
          [&](const auto id, const auto &, const bool inside) {
            if (!inside)
              return;
            const auto local = source_inclusions.index_within_set(
              inclusions.get_inclusion_id(id));
            ++found[local];
            if (this->par.remove_interior_stress_from_pressure)
              mean_stiffness[local] += stiffness[id];
          });
        for (unsigned int local = 0; local < found.size(); ++local)
          if (found[local] > 0)
            mean_stiffness[local] /= found[local];
        inclusions.for_each_source_point(
          [&](const auto id, const auto &point, const bool inside) {
            if (!inside)
              return;
            const auto inclusion = inclusions.get_inclusion_id(id);
            const auto local = source_inclusions.index_within_set(inclusion);

            const auto measure  = inclusions.get_section_measure(inclusion);
            const auto radius   = inclusions.get_radius(inclusion);
            const auto rotation = inclusions.get_rotation(inclusion);
            const auto indices  = inclusions.get_dof_indices(id);
            const auto modes    = inclusions.get_fe_values(id);
            const auto normal =
              (point - inclusions.get_center(inclusion)) / radius;
            Tensor<1, spacedim> traction;
            for (unsigned int j = 0; j < indices.size(); ++j)
              {
                const auto component = inclusions.get_component(indices[j]);
                for (unsigned int d = 0; d < spacedim; ++d)
                  traction[d] += source_lambda[indices[j]] * modes[j] *
                                 rotation[d][component] / measure;
              }
            const auto weight = inclusions.get_JxW(id);
            normal_force[local] += weight * (traction * normal);
            if (this->par.remove_interior_stress_from_pressure)
              interior_force[local] += weight * mean_stiffness[local] *
                                       nominal_displacement[local] / radius;
          });

        // Resolve the products with the normal before using their exact
        // circle moments; otherwise retain the sampled quadrature result.
        const bool resolved_moments =
          n_q > 2 && std::all_of(selection.begin(),
                                 selection.end(),
                                 [n_q](const auto index) {
                                   return (index / spacedim + 1) / 2 + 1 < n_q;
                                 });
        for (const auto inclusion : source_inclusions)
          {
            const auto local = source_inclusions.index_within_set(inclusion);
            if (resolved_moments && found[local] == n_q)
              {
                normal_force[local] = 0.;
                const auto &indices =
                  inclusions.get_dof_indices(inclusion * n_q);
                for (unsigned int j = 0; j < indices.size(); ++j)
                  if (selection[j] == spacedim ||
                      selection[j] == 2 * spacedim + 1)
                    normal_force[local] += 0.5 * source_lambda[indices[j]];
              }
          }

        std::vector<double> vessel_area(inclusions.get_n_vessels(), 0.);
        for (unsigned int inclusion = 0; inclusion < inclusions.n_inclusions();
             ++inclusion)
          vessel_area[inclusions.get_vesselID(inclusion)] +=
            inclusions.get_section_measure(inclusion);
        pressure_at_immersed_surface.reinit(
          Utilities::MPI::create_evenly_distributed_partitioning(
            this->mpi_communicator, inclusions.get_n_vessels()),
          this->mpi_communicator);
        pressure_at_immersed_surface_by_inclusion.reinit(
          Utilities::MPI::create_evenly_distributed_partitioning(
            this->mpi_communicator, inclusions.n_inclusions()),
          this->mpi_communicator);
        for (const auto inclusion : source_inclusions)
          {
            const auto local  = source_inclusions.index_within_set(inclusion);
            const auto force  = normal_force[local] + interior_force[local];
            const auto vessel = inclusions.get_vesselID(inclusion);
            pressure_at_immersed_surface_by_inclusion[inclusion] +=
              force / inclusions.get_section_measure(inclusion);
            pressure_at_immersed_surface[vessel] += force / vessel_area[vessel];
          }
        pressure_at_immersed_surface_by_inclusion.compress(
          VectorOperation::add);
        pressure_at_immersed_surface.compress(VectorOperation::add);
      }
    else if (this->uses_tensor_product_coupling())
      {
#ifdef DEAL_II_WITH_VTK
        AssertThrow(!vessel_index_of_a_cell.empty(),
                    ExcMessage("The reduced grid has no vessel identifiers."));
        const auto &coupling     = *this->tensor_product_coupling;
        const auto &reduced_dh   = coupling.get_dof_handler();
        const auto &fe           = reduced_dh.get_fe();
        const auto  n_components = fe.n_components();
        const auto &selection =
          this->par.tensor_product_coupling_parameters
            .tensor_product_space_parameters.section.selected_coefficients;
        AssertDimension(selection.size(), n_components);
        const auto relevant =
          DoFTools::extract_locally_relevant_dofs(reduced_dh);
        LA::MPI::Vector source_lambda(owned_lambda.locally_owned_elements(),
                                      relevant,
                                      this->mpi_communicator);
        source_lambda             = owned_lambda;
        const auto &source_points = coupling.locally_owned_qpoints();
        const auto &points        = coupling.get_locally_owned_qpoints();
        const auto &weights       = coupling.get_locally_owned_weights();
        const auto &modes         = coupling.get_locally_owned_mode_values();
        const auto &inside        = coupling.source_point_membership();
        AssertDimension(points.size(), source_points.n_elements());
        AssertDimension(inside.size(), points.size());
        std::map<types::global_cell_index,
                 std::vector<std::pair<unsigned int, unsigned int>>>
          samples;
        for (unsigned int q = 0; q < points.size(); ++q)
          {
            const auto [cell, representative_q, section_q] =
              coupling.particle_id_to_representative_indices(
                source_points.nth_index_in_set(q));
            (void)section_q;
            samples[cell].emplace_back(q, representative_q);
          }
        LA::MPI::Vector stiffness, source_coefficients;
        if (this->par.remove_interior_stress_from_pressure)
          {
            LA::MPI::SparseMatrix nominal_mass;
            nominal_mass.reinit(this->inclusion_matrix);
            coupling.assemble_coupling_mass_matrix(nominal_mass);
            LA::MPI::Vector nominal_rhs(owned_lambda.locally_owned_elements(),
                                        this->mpi_communicator);
            LA::MPI::Vector coefficients(owned_lambda.locally_owned_elements(),
                                         this->mpi_communicator);
            coupling.assemble_prescribed_rhs(nominal_rhs);
            LA::MPI::PreconditionJacobi preconditioner;
            preconditioner.initialize(nominal_mass);
            ReductionControl          control(1000, 1.e-14, 1.e-12);
            SolverCG<LA::MPI::Vector> solver(control);
            solver.solve(nominal_mass,
                         coefficients,
                         nominal_rhs,
                         preconditioner);
            source_coefficients.reinit(owned_lambda.locally_owned_elements(),
                                       relevant,
                                       this->mpi_communicator);
            source_coefficients = coefficients;
            stiffness.reinit(source_points, this->mpi_communicator);
            for (const auto &particle : coupling.get_particles())
              if (particle.get_surrounding_cell()->is_locally_owned())
                {
                  const auto &material = this->par.get_material_properties(
                    particle.get_surrounding_cell()->material_id());
                  stiffness[particle.get_id()] +=
                    2. * (material.Lame_lambda + material.Lame_mu);
                }
            stiffness.compress(VectorOperation::add);
          }

        // Project the covered-section mean with the nominal centerline mass.
        // The full representative mesh retains exterior-only cells.
        DoFHandler<1, spacedim> pressure_dh(coupling.get_triangulation());
        const auto             &pressure_fe = fe.base_element(0);
        pressure_dh.distribute_dofs(pressure_fe);
        const auto pressure_relevant =
          DoFTools::extract_locally_relevant_dofs(pressure_dh);
        AffineConstraints<double> constraints;
        constraints.reinit(pressure_dh.locally_owned_dofs(), pressure_relevant);
        DoFTools::make_hanging_node_constraints(pressure_dh, constraints);
        constraints.close();
        DynamicSparsityPattern sparsity(pressure_relevant);
        DoFTools::make_sparsity_pattern(pressure_dh, sparsity, constraints);
        SparsityTools::distribute_sparsity_pattern(
          sparsity,
          pressure_dh.locally_owned_dofs(),
          this->mpi_communicator,
          pressure_relevant);
        LA::MPI::SparseMatrix mass;
        mass.reinit(pressure_dh.locally_owned_dofs(),
                    pressure_dh.locally_owned_dofs(),
                    sparsity,
                    this->mpi_communicator);
        LA::MPI::Vector             rhs(pressure_dh.locally_owned_dofs(),
                            this->mpi_communicator);
        LA::MPI::Vector             projected(pressure_dh.locally_owned_dofs(),
                                  this->mpi_communicator);
        FEValues<1, spacedim>       values(fe,
                                     coupling.get_quadrature(),
                                     update_values | update_quadrature_points |
                                       update_JxW_values);
        FEValues<1, spacedim>       pressure_values(pressure_fe,
                                              coupling.get_quadrature(),
                                              update_values);
        std::vector<Vector<double>> lambda_values(
          coupling.get_quadrature().size(), Vector<double>(n_components));
        std::vector<Vector<double>> coefficient_values(
          coupling.get_quadrature().size(), Vector<double>(n_components));
        std::vector<double> nominal_radius(coupling.get_quadrature().size());
        std::vector<types::global_dof_index> pressure_indices(
          pressure_fe.n_dofs_per_cell());
        std::vector<double> local_integral(n_vessels(), 0.),
          local_length(n_vessels(), 0.);
        for (const auto &cell : reduced_dh.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto pressure_cell =
                cell->as_dof_handler_iterator(pressure_dh);
              pressure_cell->get_dof_indices(pressure_indices);
              values.reinit(cell);
              pressure_values.reinit(pressure_cell);
              values.get_function_values(source_lambda, lambda_values);
              if (this->par.remove_interior_stress_from_pressure)
                {
                  values.get_function_values(source_coefficients,
                                             coefficient_values);
                  coupling.get_thickness_values(cell,
                                                coupling.get_quadrature(),
                                                nominal_radius);
                }
              double       mean_stiffness   = 0.;
              unsigned int material_samples = 0;
              if (this->par.remove_interior_stress_from_pressure)
                {
                  for (const auto &[q, representative_q] :
                       samples[cell->global_active_cell_index()])
                    {
                      (void)representative_q;
                      if (inside[q])
                        {
                          mean_stiffness +=
                            stiffness[source_points.nth_index_in_set(q)];
                          ++material_samples;
                        }
                    }
                  if (material_samples > 0)
                    mean_stiffness /= material_samples;
                }
              const auto          n_q = coupling.get_quadrature().size();
              std::vector<double> nominal_displacement(n_q, 0.),
                nominal_area(n_q, 0.);
              for (const auto &[q, representative_q] :
                   samples[cell->global_active_cell_index()])
                {
                  nominal_area[representative_q] += weights[q][0];
                  if (this->par.remove_interior_stress_from_pressure)
                    {
                      const auto radial =
                        points[q] - values.quadrature_point(representative_q);
                      AssertThrow(
                        radial.norm() > 0.,
                        ExcMessage(
                          "Wall pressure needs a nonzero physical radius."));
                      const auto normal = radial / radial.norm();
                      for (unsigned int c = 0; c < n_components; ++c)
                        if (selection[c] == spacedim ||
                            selection[c] == 2 * spacedim + 1)
                          for (unsigned int d = 0; d < spacedim; ++d)
                            nominal_displacement[representative_q] +=
                              weights[q][0] *
                              coefficient_values[representative_q][c] *
                              modes[q][c * spacedim + d] * normal[d];
                    }
                }
              for (unsigned int q = 0; q < n_q; ++q)
                {
                  AssertThrow(
                    nominal_area[q] > 0.,
                    ExcMessage(
                      "A representative point has no nominal wall samples."));
                  nominal_displacement[q] /= nominal_area[q];
                }

              std::vector<double> normal_force(n_q, 0.);
              for (const auto &[q, representative_q] :
                   samples[cell->global_active_cell_index()])
                {
                  if (!inside[q])
                    continue;
                  const auto radial =
                    points[q] - values.quadrature_point(representative_q);
                  const auto radius = radial.norm();
                  AssertThrow(
                    radius > 0.,
                    ExcMessage(
                      "Wall pressure needs a nonzero physical radius."));
                  const auto normal        = radial / radius;
                  double     normal_stress = 0.;
                  for (unsigned int c = 0; c < n_components; ++c)
                    for (unsigned int d = 0; d < spacedim; ++d)
                      normal_stress += lambda_values[representative_q][c] *
                                       modes[q][c * spacedim + d] * normal[d];
                  if (this->par.remove_interior_stress_from_pressure)
                    normal_stress += mean_stiffness *
                                     nominal_displacement[representative_q] /
                                     nominal_radius[representative_q];
                  normal_force[representative_q] +=
                    weights[q][0] * normal_stress;
                }

              FullMatrix<double> local_mass(pressure_indices.size(),
                                            pressure_indices.size());
              Vector<double>     local_rhs(pressure_indices.size());
              const auto         vessel =
                vessel_index_of_a_cell[cell->active_cell_index()];
              for (unsigned int q = 0; q < n_q; ++q)
                {
                  const double coverage_pressure =
                    normal_force[q] / nominal_area[q];
                  const double weight = values.JxW(q);
                  local_integral[vessel] += weight * coverage_pressure;
                  local_length[vessel] += weight;
                  for (unsigned int i = 0; i < pressure_indices.size(); ++i)
                    {
                      local_rhs[i] += weight * coverage_pressure *
                                      pressure_values.shape_value(i, q);
                      for (unsigned int j = 0; j < pressure_indices.size(); ++j)
                        local_mass(i, j) += weight *
                                            pressure_values.shape_value(i, q) *
                                            pressure_values.shape_value(j, q);
                    }
                }
              constraints.distribute_local_to_global(
                local_mass, local_rhs, pressure_indices, mass, rhs);
            }
        mass.compress(VectorOperation::add);
        rhs.compress(VectorOperation::add);
        LA::MPI::PreconditionJacobi preconditioner;
        preconditioner.initialize(mass);
        SolverControl control(1000, 1.e-14 * std::max(1., rhs.l2_norm()));
        SolverCG<LA::MPI::Vector> solver(control);
        solver.solve(mass, projected, rhs, preconditioner);
        constraints.distribute(projected);
        LA::MPI::Vector relevant_pressure(pressure_dh.locally_owned_dofs(),
                                          pressure_relevant,
                                          this->mpi_communicator);
        relevant_pressure = projected;

        // Retain the public numbering by owned component-zero support points.
        const auto owned_first =
          DoFTools::extract_dofs(
            reduced_dh, fe.component_mask(FEValuesExtractors::Scalar(0))) &
          reduced_dh.locally_owned_dofs();
        const auto [first, total] =
          Utilities::MPI::partial_and_total_sum(owned_first.n_elements(),
                                                this->mpi_communicator);
        IndexSet owned_output(total);
        owned_output.add_range(first, first + owned_first.n_elements());
        pressure_at_immersed_surface_by_inclusion.reinit(
          owned_output, this->mpi_communicator);
        std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
        for (const auto &cell : reduced_dh.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(indices);
              cell->as_dof_handler_iterator(pressure_dh)
                ->get_dof_indices(pressure_indices);
              for (unsigned int i = 0; i < indices.size(); ++i)
                {
                  const auto [component, base] =
                    fe.system_to_component_index(i);
                  if (component == 0 && owned_first.is_element(indices[i]))
                    pressure_at_immersed_surface_by_inclusion
                      [first + owned_first.index_within_set(indices[i])] =
                        relevant_pressure[pressure_indices[base]];
                }
            }
        pressure_at_immersed_surface_by_inclusion.compress(
          VectorOperation::insert);
        std::vector<double> integral(n_vessels()), length(n_vessels());
        Utilities::MPI::sum(local_integral, this->mpi_communicator, integral);
        Utilities::MPI::sum(local_length, this->mpi_communicator, length);
        pressure_at_immersed_surface.reinit(
          Utilities::MPI::create_evenly_distributed_partitioning(
            this->mpi_communicator, n_vessels()),
          this->mpi_communicator);
        for (const auto vessel :
             pressure_at_immersed_surface.locally_owned_elements())
          {
            AssertThrow(
              length[vessel] > 0.,
              ExcMessage("A vessel has no representative quadrature samples."));
            pressure_at_immersed_surface[vessel] =
              integral[vessel] / length[vessel];
          }
        pressure_at_immersed_surface.compress(VectorOperation::insert);
#endif
      }
  }



  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::make_vessel_index_of_a_cell()
  {
    AssertThrow(this->tensor_product_coupling,
                ExcMessage("Tensor-product coupling not initialized"));

    const auto &bindings =
      this->tensor_product_coupling->get_properties_bindings();
    const auto binding = std::find_if(bindings.begin(),
                                      bindings.end(),
                                      [](const InputFieldBinding &b) {
                                        return b.symbol_name == "vessel_id";
                                      });
    AssertThrow(binding != bindings.end(),
                ExcMessage("vessel_id is not listed in the input file fields"));
    const unsigned int slot = binding - bindings.begin();

    const auto &reduced_dh = this->tensor_product_coupling->get_dof_handler();
    const auto &properties_dh =
      this->tensor_product_coupling->get_properties_dh();
    const auto &quadrature = this->tensor_product_coupling->get_quadrature();

    ReducedFieldValues<1, spacedim> field_values(
      properties_dh,
      quadrature,
      this->tensor_product_coupling->get_properties(),
      bindings);
    std::vector<double> values(quadrature.size() * bindings.size());

    // Indexed by the rank-local active cell index; ghost and artificial cells
    // keep invalid_unsigned_int.
    vessel_index_of_a_cell.assign(
      reduced_dh.get_triangulation().n_active_cells(),
      numbers::invalid_unsigned_int);

    // Cell fields are piecewise constant, so the first quadrature point
    // carries the value of the cell.
    unsigned int local_largest_id = 0;
    for (const auto &cell : reduced_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          field_values.extract(cell->as_dof_handler_iterator(properties_dh),
                               values);
          const double value = std::round(values[slot]);
          AssertThrow(value >= 0,
                      ExcMessage("vessel_id must be a non-negative integer"));
          const auto id = static_cast<unsigned int>(value);
          vessel_index_of_a_cell[cell->active_cell_index()] = id;
          local_largest_id = std::max(local_largest_id, id);
        }

    // Cells are distributed, so both the largest id and the set of ids
    // present must be gathered over all ranks before checking for gaps.
    const unsigned int largest_id =
      Utilities::MPI::max(local_largest_id, this->mpi_communicator);
    std::vector<unsigned int> local_present(largest_id + 1, 0);
    for (const auto id : vessel_index_of_a_cell)
      if (id != numbers::invalid_unsigned_int)
        local_present[id] = 1;
    std::vector<unsigned int> present(largest_id + 1, 0);
    Utilities::MPI::max(local_present, this->mpi_communicator, present);
    for (unsigned int id = 0; id <= largest_id; ++id)
      AssertThrow(present[id] == 1,
                  ExcMessage("vessel_id values must be consecutive from 0, "
                             "but no cell has vessel_id " +
                             std::to_string(id) + " while the largest is " +
                             std::to_string(largest_id)));

    n_vessels_from_input = largest_id + 1;
  }



  template <int dim, int spacedim>
  // TrilinosWrappers::MPI::Vector
  void
  CoupledElasticityProblem<dim, spacedim>::output_pressure_at_immersed_surface(
    bool openfilefirsttime) const
  {
    TimerOutput::Scope t(this->computing_timer,
                         "Postprocessing: output Pressure");
    if (this->par.output_pressure == false)
      {
        return;
      }
    // if (this->par.time_parameters.initial_time !=
    //     this->par.time_parameters.final_time)
    //   {
    //     this->pcout
    //       << "output_pressure file for time dependent simulation not
    //       implemented in MPI"
    //       << std::endl;
    //   }

    const auto &pressure = pressure_at_immersed_surface;

    // print .txt only sequential
    if (Utilities::MPI::n_mpi_processes(this->mpi_communicator) == 1)
      {
        const std::string filename(this->par.output_directory + "/" +
                                   this->par.output_name +
                                   "_externalPressure.txt");
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

        pressure_file << this->cycle << " ";
        pressure.print(pressure_file);
        pressure_file.close();
      }
    else
      // print .h5
      {
#ifdef DEAL_II_WITH_HDF5
        const std::string FILE_NAME(this->par.output_directory + "/" +
                                    this->par.output_name +
                                    "_externalPressure.h5");

        auto accessMode = HDF5::File::FileAccessMode::create;
        if (!openfilefirsttime)
          accessMode = HDF5::File::FileAccessMode::open;

        HDF5::File file_h5(FILE_NAME, accessMode, this->mpi_communicator);
        const std::string DATASET_NAME("externalPressure_" +
                                       std::to_string(this->cycle));

        HDF5::DataSet dataset =
          file_h5.create_dataset<double>(DATASET_NAME, {n_vessels()});

        std::vector<double> data_to_write;
        // std::vector<hsize_t> coordinates;
        data_to_write.reserve(pressure.locally_owned_size());
        // coordinates.reserve(pressure.locally_owned_size());
        const auto locally_owned_vessels =
          Utilities::MPI::create_evenly_distributed_partitioning(
            this->mpi_communicator, n_vessels());

        for (const auto &el : locally_owned_vessels)
          {
            // coordinates.emplace_back(el);
            data_to_write.emplace_back(pressure[el]);
          }
        // The prefix sum is collective: every rank must call it, including
        // ranks that own no vessel.
        const hsize_t los = pressure.locally_owned_size();
        const hsize_t prefix =
          Utilities::MPI::partial_and_total_sum(los, this->mpi_communicator)
            .first;
        if (los > 0)
          {
            std::vector<hsize_t> offset = {prefix, 1};
            std::vector<hsize_t> count  = {pressure.locally_owned_size(), 1};
            // data.write_selection(data_to_write, coordinates);
            dataset.write_hyperslab(data_to_write, offset, count);
          }
        else
          dataset.write_none<int>();
#else
        AssertThrow(false, ExcNeedsHDF5());
#endif
      }
  }



  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::run_timestep0()
  {
    this->print_parameters();
    this->make_grid();
    this->setup_fe();
    if (!this->uses_tensor_product_coupling())
      {
        TimerOutput::Scope t(this->computing_timer, "Setup inclusion");
        this->inclusions.setup_inclusions_particles(*this->tria);
      }

    this->cycle        = 0;
    this->current_time = this->par.time_parameters.initial_time;
    this->setup_dofs();
    if (this->uses_tensor_product_coupling())
      {
#ifdef DEAL_II_WITH_VTK
        make_vessel_index_of_a_cell();
#endif
      }
    // assemble_elasticity_system();
  }

  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::run_timestep()
  {
    run_timestep(this->current_time);
  }


  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::run_timestep(
    const double physical_time)
  {
    AssertThrow(std::isfinite(physical_time),
                ExcMessage("The coupled mechanical time must be finite."));
    const bool time_changed = physical_time != this->current_time;
    this->current_time      = physical_time;
    if (time_changed)
      {
        this->par.set_boundary_condition_times(physical_time);
        this->setup_constraints();
      }
    if (this->cycle == 0) // at first timestep we refine
      {
        if (this->par.refinement_strategy == "inclusions")
          {
            this->refine_and_transfer_around_inclusions();
            // this->pcout << "refining around inclusions" << std::endl;

            this->assemble_elasticity_system(); // we need to refine A too
            this->assemble_coupling();
            this->solve();
          }
        else
          {
            for (unsigned int ref_cycle = 0;
                 ref_cycle < this->par.n_refinement_cycles;
                 ++ref_cycle)
              {
                this->assemble_elasticity_system(); // questo mi serve perche
                                                    // sto raffinando
                this->assemble_coupling();
                this->solve();
                if (ref_cycle != this->par.n_refinement_cycles - 1)
                  this->refine_and_transfer();
              }
          }
      }
    else
      {
        // Reassemble the constrained mechanical operator and load together:
        // cached essential columns cannot supply a new affine lifting.
        if (time_changed)
          this->assemble_elasticity_system();
        this->assemble_coupling();
        this->solve();
      }


    // if (this->par.output_results)
    this->output_results();

    pressure_at_immersed_surface.clear();
    pressure_at_immersed_surface_by_inclusion.clear();
    // vessel_wall_pressure =
    // output_pressure(cycle == 0 ? true : false);
    compute_pressure_at_immersed_surface();
    output_pressure_at_immersed_surface(this->cycle == 0 ? true : false);

    this->compute_internal_and_boundary_stress(this->cycle == 0 ? true : false);
    this->cycle++;
  }

  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::update_vessel_radius_data(
    std::vector<double> new_data)
  {
    if (!this->uses_tensor_product_coupling())
      this->inclusions.update_inclusions_data(new_data);
    else
      {
#ifdef DEAL_II_WITH_VTK
        AssertThrow(false,
                    ExcMessage("Cannot update data for tensor coupling"));

#endif
        AssertThrow(
          false,
          ExcMessage(
            "Trying to update vessel radius data: the code uses tensor_coupling but VTK flag is OFF"));
      }
  }

  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::update_vessel_radius_data(
    std::vector<double> new_data,
    std::vector<int>    cells_per_vessel)
  {
    if (!this->uses_tensor_product_coupling())
      {
        Assert(cells_per_vessel.size() == this->inclusions.get_n_vessels(),
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
        this->inclusions.update_inclusions_data(full_vector);
      }
    else
      {
#ifdef DEAL_II_WITH_VTK
        AssertThrow(false,
                    ExcMessage("Cannot update data for tensor coupling"));

#endif
        AssertThrow(
          false,
          ExcMessage(
            "Trying to update vessel radius data: the code uses tensor_coupling but VTK flag is OFF"));
      }
  }

  template <int dim, int spacedim>
  std::vector<std::vector<double>>
  CoupledElasticityProblem<dim, spacedim>::split_pressure_over_inclusions(
    std::vector<int> number_of_cells_per_vessel,
    Vector<double>   full_press) const
  {
    std::vector<std::vector<double>> split_pressure;

    if (!this->uses_tensor_product_coupling())
      {
        AssertThrow(number_of_cells_per_vessel.size() ==
                      this->inclusions.get_n_vessels(),
                    ExcDimensionMismatch(number_of_cells_per_vessel.size(),
                                         this->inclusions.get_n_vessels()));

        AssertThrow(full_press.size() == this->inclusions.n_inclusions(),
                    ExcDimensionMismatch(full_press.size(),
                                         this->inclusions.n_inclusions()));
        const auto &pressure_by_inclusion = full_press;

        for (unsigned int vessel = 0;
             vessel < number_of_cells_per_vessel.size();
             ++vessel)
          {
            const auto &current_inclusions =
              this->inclusions.map_vessel_inclusions.at(vessel);

            const int  N2 = number_of_cells_per_vessel[vessel]; // cells
            const auto N1 = current_inclusions.size(); // inclusions in vessel

            AssertThrow(N2 > 0,
                        ExcMessage("Each vessel must have at least one cell."));

            std::vector<double> new_vector(N2, 0.);
            if (N1 == 1)
              {
                // Nothing to interpolate: every cell gets the pressure of the
                // only inclusion.
                std::fill(new_vector.begin(),
                          new_vector.end(),
                          pressure_by_inclusion[current_inclusions[0]]);
              }
            else if (N2 == 1)
              {
                // A single cell gets the mean pressure of the vessel.
                for (const auto inclusion_id : current_inclusions)
                  new_vector[0] += pressure_by_inclusion[inclusion_id];
                new_vector[0] /= N1;
              }
            else
              {
                // The cells are equally spaced from the first to the last
                // inclusion of the vessel, in the order of the inclusions file.
                for (int cell = 0; cell < N2; ++cell)
                  {
                    const double X =
                      static_cast<double>(cell) * (N1 - 1) / (N2 - 1);
                    const auto j =
                      std::min(static_cast<std::size_t>(std::floor(X)), N1 - 2);
                    const double w = X - static_cast<double>(j);
                    new_vector[cell] =
                      (1 - w) * pressure_by_inclusion[current_inclusions[j]] +
                      w * pressure_by_inclusion[current_inclusions[j + 1]];
                  }
              }
            split_pressure.push_back(new_vector);
          }

        return split_pressure;
      }
    else
      {
#ifdef DEAL_II_WITH_VTK
        AssertThrow(
          false,
          ExcMessage(
            "Cannot split the pressure over inclusions for tensor coupling"));

#endif
        AssertThrow(
          false,
          ExcMessage(
            "Trying to split the pressure over inclusions: the code uses tensor_coupling but VTK flag is OFF"));
        return split_pressure;
      }
  }


  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim,
                           spacedim>::refine_and_transfer_around_inclusions()
  {
    if (!this->uses_tensor_product_coupling())
      {
        TimerOutput::Scope t(this->computing_timer, "Refine");
        AssertThrow(this->fe->reference_cell().is_hyper_cube(),
                    ExcMessage(
                      "Refine simplex tissue meshes before distribution."));
        Vector<float> error_per_cell(this->tria->n_active_cells());
        KellyErrorEstimator<spacedim>::estimate(
          this->dh,
          QGauss<spacedim - 1>(this->par.fe_degree + 1),
          {},
          this->locally_relevant_solution.block(0),
          error_per_cell);

        const int material_id = 1;

        auto particle = this->inclusions.inclusions_as_particles.begin();
        while (particle != this->inclusions.inclusions_as_particles.end())
          {
            const auto &cell = particle->get_surrounding_cell();
            const auto  pic =
              this->inclusions.inclusions_as_particles.particles_in_cell(cell);
            Assert(pic.begin() == particle, ExcInternalError());

            cell->set_refine_flag();
            cell->set_material_id(material_id);

            // for (auto f : cell->face_indices())
            // for (unsigned int face_no = 0; face_no
            // <GeometryInfo<spacedim>::faces_per_cell; ++face_no)
            for (auto vertex : cell->vertex_indices())
              {
                for (const auto &neigh_i :
                     GridTools::find_cells_adjacent_to_vertex(this->dh, vertex))
                  // for (auto neigh_i = cell->neighbor(face_no)->begin_face();
                  // neigh_i < cell->neighbor(face_no)->end_face(); ++neigh_i)
                  {
                    if (!neigh_i->refine_flag_set())
                      {
                        neigh_i->set_refine_flag();
                        neigh_i->set_material_id(material_id);
                        for (auto vertey : neigh_i->vertex_indices())
                          for (const auto &neigh_j :
                               GridTools::find_cells_adjacent_to_vertex(
                                 this->dh, vertey))
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
        this->execute_actual_refine_and_transfer();

        for (unsigned int ref_cycle = 0;
             ref_cycle < this->par.n_refinement_cycles - 1;
             ++ref_cycle)
          {
            for (const auto &cell : this->tria->active_cell_iterators())
              {
                if (cell->material_id() == material_id)
                  cell->set_refine_flag();
              }
            this->execute_actual_refine_and_transfer();
          }
      }
    else
      this->pcout
        << "Inclusion refinement is only implemented for Point Coupling type"
        << std::endl;
  }

  // Template instantiations
  template class CoupledElasticityProblem<2>;
  template class CoupledElasticityProblem<2, 3>; // dim != spacedim
  template class CoupledElasticityProblem<3>;

} // namespace ImmersX
