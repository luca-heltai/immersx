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

#include <deal.II/grid/cell_id.h>

#include <immersx/core/reduced_field_values.h>
#include <immersx/physics/coupled_elasticity.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ImmersX
{
  namespace
  {
    // Value at normalized position s in [0, 1] of samples equally spaced from
    // s = 0 to s = 1, interpolated linearly. A single sample is constant.
    double
    interpolate_samples(const std::vector<double> &samples, const double s)
    {
      AssertThrow(!samples.empty(), ExcMessage("No samples to interpolate."));
      if (samples.size() == 1)
        return samples[0];
      const double x = std::clamp(s, 0., 1.) * (samples.size() - 1);
      const auto   j =
        std::min(static_cast<std::size_t>(std::floor(x)), samples.size() - 2);
      const double w = x - static_cast<double>(j);
      return (1 - w) * samples[j] + w * samples[j + 1];
    }

    // Exact lexicographic order of points, to identify shared vertices.
    template <int spacedim>
    struct PointLess
    {
      bool
      operator()(const Point<spacedim> &a, const Point<spacedim> &b) const
      {
        for (unsigned int d = 0; d < spacedim; ++d)
          if (a[d] != b[d])
            return a[d] < b[d];
        return false;
      }
    };
  } // namespace

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
        pressure_at_immersed_surface_by_support_point.reinit(
          Utilities::MPI::create_evenly_distributed_partitioning(
            this->mpi_communicator, inclusions.n_inclusions()),
          this->mpi_communicator);
        for (const auto inclusion : source_inclusions)
          {
            const auto local  = source_inclusions.index_within_set(inclusion);
            const auto force  = normal_force[local] + interior_force[local];
            const auto vessel = inclusions.get_vesselID(inclusion);
            pressure_at_immersed_surface_by_support_point[inclusion] +=
              force / inclusions.get_section_measure(inclusion);
            pressure_at_immersed_surface[vessel] += force / vessel_area[vessel];
          }
        pressure_at_immersed_surface_by_support_point.compress(
          VectorOperation::add);
        pressure_at_immersed_surface.compress(VectorOperation::add);
      }
    else if (this->uses_tensor_product_coupling())
      {
#ifdef DEAL_II_WITH_VTK
        // The coupling is rebuilt after mesh refinement.
        if (parametrized_coupling_generation !=
            this->tensor_product_coupling_generation)
          make_vessel_index_of_a_cell();
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
        pressure_at_immersed_surface_by_support_point.reinit(
          owned_output, this->mpi_communicator);
        std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
        // Vessel and normalized arc length of each support point of each
        // locally owned cell, by reduced dof. A support point where vessels
        // meet belongs to each of them. The entries of the output vector are
        // known on the ranks that own the dofs.
        std::vector<double> local_samples, local_entries;
        for (const auto &cell : reduced_dh.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(indices);
              cell->as_dof_handler_iterator(pressure_dh)
                ->get_dof_indices(pressure_indices);
              const auto &arc =
                reduced_cell_arcs.at(cell->global_active_cell_index());
              for (unsigned int i = 0; i < indices.size(); ++i)
                {
                  const auto [component, base] =
                    fe.system_to_component_index(i);
                  if (component != 0)
                    continue;
                  const double t = fe.unit_support_point(i)[0];
                  local_samples.insert(local_samples.end(),
                                       {static_cast<double>(indices[i]),
                                        static_cast<double>(arc.vessel),
                                        arc.first_arc_length +
                                          t * (arc.second_arc_length -
                                               arc.first_arc_length)});
                  if (owned_first.is_element(indices[i]))
                    {
                      const auto entry =
                        first + owned_first.index_within_set(indices[i]);
                      pressure_at_immersed_surface_by_support_point[entry] =
                        relevant_pressure[pressure_indices[base]];
                      local_entries.insert(local_entries.end(),
                                           {static_cast<double>(indices[i]),
                                            static_cast<double>(entry)});
                    }
                }
            }
        pressure_at_immersed_surface_by_support_point.compress(
          VectorOperation::insert);
        std::map<types::global_dof_index, std::size_t> entry_of_dof;
        for (const auto &entries :
             Utilities::MPI::all_gather(this->mpi_communicator, local_entries))
          for (unsigned int k = 0; k < entries.size(); k += 2)
            entry_of_dof[static_cast<types::global_dof_index>(entries[k])] =
              static_cast<std::size_t>(entries[k + 1]);
        n_pressure_entries = total;
        pressure_sample_entry.clear();
        pressure_sample_vessel.clear();
        pressure_sample_arc_length.clear();
        for (const auto &samples :
             Utilities::MPI::all_gather(this->mpi_communicator, local_samples))
          for (unsigned int k = 0; k < samples.size(); k += 3)
            {
              pressure_sample_entry.push_back(entry_of_dof.at(
                static_cast<types::global_dof_index>(samples[k])));
              pressure_sample_vessel.push_back(
                static_cast<unsigned int>(samples[k + 1]));
              pressure_sample_arc_length.push_back(samples[k + 2]);
            }
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

        // The particle output shows the vessel value of each wall point.
        std::vector<double> owned_pressure(n_vessels(), 0.),
          vessel_pressure(n_vessels());
        for (const auto vessel :
             pressure_at_immersed_surface.locally_owned_elements())
          owned_pressure[vessel] = pressure_at_immersed_surface[vessel];
        Utilities::MPI::sum(owned_pressure,
                            this->mpi_communicator,
                            vessel_pressure);
        this->vessel_pressure_at_wall_points.assign(points.size(), 0.);
        for (const auto &cell : reduced_dh.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto vessel =
                vessel_index_of_a_cell[cell->active_cell_index()];
              for (const auto &[q, representative_q] :
                   samples[cell->global_active_cell_index()])
                {
                  (void)representative_q;
                  this->vessel_pressure_at_wall_points[q] =
                    vessel_pressure[vessel];
                }
            }
        this->has_vessel_pressure_at_wall_points = true;
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

    // Normalized arc length along each vessel. Gather the id, vessel and
    // vertices of every reduced cell, walk each vessel from its first end,
    // and keep the result for the locally owned cells.
    std::vector<std::string> local_ids;
    std::vector<double>      local_data;
    for (const auto &cell : reduced_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          local_ids.push_back(cell->id().to_string());
          local_data.push_back(
            vessel_index_of_a_cell[cell->active_cell_index()]);
          for (const unsigned int v : {0u, 1u})
            for (unsigned int d = 0; d < spacedim; ++d)
              local_data.push_back(cell->vertex(v)[d]);
        }
    const auto all_ids =
      Utilities::MPI::all_gather(this->mpi_communicator, local_ids);
    const auto all_data =
      Utilities::MPI::all_gather(this->mpi_communicator, local_data);

    struct ChainCell
    {
      CellId                         id;
      std::array<Point<spacedim>, 2> vertices;
    };
    std::map<unsigned int, std::vector<ChainCell>> cells_of_vessel;
    for (unsigned int rank = 0; rank < all_ids.size(); ++rank)
      for (unsigned int k = 0; k < all_ids[rank].size(); ++k)
        {
          const double *data  = &all_data[rank][k * (2 * spacedim + 1)];
          ChainCell     chain = {CellId(all_ids[rank][k]), {}};
          for (const unsigned int v : {0u, 1u})
            for (unsigned int d = 0; d < spacedim; ++d)
              chain.vertices[v][d] = data[1 + v * spacedim + d];
          cells_of_vessel[static_cast<unsigned int>(data[0])].push_back(chain);
        }

    std::map<std::string, std::array<double, 2>> arc_of_cell;
    for (const auto &vessel_and_cells : cells_of_vessel)
      {
        const unsigned int vessel = vessel_and_cells.first;
        const auto        &cells  = vessel_and_cells.second;
        // Cell and local vertex index of the cells at each vertex.
        std::map<Point<spacedim>,
                 std::vector<std::pair<unsigned int, unsigned int>>,
                 PointLess<spacedim>>
          incident;
        for (unsigned int c = 0; c < cells.size(); ++c)
          for (const unsigned int v : {0u, 1u})
            incident[cells[c].vertices[v]].emplace_back(c, v);
        std::vector<std::pair<unsigned int, unsigned int>> ends;
        for (const auto &[point, at_vertex] : incident)
          {
            (void)point;
            AssertThrow(at_vertex.size() <= 2,
                        ExcMessage("Vessel " + std::to_string(vessel) +
                                   " of the reduced grid branches."));
            if (at_vertex.size() == 1)
              ends.push_back(at_vertex[0]);
          }
        AssertThrow(ends.size() == 2,
                    ExcMessage("Vessel " + std::to_string(vessel) +
                               " of the reduced grid is not a single chain of "
                               "cells."));
        const auto key = [&](const std::pair<unsigned int, unsigned int> &e) {
          return std::make_pair(cells[e.first].id, e.second);
        };
        auto [c, v] = key(ends[1]) < key(ends[0]) ? ends[1] : ends[0];

        std::vector<std::array<double, 2>> arc(cells.size());
        std::vector<bool>                  visited(cells.size(), false);
        double                             length = 0.;
        for (unsigned int k = 0; k < cells.size(); ++k)
          {
            AssertThrow(!visited[c],
                        ExcMessage("Vessel " + std::to_string(vessel) +
                                   " of the reduced grid is not a single "
                                   "chain of cells."));
            visited[c] = true;
            arc[c][v]  = length;
            length += cells[c].vertices[0].distance(cells[c].vertices[1]);
            arc[c][1 - v] = length;
            if (k + 1 < cells.size())
              for (const auto &[next, w] :
                   incident.at(cells[c].vertices[1 - v]))
                if (next != c)
                  {
                    c = next;
                    v = w;
                    break;
                  }
          }
        AssertThrow(length > 0.,
                    ExcMessage("Vessel " + std::to_string(vessel) +
                               " of the reduced grid has zero length."));
        for (unsigned int k = 0; k < cells.size(); ++k)
          arc_of_cell[cells[k].id.to_string()] = {
            {arc[k][0] / length, arc[k][1] / length}};
      }

    reduced_cell_arcs.clear();
    for (const auto &cell : reduced_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto &arc = arc_of_cell.at(cell->id().to_string());
          reduced_cell_arcs[cell->global_active_cell_index()] = {
            vessel_index_of_a_cell[cell->active_cell_index()],
            cell->vertex(0),
            cell->vertex(1),
            arc[0],
            arc[1]};
        }
    parametrized_coupling_generation = this->tensor_product_coupling_generation;
  }



  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::install_tensor_wall_data()
  {
#ifdef DEAL_II_WITH_VTK
    if (!this->uses_tensor_product_coupling())
      return;
    AssertThrow(this->tensor_product_coupling,
                ExcMessage("Tensor-product coupling not initialized"));
    // The coupling is rebuilt after mesh refinement.
    if (parametrized_coupling_generation !=
        this->tensor_product_coupling_generation)
      make_vessel_index_of_a_cell();
    if (wall_displacement_samples.empty())
      {
        this->tensor_product_coupling->modal_data = nullptr;
        return;
      }
    const auto selection =
      this->par.tensor_product_coupling_parameters
        .tensor_product_space_parameters.section.selected_coefficients;
    // The 1D wall displacement enters only through the two radial modes.
    // Without them every update would be dropped without notice.
    const auto selects = [&selection](const unsigned int mode) {
      return std::find(selection.begin(), selection.end(), mode) !=
             selection.end();
    };
    AssertThrow(selects(spacedim) && selects(2 * spacedim + 1),
                ExcMessage("The 1D wall displacement needs both radial modes, "
                           "coefficients " +
                           std::to_string(spacedim) + " and " +
                           std::to_string(2 * spacedim + 1) +
                           ", in Cross section/Selected modes."));
    this->tensor_product_coupling->modal_data =
      [this, selection](const types::global_cell_index cell,
                        const Point<spacedim>         &point) {
        const auto  &arc = reduced_cell_arcs.at(cell);
        const double t   = arc.first_vertex.distance(point) /
                         arc.first_vertex.distance(arc.second_vertex);
        const double s = arc.first_arc_length +
                         t * (arc.second_arc_length - arc.first_arc_length);
        const double value =
          interpolate_samples(wall_displacement_samples[arc.vessel], s);
        std::vector<double> coefficients(selection.size(), 0.);
        for (unsigned int c = 0; c < selection.size(); ++c)
          if (selection[c] == spacedim || selection[c] == 2 * spacedim + 1)
            coefficients[c] = value;
        return coefficients;
      };
#endif
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
    // Time-dependent runs impose the wall data at each step, as static runs
    // do.
    this->wall_displacement_constraint =
      this->par.time_mode != TimeMode::Static;
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
    install_tensor_wall_data();
    if (this->par.time_mode == TimeMode::Dynamic)
      advance_newmark(physical_time);
    else
      solve_static_timestep(physical_time);

    pressure_at_immersed_surface.clear();
    pressure_at_immersed_surface_by_support_point.clear();
    // The pressure comes first: the wall particles written by output_results()
    // carry it.
    compute_pressure_at_immersed_surface();

    // `cycle` counts the coupled steps taken so far.
    if (this->cycle % this->par.output_every_n_steps == 0)
      this->output_results();

    output_pressure_at_immersed_surface(this->cycle == 0 ? true : false);

    this->compute_internal_and_boundary_stress(this->cycle == 0 ? true : false);
    this->cycle++;
  }



  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::advance_newmark(
    const double physical_time)
  {
    AssertThrow(this->imposes_wall_displacement(),
                ExcMessage("Coupled Newmark steps impose the wall "
                           "displacement: set Pressure coupling = false."));
    if (this->cycle == 0)
      {
        this->current_time = physical_time;
        if (this->par.refinement_strategy == "inclusions")
          this->refine_and_transfer_around_inclusions();
        else
          for (unsigned int ref_cycle = 0;
               ref_cycle + 1 < this->par.n_refinement_cycles;
               ++ref_cycle)
            {
              this->assemble_elasticity_system();
              this->initialize_newmark_at_equilibrium();
              this->refine_and_transfer();
              install_tensor_wall_data();
            }
        // Mass, damping and Newmark matrices use the current `Time step`.
        this->assemble_elasticity_system();
        this->initialize_newmark_at_equilibrium();
        return;
      }

    const double time_step = this->par.time_parameters.time_step;
    AssertThrow(std::abs(physical_time - this->current_time - time_step) <=
                  1.e-8 * time_step,
                ExcMessage("A coupled Newmark step must advance the tissue by "
                           "the Time step " +
                           std::to_string(time_step) + ", from " +
                           std::to_string(this->current_time) + " to " +
                           std::to_string(physical_time) + "."));
    // Wall data and the affine part of the essential boundary data at the new
    // time; solve_newmark() sets up the constraints at both ends of the step.
    this->par.set_boundary_condition_times(physical_time);
    this->setup_constraints();
    this->compute_system_rhs(physical_time);
    this->solve_newmark();
    this->current_time = physical_time;
  }



  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::solve_static_timestep(
    const double physical_time)
  {
    const bool time_changed = physical_time != this->current_time;
    this->current_time      = physical_time;
    if (time_changed)
      {
        this->par.set_boundary_condition_times(physical_time);
        this->setup_constraints();
      }
    // Evaluate the wall data expression at the solve time. Rebuilding the
    // tensor-product coupling after refinement resets its time to zero, so
    // this runs before every coupling assembly.
    const auto set_wall_data_time = [&]() {
      if (!this->uses_tensor_product_coupling())
        this->inclusions.inclusions_rhs.set_time(physical_time);
#ifdef DEAL_II_WITH_VTK
      else
        this->tensor_product_coupling->set_time(physical_time);
#endif
    };
    if (this->cycle == 0) // at first timestep we refine
      {
        if (this->par.refinement_strategy == "inclusions")
          {
            this->refine_and_transfer_around_inclusions();
            // this->pcout << "refining around inclusions" << std::endl;

            this->assemble_elasticity_system(); // we need to refine A too
            set_wall_data_time();
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
                set_wall_data_time();
                this->assemble_coupling();
                this->solve();
                if (ref_cycle != this->par.n_refinement_cycles - 1)
                  {
                    this->refine_and_transfer();
                    install_tensor_wall_data();
                  }
              }
          }
      }
    else
      {
        // The mesh and the materials are those of the first step, so only
        // the load and the lifting of the new constraint values change.
        // Robin coefficients may depend on time and change the matrix.
        if (time_changed)
          this->assemble_elasticity_system(!this->par.robin_ids.empty());
        set_wall_data_time();
        this->assemble_coupling();
        this->solve();
      }
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
        if (new_data.empty())
          return;
        AssertThrow(new_data.size() == n_vessels(),
                    ExcMessage("Tensor-product coupling needs one radial wall "
                               "displacement per vessel, " +
                               std::to_string(n_vessels()) + ", but got " +
                               std::to_string(new_data.size()) + "."));
        wall_displacement_samples.clear();
        for (const double value : new_data)
          wall_displacement_samples.push_back({value});
        install_tensor_wall_data();
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
        // These checks also run in Release: a wrong size would read outside
        // new_data.
        AssertThrow(cells_per_vessel.size() == this->inclusions.get_n_vessels(),
                    ExcDimensionMismatch(cells_per_vessel.size(),
                                         this->inclusions.get_n_vessels()));
        std::vector<std::vector<double>> full_vector;
        std::size_t                      starting_point = 0;
        for (const int N1 : cells_per_vessel)
          {
            AssertThrow(N1 > 0,
                        ExcMessage("Each vessel needs at least one sample."));
            AssertThrow(starting_point + N1 <= new_data.size(),
                        ExcMessage("Fewer wall displacement samples than "
                                   "cells in the vessels."));
            full_vector.emplace_back(new_data.begin() + starting_point,
                                     new_data.begin() + starting_point + N1);
            starting_point += N1;
          }
        AssertThrow(starting_point == new_data.size(),
                    ExcMessage("More wall displacement samples than cells in "
                               "the vessels."));
        this->inclusions.update_inclusions_data(full_vector);
      }
    else
      {
        AssertThrow(cells_per_vessel.size() == n_vessels(),
                    ExcDimensionMismatch(cells_per_vessel.size(), n_vessels()));
        std::vector<std::vector<double>> samples;
        std::size_t                      next = 0;
        for (const int n : cells_per_vessel)
          {
            AssertThrow(n > 0,
                        ExcMessage("Each vessel needs at least one sample."));
            AssertThrow(next + n <= new_data.size(),
                        ExcMessage("Fewer wall displacement samples than "
                                   "cells in the vessels."));
            samples.emplace_back(new_data.begin() + next,
                                 new_data.begin() + next + n);
            next += n;
          }
        AssertThrow(next == new_data.size(),
                    ExcMessage("More wall displacement samples than cells in "
                               "the vessels."));
        wall_displacement_samples = std::move(samples);
        install_tensor_wall_data();
      }
  }

  template <int dim, int spacedim>
  std::vector<std::vector<double>>
  CoupledElasticityProblem<dim, spacedim>::split_pressure_over_dofs(
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
        AssertThrow(number_of_cells_per_vessel.size() == n_vessels(),
                    ExcDimensionMismatch(number_of_cells_per_vessel.size(),
                                         n_vessels()));
        AssertThrow(full_press.size() == n_pressure_entries,
                    ExcMessage("The pressure does not match the support "
                               "points of the last pressure computation."));
        // (arc length, pressure) of the support points of each vessel.
        std::vector<std::vector<std::pair<double, double>>> profile(
          n_vessels());
        for (unsigned int k = 0; k < pressure_sample_entry.size(); ++k)
          {
            AssertIndexRange(pressure_sample_vessel[k], n_vessels());
            profile[pressure_sample_vessel[k]].emplace_back(
              pressure_sample_arc_length[k],
              full_press[pressure_sample_entry[k]]);
          }
        for (unsigned int vessel = 0; vessel < n_vessels(); ++vessel)
          {
            auto     &points = profile[vessel];
            const int n      = number_of_cells_per_vessel[vessel];
            AssertThrow(n > 0,
                        ExcMessage("Each vessel must have at least one cell."));
            AssertThrow(!points.empty(),
                        ExcMessage("A vessel has no pressure samples."));
            std::sort(points.begin(), points.end());
            const auto value_at = [&points](const double s) {
              const auto next =
                std::upper_bound(points.begin(),
                                 points.end(),
                                 s,
                                 [](const double x, const auto &p) {
                                   return x < p.first;
                                 });
              if (next == points.begin())
                return points.front().second;
              if (next == points.end())
                return points.back().second;
              const auto   previous = std::prev(next);
              const double w =
                (s - previous->first) / (next->first - previous->first);
              return (1 - w) * previous->second + w * next->second;
            };
            std::vector<double> samples(n);
            if (points.size() == 1)
              std::fill(samples.begin(), samples.end(), points[0].second);
            else if (n == 1)
              {
                // Arc-length mean of the piecewise linear profile.
                double integral = 0.;
                for (unsigned int k = 0; k + 1 < points.size(); ++k)
                  integral += 0.5 * (points[k].second + points[k + 1].second) *
                              (points[k + 1].first - points[k].first);
                const double length =
                  points.back().first - points.front().first;
                samples[0] = length > 0. ? integral / length : points[0].second;
              }
            else
              for (int j = 0; j < n; ++j)
                samples[j] = value_at(static_cast<double>(j) / (n - 1));
            split_pressure.push_back(samples);
          }
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

        // Refine the inclusion cells and two layers of vertex neighbours
        // around them. Refine flags set on ghost cells are ignored by the
        // distributed triangulation, so the layers are built on the owned
        // cells: the marks are sent to the ghost cells before each layer and
        // every rank flags its own cells next to a marked cell.
        std::vector<char> marked(this->tria->n_active_cells(), 0);

        auto particle = this->inclusions.inclusions_as_particles.begin();
        while (particle != this->inclusions.inclusions_as_particles.end())
          {
            const auto &cell = particle->get_surrounding_cell();
            const auto  pic =
              this->inclusions.inclusions_as_particles.particles_in_cell(cell);
            Assert(pic.begin() == particle, ExcInternalError());

            marked[cell->active_cell_index()] = 1;
            particle                          = pic.end();
          }

        const auto vertex_to_cells = GridTools::vertex_to_cell_map(*this->tria);

        const auto add_neighbor_layer = [&]() {
          GridTools::exchange_cell_data_to_ghosts<char, DoFHandler<spacedim>>(
            this->dh,
            [&](const auto &cell) -> std::optional<char> {
              return marked[cell->active_cell_index()];
            },
            [&](const auto &cell, const char &value) {
              marked[cell->active_cell_index()] = value;
            });

          std::vector<char> next = marked;
          for (const auto &cell : this->tria->active_cell_iterators())
            if (cell->is_locally_owned() && !marked[cell->active_cell_index()])
              for (const auto vertex : cell->vertex_indices())
                {
                  bool near_marked = false;
                  for (const auto &neighbor :
                       vertex_to_cells[cell->vertex_index(vertex)])
                    near_marked =
                      near_marked || marked[neighbor->active_cell_index()];
                  if (near_marked)
                    {
                      next[cell->active_cell_index()] = 1;
                      break;
                    }
                }
          marked = std::move(next);
        };
        add_neighbor_layer();
        add_neighbor_layer();

        for (const auto &cell : this->tria->active_cell_iterators())
          if (cell->is_locally_owned() && marked[cell->active_cell_index()])
            {
              cell->set_refine_flag();
              cell->set_material_id(material_id);
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
