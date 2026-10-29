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
   * See the class documentation in coupled_elasticity.h and
   * doc/concepts/vessel-pressure.md for the derivation of every formula used
   * here.
   */
  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim,
                           spacedim>::compute_pressure_at_immersed_surface()
  {
    TimerOutput::Scope t(this->computing_timer,
                         "Postprocessing: Computing Pressure");

    if (!this->uses_tensor_product_coupling() &&
        this->inclusions.n_inclusions() > 0)
      {
        const auto locally_owned_vessels =
          Utilities::MPI::create_evenly_distributed_partitioning(
            this->mpi_communicator, this->inclusions.get_n_vessels());

        const auto locally_owned_inclusions =
          Utilities::MPI::create_evenly_distributed_partitioning(
            this->mpi_communicator, this->inclusions.n_inclusions());

        pressure_at_immersed_surface.reinit(locally_owned_vessels,
                                            this->mpi_communicator);
        auto &pressure = pressure_at_immersed_surface;
        pressure       = 0;
        pressure_at_immersed_surface_by_inclusion.reinit(
          locally_owned_inclusions, this->mpi_communicator);
        auto &pressure_at_inc = pressure_at_immersed_surface_by_inclusion;
        pressure_at_inc       = 0;

        auto &lambda_to_pressure = this->locally_relevant_solution.block(1);

        const auto used_number_modes = this->inclusions.n_dofs_per_inclusion();

        const auto local_lambda = lambda_to_pressure.locally_owned_elements();

        // Only the coefficients of cos(theta) in the first vessel-frame
        // direction and of sin(theta) in the second one exert a normal force
        // on the wall: (mode 1, x) and (mode 2, y), i.e., the indices d and
        // 2d + 1 (3 and 7 in 3D, 2 and 5 in 2D). The normal force of every
        // other coefficient averages to zero over the wall. Find where these
        // two coefficients are in the selection; a coefficient that is not
        // selected contributes nothing.
        const auto        &selection = this->inclusions.get_selected_coefficients();
        std::vector<bool> is_normal_coefficient(used_number_modes, false);
        for (unsigned int j = 0; j < used_number_modes; ++j)
          is_normal_coefficient[j] = (selection[j] == spacedim ||
                                      selection[j] == 2 * spacedim + 1);

        // Area of the lateral surface of each vessel, |Gamma_v| = sum of
        // |S_k| over its inclusions, with |S_k| = 2 pi r_k |d_k| in 3D and
        // 2 pi r_k in 2D. Every rank knows all inclusions.
        std::vector<double> vessel_area(this->inclusions.get_n_vessels(), 0.);
        for (unsigned int k = 0; k < this->inclusions.n_inclusions(); ++k)
          vessel_area[this->inclusions.get_vesselID(k)] +=
            this->inclusions.get_section_measure(k);

        // The material also fills the vessels, so the multiplier gives the
        // jump of the normal stress across the wall, tissue minus interior.
        // The interior is a disc with the imposed wall displacement on its
        // boundary. In plane strain its mean normal stress on the wall
        // depends only on the mean normal wall displacement u_n,
        //   sigma_int = 2 (lambda + mu) u_n / r,
        // and adding it back gives the normal stress of the tissue. u_n is
        // read from the imposed data: an entry g of the multiplier
        // right-hand side for the coefficient 3 or 7 contributes g to u_n of
        // its block (the mean of cos^2 or sin^2 is 1/2, as for lambda). With
        // segments the data of a block is the sum over its n_s inclusions,
        // and each inclusion gets the mean g / n_s. lambda + mu is the mean
        // over the cells that contain the support points of the inclusion.
        const auto &imposed_displacement = this->force_rhs.block(1);
        std::vector<double> interior_stiffness;
        if (this->par.remove_interior_stress_from_pressure)
          {
            std::vector<double> local_stiffness(
              this->inclusions.n_inclusions(), 0.);
            std::vector<double> local_n_points(this->inclusions.n_inclusions(),
                                               0.);
            for (const auto &particle : this->inclusions.inclusions_as_particles)
              {
                const auto  k = this->inclusions.get_inclusion_id(
                  particle.get_id());
                const auto &mp = this->par.get_material_properties(
                  particle.get_surrounding_cell()->material_id());
                local_stiffness[k] += 2. * (mp.Lame_lambda + mp.Lame_mu);
                local_n_points[k] += 1.;
              }
            interior_stiffness.resize(this->inclusions.n_inclusions());
            std::vector<double> n_points(this->inclusions.n_inclusions());
            Utilities::MPI::sum(local_stiffness,
                                this->mpi_communicator,
                                interior_stiffness);
            Utilities::MPI::sum(local_n_points,
                                this->mpi_communicator,
                                n_points);
            const auto &default_material =
              this->par.get_material_properties(numbers::invalid_material_id);
            for (unsigned int k = 0; k < interior_stiffness.size(); ++k)
              interior_stiffness[k] =
                n_points[k] > 0 ?
                  interior_stiffness[k] / n_points[k] :
                  2. * (default_material.Lame_lambda +
                        default_material.Lame_mu);
          }

        // A normal coefficient lambda of the multiplier block used by
        // inclusion k is a force: on the wall of k it produces the normal
        // force lambda / 2 (the mean of cos^2 or sin^2 over the circle).
        // Dividing by |S_k| gives the mean normal stress on the wall of k,
        //   p_k = (lambda_1x + lambda_2y) / (2 |S_k|) + sigma_int,k,
        // and dividing the force by |Gamma_v| gives the contribution to the
        // mean normal stress on the wall of vessel v,
        //   p_v = sum_k ((lambda_1x + lambda_2y) / 2 + |S_k| sigma_int,k)
        //         / |Gamma_v|.
        // `displacement` is the contribution of this entry to u_n of the
        // inclusion.
        const auto add_normal_force = [&](const unsigned int inclusion,
                                          const double       lambda,
                                          const double       displacement) {
          AssertIndexRange(inclusion, this->inclusions.n_inclusions());
          const auto vessel_id = this->inclusions.get_vesselID(inclusion);
          const auto measure = this->inclusions.get_section_measure(inclusion);
          auto interior_stress = 0.0;
          if (this->par.remove_interior_stress_from_pressure)
            interior_stress = 
              interior_stiffness[inclusion] * displacement /
                this->inclusions.get_radius(inclusion);
          pressure_at_inc[inclusion] +=
            lambda / (2. * measure) + interior_stress;
          pressure[vessel_id] +=
            (lambda / 2. + measure * interior_stress) / vessel_area[vessel_id];
        };

        if (!this->inclusions.cluster_with_segments)
          {
            // Multiplier block k belongs to inclusion k.
            for (const auto &element_of_local_lambda : local_lambda)
              {
                if (!is_normal_coefficient[element_of_local_lambda %
                                           used_number_modes])
                  continue;
                const unsigned inclusion_number = (unsigned int)floor(
                  element_of_local_lambda / (used_number_modes));
                add_normal_force(
                  inclusion_number,
                  lambda_to_pressure[element_of_local_lambda],
                  imposed_displacement[element_of_local_lambda]);
              }
          }
        else
          {
            std::vector<unsigned int> n_cells_concurring_to_each_vessel(
              this->inclusions.get_n_vessels(), 0);
            [[maybe_unused]] auto expected_dofs =
              this->inclusions.n_global_segments() *
              this->inclusions.n_dofs_per_inclusion();
            AssertDimension(expected_dofs, this->solution.block(1).size());

            // Multiplier block s belongs to segment s. The segment constraint
            // is the sum of the constraints of its inclusions, so the segment
            // multiplier acts on each inclusion of the segment as if that
            // inclusion had it as its own multiplier: every inclusion of the
            // segment receives the whole block, with its own |S_k|. If the
            // inclusions of a segment are made shorter and more numerous, the
            // segment multiplier decreases in proportion and p_k does not
            // change.
            //
            // The entries of one segment are consecutive, so look up its
            // inclusions only when the segment changes: the lookup scans all
            // inclusions.
            unsigned int current_segment = numbers::invalid_unsigned_int;
            std::vector<unsigned int> inclusions_in_segment;
            for (const auto &element_of_local_lambda : local_lambda)
              {
                const unsigned segment_number = (unsigned int)floor(
                  element_of_local_lambda / (used_number_modes));

                AssertIndexRange(segment_number,
                                 this->inclusions.n_global_segments());
                if (!is_normal_coefficient[element_of_local_lambda %
                                           used_number_modes])
                  continue;
                if (segment_number != current_segment)
                  {
                    current_segment = segment_number;
                    inclusions_in_segment =
                      this->inclusions.get_inclusions_in_a_segment(
                        segment_number);
                  }

                for (const auto i : inclusions_in_segment)
                  {
                    n_cells_concurring_to_each_vessel
                      [this->inclusions.get_vesselID(i)]++;
                    add_normal_force(
                      i,
                      lambda_to_pressure[element_of_local_lambda],
                      imposed_displacement[element_of_local_lambda] /
                        inclusions_in_segment.size());
                  }
              }
          }
        // A segment is owned by the rank that owns its cell, while both
        // vectors are split evenly by vessel or inclusion, so some of the
        // additions above go to entries owned by other ranks.
        pressure.compress(VectorOperation::add);
        pressure_at_inc.compress(VectorOperation::add);
      }
    else if (this->uses_tensor_product_coupling()) // tensor coupling
      {
#ifdef DEAL_II_WITH_VTK
        // block(1) is indexed by the DoFs of the reduced DoFHandler, so lambda
        // is a finite element field on the reduced grid: component c of
        // lambda is the coefficient Lambda_c(s) of the selected cross-section
        // mode phi_c, and the force per unit wall area is
        //   T(s, xi) = sum_c Lambda_c(s) phi_c(xi).
        const auto &reduced_dh =
          this->tensor_product_coupling->get_dof_handler();
        const auto        &lambda = this->locally_relevant_solution.block(1);
        const auto        &fe     = reduced_dh.get_fe();
        const unsigned int n_comp = fe.n_components();

        AssertThrow(!vessel_index_of_a_cell.empty(),
                    ExcMessage("error in filling the vessels_id"));

        const auto locally_owned_vessels =
          Utilities::MPI::create_evenly_distributed_partitioning(
            this->mpi_communicator, n_vessels());

        pressure_at_immersed_surface.reinit(locally_owned_vessels,
                                            this->mpi_communicator);
        auto &pressure = pressure_at_immersed_surface;
        pressure       = 0;

        // Weight of each component in the mean normal stress on the wall,
        //   p(s) = 1/|D(s)| int_{D(s)} T . n = sum_c k_c Lambda_c(s),
        //   k_c  = 1/|D_ref| int_{D_ref} phi_c . n.
        // Only the modes x e_x and y e_y, i.e., the selected indices 3 and 7,
        // exert a normal force; their weight is 1/sqrt(2) on the circle,
        // where they are sqrt(2) cos(theta) e_x and sqrt(2) sin(theta) e_y.
        // The weight is computed with the quadrature of the reference cross
        // section, so it is exact for the quadrature used by the coupling.
        const auto &section =
          this->tensor_product_coupling->get_reference_cross_section();
        const auto &selection = this->par.tensor_product_coupling_parameters
                                  .tensor_product_space_parameters.section
                                  .selected_coefficients;
        AssertDimension(selection.size(), n_comp);
        const auto         &section_quadrature = section.get_global_quadrature();
        std::vector<double> normal_weight(n_comp, 0.);
        for (unsigned int c = 0; c < n_comp; ++c)
          if (selection[c] == 3 || selection[c] == 7)
            {
              for (unsigned int q = 0; q < section_quadrature.size(); ++q)
                {
                  const auto &xi = section_quadrature.point(q);
                  const auto  n  = xi / xi.norm();
                  for (unsigned int d = 0; d < spacedim; ++d)
                    normal_weight[c] += section.shape_value(c, q, d) * n[d] *
                                        section_quadrature.weight(q);
                }
              normal_weight[c] /= section.measure();
            }

        // Interior stress (see the point-coupling branch above). The imposed
        // wall displacement is sum_c a_c(s) phi_c with M a = g, where g is the
        // multiplier right-hand side and M the coupling mass matrix
        // int psi_m psi_n |D(s)| ds, i.e., the Gram matrix of the multiplier
        // space, since int_D phi_c . phi_c = |D|. Its mean normal value is
        //   u_n(s) = sum_c k_c a_c(s),
        // and the mean normal stress of the interior material on the wall is
        //   sigma_int(s) = 2 (lambda + mu) u_n(s) / r(s),
        // with r(s) the thickness. lambda + mu is the mean over the bulk cells
        // that contain the lifted points of each reduced cell.
        LA::MPI::Vector     imposed_coefficients;
        std::vector<double> interior_stiffness;
        if (this->par.remove_interior_stress_from_pressure)
          {
            LA::MPI::Vector owned_coefficients(
              this->force_rhs.block(1).locally_owned_elements(),
              this->mpi_communicator);
            ReductionControl          control(1000, 1e-14, 1e-12);
            SolverCG<LA::MPI::Vector> cg(control);
            cg.solve(this->inclusion_matrix,
                     owned_coefficients,
                     this->force_rhs.block(1),
                     this->prec_M);
            imposed_coefficients = lambda;
            imposed_coefficients = owned_coefficients;

            const auto n_reduced_cells = this->tensor_product_coupling
                                           ->get_triangulation()
                                           .n_global_active_cells();
            std::vector<double> local_stiffness(n_reduced_cells, 0.);
            std::vector<double> local_n_points(n_reduced_cells, 0.);
            for (const auto &particle :
                 this->tensor_product_coupling->get_particles())
              {
                const auto reduced_cell = std::get<0>(
                  this->tensor_product_coupling
                    ->particle_id_to_representative_indices(particle.get_id()));
                const auto &mp = this->par.get_material_properties(
                  particle.get_surrounding_cell()->material_id());
                local_stiffness[reduced_cell] +=
                  2. * (mp.Lame_lambda + mp.Lame_mu);
                local_n_points[reduced_cell] += 1.;
              }
            interior_stiffness.resize(n_reduced_cells);
            std::vector<double> n_points(n_reduced_cells);
            Utilities::MPI::sum(local_stiffness,
                                this->mpi_communicator,
                                interior_stiffness);
            Utilities::MPI::sum(local_n_points,
                                this->mpi_communicator,
                                n_points);
            const auto &default_material =
              this->par.get_material_properties(numbers::invalid_material_id);
            for (unsigned int c = 0; c < n_reduced_cells; ++c)
              interior_stiffness[c] =
                n_points[c] > 0 ?
                  interior_stiffness[c] / n_points[c] :
                  2. * (default_material.Lame_lambda +
                        default_material.Lame_mu);
          }

        // Pressure at each support point of the reduced grid: the nodal
        // value p_n = sum_c k_c (lambda_{n,c} + 2 (lambda + mu) a_{n,c} / r_n)
        // of the pressure field p_h(s); the second term is the interior
        // stress, if it is removed.
        // Support point n is the n-th degree of freedom of component 0 in
        // the numbering of the reduced DoFHandler. Its other components are
        // on the same support point, so they are owned by the same rank.
        const IndexSet owned_dofs_of_first_component = DoFTools::extract_dofs(
          reduced_dh, fe.component_mask(FEValuesExtractors::Scalar(0)));
        const auto [first_support_point, n_support_points] =
          Utilities::MPI::partial_and_total_sum(
            owned_dofs_of_first_component.n_elements(),
            this->mpi_communicator);
        IndexSet owned_support_points(n_support_points);
        owned_support_points.add_range(
          first_support_point,
          first_support_point + owned_dofs_of_first_component.n_elements());
        pressure_at_immersed_surface_by_inclusion.reinit(owned_support_points,
                                                         this->mpi_communicator);
        auto &pressure_at_inc = pressure_at_immersed_surface_by_inclusion;
        pressure_at_inc       = 0;

        // The thickness at the support points, for the interior stress; a
        // support point shared by two cells takes the material of the last
        // one visited.
        const Quadrature<1> support_points(
          fe.base_element(0).get_unit_support_points());
        std::vector<double> support_point_thickness(support_points.size());

        std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
        for (const auto &cell : reduced_dh.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(dof_indices);
              if (this->par.remove_interior_stress_from_pressure)
                this->tensor_product_coupling->get_thickness_values(
                  cell, support_points, support_point_thickness);
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                {
                  const auto [component, base_index] =
                    fe.system_to_component_index(i);
                  if (component != 0 ||
                      !owned_dofs_of_first_component.is_element(dof_indices[i]))
                    continue;
                  double nodal_pressure = 0;
                  for (unsigned int c = 0; c < n_comp; ++c)
                    if (normal_weight[c] != 0)
                      {
                        const auto dof = dof_indices[fe.component_to_system_index(
                          c, base_index)];
                        nodal_pressure += normal_weight[c] * lambda[dof];
                        if (this->par.remove_interior_stress_from_pressure)
                          nodal_pressure +=
                            interior_stiffness[cell->global_active_cell_index()] *
                            normal_weight[c] * imposed_coefficients[dof] /
                            support_point_thickness[base_index];
                      }
                  pressure_at_inc[first_support_point +
                                  owned_dofs_of_first_component
                                    .index_within_set(dof_indices[i])] =
                    nodal_pressure;
                }
            }
        pressure_at_inc.compress(VectorOperation::insert);

        // Pressure of vessel v: the mean of p_h over its centerline,
        //   p_v = 1/|gamma_v| int_{gamma_v} p_h(s) ds,
        // which is a weighted mean of the nodal values p_n. With a constant
        // radius this is also the mean normal stress over the wall of v.
        FEValues<1, spacedim> fe_values(
          fe,
          this->tensor_product_coupling->get_quadrature(),
          update_values | update_JxW_values);
        std::vector<Vector<double>> lambda_values(fe_values.n_quadrature_points,
                                                  Vector<double>(n_comp));
        std::vector<Vector<double>> coefficient_values(
          fe_values.n_quadrature_points, Vector<double>(n_comp));
        std::vector<double> thickness(fe_values.n_quadrature_points);

        std::vector<double> local_force(n_vessels(), 0.);
        std::vector<double> local_length(n_vessels(), 0.);
        for (const auto &cell : reduced_dh.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto vessel_id =
                vessel_index_of_a_cell[cell->active_cell_index()];
              fe_values.reinit(cell);
              fe_values.get_function_values(lambda, lambda_values);
              if (this->par.remove_interior_stress_from_pressure)
                {
                  fe_values.get_function_values(imposed_coefficients,
                                                coefficient_values);
                  this->tensor_product_coupling->get_thickness_values(
                    cell,
                    this->tensor_product_coupling->get_quadrature(),
                    thickness);
                }
              for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                {
                  for (unsigned int c = 0; c < n_comp; ++c)
                    {
                      local_force[vessel_id] += normal_weight[c] *
                                                lambda_values[q][c] *
                                                fe_values.JxW(q);
                      if (this->par.remove_interior_stress_from_pressure)
                        local_force[vessel_id] +=
                          interior_stiffness[cell->global_active_cell_index()] *
                          normal_weight[c] * coefficient_values[q][c] /
                          thickness[q] * fe_values.JxW(q);
                    }
                  local_length[vessel_id] += fe_values.JxW(q);
                }
            }
        std::vector<double> force(n_vessels(), 0.);
        std::vector<double> length(n_vessels(), 0.);
        Utilities::MPI::sum(local_force, this->mpi_communicator, force);
        Utilities::MPI::sum(local_length, this->mpi_communicator, length);

        for (const auto v : pressure.locally_owned_elements())
          {
            AssertThrow(length[v] > 0,
                        ExcMessage("Vessel " + std::to_string(v) +
                                   " has no cells in the reduced grid."));
            pressure[v] = force[v] / length[v];
          }
        pressure.compress(VectorOperation::insert);
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
        if (pressure.locally_owned_size() > 0)
          {
            hsize_t prefix = 0;
            hsize_t los    = pressure.locally_owned_size();
            int     ierr   = MPI_Exscan(&los,
                                  &prefix,
                                  1,
                                  MPI_UNSIGNED_LONG_LONG,
                                  MPI_SUM,
                                  this->mpi_communicator);
            AssertThrowMPI(ierr);

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

    this->cycle = 0;
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
        this->assemble_coupling();
        this->solve();
      }


    // if (this->par.output_results)
    this->output_results();

    pressure_at_immersed_surface.clear();
    pressure_at_immersed_surface_by_inclusion.clear();
    // coupling_pressure =
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
    Vector<double> /* full_press */) const
  {
    std::vector<std::vector<double>> split_pressure;

    if (!this->uses_tensor_product_coupling())
      {
        AssertThrow(number_of_cells_per_vessel.size() ==
                      this->inclusions.get_n_vessels(),
                    ExcDimensionMismatch(number_of_cells_per_vessel.size(),
                                         this->inclusions.get_n_vessels()));

        // Copying the distributed vector into a serial one is collective, so
        // every rank must call this function.
        const Vector<double> pressure_by_inclusion(
          pressure_at_immersed_surface_by_inclusion);

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
                    const unsigned int j =
                      std::min<unsigned int>(std::floor(X), N1 - 2);
                    const double w = X - j;
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
        Vector<float>      error_per_cell(this->tria->n_active_cells());
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
