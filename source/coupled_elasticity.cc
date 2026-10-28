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

  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::reassemble_coupling_rhs()
  {
    // Reuse the point-coupling assembly so the multiplier rhs has the same
    // scaling as the coupling matrix assembled at cycle 0. Matrices are kept.
    this->assemble_point_coupling(false);
  }

  /*
  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::reassemble_coupling_rhs()
  {
    TimerOutput::Scope t(this->computing_timer, "updating coupling rhs");

    this->system_rhs.block(1) = 0;

    std::vector<types::global_dof_index> fe_dof_indices(
      this->fe->n_dofs_per_cell());
    std::vector<types::global_dof_index> inclusion_dof_indices(
      this->inclusions.n_dofs_per_inclusion());

    Vector<double> local_rhs(this->inclusions.n_dofs_per_inclusion());

    auto particle = this->inclusions.inclusions_as_particles.begin();
    while (particle != this->inclusions.inclusions_as_particles.end())
      {
        const auto &cell = particle->get_surrounding_cell();
        const auto  dh_cell =
          typename DoFHandler<spacedim>::cell_iterator(*cell, &this->dh);
        dh_cell->get_dof_indices(fe_dof_indices);
        const auto pic =
          this->inclusions.inclusions_as_particles.particles_in_cell(cell);
        Assert(pic.begin() == particle, ExcInternalError());

        auto p      = pic.begin();
        auto next_p = pic.begin();
        while (p != pic.end())
          {
            const auto inclusion_id =
              this->inclusions.get_inclusion_id(p->get_id());
            inclusion_dof_indices =
              this->inclusions.get_dof_indices(p->get_id());
            local_rhs = 0;

            // Extract all points that refer to the same inclusion
            std::vector<Point<spacedim>> ref_q_points;
            for (; next_p != pic.end() && this->inclusions.get_inclusion_id(
                                            next_p->get_id()) == inclusion_id;
                 ++next_p)
              ref_q_points.push_back(next_p->get_reference_location());
            FEValues<spacedim, spacedim> fev(*this->fe,
                                             ref_q_points,
                                             update_values | update_gradients);
            fev.reinit(dh_cell);
            // double temp = 0;
            for (unsigned int q = 0; q < ref_q_points.size(); ++q)
              {
                const auto  id = p->get_id();
                const auto &inclusion_fe_values =
                  this->inclusions.get_fe_values(id);
                const auto &real_q = p->get_location();
                const auto  ds     = this->inclusions.get_JxW(
                  id); // /inclusions.get_radius(inclusion_id);

                // Coupling and inclusions matrix
                for (unsigned int j = 0;
                     j < this->inclusions.n_dofs_per_inclusion();
                     ++j)
                  {
                    if (this->inclusions.data_file != "")
                      {
                        if (this->inclusions.inclusions_data[inclusion_id]
                              .size() > j)
                          {
                            auto temp =
                              inclusion_fe_values[j] * inclusion_fe_values[j] *
                              this->inclusions.get_inclusion_data(inclusion_id,
                                                                  j) /
                              // inclusions.inclusions_data[inclusion_id][j] /
                              // // data is always prescribed in relative
                              // coordinates
                              this->inclusions.get_radius(inclusion_id) * ds;
                            if (this->par.time_parameters.initial_time !=
                                this->par.time_parameters.final_time)
                              temp *= this->inclusions.inclusions_rhs.value(
                                real_q, this->inclusions.get_component(j));
                            local_rhs(j) += temp;
                          }
                      }
                    else
                      {
                        local_rhs(j) +=
                          inclusion_fe_values[j] *
                          this->inclusions.inclusions_rhs.value(
                            real_q, this->inclusions.get_component(j)) /
                          this->inclusions.get_radius(inclusion_id) * ds;
                      }
                  }
                ++p;
              }
            // I expect p and next_p to be the same now.
            Assert(p == next_p, ExcInternalError());
            // Add local matrices to global ones
            this->inclusion_constraints.distribute_local_to_global(
              local_rhs, inclusion_dof_indices, this->system_rhs.block(1));
          }
        particle = pic.end();
      }
    this->system_rhs.compress(VectorOperation::insert);
  }
  */


  /**
   * @brief compute tissue pressure ($\Lambda$) over the vessels
   */

  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim,
                           spacedim>::compute_pressure_at_immersed_surface()
  {
    TimerOutput::Scope t(this->computing_timer,
                         "Postprocessing: Computing Pressure");

    if (!this->uses_tensor_product_coupling() &&
        this->inclusions.n_inclusions() > 0
        // &&
        // inclusions.get_offset_coefficients() == 1 &&
        // inclusions.n_coefficients >= 2
    )
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

        const auto used_number_modes = this->inclusions.get_n_coefficients();

        const auto local_lambda = lambda_to_pressure.locally_owned_elements();

        if constexpr (spacedim == 3)
          {
            for (const auto &element_of_local_lambda : local_lambda)
              {
                const unsigned inclusion_number = (unsigned int)floor(
                  element_of_local_lambda / (used_number_modes));

                AssertIndexRange(inclusion_number,
                                 this->inclusions.n_inclusions());
                pressure[this->inclusions.get_vesselID(inclusion_number)] +=
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

                AssertIndexRange(inclusion_number,
                                 this->inclusions.n_inclusions());
                pressure_at_inc[inclusion_number] +=
                  lambda_to_pressure[element_of_local_lambda];
              }
            pressure = pressure_at_inc;
            // pressure.compress(VectorOperation::add);
            // pressure_at_inc.compress(VectorOperation::add);
          }
      }
    else if (this->uses_tensor_product_coupling()) // tensor coupling
      {
#ifdef DEAL_II_WITH_VTK
        // block(1) is indexed by the DoFs of the reduced DoFHandler, so lambda
        // is a finite element field on the reduced grid: integrate each of its
        // components over that grid.
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


        FEValues<1, spacedim> fe_values(
          fe,
          this->tensor_product_coupling->get_quadrature(),
          update_values | update_JxW_values);
        std::vector<Vector<double>> lambda_values(fe_values.n_quadrature_points,
                                                  Vector<double>(n_comp));

        std::vector<double> local_integral(1, 0.0);
        for (const auto &cell : reduced_dh.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              fe_values.reinit(cell);
              fe_values.get_function_values(lambda, lambda_values);
              for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                for (unsigned int c = 0; c < n_comp; ++c)
                  pressure[vessel_index_of_a_cell[cell->active_cell_index()]] +=
                    lambda_values[q][c] * fe_values.JxW(q);
            }
        std::vector<double> integral(1, 0.0);
        Utilities::MPI::sum(make_array_view(std::as_const(local_integral)),
                            this->mpi_communicator,
                            make_array_view(integral));

        pressure.compress(VectorOperation::add);
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
        reassemble_coupling_rhs();
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
          AssertThrow(
          false,
          ExcMessage(
            "Cannot update data for tensor coupling"));

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
        AssertThrow(
          false,
          ExcMessage(
            "Cannot update data for tensor coupling"));

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
        Assert(number_of_cells_per_vessel.size() ==
                 this->inclusions.get_n_vessels(),
               ExcInternalError());

        unsigned starting_inclusion = 0;

        for (unsigned int vessel = 0;
             vessel < number_of_cells_per_vessel.size();
             ++vessel)
          {
            auto N2 = number_of_cells_per_vessel[vessel];
            auto N1 = this->inclusions.get_inclusions_in_vessel(vessel);

            std::vector<double> new_vector;

            // const std::vector<double> pressure_of_inc_in_vessel(pressure);//
            // (*pressure[starting_inclusion],
            // *pressure[starting_inclusion+N1]);
            const Vector<double> pressure_of_inc_in_vessel(
              pressure_at_immersed_surface_by_inclusion);

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
    else
      {
#ifdef DEAL_II_WITH_VTK
        AssertThrow(
          true,
          ExcMessage(
            "Cannot update data for tensor coupling"));

#endif
        AssertThrow(
          true,
          ExcMessage(
            "Trying to update vessel radius data: the code uses tensor_coupling but VTK flag is OFF"));
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
