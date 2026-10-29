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
#include <immersx/physics/coupled_elasticity.h>

namespace ImmersX
{
  template <int dim, int spacedim>
  CoupledElasticityProblem<dim, spacedim>::CoupledElasticityProblem(
    const ElasticityProblemParameters<dim, spacedim> &par)
    : ElasticityProblem<dim, spacedim>(par)
  {}

  /**
   * @brief compute tissue pressure ($\Lambda$) over the vessels and output to a .txt file (sequential) or .h5 file (mpi)
   *
   * @param openfilefisrttime internal variable to open file
   */

  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::compute_coupling_pressure()
  {
    TimerOutput::Scope t(this->computing_timer,
                         "Postprocessing: Computing Pressure");

    if (this->inclusions.n_inclusions() > 0
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

        coupling_pressure.reinit(locally_owned_vessels, this->mpi_communicator);
        auto &pressure = coupling_pressure;
        pressure       = 0;
        coupling_pressure_at_inclusions.reinit(locally_owned_inclusions,
                                               this->mpi_communicator);
        auto &pressure_at_inc = coupling_pressure_at_inclusions;
        pressure_at_inc       = 0;

        auto &lambda_to_pressure = this->locally_relevant_solution.block(1);

        const auto used_number_modes = this->inclusions.n_dofs_per_inclusion();

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
  }

  template <int dim, int spacedim>
  // TrilinosWrappers::MPI::Vector
  void
  CoupledElasticityProblem<dim, spacedim>::output_coupling_pressure(
    bool openfilefirsttime) const
  {
    TimerOutput::Scope t(this->computing_timer,
                         "Postprocessing: output Pressure");
    if (this->par.output_pressure == false ||
        this->inclusions.n_inclusions() == 0)
      {
        std::cout << "no output" << std::endl;
        return;
      }
    if (this->par.time_parameters.initial_time !=
        this->par.time_parameters.final_time)
      {
        this->pcout
          << "output_pressure file for time dependent simulation not implemented in MPI"
          << std::endl;
      }

    const auto &pressure = coupling_pressure;

    // print .txt only sequential
    if (Utilities::MPI::n_mpi_processes(this->mpi_communicator) == 1)
      {
        const std::string filename(this->par.output_directory +
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

        pressure_file << this->cycle << " ";
        pressure.print(pressure_file);
        pressure_file.close();
      }
    else
      // print .h5
      {
#ifdef DEAL_II_WITH_HDF5
        const std::string FILE_NAME(this->par.output_directory +
                                    "/externalPressure.h5");

        auto accessMode = HDF5::File::FileAccessMode::create;
        if (!openfilefirsttime)
          accessMode = HDF5::File::FileAccessMode::open;

        HDF5::File file_h5(FILE_NAME, accessMode, this->mpi_communicator);
        const std::string DATASET_NAME("externalPressure_" +
                                       std::to_string(this->cycle));

        HDF5::DataSet dataset =
          file_h5.create_dataset<double>(DATASET_NAME,
                                         {this->inclusions.get_n_vessels()});

        std::vector<double> data_to_write;
        // std::vector<hsize_t> coordinates;
        data_to_write.reserve(pressure.locally_owned_size());
        // coordinates.reserve(pressure.locally_owned_size());
        const auto locally_owned_vessels =
          Utilities::MPI::create_evenly_distributed_partitioning(
            this->mpi_communicator, this->inclusions.get_n_vessels());

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
    {
      TimerOutput::Scope t(this->computing_timer, "Setup inclusion");
      this->inclusions.setup_inclusions_particles(*this->tria);
    }
    this->cycle = 0;
    this->setup_dofs();
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

    coupling_pressure.clear();
    coupling_pressure_at_inclusions.clear();
    // coupling_pressure =
    // output_pressure(cycle == 0 ? true : false);
    compute_coupling_pressure();
    output_coupling_pressure(this->cycle == 0 ? true : false);

    this->compute_internal_and_boundary_stress(this->cycle == 0 ? true : false);
    this->cycle++;
  }

  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::update_inclusions_data(
    std::vector<double> new_data)
  {
    this->inclusions.update_inclusions_data(new_data);
  }

  template <int dim, int spacedim>
  void
  CoupledElasticityProblem<dim, spacedim>::update_inclusions_data(
    std::vector<double> new_data,
    std::vector<int>    cells_per_vessel)
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

  template <int dim, int spacedim>
  std::vector<std::vector<double>>
  CoupledElasticityProblem<dim, spacedim>::split_pressure_over_inclusions(
    std::vector<int> number_of_cells_per_vessel,
    Vector<double>   full_press) const
  {
    Assert(number_of_cells_per_vessel.size() ==
             this->inclusions.get_n_vessels(),
           ExcInternalError());

    std::vector<std::vector<double>> split_pressure;
    unsigned                         starting_inclusion = 0;

    for (unsigned int vessel = 0; vessel < number_of_cells_per_vessel.size();
         ++vessel)
      {
        auto N2 = number_of_cells_per_vessel[vessel];
        auto N1 = this->inclusions.get_inclusions_in_vessel(vessel);

        std::vector<double> new_vector;

        // const std::vector<double> pressure_of_inc_in_vessel(pressure);//
        // (*pressure[starting_inclusion], *pressure[starting_inclusion+N1]);
        AssertThrow(
          N1 >= 2 && N2 >= 2,
          ExcMessage(
            "Pressure interpolation needs at least two samples per vessel."));
        AssertDimension(full_press.size(), this->inclusions.n_inclusions());
        const auto &pressure_of_inc_in_vessel = full_press;

        new_vector.push_back(pressure_of_inc_in_vessel[starting_inclusion]);

        for (auto cell = 1; cell < N2 - 1; ++cell)
          {
            const double X = static_cast<double>(cell) / (N2 - 1) * (N1 - 1);
            const auto   j = static_cast<unsigned int>(std::floor(X));
            const double w = X - j;

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
  CoupledElasticityProblem<dim,
                           spacedim>::refine_and_transfer_around_inclusions()
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
                           GridTools::find_cells_adjacent_to_vertex(this->dh,
                                                                    vertey))
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

  // Template instantiations
  template class CoupledElasticityProblem<2>;
  template class CoupledElasticityProblem<2, 3>; // dim != spacedim
  template class CoupledElasticityProblem<3>;

} // namespace ImmersX
