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
 * Modified by: Luca Heltai, 2020, Camilla Belponer 2023
 */

// 3D tissue coupled to the legacy 1D blood flow model.
//
// Usage:
//   coupled_elasticity parameters.prm [input_1d.pot [sampling start [mode]]]
// Without a 1D input file only the tissue is solved. With it, the tissue
// receives the wall displacement of the 1D model every `sampling` 1D steps
// once the 1D time exceeds `start`, and returns the external pressure
// -10 p_v (Pa to dyn/cm^2) for the next 1D step. With start >= 100 only the
// 1D model runs. mode 0 exchanges one value per vessel, mode 1 one value per
// 1D cell; mode 1 needs tensor-product coupling or point coupling without
// clustering.
//
// The parameter file selects the tissue model: a static solve at each
// exchange when Initial time = Final time, or when every material has zero
// Density; otherwise one Newmark step of the damped elastodynamics per
// exchange, whose time step is `sampling` times the 1D time step.

#include <immersx/io/utils.h>
#include <immersx/physics/vessel_elasticity.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef ENABLE_COUPLED_PROBLEMS
#  include <immersx/fvm/coupledModel1d.h>
#endif

using namespace dealii;
using namespace ImmersX;

namespace
{
  // Solve the tissue alone: one static solve, or steps from the initial time
  // to the final time. A dynamic tissue takes Newmark steps from the
  // equilibrium at the initial time; a quasi-static tissue takes a static
  // solve at each time.
  void
  run_tissue(const std::string &prm_file)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Running uncoupled 3D problem" << std::endl;
    ElasticityProblemParameters<3> par;
    VesselElasticityProblem<3>     problem3D(par);
    initialize_parameters(prm_file);
    problem3D.run_timestep0();
    problem3D.run_timestep();
    if (par.time_mode == TimeMode::Dynamic ||
        par.time_mode == TimeMode::QuasiStatic)
      {
        const auto  &time = par.time_parameters;
        unsigned int step = 1;
        for (; time.initial_time + step * time.time_step <=
               time.final_time + 1.e-12 * time.time_step;
             ++step)
          problem3D.run_timestep(time.initial_time + step * time.time_step);
      }
  }

#ifdef ENABLE_COUPLED_PROBLEMS
  // Same unit conversion for both exchange modes.
  constexpr double Pa_to_dyn_conversion = 10.;

  // Run `action` on rank zero, which alone owns the 1D model. An exception
  // there is sent to all ranks and rethrown on each of them, so that no rank
  // waits for rank zero in the next collective call. Collective.
  template <typename Action>
  void
  on_rank_zero(Action &&action)
  {
    std::string error;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        try
          {
            action();
          }
        catch (const std::exception &exception)
          {
            error = exception.what();
          }
        catch (...)
          {
            error = "unknown exception";
          }
        if (error.empty() == false)
          error = "On rank zero: " + error;
      }
    error = Utilities::MPI::broadcast(MPI_COMM_WORLD, error);
    AssertThrow(error.empty(), ExcMessage(error));
  }

  // Send the external pressure -10 p to the 1D model, either one value per
  // vessel or one value per 1D cell, and copy it to the junctions.
  void
  send_external_pressure(CoupledModel1d                   &pb1D,
                         const VesselElasticityProblem<3> &problem3D,
                         const unsigned int                coupling_mode,
                         const std::vector<int>           &cells_per_vessel)
  {
    AssertThrow(coupling_mode == 0 || coupling_mode == 1,
                ExcMessage("Unknown coupling mode."));
    // Gathering the distributed pressure is collective.
    const Vector<double> pressure(
      coupling_mode == 0 ? problem3D.pressure_at_immersed_surface :
                           problem3D.pressure_at_immersed_surface_by_support_point);

    on_rank_zero([&]() {
      if (coupling_mode == 0)
        {
          AssertThrow(pressure.size() == static_cast<unsigned int>(pb1D.NV),
                      ExcDimensionMismatch(pressure.size(), pb1D.NV));
          for (int i = 0; i < pb1D.NV; ++i)
            for (int j = 0; j < pb1D.vess[i].NCELLS; ++j)
              pb1D.vess[i].setpeconst(j, -Pa_to_dyn_conversion * pressure[i]);
        }
      else
        {
          const auto pointwise =
            problem3D.split_pressure_over_dofs(cells_per_vessel, pressure);
          AssertThrow(pointwise.size() == static_cast<unsigned int>(pb1D.NV),
                      ExcDimensionMismatch(pointwise.size(), pb1D.NV));
          for (int i = 0; i < pb1D.NV; ++i)
            {
              AssertThrow(pointwise[i].size() ==
                            static_cast<unsigned int>(pb1D.vess[i].NCELLS),
                          ExcDimensionMismatch(pointwise[i].size(),
                                               pb1D.vess[i].NCELLS));
              for (int j = 0; j < pb1D.vess[i].NCELLS; ++j)
                pb1D.vess[i].setpeconst(j,
                                        -Pa_to_dyn_conversion *
                                          pointwise[i][j]);
            }
        }

      for (int i = 0; i < pb1D.NV; ++i)
        {
          if (pb1D.vess[i].bcType[0] > 1)
            pb1D.junctionsData[pb1D.vess[i].iJuncL]
              .peJ[pb1D.vess[i].iJuncLorder] = pb1D.vess[i].getpe(0, 0);
          if (pb1D.vess[i].bcType[1] > 1)
            pb1D.junctionsData[pb1D.vess[i].iJuncR]
              .peJ[pb1D.vess[i].iJuncRorder] =
              pb1D.vess[i].getpe(pb1D.vess[i].NCELLS - 1,
                                 pb1D.vess[i].nDOFs - 1);
        }
    });
  }

  void
  run_coupled(const std::string &prm_file,
              const std::string &input_file_name,
              const unsigned int coupling_sampling,
              const double       coupling_start,
              unsigned int       coupling_mode)
  {
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);
#  ifdef USE_MPI
    AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
                ExcMessage(
                  "The MPI-enabled legacy 1dsolver backend uses MPI_COMM_WORLD "
                  "but this driver executes its 1D model only on rank zero. "
                  "Use one MPI rank for this backend, or supply a matching "
                  "serial backend with IMMERSX_FVM_USE_MPI=OFF for distributed "
                  "tissue solves."));
#  endif
    AssertThrow(coupling_sampling > 0,
                ExcMessage("The coupling sampling must be positive."));

    TimerOutput timer(MPI_COMM_WORLD,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times);

    ElasticityProblemParameters<3> par;
    VesselElasticityProblem<3>     problem3D(par);
    initialize_parameters(prm_file);
    if (coupling_mode == 1 && !problem3D.uses_tensor_product_coupling() &&
        problem3D.inclusions.cluster_with_segments)
      {
        coupling_mode = 0;
        pcout << "coupling_mode = 1 is incompatible with Cluster inclusions "
                 "with segments = true, coupling_mode set to 0"
              << std::endl;
      }

    CoupledModel1d   pb1D;
    std::vector<int> cells_per_vessel;
    pb1D.partitionID = 0;
    pb1D.nproc       = 1;
    pb1D.verbose     = 0;
    on_rank_zero([&]() {
      pb1D.init(input_file_name);
      for (int i = 0; i < pb1D.NV; ++i)
        cells_per_vessel.push_back(pb1D.vess[i].NCELLS);
    });
    cells_per_vessel =
      Utilities::MPI::broadcast(MPI_COMM_WORLD, cells_per_vessel);
    const double t_end = Utilities::MPI::broadcast(MPI_COMM_WORLD, pb1D.tEnd);
    const double t_ini = Utilities::MPI::broadcast(MPI_COMM_WORLD, pb1D.tIni);
    const double dt =
      Utilities::MPI::broadcast(MPI_COMM_WORLD, pb1D.dtMaxLTSLIMIT);

    const bool dynamic = par.time_mode == TimeMode::Dynamic;
    if (dynamic)
      {
        // One Newmark step per exchange.
        const double newmark_step = coupling_sampling * dt;
        if (par.time_parameters.time_step != newmark_step)
          pcout << "Newmark time step set to " << newmark_step << " ("
                << coupling_sampling << " x 1D time step), replacing "
                << par.time_parameters.time_step << std::endl;
        par.time_parameters.time_step = newmark_step;
      }
    pcout << "Running coupled problem with parameters: Sampling = "
          << coupling_sampling << " Start = " << coupling_start
          << " mode = " << coupling_mode
          << ", tissue model = " << (dynamic ? "Newmark" : "static")
          << std::endl;

    problem3D.run_timestep0();
    AssertThrow(cells_per_vessel.size() == problem3D.n_vessels(),
                ExcDimensionMismatch(cells_per_vessel.size(),
                                     problem3D.n_vessels()));

    pb1D.iT = 0;
    for (unsigned int iter = 0; iter * dt < t_end - t_ini; ++iter)
      {
        timer.enter_subsection("1 timestep 1d");
        on_rank_zero([&]() { pb1D.solveTimeStep(pb1D.dtMaxLTSLIMIT); });
        timer.leave_subsection();

        // The time reached by the 1D step, without accumulated rounding, so
        // that consecutive exchanges are exactly one Newmark step apart.
        const double time = t_ini + (iter + 1) * dt;
        if (iter * dt > coupling_start && iter % coupling_sampling == 0)
          {
            on_rank_zero([&]() {
              pb1D.compute_new_displacement_for_coupling(coupling_mode);
            });
            const std::vector<double> displacement =
              Utilities::MPI::broadcast(MPI_COMM_WORLD, pb1D.new_displacement);
            if (coupling_mode == 0)
              problem3D.update_vessel_radius_data(displacement);
            else
              problem3D.update_vessel_radius_data(displacement,
                                                  cells_per_vessel);

            timer.enter_subsection("3 3d timestep");
            problem3D.run_timestep(time);
            timer.leave_subsection();

            send_external_pressure(pb1D,
                                   problem3D,
                                   coupling_mode,
                                   cells_per_vessel);
          }
        pb1D.iT += 1;
      }
    on_rank_zero([&]() { pb1D.end(); });
  }

  void
  run_1d(const std::string &input_file_name)
  {
    if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
      {
        std::cout << " !! to Run uncoupled 1D problem please use OMP "
                     "parallelizazion instead of MPI"
                  << std::endl;
        return;
      }
    Timer timer; // creating a timer also starts it
    std::cout << "Running uncoupled 1D problem" << std::endl;

    CoupledModel1d pb1D;
    pb1D.partitionID = 0;
    pb1D.nproc       = 1;
    pb1D.verbose     = 0;
    pb1D.init(input_file_name);

    pb1D.iT         = 0; // not simple iteration counter !
    double timestep = 0.0;
    while (timestep < pb1D.tEnd)
      {
        pb1D.solveTimeStep(pb1D.dtMaxLTSLIMIT);
        pb1D.iT += 1;
        timestep += pb1D.dtMaxLTSLIMIT;
      }
    pb1D.end();

    timer.stop();
    std::cout << "Elapsed wall time: " << timer.wall_time() << " seconds.\n";
  }
#endif
} // namespace

int
main(int argc, char *argv[])
{
  deallog.depth_console(1);
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      const std::string prm_file = argc > 1 ? argv[1] : "parameters.prm";

#ifdef ENABLE_COUPLED_PROBLEMS
      if (argc > 2)
        {
          const unsigned int sampling =
            argc > 4 ? std::strtoul(argv[3], nullptr, 10) : 1;
          const double start = argc > 4 ? std::strtod(argv[4], nullptr) : 9.;
          const unsigned int mode =
            argc > 5 ? std::strtoul(argv[5], nullptr, 10) : 0;
          if (start < 100)
            run_coupled(prm_file, argv[2], sampling, start, mode);
          else
            run_1d(argv[2]);
        }
      else
        run_tissue(prm_file);
#else
      run_tissue(prm_file);
#endif
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
