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
#ifdef ENABLE_COUPLED_PROBLEMS

#  include <iomanip>

#  if 1
#    include "coupled_elasticity.h"
#  endif
#  include "coupledModel1d.h"
#  include "utils.h"

int
main(int argc, char *argv[])
{
  using namespace dealii;
  deallog.depth_console(1);
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      std::string                      prm_file;
      std::string                      input_file_name;
      unsigned int                     couplingSampling     = 1;
      unsigned int                     Pa_to_dyn_conversion = 10;
      unsigned int                     couplingStart        = 9;
      unsigned int                     coupling_mode        = 0;

      ConditionalOStream pcout(
        std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      if (argc > 1)
        prm_file = argv[1];
      else
        prm_file = "parameters.prm";

      if (argc > 2)
        {
          input_file_name = argv[2];
          if (argc > 4)
            {
              couplingSampling = std::strtol(argv[3], NULL, 10);
              couplingStart    = std::strtol(argv[4], NULL, 10);
              if (argc > 5)
                coupling_mode = std::strtol(argv[5], NULL, 10);
              // mode 0 = constant pressure over vessels
              // mode 1 = coupling at inclusions/cells level
            }

          if (couplingStart < 100)
            {
              if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                {
                  std::cout
                    << "Running coupled problem with parameters: Sampling = "
                    << couplingSampling << " Start = " << couplingStart
                    << " mode (DEPRECATED) = " << coupling_mode << std::endl;
                }
              TimerOutput timer(MPI_COMM_WORLD,
                                pcout,
                                TimerOutput::summary,
                                TimerOutput::wall_times);

              ElasticityProblemParameters<3> par;
              CoupledElasticityProblem<3>    problem3D(par);
              initialize_parameters(prm_file);
              problem3D.run_timestep0();

              {
                CoupledModel1d pb1D;
                int            id = 0, p = 1;
                // define process ID and number of processes
                pb1D.partitionID = id;
                pb1D.nproc       = p;
                // initialize model
                pb1D.verbose = 0;

                std::vector<int> number_of_cells_per_vessel;
                if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                  {
                    pb1D.init(input_file_name);
                    AssertDimension(problem3D.n_vessels(), pb1D.NV);

                    for (int i = 0; i < pb1D.NV; i++)
                      number_of_cells_per_vessel.push_back(pb1D.vess[i].NCELLS);
                  }
                // else
                //   number_of_cells_per_vessel.resize(problem3D.n_vessels());

                MPI_Barrier(MPI_COMM_WORLD);
                number_of_cells_per_vessel =
                  Utilities::MPI::broadcast(MPI_COMM_WORLD,
                                            number_of_cells_per_vessel);

                // enter time loop
                pb1D.iT         = 0; // not simple iteration counter !
                int    iter     = 0;
                double timestep = 0.0;
                double tEnd =
                  Utilities::MPI::broadcast(MPI_COMM_WORLD, pb1D.tEnd);
                double tIni =
                  Utilities::MPI::broadcast(MPI_COMM_WORLD, pb1D.tIni);
                double dt =
                  Utilities::MPI::broadcast(MPI_COMM_WORLD, pb1D.dtMaxLTSLIMIT);
                MPI_Barrier(MPI_COMM_WORLD);

                while (timestep < (tEnd - tIni))
                  {
                    timer.enter_subsection("1 timestep 1d");
                    // solve time step
                    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                      {
                        pb1D.solveTimeStep(pb1D.dtMaxLTSLIMIT);
                      }
                    MPI_Barrier(MPI_COMM_WORLD);
                    timer.leave_subsection();
                    // every iterSampling we aso perform the 3D
                    if (timestep > couplingStart &&
                        iter % couplingSampling == 0)
                      {
                        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                            0)
                          {
                            pb1D.compute_new_displacement_for_coupling(
                              coupling_mode);
                          }
                        MPI_Barrier(MPI_COMM_WORLD);

                        std::vector<double> new_displacement_data =
                          Utilities::MPI::broadcast(MPI_COMM_WORLD,
                                                    pb1D.new_displacement);

                        if (coupling_mode == 0)
                          {
                            problem3D.update_inclusions_data(
                              new_displacement_data);
                          }
                        else // if (coupling_mode == 1)
                          {
                            problem3D.update_inclusions_data(
                              new_displacement_data,
                              number_of_cells_per_vessel);
                          }
                        timer.enter_subsection("3 3d timestep");
                        problem3D.run_timestep();
                        // problem3D.compute_coupling_pressure();

                        if (coupling_mode == 0)
                          {
                            Vector<double> coupling_pressure(
                              problem3D.coupling_pressure);
                            coupling_pressure *= Pa_to_dyn_conversion;
                            coupling_pressure *= -1;


                            if (Utilities::MPI::this_mpi_process(
                                  MPI_COMM_WORLD) == 0)
                              {
                                AssertDimension(
                                  problem3D.coupling_pressure.size(), pb1D.NV);

                                for (int i = 0; i < pb1D.NV; i++)
                                  for (int j = 0; j < pb1D.vess[i].NCELLS; j++)
                                    {
                                      pb1D.vess[i].setpeconst(
                                        j, (coupling_pressure[i]));
                                    }
                              }
                          }
                        else // if (coupling_mode == 1)
                          {
                            Vector<double> pressure(
                              problem3D.coupling_pressure_at_inclusions);

                            if (Utilities::MPI::this_mpi_process(
                                  MPI_COMM_WORLD) == 0)
                              {
                                std::vector<std::vector<double>>
                                  coupling_pressure_pointwise =
                                    problem3D.split_pressure_over_inclusions(
                                      number_of_cells_per_vessel, pressure);

                                AssertDimension(
                                  coupling_pressure_pointwise.size(), pb1D.NV);

                                for (int i = 0; i < pb1D.NV; i++)
                                  for (int j = 0; j < pb1D.vess[i].NCELLS; j++)
                                    {
                                      auto pe =
                                        coupling_pressure_pointwise[i][j] *
                                        Pa_to_dyn_conversion;
                                      pe *= -1;
                                      pb1D.vess[i].setpeconst(j, pe);
                                    }
                              }
                          }
                        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                            0)
                          { // update the junction
                            for (int i = 0; i < pb1D.NV; i++)
                              {
                                if (pb1D.vess[i].bcType[0] > 1)
                                  {
                                    const auto iJunc = pb1D.vess[i].iJuncL;
                                    const auto iJuncOrder =
                                      pb1D.vess[i].iJuncLorder;
                                    pb1D.junctionsData[iJunc].peJ[iJuncOrder] =
                                      pb1D.vess[i].getpe(0, 0);
                                  }
                                if (pb1D.vess[i].bcType[1] > 1)
                                  {
                                    const auto iJunc = pb1D.vess[i].iJuncR;
                                    const auto iJuncOrder =
                                      pb1D.vess[i].iJuncRorder;
                                    pb1D.junctionsData[iJunc].peJ[iJuncOrder] =
                                      pb1D.vess[i].getpe(pb1D.vess[i].NCELLS -
                                                           1,
                                                         pb1D.vess[i].nDOFs -
                                                           1);
                                  }
                              }
                          }
                      }
                    iter++;
                    pb1D.iT += 1;
                    timestep += dt;
                  }
                if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                  {
                    pb1D.end();
                  }
              }
            }
          else // uncoupled 1d
            {
              if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
                {
                  std::cout
                    << " !! to Run uncoupled 1D problem please use OMP parallelizazion instead of MPI"
                    << std::endl;
                }
              else
                {
                  Timer timer; // creating a timer also starts it
                  std::cout << "Running uncoupled 1D problem" << std::endl;

                  CoupledModel1d pb1D;
                  int            id = 0, p = 1;
                  // define process ID and number of processes
                  pb1D.partitionID = id;
                  pb1D.nproc       = p;
                  // initialize model
                  pb1D.verbose = 0;

                  pb1D.init(input_file_name);

                  // enter time loop
                  pb1D.iT         = 0; // not simple iteration counter !
                  int    iter     = 0;
                  double timestep = 0.0;
                  double tEnd     = pb1D.tEnd;
                  double dt       = pb1D.dtMaxLTSLIMIT;
                  while (timestep < tEnd)
                    {
                      // solve time step
                      pb1D.solveTimeStep(pb1D.dtMaxLTSLIMIT);

                      iter++;
                      pb1D.iT += 1;
                      timestep += dt;
                    }
                  pb1D.end();

                  timer.stop();
                  std::cout << "Elapsed wall time: " << timer.wall_time()
                            << " seconds.\n";
                }
            }
        }
      else // run uncuopled problem
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Running uncoupled 3D problem" << std::endl;
          ElasticityProblemParameters<3> par;
          CoupledElasticityProblem<3>    problem3D(par);
          initialize_parameters(prm_file);
          problem3D.run_timestep0();
          problem3D.run_timestep();
        }
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
#else
int
main()
{
  return 0;
}
#endif
