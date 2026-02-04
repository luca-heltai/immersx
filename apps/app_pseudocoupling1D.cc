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
#include <iostream>

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
      unsigned int                     couplingSampling           = 1;
      unsigned int                     couplingStart              = 9;
      unsigned int                     pseudocoupling_coefficient = 1;
      ConditionalOStream               pcout(
        std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      if (argc > 5)
        {
          prm_file                   = argv[1];
          input_file_name            = argv[2];
          couplingSampling           = std::strtol(argv[3], NULL, 10);
          couplingStart              = std::strtol(argv[4], NULL, 10);
          pseudocoupling_coefficient = std::strtol(argv[5], NULL, 10);

          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
              std::cout
                << "Running PSEUDO coupled problem with parameters: Sampling = "
                << couplingSampling << " Start = " << couplingStart
                << std::endl;
            }
          TimerOutput timer(MPI_COMM_WORLD,
                            pcout,
                            TimerOutput::summary,
                            TimerOutput::wall_times);

          CoupledElasticityProblemParameters<3> par;
          CoupledElasticityProblem<3>           problem3D(par);
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

            if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
              pb1D.init(input_file_name);

            // enter time loop
            pb1D.iT         = 0; // not simple iteration counter !
            int    iter     = 0;
            double timestep = 0.0;
            double tEnd = Utilities::MPI::broadcast(MPI_COMM_WORLD, pb1D.tEnd);
            double tIni = Utilities::MPI::broadcast(MPI_COMM_WORLD, pb1D.tIni);
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
                if (timestep > couplingStart && iter % couplingSampling == 0)
                  {
                    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                      {
                        pb1D.compute_new_displacement_for_coupling(0);
                      }
                    MPI_Barrier(MPI_COMM_WORLD);

                    std::vector<double> new_displacement_data =
                      Utilities::MPI::broadcast(MPI_COMM_WORLD,
                                                pb1D.new_displacement);

                    problem3D.update_inclusions_data(new_displacement_data);
                    timer.enter_subsection("3 3d timestep");
                    problem3D.run_timestep();
                    timer.leave_subsection();

                    pb1D.solvePseudo3D(pseudocoupling_coefficient);

                    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
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
                                  pb1D.vess[i].getpe(pb1D.vess[i].NCELLS - 1,
                                                     pb1D.vess[i].nDOFs - 1);
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
      else
        {
          std::cout << "insufficient number of input parameters" << std::endl;
          return 1;
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
  std::cout << "This program can only be run with ENABLE_COUPLED_PROBLEMS"
            << std::endl;
  return 0;
}
#endif