// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <immersx/io/utils.h>
#  include <immersx/physics/metric_flow_x_elastostatic_replay.h>

#  include <iostream>
#  include <string>

int
main(int argc, char *argv[])
{
  try
    {
      AssertThrow(argc == 2,
                  dealii::ExcMessage(
                    "usage: metric_flow_x_elastostatic_replay PARAMETER_FILE"));
      const std::string parameter_file = argv[1];
      // Keep application options out of PETSc's command-line database.
      int    runtime_argc        = 1;
      char  *runtime_arguments[] = {argv[0], nullptr};
      char **runtime_argv        = runtime_arguments;
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(runtime_argc,
                                                                  runtime_argv,
                                                                  1);
      dealii::ParameterAcceptor::clear();
      ImmersX::MetricFlowXElastostaticReplayParameters parameters;
      ImmersX::initialize_parameters(parameter_file);
      ImmersX::MetricFlowXElastostaticReplay replay(parameters);
      replay.run();
    }
  catch (const std::exception &exc)
    {
      std::cerr << "Exception on processing: " << exc.what() << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << "Unknown exception!" << std::endl;
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

#endif // IMMERSX_WITH_METRIC_FLOW_X
