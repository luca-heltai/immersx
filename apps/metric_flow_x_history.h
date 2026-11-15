// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of the ImmersX application, based on deal.II.
// ---------------------------------------------------------------------
#ifndef immersx_app_metric_flow_x_history_h
#define immersx_app_metric_flow_x_history_h

#include <deal.II/base/mpi.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace ImmersX::metric_flow_x_app_detail
{
  // The pressure provider must still be attached while fields are written.
  // Detach it only for the layout checkpoint; replay reads saved pressure,
  // rather than reconstructing this callable from the layout.
  template <class Flow, class Vector>
  auto
  write_flow_history(Flow              &flow,
                     const std::string &directory,
                     const double       time,
                     const Vector      &state,
                     const Vector      &derivative,
                     const bool         first,
                     int)
    -> decltype(flow.save_coupling_fields(directory,
                                          time,
                                          state,
                                          derivative,
                                          {"area", "velocity", "pressure"}),
                void())
  {
    using namespace dealii;
    const auto                  comm = flow.mpi_communicator();
    const auto                  rank = Utilities::MPI::this_mpi_process(comm);
    const std::filesystem::path root(directory);
    std::string                 error;
    if (first && rank == 0)
      try
        {
          AssertThrow(!std::filesystem::exists(root),
                      ExcMessage("Flow history destination already exists."));
          std::filesystem::create_directories(root);
        }
      catch (const std::exception &exception)
        {
          error = exception.what();
        }
    error = Utilities::MPI::broadcast(comm, error, 0);
    AssertThrow(error.empty(), ExcMessage(error));

    std::ostringstream name;
    name << "fields-" << std::hexfloat << time;
    flow.save_coupling_fields((root / name.str()).string(),
                              time,
                              state,
                              derivative,
                              {"area", "velocity", "pressure"});
    if (first)
      {
        flow.clear_external_pressure_provider();
        flow.save_checkpoint((root / "layout").string(),
                             time,
                             state,
                             derivative);
      }
    if (rank == 0)
      try
        {
          std::ofstream index(root / "coupling_fields.csv", std::ios::app);
          if (first)
            index << "time_s,directory\n";
          index << std::setprecision(17) << time << ',' << name.str() << '\n';
          index.close();
          AssertThrow(index.good(),
                      ExcMessage("Cannot write flow history index."));
        }
      catch (const std::exception &exception)
        {
          error = exception.what();
        }
    error = Utilities::MPI::broadcast(comm, error, 0);
    AssertThrow(error.empty(), ExcMessage(error));
  }

  template <class Flow, class Vector>
  void
  write_flow_history(Flow &,
                     const std::string &,
                     const double,
                     const Vector &,
                     const Vector &,
                     const bool,
                     long)
  {
    AssertThrow(
      false,
      dealii::ExcMessage(
        "Flow history requires MetricFlowX with save_coupling_fields()."));
  }
} // namespace ImmersX::metric_flow_x_app_detail
#endif
