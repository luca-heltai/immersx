// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/physics/metric_flow_x.h>

#  include <iostream>
#  include <stdexcept>
#  include <string>

namespace
{
  void
  run_metric_flow_x(const std::string &parameter_file)
  {
    using Problem      = MetricFlowX::BloodFlowSystem<1, 3>;
    using FieldVector  = MetricFlowX::VectorType;
    using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
    using Adapter      = ImmersX::IDAAdapter<FieldVector, GlobalVector>;

    Problem problem(MPI_COMM_WORLD);
    problem.initialize_params(parameter_file);
    problem.setup();

    Adapter::AdditionalData data;
    data.initial_time                  = 0.;
    data.final_time                    = 1.e-4;
    data.initial_step_size             = 1.e-5;
    data.maximum_order                 = 1;
    data.maximum_non_linear_iterations = 10;
    data.absolute_tolerance            = 1.e-6;
    data.relative_tolerance            = 1.e-5;
    data.ic_type                       = Adapter::AdditionalData::use_y_diff;
    data.reset_type                    = Adapter::AdditionalData::none;

    Adapter    adapter(data, MPI_COMM_WORLD);
    const auto fields =
      adapter.add(ImmersX::metric_flow_x(problem), "blood-flow");

    auto state     = adapter.make_state();
    auto state_dot = adapter.make_state();
    problem.initialize_state(adapter.field(state, fields.fields().state),
                             data.initial_time);
    problem.initialize_state_derivative(adapter.field(state_dot,
                                                      fields.fields().state),
                                        data.initial_time);

    adapter.set_compute_consistent_initial_conditions(
      [&problem, &adapter, fields](const double  time,
                                   GlobalVector &state,
                                   GlobalVector &state_dot) {
        problem.initialize_state_derivative(
          adapter.field(state_dot, fields.fields().state), time);
        (void)state;
      });

    adapter.solve(state, state_dot);
  }
} // namespace

int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                                  argv,
                                                                  1);
      if (argc != 2)
        throw std::invalid_argument("usage: metric_flow_x PARAMETER_FILE");

      run_metric_flow_x(argv[1]);
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
