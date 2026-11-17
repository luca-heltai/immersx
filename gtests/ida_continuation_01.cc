// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on deal.II.
// ---------------------------------------------------------------------
#include <deal.II/base/scope_exit.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <cmath>
#include <sstream>

#ifdef DEAL_II_WITH_SUNDIALS
TEST(DistributedIDA, BOTH_NativeConvergenceFailureRetriesFromCheckpoint)
{
  using namespace ImmersX;
  using Adapter =
    IDAAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector>;
  TimeIntervalParameters time;
  time.final_time = 0.;
  IDAParameters parameters;
  parameters.correction_type_at_initial_time = "use_y_dot";
  parameters.initial_residual_continuation   = true;
  parameters.initial_continuation_step       = 1.;
  parameters.absolute_tolerance = parameters.relative_tolerance = 1.e-8;
  Adapter          adapter(time, parameters, MPI_COMM_WORLD);
  dealii::IndexSet owned(1);
  const auto rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  if (rank == 0)
    owned.add_index(0);
  owned.compress();
  adapter.add([&](auto &builder) {
    return builder.differential_field("nonlinear", owned);
  });
  auto state = adapter.make_state(), derivative = adapter.make_state(),
       diagonal       = adapter.make_state();
  state               = 0.;
  derivative          = 0.;
  unsigned int resets = 0, outputs = 0;
  bool         first_residual = false;
  adapter.set_output_step(
    [&](double t, const auto &y, const auto &, unsigned int) {
      ++outputs;
      EXPECT_EQ(t, 0.);
      for (const auto i : owned)
        EXPECT_NEAR(y.block(0)[i], std::log(2.), 1.e-7);
    });
  auto &solver    = adapter.solver();
  solver.residual = [&](double, const auto &y, const auto &, auto &residual) {
    for (const auto i : owned)
      {
        if (resets == 2 && first_residual)
          {
            EXPECT_EQ(y.block(0)[i], 0.);
          }
        residual.block(0)[i] = std::exp(y.block(0)[i]) - 2.;
      }
    residual.compress(dealii::VectorOperation::insert);
    first_residual = false;
  };
  solver.setup_jacobian = [&](double, const auto &y, const auto &, double) {
    for (const auto i : owned)
      diagonal.block(0)[i] = std::exp(y.block(0)[i]);
    diagonal.compress(dealii::VectorOperation::insert);
  };
  solver.solve_with_jacobian = [&](const auto &rhs, auto &dst, double) {
    dst = rhs;
    for (const auto i : owned)
      dst.block(0)[i] = rhs.block(0)[i] / diagonal.block(0)[i];
    dst.compress(dealii::VectorOperation::insert);
  };
  const auto original_setup = solver.custom_setup;
  solver.custom_setup       = [&](void *memory) {
    if (original_setup)
      original_setup(memory);
    ++resets;
    first_residual = true;
    // A real native IC iteration limit forces the first nonlinear solve to
    // fail. Later resets permit convergence; no callback exception is injected.
    EXPECT_EQ(IDASetMaxNumItersIC(memory, resets == 1 ? 1 : 50), IDA_SUCCESS);
    EXPECT_EQ(IDASetMaxNumJacsIC(memory,
                                 resets == 1 ?
                                         1 :
                                         parameters.maximum_number_of_jacobians_ic),
              IDA_SUCCESS);
    EXPECT_EQ(IDASetMaxNumStepsIC(memory,
                                  resets == 1 ?
                                          1 :
                                          parameters.maximum_number_of_steps_ic),
              IDA_SUCCESS);
  };
  std::ostringstream      log;
  auto                   *original_stream = std::cout.rdbuf(log.rdbuf());
  const dealii::ScopeExit restore([&] { std::cout.rdbuf(original_stream); });
  EXPECT_EQ(adapter.solve(state, derivative), 0u);
  EXPECT_EQ(outputs, 1u);
  EXPECT_EQ(resets, 4u);
  if (rank == 0)
    {
      EXPECT_NE(log.str().find("rejected eta=1 retry_step=0.5"),
                std::string::npos);
    }
  auto residual = adapter.make_state();
  solver.residual(0., state, derivative, residual);
  EXPECT_LT(residual.linfty_norm(), 1.e-7);
}
#endif
