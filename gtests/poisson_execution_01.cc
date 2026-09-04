// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>

#include <cmath>
#include <filesystem>
#include <string>

#include "test_paths.h"

namespace
{
  std::string
  parameters_for(const std::string &section, const std::string &output)
  {
    return "subsection " + section +
           "\n"
           "  set FE degree = 1\n"
           "  set Initial refinement = 1\n"
           "  set Dirichlet boundary ids = 0, 1\n"
           "  set Output directory = " +
           output +
           "\n"
           "  set Output name = solution\n"
           "  subsection Grid generation\n"
           "    set Grid generator = hyper_cube\n"
           "    set Grid generator arguments = -1: 1: false\n"
           "    set Triangulation type = distributed\n"
           "  end\n"
           "  subsection Refinement and remeshing\n"
           "    set Number of refinement cycles = 1\n"
           "    set Strategy = global\n"
           "  end\n"
           "  subsection Right hand side\n"
           "    set Function expression = 1\n"
           "    set Variable names = x,y,z,t\n"
           "  end\n"
           "  subsection Dirichlet boundary conditions\n"
           "    set Function expression = 0\n"
           "    set Variable names = x,y,z,t\n"
           "  end\n"
           "  subsection Solver\n"
           "    subsection Control\n"
           "      set Max steps = 1000\n"
           "      set Reduction = 1.e-10\n"
           "      set Tolerance = 1.e-12\n"
           "      set Log result = false\n"
           "    end\n"
           "  end\n"
           "end\n";
  }
} // namespace

TEST(PoissonExecution, NativeAndAdapterEmbeddedOneDimensional)
{
  using namespace ImmersX;
  using Problem      = PoissonSolver<1, 3>;
  using FieldVector  = typename Problem::VectorType;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

  dealii::ParameterAcceptor::clear();
  const auto native_output  = TestPaths::output_directory("poisson-native");
  const auto adapter_output = TestPaths::output_directory("poisson-adapter");
  PoissonParameters<1, 3> native_parameters("/Native Poisson/");
  PoissonParameters<1, 3> adapter_parameters("/Adapter Poisson/");
  initialize_parameters_from_string(
    parameters_for("Native Poisson", native_output) +
    parameters_for("Adapter Poisson", adapter_output));

  Problem native_problem(native_parameters);
  native_problem.run();

  Problem adapter_problem(adapter_parameters);
  adapter_problem.make_grid();
  adapter_problem.setup_fe();
  adapter_problem.setup_system();
  adapter_problem.assemble_system();

  LinearSolverParameters options;
  options.solver             = LinearSolver::iterative;
  options.preconditioner     = LinearPreconditioner::block_diagonal;
  options.maximum_iterations = 1000;
  options.tolerance          = 1.e-12;
  Adapter    adapter(options, MPI_COMM_WORLD);
  const auto fields = adapter.add(adapter_problem, "poisson");
  auto       state  = adapter.make_state();
  adapter.solve(state);
  adapter_problem.set_solution(adapter.field(state, fields.fields().solution));

  FieldVector difference;
  difference.reinit(native_problem.solution());
  difference = native_problem.solution();
  difference -= adapter_problem.solution();
  EXPECT_LT(difference.l2_norm(), 1.e-9);
  EXPECT_TRUE(native_problem.solution_is_finite());
  EXPECT_TRUE(std::isfinite(adapter_problem.solution().l2_norm()));
  EXPECT_TRUE(std::filesystem::exists(native_output + "/solution.pvd"));
}
