// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_lib.h>
#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/physics/elastic_static.h>

#include <cmath>
#include <string>
#include <vector>

namespace
{
  template <int dim, int spacedim>
  std::string
  parameter_text(const std::string &section)
  {
    return "subsection " + section +
           "\n"
           "  set Initial refinement = 1\n"
           "  set Dirichlet boundary ids = 0, 1, 2, 3, 4, 5\n"
           "  subsection Grid generation\n"
           "    set Grid generator = hyper_cube\n"
           "    set Grid generator arguments = 0: 1: true\n"
           "    set Triangulation type = distributed\n"
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

  template <int dim, int spacedim>
  void
  check_static_problem(const bool use_distributed_backend)
  {
    using namespace ImmersX;

    dealii::ParameterAcceptor::clear();
    ElasticStaticParameters<dim, spacedim> parameters;
    initialize_parameters_from_string(
      parameter_text<dim, spacedim>("Elastic static"));
    if (!use_distributed_backend)
      parameters.triangulation_type = "fullydistributed";

    ElasticStaticProblem<dim, spacedim> problem(parameters);
    problem.setup();
    EXPECT_GT(problem.dof_handler().n_dofs(), 0U);
    EXPECT_EQ(problem.dof_handler().get_fe().n_components(),
              static_cast<unsigned int>(spacedim));

    problem.set_forcing(dealii::Functions::ConstantFunction<spacedim>(
      std::vector<double>(spacedim, 1.)));
    problem.solve();

    EXPECT_TRUE(std::isfinite(problem.solution().l2_norm()));
    EXPECT_GT(problem.solution().l2_norm(), 0.);

    typename ElasticStaticProblem<dim, spacedim>::VectorType residual;
    residual.reinit(problem.solution());
    problem.stiffness_operator().vmult(residual, problem.solution());
    residual -= problem.forcing();
    problem.constraints().set_zero(residual);
    EXPECT_LT(residual.l2_norm(), 1.e-9);
  }

  template <int dim, int spacedim>
  void
  check_parity()
  {
    using namespace ImmersX;
    using Problem      = ElasticStaticProblem<dim, spacedim>;
    using FieldVector  = typename Problem::VectorType;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

    dealii::ParameterAcceptor::clear();
    ElasticStaticParameters<dim, spacedim> native_parameters("/Native static/");
    ElasticStaticParameters<dim, spacedim> adapter_parameters(
      "/Adapter static/");
    initialize_parameters_from_string(
      parameter_text<dim, spacedim>("Native static") +
      parameter_text<dim, spacedim>("Adapter static"));

    Problem native_problem(native_parameters);
    Problem adapter_problem(adapter_parameters);
    native_problem.setup();
    adapter_problem.setup();

    const dealii::Functions::ConstantFunction<spacedim> body_force(
      std::vector<double>(spacedim, 1.));
    native_problem.set_forcing(body_force);
    adapter_problem.set_forcing(body_force);

    native_problem.solve();

    LinearSolverParameters options;
    options.solver             = LinearSolver::iterative;
    options.preconditioner     = LinearPreconditioner::block_diagonal;
    options.maximum_iterations = 1000;
    options.tolerance          = 1.e-12;
    Adapter    adapter(options, MPI_COMM_WORLD);
    const auto fields = adapter.add(adapter_problem, "elastic-static");
    auto       state  = adapter.make_state();
    adapter.solve(state);
    adapter_problem.set_solution(
      adapter.field(state, fields.fields().displacement));

    FieldVector difference;
    difference.reinit(native_problem.solution());
    difference = native_problem.solution();
    difference -= adapter_problem.solution();
    EXPECT_LT(difference.l2_norm(), 1.e-9);

    const auto physical_residual = [](const Problem &problem) {
      FieldVector residual;
      residual.reinit(problem.solution());
      problem.stiffness_operator().vmult(residual, problem.solution());
      residual -= problem.forcing();
      problem.constraints().set_zero(residual);
      return residual.l2_norm();
    };
    EXPECT_LT(physical_residual(native_problem), 1.e-9);
    EXPECT_LT(physical_residual(adapter_problem), 1.e-9);
  }
} // namespace

TEST(ElasticStaticDimensions, AllSupportedCombinations)
{
  check_static_problem<1, 1>(false);
  check_static_problem<1, 2>(false);
  check_static_problem<1, 3>(false);
  check_static_problem<2, 2>(false);
  check_static_problem<2, 3>(false);
  check_static_problem<3, 3>(false);
}

TEST(ElasticStaticDimensions, MPI_AllSupportedCombinations)
{
  check_static_problem<1, 1>(true);
  check_static_problem<1, 2>(true);
  check_static_problem<1, 3>(true);
  check_static_problem<2, 2>(true);
  check_static_problem<2, 3>(true);
  check_static_problem<3, 3>(true);
}

TEST(ElasticStaticExecution, NativeAndAdapterParity)
{
  check_parity<1, 3>();
  check_parity<2, 3>();
  check_parity<3, 3>();
}
