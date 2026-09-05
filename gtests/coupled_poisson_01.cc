// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License or (at your option) any later version. The
// full text of the license can be found in the LICENSE.md file at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/numerics/data_out.h>

#include <gtest/gtest.h>
#include <immersx/core/constraint.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/linear_adapter.h>

#include <filesystem>
#include <vector>

using namespace ImmersX;
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>

#include <fstream>

#include "test_paths.h"


using namespace dealii;

namespace
{
  bool
  output_contains(const std::filesystem::path &directory,
                  const std::string           &stem,
                  const std::string           &text)
  {
    for (const auto &entry : std::filesystem::directory_iterator(directory))
      if (entry.path().filename().string().find(stem) != std::string::npos &&
          (entry.path().extension() == ".vtu" ||
           entry.path().extension() == ".pvtu"))
        {
          std::ifstream input(entry.path());
          std::string   contents((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
          if (contents.find(text) != std::string::npos)
            return true;
        }
    return false;
  }
} // namespace


TEST(CoupledPoisson, MPI_UnifiedConstraintSolve) // NOLINT
{
  ParameterAcceptor::clear();

  PoissonParameters<2>    bulk_parameters("/Bulk Poisson/");
  PoissonParameters<1, 2> embedded_parameters("/Embedded Poisson/");

  initialize_parameters_from_string(R"(
    subsection Bulk Poisson
      set FE degree                   = 1
      set Initial refinement          = 2
      set Dirichlet boundary ids      = 0
      subsection Grid generation
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Right hand side
        set Function expression = 0
        set Variable names      = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names      = x,y,t
      end
      subsection Solver
        subsection Control
          set Max steps  = 200
          set Reduction  = 1.e-12
          set Tolerance  = 1.e-14
          set Log result = false
        end
      end
    end
    subsection Embedded Poisson
      set FE degree                   = 1
      set Initial refinement          = 2
      set Dirichlet boundary ids      = 0,1
      subsection Grid generation
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Right hand side
        set Function expression = 1
        set Variable names      = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names      = x,y,t
      end
      subsection Solver
        subsection Control
          set Max steps  = 200
          set Reduction  = 1.e-12
          set Tolerance  = 1.e-14
          set Log result = false
        end
      end
    end
  )");

  const auto output_directory =
    ImmersX::TestPaths::output_directory("coupled-poisson-01");
  bulk_parameters.output_directory     = output_directory;
  embedded_parameters.output_directory = output_directory;
  bulk_parameters.output_name          = "coupled_bulk";
  embedded_parameters.output_name      = "coupled_embedded";

  PoissonSolver<2>    bulk_problem(bulk_parameters);
  PoissonSolver<1, 2> embedded_problem(embedded_parameters);

  // Keep this first proving ground fixed-mesh. Adaptivity and representation
  // transfer are deliberately outside this prototype.
  bulk_problem.make_grid();
  bulk_problem.setup_fe();
  bulk_problem.setup_system();
  bulk_problem.assemble_system();

  embedded_problem.make_grid();
  embedded_problem.setup_fe();
  embedded_problem.setup_system();
  embedded_problem.assemble_system();

  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = LinearAdapter<FieldVector, GlobalVector>;
  LinearSolverParameters linear_parameters;
  Adapter                adapter(linear_parameters, MPI_COMM_WORLD);
  const auto             bulk     = adapter.add(bulk_problem, "bulk");
  const auto             embedded = adapter.add(embedded_problem, "embedded");

  const auto       bulk_view     = fe_space(bulk_problem.dof_handler(),
                                  StaticMappingQ1<2>::mapping,
                                  bulk_problem.constraints(),
                                  bulk_problem.locally_relevant_dofs());
  const auto       embedded_view = fe_space(embedded_problem.dof_handler(),
                                      StaticMappingQ1<1, 2>::mapping,
                                      embedded_problem.constraints(),
                                      embedded_problem.locally_relevant_dofs());
  FE_DGQ<1, 2>     multiplier_fe(0);
  DoFHandler<1, 2> multiplier_dh(embedded_problem.triangulation());
  multiplier_dh.distribute_dofs(multiplier_fe);
  const auto multiplier_owned = multiplier_dh.locally_owned_dofs();
  const auto multiplier_relevant =
    DoFTools::extract_locally_relevant_dofs(multiplier_dh);
  AffineConstraints<double> multiplier_constraints;
  multiplier_constraints.reinit(multiplier_owned, multiplier_relevant);
  multiplier_constraints.close();
  const auto multiplier_view = fe_space(multiplier_dh,
                                        StaticMappingQ1<1, 2>::mapping,
                                        multiplier_constraints,
                                        multiplier_relevant);
  const auto bulk_field =
    bulk_view.field(bulk.fields().solution, "bulk_solution");
  const auto embedded_field =
    embedded_view.field(embedded.fields().solution, "embedded_solution");
  const auto lambda = multiplier_view.field("lambda");
  const auto constraint =
    make_constraint(weak_term(value(bulk_field), lambda) -
                    weak_term(value(embedded_field), lambda));
  const auto coupling = adapter.add(constraint, "continuity");

  auto state = adapter.make_state();
  adapter.solve(state);
  bulk_problem.set_solution(adapter.field(state, bulk.fields().solution));
  embedded_problem.set_solution(
    adapter.field(state, embedded.fields().solution));

  ASSERT_EQ(adapter.saddle_points().size(), 1u);
  EXPECT_EQ(adapter.saddle_points().front().participants.size(), 2u);
  EXPECT_EQ(adapter.field(state, coupling.fields().multiplier).size(),
            multiplier_dh.n_dofs());
  EXPECT_TRUE(std::isfinite(
    adapter.field(state, coupling.fields().multiplier).l2_norm()));
  EXPECT_TRUE(bulk_problem.solution_is_finite());
  EXPECT_TRUE(embedded_problem.solution_is_finite());
  EXPECT_GT(bulk_problem.solution_l2_norm(), 1.e-12);
  EXPECT_GT(embedded_problem.solution_l2_norm(), 1.e-12);
  GlobalVector residual;
  adapter.evaluate_residual(state, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-7);

  // The externally computed states remain valid Poisson states, including
  // ghost updates used by the existing output path.
  bulk_problem.output_results();
  embedded_problem.output_results();
}

TEST(CoupledPoisson, MPI_LinearAdapterComposesStandaloneProblems) // NOLINT
{
  ParameterAcceptor::clear();

  PoissonParameters<2>    bulk_parameters("/Adapter Bulk/");
  PoissonParameters<1, 2> embedded_parameters("/Adapter Embedded/");
  initialize_parameters_from_string(R"(
    subsection Adapter Bulk
      set FE degree = 1
      set Initial refinement = 2
      set Dirichlet boundary ids = 0
      subsection Grid generation
        set Grid generator = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Right hand side
        set Function expression = 0
        set Variable names = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names = x,y,t
      end
    end
    subsection Adapter Embedded
      set FE degree = 1
      set Initial refinement = 1
      set Dirichlet boundary ids = 0,1
      subsection Grid generation
        set Grid generator = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Right hand side
        set Function expression = 1
        set Variable names = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names = x,y,t
      end
    end
  )");

  const auto output_directory =
    ImmersX::TestPaths::output_directory("linear-adapter-poisson");
  bulk_parameters.output_directory     = output_directory;
  embedded_parameters.output_directory = output_directory;

  PoissonSolver<2>    bulk_problem(bulk_parameters);
  PoissonSolver<1, 2> embedded_problem(embedded_parameters);
  const auto          initialize_problem = [](auto &problem) {
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_system();
  };
  initialize_problem(bulk_problem);
  initialize_problem(embedded_problem);

  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = ImmersX::LinearAdapter<FieldVector, GlobalVector>;
  ImmersX::LinearSolverParameters linear_parameters;
  Adapter                         linear(linear_parameters, MPI_COMM_WORLD);
  const auto                      bulk = linear.add(bulk_problem, "bulk");
  const auto embedded = linear.add(embedded_problem, "embedded");

  const auto       bulk_view     = fe_space(bulk_problem.dof_handler(),
                                  StaticMappingQ1<2>::mapping,
                                  bulk_problem.constraints(),
                                  bulk_problem.locally_relevant_dofs());
  const auto       embedded_view = fe_space(embedded_problem.dof_handler(),
                                      StaticMappingQ1<1, 2>::mapping,
                                      embedded_problem.constraints(),
                                      embedded_problem.locally_relevant_dofs());
  FE_DGQ<1, 2>     multiplier_fe(0);
  DoFHandler<1, 2> multiplier_dh(embedded_problem.triangulation());
  multiplier_dh.distribute_dofs(multiplier_fe);
  const auto multiplier_owned = multiplier_dh.locally_owned_dofs();
  const auto multiplier_relevant =
    DoFTools::extract_locally_relevant_dofs(multiplier_dh);
  AffineConstraints<double> multiplier_constraints;
  multiplier_constraints.reinit(multiplier_owned, multiplier_relevant);
  multiplier_constraints.close();
  const auto multiplier_view = fe_space(multiplier_dh,
                                        StaticMappingQ1<1, 2>::mapping,
                                        multiplier_constraints,
                                        multiplier_relevant);
  const auto bulk_field =
    bulk_view.field(bulk.fields().solution, "bulk_solution");
  const auto embedded_field =
    embedded_view.field(embedded.fields().solution, "embedded_solution");
  const auto lambda = multiplier_view.field("lambda");
  const auto constraint =
    make_constraint(weak_term(value(bulk_field), lambda) -
                    weak_term(value(embedded_field), lambda));
  const auto coupling = linear.add(constraint, "continuity");
  auto       state    = linear.make_state();
  linear.solve(state);

  const auto &bulk_state     = linear.field(state, bulk.fields().solution);
  const auto &embedded_state = linear.field(state, embedded.fields().solution);
  bulk_problem.set_solution(bulk_state);
  embedded_problem.set_solution(embedded_state);

  const auto multiplier_output =
    ImmersX::TestPaths::output_directory("linear-adapter-multiplier");
  std::filesystem::create_directories(multiplier_output);
  const auto    rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  DataOut<1, 2> multiplier_data;
  multiplier_data.attach_dof_handler(multiplier_dh);
  multiplier_data.add_data_vector(linear.field(state,
                                               coupling.fields().multiplier),
                                  "scalar_multiplier",
                                  DataOut<1, 2>::type_dof_data);
  multiplier_data.build_patches();
  std::ofstream multiplier_file(
    std::filesystem::path(multiplier_output) /
    ("scalar_multiplier." + std::to_string(rank) + ".vtu"));
  multiplier_data.write_vtu(multiplier_file);
  MPI_Barrier(MPI_COMM_WORLD);
  EXPECT_TRUE(output_contains(multiplier_output,
                              "scalar_multiplier.",
                              "scalar_multiplier"));

  ImmersXLA::MPI::Vector bulk_difference;
  bulk_difference.reinit(bulk_problem.solution());
  bulk_difference = bulk_problem.solution();
  bulk_difference -= bulk_state;
  EXPECT_LT(bulk_difference.l2_norm(), 1.e-12);

  ImmersXLA::MPI::Vector embedded_difference;
  embedded_difference.reinit(embedded_problem.solution());
  embedded_difference = embedded_problem.solution();
  embedded_difference -= embedded_state;
  EXPECT_LT(embedded_difference.l2_norm(), 1.e-12);

  GlobalVector residual;
  linear.evaluate_residual(state, residual);
  EXPECT_TRUE(
    std::isfinite(linear.field(state, bulk.fields().solution).l2_norm()));
  EXPECT_TRUE(
    std::isfinite(linear.field(state, embedded.fields().solution).l2_norm()));
  EXPECT_TRUE(
    std::isfinite(linear.field(state, coupling.fields().multiplier).l2_norm()));
  EXPECT_GT(linear.field(state, embedded.fields().solution).l2_norm(), 1.e-12);
  EXPECT_LT(residual.l2_norm(), 1.e-7);

  ImmersX::LinearSolverParameters augmented_options;
  augmented_options.solver = ImmersX::LinearSolver::iterative;
  augmented_options.preconditioner =
    ImmersX::LinearPreconditioner::augmented_lagrangian;
  augmented_options.augmented_lagrangian_parameter = 2.;
  Adapter    augmented(augmented_options, MPI_COMM_WORLD);
  const auto augmented_bulk = augmented.add(bulk_problem, "bulk-al");
  const auto augmented_embedded =
    augmented.add(embedded_problem, "embedded-al");
  const auto augmented_bulk_field =
    bulk_view.field(augmented_bulk.fields().solution, "bulk_solution");
  const auto augmented_embedded_field =
    embedded_view.field(augmented_embedded.fields().solution,
                        "embedded_solution");
  const auto augmented_lambda = multiplier_view.field("lambda");
  augmented.add(make_constraint(
                  weak_term(value(augmented_bulk_field), augmented_lambda) -
                  weak_term(value(augmented_embedded_field), augmented_lambda)),
                "continuity-al");
  auto augmented_state = augmented.make_state();
  augmented.solve(augmented_state);
  GlobalVector augmented_residual;
  augmented.evaluate_residual(augmented_state, augmented_residual);
  EXPECT_LT(augmented_residual.l2_norm(), 1.e-7);

  ImmersX::LinearSolverParameters direct_options;
  direct_options.solver = ImmersX::LinearSolver::direct;
  Adapter    direct(direct_options, MPI_COMM_WORLD);
  const auto direct_bulk     = direct.add(bulk_problem, "bulk-direct");
  const auto direct_embedded = direct.add(embedded_problem, "embedded-direct");
  const auto direct_bulk_field =
    bulk_view.field(direct_bulk.fields().solution, "bulk_solution");
  const auto direct_embedded_field =
    embedded_view.field(direct_embedded.fields().solution, "embedded_solution");
  const auto direct_lambda = multiplier_view.field("lambda");
  direct.add(make_constraint(
               weak_term(value(direct_bulk_field), direct_lambda) -
               weak_term(value(direct_embedded_field), direct_lambda)),
             "continuity-direct");
  auto direct_state = direct.make_state();
  EXPECT_TRUE(direct.can_materialize_matrix(direct_state));
  const auto direct_matrix = direct.monolithic_matrix(direct_state);
  const auto expected_system_size =
    bulk_problem.n_dofs() + embedded_problem.n_dofs() + multiplier_dh.n_dofs();
  EXPECT_EQ(direct_matrix.m(), expected_system_size);
  EXPECT_EQ(direct_matrix.n(), expected_system_size);

  auto sample_state = direct.make_state();
  for (unsigned int block = 0; block < sample_state.n_blocks(); ++block)
    sample_state.block(block) = 1.;
  const auto             sample = direct.pack(sample_state);
  ImmersXLA::MPI::Vector matrix_action;
  matrix_action.reinit(sample);
  direct_matrix.vmult(matrix_action, sample);
  auto operator_action = direct.make_state();
  direct.jacobian(sample_state).vmult(operator_action, sample_state);
  matrix_action -= direct.pack(operator_action);
  EXPECT_LT(matrix_action.l2_norm(), 1.e-10);

  ASSERT_EQ(direct.saddle_points().size(), 1u);
  ASSERT_TRUE(
    direct.has_multiplier_metric(direct.saddle_points().front().multiplier));
  const auto augmented_matrix =
    direct.augmented_lagrangian_matrix(sample_state, 2.);
  auto augmented_matrix_action = direct.make_state();
  augmented_matrix.vmult(augmented_matrix_action, sample_state);
  const auto augmented_operator =
    direct.augmented_lagrangian_operator(sample_state, 2.);
  auto augmented_operator_action = direct.make_state();
  augmented_operator.vmult(augmented_operator_action, sample_state);
  augmented_matrix_action -= augmented_operator_action;
  EXPECT_LT(augmented_matrix_action.l2_norm(), 1.e-10);

  try
    {
      direct.solve(direct_state);
    }
  catch (const std::exception &exception)
    {
      FAIL() << exception.what();
    }
  catch (...)
    {
      FAIL() << "unknown exception from the parallel direct solver";
    }
  GlobalVector direct_residual;
  direct.evaluate_residual(direct_state, direct_residual);
  EXPECT_LT(direct_residual.l2_norm(), 1.e-7);
}
