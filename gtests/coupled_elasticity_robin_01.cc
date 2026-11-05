// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/coupled_elasticity.h>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

TEST(CoupledElasticityRobin, BOTH_PhysicalTimeUpdatesSpringLoadAndLifting)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2> parameters;
  parameters.initial_refinement  = 1;
  parameters.n_refinement_cycles = 1;
  parameters.dirichlet_ids       = {0};
  parameters.robin_ids           = {1, 2, 3};
  parameters.output_directory =
    TestPaths::output_directory("coupled-elasticity-robin-time");
  parameters.displacement_solver_control.set_tolerance(1.e-12);
  parameters.displacement_solver_control.set_reduction(1.e-12);
  initialize_parameters_from_string(R"(
subsection Immersed Problem
  subsection Grid generation
    set Grid generator = hyper_cube
    set Grid generator arguments = 0:1:true
  end
  subsection Time parameters
    set Initial time = 0.1
    set Final time = 1
    set Time step = 0.1
  end
  subsection Material properties
    subsection default
      set Density = 1
      set Lame mu = 1
      set Lame lambda = 1
    end
  end
end
subsection Functions
  subsection Dirichlet boundary conditions
    set Function expression = 1+t; 2*(1+t)
  end
  subsection Robin coefficient
    set Function expression = 2+x+t
  end
  subsection Robin boundary conditions
    set Function expression = (2+x+t)*(1+t); 2*(2+x+t)*(1+t)
  end
end
)");
  parameters.time_mode = TimeMode::Static;
  CoupledElasticityProblem<2> problem(parameters);
  problem.run_timestep0();
  EXPECT_DOUBLE_EQ(problem.current_time, 0.1);
  const auto cells              = problem.tria->n_global_active_cells();
  double     previous_stiffness = 0.;
  for (const double time : {0.2, 0.7})
    {
      problem.run_timestep(time);
      EXPECT_DOUBLE_EQ(problem.current_time, time);
      EXPECT_EQ(problem.tria->n_global_active_cells(), cells);
      const auto                          &fe = problem.dh.get_fe();
      std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
      double                               local_error = 0.;
      for (const auto &cell : problem.dh.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(ids);
            for (unsigned int i = 0; i < ids.size(); ++i)
              if (problem.owned_dofs[0].is_element(ids[i]))
                local_error =
                  std::max(local_error,
                           std::abs(
                             problem.solution.block(0)[ids[i]] -
                             (1. + time) *
                               (1. + fe.system_to_component_index(i).first)));
          }
      EXPECT_LT(Utilities::MPI::max(local_error, MPI_COMM_WORLD), 1.e-9);
      const double stiffness = problem.stiffness_matrix.frobenius_norm();
      EXPECT_GT(stiffness, previous_stiffness);
      previous_stiffness = stiffness;
    }
}

TEST(CoupledElasticityRobin, BOTH_TimeUpdatesWithNonemptyPointMultipliers)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<3> parameters;
  CoupledElasticityProblem<3>    problem(parameters);
  parameters.output_directory =
    TestPaths::output_directory("coupled-robin-time-point-multipliers");
  initialize_parameters_from_string(R"(
subsection Immersed Problem
  set FE degree = 2
  set Initial refinement = 1
  set Dirichlet boundary ids = 0
  set Robin boundary ids = 1,2,3,4,5
  subsection Grid generation
    set Grid generator = hyper_cube
    set Grid generator arguments = 0:1:true
  end
  subsection Refinement and remeshing
    set Number of refinement cycles = 1
  end
  subsection Immersed inclusions
    set Inclusions file =
    set Inclusions refinement = 16
    set Number of fourier coefficients = 2
    set Selection of Fourier coefficients = 3,7
    set Reference inclusion data = 0,0,0,0,0,0,0,0,0
  end
  subsection Material properties
    subsection default
      set Lame mu = 1
      set Lame lambda = 1
    end
  end
end
subsection Functions
  subsection Dirichlet boundary conditions
    set Function expression = 1+t;2*(1+t);3*(1+t)
  end
  subsection Robin coefficient
    set Function expression = 2+x+t
  end
  subsection Robin boundary conditions
    set Function expression = (2+x+t)*(1+t);2*(2+x+t)*(1+t);3*(2+x+t)*(1+t)
  end
end
)");
  parameters.time_mode = TimeMode::Static;
  for (auto *control : {&parameters.displacement_solver_control,
                        &parameters.augmented_lagrange_solver_control,
                        &parameters.reduced_mass_solver_control})
    {
      control->set_max_steps(1000);
      control->set_tolerance(1.e-13);
      control->set_reduction(1.e-13);
    }
  problem.inclusions.inclusions = {{0.35, 0.5, 0.5, 0., 0., 1., 0.05, 0.},
                                   {0.50, 0.5, 0.5, 0., 0., 1., 0.05, 1.},
                                   {0.65, 0.5, 0.5, 0., 0., 1., 0.05, 2.}};
  problem.run_timestep0();
  ASSERT_EQ(problem.n_vessels(), 3u);
  ASSERT_EQ(problem.solution.block(1).size(), 6u);
  const auto                          &fe = problem.dh.get_fe();
  std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
  for (const double time : {0.2, 0.7})
    {
      // The first coupling leaves a nonempty multiplier RHS in insert mode;
      // reassembling the bulk at the second time must compress only its block.
      problem.run_timestep(time);
      EXPECT_DOUBLE_EQ(problem.current_time, time);
      double local_error = 0.;
      for (const auto &cell : problem.dh.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(ids);
            for (unsigned int i = 0; i < ids.size(); ++i)
              if (problem.owned_dofs[0].is_element(ids[i]))
                local_error =
                  std::max(local_error,
                           std::abs(
                             problem.solution.block(0)[ids[i]] -
                             (1. + time) *
                               (1. + fe.system_to_component_index(i).first)));
          }
      EXPECT_LT(Utilities::MPI::max(local_error, MPI_COMM_WORLD), 1.e-9);
      EXPECT_LT(problem.solution.block(1).l2_norm(), 1.e-8);
    }
}
