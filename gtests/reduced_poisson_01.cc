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
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/reduced_poisson.h>

#include <string>
#include <vector>

#include "test_paths.h"

#ifdef DEAL_II_WITH_VTK
using namespace ImmersX;
using namespace dealii;

namespace
{
  void
  check_augmented_lagrangian(const unsigned int inclusion_degree)
  {
    ParameterAcceptor::clear();
    ReducedPoissonParameters<3> parameters;
    initialize_parameters_from_string(R"(
      subsection Reduced Poisson
        set Dirichlet boundary ids = 0
        subsection Grid generation
          set Grid generator = hyper_cube
          set Grid generator arguments = 0:1:false
          set Initial refinement = 2
        end
        subsection Right hand side
          set Function expression = 0
        end
      end
    )");
    auto &coupling = parameters.reduced_coupling_parameters;
    auto &tensor   = coupling.tensor_product_space_parameters;
    tensor.reduced_grid_name =
      TestPaths::data_filename("tests/simplex_interior_line_3d.vtk");
    tensor.thickness                     = "0.05";
    tensor.fe_degree                     = 1;
    tensor.n_q_points                    = 3;
    tensor.section.inclusion_degree      = inclusion_degree;
    tensor.section.selected_coefficients = inclusion_degree == 0 ?
                                             std::vector<unsigned int>{0} :
                                             std::vector<unsigned int>{0, 1, 2};
    coupling.coupling_rhs_expressions =
      inclusion_degree == 0 ? std::vector<std::string>{"1"} :
                              std::vector<std::string>{"1", "0", "0"};
    auto &refinement                           = coupling.refinement_parameters;
    refinement.space_pre_refinement_cycles     = 0;
    refinement.space_post_refinement_cycles    = 0;
    refinement.embedded_pre_refinement_cycles  = 0;
    refinement.embedded_post_refinement_cycles = 0;
    refinement.max_refinement_level            = 0;
    parameters.inner_control.set_max_steps(1000);
    parameters.inner_control.set_tolerance(1.e-12);
    parameters.inner_control.set_reduction(1.e-12);
    parameters.outer_control.set_max_steps(1000);
    parameters.outer_control.set_tolerance(1.e-12);
    parameters.outer_control.set_reduction(1.e-12);
    parameters.solver_name = "AL";

    ReducedPoisson<3> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_dofs();
    problem.assemble_poisson_system();
    problem.assemble_coupling_system();
    ASSERT_GT(problem.coupling_matrix_frobenius_norm(), 0.);
    problem.solve();
    ASSERT_TRUE(problem.bulk_solution_is_finite());
    ASSERT_TRUE(problem.multiplier_solution_is_finite());
    ASSERT_GT(problem.bulk_solution_l2_norm(), 0.);
    ASSERT_GT(problem.multiplier_solution_l2_norm(), 0.);
    auto constraint_residual = problem.modal_rhs();
    problem.assembled_coupling().Tvmult(constraint_residual,
                                        problem.bulk_values());
    constraint_residual -= problem.modal_rhs();
    EXPECT_LT(constraint_residual.l2_norm(),
              1.e-8 * problem.modal_rhs().l2_norm());
    const auto augmented_bulk       = problem.bulk_values();
    const auto augmented_multiplier = problem.modal_values();

    parameters.solver_name = "Schur";
    problem.solve();
    auto bulk_difference       = problem.bulk_values();
    auto multiplier_difference = problem.modal_values();
    bulk_difference -= augmented_bulk;
    multiplier_difference -= augmented_multiplier;
    EXPECT_LT(bulk_difference.l2_norm(), 1.e-8 * augmented_bulk.l2_norm());
    EXPECT_LT(multiplier_difference.l2_norm(),
              1.e-8 * augmented_multiplier.l2_norm());
  }
} // namespace

TEST(ReducedPoisson, BOTH_ConstantModeAugmentedLagrangianMatchesSchur)
{
  check_augmented_lagrangian(0);
}

TEST(ReducedPoisson, BOTH_LinearModesAugmentedLagrangianMatchesSchur)
{
  check_augmented_lagrangian(1);
}
#endif
