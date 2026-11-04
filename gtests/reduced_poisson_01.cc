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
#include <immersx/algebra/augmented_lagrangian_preconditioner.h>
#include <immersx/io/utils.h>
#include <immersx/physics/reduced_poisson.h>

#include <algorithm>
#include <string>
#include <vector>

#include "test_paths.h"

#ifdef DEAL_II_WITH_VTK
using namespace ImmersX;
using namespace dealii;

namespace
{
  template <int representative_dimension = 1, int ambient_dimension = 3>
  void
  check_augmented_lagrangian(
    const unsigned int inclusion_degree,
    const std::string &grid     = "simplex_interior_line_3d.vtk",
    const bool         disjoint = false,
    const bool         assembled_preconditioner = false,
    const bool         thin_case                = false)
  {
    ParameterAcceptor::clear();
    ReducedPoissonParameters<ambient_dimension, representative_dimension>
                parameters;
    std::string input = R"(
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
    )";
    if (disjoint || thin_case)
      input.replace(input.find("Function expression = 0"),
                    std::string("Function expression = 0").size(),
                    "Function expression = 1");
    initialize_parameters_from_string(input);
    auto &coupling = parameters.reduced_coupling_parameters;
    auto &tensor   = coupling.tensor_product_space_parameters;
    if constexpr (representative_dimension > 0)
      tensor.reduced_grid_name = TestPaths::data_filename("tests/" + grid);
    else
      {
        if (!disjoint)
          tensor.point_cloud.points.emplace_back(-0.1, 0.5);
        tensor.point_cloud.points.emplace_back(2., 2.);
      }
    tensor.thickness =
      grid == "overlap_line_3d.vtk" && !thin_case ? "0.2" : "0.05";
    if constexpr (representative_dimension == 0)
      tensor.thickness = "0.3";
    tensor.fe_degree                     = 1;
    tensor.n_q_points                    = 3;
    tensor.section.inclusion_degree      = inclusion_degree;
    tensor.section.selected_coefficients = inclusion_degree == 0 ?
                                             std::vector<unsigned int>{0} :
                                             std::vector<unsigned int>{0, 1, 2};
    coupling.coupling_rhs_expressions =
      inclusion_degree == 0 ? std::vector<std::string>{"1"} :
                              std::vector<std::string>{"1", "0", "0"};
    auto &refinement                        = coupling.refinement_parameters;
    refinement.space_pre_refinement_cycles  = 0;
    refinement.space_post_refinement_cycles = 0;
    if constexpr (representative_dimension > 0)
      {
        refinement.embedded_pre_refinement_cycles  = 0;
        refinement.embedded_post_refinement_cycles = 0;
      }
    refinement.max_refinement_level = 0;
    parameters.inner_control.set_max_steps(1000);
    parameters.inner_control.set_tolerance(1.e-14);
    parameters.inner_control.set_reduction(1.e-14);
    parameters.outer_control.set_max_steps(1000);
    parameters.outer_control.set_tolerance(1.e-14);
    parameters.outer_control.set_reduction(1.e-14);
    parameters.solver_name             = "AL";
    parameters.assemble_full_AL_system = assembled_preconditioner;

    ReducedPoisson<ambient_dimension,
                   ambient_dimension,
                   representative_dimension>
      problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_dofs();
    if constexpr (representative_dimension == 0)
      {
        EXPECT_EQ(problem.n_reduced_dofs(), disjoint ? 3u : 6u);
      }
    problem.assemble_poisson_system();
    problem.assemble_coupling_system();
    if (disjoint)
      {
        EXPECT_EQ(problem.multiplier_load_matrix_frobenius_norm(), 0.);
      }
    else
      {
        ASSERT_GT(problem.multiplier_load_matrix_frobenius_norm(), 0.);
      }
    if (grid != "simplex_interior_line_3d.vtk")
      {
        EXPECT_GT(problem.assembled_outside_mass().frobenius_norm(), 0.);
      }
    problem.solve();
    ASSERT_TRUE(problem.bulk_solution_is_finite());
    ASSERT_TRUE(problem.multiplier_solution_is_finite());
    ASSERT_GT(problem.bulk_solution_l2_norm(), 0.);
    if (disjoint)
      {
        EXPECT_LT(problem.multiplier_solution_l2_norm(), 1.e-10);
      }
    else
      {
        ASSERT_GT(problem.multiplier_solution_l2_norm(), 0.);
      }
    auto bulk_residual = problem.bulk_rhs();
    problem.assembled_bulk().vmult(bulk_residual, problem.bulk_values());
    auto reaction = problem.bulk_rhs();
    problem.assembled_multiplier_load_matrix().vmult(reaction,
                                                     problem.modal_values());
    bulk_residual += reaction;
    bulk_residual -= problem.bulk_rhs();
    const double bulk_scale =
      std::max(1.,
               problem.assembled_bulk().frobenius_norm() *
                   problem.bulk_values().l2_norm() +
                 problem.assembled_multiplier_load_matrix().frobenius_norm() *
                   problem.modal_values().l2_norm() +
                 problem.bulk_rhs().l2_norm());
    EXPECT_LT(bulk_residual.l2_norm(), 1.e-8 * bulk_scale);
    auto constraint_residual = problem.modal_rhs();
    problem.assembled_multiplier_load_matrix().Tvmult(constraint_residual,
                                                      problem.bulk_values());
    auto outside_action = problem.modal_values();
    problem.assembled_outside_mass().vmult(outside_action,
                                           problem.modal_values());
    if (grid == "overlap_line_3d.vtk")
      {
        EXPECT_GT(outside_action.l2_norm(), 1.e-8);
      }
    constraint_residual -= outside_action;
    constraint_residual -= problem.modal_rhs();
    EXPECT_LT(constraint_residual.l2_norm(),
              1.e-8 * (disjoint ? 1. : problem.modal_rhs().l2_norm()));
    if (thin_case)
      return;
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
              1.e-8 * std::max(1., augmented_multiplier.l2_norm()));
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
TEST(ReducedPoisson, BOTH_OverlapModesAugmentedLagrangianMatchesSchur)
{
  check_augmented_lagrangian(1, "overlap_line_3d.vtk");
}

TEST(ReducedPoisson, BOTH_PointOverlapModesAugmentedLagrangianMatchesSchur)
{
  check_augmented_lagrangian<0, 2>(1, "overlap_line_3d.vtk");
}

TEST(ReducedPoisson, BOTH_DisjointPointMultiplierMassDecouplesBulk)
{
  check_augmented_lagrangian<0, 2>(1, "outside_line_3d.vtk", true);
}

TEST(ReducedPoisson, BOTH_ThinOverlapSatisfiesPhysicalRows)
{
  check_augmented_lagrangian(1, "overlap_line_3d.vtk", false, false, true);
}

TEST(ReducedPoisson, BOTH_OverlapAssembledALPreconditionerMatchesSchur)
{
  check_augmented_lagrangian(1, "overlap_line_3d.vtk", false, true);
}

TEST(ReducedPoisson, BOTH_DisjointMultiplierMassDecouplesBulk)
{
  check_augmented_lagrangian(1, "outside_line_3d.vtk", true);
}
#  ifdef DEAL_II_WITH_TRILINOS
TEST(ReducedPoisson, BOTH_AssembledAugmentationMatchesDenseProduct)
{
  const auto bulk =
    Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, 4);
  const auto multiplier =
    Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, 3);
  DynamicSparsityPattern bulk_graph(4, 4, bulk), coupling_graph(4, 3, bulk);
  for (const auto i : bulk)
    {
      for (unsigned int j = 0; j < 4; ++j)
        bulk_graph.add(i, j);
      for (unsigned int j = 0; j < 3; ++j)
        coupling_graph.add(i, j);
    }
  TrilinosWrappers::SparseMatrix A, Bt, augmented;
  A.reinit(bulk, bulk, bulk_graph, MPI_COMM_WORLD);
  Bt.reinit(bulk, multiplier, coupling_graph, MPI_COMM_WORLD);
  const double coefficients[4][3]{{1., 2., 0.},
                                  {0., -1., 3.},
                                  {2., 0.5, -0.5},
                                  {-1., 0., 1.}};
  const double weights[3]{0.5, 2., 1.5};
  const auto   entry = [](const unsigned int i, const unsigned int j) {
    return i == j ? 4. + i : 0.1 * (1. + std::min(i, j));
  };
  for (const auto i : bulk)
    {
      for (unsigned int j = 0; j < 4; ++j)
        A.set(i, j, entry(i, j));
      for (unsigned int j = 0; j < 3; ++j)
        Bt.set(i, j, coefficients[i][j]);
    }
  A.compress(VectorOperation::insert);
  Bt.compress(VectorOperation::insert);
  TrilinosWrappers::MPI::Vector scaling(multiplier, MPI_COMM_WORLD);
  for (const auto j : multiplier)
    scaling[j] = weights[j];
  scaling.compress(VectorOperation::insert);
  UtilitiesAL::create_augmented_block(A, Bt, scaling, 0.7, augmented);
  for (const auto i : bulk)
    for (unsigned int j = 0; j < 4; ++j)
      {
        double expected = entry(i, j);
        for (unsigned int k = 0; k < 3; ++k)
          expected +=
            0.7 * coefficients[i][k] * weights[k] * coefficients[j][k];
        EXPECT_NEAR(augmented.el(i, j), expected, 1.e-12);
      }
}
#  endif
#endif
