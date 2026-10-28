// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
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

// These tests replicate Sections 5.1 and 5.2 of
//
//   C. Belponer, A. Caiazzo, L. Heltai,
//   "Mixed-dimensional modeling of vascular tissues with reduced Lagrange
//   multipliers"
//
// Section, figure and table numbers below refer to that paper. The
// parameters are in gtests/parameters/elasticity_04_*.prm.in; each run
// overrides only the entries that change between the cases.

#include <deal.II/base/mpi.h>

#include <gtest/gtest.h>
#include <immersx/physics/elasticity.h>

#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace ImmersX;
#include <immersx/io/utils.h>

#include "test_paths.h"

using namespace dealii;

// Section 5.1, Fig. 2, global refinement: unit disk with one inclusion of
// radius ri = 0.2 and normal inflation ubar = 0.1, initial refinement 3 to 7.
// The reference values are the data behind Fig. 2
// (convergence_global_errors.txt and
// convergence_global_externalPressure.txt). Their multiplier column is
// lambda_1x + lambda_2y = 2 lambda_1x; Fig. 2 (right) plots the error of
// lambda_1x against the analytical value 12 pi ubar / (R^2 - ri^2).
//
// The data were computed with an older solver (MinRes on the Schur
// complement, prms/elasticity/circle_one_hole.prm) converged to a reduction
// of 1e-10. This test uses the augmented Lagrangian solver with the same
// tolerances. Both solve the same discrete system, and the values agree to
// the four digits of the data files.
TEST(ElasticityValidation, BOTH_PaperAxisSymmetricGlobalConvergence)
{
  // dofs, L2, Linfty, H1, lambda_1x + lambda_2y.
  const std::vector<std::array<double, 5>> fig2_global = {
    {{674, 7.010e-03, 1.503e-02, 1.143e-01, -9.005}},
    {{2626, 2.957e-03, 6.197e-03, 6.297e-02, -8.335}},
    {{10370, 1.303e-03, 2.950e-03, 4.828e-02, -8.062}},
    {{41218, 6.771e-04, 1.466e-03, 3.359e-02, -7.962}},
    {{164354, 3.504e-04, 9.467e-04, 2.306e-02, -7.910}}};
  const double lambda_exact = -12 * numbers::PI * 0.1 / (1 - 0.2 * 0.2);

  double previous_lambda_error = std::numeric_limits<double>::max();
  for (unsigned int i = 0; i < fig2_global.size(); ++i)
    {
      const auto &reference = fig2_global[i];

      ParameterAcceptor::clear();
      ElasticityProblemParameters<2> par;
      ElasticityProblem<2>           problem(par);
      initialize_parameters(TestPaths::parameter_path(
        "gtests/parameters/elasticity_04_axis_symmetric.prm"));
      ParameterAcceptor::prm.parse_input_from_string(
        "subsection Immersed Problem\n"
        "  set Initial refinement = " +
        std::to_string(3 + i) +
        "\n"
        "  subsection Refinement and remeshing\n"
        "    set Number of refinement cycles = 1\n"
        "    set Strategy                    = global\n"
        "  end\n"
        "end\n");
      ParameterAcceptor::parse_all_parameters();
      problem.run();

      // Header, then one row: cells, dofs, and the L2, Linfty and H1 errors,
      // each followed by its (here undefined) convergence rate.
      std::stringstream table;
      par.convergence_table.output_table(table);
      std::string header, rate;
      double      cells, dofs, l2, linfty, h1;
      std::getline(table, header);
      table >> cells >> dofs >> l2 >> rate >> linfty >> rate >> h1 >> rate;

      const auto  &lambda = problem.solution.block(1);
      const double lambda_1x =
        Utilities::MPI::sum(lambda.in_local_range(0) ? lambda[0] : 0.,
                            MPI_COMM_WORLD);
      const double lambda_2y =
        Utilities::MPI::sum(lambda.in_local_range(1) ? lambda[1] : 0.,
                            MPI_COMM_WORLD);
      problem.pcout << "dofs " << dofs << " L2 " << l2 << " Linfty " << linfty
                    << " H1 " << h1 << " lambda " << lambda_1x << " "
                    << lambda_2y << std::endl;

      EXPECT_EQ(dofs, reference[0]);
      EXPECT_NEAR(l2, reference[1], 1e-3 * reference[1]);
      EXPECT_NEAR(linfty, reference[2], 1e-3 * reference[2]);
      EXPECT_NEAR(h1, reference[3], 1e-3 * reference[3]);
      EXPECT_NEAR(lambda_1x + lambda_2y,
                  reference[4],
                  1e-3 * std::abs(reference[4]));
      // Axis symmetry: the (1, x) and (2, y) modes coincide.
      EXPECT_NEAR(lambda_1x, lambda_2y, 1e-6 * std::abs(lambda_1x));

      const double lambda_error = std::abs(lambda_1x - lambda_exact);
      EXPECT_LT(lambda_error, previous_lambda_error);
      previous_lambda_error = lambda_error;
    }
}



// Section 5.1, Fig. 2, adaptive refinement: the same problem with six
// fixed_number refinement cycles. The reference values are the data behind
// Fig. 2 (convergence_fixed_number_errors.txt and
// convergence_fixed_number_externalPressure.txt), whose multiplier column is
// lambda_1x + lambda_2y. Only the multiplier of the last cycle is available
// after the run.
TEST(ElasticityValidation, BOTH_PaperAxisSymmetricAdaptiveRefinement)
{
  // cells, dofs, L2, Linfty, H1.
  const std::vector<std::array<double, 5>> fig2_adaptive = {
    {{320, 674, 7.010e-03, 1.503e-02, 1.143e-01}},
    {{620, 1330, 2.920e-03, 6.187e-03, 6.398e-02}},
    {{1184, 2546, 1.270e-03, 2.944e-03, 4.987e-02}},
    {{2228, 4826, 6.459e-04, 1.474e-03, 3.539e-02}},
    {{4352, 9434, 3.286e-04, 9.467e-04, 2.475e-02}},
    {{8540, 18522, 1.546e-04, 6.874e-04, 1.803e-02}}};
  const double last_lambda = -7.881;

  ParameterAcceptor::clear();
  ElasticityProblemParameters<2> par;
  ElasticityProblem<2>           problem(par);
  initialize_parameters(TestPaths::parameter_path(
    "gtests/parameters/elasticity_04_axis_symmetric.prm"));
  problem.run();

  // Header, then one row per cycle: cells, dofs, and the L2, Linfty and H1
  // errors, each followed by its convergence rate.
  std::stringstream table;
  par.convergence_table.output_table(table);
  std::string header, rate;
  std::getline(table, header);
  for (const auto &reference : fig2_adaptive)
    {
      double cells, dofs, l2, linfty, h1;
      table >> cells >> dofs >> l2 >> rate >> linfty >> rate >> h1 >> rate;
      ASSERT_TRUE(table);
      problem.pcout << "dofs " << dofs << " L2 " << l2 << " Linfty " << linfty
                    << " H1 " << h1 << std::endl;

      EXPECT_EQ(cells, reference[0]);
      EXPECT_EQ(dofs, reference[1]);
      EXPECT_NEAR(l2, reference[2], 1e-3 * reference[2]);
      EXPECT_NEAR(linfty, reference[3], 1e-3 * reference[3]);
      EXPECT_NEAR(h1, reference[4], 1e-3 * reference[4]);
    }

  const auto  &lambda = problem.solution.block(1);
  const double lambda_1x =
    Utilities::MPI::sum(lambda.in_local_range(0) ? lambda[0] : 0.,
                        MPI_COMM_WORLD);
  const double lambda_2y =
    Utilities::MPI::sum(lambda.in_local_range(1) ? lambda[1] : 0.,
                        MPI_COMM_WORLD);
  problem.pcout << "lambda " << lambda_1x << " " << lambda_2y << std::endl;
  EXPECT_NEAR(lambda_1x + lambda_2y, last_lambda, 1e-3 * std::abs(last_lambda));
}



// Section 5.2, Table 1 and Fig. 4: three inclusions at (0.3, 0.3),
// (-0.4, 0.3), (0.1, -0.3) in [-1, 1]^2 with ubar = 0.1 (Fig. 3), radius ri
// and N modes. Table 1 reports the multiplier of the second inclusion.
//
// With N modes the selected Fourier coefficients are 2, 5, 6, ..., 2N + 1:
// the leading modes only carry the normal components (1, x) and (2, y), the
// modes 3, ..., N carry both components, so the multiplier has 2N - 2
// entries per inclusion.
//
// Table 1 was computed with an older solver: MinRes on the Schur complement
// for the multiplier, stopped at a residual reduction of 1e-2 or after 100
// steps (Outer control in prms/elasticity/square_three_holes.prm), with
// inner displacement solves to a reduction of 1e-6. That solve is not
// converged. This test solves the same discrete system with the augmented
// Lagrangian solver to a reduction of 1e-10: the term gamma B^T W^-1 (B u - g)
// it adds vanishes at the solution, and tightening the tolerances to 1e-14
// does not change any printed digit. The differences from Table 1 come from
// the early stop, not from the solver type. Stopping the augmented
// Lagrangian solver at a reduction of 1e-2 moves the values by a similar
// amount (e_N,i from 0.30% to 0.53% for ri = 0.05 and N = 8, ||Lambda||^2
// by up to 1%).
//
// The test therefore keeps the tight tolerances and checks the converged
// values against Table 1 with tolerances that cover the early stop. The
// column labeled ||Lambda||_l2 in Table 1 matches the squared norm computed
// here to 0.7%. The shares of the leading modes match to 0.5 percentage
// points. The truncation e_N,i is up to twice the value of Table 1 (0.30%
// instead of 0.14% for ri = 0.05 and N = 8), so it is only checked to be at
// most 2 e_N,i + 0.05%.
TEST(ElasticityValidation, BOTH_PaperHigherOrderModes)
{
  struct Table1Row
  {
    std::string  ri;
    unsigned int highest_mode; // N in Table 1: highest Fourier mode retained
    double       norm;         // ||Lambda||_l2 column of Table 1
    double       share_1x;     // |Lambda_1,x|^2 / ||Lambda||^2, in percent
    double       share_2y;     // |Lambda_2,y|^2 / ||Lambda||^2, in percent
    double       e;            // truncation e_N,i, in percent
  };
  const std::vector<Table1Row> table1 = {
    {"0.05", 2, 34.5814, 51.07, 48.93, 0},
    {"0.05", 3, 34.5305, 50.6, 49.39, 8.01e-3},
    {"0.05", 4, 34.5312, 50.59, 49.4, 9e-3},
    {"0.05", 5, 34.6871, 50.95, 48.99, 6.1e-2},
    {"0.05", 6, 34.6873, 50.96, 48.98, 6.3e-2},
    {"0.05", 7, 34.7736, 51.01, 48.86, 0.13},
    {"0.05", 8, 34.7862, 51.01, 48.85, 0.14},
    {"0.1", 2, 32.78826, 51.07, 48.93, 0},
    {"0.1", 3, 32.7818, 50.76, 49.23, 6.01e-3},
    {"0.1", 4, 32.7801, 50.76, 49.24, 6.01e-3},
    {"0.1", 5, 32.7835, 50.77, 49.22, 9e-3},
    {"0.1", 6, 32.7871, 50.79, 49.2, 1.2e-2},
    {"0.1", 7, 32.7919, 50.81, 49.17, 1.6e-2},
    {"0.1", 8, 32.788, 50.81, 49.17, 1.5e-2},
    {"0.2", 2, 37.4164, 54.66, 45.34, 0},
    {"0.2", 3, 37.4083, 54.73, 45.26, 6.01e-3},
    {"0.2", 4, 37.4084, 54.73, 45.26, 9e-3},
    {"0.2", 5, 37.4177, 54.73, 45.26, 1.3e-2},
    {"0.2", 6, 37.4321, 54.7, 45.28, 1.7e-2},
    {"0.2", 7, 37.434, 54.71, 45.27, 1.8e-2},
    {"0.2", 8, 37.4366, 54.71, 45.27, 2e-2}};

  for (const auto &reference : table1)
    {
      std::string selection = "2, 5";
      std::string data      = "0, 0, 0.1, 0, 0, 0.1";
      for (unsigned int c = 6; c < 2 * (reference.highest_mode + 1); ++c)
        {
          selection += ", " + std::to_string(c);
          data += ", 0";
        }
      const std::string &ri = reference.ri;

      ParameterAcceptor::clear();
      ElasticityProblemParameters<2> par;
      ElasticityProblem<2>           problem(par);
      initialize_parameters(TestPaths::parameter_path(
        "gtests/parameters/elasticity_04_higher_order_modes.prm"));
      ParameterAcceptor::prm.parse_input_from_string(
        "subsection Immersed Problem\n"
        "  subsection Immersed inclusions\n"
        "    set Inclusions = 0.3, 0.3, " +
        ri + "; -0.4, 0.3, " + ri + "; 0.1, -0.3, " + ri +
        "\n"
        "    set Number of fourier coefficients    = " +
        std::to_string(2 * reference.highest_mode - 2) +
        "\n"
        "    set Reference inclusion data          = " +
        data +
        "\n"
        "    set Selection of Fourier coefficients = " +
        selection +
        "\n"
        "  end\n"
        "end\n");
      ParameterAcceptor::parse_all_parameters();
      problem.run();

      const auto        &lambda = problem.solution.block(1);
      const unsigned int n      = problem.inclusions.n_dofs_per_inclusion();
      ASSERT_EQ(n, 2 * reference.highest_mode - 2);
      std::vector<double> lambda_2(n);
      for (unsigned int k = 0; k < n; ++k)
        lambda_2[k] =
          Utilities::MPI::sum(lambda.in_local_range(n + k) ? lambda[n + k] : 0.,
                              MPI_COMM_WORLD);

      double norm2 = 0;
      for (const auto l : lambda_2)
        norm2 += l * l;
      const double share_1x   = 100 * lambda_2[0] * lambda_2[0] / norm2;
      const double share_2y   = 100 * lambda_2[1] * lambda_2[1] / norm2;
      const double truncation = 100 - share_1x - share_2y;
      problem.pcout << "ri " << ri << " N " << reference.highest_mode
                    << " |Lambda|^2 " << norm2 << " 1x " << share_1x << " 2y "
                    << share_2y << " truncation " << truncation << " lambda";
      for (const auto l : lambda_2)
        problem.pcout << " " << l;
      problem.pcout << std::endl;

      EXPECT_NEAR(norm2, reference.norm, 1e-2 * reference.norm);
      EXPECT_NEAR(share_1x, reference.share_1x, 0.5);
      EXPECT_NEAR(share_2y, reference.share_2y, 0.5);
      EXPECT_NEAR(truncation, reference.e, reference.e + 0.05);
    }
}
