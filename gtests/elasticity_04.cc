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
// Section, figure and table numbers below refer to that paper. Small
// representative cases retain its independent numerical reference values;
// the full refinement and mode-count studies are not regression tests.

#include <deal.II/base/mpi.h>

#include <gtest/gtest.h>
#include <immersx/physics/vessel_elasticity.h>

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
// radius ri = 0.2 and normal inflation ubar = 0.1, on two representative
// global meshes at refinement 3 and 4.
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
TEST(ElasticityValidation, BOTH_PaperAxisSymmetricGlobalRegression)
{
  // dofs, L2, Linfty, H1, lambda_1x + lambda_2y.
  const std::vector<std::array<double, 5>> fig2_global = {
    {{674, 7.010e-03, 1.503e-02, 1.143e-01, -9.005}},
    {{2626, 2.957e-03, 6.197e-03, 6.297e-02, -8.335}}};
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



// Section 5.1, Fig. 2, adaptive refinement: the same problem with two
// fixed_number refinement cycles. The reference values are the data behind
// Fig. 2 (convergence_fixed_number_errors.txt and
// convergence_fixed_number_externalPressure.txt), whose multiplier column is
// lambda_1x + lambda_2y. The global-mesh and pressure regressions below
// independently check the multiplier; this case checks mesh adaptation.
TEST(ElasticityValidation, BOTH_PaperAxisSymmetricAdaptiveRegression)
{
  // cells, dofs, L2, Linfty, H1.
  const std::vector<std::array<double, 5>> fig2_adaptive = {
    {{320, 674, 7.010e-03, 1.503e-02, 1.143e-01}},
    {{620, 1330, 2.920e-03, 6.187e-03, 6.398e-02}}};

  ParameterAcceptor::clear();
  ElasticityProblemParameters<2> par;
  ElasticityProblem<2>           problem(par);
  initialize_parameters(TestPaths::parameter_path(
    "gtests/parameters/elasticity_04_axis_symmetric.prm"));
  par.n_refinement_cycles = 2;
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
}



// Section 5.1 with the pressure of VesselElasticityProblem, on the annulus
// between the inclusion (ri = 0.2) and the unit circle (R = 1), with normal
// inflation ubar = 0.1 and Lame lambda = 3, mu = 1. The exact solution of the
// parameter file does not depend on the material:
//   outside the inclusion: u_r = A r + B / r, with
//     A = -ri ubar / (R^2 - ri^2) and B = ri ubar R^2 / (R^2 - ri^2);
//   inside the inclusion, which is filled with the same material: the
//     uniform expansion u = ubar x / ri.
// At r = ri the exact radial stresses are
//   outside: sigma_rr = 2 (lambda + mu) A - 2 mu B / ri^2,
//   inside:  sigma_rr = 2 (lambda + mu) ubar / ri.
// The multiplier gives their difference. With "Remove interior stress from
// pressure" the computed pressure adds back the inside stress, and it is the
// outside stress. A single resolved reference mesh checks these analytical
// stresses without a mesh-convergence sweep.
TEST(ElasticityValidation, BOTH_PaperAxisSymmetricPressure)
{
  constexpr double ri = 0.2, R = 1., ubar = 0.1, lambda = 3., mu = 1.;
  const double     A       = -ri * ubar / (R * R - ri * ri);
  const double     B       = ri * ubar * R * R / (R * R - ri * ri);
  const double     outside = 2 * (lambda + mu) * A - 2 * mu * B / (ri * ri);
  const double     inside  = 2 * (lambda + mu) * ubar / ri;

  ParameterAcceptor::clear();
  ElasticityProblemParameters<2> par;
  VesselElasticityProblem<2>     problem(par);
  initialize_parameters(TestPaths::parameter_path(
    "gtests/parameters/elasticity_04_axis_symmetric.prm"));
  ParameterAcceptor::prm.parse_input_from_string(
    "subsection Immersed Problem\n"
    "  set FE degree = 2\n"
    "  set Initial refinement = 4\n"
    "  set Remove interior stress from pressure = false\n"
    "  subsection Material properties\n"
    "    subsection default\n"
    "      set Lame lambda = 3\n"
    "      set Lame mu = 1\n"
    "    end\n"
    "  end\n"
    "  subsection Refinement and remeshing\n"
    "    set Number of refinement cycles = 1\n"
    "    set Strategy = global\n"
    "  end\n"
    "end\n");
  ParameterAcceptor::parse_all_parameters();
  problem.run_timestep0();
  // Resolve the stress jump near the circle rather than refining the disk.
  for (const auto &particle : problem.inclusions.inclusions_as_particles)
    particle.get_surrounding_cell()->set_refine_flag();
  problem.execute_actual_refine_and_transfer();
  problem.run_timestep();

  const auto pressure = [&]() {
    const auto &p = problem.pressure_at_immersed_surface;
    EXPECT_EQ(p.size(), 1u);
    return Utilities::MPI::sum(p.in_local_range(0) ? p[0] : 0., MPI_COMM_WORLD);
  };
  const double jump = pressure();
  // Interior-stress removal changes pressure observation, not the solve.
  par.remove_interior_stress_from_pressure = true;
  problem.compute_pressure_at_immersed_surface();
  const double tissue = pressure();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "jump " << jump << " (exact " << outside - inside
              << "), tissue " << tissue << " (exact " << outside << ")"
              << std::endl;

  const double tolerance = 2e-2 * std::abs(outside - inside);
  EXPECT_NEAR(jump, outside - inside, tolerance);
  EXPECT_NEAR(tissue - jump, inside, 1e-10 * inside);
  EXPECT_NEAR(tissue, outside, tolerance);
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
// Table 1 used an early-stopped Schur solve, whereas this regression solves
// the same discrete equations with the augmented Lagrangian method. Keep the
// original tolerances accounting for that published stopping error: one
// percent for the squared multiplier norm, half a percentage point for its
// leading modal shares, and twice the tabulated truncation plus 0.05 percent.
TEST(ElasticityValidation, BOTH_PaperHigherOrderModesRegression)
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
  // One multi-mode configuration exercises the retained-mode indexing and
  // the three-inclusion solve against the published values. The full table
  // is a parameter study rather than a separate correctness oracle.
  const std::vector<Table1Row> table1 = {
    {"0.1", 4, 32.7801, 50.76, 49.24, 6.01e-3}};

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
