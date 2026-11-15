// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <cmath>

#ifdef DEAL_II_WITH_SUNDIALS
namespace
{
  using Vector = ImmersX::ImmersXLA::MPI::Vector;

  void
  check_saddle_refinement(const unsigned int budget,
                          const bool         expect_success,
                          const bool         fail_backsolve = false)
  {
    const auto comm  = MPI_COMM_WORLD;
    const bool owner = dealii::Utilities::MPI::this_mpi_process(comm) == 0;
    dealii::IndexSet owned(2);
    if (owner)
      owned.add_range(0, 2);
    Vector rhs(owned, comm), solution(owned, comm);
    if (owner)
      {
        rhs[0] = 2.;
        rhs[1] = 1.e-4;
        // For A=[[1e4,1],[1,0]], the exact solution is [1e-4,1].
        // With inverse diag(1e-4,-1e4), three unit-step Richardson
        // corrections from M^-1 rhs give [0,3], not the exact solution.
        solution[0] = 0.;
        solution[1] = 3.;
      }
    rhs.compress(dealii::VectorOperation::insert);
    solution.compress(dealii::VectorOperation::insert);
    dealii::LinearOperator<Vector> matrix;
    matrix.reinit_range_vector = [owned, comm](Vector &v, bool) {
      v.reinit(owned, comm);
    };
    matrix.reinit_domain_vector = matrix.reinit_range_vector;
    matrix.vmult                = [owner](Vector &dst, const Vector &src) {
      if (owner)
        {
          dst[0] = 1.e4 * src[0] + src[1];
          dst[1] = src[0];
        }
      dst.compress(dealii::VectorOperation::insert);
    };
    unsigned int                                        applications = 0;
    const std::function<void(const Vector &, Vector &)> inverse =
      [owner, &applications, fail_backsolve](const Vector &src, Vector &dst) {
        ++applications;
        if (fail_backsolve)
          throw dealii::SolverControl::NoConvergence(0, 1.);
        if (owner)
          {
            dst[0] = 1.e-4 * src[0];
            dst[1] = -1.e4 * src[1];
          }
        dst.compress(dealii::VectorOperation::insert);
      };
    const double target     = 1.e-10 * rhs.l2_norm();
    const auto   iterations = ImmersX::detail::mumps_krylov_refinement<Vector>(
      matrix, inverse, rhs, solution, budget, target);
    EXPECT_LE(iterations, budget);
    EXPECT_LE(applications, budget);
    // Independent physical equations, not the operator used by FGMRES.
    const double local_defect =
      owner ?
        std::hypot(1.e4 * solution[0] + solution[1] - 2., solution[0] - 1.e-4) :
        0.;
    const double defect = dealii::Utilities::MPI::max(local_defect, comm);
    if (expect_success)
      {
        EXPECT_LE(defect, target);
        if (owner)
          {
            EXPECT_NEAR(solution[0], 1.e-4, 1.e-10);
            EXPECT_NEAR(solution[1], 1., 1.e-6);
          }
      }
    else
      {
        EXPECT_GT(defect, target);
      }
  }
} // namespace

TEST(DistributedIDA, BOTH_KrylovRecoversOscillatingSaddleRefinement)
{
  check_saddle_refinement(3, true);
}

TEST(DistributedIDA, BOTH_KrylovBudgetCannotCertifyInaccurateCorrection)
{
  check_saddle_refinement(1, false);
  check_saddle_refinement(0, false);
}

TEST(DistributedIDA, BOTH_KrylovPropagatesBacksolveFailure)
{
  EXPECT_THROW(check_saddle_refinement(3, true, true),
               dealii::SolverControl::NoConvergence);
}
#endif
