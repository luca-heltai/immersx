// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of the ImmersX application, based on deal.II.
// ---------------------------------------------------------------------
#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <array>
#include <cmath>
#include <limits>

#ifdef DEAL_II_WITH_SUNDIALS
namespace
{
  using FieldVector = ImmersX::ImmersXLA::MPI::Vector;
  using Vector      = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter     = ImmersX::IDAAdapter<FieldVector, Vector>;
  using Operator    = dealii::LinearOperator<Vector>;

  struct FourFields
  {
    Vector   rhs, weights, solution;
    Operator matrix, inverse;
    FourFields(const bool empty_ranks)
    {
      const auto rank =
        dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      const auto ranks =
        dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      std::vector<dealii::IndexSet> partitions(4, dealii::IndexSet(1));
      for (unsigned int k = 0; k < 4; ++k)
        if (rank == (empty_ranks ? 0 : k % ranks))
          partitions[k].add_index(0);
      rhs.reinit(partitions, MPI_COMM_WORLD);
      weights.reinit(rhs);
      solution.reinit(rhs);
      matrix.reinit_range_vector = [partitions](Vector &v, const bool) {
        v.reinit(partitions, MPI_COMM_WORLD);
      };
      matrix.reinit_domain_vector = matrix.reinit_range_vector;
      const auto gather           = [](const Vector &src) {
        std::array<double, 4> v{};
        for (unsigned int k = 0; k < 4; ++k)
          v[k] = dealii::Utilities::MPI::sum(
            src.block(k).locally_owned_elements().n_elements() ?
              src.block(k)[0] :
              0.,
            MPI_COMM_WORLD);
        return v;
      };
      matrix.vmult = [gather](Vector &dst, const Vector &src) {
        const auto                  x = gather(src);
        const std::array<double, 4> values{{2 * x[0] + 1e-9 * x[1],
                                            3 * x[1] + .25 * x[2],
                                            5 * x[2] + .5 * x[3],
                                            7 * x[3]}};
        for (unsigned int k = 0; k < 4; ++k)
          if (dst.block(k).locally_owned_elements().n_elements())
            dst.block(k)[0] = values[k];
        dst.compress(dealii::VectorOperation::insert);
      };
      inverse       = matrix;
      inverse.vmult = [gather](Vector &dst, const Vector &src) {
        const auto            b = gather(src);
        std::array<double, 4> x{};
        x[3] = b[3] / 7.;
        x[2] = (b[2] - .5 * x[3]) / 5.;
        x[1] = (b[1] - .25 * x[2]) / 3.;
        x[0] = (b[0] - 1e-9 * x[1]) / 2.;
        for (unsigned int k = 0; k < 4; ++k)
          if (dst.block(k).locally_owned_elements().n_elements())
            dst.block(k)[0] = x[k];
        dst.compress(dealii::VectorOperation::insert);
      };
    }
    void
    set(Vector &v, const std::array<double, 4> &values)
    {
      for (unsigned int k = 0; k < 4; ++k)
        if (v.block(k).locally_owned_elements().n_elements())
          v.block(k)[0] = values[k];
      v.compress(dealii::VectorOperation::insert);
    }
  };
} // namespace

TEST(DistributedIDA, BOTH_WeightedKrylovHomogeneousAndMixedFields)
{
  for (const bool empty : {false, true})
    for (const bool flexible : {false, true})
      for (const bool mixed : {false, true})
        {
          FourFields f(empty);
          f.set(f.rhs, {{1.000000001, 3.5, 11.5, 21.}});
          f.set(f.weights,
                mixed ? std::array<double, 4>{{1e12, 1, 1e3, 1e-3}} :
                        std::array<double, 4>{{7, 7, 7, 7}});
          const auto count =
            ImmersX::detail::ida_weighted_krylov(f.matrix,
                                                 &f.inverse,
                                                 f.rhs,
                                                 f.solution,
                                                 f.weights,
                                                 mixed ? .01 : 1e-10,
                                                 1e-8 * f.rhs.l2_norm(),
                                                 8,
                                                 flexible,
                                                 MPI_COMM_WORLD);
          EXPECT_GT(count, 0u);
          EXPECT_LE(count, 8u);
          const std::array<double, 4> exact{{.5, 1., 2., 3.}};
          for (unsigned int k = 0; k < 4; ++k)
            if (f.solution.block(k).locally_owned_elements().n_elements())
              EXPECT_NEAR(f.solution.block(k)[0], exact[k], 1e-10);
          Vector defect = f.rhs;
          f.matrix.vmult(defect, f.solution);
          defect -= f.rhs;
          EXPECT_LE(defect.l2_norm(), 1e-8 * f.rhs.l2_norm());
          defect.scale(f.weights);
          EXPECT_LE(defect.l2_norm(), mixed ? .01 : 1e-10);
        }
}

TEST(DistributedIDA, BOTH_WeightedKrylovRequiresBothTrueResiduals)
{
  FourFields f(true);
  f.matrix.vmult = [](Vector &dst, const Vector &src) { dst = src; };
  f.set(f.weights, {{1e12, 1, 1e3, 1e-3}});
  // Zero is raw-acceptable, but has weighted residual 10 > .01.
  f.set(f.rhs, {{1e-11, 0, 0, 0}});
  EXPECT_GT(ImmersX::detail::ida_weighted_krylov<Vector>(f.matrix,
                                                         nullptr,
                                                         f.rhs,
                                                         f.solution,
                                                         f.weights,
                                                         .01,
                                                         1e-8,
                                                         4,
                                                         true,
                                                         MPI_COMM_WORLD),
            0u);
  // Zero is weighted-acceptable, but fails the raw residual requirement.
  f.set(f.rhs, {{0, 0, 0, 1e-6}});
  EXPECT_GT(ImmersX::detail::ida_weighted_krylov<Vector>(f.matrix,
                                                         nullptr,
                                                         f.rhs,
                                                         f.solution,
                                                         f.weights,
                                                         .01,
                                                         1e-14,
                                                         4,
                                                         true,
                                                         MPI_COMM_WORLD),
            0u);
  Vector error = f.solution;
  error -= f.rhs;
  EXPECT_LE(error.l2_norm(), 1e-14);
  EXPECT_THROW(ImmersX::detail::ida_weighted_krylov<Vector>(f.matrix,
                                                            nullptr,
                                                            f.rhs,
                                                            f.solution,
                                                            f.weights,
                                                            .01,
                                                            1e-14,
                                                            0,
                                                            true,
                                                            MPI_COMM_WORLD),
               dealii::SolverControl::NoConvergence);
  f.rhs = 0.;
  EXPECT_EQ(ImmersX::detail::ida_weighted_krylov<Vector>(f.matrix,
                                                         nullptr,
                                                         f.rhs,
                                                         f.solution,
                                                         f.weights,
                                                         .01,
                                                         0.,
                                                         0,
                                                         true,
                                                         MPI_COMM_WORLD),
            0u);
}

TEST(DistributedIDA, BOTH_WeightedKrylovRejectsInvalidWeightsCollectively)
{
  FourFields f(true);
  f.rhs = 1.;
  for (const double invalid : {0.,
                               -1.,
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()})
    {
      f.set(f.weights, {{invalid, 1, 1, 1}});
      EXPECT_THROW(ImmersX::detail::ida_weighted_krylov<Vector>(f.matrix,
                                                                nullptr,
                                                                f.rhs,
                                                                f.solution,
                                                                f.weights,
                                                                .01,
                                                                1e-8,
                                                                4,
                                                                true,
                                                                MPI_COMM_WORLD),
                   dealii::ExceptionBase);
    }
  f.weights = 1.;
  EXPECT_THROW(ImmersX::detail::ida_weighted_krylov<Vector>(f.matrix,
                                                            nullptr,
                                                            f.rhs,
                                                            f.solution,
                                                            f.weights,
                                                            0.,
                                                            1e-8,
                                                            4,
                                                            true,
                                                            MPI_COMM_WORLD),
               dealii::ExceptionBase);
}

TEST(DistributedIDA, BOTH_MixedScaleWitnessSeparatesNativeAndRawTargets)
{
  // Four independent state units. This witness tests the contracts, not a
  // projected Krylov residual or a physiological configuration.
  const std::array<double, 4> w{{1e12, 1, 1e3, 1e-3}};
  const std::array<double, 4> r{{1e-16, 1e-6, 1e-8, 1e-6}};
  double                      raw = 0., weighted = 0., mean = 0.;
  for (unsigned int i = 0; i < 4; ++i)
    {
      raw += r[i] * r[i];
      weighted += w[i] * w[i] * r[i] * r[i];
      mean += w[i] * w[i] / 4;
    }
  const double delta = .01, tau = delta / std::sqrt(mean);
  EXPECT_GT(std::sqrt(raw), tau);
  EXPECT_LT(std::sqrt(weighted), delta);
  EXPECT_LT(std::sqrt(raw), 1e-8 * 1000.);
  EXPECT_DOUBLE_EQ(tau * std::sqrt(mean), delta);
}
TEST(DistributedIDA, BOTH_NativeIdaMixedWeightsUseBothLinearGates)
{
  ImmersX::TimeIntervalParameters time;
  ImmersX::IDAParameters          parameters;
  time.initial_time                          = 0.;
  time.final_time                            = .01;
  time.output_time_interval                  = .005;
  parameters.initial_step_size               = .005;
  parameters.maximum_step_size               = .005;
  parameters.relative_tolerance              = 0.;
  parameters.absolute_tolerance              = 1.;
  parameters.correction_type_at_initial_time = "none";
  Adapter adapter(time, parameters, MPI_COMM_WORLD);
  adapter.set_stop_at_output_times(true);
  const std::array<double, 4> errors{{1e-12, 1., 1e-3, 1e3}};
  const std::array<double, 4> initial{{1e-12, 1000., 1e-3, 1.}};
  const auto                  contributor = adapter.add(
    [&](auto &builder) {
      using Model = typename std::decay_t<decltype(builder)>::Model;
      std::array<ImmersX::FieldId, 4> fields;
      for (unsigned int k = 0; k < 4; ++k)
        {
          dealii::IndexSet owned(1);
          if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
              k % dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
            owned.add_index(0);
          const auto id =
            builder.differential_field("scale_" + std::to_string(k), owned);
          fields[k] = id;
          typename Model::OperatorFactory identity = [id](const auto &context) {
            const auto              &reference = context.state(id);
            typename Model::Operator op;
            op.reinit_range_vector = [reference](FieldVector &v, const bool) {
              v.reinit(reference);
            };
            op.reinit_domain_vector = op.reinit_range_vector;
            op.vmult = [](FieldVector &dst, const FieldVector &src) {
              dst = src;
            };
            op.vmult_add = [](FieldVector &dst, const FieldVector &src) {
              dst += src;
            };
            op.Tvmult     = op.vmult;
            op.Tvmult_add = op.vmult_add;
            return op;
          };
          builder.term(id, "decay")
            .residual([id](const auto &context) {
              const auto state = context.state(id),
                         dot   = context.derivative(id);
              typename Model::Operation op;
              op.reinit_vector = [state](FieldVector &v, const bool) {
                v.reinit(state);
              };
              op.apply = [state, dot](FieldVector &dst) {
                dst = dot;
                dst += state;
              };
              op.apply_add = [state, dot](FieldVector &dst) {
                dst += dot;
                dst += state;
              };
              return op;
            })
            .state(id, identity)
            .derivative(id, identity);
        }
      return fields;
    },
    "four native error scales");
  for (unsigned int k = 0; k < 4; ++k)
    adapter.set_absolute_tolerance(contributor.fields()[k], errors[k]);
  auto state = adapter.make_state(), rate = adapter.make_state();
  for (unsigned int k = 0; k < 4; ++k)
    adapter.field(state, contributor.fields()[k]) = initial[k];
  state.compress(dealii::VectorOperation::insert);
  rate = state;
  rate *= -1.;
  auto        &ida        = adapter.solver();
  const auto   original   = ida.solve_with_jacobian;
  unsigned int checks     = 0;
  ida.solve_with_jacobian = [&](const Vector &rhs,
                                Vector       &answer,
                                const double  tau) {
    auto weights = adapter.make_state(), ones = adapter.make_state();
    ones       = 1.;
    auto wview = ImmersX::Compatibility::SUNDIALS::internal::make_nvector_view(
      weights
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
      ,
      ida.get_sun_context()
#  endif
    );
    auto oview = ImmersX::Compatibility::SUNDIALS::internal::make_nvector_view(
      ones
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
      ,
      ida.get_sun_context()
#  endif
    );
    EXPECT_EQ(IDAGetErrWeights(ida.get_ida_memory(), wview), IDA_SUCCESS);
    for (unsigned int k = 0; k < 4; ++k)
      for (const auto i : adapter.field(weights, contributor.fields()[k])
                            .locally_owned_elements())
        EXPECT_DOUBLE_EQ(adapter.field(weights, contributor.fields()[k])[i],
                         1. / errors[k]);
    const double delta = tau * N_VWrmsNorm(wview, oview);
    original(rhs, answer, tau);
    auto defect = adapter.make_state();
    adapter.current_jacobian().vmult(defect, answer);
    defect -= rhs;
    EXPECT_LE(defect.l2_norm(),
              parameters.linear_solver_controls.krylov_relative_tolerance *
                rhs.l2_norm());
    defect.scale(weights);
    EXPECT_LE(defect.l2_norm(), delta);
    ++checks;
  };
  EXPECT_GT(adapter.solve(state, rate), 0u);
  EXPECT_GT(checks, 0u);
  for (unsigned int k = 0; k < 4; ++k)
    for (const auto i :
         adapter.field(state, contributor.fields()[k]).locally_owned_elements())
      EXPECT_NEAR(adapter.field(state, contributor.fields()[k])[i] / initial[k],
                  std::exp(-time.final_time),
                  1e-3);
}
TEST(DistributedIDA, BOTH_IndeterminateTargetRequiresExactResidualEntries)
{
  FourFields f(true);
  f.set(f.rhs, {{1, 0, 0, 0}});
  f.matrix.vmult = [](Vector &dst, const Vector &src) { dst = src; };
  EXPECT_GT(
    ImmersX::detail::ida_exact_residual_krylov<Vector>(
      f.matrix, nullptr, f.rhs, f.solution, 1e-8, 4, true, MPI_COMM_WORLD),
    0u);
  // The nearest binary64 value of 1/49 leaves a nonzero true defect. Its
  // raw norm meets 1e-8, but cannot certify an indeterminate weighted target.
  f.matrix.vmult = [](Vector &dst, const Vector &src) {
    dst = src;
    dst *= 49.;
  };
  EXPECT_THROW(
    ImmersX::detail::ida_exact_residual_krylov<Vector>(
      f.matrix, nullptr, f.rhs, f.solution, 1e-8, 4, true, MPI_COMM_WORLD),
    dealii::ExceptionBase);
  bool   exact  = true;
  Vector defect = f.rhs;
  f.matrix.vmult(defect, f.solution);
  defect -= f.rhs;
  for (const auto i : defect.locally_owned_elements())
    exact = exact && defect[i] == 0.;
  EXPECT_EQ(dealii::Utilities::MPI::min(exact ? 1u : 0u, MPI_COMM_WORLD), 0u);
  EXPECT_LE(defect.l2_norm(), 1e-8);
}
#endif
