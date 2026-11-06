// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <gtest/gtest.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/matrix_operator.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <array>
#include <limits>
#include <memory>

using namespace dealii;
using namespace ImmersX;

namespace
{
  using FieldVector = ImmersXLA::MPI::Vector;
  using Matrix      = ImmersXLA::MPI::SparseMatrix;
  using Dense       = std::array<std::array<double, 3>, 3>;
  const Dense values{{{{4., 1., 0.}}, {{2., 5., -1.}}, {{0., 3., 6.}}}};
  const Dense unit_entries{{{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};

  IndexSet
  partition()
  {
    return Utilities::MPI::create_evenly_distributed_partitioning(
      MPI_COMM_WORLD, 3);
  }

  void
  initialize(Matrix &matrix, const IndexSet &owned, const Dense &entries)
  {
    DynamicSparsityPattern sparsity(3, 3, owned);
    for (const auto row : owned)
      for (unsigned int column = 0; column < 3; ++column)
        sparsity.add(row, column);
    sparsity.compress();
    matrix.reinit(owned, owned, sparsity, MPI_COMM_WORLD);
    for (const auto row : owned)
      for (unsigned int column = 0; column < 3; ++column)
        matrix.set(row, column, entries[row][column]);
    matrix.compress(VectorOperation::insert);
  }

  auto
  reinitializer(const IndexSet &owned)
  {
    return std::function<void(FieldVector &, bool)>(
      [owned](FieldVector &v, const bool) { v.reinit(owned, MPI_COMM_WORLD); });
  }

  void
  fill(FieldVector &v)
  {
    for (const auto i : v.locally_owned_elements())
      v(i) = 0.7 + 1.3 * i;
    v.compress(VectorOperation::insert);
  }
} // namespace

TEST(ReplicatedOperators, BOTH_DirectInverseAndTransposeWithEmptyRanks)
{
  const auto owned  = partition();
  const auto reinit = reinitializer(owned);
  Matrix     matrix;
  initialize(matrix, owned, values);
  const auto inverse =
    make_replicated_direct_preconditioner<FieldVector>(matrix, reinit, 3);
  FieldVector rhs, dst, defect;
  reinit(rhs, false);
  reinit(dst, false);
  reinit(defect, false);
  fill(rhs);
  // Import into an owned destination before overwrite, as in the native flow
  // pressure callback. This must not leave an Insert/Add state conflict.
  FieldVector imported;
  imported.reinit(owned, complete_index_set(3), MPI_COMM_WORLD);
  imported = rhs;
  dst      = imported;
  inverse.vmult(dst, rhs);
  matrix.vmult(defect, dst);
  defect -= rhs;
  EXPECT_LT(defect.l2_norm(), 1.e-12);
  inverse.Tvmult(dst, rhs);
  matrix.Tvmult(defect, dst);
  defect -= rhs;
  EXPECT_LT(defect.l2_norm(), 1.e-12);
  inverse.vmult(dst, rhs);
  FieldVector previous;
  reinit(previous, false);
  previous = dst;
  inverse.vmult_add(dst, rhs);
  dst.add(-2., previous);
  EXPECT_LT(dst.l2_norm(), 1.e-12);
  inverse.Tvmult(dst, rhs);
  previous = dst;
  inverse.Tvmult_add(dst, rhs);
  dst.add(-2., previous);
  EXPECT_LT(dst.l2_norm(), 1.e-12);
}

TEST(ReplicatedOperators, BOTH_MaterializedSnapshotRetainsActionsAndTranspose)
{
  const auto owned  = partition();
  const auto reinit = reinitializer(owned);
  Matrix     matrix;
  initialize(matrix, owned, values);
  auto       original     = matrix_operator<FieldVector, Matrix>(matrix).view;
  auto       applications = std::make_shared<unsigned int>(0);
  const auto action       = original.vmult;
  original.vmult          = [action, applications](FieldVector       &dst,
                                          const FieldVector &src) {
    ++*applications;
    action(dst, src);
  };
  const auto cached =
    detail::materialize_replicated_operator(original, MPI_COMM_WORLD, 3);
  EXPECT_EQ(*applications, 3u);
  FieldVector rhs, expected, result;
  reinit(rhs, false);
  reinit(expected, false);
  reinit(result, false);
  fill(rhs);
  matrix.vmult(expected, rhs);
  cached.vmult(result, rhs);
  result -= expected;
  EXPECT_LT(result.l2_norm(), 1.e-12);
  matrix.Tvmult(expected, rhs);
  cached.Tvmult(result, rhs);
  result -= expected;
  EXPECT_LT(result.l2_norm(), 1.e-12);
  matrix.vmult(expected, rhs);
  cached.vmult(result, rhs);
  result -= expected;
  EXPECT_LT(result.l2_norm(), 1.e-12);
  EXPECT_EQ(*applications, 3u);
  // A fresh factory sees the changed matrix; the old snapshot remains owned.
  for (const auto i : owned)
    matrix.add(i, i, 1.);
  matrix.compress(VectorOperation::add);
  const auto refreshed =
    detail::materialize_replicated_operator(original, MPI_COMM_WORLD, 3);
  cached.vmult(expected, rhs);
  refreshed.vmult(result, rhs);
  result -= expected;
  result -= rhs;
  EXPECT_LT(result.l2_norm(), 1.e-12);
  EXPECT_EQ(*applications, 6u);
}

TEST(ReplicatedOperators, BOTH_CollectiveBoundsAndSingularity)
{
  const auto owned  = partition();
  const auto reinit = reinitializer(owned);
  Matrix     matrix;
  initialize(matrix, owned, values);
  EXPECT_ANY_THROW(
    make_replicated_direct_preconditioner<FieldVector>(matrix, reinit, 0));
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  // All ranks must reject before one enters factorization with a different cap.
  EXPECT_ANY_THROW(make_replicated_direct_preconditioner<FieldVector>(
    matrix, reinit, rank == 0 ? 2 : 3));
  const Dense singular{{{{4., 1., 0.}}, {{8., 2., 0.}}, {{0., 0., 1.}}}};
  Matrix      bad;
  initialize(bad, owned, singular);
  EXPECT_ANY_THROW(
    make_replicated_direct_preconditioner<FieldVector>(bad, reinit, 3));
}

#ifndef IMMERSX_USE_PETSC_LA
TEST(ReplicatedOperators, BOTH_RejectNonfiniteInputBeforeSerialAction)
{
  const auto owned  = partition();
  const auto reinit = reinitializer(owned);
  auto       calls  = std::make_shared<unsigned int>(0);
  const auto op     = detail::replicated_vector_operator<FieldVector>(
    3,
    MPI_COMM_WORLD,
    reinit,
    [calls](dealii::Vector<double>       &dst,
            const dealii::Vector<double> &src,
            const bool) {
      ++*calls;
      dst = src;
    });
  FieldVector rhs, dst;
  reinit(rhs, false);
  reinit(dst, false);
  fill(rhs);
  // Inject through the native storage: public vector setters may reject NaN
  // before the operator contract can be exercised in Debug builds.
  if (owned.is_element(0))
    rhs.trilinos_vector()[0][owned.index_within_set(0)] =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_ANY_THROW(op.vmult(dst, rhs));
  EXPECT_EQ(*calls, 0u);
}
#endif

#ifdef DEAL_II_WITH_SUNDIALS
namespace
{
  using Adapter = IDAAdapter<FieldVector, ImmersXLA::MPI::BlockVector>;

  auto
  add_saddle(Adapter                             &adapter,
             const IndexSet                      &owned,
             Matrix                              &unit,
             const std::shared_ptr<unsigned int> &applications,
             const bool                           nonlinear,
             Matrix                              *coupling = nullptr)
  {
    return adapter
      .add(
        [&](auto &builder) {
          const auto u      = builder.algebraic_field("u", owned);
          const auto lambda = builder.algebraic_field("lambda", owned);
          const auto matrix = builder.matrix_operator(unit);
          const auto reaction =
            builder.matrix_operator(coupling ? *coupling : unit);
          builder.term(u, "diagonal").state(u, matrix).state(lambda, reaction);
          builder.term(lambda, "constraint").state(u, matrix);
          builder.preconditioner(
            u, [applications, nonlinear](const auto &, const auto &reinit) {
              auto result      = dealii::identity_operator<FieldVector>(reinit);
              const auto apply = [reinit, applications, nonlinear](
                                   FieldVector &dst, const FieldVector &src) {
                ++*applications;
                const double scale = nonlinear && src.l2_norm() > 2. ? 2. : 1.;
                reinit(dst, false);
                dst = src;
                dst *= scale;
              };
              result.vmult     = apply;
              result.Tvmult    = apply;
              result.vmult_add = [apply, reinit](FieldVector       &dst,
                                                 const FieldVector &src) {
                FieldVector tmp;
                reinit(tmp, false);
                apply(tmp, src);
                dst += tmp;
              };
              result.Tvmult_add = result.vmult_add;
              return result;
            });
          builder.saddle_point(lambda, {u}, matrix);
          return std::make_pair(u, lambda);
        },
        "saddle")
      .fields();
  }
} // namespace

TEST(ReplicatedOperators, BOTH_SchurRefinesAgainstOriginalAction)
{
  const auto owned = partition();
  Matrix     unit, coupling;
  initialize(unit, owned, unit_entries);
  Dense coupling_values = values;
  for (auto &row : coupling_values)
    for (auto &value : row)
      value *= .25;
  initialize(coupling, owned, coupling_values);
  auto applications = std::make_shared<unsigned int>(0);
  ParameterAcceptor::clear();
  TimeIntervalParameters time("/refinement/time");
  IDAParameters          parameters("/refinement/ida", &time);
  Adapter                adapter(time, parameters, MPI_COMM_WORLD);
  const auto             fields =
    add_saddle(adapter, owned, unit, applications, false, &coupling);
  auto state = adapter.make_state();
  adapter.set_schur_materialization_limit(3);
  const auto inverse = adapter.schur_preconditioner(fields.second, state);
  EXPECT_EQ(*applications, 3u);
  // Make the cached matrix slightly inexact while keeping the true action
  // linear. Check the physical residual against the original relative target.
  for (const auto i : owned)
    coupling.add(i, i, 1.e-4);
  if (owned.is_element(0))
    coupling.add(0, 1, 1.e-4);
  coupling.compress(VectorOperation::add);
  auto rhs = adapter.make_state(), dst = adapter.make_state();
  auto dot = adapter.make_state(), defect = adapter.make_state();
  rhs = 0.;
  for (const auto i : owned)
    adapter.field(rhs, fields.second)(i) = 4.;
  adapter.field(rhs, fields.second).compress(VectorOperation::insert);
  auto physical = adapter.block_matrix(state, dot, 0.);
  inverse.vmult(dst, rhs);
  physical.vmult(defect, dst);
  defect -= rhs;
  EXPECT_LE(defect.l2_norm(), 1.e-9 * rhs.l2_norm());
  EXPECT_GE(*applications, 8u);
  inverse.Tvmult(dst, rhs);
  physical.Tvmult(defect, dst);
  defect -= rhs;
  EXPECT_LE(defect.l2_norm(), 1.e-9 * rhs.l2_norm());
  EXPECT_GE(*applications, 13u);
}

TEST(ReplicatedOperators, BOTH_IDAPublicSchurLimitAndOriginalActionGuard)
{
  const auto owned = partition();
  Matrix     unit;
  initialize(unit, owned, unit_entries);
  auto applications = std::make_shared<unsigned int>(0);
  ParameterAcceptor::clear();
  TimeIntervalParameters time("/replicated/time");
  IDAParameters          parameters("/replicated/ida", &time);
  Adapter                adapter(time, parameters, MPI_COMM_WORLD);
  const auto fields = add_saddle(adapter, owned, unit, applications, false);
  auto       state  = adapter.make_state();
  adapter.set_schur_materialization_limit(3);
  const auto inverse = adapter.schur_preconditioner(fields.second, state);
  EXPECT_EQ(*applications, 3u);
  auto rhs = adapter.make_state(), dst = adapter.make_state();
  rhs = 0.;
  for (const auto i : owned)
    adapter.field(rhs, fields.second)(i) = 4.;
  adapter.field(rhs, fields.second).compress(VectorOperation::insert);
  inverse.vmult(dst, rhs);
  auto dot      = adapter.make_state();
  auto physical = adapter.block_matrix(state, dot, 0.);
  auto defect   = adapter.make_state();
  physical.vmult(defect, dst);
  defect -= rhs;
  EXPECT_LT(defect.l2_norm(), 1.e-11);
  const auto before = *applications;
  adapter.set_schur_materialization_limit(2);
  const auto uncached = adapter.schur_preconditioner(fields.second, state);
  EXPECT_EQ(*applications, before);
  (void)uncached;

  // Deliberately violate the fixed-action contract. Sampling columns must not
  // silently accept an inner solution that fails the original Schur action.
  Adapter    bad_adapter(time, parameters, MPI_COMM_WORLD);
  auto       count      = std::make_shared<unsigned int>(0);
  const auto bad_fields = add_saddle(bad_adapter, owned, unit, count, true);
  auto       bad_state  = bad_adapter.make_state();
  bad_adapter.set_schur_materialization_limit(3);
  const auto bad_inverse =
    bad_adapter.schur_preconditioner(bad_fields.second, bad_state);
  auto bad_rhs = bad_adapter.make_state(), bad_dst = bad_adapter.make_state();
  bad_rhs = 0.;
  for (const auto i : owned)
    bad_adapter.field(bad_rhs, bad_fields.second)(i) = 4.;
  bad_adapter.field(bad_rhs, bad_fields.second)
    .compress(VectorOperation::insert);
  EXPECT_THROW(bad_inverse.vmult(bad_dst, bad_rhs),
               SolverControl::NoConvergence);
}
#endif
