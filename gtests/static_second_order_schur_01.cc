// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/core/second_order_preconditioner.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <array>
#include <memory>
#include <utility>
#include <vector>

using namespace dealii;
using namespace ImmersX;

namespace
{
  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Matrix       = ImmersXLA::MPI::SparseMatrix;
  using Dense        = FullMatrix<double>;

  const std::array<std::array<double, 3>, 3> mass_values{
    {{{4., 1., 0.5}}, {{1., 3., 0.2}}, {{0.5, 0.2, 2.}}}};
  const std::array<std::array<double, 3>, 3> stiffness_values{
    {{{6., -2., 0.3}}, {{-2., 5., -1.}}, {{0.3, -1., 3.}}}};
  const std::array<std::array<double, 3>, 3> damping_values{
    {{{1., -1., 0.}}, {{-1., 1., 0.}}, {{0., 0., 0.}}}};

  template <typename Values>
  void
  initialize_matrix(Matrix         &matrix,
                    const IndexSet &rows,
                    const IndexSet &columns,
                    const Values   &values)
  {
    DynamicSparsityPattern sparsity(rows.size(), columns.size(), rows);
    for (const auto i : rows)
      for (unsigned int j = 0; j < columns.size(); ++j)
        sparsity.add(i, j);
    matrix.reinit(rows, columns, sparsity, MPI_COMM_WORLD);
    for (const auto i : rows)
      for (unsigned int j = 0; j < columns.size(); ++j)
        matrix.set(i, j, values(i, j));
    matrix.compress(VectorOperation::insert);
  }

  Dense
  spatial_matrix(const std::array<std::array<double, 3>, 3> &values,
                 const bool constrained_identity = false)
  {
    Dense result(3, 3);
    for (unsigned int i = 0; i < 2; ++i)
      for (unsigned int j = 0; j < 2; ++j)
        result(i, j) = values[i][j];
    result(2, 2) = constrained_identity ? 1. : 0.;
    return result;
  }

  Dense
  static_matrix(const Dense &mass,
                const Dense &stiffness,
                const Dense &damping,
                const bool   multiplier)
  {
    Dense result(multiplier ? 8 : 6, multiplier ? 8 : 6);
    for (unsigned int i = 0; i < 2; ++i)
      for (unsigned int j = 0; j < 2; ++j)
        {
          result(i, 3 + j)     = -mass(i, j);
          result(3 + i, j)     = stiffness(i, j);
          result(3 + i, 3 + j) = damping(i, j);
        }
    result(2, 2) = result(5, 5) = 1.;
    if (multiplier)
      for (unsigned int i = 0; i < 2; ++i)
        {
          result(3 + i, 6 + i) = 1.;
          result(6 + i, i)     = 1.;
        }
    return result;
  }

  SemiDiscreteModel<FieldVector>::Operation
  value_operation(FieldVector value)
  {
    SemiDiscreteModel<FieldVector>::Operation result;
    result.reinit_vector = [value](FieldVector &v, const bool) {
      v.reinit(value);
    };
    result.apply     = [value](FieldVector &v) { v.equ(1., value); };
    result.apply_add = [value](FieldVector &v) { v += value; };
    return result;
  }

  dealii::LinearOperator<FieldVector>
  assembling_operator(const Matrix &matrix)
  {
    auto       result = matrix_operator<FieldVector>(matrix).view;
    const auto base   = result;
    result.vmult_add  = [base](FieldVector &dst, const FieldVector &src) {
      FieldVector value;
      base.reinit_range_vector(value, false);
      base.vmult(value, src);
      std::vector<types::global_dof_index> indices;
      std::vector<double>                  values;
      for (const auto i : value.locally_owned_elements())
        {
          indices.push_back(i);
          values.push_back(value[i]);
        }
      if (!indices.empty())
        dst.add(indices, values);
      dst.compress(VectorOperation::add);
    };
    result.Tvmult_add = [base](FieldVector &dst, const FieldVector &src) {
      FieldVector value;
      base.reinit_domain_vector(value, false);
      base.Tvmult(value, src);
      std::vector<types::global_dof_index> indices;
      std::vector<double>                  values;
      for (const auto i : value.locally_owned_elements())
        {
          indices.push_back(i);
          values.push_back(value[i]);
        }
      if (!indices.empty())
        dst.add(indices, values);
      dst.compress(VectorOperation::add);
    };
    return result;
  }

  void
  initialize_multilevel_matrices(Matrix         &mass,
                                 Matrix         &stiffness,
                                 Matrix         &damping,
                                 const IndexSet &owned)
  {
    const auto             size = owned.size();
    DynamicSparsityPattern sparsity(size, size, owned);
    for (const auto i : owned)
      {
        sparsity.add(i, i);
        if (i > 0)
          sparsity.add(i, i - 1);
        if (i + 1 < size)
          sparsity.add(i, i + 1);
      }
    for (auto *matrix : {&mass, &stiffness, &damping})
      matrix->reinit(owned, owned, sparsity, MPI_COMM_WORLD);
    for (const auto i : owned)
      {
        mass.set(i, i, 3. + 0.1 * (i % 7));
        stiffness.set(i, i, 2.05 + 0.01 * (i % 3));
        damping.set(i, i, 0.3 + 0.05 * (i % 5));
        for (const auto j : {i == 0 ? i : i - 1, i + 1 == size ? i : i + 1})
          if (j != i)
            {
              mass.set(i, j, 0.1);
              stiffness.set(i, j, -1.);
            }
      }
    for (auto *matrix : {&mass, &stiffness, &damping})
      matrix->compress(VectorOperation::insert);
  }
} // namespace

TEST(Elastodynamics, BOTH_StaticPairPreservesConstraintsAndTranspose)
{
  const auto owned =
    Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, 3);
  IndexSet constrained(3);
  constrained.add_index(2);
  const auto m = spatial_matrix(mass_values);
  const auto k = spatial_matrix(stiffness_values);
  for (const bool zero_damping : {false, true})
    {
      const auto d =
        zero_damping ? Dense(3, 3) : spatial_matrix(damping_values);
      Matrix mass, stiffness, damping;
      initialize_matrix(mass, owned, owned, m);
      initialize_matrix(stiffness, owned, owned, k);
      initialize_matrix(damping, owned, owned, d);
      SecondOrderBlockPreconditioner<FieldVector> inverse;
      inverse.initialize(mass,
                         stiffness,
                         damping,
                         constrained,
                         [owned](FieldVector &v, const bool) {
                           v.reinit(owned, MPI_COMM_WORLD);
                         });
      inverse.prepare_static_inverse();
      const auto     matrix = static_matrix(m, k, d, false);
      Vector<double> expected(6), rhs(6);
      for (unsigned int i = 0; i < 6; ++i)
        expected[i] = 0.5 + i;
      GlobalVector input(std::vector<IndexSet>(2, owned), MPI_COMM_WORLD);
      GlobalVector output(input);
      for (const bool transpose : {false, true})
        for (const bool fixed : {false, true})
          {
            if (transpose)
              matrix.Tvmult(rhs, expected);
            else
              matrix.vmult(rhs, expected);
            for (const auto i : owned)
              {
                input.block(0)(i) = rhs[i];
                input.block(1)(i) = rhs[3 + i];
              }
            input.compress(VectorOperation::insert);
            if (transpose)
              inverse.static_Tvmult(output.block(0),
                                    output.block(1),
                                    input.block(0),
                                    input.block(1),
                                    fixed);
            else
              inverse.static_vmult(output.block(0),
                                   output.block(1),
                                   input.block(0),
                                   input.block(1),
                                   fixed);
            for (const auto i : owned)
              {
                EXPECT_NEAR(output.block(0)[i], expected[i], 1.e-10);
                EXPECT_NEAR(output.block(1)[i], expected[3 + i], 1.e-10);
              }
          }
    }
}

TEST(Elastodynamics, BOTH_StaticPairFixedAMGAdjoint)
{
  // Exceed the native ML coarse threshold to exercise a multilevel cycle,
  // rather than only an exact tiny coarse solve.
  const unsigned int size = 3000;
  const auto         owned =
    Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,
                                                           size);
  Matrix mass, stiffness, damping;
  initialize_multilevel_matrices(mass, stiffness, damping, owned);
  SecondOrderBlockPreconditioner<FieldVector> inverse;
  inverse.initialize(mass,
                     stiffness,
                     damping,
                     IndexSet(size),
                     [owned](FieldVector &v, const bool) {
                       v.reinit(owned, MPI_COMM_WORLD);
                     });
  inverse.prepare_static_inverse();
  GlobalVector x(std::vector<IndexSet>(2, owned), MPI_COMM_WORLD), y(x), px(x),
    pty(x), sum(x), psum(x);
  for (const auto i : owned)
    for (unsigned int block = 0; block < 2; ++block)
      {
        x.block(block)(i) = 0.1 * (i % 13) - block;
        y.block(block)(i) = 0.2 * (i % 11) + block;
      }
  x.compress(VectorOperation::insert);
  y.compress(VectorOperation::insert);
  inverse.static_vmult(px.block(0), px.block(1), x.block(0), x.block(1), true);
  inverse.static_Tvmult(
    pty.block(0), pty.block(1), y.block(0), y.block(1), true);
  EXPECT_NEAR(y * px,
              pty * x,
              1.e-10 *
                (y.l2_norm() * px.l2_norm() + pty.l2_norm() * x.l2_norm()));
  sum = x;
  sum += y;
  inverse.static_vmult(
    psum.block(0), psum.block(1), sum.block(0), sum.block(1), true);
  inverse.static_vmult(
    pty.block(0), pty.block(1), y.block(0), y.block(1), true);
  psum -= px;
  psum -= pty;
  EXPECT_LE(psum.l2_norm(), 1.e-11 * (px.l2_norm() + pty.l2_norm()));
}

#ifdef DEAL_II_WITH_SUNDIALS
namespace
{
  void
  check_multilevel_schur_multiplier_row(const double alpha)
  {
    SCOPED_TRACE(::testing::Message() << "alpha=" << alpha);
    using Adapter           = IDAAdapter<FieldVector, GlobalVector>;
    const unsigned int size = 3000;
    const auto         owned =
      Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,
                                                             size);
    const auto multiplier_owned =
      Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, 2);
    const auto modes = [](const auto i, const unsigned int column) {
      return column == 0 ? 1. + 0.1 * (i % 7) :
                           (i % 2 == 0 ? -1. : 1.) + 0.03 * (i % 11);
    };
    Matrix mass, stiffness, damping, force, trace, self, metric;
    initialize_multilevel_matrices(mass, stiffness, damping, owned);
    initialize_matrix(force, owned, multiplier_owned, modes);
    initialize_matrix(trace,
                      multiplier_owned,
                      owned,
                      [&](const auto row, const auto column) {
                        return modes(column, row) / size;
                      });
    initialize_matrix(self,
                      multiplier_owned,
                      multiplier_owned,
                      [](const auto row, const auto column) {
                        return row == column ? -0.05 : 0.;
                      });
    initialize_matrix(metric,
                      multiplier_owned,
                      multiplier_owned,
                      [](const auto row, const auto column) {
                        return row == column ? 1. : 0.;
                      });
    const auto reinit = [owned](FieldVector &v, const bool) {
      v.reinit(owned, MPI_COMM_WORLD);
    };
    FieldVector b(owned, MPI_COMM_WORLD), zero(b), fixed_u(b), fixed_v(b),
      error(b);
    for (const auto i : owned)
      b(i) = modes(i, 0) + 0.3 * modes(i, 1);
    b.compress(VectorOperation::insert);
    SecondOrderBlockPreconditioner<FieldVector> fixed_inverse;
    fixed_inverse.initialize(mass, stiffness, damping, IndexSet(size), reinit);
    if (alpha == 0.)
      {
        fixed_inverse.prepare_static_inverse();
        fixed_inverse.static_vmult(fixed_u, fixed_v, zero, b, true);
        stiffness.vmult(error, fixed_u);
      }
    else
      {
        fixed_inverse.set_coefficients(alpha, 1., 1., 1.);
        fixed_inverse.apply_schur_amg(fixed_u, b);
        mass.vmult(error, fixed_u);
        error *= alpha;
        damping.vmult_add(error, fixed_u);
        stiffness.vmult(fixed_v, fixed_u);
        error.add(1. / alpha, fixed_v);
      }
    error -= b;
    const double amg_residual = error.l2_norm() / b.l2_norm();
    // An exact coarse solve would hide different Schur and recovery actions.
    ASSERT_GT(amg_residual, 1.e-8);
    SCOPED_TRACE(::testing::Message()
                 << "one-cycle AMG relative residual=" << amg_residual);

    const auto contribute = [&](auto &builder) {
      const auto u      = builder.field("u", owned, owned, owned);
      const auto v      = builder.field("v", owned, owned, owned);
      const auto lambda = builder.algebraic_field("lambda", multiplier_owned);
      const auto mass_operator = builder.matrix_operator(mass);
      builder.term(u, "kinematic")
        .derivative(u, mass_operator)
        .state(v, -1. * mass_operator);
      builder.term(v, "momentum")
        .derivative(v, mass_operator)
        .state(u, builder.matrix_operator(stiffness))
        .state(v, builder.matrix_operator(damping))
        .state(lambda, assembling_operator(force));
      builder.term(lambda, "compatibility")
        .state(u, assembling_operator(trace))
        .state(lambda, builder.matrix_operator(self));
      builder.second_order_pair(
        u,
        v,
        [] { return 0ul; },
        [size] {
          SecondOrderPreconditionerData data;
          data.constrained = IndexSet(size);
          return data;
        });
      builder.saddle_point(lambda, {u, v}, builder.matrix_operator(metric));
      return std::array<FieldId, 3>{{u, v, lambda}};
    };
    Adapter::Composition composition(MPI_COMM_WORLD);
    const auto           fields  = composition.add(contribute, "solid");
    auto                 state   = composition.make_state();
    auto                 dot     = composition.make_state();
    const auto           inverse = composition.schur_preconditioner(
      fields[2], state, &dot, alpha, 100, 1.e-12, true);
    auto  rhs        = composition.make_state();
    auto  result     = composition.make_state();
    auto &lambda_rhs = composition.field(rhs, fields[2]);
    for (const auto i : multiplier_owned)
      lambda_rhs(i) = i == 0 ? 0.4 : -0.7;
    rhs.compress(VectorOperation::insert);
    const auto n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const auto empty_ranks =
      Utilities::MPI::sum(static_cast<unsigned int>(
                            multiplier_owned.n_elements() == 0),
                          MPI_COMM_WORLD);
    EXPECT_EQ(empty_ranks, n_ranks > 2 ? n_ranks - 2 : 0);

    for (const bool transpose : {false, true})
      {
        SCOPED_TRACE(transpose ? "transpose" : "forward");
        if (transpose)
          inverse.Tvmult(result, rhs);
        else
          inverse.vmult(result, rhs);
        FieldVector row(multiplier_owned, MPI_COMM_WORLD);
        if (transpose)
          {
            force.Tvmult(row, composition.field(result, fields[1]));
            self.Tvmult_add(row, composition.field(result, fields[2]));
          }
        else
          {
            trace.vmult(row, composition.field(result, fields[0]));
            self.vmult_add(row, composition.field(result, fields[2]));
          }
        row -= lambda_rhs;
        // This physical row oracle does not construct its RHS through Schur.
        EXPECT_LE(row.l2_norm(), 1.e-10 * lambda_rhs.l2_norm());
      }
  }
} // namespace

TEST(DistributedIDA, BOTH_StaticPairMultilevelSchurPreservesMultiplierRow)
{
  check_multilevel_schur_multiplier_row(0.);
}

TEST(DistributedIDA, BOTH_PositivePairMultilevelSchurPreservesMultiplierRow)
{
  check_multilevel_schur_multiplier_row(0.125);
}

TEST(DistributedIDA, BOTH_StaticPairSchurAndConsistentInitialConditions)
{
  using Adapter = IDAAdapter<FieldVector, GlobalVector>;
  const auto owned =
    Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, 3);
  const auto multiplier_owned =
    Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, 2);
  IndexSet constrained(3);
  constrained.add_index(2);
  IndexSet free = owned;
  free.subtract_set(constrained);
  const auto m = spatial_matrix(mass_values);
  const auto k = spatial_matrix(stiffness_values);
  const auto d = spatial_matrix(damping_values, true);
  Dense      q(3, 3), b(2, 3), c(3, 2), w(2, 2);
  q(2, 2) = 1.;
  for (unsigned int i = 0; i < 2; ++i)
    b(i, i) = c(i, i) = w(i, i) = 1.;
  Matrix mass, stiffness, damping, constraints, trace, force, metric;
  initialize_matrix(mass, owned, owned, m);
  initialize_matrix(stiffness, owned, owned, k);
  initialize_matrix(damping, owned, owned, d);
  initialize_matrix(constraints, owned, owned, q);
  initialize_matrix(trace, multiplier_owned, owned, b);
  initialize_matrix(force, owned, multiplier_owned, c);
  initialize_matrix(metric, multiplier_owned, multiplier_owned, w);
  const auto     matrix = static_matrix(m, k, d, true);
  Vector<double> equilibrium(8), forcing(8);
  equilibrium[0] = 1.;
  equilibrium[1] = 2.;
  equilibrium[2] = 3.;
  equilibrium[6] = 4.;
  equilibrium[7] = 5.;
  matrix.vmult(forcing, equilibrium);
  FieldVector f_u(owned, MPI_COMM_WORLD), f_v(owned, MPI_COMM_WORLD),
    f_lambda(multiplier_owned, MPI_COMM_WORLD);
  for (const auto i : owned)
    {
      f_u(i) = forcing[i];
      f_v(i) = forcing[3 + i];
    }
  for (const auto i : multiplier_owned)
    f_lambda(i) = forcing[6 + i];
  f_u.compress(VectorOperation::insert);
  f_v.compress(VectorOperation::insert);
  f_lambda.compress(VectorOperation::insert);
  const auto contribute = [&](auto &builder) {
    const auto u      = builder.field("u", owned, owned, free);
    const auto v      = builder.field("v", owned, owned, free);
    const auto lambda = builder.algebraic_field("lambda", multiplier_owned);
    auto       mass_operator  = builder.matrix_operator(mass);
    const auto force_operator = assembling_operator(force);
    const auto trace_operator = assembling_operator(trace);
    builder.term(u, "kinematic")
      .residual([&, u, v](const auto &context) {
        FieldVector value(owned, MPI_COMM_WORLD), temporary(value);
        mass.vmult(value, context.derivative(u));
        mass.vmult(temporary, context.state(v));
        value -= temporary;
        constraints.vmult_add(value, context.state(u));
        value -= f_u;
        return value_operation(std::move(value));
      })
      .derivative(u, mass_operator)
      .state(v, -1. * mass_operator)
      .state(u, builder.matrix_operator(constraints));
    builder.term(v, "momentum")
      .residual([&, u, v, lambda](const auto &context) {
        FieldVector value(owned, MPI_COMM_WORLD);
        mass.vmult(value, context.derivative(v));
        stiffness.vmult_add(value, context.state(u));
        damping.vmult_add(value, context.state(v));
        force.vmult_add(value, context.state(lambda));
        value -= f_v;
        return value_operation(std::move(value));
      })
      .derivative(v, mass_operator)
      .state(u, builder.matrix_operator(stiffness))
      .state(v, builder.matrix_operator(damping))
      .state(lambda, force_operator);
    builder.term(lambda, "compatibility")
      .residual([&, u](const auto &context) {
        FieldVector value(multiplier_owned, MPI_COMM_WORLD);
        trace.vmult(value, context.state(u));
        value -= f_lambda;
        return value_operation(std::move(value));
      })
      .state(u, trace_operator);
    builder.second_order_pair(
      u,
      v,
      [] { return 0ul; },
      [constrained] {
        SecondOrderPreconditionerData data;
        data.constrained = constrained;
        return data;
      });
    builder.saddle_point(lambda, {u, v}, builder.matrix_operator(metric));
    return std::array<FieldId, 3>{{u, v, lambda}};
  };

  Adapter::Composition composition(MPI_COMM_WORLD);
  const auto           fields  = composition.add(contribute, "solid");
  auto                 state   = composition.make_state();
  auto                 dot     = composition.make_state();
  const auto           inverse = composition.schur_preconditioner(
    fields[2], state, &dot, 0., 100, 1.e-12, true);
  auto           rhs    = composition.make_state();
  auto           result = composition.make_state();
  Vector<double> expected(8), dense_rhs(8);
  for (unsigned int i = 0; i < 8; ++i)
    expected[i] = 0.25 + i;
  for (const bool transpose : {false, true})
    {
      if (transpose)
        matrix.Tvmult(dense_rhs, expected);
      else
        matrix.vmult(dense_rhs, expected);
      for (unsigned int field = 0; field < 3; ++field)
        {
          auto &part = composition.field(rhs, fields[field]);
          for (const auto i : part.locally_owned_elements())
            part(i) = dense_rhs[3 * field + i];
        }
      rhs.compress(VectorOperation::insert);
      if (transpose)
        inverse.Tvmult(result, rhs);
      else
        inverse.vmult(result, rhs);
      for (unsigned int field = 0; field < 3; ++field)
        for (const auto i :
             composition.field(result, fields[field]).locally_owned_elements())
          EXPECT_NEAR(composition.field(result, fields[field])[i],
                      expected[3 * field + i],
                      1.e-9);
    }
  const auto schur =
    composition.schur_operator(fields[2], state, &dot, 0., true);
  FieldVector x(multiplier_owned, MPI_COMM_WORLD), y(x), action_x(x),
    action_y(x), sum(x), action_sum(x);
  for (const auto i : multiplier_owned)
    {
      x(i) = 1. + i;
      y(i) = 0.3 - 0.7 * i;
    }
  sum = x;
  sum += y;
  schur.vmult(action_x, x);
  schur.vmult(action_y, y);
  schur.vmult(action_sum, sum);
  action_sum -= action_x;
  action_sum -= action_y;
  EXPECT_LE(action_sum.l2_norm(),
            1.e-12 * (action_x.l2_norm() + action_y.l2_norm()));
  schur.Tvmult(action_sum, x);
  EXPECT_NEAR(x * action_y, action_sum * y, 1.e-12);
  // Epetra Import leaves an Insert state. The Schur overwrite must reset its
  // multiplier destination before assembling Add contributions, including
  // ranks with no multiplier rows (two such ranks in the MPI4 gate).
  IndexSet alternate(2);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    alternate.add_range(0, 2);
  FieldVector imported(multiplier_owned, MPI_COMM_WORLD),
    imported_source(alternate, MPI_COMM_WORLD);
  imported_source = 1.;
  imported_source.compress(VectorOperation::insert);
  imported = imported_source;
  schur.vmult(imported, x);
  imported -= action_x;
  EXPECT_LT(imported.l2_norm(), 1.e-12);
  imported = imported_source;
  schur.Tvmult(imported, x);
  imported -= action_sum;
  EXPECT_LT(imported.l2_norm(), 1.e-12);

  TimeIntervalParameters time;
  time.initial_time         = 0.;
  time.final_time           = 0.002;
  time.output_time_interval = 0.001;
  IDAParameters parameters;
  parameters.initial_step_size                = 0.001;
  parameters.maximum_order                    = 2;
  parameters.minimum_step_size                = 1.e-10;
  parameters.maximum_non_linear_iterations_ic = 30;
  parameters.absolute_tolerance               = 1.e-9;
  parameters.relative_tolerance               = 1.e-8;
  parameters.correction_type_at_initial_time  = "use_y_dot";
  parameters.correction_type_after_restart    = "none";
  Adapter adapter(time, parameters, MPI_COMM_WORLD);
  adapter.set_schur_primal_coupling();
  const auto ids       = adapter.add(contribute, "solid").fields();
  auto       y_state   = adapter.make_state();
  auto       y_dot     = adapter.make_state();
  bool       corrected = false;
  adapter.set_output_step([&](const double,
                              const GlobalVector &value,
                              const GlobalVector &,
                              const unsigned int step) {
    if (step == 0)
      corrected = true;
    for (unsigned int field = 0; field < 3; ++field)
      for (const auto i :
           adapter.field(value, ids[field]).locally_owned_elements())
        EXPECT_NEAR(adapter.field(value, ids[field])[i],
                    equilibrium[3 * field + i],
                    1.e-6);
  });
  EXPECT_GT(adapter.solve(y_state, y_dot), 0u);
  EXPECT_TRUE(corrected);
  EXPECT_GT(adapter.field(y_state, ids[0]).l2_norm(), 1.);
  EXPECT_GT(adapter.field(y_state, ids[2]).l2_norm(), 1.);
}
#endif
