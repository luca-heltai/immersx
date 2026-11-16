// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/mpi.h>

#include <gtest/gtest.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <array>
#if defined(DEAL_II_WITH_SUNDIALS) && !defined(IMMERSX_USE_PETSC_LA)
using namespace ImmersX;
using Vector  = ImmersXLA::MPI::Vector;
using Global  = ImmersXLA::MPI::BlockVector;
using Matrix  = ImmersXLA::MPI::SparseMatrix;
using Adapter = IDAAdapter<Vector, Global>;
TEST(DistributedIDA, BOTH_SteadyScalingWithoutLinearSolverSetup)
{
  for (const auto solver : {Adapter::LinearSolver::mumps,
                            Adapter::LinearSolver::mumps_blr_fgmres,
                            Adapter::LinearSolver::iterative})
    {
      dealii::ParameterAcceptor::clear();
      TimeIntervalParameters time("/", false);
      IDAParameters          parameters("/", &time);
      Adapter                adapter(time, parameters, MPI_COMM_WORLD);
      adapter.set_linear_solver(solver);
      const auto registration = adapter.add([](auto &builder) {
        using Model = typename std::decay_t<decltype(builder)>::Model;
        dealii::IndexSet owned(2);
        const auto       rank =
                     dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
                   size =
                     dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
        for (unsigned int i = 0; i < 2; ++i)
          if (i % size == rank)
            owned.add_index(i);
        owned.compress();
        const std::array<FieldId, 4> fields{
          {builder.differential_field("u", owned),
           builder.differential_field("v", owned),
           builder.differential_field("a", owned),
           builder.algebraic_field("lambda", owned)}};
        const auto diagonal =
          [owned](const std::function<double(unsigned int)> &value) {
            dealii::DynamicSparsityPattern pattern(2, 2, owned);
            for (const auto i : owned)
              pattern.add(i, i);
            auto matrix = std::make_shared<Matrix>();
            matrix->reinit(owned, owned, pattern, MPI_COMM_WORLD);
            for (const auto i : owned)
              matrix->set(i, i, value(i));
            matrix->compress(dealii::VectorOperation::insert);
            return matrix;
          };
        for (unsigned int row = 0; row < 4; ++row)
          {
            auto term =
              builder.term(fields[row], "analytic row" + std::to_string(row));
            typename Model::ResidualFactory residual =
              [=](const auto &context) {
                auto values = std::make_shared<Vector>();
                values->reinit(owned, MPI_COMM_WORLD);
                for (const auto i : owned)
                  {
                    const double u = context.state(fields[0])[i],
                                 v = context.state(fields[1])[i],
                                 a = context.state(fields[2])[i],
                                 l = context.state(fields[3])[i];
                    (*values)[i] =
                      row == 0 ?
                        2. * (context.derivative(fields[0])[i] - v) :
                      row == 1 ?
                        4. * context.derivative(fields[1])[i] + u + l - 2. :
                      row == 2 ? a * a - l :
                                 u - a;
                  }
                values->compress(dealii::VectorOperation::insert);
                typename Model::Operation op;
                op.reinit_vector = [owned](Vector &v, const bool omit) {
                  v.reinit(owned, MPI_COMM_WORLD, omit);
                };
                op.apply     = [values](Vector &v) { v = *values; };
                op.apply_add = [values](Vector &v) { v += *values; };
                return op;
              };
            term.residual(residual);
            for (unsigned int col = 0; col < 4; ++col)
              if ((row == 0 && col == 1) ||
                  (row == 1 && (col == 0 || col == 3)) ||
                  (row == 2 && (col == 2 || col == 3)) ||
                  (row == 3 && (col == 0 || col == 2)))
                {
                  typename Model::MatrixOperatorFactory matrix =
                    [=](const auto &context) {
                      return matrix_operator<Vector>(diagonal([&](
                                                                const auto i) {
                        return row == 0 ? -2. :
                               row == 1 ? 1. :
                               row == 2 ? (col == 2 ?
                                             2. * context.state(fields[2])[i] :
                                             -1.) :
                                          (col == 0 ? 1. : -1.);
                      }));
                    };
                  term.state(fields[col], matrix);
                }
            if (row <= 1)
              {
                auto entries =
                  diagonal([=](auto) { return row == 0 ? 2. : 4.; });
                typename Model::MatrixOperatorFactory mass =
                  [entries](const auto &) {
                    return matrix_operator<Vector>(entries);
                  };
                term.derivative(fields[row], mass);
              }
          }
        builder.second_order_pair(
          fields[0],
          fields[1],
          [] { return 0ul; },
          [] {
            SecondOrderPreconditionerData d;
            d.constrained = dealii::IndexSet(2);
            return d;
          });
        return fields;
      });

      const auto &fields = registration.fields();
      auto        y = adapter.make_state(), yd = adapter.make_state(),
           chars = adapter.make_state(), scaling = adapter.make_state();
      yd = .125;
      const std::array<double, 4> initial{{.5, 0., .5, 0.}},
        characteristics{{2., 3., 5., 7.}},
        expected{{1. / 6., 1. / 9., 1. / 12., 1. / 7.}};
      for (unsigned int k = 0; k < 4; ++k)
        {
          adapter.field(y, fields[k])     = initial[k];
          adapter.field(chars, fields[k]) = characteristics[k];
        }
      y.compress(dealii::VectorOperation::insert);
      chars.compress(dealii::VectorOperation::insert);
      // At a=.5 the full alpha=0 row sums are [2*c_v,c_u+c_l,
      // 2*a*c_a+c_l,c_u+c_a] = [6,9,12,7]. Kinematic rows are
      // retained even though the registered second-order pair can condense.
      // MPI4 includes two empty ranks; semantic execution ownership has holes.
      const auto before     = y;
      const auto before_dot = yd;
      adapter.steady_function_scaling(0., y, yd, chars, scaling);
      EXPECT_THROW(adapter.current_jacobian(), dealii::ExceptionBase);
      EXPECT_FALSE(adapter.has_current_preconditioner());
      EXPECT_EQ(adapter.statistics().jacobian_setups, 0u);
      EXPECT_EQ(adapter.statistics().linear_solves, 0u);
      EXPECT_EQ(adapter.statistics().kinematic_factorizations, 0u);
      for (unsigned int k = 0; k < 4; ++k)
        for (const auto i :
             adapter.field(scaling, fields[k]).locally_owned_elements())
          EXPECT_NEAR(adapter.field(scaling, fields[k])[i], expected[k], 1e-15);
      auto difference = y;
      difference -= before;
      EXPECT_DOUBLE_EQ(difference.l2_norm(), 0.);
      auto rate_difference = yd;
      rate_difference -= before_dot;
      EXPECT_DOUBLE_EQ(rate_difference.l2_norm(), 0.);
    }
  dealii::ParameterAcceptor::clear();
}
#endif
