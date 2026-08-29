#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/linear_adapter.h>

namespace ImmersX
{
  struct FakeProblem
  {
    ImmersXLA::MPI::SparseMatrix matrix;
    ImmersXLA::MPI::Vector       rhs;
    dealii::IndexSet             owned;
  };

  struct FakeFields
  {
    ImmersX::FieldId solution;
  };
} // namespace ImmersX

namespace ImmersX
{
  template <typename Builder>
  FakeFields
  contribute(Builder &builder, const FakeProblem &problem)
  {
    const auto solution = builder.algebraic_field("solution", problem.owned);
    using VectorType    = ImmersXLA::MPI::Vector;
    const auto matrix   = builder.matrix_operator(problem.matrix);
    builder.preconditioner(
      solution, [](const auto &linearized_matrix, const auto &reinit_vector) {
        return make_amg_preconditioner(linearized_matrix, reinit_vector);
      });
    builder.term(solution, "fake")
      .residual([solution, &problem](const auto &context) {
        const auto &state = context.state(solution);
        dealii::PackagedOperation<ImmersXLA::MPI::Vector> operation;
        operation.reinit_vector = [state](auto &vector, const bool omit) {
          vector.reinit(state, omit);
        };
        operation.apply = [&problem, &state](auto &vector) {
          problem.matrix.vmult(vector, state);
          vector -= problem.rhs;
        };
        operation.apply_add = [&problem, &state](auto &vector) {
          ImmersXLA::MPI::Vector contribution;
          contribution.reinit(state);
          problem.matrix.vmult(contribution, state);
          contribution -= problem.rhs;
          vector += contribution;
        };
        return operation;
      })
      .state(solution, matrix);
    return {solution};
  }
} // namespace ImmersX

TEST(LinearAdapter, DirectContributorAndSemanticFieldAccess)
{
  ImmersX::FakeProblem problem;
  problem.owned = dealii::IndexSet(2);
  problem.owned.add_range(0, 2);
  problem.owned.compress();
  dealii::DynamicSparsityPattern sparsity(2, 2);
  sparsity.add(0, 0);
  sparsity.add(1, 1);
  problem.matrix.reinit(problem.owned, problem.owned, sparsity, MPI_COMM_WORLD);
  problem.matrix.set(0, 0, 2.);
  problem.matrix.set(1, 1, 4.);
  problem.matrix.compress(dealii::VectorOperation::insert);
  problem.rhs.reinit(problem.owned, MPI_COMM_WORLD);
  problem.rhs[0] = 2.;
  problem.rhs[1] = 8.;
  using Adapter  = ImmersX::LinearAdapter<ImmersX::ImmersXLA::MPI::Vector,
                                         ImmersX::ImmersXLA::MPI::BlockVector>;
  Adapter    adapter(MPI_COMM_WORLD,
                  [](const auto &operator_view,
                     const auto &rhs,
                     auto       &solution) {
                    (void)operator_view;
                    solution = rhs;
                    solution.block(0)[0] /= 2.;
                    solution.block(0)[1] /= 4.;
                  });
  const auto fields   = adapter.add(problem, "fake");
  const auto observed = fields.observe(fields.fields().solution);
  auto       state    = adapter.make_state();
  EXPECT_EQ(observed.source(), fields.fields().solution);
  EXPECT_EQ(state.n_blocks(), 1u);
  adapter.solve(state);

  EXPECT_NEAR(adapter.field(state, fields.fields().solution)[0], 1., 1.e-12);
  EXPECT_NEAR(adapter.field(state, fields.fields().solution)[1], 2., 1.e-12);

  EXPECT_TRUE(adapter.has_local_preconditioner(fields.fields().solution));
  const auto local =
    adapter.local_preconditioner(fields.fields().solution, state);
  ASSERT_TRUE(local.has_value());
  const auto input          = adapter.field(state, fields.fields().solution);
  auto       preconditioned = input;
  (*local).vmult(preconditioned, input);
  EXPECT_GT(preconditioned.l2_norm(), 0.);
  const auto block_diagonal = adapter.block_diagonal_preconditioner(state);
  auto       block_diagonal_action = adapter.make_state();
  block_diagonal.vmult(block_diagonal_action, state);
  EXPECT_GT(block_diagonal_action.l2_norm(), 0.);

  EXPECT_TRUE(adapter.can_materialize_matrix(state));
  const auto block_matrix = adapter.block_matrix(state);
  EXPECT_EQ(block_matrix.n_block_rows(), 1u);
  const auto monolithic = adapter.monolithic_matrix(state);
  EXPECT_EQ(monolithic.m(), 2u);
  auto block_action = adapter.make_state();
  block_matrix.vmult(block_action, state);
  auto operator_action = adapter.make_state();
  adapter.jacobian(state).vmult(operator_action, state);
  EXPECT_NEAR(adapter.field(block_action, fields.fields().solution)[0],
              adapter.field(operator_action, fields.fields().solution)[0],
              1.e-12);
  EXPECT_NEAR(adapter.field(block_action, fields.fields().solution)[1],
              adapter.field(operator_action, fields.fields().solution)[1],
              1.e-12);
  const auto                      flat = adapter.pack(state);
  ImmersX::ImmersXLA::MPI::Vector monolithic_action;
  monolithic_action.reinit(flat);
  monolithic.vmult(monolithic_action, flat);
  EXPECT_NEAR(monolithic_action[0],
              adapter.field(operator_action, fields.fields().solution)[0],
              1.e-12);
  EXPECT_NEAR(monolithic_action[1],
              adapter.field(operator_action, fields.fields().solution)[1],
              1.e-12);
  auto unpacked = adapter.make_state();
  adapter.unpack(flat, unpacked);
  EXPECT_NEAR(adapter.field(unpacked, fields.fields().solution)[0], 1., 1.e-12);
  EXPECT_NEAR(adapter.field(unpacked, fields.fields().solution)[1], 2., 1.e-12);

  adapter.solve_direct(state);
  EXPECT_NEAR(adapter.field(state, fields.fields().solution)[0], 1., 1.e-12);
  EXPECT_NEAR(adapter.field(state, fields.fields().solution)[1], 2., 1.e-12);

  ImmersX::ImmersXLA::MPI::BlockVector residual;
  adapter.evaluate_residual(state, residual);
  EXPECT_NEAR(residual.l2_norm(), 0., 1.e-12);
}
