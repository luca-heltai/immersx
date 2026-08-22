#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
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
    const auto solution =
      builder.field("solution", TimeRole::algebraic, problem.owned);
    using VectorType  = ImmersXLA::MPI::Vector;
    const auto matrix = ImmersX::payload_free(
      dealii::linear_operator<VectorType, VectorType>(problem.matrix));
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
  const auto fields = adapter.add(problem, "fake");
  auto       state  = adapter.make_state();
  adapter.solve(state);

  EXPECT_NEAR(adapter.field(state, fields.solution)[0], 1., 1.e-12);
  EXPECT_NEAR(adapter.field(state, fields.solution)[1], 2., 1.e-12);

  ImmersX::ImmersXLA::MPI::BlockVector residual;
  adapter.evaluate_residual(state, residual);
  EXPECT_NEAR(residual.l2_norm(), 0., 1.e-12);
}
