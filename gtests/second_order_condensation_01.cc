// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/detail/second_order_condensation.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <functional>
#include <vector>

using namespace dealii;

namespace
{
  using FieldVector  = ImmersX::ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Matrix       = ImmersX::ImmersXLA::MPI::SparseMatrix;
  using BlockMatrix  = ImmersX::ImmersXLA::MPI::BlockSparseMatrix;
  using Condensation =
    ImmersX::detail::SecondOrderCondensation<Matrix, FieldVector>;

  constexpr unsigned int n           = 7; // displacement and velocity size
  constexpr unsigned int m           = 2; // multiplier size
  constexpr unsigned int constrained = 4; // one constrained displacement DoF

  // With condensed_mass the constrained mass row and column are removed;
  // otherwise free mass rows also reach the constrained column.
  bool condensed_mass = true;

  double
  mass(const unsigned int i, const unsigned int j)
  {
    if (i == constrained || (condensed_mass && j == constrained))
      return 0.;
    return i == j ? 4. + 0.1 * i : (i + 1 == j || j + 1 == i ? 1. : 0.);
  }

  double
  stiffness(const unsigned int i, const unsigned int j)
  {
    if (i == constrained || j == constrained)
      return 0.;
    return i == j ? 6. : (i + 1 == j || j + 1 == i ? -2. : 0.);
  }

  double
  damping(const unsigned int i, const unsigned int j)
  {
    return 0.05 * mass(i, j);
  }

  // The multiplier rows see the displacement, including the constrained DoF.
  double
  coupling(const unsigned int row, const unsigned int column)
  {
    return row == 0 ? (column < 3 ? 1. + column : 0.) :
                      (column >= 3 ? 0.5 * column : 0.);
  }

  void
  fill_block(BlockMatrix                 &jacobian,
             const unsigned int           row_block,
             const unsigned int           column_block,
             const std::vector<IndexSet> &partitions,
             const std::function<double(unsigned int, unsigned int)> &value)
  {
    const auto            &rows    = partitions[row_block];
    const auto            &columns = partitions[column_block];
    DynamicSparsityPattern sparsity(rows.size(), columns.size(), rows);
    for (const auto row : rows)
      for (unsigned int column = 0; column < columns.size(); ++column)
        if (value(row, column) != 0.)
          sparsity.add(row, column);
    auto &block = jacobian.block(row_block, column_block);
    block.reinit(rows, columns, sparsity, MPI_COMM_WORLD);
    for (const auto row : rows)
      for (unsigned int column = 0; column < columns.size(); ++column)
        if (value(row, column) != 0.)
          block.set(row, column, value(row, column));
    block.compress(VectorOperation::insert);
  }

  // IDA's J = dF/dy + alpha dF/dydot for M u' - M v = 0 and
  // M v' + D v + K u + B^T lambda = f, B u - eps lambda = g.
  void
  assemble(BlockMatrix                 &jacobian,
           const std::vector<IndexSet> &partitions,
           const double                 alpha,
           const double                 kinematic_perturbation = 0.,
           const bool                   unit_velocity_row      = true)
  {
    jacobian.reinit(3, 3);
    const auto zero = [](unsigned int, unsigned int) { return 0.; };
    const auto free = [](const unsigned int i) { return i != constrained; };
    fill_block(jacobian, 0, 0, partitions, [&](auto i, auto j) {
      return free(i) ? alpha * mass(i, j) : (i == j ? 2.5 : 0.);
    });
    fill_block(jacobian, 0, 1, partitions, [&](auto i, auto j) {
      return free(i) ? -mass(i, j) *
                         (i == 0 && j == 0 ? 1. + kinematic_perturbation : 1.) :
                       0.;
    });
    fill_block(jacobian, 0, 2, partitions, zero);
    fill_block(jacobian, 1, 0, partitions, [&](auto i, auto j) {
      if (!unit_velocity_row && i == constrained && j + 1 == constrained)
        return 0.5;
      return free(i) ? stiffness(i, j) : 0.;
    });
    fill_block(jacobian, 1, 1, partitions, [&](auto i, auto j) {
      return free(i) ? alpha * mass(i, j) + damping(i, j) : (i == j ? 1. : 0.);
    });
    fill_block(jacobian, 1, 2, partitions, [&](auto i, auto j) {
      return free(i) ? coupling(j, i) : 0.;
    });
    fill_block(jacobian, 2, 0, partitions, coupling);
    fill_block(jacobian, 2, 1, partitions, zero);
    fill_block(jacobian, 2, 2, partitions, [](auto i, auto j) {
      return i == j ? -1.e-2 : 0.;
    });
    jacobian.collect_sizes();
  }

  void
  solve(const Matrix &matrix, FieldVector &solution, const FieldVector &rhs)
  {
    SolverControl            control(500, 1.e-14 * rhs.l2_norm());
    SolverGMRES<FieldVector> solver(control);
    solution.reinit(rhs);
    solver.solve(matrix, solution, rhs, PreconditionIdentity());
  }

  std::vector<IndexSet>
  partitions()
  {
    return {
      Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, n),
      Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, n),
      Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,
                                                             m)};
  }
} // namespace


TEST(SecondOrderCondensation, BOTH_ReducedSolveMatchesCoupledJacobian)
{
  const auto layout = partitions();
  for (const bool condensed : {true, false})
    {
      condensed_mass = condensed;
      std::shared_ptr<const Condensation> previous;
      for (const double alpha : {3., 0.})
        {
          SCOPED_TRACE(alpha);
          SCOPED_TRACE(condensed);
          BlockMatrix jacobian;
          assemble(jacobian, layout, alpha);
          const auto condensation =
            Condensation::build(jacobian, {{0, 1}}, layout, MPI_COMM_WORLD);
          ASSERT_TRUE(condensation);
          ASSERT_EQ(condensation->n_pairs(), 1u);
          EXPECT_EQ(condensation->reduced_matrix().m(), n + m);
          // The kinematic mass does not depend on the derivative weight.
          if (previous)
            {
              EXPECT_TRUE(previous->kinematic_mass(0) ==
                          condensation->kinematic_mass(0));
            }
          previous = condensation;

          GlobalVector rhs(layout, MPI_COMM_WORLD);
          for (unsigned int block = 0; block < 3; ++block)
            for (const auto index : layout[block])
              rhs.block(block)(index) = 1. + 0.3 * index - 0.7 * block;
          rhs.compress(VectorOperation::insert);

          const auto mass_matrix = condensation->kinematic_mass(0).make_matrix(
            condensation->kinematic_partition(0), MPI_COMM_WORLD);
          FieldVector              kinematic_rhs;
          std::vector<FieldVector> kinematic(1);
          condensation->kinematic_rhs(rhs, 0, kinematic_rhs);
          solve(*mass_matrix, kinematic[0], kinematic_rhs);
          condensation->finish_kinematic_solution(rhs, 0, kinematic[0]);
          FieldVector reduced_rhs, reduced_solution;
          condensation->reduced_rhs(rhs, kinematic, reduced_rhs);
          solve(condensation->reduced_matrix(), reduced_solution, reduced_rhs);
          GlobalVector solution(layout, MPI_COMM_WORLD);
          condensation->expand(reduced_solution, kinematic, solution);

          GlobalVector residual(layout, MPI_COMM_WORLD);
          jacobian.vmult(residual, solution);
          residual -= rhs;
          EXPECT_LT(residual.l2_norm(), 1.e-10 * rhs.l2_norm());
        }
    }
  condensed_mass = true;
}


#ifdef DEAL_II_WITH_SUNDIALS
// Unlike contiguous field ownership, cyclic ownership leaves gaps between
// owned sparse rows and requires importing constrained flags from other ranks.
TEST(SecondOrderCondensation, BOTH_RepeatedCyclicOwnershipPreservesRemoteFlags)
{
  using Adapter              = ImmersX::IDAAdapter<FieldVector, GlobalVector>;
  const auto            rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto            ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  std::vector<IndexSet> layout{IndexSet(n), IndexSet(n), IndexSet(m)};
  for (auto &owned : layout)
    {
      for (unsigned int i = 0; i < owned.size(); ++i)
        if (i % ranks == rank)
          owned.add_index(i);
      owned.compress();
    }
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(layout[2].is_empty()),
                                MPI_COMM_WORLD),
            ranks > m ? ranks - m : 0u);
  BlockMatrix          jacobian;
  Adapter::Composition composition(MPI_COMM_WORLD);
  composition.add(
    [&](auto &builder) {
      using Model = typename std::decay_t<decltype(builder)>::Model;
      const std::array<ImmersX::FieldId, 3> ids{
        {builder.differential_field("u", layout[0]),
         builder.differential_field("v", layout[1]),
         builder.algebraic_field("lambda", layout[2])}};
      for (unsigned int row = 0; row < 3; ++row)
        for (unsigned int col = 0; col < 3; ++col)
          {
            typename Model::MatrixOperatorFactory factory =
              [&, row, col](const auto &) {
                return ImmersX::matrix_operator<FieldVector>(
                  jacobian.block(row, col));
              };
            builder
              .term(ids[row],
                    "block-" + std::to_string(row) + "-" + std::to_string(col))
              .state(ids[col], factory);
          }
      builder.second_order_pair(
        ids[0],
        ids[1],
        [] { return 0ul; },
        [] {
          ImmersX::SecondOrderPreconditionerData data;
          data.constrained = IndexSet(n);
          data.constrained.add_index(constrained);
          return data;
        });
      return ids;
    },
    "solid");
  auto state = composition.make_state();
  state      = 0.;
  // Independent prescribed correction, including nonzero constrained u/v.
  const auto exact = [](unsigned int block, unsigned int i) {
    return .17 * (i + 1) - .31 * block;
  };
  for (const bool condensed : {true, false})
    {
      condensed_mass = condensed;
      for (const double stiffness_factor : {1., 1.7})
        for (const double alpha : {3., 0.})
          {
            SCOPED_TRACE(condensed);
            SCOPED_TRACE(stiffness_factor);
            SCOPED_TRACE(alpha);
            assemble(jacobian, layout, alpha);
            jacobian.block(1, 0) *= stiffness_factor;
            const auto blocks  = composition.block_matrix(state);
            const auto reduced = composition.second_order_condensation(blocks);
            ASSERT_TRUE(reduced);
            // 14 free stiffness entries, six B and six transpose-B entries,
            // two multiplier diagonals and one prescribed velocity row.
            // An uncondensed mass adds two remote constrained-v columns.
            EXPECT_EQ(reduced->reduced_matrix().n_nonzero_elements(),
                      condensed ? 29u : 31u);
            GlobalVector rhs(layout, MPI_COMM_WORLD);
            // Evaluate the original equations directly, without multiplying
            // either the assembled Jacobian or its condensed representation.
            for (const auto i : layout[0])
              {
                rhs.block(0)[i] = i == constrained ? 2.5 * exact(0, i) : 0.;
                rhs.block(1)[i] = i == constrained ? exact(1, i) : 0.;
                for (unsigned int j = 0; j < n; ++j)
                  {
                    rhs.block(0)[i] +=
                      mass(i, j) * (alpha * exact(0, j) - exact(1, j));
                    rhs.block(1)[i] +=
                      stiffness_factor * stiffness(i, j) * exact(0, j) +
                      (alpha * mass(i, j) + damping(i, j)) * exact(1, j);
                  }
                if (i != constrained)
                  for (unsigned int j = 0; j < m; ++j)
                    rhs.block(1)[i] += coupling(j, i) * exact(2, j);
              }
            for (const auto i : layout[2])
              {
                rhs.block(2)[i] = -.01 * exact(2, i);
                for (unsigned int j = 0; j < n; ++j)
                  rhs.block(2)[i] += coupling(i, j) * exact(0, j);
              }
            rhs.compress(VectorOperation::insert);
            const auto mass_matrix = reduced->kinematic_mass(0).make_matrix(
              reduced->kinematic_partition(0), MPI_COMM_WORLD);
            FieldVector              kinematic_rhs;
            std::vector<FieldVector> kinematic(1);
            reduced->kinematic_rhs(rhs, 0, kinematic_rhs);
            solve(*mass_matrix, kinematic[0], kinematic_rhs);
            reduced->finish_kinematic_solution(rhs, 0, kinematic[0]);
            FieldVector reduced_rhs, reduced_solution;
            reduced->reduced_rhs(rhs, kinematic, reduced_rhs);
            solve(reduced->reduced_matrix(), reduced_solution, reduced_rhs);
            GlobalVector solution(layout, MPI_COMM_WORLD);
            reduced->expand(reduced_solution, kinematic, solution);
            for (unsigned int block = 0; block < 3; ++block)
              for (const auto i : layout[block])
                EXPECT_NEAR(solution.block(block)[i], exact(block, i), 1.e-9);
            GlobalVector defect(layout, MPI_COMM_WORLD);
            jacobian.vmult(defect, solution);
            defect -= rhs;
            EXPECT_LT(defect.l2_norm(), 1.e-10 * rhs.l2_norm());
          }
    }
  condensed_mass = true;
}

#endif

TEST(SecondOrderCondensation, BOTH_RejectsNonKinematicRows)
{
  const auto  layout = partitions();
  BlockMatrix jacobian;
  assemble(jacobian, layout, 3., 1.e-3);
  EXPECT_FALSE(Condensation::build(jacobian, {{0, 1}}, layout, MPI_COMM_WORLD));
  EXPECT_FALSE(Condensation::build(jacobian, {}, layout, MPI_COMM_WORLD));

  // An uncondensed mass needs v_c from a unit velocity row.
  condensed_mass = false;
  assemble(jacobian, layout, 3., 0., false);
  EXPECT_FALSE(Condensation::build(jacobian, {{0, 1}}, layout, MPI_COMM_WORLD));
  condensed_mass = true;
  assemble(jacobian, layout, 3., 0., false);
  EXPECT_TRUE(Condensation::build(jacobian, {{0, 1}}, layout, MPI_COMM_WORLD));
}


#if defined(DEAL_II_WITH_SUNDIALS) && defined(DEAL_II_WITH_MUMPS) && \
  !defined(IMMERSX_USE_PETSC_LA)
namespace
{
  using BlrAdapter = ImmersX::IDAAdapter<FieldVector, GlobalVector>;

  void
  add_blr_second_order_matrix(BlrAdapter                  &adapter,
                              BlockMatrix                 &jacobian,
                              const std::vector<IndexSet> &layout)
  {
    adapter.add([&](auto &builder) {
      using Model       = typename std::decay_t<decltype(builder)>::Model;
      const auto u      = builder.differential_field("u", layout[0]);
      const auto v      = builder.differential_field("v", layout[1]);
      const auto lambda = builder.algebraic_field("lambda", layout[2]);
      const std::array<ImmersX::FieldId, 3> fields{{u, v, lambda}};
      for (unsigned int row = 0; row < 3; ++row)
        for (unsigned int column = 0; column < 3; ++column)
          {
            typename Model::MatrixOperatorFactory factory =
              [&, row, column](const auto &) {
                return ImmersX::matrix_operator<FieldVector>(
                  jacobian.block(row, column));
              };
            builder
              .term(fields[row],
                    "entry-" + std::to_string(row) + "-" +
                      std::to_string(column))
              .state(fields[column], factory);
          }
      builder.second_order_pair(
        u,
        v,
        [] { return 0ul; },
        [&layout] {
          ImmersX::SecondOrderPreconditionerData data;
          data.constrained = IndexSet(layout[0].size());
          if (layout[0].size() > constrained)
            data.constrained.add_index(constrained);
          return data;
        });
      return fields;
    });
  }
} // namespace

TEST(SecondOrderCondensation, BOTH_DirectSolveHandlesDisjointRowsAndStoredZeros)
{
  using Adapter   = ImmersX::IDAAdapter<FieldVector, GlobalVector>;
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  IndexSet   owned(2);
  if (rank < 2)
    owned.add_index(rank);
  owned.compress();
  for (const bool diagonal : {false, true})
    {
      ParameterAcceptor::clear();
      const std::array<std::array<double, 4>, 4> coefficients =
        diagonal ? std::array<std::array<double, 4>, 4>{{{{2., 0., 0., 0.}},
                                                         {{0., -3., 0., 0.}},
                                                         {{0., 0., 4., 0.}},
                                                         {{0., 0., 0., -5.}}}} :
                   std::array<std::array<double, 4>, 4>{{{{2., -1., 1., 0.}},
                                                         {{-1., 3., 0., -1.}},
                                                         {{1., 0., 4., -2.}},
                                                         {{0., 1., -2., 5.}}}};
      ImmersX::TimeIntervalParameters time("/", false);
      ImmersX::IDAParameters          parameters("/", &time);
      Adapter                         adapter(time, parameters, MPI_COMM_WORLD);
      adapter.set_linear_solver(Adapter::LinearSolver::mumps);
      const auto fields = adapter.add([&](auto &builder) {
        using Model = typename std::decay_t<decltype(builder)>::Model;
        const std::array<ImmersX::FieldId, 2> ids{
          {builder.algebraic_field("first", owned),
           builder.algebraic_field("second", owned)}};
        for (unsigned int row = 0; row < 2; ++row)
          for (unsigned int col = 0; col < 2; ++col)
            {
              typename Model::MatrixOperatorFactory factory =
                [owned, coefficients, row, col](const auto &) {
                  DynamicSparsityPattern pattern(2, 2, owned);
                  for (const auto i : owned)
                    for (unsigned int j = 0; j < 2; ++j)
                      pattern.add(i, j);
                  auto matrix = std::make_shared<Matrix>();
                  matrix->reinit(owned, owned, pattern, MPI_COMM_WORLD);
                  for (const auto i : owned)
                    for (unsigned int j = 0; j < 2; ++j)
                      matrix->set(i, j, coefficients[2 * row + i][2 * col + j]);
                  matrix->compress(VectorOperation::insert);
                  return ImmersX::matrix_operator<FieldVector>(matrix);
                };
              builder
                .term(ids[row],
                      "block-" + std::to_string(row) + "-" +
                        std::to_string(col))
                .state(ids[col], factory);
            }
        return ids;
      });
      auto       state = adapter.make_state(), dot = adapter.make_state();
      state = 0.;
      dot   = 0.;
      adapter.solver().setup_jacobian(0., state, dot, 0.);
      // Dense independent oracle; negative diagonals and explicit structural
      // off-diagonal zeros must remain eligible for entrywise inversion.
      const std::array<double, 4> exact{{1., -2., 3., -4.}};
      const std::array<double, 4> forcing =
        diagonal ? std::array<double, 4>{{2., 6., 12., 20.}} :
                   std::array<double, 4>{{7., -3., 21., -28.}};
      auto rhs = adapter.make_state(), solution = adapter.make_state();
      for (unsigned int block = 0; block < 2; ++block)
        for (const auto i : owned)
          adapter.field(rhs, fields.fields()[block])[i] =
            forcing[2 * block + i];
      rhs.compress(VectorOperation::insert);
      adapter.solver().solve_with_jacobian(rhs, solution, 1.e-12);
      for (unsigned int block = 0; block < 2; ++block)
        for (const auto i : owned)
          EXPECT_NEAR(adapter.field(solution, fields.fields()[block])[i],
                      exact[2 * block + i],
                      1.e-12);
      auto defect = adapter.make_state();
      adapter.current_jacobian().vmult(defect, solution);
      defect -= rhs;
      EXPECT_LT(defect.l2_norm(), 1.e-12 * rhs.l2_norm());
      ParameterAcceptor::clear();
    }
}

TEST(SecondOrderCondensation, BOTH_BlrAdapterSolvesFullConstrainedSystem)
{
  using Adapter     = ImmersX::IDAAdapter<FieldVector, GlobalVector>;
  const auto layout = partitions();
  // Both mass boundary treatments exercise nonzero prescribed u/v entries.
  for (const bool condensed : {true, false})
    {
      condensed_mass = condensed;
      ImmersX::TimeIntervalParameters time;
      ImmersX::IDAParameters          parameters;
      parameters.linear_solver_controls.krylov_maximum_iterations = 100;
      Adapter adapter(time, parameters, MPI_COMM_WORLD);
      adapter.set_linear_solver(Adapter::LinearSolver::mumps_blr_fgmres);
      adapter.set_mumps_blr_parameters(1.e-5);
      BlockMatrix jacobian;
      assemble(jacobian, layout, 3.);
      add_blr_second_order_matrix(adapter, jacobian, layout);
      auto state = adapter.make_state(), dot = adapter.make_state();
      state    = 0.;
      dot      = 0.;
      auto rhs = adapter.make_state(), solution = adapter.make_state();
      auto residual = adapter.make_state();
      for (unsigned int block = 0; block < 3; ++block)
        for (const auto i : rhs.block(block).locally_owned_elements())
          rhs.block(block)[i] = 1. + .3 * i - .7 * block;
      rhs.compress(VectorOperation::insert);
      // Alpha changes alone reuse Mhat. A changed kinematic mass must rebuild.
      for (unsigned int step = 0; step < 3; ++step)
        {
          const double alpha = step == 0 ? 3. : step == 1 ? 0. : 2.;
          assemble(jacobian, layout, alpha);
          if (step == 2)
            {
              jacobian.block(0, 0) *= 1.25;
              jacobian.block(0, 1) *= 1.25;
            }
          adapter.solver().setup_jacobian(1. + step, state, dot, alpha);
          solution = 0.;
          adapter.solver().solve_with_jacobian(rhs, solution, 1.e-11);
          // Independent uncondensed matrix oracle includes the nonzero
          // kinematic RHS and both constrained rows; never just S*x=b.
          jacobian.vmult(residual, solution);
          residual -= rhs;
          EXPECT_LE(residual.l2_norm(), 1.e-10 * rhs.l2_norm());
          const auto statistics = adapter.statistics();
          EXPECT_EQ(statistics.kinematic_factorizations, step < 2 ? 1u : 2u);
          EXPECT_EQ(statistics.kinematic_factorization_reuses,
                    step == 0 ? 0u : 1u);
        }
    }
  condensed_mass = true;
}

TEST(SecondOrderCondensation, BOTH_BlrSmallMassPreservesFullFallback)
{
  const unsigned int size = 3;
  const auto         owned =
    Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,
                                                           size);
  const auto multipliers =
    Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, m);
  const std::vector<IndexSet> layout{owned, owned, multipliers};
  BlockMatrix                 jacobian;
  assemble(jacobian, layout, 2.);
  ImmersX::TimeIntervalParameters time;
  ImmersX::IDAParameters          parameters;
  BlrAdapter                      adapter(time, parameters, MPI_COMM_WORLD);
  adapter.set_linear_solver(BlrAdapter::LinearSolver::mumps_blr_fgmres);
  add_blr_second_order_matrix(adapter, jacobian, layout);
  auto state = adapter.make_state(), dot = adapter.make_state();
  auto rhs = adapter.make_state(), solution = adapter.make_state();
  state = 0.;
  dot   = 0.;
  for (unsigned int block = 0; block < 3; ++block)
    for (const auto i : rhs.block(block).locally_owned_elements())
      rhs.block(block)[i] = 1. + .2 * i + .4 * block;
  rhs.compress(VectorOperation::insert);
  adapter.solver().setup_jacobian(1., state, dot, 2.);
  adapter.solver().solve_with_jacobian(rhs, solution, 1.e-11);
  auto error = adapter.make_state();
  jacobian.vmult(error, solution);
  error -= rhs;
  EXPECT_LE(error.l2_norm(), 1.e-10 * rhs.l2_norm());
  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > size)
    {
      EXPECT_EQ(adapter.statistics().kinematic_factorizations, 0u);
    }
}
#endif
