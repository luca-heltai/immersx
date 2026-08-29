// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_linear_adapter_h
#define immersx_linear_adapter_h

#include <deal.II/lac/solver_control.h>

#ifdef DEAL_II_WITH_TRILINOS
#  include <Amesos2.hpp>
#  include <Epetra_CrsMatrix.h>
#  include <Epetra_Export.h>
#  include <Epetra_Map.h>
#  include <Epetra_MultiVector.h>
#endif

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/detail/execution_composition.h>
#include <immersx/core/problem_handle.h>
#include <immersx/core/representation.h>

#include <functional>
#include <limits>
#include <string>
#include <utility>

namespace ImmersX
{
  /**
   * Execution adapter for affine steady semantic systems.
   *
   * `solve()` evaluates the affine residual at zero and overwrites its state
   * with the result of the supplied direct linear solve. The incoming state is
   * therefore storage, not a nonlinear initial guess.
   */
  template <typename FieldVectorType, typename GlobalVectorType>
  class LinearAdapter
  {
    using Composition =
      detail::ExecutionComposition<FieldVectorType, GlobalVectorType>;

  public:
    using RepresentationType = Representation<FieldVectorType>;
    using ComponentRepresentationType =
      ComponentRepresentation<FieldVectorType>;
    using Operator        = dealii::LinearOperator<GlobalVectorType>;
    using LocalOperator   = dealii::LinearOperator<FieldVectorType>;
    using MatrixType      = typename Composition::MatrixType;
    using BlockMatrixType = typename Composition::BlockMatrixType;
    using SolveFunction   = std::function<
      void(const Operator &, const GlobalVectorType &, GlobalVectorType &)>;

    LinearAdapter(const MPI_Comm communicator, SolveFunction solve)
      : composition_(communicator)
      , solve_(std::move(solve))
    {
      AssertThrow(solve_,
                  dealii::ExcMessage(
                    "LinearAdapter requires a linear solve callback."));
    }

    template <typename Problem, typename... Arguments>
    auto
    add(const Problem     &problem,
        const std::string &prefix = {},
        const Arguments &...arguments)
    {
      auto fields = composition_.add(problem, prefix, arguments...);
      return ProblemHandle<LinearAdapter, decltype(fields)>(*this,
                                                            std::move(fields));
    }

    template <typename Quantity, typename Target, typename Coupling>
    auto
    couple(const Quantity &quantity,
           const Target   &target,
           const Coupling &coupling)
    {
      auto interaction = detail::invoke_coupling(quantity, target, coupling, 0);
      return add(std::move(interaction),
                 "coupling" + std::to_string(coupling_count_++));
    }

    GlobalVectorType
    make_state() const
    {
      return composition_.make_state();
    }

    void
    reinit(GlobalVectorType &state) const
    {
      composition_.reinit(state);
    }

    FieldVectorType &
    field(GlobalVectorType &state, const FieldId id) const
    {
      return composition_.field(state, id);
    }

    const FieldVectorType &
    field(const GlobalVectorType &state, const FieldId id) const
    {
      return composition_.field(state, id);
    }

    RepresentationType
    observe(const FieldId id) const
    {
      AssertThrow(composition_.state_layout().contains(id),
                  dealii::ExcMessage("Cannot observe an unknown Field."));
      return RepresentationType(id);
    }

    FieldComponentView
    component(const FieldId id, const dealii::IndexSet &components) const
    {
      AssertThrow(composition_.state_layout().contains(id),
                  dealii::ExcMessage("Cannot select an unknown Field."));
      AssertThrow(components.size() ==
                    composition_.state_layout().field(id).locally_owned.size(),
                  dealii::ExcMessage(
                    "Component view must have the Field's global size."));
      return FieldComponentView(id, components);
    }

    ComponentRepresentationType
    observe(const FieldComponentView &view) const
    {
      AssertThrow(composition_.state_layout().contains(view.source()),
                  dealii::ExcMessage("Cannot observe an unknown Field."));
      return ComponentRepresentationType(view);
    }

    void
    evaluate_residual(const GlobalVectorType &state,
                      GlobalVectorType       &residual) const
    {
      composition_.evaluate_residual(0., state, nullptr, residual);
    }

    Operator
    jacobian(const GlobalVectorType &state) const
    {
      return composition_.jacobian(0., state, nullptr, 0.);
    }

    bool
    can_materialize_matrix(const GlobalVectorType &state) const
    {
      return composition_.can_materialize_matrix(state);
    }

    BlockMatrixType
    block_matrix(const GlobalVectorType &state) const
    {
      return composition_.block_matrix(state);
    }

    MatrixType
    monolithic_matrix(const GlobalVectorType &state) const
    {
      return composition_.monolithic_matrix(state);
    }

    bool
    has_local_preconditioner(const FieldId field) const
    {
      return composition_.has_local_preconditioner(field);
    }

    std::optional<LocalOperator>
    local_preconditioner(const FieldId           field,
                         const GlobalVectorType &state) const
    {
      return composition_.local_preconditioner(field, state);
    }

    Operator
    block_diagonal_preconditioner(const GlobalVectorType &state) const
    {
      return composition_.block_diagonal_preconditioner(state);
    }

    FieldVectorType
    pack(const GlobalVectorType &state) const
    {
      return composition_.pack(state);
    }

    void
    unpack(const FieldVectorType &flat, GlobalVectorType &state) const
    {
      composition_.unpack(flat, state);
    }

    /** Solve using a freshly materialized matrix and the configured direct
     * backend. */
    void
    solve_direct(GlobalVectorType &state) const
    {
      const auto &model = composition_.model();
      AssertThrow(!model.has_derivative_terms(),
                  dealii::ExcMessage(
                    "LinearAdapter cannot solve a model with derivative "
                    "terms."));
      auto residual = composition_.make_state();
      state         = composition_.make_state();
      composition_.evaluate_residual(0., state, nullptr, residual);
      residual *= -1.;

      auto matrix   = composition_.monolithic_matrix(state);
      auto rhs      = composition_.pack(residual);
      auto result   = composition_.make_state();
      auto solution = composition_.pack(result);
      if (dealii::Utilities::MPI::n_mpi_processes(
            matrix.get_mpi_communicator()) > 1)
        {
          AssertThrow(matrix.m() <= static_cast<typename MatrixType::size_type>(
                                      std::numeric_limits<int>::max()),
                      dealii::ExcMessage(
                        "Amesos2 MUMPS requires a 32-bit global matrix "
                        "index range."));

          const auto      &epetra_matrix = matrix.trilinos_matrix();
          const auto      &row_map       = epetra_matrix.RowMap();
          const auto      &column_map    = epetra_matrix.ColMap();
          std::vector<int> owned_indices;
          owned_indices.reserve(row_map.NumMyElements());
          for (int local_row = 0; local_row < row_map.NumMyElements();
               ++local_row)
            owned_indices.push_back(static_cast<int>(row_map.GID64(local_row)));

          const Epetra_Map int_map(static_cast<int>(matrix.m()),
                                   static_cast<int>(owned_indices.size()),
                                   owned_indices.data(),
                                   0,
                                   matrix.trilinos_matrix().Comm());
          Epetra_CrsMatrix int_matrix(Copy, int_map, 0);
          for (int local_row = 0; local_row < row_map.NumMyElements();
               ++local_row)
            {
              int     n_entries = 0;
              double *values    = nullptr;
              int    *columns   = nullptr;
              AssertThrow(epetra_matrix.ExtractMyRowView(
                            local_row, n_entries, values, columns) == 0,
                          dealii::ExcMessage(
                            "Unable to inspect the Epetra matrix row."));
              std::vector<int>    global_columns(n_entries);
              std::vector<double> global_values(values, values + n_entries);
              for (int entry = 0; entry < n_entries; ++entry)
                global_columns[entry] =
                  static_cast<int>(column_map.GID64(columns[entry]));
              AssertThrow(
                int_matrix.InsertGlobalValues(owned_indices[local_row],
                                              n_entries,
                                              global_values.data(),
                                              global_columns.data()) == 0,
                dealii::ExcMessage(
                  "Unable to convert the matrix to 32-bit Epetra "
                  "global indices."));
            }
          AssertThrow(int_matrix.FillComplete(int_map, int_map) == 0,
                      dealii::ExcMessage(
                        "Unable to complete the 32-bit Epetra matrix for "
                        "Amesos2 MUMPS."));
          const Epetra_Map solver_map(static_cast<int>(matrix.m()),
                                      0,
                                      matrix.trilinos_matrix().Comm());
          Epetra_Export    exporter(int_map, solver_map);
          Epetra_CrsMatrix solver_matrix(Copy, solver_map, 0);
          AssertThrow(solver_matrix.Export(int_matrix, exporter, Insert) == 0,
                      dealii::ExcMessage(
                        "Unable to redistribute the Epetra matrix for "
                        "Amesos2 MUMPS."));
          AssertThrow(solver_matrix.FillComplete(solver_map, solver_map) == 0,
                      dealii::ExcMessage(
                        "Unable to complete the contiguous Epetra matrix for "
                        "Amesos2 MUMPS."));

          std::vector<double> global_rhs(matrix.m(), 0.);
          const auto         &rhs_epetra = rhs.trilinos_vector();
          const auto         &rhs_map    = rhs_epetra.Map();
          for (int local = 0; local < rhs_map.NumMyElements(); ++local)
            global_rhs[static_cast<std::size_t>(rhs_map.GID64(local))] =
              rhs_epetra[0][local];
          AssertThrow(
            MPI_Allreduce(MPI_IN_PLACE,
                          global_rhs.data(),
                          static_cast<int>(global_rhs.size()),
                          MPI_DOUBLE,
                          MPI_SUM,
                          matrix.get_mpi_communicator()) == MPI_SUCCESS,
            dealii::ExcMessage(
              "Unable to redistribute the direct-solver right-hand side."));

          Epetra_MultiVector int_rhs(solver_map, 1);
          Epetra_MultiVector int_solution(solver_map, 1);
          for (int local_row = 0; local_row < solver_map.NumMyElements();
               ++local_row)
            {
              const int row         = solver_map.GID(local_row);
              int_rhs[0][local_row] = global_rhs[row];
            }
          auto int_matrix_rcp =
            Teuchos::rcp(new Epetra_CrsMatrix(solver_matrix));
          auto int_rhs_rcp = Teuchos::rcp(new Epetra_MultiVector(int_rhs));
          auto int_solution_rcp =
            Teuchos::rcp(new Epetra_MultiVector(int_solution));
          auto mumps_solver =
            Amesos2::create<Epetra_CrsMatrix, Epetra_MultiVector>(
              "MUMPS", int_matrix_rcp, int_solution_rcp, int_rhs_rcp);
          mumps_solver->symbolicFactorization();
          mumps_solver->numericFactorization();
          mumps_solver->solve();
          std::vector<double> global_solution(matrix.m(), 0.);
          for (int local_row = 0; local_row < solver_map.NumMyElements();
               ++local_row)
            global_solution[solver_map.GID(local_row)] =
              (*int_solution_rcp)[0][local_row];
          AssertThrow(MPI_Allreduce(MPI_IN_PLACE,
                                    global_solution.data(),
                                    static_cast<int>(global_solution.size()),
                                    MPI_DOUBLE,
                                    MPI_SUM,
                                    matrix.get_mpi_communicator()) ==
                        MPI_SUCCESS,
                      dealii::ExcMessage(
                        "Unable to collect the direct-solver solution."));
          const auto &solution_epetra = solution.trilinos_vector();
          const auto &solution_map    = solution_epetra.Map();
          for (int local = 0; local < solution_map.NumMyElements(); ++local)
            solution[solution_map.GID64(local)] =
              global_solution[solution_map.GID64(local)];
          solution.compress(dealii::VectorOperation::insert);
        }
      else
        {
          dealii::SolverControl   control(0, 1.e-12);
          ImmersXLA::SolverDirect solver(control);
          solver.initialize(matrix);
          solver.solve(solution, rhs);
        }
      composition_.unpack(solution, state);
    }

    void
    solve(GlobalVectorType &state) const
    {
      const auto &model = composition_.model();
      AssertThrow(!model.has_derivative_terms(),
                  dealii::ExcMessage(
                    "LinearAdapter cannot solve a model with derivative "
                    "terms."));
      auto residual = composition_.make_state();
      state         = composition_.make_state();
      composition_.evaluate_residual(0., state, nullptr, residual);
      residual *= -1.;
      solve_(composition_.jacobian(0., state, nullptr, 0.), residual, state);
    }

  private:
    Composition   composition_;
    SolveFunction solve_;
    std::size_t   coupling_count_ = 0;
  };
} // namespace ImmersX

#endif // immersx_linear_adapter_h
