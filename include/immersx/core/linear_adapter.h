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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/patterns.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#ifdef DEAL_II_WITH_MUMPS
#  include <deal.II/lac/sparse_direct.h>
#endif

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/detail/execution_composition.h>
#include <immersx/core/problem_handle.h>

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace ImmersX
{
  enum class LinearSolver
  {
    automatic,
    iterative,
    direct,
    mumps
  };

  enum class LinearPreconditioner
  {
    automatic,
    none,
    block_diagonal,
    block_triangular,
    schur,
    augmented_lagrangian
  };

  /** Parameter object for the standard LinearAdapter path. */
  struct LinearAdapterParameters : public dealii::ParameterAcceptor
  {
    LinearAdapterParameters(const std::string &section_name = "Linear adapter")
      : dealii::ParameterAcceptor(section_name)
    {}

    void
    declare_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter("Solver", solver);
      prm.add_parameter("Preconditioner", preconditioner);
      prm.add_parameter("Maximum iterations", maximum_iterations);
      prm.add_parameter("Tolerance", tolerance);
      prm.add_parameter("Augmented Lagrangian parameter",
                        augmented_lagrangian_parameter);
    }

    LinearSolver         solver             = LinearSolver::automatic;
    LinearPreconditioner preconditioner     = LinearPreconditioner::automatic;
    unsigned int         maximum_iterations = 1000;
    double               tolerance          = 1.e-12;
    double               augmented_lagrangian_parameter = 1.e1;
  };

  using LinearSolverParameters = LinearAdapterParameters;

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
    using Parameters          = LinearAdapterParameters;
    using Operator            = dealii::LinearOperator<GlobalVectorType>;
    using LocalOperator       = dealii::LinearOperator<FieldVectorType>;
    using MatrixType          = typename Composition::MatrixType;
    using BlockMatrixType     = typename Composition::BlockMatrixType;
    using SaddlePointMetadata = typename Composition::SaddlePointMetadata;
    using SolveFunction       = std::function<
      void(const Operator &, const GlobalVectorType &, GlobalVectorType &)>;

    LinearAdapter(const LinearAdapterParameters &parameters,
                  const MPI_Comm                 communicator,
                  SolveFunction                  solve = {})
      : composition_(communicator)
      , solve_(std::move(solve))
      , parameters_(parameters)
      , pcout(std::cout,
              dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
    {}

    const LinearAdapterParameters &
    solver_options() const
    {
      return parameters_;
    }

    bool
    has_multiplier_metric(const FieldId multiplier) const
    {
      return composition_.model().has_multiplier_metric(multiplier);
    }

    /** Factorize the current linearization for subsequent direct solves. */
    void
    setup_direct(const GlobalVectorType &state) const
    {
      pcout << "LinearAdapter: materializing and factorizing the linear "
               "system."
            << std::endl;
      direct_matrix_ = std::make_shared<MatrixType>();
      composition_.monolithic_matrix(state, *direct_matrix_);
      direct_solver_.reset();
      direct_control_ =
        std::make_unique<dealii::SolverControl>(0, parameters_.tolerance);
#ifdef DEAL_II_WITH_MUMPS
      if (parameters_.solver == LinearSolver::mumps)
        {
          mumps_solver_ = std::make_unique<dealii::SparseDirectMUMPS>(
            dealii::SparseDirectMUMPS::AdditionalData(),
            composition_.communicator());
          mumps_solver_->initialize(*direct_matrix_);
          pcout << "LinearAdapter: MUMPS factorization ready." << std::endl;
          return;
        }
#else
      AssertThrow(parameters_.solver != LinearSolver::mumps,
                  dealii::ExcMessage(
                    "LinearAdapter solver 'mumps' requires deal.II to be "
                    "configured with MUMPS."));
#endif
      direct_solver_ =
        std::make_unique<ImmersXLA::SolverDirect>(*direct_control_);
      direct_solver_->initialize(*direct_matrix_);
      pcout << "LinearAdapter: direct factorization ready." << std::endl;
    }

    template <typename Problem, typename... Arguments>
    auto
    add(const Problem     &problem,
        const std::string &prefix = {},
        const Arguments &...arguments)
    {
      auto fields = composition_.add(problem, prefix, arguments...);
      pcout << "LinearAdapter: registered contributor";
      if (!prefix.empty())
        pcout << " '" << prefix << "'";
      pcout << "; " << composition_.n_fields()
            << " semantic field(s) available." << std::endl;
      return ProblemHandle<LinearAdapter, decltype(fields)>(*this,
                                                            std::move(fields));
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

    bool
    has_complete_local_preconditioners() const
    {
      return composition_.has_complete_local_preconditioners();
    }

    const std::vector<SaddlePointMetadata> &
    saddle_points() const
    {
      return composition_.saddle_points();
    }

    LocalOperator
    schur_operator(const FieldId           multiplier,
                   const GlobalVectorType &state) const
    {
      return composition_.schur_operator(multiplier, state);
    }

    Operator
    schur_preconditioner(const FieldId           multiplier,
                         const GlobalVectorType &state) const
    {
      return composition_.schur_preconditioner(multiplier, state);
    }

    Operator
    augmented_lagrangian_operator(const GlobalVectorType &state,
                                  const double            gamma = 1.e1) const
    {
      return composition_.augmented_lagrangian_operator(state, gamma);
    }

    BlockMatrixType
    augmented_lagrangian_matrix(const GlobalVectorType &state,
                                const double            gamma = 1.e1) const
    {
      return composition_.augmented_lagrangian_matrix(state, gamma);
    }

    Operator
    augmented_lagrangian_preconditioner(const GlobalVectorType &state,
                                        const double gamma = 1.e1) const
    {
      return composition_.augmented_lagrangian_preconditioner(state, gamma);
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

    Operator
    block_triangular_preconditioner(const GlobalVectorType &state,
                                    const bool              lower = true) const
    {
      return composition_.block_triangular_preconditioner(state, lower);
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
      setup_direct(state);
      solve_with_current_direct(state);
    }

    /** Reuse an explicitly prepared direct factorization. */
    void
    solve_with_current_direct(GlobalVectorType &state) const
    {
      bool has_factorization = static_cast<bool>(direct_solver_);
#ifdef DEAL_II_WITH_MUMPS
      has_factorization = has_factorization || static_cast<bool>(mumps_solver_);
#endif
      AssertThrow(
        has_factorization,
        dealii::ExcMessage(
          "No direct factorization is available; call setup_direct first."));

      const auto &model = composition_.model();
      AssertThrow(!model.has_derivative_terms(),
                  dealii::ExcMessage(
                    "LinearAdapter cannot solve a model with derivative "
                    "terms."));
      auto residual = composition_.make_state();
      state         = composition_.make_state();
      composition_.evaluate_residual(0., state, nullptr, residual);
      residual *= -1.;

      auto rhs      = composition_.pack(residual);
      auto result   = composition_.make_state();
      auto solution = composition_.pack(result);
#ifdef DEAL_II_WITH_MUMPS
      if (mumps_solver_)
        mumps_solver_->vmult(solution, rhs);
      else
#endif
        direct_solver_->solve(solution, rhs);
      composition_.unpack(solution, state);
      pcout << "LinearAdapter: direct solve finished." << std::endl;
    }

    void
    solve(GlobalVectorType &state) const
    {
      pcout << "LinearAdapter: starting solve with " << composition_.n_fields()
            << " semantic field(s), solver="
            << magic_enum::enum_name(parameters_.solver) << ", preconditioner="
            << magic_enum::enum_name(parameters_.preconditioner) << "."
            << std::endl;
      if (parameters_.solver == LinearSolver::direct ||
          parameters_.solver == LinearSolver::mumps)
        {
          solve_direct(state);
          return;
        }
      AssertThrow(parameters_.solver == LinearSolver::automatic ||
                    parameters_.solver == LinearSolver::iterative,
                  dealii::ExcMessage(
                    "LinearAdapter solver must be auto, iterative, direct, "
                    "or mumps."));

      const auto &model = composition_.model();
      AssertThrow(!model.has_derivative_terms(),
                  dealii::ExcMessage(
                    "LinearAdapter cannot solve a model with derivative "
                    "terms."));
      auto residual = composition_.make_state();
      state         = composition_.make_state();
      composition_.evaluate_residual(0., state, nullptr, residual);
      residual *= -1.;
      const auto operator_view = composition_.jacobian(0., state, nullptr, 0.);
      if (solve_)
        {
          pcout << "LinearAdapter: invoking the configured linear solver."
                << std::endl;
          solve_(operator_view, residual, state);
          pcout << "LinearAdapter: configured linear solver finished."
                << std::endl;
        }
      else
        {
          const auto preconditioner = make_preconditioner(operator_view, state);
          dealii::SolverControl control(parameters_.maximum_iterations,
                                        parameters_.tolerance);
          const typename dealii::SolverGMRES<GlobalVectorType>::AdditionalData
                     data(30, false, false);
          const bool flexible =
            parameters_.preconditioner ==
              LinearPreconditioner::augmented_lagrangian ||
            parameters_.preconditioner ==
              LinearPreconditioner::block_triangular ||
            parameters_.preconditioner == LinearPreconditioner::schur ||
            (parameters_.preconditioner == LinearPreconditioner::automatic &&
             (composition_.n_fields() > 1 ||
              !composition_.saddle_points().empty()));
          if (flexible)
            {
              const typename dealii::SolverFGMRES<
                GlobalVectorType>::AdditionalData    right_data(100);
              dealii::SolverFGMRES<GlobalVectorType> solver(control,
                                                            right_data);
              solver.solve(operator_view, state, residual, preconditioner);
              pcout << "LinearAdapter: FGMRES converged in "
                    << control.last_step() << " iteration(s), residual "
                    << control.last_value() << "." << std::endl;
            }
          else
            {
              dealii::SolverGMRES<GlobalVectorType> solver(control, data);
              solver.solve(operator_view, state, residual, preconditioner);
              pcout << "LinearAdapter: GMRES converged in "
                    << control.last_step() << " iteration(s), residual "
                    << control.last_value() << "." << std::endl;
            }
        }
      pcout << "LinearAdapter: solve finished." << std::endl;
    }

  private:
    Operator
    make_preconditioner(const Operator         &operator_view,
                        const GlobalVectorType &state) const
    {
      auto choice = parameters_.preconditioner;
      if (choice == LinearPreconditioner::automatic)
        choice = !composition_.saddle_points().empty() ?
                   LinearPreconditioner::schur :
                   (composition_.n_fields() == 1 ?
                      LinearPreconditioner::block_diagonal :
                      LinearPreconditioner::block_triangular);

      pcout << "LinearAdapter: selected " << magic_enum::enum_name(choice)
            << " preconditioner." << std::endl;

      if (choice == LinearPreconditioner::none)
        return dealii::identity_operator(operator_view);
      if (choice == LinearPreconditioner::block_diagonal)
        {
          AssertThrow(composition_.has_complete_local_preconditioners(),
                      dealii::ExcMessage(
                        "Block diagonal preconditioning needs local "
                        "preconditioners for every field."));
          return composition_.block_diagonal_preconditioner(state);
        }
      if (choice == LinearPreconditioner::block_triangular)
        {
          AssertThrow(composition_.has_complete_local_preconditioners(),
                      dealii::ExcMessage(
                        "Block triangular preconditioning needs local "
                        "preconditioners for every field."));
          return composition_.block_triangular_preconditioner(state);
        }
      if (choice == LinearPreconditioner::schur)
        {
          AssertThrow(composition_.saddle_points().size() == 1u,
                      dealii::ExcMessage(
                        "Schur preconditioning currently requires exactly "
                        "one saddle-point relation."));
          return composition_.schur_preconditioner(
            composition_.saddle_points().front().multiplier, state);
        }
      if (choice == LinearPreconditioner::augmented_lagrangian)
        return composition_.augmented_lagrangian_preconditioner(
          state, parameters_.augmented_lagrangian_parameter);
      AssertThrow(false,
                  dealii::ExcMessage(
                    "The requested preconditioner policy is not available "
                    "for LinearAdapter."));
      return dealii::identity_operator(operator_view);
    }

    Composition                                      composition_;
    SolveFunction                                    solve_;
    const LinearAdapterParameters                   &parameters_;
    mutable dealii::ConditionalOStream               pcout;
    mutable std::shared_ptr<MatrixType>              direct_matrix_;
    mutable std::unique_ptr<dealii::SolverControl>   direct_control_;
    mutable std::unique_ptr<ImmersXLA::SolverDirect> direct_solver_;
#ifdef DEAL_II_WITH_MUMPS
    mutable std::unique_ptr<dealii::SparseDirectMUMPS> mumps_solver_;
#endif
  };
} // namespace ImmersX

#endif // immersx_linear_adapter_h
