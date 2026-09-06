// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_sundials_kinsol_adapter_h
#define immersx_sundials_kinsol_adapter_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_SUNDIALS
#  include <deal.II/base/conditional_ostream.h>
#  include <deal.II/base/mpi.h>
#  include <deal.II/base/parameter_acceptor.h>
#  include <deal.II/base/patterns.h>

#  include <deal.II/lac/precondition.h>
#  include <deal.II/lac/solver_gmres.h>

#  include <deal.II/sundials/kinsol.h>

#  include <immersx/core/detail/execution_composition.h>
#  include <immersx/core/problem_handle.h>

#  include <algorithm>
#  include <functional>
#  include <iostream>
#  include <memory>
#  include <optional>
#  include <string>
#  include <utility>
#endif

#ifdef DEAL_II_WITH_SUNDIALS
namespace ImmersX
{
  /** Parameters for the steady nonlinear KINSOL execution path. */
  struct KINSOLAdapterParameters : public dealii::ParameterAcceptor
  {
    explicit KINSOLAdapterParameters(
      const std::string &section_name = "KINSOL adapter")
      : dealii::ParameterAcceptor(section_name)
    {}

    void
    declare_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter("Solution strategy",
                        strategy,
                        "Use Newton or line-search Newton.",
                        dealii::Patterns::Selection("newton|linesearch"));
      prm.add_parameter("Maximum number of nonlinear iterations",
                        maximum_non_linear_iterations);
      prm.add_parameter("Function norm stopping tolerance",
                        function_tolerance,
                        "",
                        dealii::Patterns::Double(0.));
      prm.add_parameter("Scaled step stopping tolerance",
                        step_tolerance,
                        "",
                        dealii::Patterns::Double(0.));
      prm.add_parameter("No initial Jacobian setup", no_initial_setup);
      prm.add_parameter("Maximum iterations without matrix setup",
                        maximum_setup_calls);
      prm.add_parameter("Maximum allowable scaled length of Newton step",
                        maximum_newton_step,
                        "",
                        dealii::Patterns::Double(0.));
    }

    template <typename GlobalVectorType>
    typename dealii::SUNDIALS::KINSOL<GlobalVectorType>::AdditionalData
    kinsol_parameters() const
    {
      using AdditionalData =
        typename dealii::SUNDIALS::KINSOL<GlobalVectorType>::AdditionalData;

      const auto solution_strategy = [&]() {
        if (strategy == "newton")
          return AdditionalData::newton;
        AssertThrow(strategy == "linesearch",
                    dealii::ExcMessage(
                      "KINSOL solution strategy must be newton or "
                      "linesearch."));
        return AdditionalData::linesearch;
      }();

      return AdditionalData(solution_strategy,
                            maximum_non_linear_iterations,
                            function_tolerance,
                            step_tolerance,
                            no_initial_setup,
                            maximum_setup_calls,
                            maximum_newton_step);
    }

    std::string  strategy                      = "linesearch";
    unsigned int maximum_non_linear_iterations = 200;
    double       function_tolerance            = 0.;
    double       step_tolerance                = 0.;
    bool         no_initial_setup              = false;
    unsigned int maximum_setup_calls           = 0;
    double       maximum_newton_step           = 0.;
  };

  /** Public steady nonlinear adapter backed by the private composition engine.
   */
  template <typename FieldVectorType, typename GlobalVectorType>
  class KINSOLAdapter
  {
  public:
    using Composition =
      detail::ExecutionComposition<FieldVectorType, GlobalVectorType>;
    using Operator            = dealii::LinearOperator<GlobalVectorType>;
    using LocalOperator       = dealii::LinearOperator<FieldVectorType>;
    using SaddlePointMetadata = typename Composition::SaddlePointMetadata;
    using LinearSolveFunction = std::function<void(const Operator &,
                                                   const GlobalVectorType &,
                                                   GlobalVectorType &,
                                                   double)>;
    using AdditionalData =
      typename dealii::SUNDIALS::KINSOL<GlobalVectorType>::AdditionalData;

    KINSOLAdapter(const KINSOLAdapterParameters &parameters,
                  const MPI_Comm                 communicator,
                  LinearSolveFunction            solve = {})
      : composition_(communicator)
      , solve_(std::move(solve))
      , parameters_(parameters)
      , pcout(std::cout,
              dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
    {}

    const KINSOLAdapterParameters &
    solver_options() const
    {
      return parameters_;
    }

    const AdditionalData &
    additional_data() const
    {
      if (!additional_data_.has_value())
        additional_data_ =
          parameters_.template kinsol_parameters<GlobalVectorType>();
      return *additional_data_;
    }

    template <typename Problem, typename... Arguments>
    auto
    add(const Problem     &problem,
        const std::string &prefix = {},
        const Arguments &...arguments)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "KINSOLAdapter contributors must be added before "
                    "reinit or solve."));
      auto fields = composition_.add(problem, prefix, arguments...);
      pcout << "KINSOLAdapter: registered contributor";
      if (!prefix.empty())
        pcout << " '" << prefix << "'";
      pcout << "; " << composition_.n_fields()
            << " semantic field(s) available." << std::endl;
      return ProblemHandle<KINSOLAdapter, decltype(fields)>(*this,
                                                            std::move(fields));
    }

    void
    reinit(GlobalVectorType &state)
    {
      composition_.reinit(state);
      pcout << "KINSOLAdapter: execution vector initialized with "
            << composition_.n_fields() << " semantic field(s)." << std::endl;
    }

    GlobalVectorType
    make_state() const
    {
      return composition_.make_state();
    }

    FieldVectorType &
    field(GlobalVectorType &state, const FieldId id)
    {
      return composition_.field(state, id);
    }

    const FieldVectorType &
    field(const GlobalVectorType &state, const FieldId id) const
    {
      return composition_.field(state, id);
    }

    dealii::SUNDIALS::KINSOL<GlobalVectorType> &
    solver()
    {
      finalize();
      return *kinsol_;
    }

    const Operator &
    current_jacobian() const
    {
      AssertThrow(current_jacobian_.has_value(),
                  dealii::ExcMessage("KINSOL has not prepared a Jacobian."));
      return *current_jacobian_;
    }

    bool
    has_current_preconditioner() const
    {
      return current_preconditioner_.has_value();
    }

    /** Solve the steady nonlinear system F(y)=0 from the supplied guess. */
    unsigned int
    solve(GlobalVectorType &state)
    {
      AssertThrow(!composition_.model().has_derivative_terms(),
                  dealii::ExcMessage(
                    "KINSOLAdapter solves steady systems F(y)=0; derivative "
                    "terms belong to a transient execution adapter."));
      finalize();
      pcout << "KINSOLAdapter: starting steady nonlinear solve with "
            << composition_.n_fields() << " semantic field(s)." << std::endl;
      const auto n_iterations = kinsol_->solve(state);
      pcout << "KINSOLAdapter: nonlinear solve finished after " << n_iterations
            << " iteration(s)." << std::endl;
      return n_iterations;
    }

  private:
    void
    finalize()
    {
      if (connected_)
        return;

      AssertThrow(composition_.n_fields() > 0,
                  dealii::ExcMessage("KINSOLAdapter has no semantic fields."));
      kinsol_ = std::make_unique<dealii::SUNDIALS::KINSOL<GlobalVectorType>>(
        additional_data(), composition_.communicator());
      kinsol_->reinit_vector = [this](GlobalVectorType &vector) {
        composition_.reinit(vector);
      };
      kinsol_->residual = [this](const GlobalVectorType &state,
                                 GlobalVectorType       &residual) {
        composition_.evaluate_residual(0., state, nullptr, residual);
      };
      kinsol_->setup_jacobian = [this](const GlobalVectorType &state,
                                       const GlobalVectorType &) {
        prepare_jacobian(state);
      };
      kinsol_->solve_with_jacobian = [this](const GlobalVectorType &rhs,
                                            GlobalVectorType       &dst,
                                            const double            tolerance) {
        AssertThrow(current_jacobian_.has_value(),
                    dealii::ExcMessage(
                      "KINSOL requested a solve without a current Jacobian."));
        if (solve_)
          {
            pcout << "KINSOLAdapter: invoking the configured linear solver."
                  << std::endl;
            solve_(*current_jacobian_, rhs, dst, tolerance);
            pcout << "KINSOLAdapter: configured linear solver finished."
                  << std::endl;
          }
        else
          {
            dst = 0.;
            dealii::SolverControl control(5000, std::max(1.e-12, tolerance));
            dealii::SolverGMRES<GlobalVectorType> solver(control);
            if (current_preconditioner_.has_value())
              if (current_solver_is_flexible_)
                {
                  typename dealii::SolverFGMRES<
                    GlobalVectorType>::AdditionalData    flexible_data(100);
                  dealii::SolverFGMRES<GlobalVectorType> flexible_solver(
                    control, flexible_data);
                  flexible_solver.solve(*current_jacobian_,
                                        dst,
                                        rhs,
                                        *current_preconditioner_);
                  pcout << "KINSOLAdapter: FGMRES converged in "
                        << control.last_step() << " iteration(s), residual "
                        << control.last_value() << "." << std::endl;
                }
              else
                {
                  solver.solve(*current_jacobian_,
                               dst,
                               rhs,
                               *current_preconditioner_);
                  pcout << "KINSOLAdapter: GMRES converged in "
                        << control.last_step() << " iteration(s), residual "
                        << control.last_value() << "." << std::endl;
                }
            else
              {
                solver.solve(*current_jacobian_,
                             dst,
                             rhs,
                             dealii::PreconditionIdentity());
                pcout << "KINSOLAdapter: GMRES converged in "
                      << control.last_step() << " iteration(s), residual "
                      << control.last_value() << "." << std::endl;
              }
          }
      };
      connected_ = true;
    }

    Operator
    retain_snapshot(const Operator &operator_view,
                    const std::shared_ptr<typename Composition::StateSnapshot>
                      &snapshot) const
    {
      Operator result;
      result.reinit_range_vector =
        [operator_view, snapshot](GlobalVectorType &vector, const bool omit) {
          operator_view.reinit_range_vector(vector, omit);
        };
      result.reinit_domain_vector =
        [operator_view, snapshot](GlobalVectorType &vector, const bool omit) {
          operator_view.reinit_domain_vector(vector, omit);
        };
      result.vmult = [operator_view, snapshot](GlobalVectorType &destination,
                                               const GlobalVectorType &source) {
        operator_view.vmult(destination, source);
      };
      result.vmult_add = [operator_view,
                          snapshot](GlobalVectorType       &destination,
                                    const GlobalVectorType &source) {
        operator_view.vmult_add(destination, source);
      };
      result.Tvmult = [operator_view,
                       snapshot](GlobalVectorType       &destination,
                                 const GlobalVectorType &source) {
        operator_view.Tvmult(destination, source);
      };
      result.Tvmult_add = [operator_view,
                           snapshot](GlobalVectorType       &destination,
                                     const GlobalVectorType &source) {
        operator_view.Tvmult_add(destination, source);
      };
      return result;
    }

    void
    prepare_jacobian(const GlobalVectorType &state)
    {
      const auto snapshot      = composition_.make_snapshot(0., state);
      const auto operator_view = composition_.jacobian(snapshot->context, 0.);
      current_jacobian_        = retain_snapshot(operator_view, snapshot);
      current_preconditioner_.reset();
      current_solver_is_flexible_ = false;

      if (!solve_)
        {
          if (!composition_.saddle_points().empty())
            {
              const auto &saddle = composition_.saddle_points().front();
              current_preconditioner_ =
                retain_snapshot(composition_.schur_preconditioner(
                                  saddle.multiplier, snapshot->state_storage),
                                snapshot);
              current_solver_is_flexible_ = true;
            }
          else if (composition_.has_complete_local_preconditioners())
            {
              current_preconditioner_ =
                retain_snapshot(composition_.n_fields() > 1 ?
                                  composition_.block_triangular_preconditioner(
                                    snapshot->state_storage) :
                                  composition_.block_diagonal_preconditioner(
                                    snapshot->state_storage),
                                snapshot);
              current_solver_is_flexible_ = composition_.n_fields() > 1;
            }
        }
    }

    Composition                           composition_;
    LinearSolveFunction                   solve_;
    const KINSOLAdapterParameters        &parameters_;
    mutable std::optional<AdditionalData> additional_data_;
    std::unique_ptr<dealii::SUNDIALS::KINSOL<GlobalVectorType>> kinsol_;
    mutable dealii::ConditionalOStream                          pcout;
    std::optional<Operator> current_jacobian_;
    std::optional<Operator> current_preconditioner_;
    bool                    current_solver_is_flexible_ = false;
    bool                    connected_                  = false;
  };
} // namespace ImmersX
#endif

#endif // immersx_sundials_kinsol_adapter_h
