// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_sundials_ida_adapter_h
#define immersx_sundials_ida_adapter_h

#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include <immersx/core/detail/execution_composition.h>
#include <immersx/core/problem_handle.h>
#include <immersx/core/representation.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifdef DEAL_II_WITH_SUNDIALS
#  include <deal.II/sundials/ida.h>
#endif

#ifdef DEAL_II_WITH_SUNDIALS
namespace ImmersX
{
  /** Public transient DAE adapter backed by the private composition engine. */
  template <typename FieldVectorType, typename GlobalVectorType>
  class IDAAdapter : public dealii::ParameterAcceptor
  {
  public:
    using Composition =
      detail::ExecutionComposition<FieldVectorType, GlobalVectorType>;
    using RepresentationType = Representation<FieldVectorType>;
    using ComponentRepresentationType =
      ComponentRepresentation<FieldVectorType>;
    using Operator            = dealii::LinearOperator<GlobalVectorType>;
    using LocalOperator       = dealii::LinearOperator<FieldVectorType>;
    using SaddlePointMetadata = typename Composition::SaddlePointMetadata;
    using LinearSolveFunction = std::function<void(const Operator &,
                                                   const GlobalVectorType &,
                                                   GlobalVectorType &,
                                                   double)>;
    using OutputFunction      = std::function<void(const double,
                                              const GlobalVectorType &,
                                              const GlobalVectorType &,
                                              const unsigned int)>;
    using AdditionalData =
      typename dealii::SUNDIALS::IDA<GlobalVectorType>::AdditionalData;

    IDAAdapter(const AdditionalData &data,
               const MPI_Comm        communicator,
               LinearSolveFunction   solve        = {},
               const std::string    &section_name = "IDA adapter")
      : dealii::ParameterAcceptor(section_name)
      , composition_(communicator)
      , solve_(std::move(solve))
      , data_(data)
      , pcout(std::cout,
              dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
    {}

    void
    declare_parameters(dealii::ParameterHandler &prm) override
    {
      data_.add_parameters(prm);
    }

    const AdditionalData &
    additional_data() const
    {
      return data_;
    }

    template <typename Problem, typename... Arguments>
    auto
    add(const Problem     &problem,
        const std::string &prefix = {},
        const Arguments &...arguments)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "IDAAdapter contributors must be added before reinit or "
                    "solve."));
      auto fields = composition_.add(problem, prefix, arguments...);
      pcout << "IDAAdapter: registered contributor";
      if (!prefix.empty())
        pcout << " '" << prefix << "'";
      pcout << "; " << composition_.n_fields()
            << " semantic field(s) available." << std::endl;
      return ProblemHandle<IDAAdapter, decltype(fields)>(*this,
                                                         std::move(fields));
    }

    template <typename Quantity, typename Target, typename Coupling>
    auto
    couple(const Quantity &quantity,
           const Target   &target,
           const Coupling &coupling)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "IDAAdapter couplings must be added before reinit or "
                    "solve."));
      auto interaction = detail::invoke_coupling(quantity, target, coupling, 0);
      return add(std::move(interaction),
                 "coupling" + std::to_string(coupling_count_++));
    }

    void
    reinit(GlobalVectorType &vector)
    {
      composition_.reinit(vector);
      pcout << "IDAAdapter: execution vector initialized with "
            << composition_.n_fields() << " semantic field(s)." << std::endl;
    }

    dealii::SUNDIALS::IDA<GlobalVectorType> &
    solver()
    {
      finalize();
      return *ida_;
    }

    unsigned int
    solve(GlobalVectorType &state, GlobalVectorType &state_dot)
    {
      finalize();
      pcout << "IDAAdapter: starting DAE solve with " << composition_.n_fields()
            << " semantic field(s)." << std::endl;
      const auto n_steps = ida_->solve_dae(state, state_dot);
      pcout << "IDAAdapter: DAE solve finished after " << n_steps
            << " accepted step(s)." << std::endl;
      return n_steps;
    }

    /** Register native output for IDA's accepted/interpolated output states. */
    void
    set_output_step(OutputFunction output)
    {
      AssertThrow(!connected_,
                  dealii::ExcMessage(
                    "IDAAdapter output must be configured before solve."));
      output_ = std::move(output);
      pcout << "IDAAdapter: accepted-state output callback configured."
            << std::endl;
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

    dealii::IndexSet
    differential_components()
    {
      finalize();
      return composition_.differential_components();
    }

    const Operator &
    current_jacobian() const
    {
      AssertThrow(current_jacobian_.has_value(),
                  dealii::ExcMessage("IDA has not prepared a Jacobian."));
      return *current_jacobian_;
    }

    bool
    has_current_preconditioner() const
    {
      return current_preconditioner_.has_value();
    }

    bool
    can_materialize_matrix(const GlobalVectorType &state,
                           const GlobalVectorType &state_dot,
                           const double            alpha) const
    {
      return composition_.can_materialize_matrix(state, &state_dot, alpha);
    }

    typename Composition::BlockMatrixType
    block_matrix(const GlobalVectorType &state,
                 const GlobalVectorType &state_dot,
                 const double            alpha) const
    {
      return composition_.block_matrix(state, &state_dot, alpha);
    }

    typename Composition::MatrixType
    monolithic_matrix(const GlobalVectorType &state,
                      const GlobalVectorType &state_dot,
                      const double            alpha) const
    {
      return composition_.monolithic_matrix(
        composition_.block_matrix(state, &state_dot, alpha));
    }

    bool
    has_local_preconditioner(const FieldId field) const
    {
      return composition_.has_local_preconditioner(field);
    }

    LocalOperator
    local_preconditioner(const FieldId           field,
                         const GlobalVectorType &state) const
    {
      const auto result = composition_.local_preconditioner(field, state);
      AssertThrow(result.has_value(),
                  dealii::ExcMessage("No local preconditioner is registered "
                                     "for this Field."));
      return *result;
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

  private:
    void
    finalize()
    {
      if (connected_)
        return;

      AssertThrow(composition_.n_fields() > 0,
                  dealii::ExcMessage("IDAAdapter has no semantic fields."));
      ida_ = std::make_unique<dealii::SUNDIALS::IDA<GlobalVectorType>>(
        data_, composition_.communicator());
      ida_->reinit_vector = [this](GlobalVectorType &vector) {
        composition_.reinit(vector);
      };
      ida_->residual = [this](const double            time,
                              const GlobalVectorType &state,
                              const GlobalVectorType &state_dot,
                              GlobalVectorType       &residual) {
        composition_.evaluate_residual(time, state, &state_dot, residual);
      };
      ida_->setup_jacobian = [this](const double            time,
                                    const GlobalVectorType &state,
                                    const GlobalVectorType &state_dot,
                                    const double            alpha) {
        prepare_jacobian(time, state, state_dot, alpha);
      };
      ida_->solve_with_jacobian = [this](const GlobalVectorType &rhs,
                                         GlobalVectorType       &dst,
                                         const double            tolerance) {
        AssertThrow(current_jacobian_.has_value(),
                    dealii::ExcMessage("IDA requested a solve without a "
                                       "current Jacobian."));
        if (solve_)
          {
            pcout << "IDAAdapter: invoking the configured linear solver."
                  << std::endl;
            solve_(*current_jacobian_, rhs, dst, tolerance);
            pcout << "IDAAdapter: configured linear solver finished."
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
                  pcout << "IDAAdapter: FGMRES converged in "
                        << control.last_step() << " iteration(s), residual "
                        << control.last_value() << "." << std::endl;
                }
              else
                {
                  solver.solve(*current_jacobian_,
                               dst,
                               rhs,
                               *current_preconditioner_);
                  pcout << "IDAAdapter: GMRES converged in "
                        << control.last_step() << " iteration(s), residual "
                        << control.last_value() << "." << std::endl;
                }
            else
              {
                solver.solve(*current_jacobian_,
                             dst,
                             rhs,
                             dealii::PreconditionIdentity());
                pcout << "IDAAdapter: GMRES converged in "
                      << control.last_step() << " iteration(s), residual "
                      << control.last_value() << "." << std::endl;
              }
          }
      };
      ida_->differential_components = [this]() {
        return composition_.differential_components();
      };
      if (output_)
        {
          ida_->output_step =
            [this, output = output_](const double            time,
                                     const GlobalVectorType &state,
                                     const GlobalVectorType &state_dot,
                                     const unsigned int      step) {
              pcout << "IDAAdapter: accepted output step " << step
                    << " at time " << time << "." << std::endl;
              output(time, state, state_dot, step);
            };
        }
      pcout << "IDAAdapter: finalized with " << composition_.n_fields()
            << " semantic field(s)." << std::endl;
      connected_ = true;
    }

    void
    prepare_jacobian(const double            time,
                     const GlobalVectorType &state,
                     const GlobalVectorType &state_dot,
                     const double            alpha)
    {
      const auto snapshot = composition_.make_snapshot(time, state, state_dot);
      const auto operator_view =
        composition_.jacobian(snapshot->context, alpha);

      Operator stable;
      stable.reinit_range_vector = [operator_view](GlobalVectorType &vector,
                                                   const bool        omit) {
        operator_view.reinit_range_vector(vector, omit);
      };
      stable.reinit_domain_vector = [operator_view](GlobalVectorType &vector,
                                                    const bool        omit) {
        operator_view.reinit_domain_vector(vector, omit);
      };
      stable.vmult = [operator_view, snapshot](GlobalVectorType &destination,
                                               const GlobalVectorType &source) {
        operator_view.vmult(destination, source);
      };
      stable.vmult_add = [operator_view,
                          snapshot](GlobalVectorType       &destination,
                                    const GlobalVectorType &source) {
        operator_view.vmult_add(destination, source);
      };
      stable.Tvmult = [operator_view,
                       snapshot](GlobalVectorType       &destination,
                                 const GlobalVectorType &source) {
        operator_view.Tvmult(destination, source);
      };
      stable.Tvmult_add = [operator_view,
                           snapshot](GlobalVectorType       &destination,
                                     const GlobalVectorType &source) {
        operator_view.Tvmult_add(destination, source);
      };
      current_jacobian_ = std::move(stable);
      current_preconditioner_.reset();
      current_solver_is_flexible_ = false;
      if (!solve_)
        {
          const auto keep_snapshot = [snapshot](Operator operator_view) {
            Operator result;
            result.reinit_range_vector = [operator_view,
                                          snapshot](GlobalVectorType &vector,
                                                    const bool        omit) {
              operator_view.reinit_range_vector(vector, omit);
            };
            result.reinit_domain_vector = [operator_view,
                                           snapshot](GlobalVectorType &vector,
                                                     const bool        omit) {
              operator_view.reinit_domain_vector(vector, omit);
            };
            result.vmult = [operator_view,
                            snapshot](GlobalVectorType       &destination,
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
          };
          if (!composition_.saddle_points().empty())
            {
              const auto &saddle      = composition_.saddle_points().front();
              current_preconditioner_ = keep_snapshot(
                composition_.schur_preconditioner(saddle.multiplier,
                                                  snapshot->state_storage,
                                                  &snapshot->derivative_storage,
                                                  alpha));
              current_solver_is_flexible_ = true;
            }
          else if (composition_.has_complete_local_preconditioners())
            {
              current_preconditioner_ =
                keep_snapshot(composition_.n_fields() > 1 ?
                                composition_.block_triangular_preconditioner(
                                  snapshot->state_storage,
                                  true,
                                  &snapshot->derivative_storage,
                                  alpha) :
                                composition_.block_diagonal_preconditioner(
                                  snapshot->state_storage,
                                  &snapshot->derivative_storage,
                                  alpha));
              current_solver_is_flexible_ = composition_.n_fields() > 1;
            }
        }
    }

    Composition                                              composition_;
    LinearSolveFunction                                      solve_;
    AdditionalData                                           data_;
    std::unique_ptr<dealii::SUNDIALS::IDA<GlobalVectorType>> ida_;
    mutable dealii::ConditionalOStream                       pcout;
    std::optional<Operator>                                  current_jacobian_;
    std::optional<Operator> current_preconditioner_;
    OutputFunction          output_;
    bool                    current_solver_is_flexible_ = false;
    std::size_t             coupling_count_             = 0;
    bool                    connected_                  = false;
  };
} // namespace ImmersX
#endif

#endif // immersx_sundials_ida_adapter_h
