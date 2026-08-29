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

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include <immersx/core/detail/execution_composition.h>
#include <immersx/core/problem_handle.h>
#include <immersx/core/representation.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#ifdef DEAL_II_WITH_SUNDIALS
#  include <deal.II/sundials/ida.h>
#endif

#ifdef DEAL_II_WITH_SUNDIALS
namespace ImmersX
{
  /** Public transient DAE adapter backed by the private composition engine. */
  template <typename FieldVectorType, typename GlobalVectorType>
  class IDAAdapter
  {
  public:
    using RepresentationType = Representation<FieldVectorType>;
    using ComponentRepresentationType =
      ComponentRepresentation<FieldVectorType>;
    using Operator            = dealii::LinearOperator<GlobalVectorType>;
    using LinearSolveFunction = std::function<void(const Operator &,
                                                   const GlobalVectorType &,
                                                   GlobalVectorType &,
                                                   double)>;
    using AdditionalData =
      typename dealii::SUNDIALS::IDA<GlobalVectorType>::AdditionalData;

    IDAAdapter(const AdditionalData &data,
               const MPI_Comm        communicator,
               LinearSolveFunction   solve = {})
      : composition_(communicator)
      , solve_(std::move(solve))
      , ida_(data, communicator)
    {}

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
    }

    dealii::SUNDIALS::IDA<GlobalVectorType> &
    solver()
    {
      finalize();
      return ida_;
    }

    unsigned int
    solve(GlobalVectorType &state, GlobalVectorType &state_dot)
    {
      finalize();
      return ida_.solve_dae(state, state_dot);
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

  private:
    using Composition =
      detail::ExecutionComposition<FieldVectorType, GlobalVectorType>;

    void
    finalize()
    {
      if (connected_)
        return;

      AssertThrow(composition_.n_fields() > 0,
                  dealii::ExcMessage("IDAAdapter has no semantic fields."));
      ida_.reinit_vector = [this](GlobalVectorType &vector) {
        composition_.reinit(vector);
      };
      ida_.residual = [this](const double            time,
                             const GlobalVectorType &state,
                             const GlobalVectorType &state_dot,
                             GlobalVectorType       &residual) {
        composition_.evaluate_residual(time, state, &state_dot, residual);
      };
      ida_.setup_jacobian = [this](const double            time,
                                   const GlobalVectorType &state,
                                   const GlobalVectorType &state_dot,
                                   const double            alpha) {
        prepare_jacobian(time, state, state_dot, alpha);
      };
      ida_.solve_with_jacobian = [this](const GlobalVectorType &rhs,
                                        GlobalVectorType       &dst,
                                        const double            tolerance) {
        AssertThrow(current_jacobian_.has_value(),
                    dealii::ExcMessage("IDA requested a solve without a "
                                       "current Jacobian."));
        if (solve_)
          solve_(*current_jacobian_, rhs, dst, tolerance);
        else
          {
            dealii::SolverControl                 control(1000, tolerance);
            dealii::SolverGMRES<GlobalVectorType> solver(control);
            solver.solve(*current_jacobian_,
                         dst,
                         rhs,
                         dealii::PreconditionIdentity());
          }
      };
      ida_.differential_components = [this]() {
        return composition_.differential_components();
      };
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
    }

    Composition                             composition_;
    LinearSolveFunction                     solve_;
    dealii::SUNDIALS::IDA<GlobalVectorType> ida_;
    std::optional<Operator>                 current_jacobian_;
    std::size_t                             coupling_count_ = 0;
    bool                                    connected_      = false;
  };
} // namespace ImmersX
#endif

#endif // immersx_sundials_ida_adapter_h
