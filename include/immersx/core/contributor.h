// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_contributor_h
#define immersx_contributor_h

#include <deal.II/lac/linear_operator.h>

#include <immersx/core/field.h>
#include <immersx/core/time_residual.h>

#include <functional>
#include <string>
#include <utility>

namespace ImmersX
{
  /**
   * Erase a backend-specific LinearOperator payload at an execution boundary.
   * This is needed when a global operator combines blocks with different
   * backend payload types.
   */
  template <typename Range, typename Domain, typename Payload>
  dealii::LinearOperator<Range, Domain>
  payload_free(const dealii::LinearOperator<Range, Domain, Payload> &op)
  {
    dealii::LinearOperator<Range, Domain> result;
    result.vmult = [op](Range &dst, const Domain &src) { op.vmult(dst, src); };
    result.vmult_add = [op](Range &dst, const Domain &src) {
      op.vmult_add(dst, src);
    };
    result.Tvmult = [op](Domain &dst, const Range &src) {
      op.Tvmult(dst, src);
    };
    result.Tvmult_add = [op](Domain &dst, const Range &src) {
      op.Tvmult_add(dst, src);
    };
    result.reinit_range_vector = [op](Range &dst, const bool omit) {
      op.reinit_range_vector(dst, omit);
    };
    result.reinit_domain_vector = [op](Domain &dst, const bool omit) {
      op.reinit_domain_vector(dst, omit);
    };
    return result;
  }

  /** Fallback customization point for callable contributors. */
  template <typename Builder, typename Callable, typename... Arguments>
  auto
  contribute(Builder        &builder,
             const Callable &callable,
             Arguments &&...arguments)
    -> decltype(callable(builder, std::forward<Arguments>(arguments)...))
  {
    return callable(builder, std::forward<Arguments>(arguments)...);
  }

  /** Direct handle for one semantically coherent residual term. */
  template <typename VectorType>
  class SemidiscreteTerm
  {
  public:
    using Model           = SemiDiscreteModel<VectorType>;
    using Operator        = typename Model::Operator;
    using ResidualFactory = typename Model::ResidualFactory;
    using OperatorFactory = typename Model::OperatorFactory;

    SemidiscreteTerm(Model &model, const FieldId row, std::string name)
      : model_(&model)
      , row_(row)
      , name_(std::move(name))
    {}

    template <typename Factory>
    SemidiscreteTerm &
    residual(Factory factory)
    {
      model_->add_residual(row_, name_, ResidualFactory(std::move(factory)));
      return *this;
    }

    SemidiscreteTerm &
    state(const FieldId column, const Operator &op)
    {
      model_->add_state_operator(row_, column, name_, op);
      return *this;
    }

    SemidiscreteTerm &
    state(const FieldId column, OperatorFactory factory)
    {
      model_->add_state_operator(row_, column, name_, std::move(factory));
      return *this;
    }

    SemidiscreteTerm &
    derivative(const FieldId column, const Operator &op)
    {
      model_->add_derivative_operator(row_, column, name_, op);
      return *this;
    }

    SemidiscreteTerm &
    derivative(const FieldId column, OperatorFactory factory)
    {
      model_->add_derivative_operator(row_, column, name_, std::move(factory));
      return *this;
    }

    FieldId
    row() const
    {
      return row_;
    }

    const std::string &
    name() const
    {
      return name_;
    }

  private:
    Model      *model_;
    FieldId     row_;
    std::string name_;
  };

  /** Minimal solver-neutral semantic authoring API. */
  template <typename VectorType>
  class SemidiscreteBuilder
  {
  public:
    using Model = SemiDiscreteModel<VectorType>;
    using Term  = SemidiscreteTerm<VectorType>;

    SemidiscreteBuilder(StateLayout &layout,
                        Model       &model,
                        std::string  prefix = {})
      : layout_(layout)
      , model_(model)
      , prefix_(std::move(prefix))
    {}

    FieldId
    field(const std::string      &local_name,
          const dealii::IndexSet &locally_owned,
          const dealii::IndexSet &locally_relevant        = {},
          const dealii::IndexSet &differential_components = {},
          const HistoryGroupId    history_group           = HistoryGroupId())
    {
      FieldDescriptor descriptor;
      descriptor.name =
        prefix_.empty() ? local_name : prefix_ + "." + local_name;
      descriptor.history_group           = history_group;
      descriptor.locally_owned           = locally_owned;
      descriptor.locally_relevant        = locally_relevant;
      descriptor.differential_components = differential_components;
      return layout_.add_field(std::move(descriptor));
    }

    FieldId
    differential_field(const std::string      &local_name,
                       const dealii::IndexSet &locally_owned,
                       const dealii::IndexSet &locally_relevant = {},
                       const HistoryGroupId    history_group = HistoryGroupId())
    {
      return field(local_name,
                   locally_owned,
                   locally_relevant,
                   locally_owned,
                   history_group);
    }

    FieldId
    algebraic_field(const std::string      &local_name,
                    const dealii::IndexSet &locally_owned,
                    const dealii::IndexSet &locally_relevant = {},
                    const HistoryGroupId    history_group    = HistoryGroupId())
    {
      return field(local_name,
                   locally_owned,
                   locally_relevant,
                   dealii::IndexSet(locally_owned.size()),
                   history_group);
    }

    Term
    term(const FieldId row, const std::string &local_name)
    {
      const auto qualified_name =
        prefix_.empty() ? local_name : prefix_ + "." + local_name;
      return Term(model_, row, qualified_name);
    }

  private:
    StateLayout &layout_;
    Model       &model_;
    std::string  prefix_;
  };
} // namespace ImmersX

#endif // immersx_contributor_h
