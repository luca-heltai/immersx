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

#include <immersx/core/field.h>
#include <immersx/core/time_residual.h>

#include <functional>
#include <string>
#include <utility>

namespace ImmersX
{
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
  template <typename VectorType,
            typename MatrixType = ImmersXLA::MPI::SparseMatrix>
  class SemidiscreteTerm
  {
  public:
    using Model                 = SemiDiscreteModel<VectorType, MatrixType>;
    using Operator              = typename Model::Operator;
    using MatrixOperator        = typename Model::MatrixOperator;
    using MatrixOperatorFactory = typename Model::MatrixOperatorFactory;
    using ResidualFactory       = typename Model::ResidualFactory;
    using OperatorFactory       = typename Model::OperatorFactory;

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
    state(const FieldId column, const MatrixOperator &op)
    {
      model_->add_state_operator(row_, column, name_, op);
      return *this;
    }

    SemidiscreteTerm &
    state(const FieldId column, MatrixOperatorFactory factory)
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

    SemidiscreteTerm &
    derivative(const FieldId column, const MatrixOperator &op)
    {
      model_->add_derivative_operator(row_, column, name_, op);
      return *this;
    }

    SemidiscreteTerm &
    derivative(const FieldId column, MatrixOperatorFactory factory)
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
  template <typename VectorType,
            typename MatrixType = ImmersXLA::MPI::SparseMatrix>
  class SemidiscreteBuilder
  {
  public:
    using Model = SemiDiscreteModel<VectorType, MatrixType>;
    using Term  = SemidiscreteTerm<VectorType, MatrixType>;

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

    /** Create a provenance-preserving operator from an assembled matrix. */
    typename Model::MatrixOperator
    matrix_operator(const MatrixType &matrix) const
    {
      return ImmersX::matrix_operator<VectorType, MatrixType>(matrix);
    }

    /** Register a Problem-local approximate inverse factory. */
    template <typename Factory>
    void
    preconditioner(const FieldId field, Factory factory)
    {
      model_.add_preconditioner(
        field, typename Model::PreconditionerFactory(std::move(factory)));
    }

    /** Register a semantic multiplier/primal saddle-point relation. */
    void
    saddle_point(const FieldId multiplier, std::vector<FieldId> participants)
    {
      model_.add_saddle_point({multiplier, std::move(participants)});
    }

    /** Register a saddle relation and its physical multiplier metric. */
    void
    saddle_point(const FieldId                         multiplier,
                 std::vector<FieldId>                  participants,
                 const typename Model::MatrixOperator &metric)
    {
      model_.add_saddle_point({multiplier, std::move(participants), true});
      model_.add_multiplier_metric(multiplier,
                                   typename Model::MatrixOperatorFactory(
                                     [metric](const auto &) {
                                       return metric;
                                     }));
    }

  private:
    StateLayout &layout_;
    Model       &model_;
    std::string  prefix_;
  };
} // namespace ImmersX

#endif // immersx_contributor_h
