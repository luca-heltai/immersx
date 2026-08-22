// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
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
   * Erase a backend-specific LinearOperator payload while retaining the
   * standard deal.II operator interface. This is useful at the execution
   * boundary, where a global operator may combine Trilinos and matrix-free
   * blocks without exposing a backend payload in its type.
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

  /**
   * Handle for one semantic residual term.
   *
   * Residual and both Jacobian parts are registered through the same name, so
   * term selection cannot accidentally retain only part of a linearization.
   */
  template <typename VectorType>
  class SemidiscreteTermHandle
  {
  public:
    using Model           = SemiDiscreteModel<VectorType>;
    using Operation       = typename Model::Operation;
    using Operator        = typename Model::Operator;
    using Context         = typename Model::Context;
    using ResidualFactory = typename Model::ResidualFactory;
    using OperatorFactory = typename Model::OperatorFactory;

    SemidiscreteTermHandle(
      std::function<void(const std::string &, ResidualFactory)>   add_residual,
      std::function<void(const std::string &, FieldId, Operator)> add_state,
      std::function<void(const std::string &, FieldId, Operator)>
        add_derivative,
      std::function<void(const std::string &, FieldId, OperatorFactory)>
        add_state_factory,
      std::function<void(const std::string &, FieldId, OperatorFactory)>
                    add_derivative_factory,
      const FieldId row,
      std::string   name)
      : add_residual_(std::move(add_residual))
      , add_state_(std::move(add_state))
      , add_derivative_(std::move(add_derivative))
      , add_state_factory_(std::move(add_state_factory))
      , add_derivative_factory_(std::move(add_derivative_factory))
      , row_(row)
      , name_(std::move(name))
    {}

    template <typename Factory>
    SemidiscreteTermHandle &
    residual(Factory factory)
    {
      add_residual_(name_, ResidualFactory(std::move(factory)));
      return *this;
    }

    SemidiscreteTermHandle &
    state(const FieldId column, const Operator &op)
    {
      add_state_(name_, column, op);
      return *this;
    }

    SemidiscreteTermHandle &
    state(const FieldId column, OperatorFactory factory)
    {
      add_state_factory_(name_, column, std::move(factory));
      return *this;
    }

    SemidiscreteTermHandle &
    derivative(const FieldId column, const Operator &op)
    {
      add_derivative_(name_, column, op);
      return *this;
    }

    SemidiscreteTermHandle &
    derivative(const FieldId column, OperatorFactory factory)
    {
      add_derivative_factory_(name_, column, std::move(factory));
      return *this;
    }

    FieldId
    row() const
    {
      return row_;
    }

  private:
    std::function<void(const std::string &, ResidualFactory)>   add_residual_;
    std::function<void(const std::string &, FieldId, Operator)> add_state_;
    std::function<void(const std::string &, FieldId, Operator)> add_derivative_;
    std::function<void(const std::string &, FieldId, OperatorFactory)>
      add_state_factory_;
    std::function<void(const std::string &, FieldId, OperatorFactory)>
                add_derivative_factory_;
    FieldId     row_;
    std::string name_;
  };

  /** Minimal solver-neutral semantic authoring API. */
  template <typename VectorType>
  class SemidiscreteBuilder
  {
  public:
    using Model           = SemiDiscreteModel<VectorType>;
    using Operation       = typename Model::Operation;
    using Operator        = typename Model::Operator;
    using Context         = typename Model::Context;
    using ResidualFactory = typename Model::ResidualFactory;
    using OperatorFactory = typename Model::OperatorFactory;

    SemidiscreteBuilder(StateLayout &layout,
                        Model       &model,
                        std::string  prefix = {})
      : layout_(layout)
      , model_(model)
      , prefix_(std::move(prefix))
    {}

    /** Add a field below this builder's contributor scope. */
    FieldId
    field(const std::string      &local_name,
          const TimeRole          role,
          const dealii::IndexSet &locally_owned,
          const dealii::IndexSet &locally_relevant = {},
          const HistoryGroupId    history_group    = HistoryGroupId())
    {
      FieldDescriptor descriptor;
      descriptor.name =
        prefix_.empty() ? local_name : prefix_ + "." + local_name;
      descriptor.time_role        = role;
      descriptor.history_group    = history_group;
      descriptor.locally_owned    = locally_owned;
      descriptor.locally_relevant = locally_relevant;
      return layout_.add_field(std::move(descriptor));
    }

    /** Register one semantically coherent residual term. */
    SemidiscreteTermHandle<VectorType>
    term(const FieldId row, const std::string &name)
    {
      using TermHandle = SemidiscreteTermHandle<VectorType>;
      return TermHandle(
        [this, row](const std::string &term, ResidualFactory factory) {
          add_residual(row, term, std::move(factory));
        },
        [this, row](const std::string &term,
                    const FieldId      column,
                    Operator op) { add_state_operator(row, column, term, op); },
        [this,
         row](const std::string &term, const FieldId column, Operator op) {
          add_derivative_operator(row, column, term, op);
        },
        [this, row](const std::string &term,
                    const FieldId      column,
                    OperatorFactory    factory) {
          add_state_operator(row, column, term, std::move(factory));
        },
        [this, row](const std::string &term,
                    const FieldId      column,
                    OperatorFactory    factory) {
          add_derivative_operator(row, column, term, std::move(factory));
        },
        row,
        name);
    }

    const std::string &
    prefix() const
    {
      return prefix_;
    }

    void
    add_residual(const FieldId row, std::string term, ResidualFactory factory)
    {
      model_.add_residual(row, std::move(term), std::move(factory));
    }

    void
    add_state_operator(const FieldId   row,
                       const FieldId   column,
                       std::string     term,
                       const Operator &op)
    {
      model_.add_state_operator(row, column, std::move(term), op);
    }

    void
    add_state_operator(const FieldId                   row,
                       const FieldId                   column,
                       std::string                     term,
                       typename Model::OperatorFactory factory)
    {
      model_.add_state_operator(row,
                                column,
                                std::move(term),
                                std::move(factory));
    }

    void
    add_derivative_operator(const FieldId   row,
                            const FieldId   column,
                            std::string     term,
                            const Operator &op)
    {
      model_.add_derivative_operator(row, column, std::move(term), op);
    }

    void
    add_derivative_operator(const FieldId                   row,
                            const FieldId                   column,
                            std::string                     term,
                            typename Model::OperatorFactory factory)
    {
      model_.add_derivative_operator(row,
                                     column,
                                     std::move(term),
                                     std::move(factory));
    }

  private:
    StateLayout &layout_;
    Model       &model_;
    std::string  prefix_;
  };
} // namespace ImmersX

#endif // immersx_contributor_h
