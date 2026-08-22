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

    SemidiscreteBuilder(StateLayout &layout, Model &model)
      : layout_(layout)
      , model_(model)
    {}

    FieldId
    add_field(FieldDescriptor descriptor)
    {
      return layout_.add_field(std::move(descriptor));
    }

    FieldId
    add_field(const std::string      &prefix,
              const std::string      &local_name,
              const TimeRole          role,
              const dealii::IndexSet &locally_owned,
              const dealii::IndexSet &locally_relevant = {},
              const HistoryGroupId    history_group    = HistoryGroupId())
    {
      FieldDescriptor descriptor;
      descriptor.name = prefix.empty() ? local_name : prefix + "." + local_name;
      descriptor.time_role        = role;
      descriptor.history_group    = history_group;
      descriptor.locally_owned    = locally_owned;
      descriptor.locally_relevant = locally_relevant;
      return add_field(std::move(descriptor));
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

    StateLayout &
    layout()
    {
      return layout_;
    }

    Model &
    model()
    {
      return model_;
    }

  private:
    StateLayout &layout_;
    Model       &model_;
  };
} // namespace ImmersX

#endif // immersx_contributor_h
