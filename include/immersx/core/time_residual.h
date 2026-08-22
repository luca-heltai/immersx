// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_time_residual_h
#define immersx_time_residual_h

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <immersx/core/field.h>
#include <immersx/core/state.h>

#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ImmersX
{
  /**
   * A term-wise semi-discrete residual model for F(t,y,ydot)=0.
   *
   * Each semantic term owns its residual, state-Jacobian, and
   * state-derivative-Jacobian contributions. The execution adapter selects
   * terms through EvaluationContext and constructs the global operator.
   */
  template <typename VectorType>
  class SemiDiscreteModel
  {
  public:
    using Context         = EvaluationContext<VectorType>;
    using Operation       = dealii::PackagedOperation<VectorType>;
    using Operator        = dealii::LinearOperator<VectorType, VectorType>;
    using ResidualFactory = std::function<Operation(const Context &)>;
    using OperatorFactory = std::function<Operator(const Context &)>;

    void
    add_residual(const FieldId row, std::string term, ResidualFactory factory)
    {
      AssertThrow(factory,
                  dealii::ExcMessage("A residual factory cannot be empty."));
      residuals_[row].push_back({std::move(term), std::move(factory)});
    }

    void
    add_state_operator(const FieldId   row,
                       const FieldId   column,
                       std::string     term,
                       const Operator &op)
    {
      OperatorFactory factory = [op](const Context &) { return op; };
      add_state_operator(row, column, std::move(term), std::move(factory));
    }

    void
    add_state_operator(const FieldId   row,
                       const FieldId   column,
                       std::string     term,
                       OperatorFactory factory)
    {
      AssertThrow(factory,
                  dealii::ExcMessage("An operator factory cannot be empty."));
      state_operators_[{row.value(), column.value()}].push_back(
        {std::move(term), std::move(factory)});
    }

    void
    add_derivative_operator(const FieldId   row,
                            const FieldId   column,
                            std::string     term,
                            const Operator &op)
    {
      OperatorFactory factory = [op](const Context &) { return op; };
      add_derivative_operator(row, column, std::move(term), std::move(factory));
    }

    void
    add_derivative_operator(const FieldId   row,
                            const FieldId   column,
                            std::string     term,
                            OperatorFactory factory)
    {
      AssertThrow(factory,
                  dealii::ExcMessage("An operator factory cannot be empty."));
      derivative_operators_[{row.value(), column.value()}].push_back(
        {std::move(term), std::move(factory)});
    }

    void
    evaluate_row(const FieldId  row,
                 const Context &context,
                 VectorType    &destination) const
    {
      const auto it = residuals_.find(row);
      if (it == residuals_.end())
        return;
      for (const auto &entry : it->second)
        if (context.terms().includes(entry.term, TermTreatment::all))
          entry.factory(context).apply_add(destination);
    }

    Operator
    state_operator(const FieldId  row,
                   const FieldId  column,
                   const Context &context) const
    {
      return combine_operators(state_operators_, row, column, context);
    }

    Operator
    derivative_operator(const FieldId  row,
                        const FieldId  column,
                        const Context &context) const
    {
      return combine_operators(derivative_operators_, row, column, context);
    }

    bool
    has_derivative_terms() const
    {
      return !derivative_operators_.empty();
    }

  private:
    struct ResidualEntry
    {
      std::string     term;
      ResidualFactory factory;
    };

    struct OperatorEntry
    {
      std::string     term;
      OperatorFactory factory;
    };

    using BlockKey = std::pair<std::size_t, std::size_t>;

    template <typename Registry>
    Operator
    combine_operators(const Registry &registry,
                      const FieldId   row,
                      const FieldId   column,
                      const Context  &context) const
    {
      const auto  it     = registry.find({row.value(), column.value()});
      const auto &range  = context.state(row);
      const auto &domain = context.state(column);
      if (it == registry.end())
        return zero_operator(range, domain);

      std::vector<Operator> operators;
      for (const auto &entry : it->second)
        if (context.terms().includes(entry.term, TermTreatment::all))
          operators.push_back(entry.factory(context));
      if (operators.empty())
        return zero_operator(range, domain);

      Operator result = operators.front();
      for (std::size_t i = 1; i < operators.size(); ++i)
        result += operators[i];
      return result;
    }

    static Operator
    zero_operator(const VectorType &range, const VectorType &domain)
    {
      Operator result;
      result.reinit_range_vector = [range](VectorType &v, const bool omit) {
        v.reinit(range, omit);
      };
      result.reinit_domain_vector = [domain](VectorType &v, const bool omit) {
        v.reinit(domain, omit);
      };
      result.vmult      = [](VectorType &v, const VectorType &) { v = 0.; };
      result.vmult_add  = [](VectorType &, const VectorType &) {};
      result.Tvmult     = [](VectorType &v, const VectorType &) { v = 0.; };
      result.Tvmult_add = [](VectorType &, const VectorType &) {};
      return result;
    }

    std::map<FieldId, std::vector<ResidualEntry>>  residuals_;
    std::map<BlockKey, std::vector<OperatorEntry>> state_operators_;
    std::map<BlockKey, std::vector<OperatorEntry>> derivative_operators_;
  };
} // namespace ImmersX

#endif // immersx_time_residual_h
