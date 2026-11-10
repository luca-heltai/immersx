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

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/field.h>
#include <immersx/core/matrix_operator.h>
#include <immersx/core/state.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ImmersX
{
  namespace detail
  {
    template <typename FieldVectorType, typename GlobalVectorType>
    class ExecutionComposition;
  }

  template <typename VectorType, typename MatrixType>
  class SemidiscreteTerm;

  template <typename VectorType, typename MatrixType>
  class SemidiscreteBuilder;

  /** Semantic description of one multiplier/primal saddle-point relation.
   * The physical multiplier metric is state-dependent model data and is kept
   * in the model's metric-factory registry, rather than duplicated here. */
  template <typename VectorType, typename MatrixType>
  struct SaddlePointMetadata
  {
    using MatrixOperator = MaterializedOperator<VectorType, MatrixType>;

    FieldId              multiplier;
    std::vector<FieldId> participants;
  };

  /** Mesh-dependent data for the symmetric spatial block solves. */
  struct SecondOrderPreconditionerData
  {
    dealii::IndexSet                                constrained;
    ImmersXLA::MPI::PreconditionAMG::AdditionalData amg_data;
  };

  /** Two fields satisfying u_dot = v and M v_dot + K u + D v = f.
   * This physical relation permits sparse Schur reduction in an execution
   * adapter without exposing integration coefficients to the contributor.
   * The revision changes when the spatial operators or constraint topology
   * change. Mesh data are requested only when that revision changes.
   */
  struct SecondOrderPair
  {
    FieldId                                        displacement;
    FieldId                                        velocity;
    std::function<unsigned long()>                 revision;
    std::function<SecondOrderPreconditionerData()> preconditioner_data;
  };

  /** A term-wise semi-discrete residual model for F(t,y,ydot)=0. */
  template <typename VectorType,
            typename MatrixType = ImmersXLA::MPI::SparseMatrix>
  class SemiDiscreteModel
  {
  public:
    using Context              = EvaluationContext<VectorType>;
    using Operation            = dealii::PackagedOperation<VectorType>;
    using Operator             = dealii::LinearOperator<VectorType, VectorType>;
    using MatrixOperator       = MaterializedOperator<VectorType, MatrixType>;
    using ResidualFactory      = std::function<Operation(const Context &)>;
    using ContextUpdateFactory = std::function<void(const Context &)>;
    using OperatorFactory      = std::function<Operator(const Context &)>;
    using MatrixOperatorFactory =
      std::function<MatrixOperator(const Context &)>;
    using SaddlePointMetadata =
      ImmersX::SaddlePointMetadata<VectorType, MatrixType>;
    using VectorReinitializer = std::function<void(VectorType &, bool)>;
    using PreconditionerFactory =
      std::function<Operator(const MatrixType &, const VectorReinitializer &)>;

    void
    update_context(const Context &context) const
    {
      for (const auto &factory : context_updates_)
        factory(context);
    }

    /** Retain a linearization at one execution-owned evaluation point.
     * Materialize each eligible contribution before another factory can change
     * a native matrix observed by its view. Shared materialized storage is
     * cloned, and each retained view owns its matrix. Matrix-free actions
     * remain actions.
     */
    SemiDiscreteModel
    retained_linearization(const Context &context) const
    {
      SemiDiscreteModel result = *this;
      const auto        retain = [&context](auto &registry) {
        for (auto &[key, entries] : registry)
          for (auto &entry : entries)
            if (context.terms().includes(entry.term, TermTreatment::all))
              {
                if (entry.matrix_factory)
                  {
                    auto op = entry.matrix_factory(context);
                    if (op.is_materializable())
                      {
                        auto matrix = op.matrix();
                        if (matrix.use_count() > 1)
                          matrix = detail::clone_matrix(*matrix);
                        op = matrix_operator<VectorType>(std::move(matrix));
                      }
                    entry.factory = [op](const Context &) { return op.view; };
                    entry.matrix_factory = [op](const Context &) { return op; };
                  }
                else
                  {
                    const auto op = entry.factory(context);
                    entry.factory = [op](const Context &) { return op; };
                  }
              }
      };
      retain(result.state_operators_);
      retain(result.derivative_operators_);
      result.context_updates_.clear();
      return result;
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

    /** Return a matrix-backed block, if every active term has provenance. */
    std::optional<MatrixOperator>
    state_matrix_operator(const FieldId  row,
                          const FieldId  column,
                          const Context &context) const
    {
      return combine_matrix_operators(state_operators_, row, column, context);
    }

    Operator
    derivative_operator(const FieldId  row,
                        const FieldId  column,
                        const Context &context) const
    {
      return combine_operators(derivative_operators_, row, column, context);
    }

    /** Return a materializable dF/dydot block, when available. */
    std::optional<MatrixOperator>
    derivative_matrix_operator(const FieldId  row,
                               const FieldId  column,
                               const Context &context) const
    {
      return combine_matrix_operators(derivative_operators_,
                                      row,
                                      column,
                                      context);
    }

    bool
    has_derivative_terms() const
    {
      return !derivative_operators_.empty();
    }

    bool
    has_preconditioner(const FieldId field) const
    {
      return preconditioners_.find(field) != preconditioners_.end();
    }

    const std::vector<SaddlePointMetadata> &
    saddle_points() const
    {
      return saddle_points_;
    }

    const std::vector<SecondOrderPair> &
    second_order_pairs() const
    {
      return second_order_pairs_;
    }

    void
    add_multiplier_metric(const FieldId         multiplier,
                          MatrixOperatorFactory factory)
    {
      AssertThrow(factory,
                  dealii::ExcMessage(
                    "A multiplier metric factory cannot be empty."));
      AssertThrow(
        multiplier_metrics_.find(multiplier) == multiplier_metrics_.end(),
        dealii::ExcMessage("A multiplier metric was registered twice."));
      multiplier_metrics_.emplace(multiplier, std::move(factory));
    }

    bool
    has_multiplier_metric(const FieldId multiplier) const
    {
      return multiplier_metrics_.find(multiplier) != multiplier_metrics_.end();
    }

    std::optional<MatrixOperator>
    multiplier_metric(const FieldId multiplier, const Context &context) const
    {
      const auto it = multiplier_metrics_.find(multiplier);
      if (it == multiplier_metrics_.end())
        return std::nullopt;
      auto metric = it->second(context);
      if (!metric.is_materializable())
        return std::nullopt;
      return metric;
    }

    std::optional<Operator>
    preconditioner(const FieldId              field,
                   const MatrixType          &matrix,
                   const VectorReinitializer &reinitializer) const
    {
      const auto it = preconditioners_.find(field);
      if (it == preconditioners_.end())
        return std::nullopt;
      return it->second(matrix, reinitializer);
    }

    bool
    has_state_operator(const FieldId row, const FieldId column) const
    {
      return state_operators_.find({row.value(), column.value()}) !=
             state_operators_.end();
    }

    bool
    has_derivative_operator(const FieldId row, const FieldId column) const
    {
      return derivative_operators_.find({row.value(), column.value()}) !=
             derivative_operators_.end();
    }

  private:
    template <typename, typename>
    friend class SemidiscreteTerm;

    template <typename, typename>
    friend class SemidiscreteBuilder;

    template <typename, typename>
    friend class detail::ExecutionComposition;

    void
    add_residual(const FieldId row, std::string term, ResidualFactory factory)
    {
      AssertThrow(factory,
                  dealii::ExcMessage("A residual factory cannot be empty."));
      residuals_[row].push_back({std::move(term), std::move(factory)});
    }

    void
    add_context_update(ContextUpdateFactory factory)
    {
      AssertThrow(factory,
                  dealii::ExcMessage(
                    "A context update factory cannot be empty."));
      context_updates_.push_back(std::move(factory));
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
    add_state_operator(const FieldId         row,
                       const FieldId         column,
                       std::string           term,
                       const MatrixOperator &op)
    {
      add_state_operator(row,
                         column,
                         std::move(term),
                         MatrixOperatorFactory(
                           [op](const Context &) { return op; }));
    }

    void
    add_state_operator(const FieldId         row,
                       const FieldId         column,
                       std::string           term,
                       MatrixOperatorFactory factory)
    {
      AssertThrow(factory,
                  dealii::ExcMessage(
                    "A materialized operator factory cannot be empty."));
      const auto view_factory = [factory](const Context &context) {
        return factory(context).view;
      };
      state_operators_[{row.value(), column.value()}].push_back(
        {std::move(term), view_factory, std::move(factory)});
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
        {std::move(term), std::move(factory), {}});
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
    add_derivative_operator(const FieldId         row,
                            const FieldId         column,
                            std::string           term,
                            const MatrixOperator &op)
    {
      add_derivative_operator(row,
                              column,
                              std::move(term),
                              MatrixOperatorFactory(
                                [op](const Context &) { return op; }));
    }

    void
    add_derivative_operator(const FieldId         row,
                            const FieldId         column,
                            std::string           term,
                            MatrixOperatorFactory factory)
    {
      AssertThrow(factory,
                  dealii::ExcMessage(
                    "A materialized operator factory cannot be empty."));
      const auto view_factory = [factory](const Context &context) {
        return factory(context).view;
      };
      derivative_operators_[{row.value(), column.value()}].push_back(
        {std::move(term), view_factory, std::move(factory)});
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
        {std::move(term), std::move(factory), {}});
    }

    void
    add_preconditioner(const FieldId field, PreconditionerFactory factory)
    {
      AssertThrow(factory,
                  dealii::ExcMessage(
                    "A preconditioner factory cannot be empty."));
      AssertThrow(!has_preconditioner(field),
                  dealii::ExcMessage(
                    "A local preconditioner was registered twice."));
      preconditioners_.emplace(field, std::move(factory));
    }

    using SecondOrderOperators = std::array<MatrixOperator, 3>;

    /** Supply sparse approximate M, K, D for second-order preconditioning.
     * Exact residual/Jacobian operators are unchanged.
     */
    void
    set_second_order_preconditioner_operators(const FieldId        displacement,
                                              SecondOrderOperators operators)
    {
      AssertThrow(std::any_of(second_order_pairs_.begin(),
                              second_order_pairs_.end(),
                              [displacement](const auto &pair) {
                                return pair.displacement == displacement;
                              }),
                  dealii::ExcMessage(
                    "A second-order preconditioner needs a registered pair."));
      const bool inserted =
        second_order_surrogates_.emplace(displacement, std::move(operators))
          .second;
      AssertThrow(
        inserted,
        dealii::ExcMessage(
          "A second-order preconditioning surrogate was registered twice."));
    }

    std::optional<SecondOrderOperators>
    second_order_preconditioner_operators(const FieldId displacement) const
    {
      const auto entry = second_order_surrogates_.find(displacement);
      if (entry == second_order_surrogates_.end())
        return std::nullopt;
      return entry->second;
    }

    void
    add_second_order_pair(SecondOrderPair pair)
    {
      AssertThrow(pair.displacement != pair.velocity,
                  dealii::ExcMessage("A second-order pair needs two fields."));
      AssertThrow(pair.revision && pair.preconditioner_data,
                  dealii::ExcMessage("A second-order pair needs a spatial "
                                     "revision and mesh-data factory."));
      for (const auto &existing : second_order_pairs_)
        AssertThrow(existing.displacement != pair.displacement &&
                      existing.displacement != pair.velocity &&
                      existing.velocity != pair.displacement &&
                      existing.velocity != pair.velocity,
                    dealii::ExcMessage(
                      "A field belongs to more than one second-order pair."));
      second_order_pairs_.push_back(std::move(pair));
    }

    void
    add_saddle_point(SaddlePointMetadata metadata)
    {
      AssertThrow(!metadata.participants.empty(),
                  dealii::ExcMessage(
                    "A saddle-point relation needs a primal field."));
      for (const auto &entry : saddle_points_)
        AssertThrow(entry.multiplier != metadata.multiplier,
                    dealii::ExcMessage(
                      "A multiplier FieldId was registered twice."));
      saddle_points_.push_back(std::move(metadata));
    }

    std::vector<std::pair<FieldId, FieldId>>
    active_operator_blocks() const
    {
      std::vector<std::pair<FieldId, FieldId>> result;
      for (const auto &entry : state_operators_)
        result.emplace_back(FieldId(entry.first.first),
                            FieldId(entry.first.second));
      for (const auto &entry : derivative_operators_)
        {
          const auto pair = std::make_pair(FieldId(entry.first.first),
                                           FieldId(entry.first.second));
          if (std::find(result.begin(), result.end(), pair) == result.end())
            result.push_back(pair);
        }
      return result;
    }

    struct ResidualEntry
    {
      std::string     term;
      ResidualFactory factory;
    };

    struct OperatorEntry
    {
      std::string           term;
      OperatorFactory       factory;
      MatrixOperatorFactory matrix_factory;
    };

    using BlockKey = std::pair<std::size_t, std::size_t>;

    static Operator
    sum_operator_views(std::vector<Operator> operators)
    {
      Assert(!operators.empty(), dealii::ExcInternalError());
      if (operators.size() == 1)
        return std::move(operators.front());

      // Repeated LinearOperator additions copy the entire previous sum into
      // each action callback. Share the flat list so storage grows linearly
      // with the number of contributions, including when the sum is copied.
      const auto terms =
        std::make_shared<const std::vector<Operator>>(std::move(operators));
      Operator result;
      result.reinit_range_vector  = terms->front().reinit_range_vector;
      result.reinit_domain_vector = terms->front().reinit_domain_vector;
      result.is_null_operator =
        std::all_of(terms->begin(), terms->end(), [](const auto &op) {
          return op.is_null_operator;
        });
      result.vmult = [terms](VectorType &dst, const VectorType &src) {
        terms->front().vmult(dst, src);
        for (std::size_t i = 1; i < terms->size(); ++i)
          (*terms)[i].vmult_add(dst, src);
      };
      result.vmult_add = [terms](VectorType &dst, const VectorType &src) {
        for (const auto &op : *terms)
          op.vmult_add(dst, src);
      };
      result.Tvmult = [terms](VectorType &dst, const VectorType &src) {
        terms->back().Tvmult(dst, src);
        for (std::size_t i = terms->size() - 1; i > 0; --i)
          (*terms)[i - 1].Tvmult_add(dst, src);
      };
      result.Tvmult_add = [terms](VectorType &dst, const VectorType &src) {
        for (auto it = terms->rbegin(); it != terms->rend(); ++it)
          it->Tvmult_add(dst, src);
      };
      return result;
    }

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

      return sum_operator_views(std::move(operators));
    }

    template <typename Registry>
    std::optional<MatrixOperator>
    combine_matrix_operators(const Registry &registry,
                             const FieldId   row,
                             const FieldId   column,
                             const Context  &context) const
    {
      const auto it = registry.find({row.value(), column.value()});
      if (it == registry.end())
        return std::nullopt;

      std::vector<MatrixOperator> operators;
      for (const auto &entry : it->second)
        if (context.terms().includes(entry.term, TermTreatment::all))
          {
            if (!entry.matrix_factory)
              return std::nullopt;
            operators.push_back(entry.matrix_factory(context));
            if (!operators.back().is_materializable())
              return std::nullopt;
          }
      if (operators.empty())
        return std::nullopt;
      if (operators.size() == 1)
        return operators.front();

      MatrixOperator        result;
      std::vector<Operator> views;
      views.reserve(operators.size());
      for (const auto &op : operators)
        views.push_back(op.view);
      result.view        = sum_operator_views(std::move(views));
      result.materialize = [operators]() {
        return detail::sum_matrices(operators);
      };
      result.materialize_into = [operators](MatrixType &destination) {
        const auto matrix = detail::sum_matrices(operators);
        if constexpr (detail::has_sparse_reinit<MatrixType>::value &&
                      !detail::has_distributed_partitions<MatrixType>::value)
          {
            AssertThrow(
              destination.m() == matrix->m() && destination.n() == matrix->n(),
              dealii::ExcMessage(
                "Initialize the destination sparsity before assembling a serial sparse matrix."));
            destination = 0.;
            detail::set_matrix_values(destination, *matrix);
          }
        else
          detail::copy_matrix(destination, *matrix);
      };
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
      return dealii::null_operator(result);
    }

    std::map<FieldId, std::vector<ResidualEntry>>  residuals_;
    std::map<BlockKey, std::vector<OperatorEntry>> state_operators_;
    std::map<BlockKey, std::vector<OperatorEntry>> derivative_operators_;
    std::map<FieldId, PreconditionerFactory>       preconditioners_;
    std::map<FieldId, MatrixOperatorFactory>       multiplier_metrics_;
    std::vector<SaddlePointMetadata>               saddle_points_;
    std::vector<SecondOrderPair>                   second_order_pairs_;
    std::map<FieldId, SecondOrderOperators>        second_order_surrogates_;
    std::vector<ContextUpdateFactory>              context_updates_;
  };
} // namespace ImmersX

#endif // immersx_time_residual_h
