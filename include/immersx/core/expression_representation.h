// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_expression_representation_h
#define immersx_expression_representation_h

#include <immersx/core/representation.h>
#include <immersx/core/symbolic_expression_kernel.h>

#include <map>
#include <memory>
#include <string>
#include <utility>

namespace ImmersX
{
  /** A pointwise symbolic transform of one retained scalar FE observable. */
  template <int spacedim,
            typename StateVectorType    = ImmersXLA::MPI::Vector,
            typename QuantityVectorType = ImmersXLA::MPI::Vector>
  class ExpressionRepresentation
  {
  public:
    using value_type          = QuantityVectorType;
    using state_type          = StateVectorType;
    using quantity_space_type = QuantitySpace<value_type>;
    using SamplingPlan =
      RetainedSamplingPlan<spacedim, StateVectorType, QuantityVectorType>;
    using Operator      = RepresentationOperator<value_type, state_type>;
    using ValueOperator = RepresentationOperator<value_type, value_type>;

    ExpressionRepresentation(const FieldId            source,
                             const SamplingPlan      &sampling,
                             SymbolicExpressionKernel kernel)
      : source_(source)
      , sampling_(sampling)
      , kernel_(std::make_shared<SymbolicExpressionKernel>(std::move(kernel)))
    {
      AssertThrow(kernel_->n_independent_symbols() == 1,
                  dealii::ExcMessage(
                    "A scalar ExpressionRepresentation requires exactly one "
                    "independent symbol."));
    }

    ExpressionRepresentation(
      const FieldId                        source,
      const SamplingPlan                  &sampling,
      const std::string                   &expression,
      const std::string                   &symbol    = "A",
      const std::map<std::string, double> &constants = {})
      : ExpressionRepresentation(source,
                                 sampling,
                                 make_kernel(expression, symbol, constants))
    {}

    FieldId
    source() const
    {
      return source_;
    }

    const SamplingPlan &
    sampling_plan() const
    {
      return sampling_;
    }

    const SymbolicExpressionKernel &
    kernel() const
    {
      return *kernel_;
    }

    quantity_space_type
    quantity_space() const
    {
      return quantity_space_type(
        RepresentationDomain(spacedim, spacedim, "retained-fe-sampling"));
    }

    value_type
    evaluate(const EvaluationContext<state_type> &context,
             const EvaluationRequest             &request = {}) const
    {
      (void)request;
      const auto samples = sampled_values(context.state(source_));
      value_type result;
      result.reinit(sampling_.locally_owned_points(),
                    sampling_.locally_relevant_points(),
                    sampling_.mpi_communicator());
      for (std::size_t q = 0; q < sampling_.points().size(); ++q)
        result[sampling_.point_index(q)] =
          kernel_
            ->evaluate(sampling_.points()[q].point,
                       context.time(),
                       {samples[sampling_.point_index(q)]})
            .value;
      return result;
    }

    Operator
    linearize(const EvaluationContext<state_type> &context,
              const EvaluationRequest             &request = {}) const
    {
      (void)request;
      const auto &state    = context.state(source_);
      const auto  samples  = sampled_values(state);
      auto        diagonal = std::make_shared<value_type>();
      diagonal->reinit(sampling_.locally_owned_points(),
                       sampling_.locally_relevant_points(),
                       sampling_.mpi_communicator());
      for (std::size_t q = 0; q < sampling_.points().size(); ++q)
        (*diagonal)[sampling_.point_index(q)] =
          kernel_
            ->evaluate(sampling_.points()[q].point,
                       context.time(),
                       {samples[sampling_.point_index(q)]})
            .derivatives[0];

      const auto sampling_operator  = sampling_.linearize(state);
      auto       pointwise_operator = make_diagonal_operator(diagonal);
      return pointwise_operator * sampling_operator;
    }

  private:
    static SymbolicExpressionKernel
    make_kernel(const std::string                   &expression,
                const std::string                   &symbol,
                const std::map<std::string, double> &constants)
    {
      SymbolicExpressionKernel kernel;
      kernel.initialize(expression, {symbol}, constants);
      return kernel;
    }

    value_type
    sampled_values(const state_type &state) const
    {
      const auto sampling_operator = sampling_.linearize(state);
      value_type result;
      sampling_operator.reinit_range_vector(result, false);
      sampling_operator.vmult(result, state);
      return result;
    }

    ValueOperator
    make_diagonal_operator(const std::shared_ptr<value_type> &diagonal) const
    {
      ValueOperator result;
      const auto    owned        = sampling_.locally_owned_points();
      const auto    relevant     = sampling_.locally_relevant_points();
      const auto    communicator = sampling_.mpi_communicator();
      result.reinit_range_vector =
        [owned, relevant, communicator](value_type &vector, const bool omit) {
          vector.reinit(owned, relevant, communicator);
          if (!omit)
            vector = 0.;
        };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult                = [diagonal](value_type       &destination,
                                const value_type &source) {
        destination = source;
        destination.scale(*diagonal);
      };
      result.vmult_add = [diagonal](value_type       &destination,
                                    const value_type &source) {
        value_type contribution = source;
        contribution.scale(*diagonal);
        destination += contribution;
      };
      result.Tvmult     = result.vmult;
      result.Tvmult_add = result.vmult_add;
      return result;
    }

    FieldId                                   source_;
    SamplingPlan                              sampling_;
    std::shared_ptr<SymbolicExpressionKernel> kernel_;
  };

  /** Construct a scalar expression representation from expression text. */
  template <int spacedim, typename StateVectorType, typename QuantityVectorType>
  auto
  make_expression_representation(
    const FieldId source,
    const RetainedSamplingPlan<spacedim, StateVectorType, QuantityVectorType>
                                        &sampling,
    const std::string                   &expression,
    const std::string                   &symbol    = "A",
    const std::map<std::string, double> &constants = {})
    -> ExpressionRepresentation<spacedim, StateVectorType, QuantityVectorType>
  {
    return ExpressionRepresentation<spacedim,
                                    StateVectorType,
                                    QuantityVectorType>(
      source, sampling, expression, symbol, constants);
  }
} // namespace ImmersX

#endif // immersx_expression_representation_h
