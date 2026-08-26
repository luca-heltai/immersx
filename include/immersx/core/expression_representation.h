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

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ImmersX
{
  /** A symbolic pointwise transform of one or more retained observables. */
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

    /** A semantic Field and the independent symbol bound to it. */
    struct Binding
    {
      FieldId     field;
      std::string symbol;
    };

    ExpressionRepresentation(const std::vector<Binding> &bindings,
                             const SamplingPlan         &sampling,
                             SymbolicExpressionKernel    kernel)
      : kernel_(std::make_shared<SymbolicExpressionKernel>(std::move(kernel)))
    {
      AssertThrow(!bindings.empty(),
                  dealii::ExcMessage(
                    "An expression representation needs at least one "
                    "binding."));
      AssertThrow(kernel_->n_independent_symbols() == bindings.size(),
                  dealii::ExcMessage(
                    "The number of expression symbols must match the number "
                    "of field bindings."));

      bindings_.reserve(bindings.size());
      for (const auto &binding : bindings)
        {
          bindings_.push_back({binding.field, binding.symbol, sampling});
          if (std::find(dependencies_.begin(),
                        dependencies_.end(),
                        binding.field) == dependencies_.end())
            dependencies_.push_back(binding.field);
        }
    }

    ExpressionRepresentation(const FieldId            source,
                             const SamplingPlan      &sampling,
                             SymbolicExpressionKernel kernel)
      : ExpressionRepresentation(std::vector<Binding>{{source, "A"}},
                                 sampling,
                                 std::move(kernel))
    {}

    ExpressionRepresentation(
      const std::vector<Binding>          &bindings,
      const SamplingPlan                  &sampling,
      const std::string                   &expression,
      const std::map<std::string, double> &constants = {})
      : ExpressionRepresentation(bindings,
                                 sampling,
                                 make_kernel(expression,
                                             symbols(bindings),
                                             constants))
    {}

    ExpressionRepresentation(
      const FieldId                        source,
      const SamplingPlan                  &sampling,
      const std::string                   &expression,
      const std::string                   &symbol    = "A",
      const std::map<std::string, double> &constants = {})
      : ExpressionRepresentation(std::vector<Binding>{{source, symbol}},
                                 sampling,
                                 expression,
                                 constants)
    {}

    /** Return the sole source Field; only valid for scalar expressions. */
    FieldId
    source() const
    {
      AssertThrow(bindings_.size() == 1,
                  dealii::ExcMessage(
                    "A multi-field expression has no single source field."));
      return bindings_.front().field;
    }

    const std::vector<FieldId> &
    dependencies() const
    {
      return dependencies_;
    }

    const SamplingPlan &
    sampling_plan() const
    {
      return bindings_.front().sampling;
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
      const auto samples = sample_bindings(context);
      value_type result;
      result.reinit(sampling_plan().locally_owned_points(),
                    sampling_plan().locally_relevant_points(),
                    sampling_plan().mpi_communicator());
      for (std::size_t q = 0; q < sampling_plan().points().size(); ++q)
        {
          std::vector<double> values;
          values.reserve(samples.size());
          for (const auto &sample : samples)
            values.push_back(sample[sampling_plan().point_index(q)]);
          result[sampling_plan().point_index(q)] =
            kernel_
              ->evaluate(sampling_plan().points()[q].point,
                         context.time(),
                         values)
              .value;
        }
      return result;
    }

    Operator
    linearize(const EvaluationContext<state_type> &context,
              const EvaluationRequest             &request = {}) const
    {
      (void)request;
      AssertThrow(dependencies_.size() == 1,
                  dealii::ExcMessage(
                    "Select a dependency when linearizing a multi-field "
                    "expression."));
      return linearize(context, dependencies_.front());
    }

    /** Return d p / d field for one semantic dependency. */
    Operator
    linearize(const EvaluationContext<state_type> &context,
              const FieldId                        field,
              const EvaluationRequest             &request = {}) const
    {
      (void)request;
      const auto samples = sample_bindings(context);
      Operator   result;
      bool       has_term = false;
      for (std::size_t binding_index = 0; binding_index < bindings_.size();
           ++binding_index)
        if (bindings_[binding_index].field == field)
          {
            auto diagonal = std::make_shared<value_type>();
            diagonal->reinit(sampling_plan().locally_owned_points(),
                             sampling_plan().locally_relevant_points(),
                             sampling_plan().mpi_communicator());
            for (std::size_t q = 0; q < sampling_plan().points().size(); ++q)
              {
                std::vector<double> values;
                values.reserve(samples.size());
                for (const auto &sample : samples)
                  values.push_back(sample[sampling_plan().point_index(q)]);
                (*diagonal)[sampling_plan().point_index(q)] =
                  kernel_
                    ->evaluate(sampling_plan().points()[q].point,
                               context.time(),
                               values)
                    .derivatives[binding_index];
              }

            const auto &state = context.state(field);
            auto        term =
              make_diagonal_operator(bindings_[binding_index].sampling,
                                     diagonal) *
              bindings_[binding_index].sampling.linearize(state);
            if (has_term)
              result += term;
            else
              {
                result   = term;
                has_term = true;
              }
          }

      AssertThrow(has_term,
                  dealii::ExcMessage(
                    "The requested field is not an expression dependency."));
      return result;
    }

  private:
    struct InstalledBinding
    {
      FieldId      field;
      std::string  symbol;
      SamplingPlan sampling;
    };

    static std::vector<std::string>
    symbols(const std::vector<Binding> &bindings)
    {
      std::vector<std::string> result;
      result.reserve(bindings.size());
      for (const auto &binding : bindings)
        result.push_back(binding.symbol);
      return result;
    }

    static SymbolicExpressionKernel
    make_kernel(const std::string                   &expression,
                const std::vector<std::string>      &symbols,
                const std::map<std::string, double> &constants)
    {
      SymbolicExpressionKernel kernel;
      kernel.initialize(expression, symbols, constants);
      return kernel;
    }

    std::vector<value_type>
    sample_bindings(const EvaluationContext<state_type> &context) const
    {
      std::vector<value_type> result;
      result.reserve(bindings_.size());
      for (const auto &binding : bindings_)
        {
          const auto sampling_operator =
            binding.sampling.linearize(context.state(binding.field));
          value_type samples;
          sampling_operator.reinit_range_vector(samples, false);
          sampling_operator.vmult(samples, context.state(binding.field));
          result.push_back(std::move(samples));
        }
      return result;
    }

    ValueOperator
    make_diagonal_operator(const SamplingPlan                &sampling,
                           const std::shared_ptr<value_type> &diagonal) const
    {
      ValueOperator result;
      const auto    owned        = sampling.locally_owned_points();
      const auto    relevant     = sampling.locally_relevant_points();
      const auto    communicator = sampling.mpi_communicator();
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

    std::vector<InstalledBinding>             bindings_;
    std::vector<FieldId>                      dependencies_;
    std::shared_ptr<SymbolicExpressionKernel> kernel_;
  };

  /** Construct a multi-field expression representation from expression text. */
  template <int spacedim, typename StateVectorType, typename QuantityVectorType>
  auto
  make_expression_representation(
    const std::vector<
      typename ExpressionRepresentation<spacedim,
                                        StateVectorType,
                                        QuantityVectorType>::Binding> &bindings,
    const RetainedSamplingPlan<spacedim, StateVectorType, QuantityVectorType>
                                        &sampling,
    const std::string                   &expression,
    const std::map<std::string, double> &constants = {})
    -> ExpressionRepresentation<spacedim, StateVectorType, QuantityVectorType>
  {
    return ExpressionRepresentation<spacedim,
                                    StateVectorType,
                                    QuantityVectorType>(bindings,
                                                        sampling,
                                                        expression,
                                                        constants);
  }

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
