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
#include <variant>
#include <vector>

namespace ImmersX
{
  /** A value observable bound to one independent expression symbol. */
  struct ValueBinding
  {
    FieldId     field;
    std::string symbol;
  };

  /** A physical gradient component bound to one independent symbol. */
  struct GradientBinding
  {
    FieldId      field;
    std::string  symbol;
    unsigned int component;
  };

  /** Semantic expression observable, independent of FE storage details. */
  using ExpressionBinding = std::variant<ValueBinding, GradientBinding>;

  /** Bind a semantic value observable to an expression symbol. */
  inline ExpressionBinding
  value(const FieldId field, std::string symbol)
  {
    return ValueBinding{field, std::move(symbol)};
  }

  /** Bind a physical gradient component to an expression symbol. */
  inline ExpressionBinding
  gradient(const FieldId      field,
           std::string        symbol,
           const unsigned int component)
  {
    return GradientBinding{field, std::move(symbol), component};
  }

  namespace detail
  {
    inline dealii::UpdateFlags
    expression_update_flags(const ExpressionBinding &binding)
    {
      return std::visit(
        [](const auto &observable) {
          using Observable = std::decay_t<decltype(observable)>;
          if constexpr (std::is_same_v<Observable, GradientBinding>)
            return dealii::UpdateFlags(dealii::update_values |
                                       dealii::update_gradients);
          else
            return dealii::update_values;
        },
        binding);
    }

    inline dealii::UpdateFlags
    expression_update_flags(const std::vector<ExpressionBinding> &bindings)
    {
      auto result = dealii::update_values;
      for (const auto &binding : bindings)
        result |= expression_update_flags(binding);
      return result;
    }
  } // namespace detail

  /**
   * A symbolic pointwise transform of one or more retained observables.
   *
   * All bindings use one common retained sampling space. The normal factory
   * constructs that space from the supplied finite-element representation and
   * quadrature; the retained-plan overload is an advanced escape hatch for
   * tests and adapters.
   */
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

    using Binding = ExpressionBinding;

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
        std::visit(
          [&](const auto &observable) {
            using Observable = std::decay_t<decltype(observable)>;
            const bool is_gradient =
              std::is_same_v<Observable, GradientBinding>;
            unsigned int component = 0;
            if constexpr (is_gradient)
              {
                component = observable.component;
                AssertThrow(component < spacedim,
                            dealii::ExcMessage(
                              "A gradient binding has an invalid "
                              "component."));
                AssertThrow(
                  (sampling.update_flags() & dealii::update_gradients) != 0,
                  dealii::ExcMessage(
                    "A gradient binding requires a sampling plan with "
                    "update_gradients."));
              }
            bindings_.push_back({observable.field,
                                 observable.symbol,
                                 is_gradient,
                                 component,
                                 sampling});
            if (std::find(dependencies_.begin(),
                          dependencies_.end(),
                          observable.field) == dependencies_.end())
              dependencies_.push_back(observable.field);
          },
          binding);
    }

    ExpressionRepresentation(const FieldId            source,
                             const SamplingPlan      &sampling,
                             SymbolicExpressionKernel kernel)
      : ExpressionRepresentation(std::vector<Binding>{value(source, "A")},
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
      : ExpressionRepresentation(std::vector<Binding>{value(source, symbol)},
                                 sampling,
                                 expression,
                                 constants)
    {}

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
      const auto samples     = sample_bindings(context);
      const auto evaluations = evaluate_kernel(context, samples);
      value_type result;
      result.reinit(sampling_plan().locally_owned_points(),
                    sampling_plan().locally_relevant_points(),
                    sampling_plan().mpi_communicator());
      for (std::size_t q = 0; q < sampling_plan().points().size(); ++q)
        result[sampling_plan().point_index(q)] = evaluations[q].value;
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
      const auto samples     = sample_bindings(context);
      const auto evaluations = evaluate_kernel(context, samples);
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
                (*diagonal)[sampling_plan().point_index(q)] =
                  evaluations[q].derivatives[binding_index];
              }

            const auto &state = context.state(field);
            auto        term =
              make_diagonal_operator(bindings_[binding_index].sampling,
                                     diagonal) *
              sampling_operator_for(bindings_[binding_index], state);
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
      bool         gradient;
      unsigned int component;
      SamplingPlan sampling;
    };

    static std::vector<std::string>
    symbols(const std::vector<Binding> &bindings)
    {
      std::vector<std::string> result;
      result.reserve(bindings.size());
      for (const auto &binding : bindings)
        std::visit(
          [&result](const auto &observable) {
            result.push_back(observable.symbol);
          },
          binding);
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
            sampling_operator_for(binding, context.state(binding.field));
          value_type samples;
          sampling_operator.reinit_range_vector(samples, false);
          sampling_operator.vmult(samples, context.state(binding.field));
          result.push_back(std::move(samples));
        }
      return result;
    }

    std::vector<SymbolicExpressionKernel::Evaluation>
    evaluate_kernel(const EvaluationContext<state_type> &context,
                    const std::vector<value_type>       &samples) const
    {
      std::vector<SymbolicExpressionKernel::Evaluation> result;
      result.reserve(sampling_plan().points().size());
      for (std::size_t q = 0; q < sampling_plan().points().size(); ++q)
        {
          std::vector<double> values;
          values.reserve(samples.size());
          for (const auto &sample : samples)
            values.push_back(sample[sampling_plan().point_index(q)]);
          result.push_back(kernel_->evaluate(sampling_plan().points()[q].point,
                                             context.time(),
                                             values));
        }
      return result;
    }

    Operator
    sampling_operator_for(const InstalledBinding &binding,
                          const state_type       &state) const
    {
      if (binding.gradient)
        return binding.sampling.gradient_linearize(state, binding.component);
      return binding.sampling.linearize(state);
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

  /**
   * Construct a value-only expression and infer its FE requirements.
   *
   * This is the normal user-facing path. All bindings are sampled on the
   * supplied finite-element representation and quadrature, so the expression
   * has one common sampling point space by construction.
   */
  template <int dim, int spacedim, typename ValueType, typename Extractor>
  auto
  make_expression_representation(
    const FiniteElementRepresentation<dim, spacedim, ValueType, Extractor>
                                         &representation,
    const dealii::Quadrature<dim>        &quadrature,
    const std::vector<ExpressionBinding> &bindings,
    const std::string                    &expression,
    const std::map<std::string, double>  &constants = {})
    -> ExpressionRepresentation<spacedim>
  {
    static_assert(std::is_same_v<ValueType, double>,
                  "Retained scalar expressions require a scalar FE view.");
    const auto sampling =
      make_retained_sampling_plan(representation,
                                  quadrature,
                                  detail::expression_update_flags(bindings));
    return ExpressionRepresentation<spacedim>(bindings,
                                              sampling,
                                              expression,
                                              constants);
  }

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
