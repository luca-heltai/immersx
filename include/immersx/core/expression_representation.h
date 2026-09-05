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

#include <immersx/core/lifting.h>
#include <immersx/core/representation.h>
#include <immersx/core/symbolic_expression_kernel.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ImmersX
{
  /** An active FE source whose coefficients come from one state field. */
  template <typename FERepresentation>
  struct StateField
  {
    FERepresentation representation;
    FieldId          field;

    FERepresentation
    current_representation() const
    {
      return representation;
    }
  };

  /** A frozen FE source whose coefficients are fixed for the solve. */
  template <typename FERepresentation>
  struct FrozenField
  {
    FERepresentation                              representation;
    std::shared_ptr<const ImmersXLA::MPI::Vector> coefficients;
    /** Optional factory for representations whose storage may be refined. */
    std::function<FERepresentation()> representation_factory;

    FERepresentation
    current_representation() const
    {
      return representation_factory ? representation_factory() : representation;
    }
  };

  template <typename FERepresentation>
  using FEFieldSource =
    std::variant<StateField<FERepresentation>, FrozenField<FERepresentation>>;

  /** A value observable bound to one FE source and expression symbol. */
  template <typename FERepresentation>
  struct ValueBinding
  {
    FEFieldSource<FERepresentation> source;
    std::string                     symbol;
  };

  /** A physical gradient component bound to one FE source and symbol. */
  template <typename FERepresentation>
  struct GradientBinding
  {
    FEFieldSource<FERepresentation> source;
    std::string                     symbol;
    unsigned int                    component;
  };

  template <typename FERepresentation>
  using ExpressionBinding = std::variant<ValueBinding<FERepresentation>,
                                         GradientBinding<FERepresentation>>;

  /** Bind a finite-element view to an active state field. */
  template <typename FERepresentation>
  StateField<std::decay_t<FERepresentation>>
  state_field(const FERepresentation &representation, const FieldId field)
  {
    return {representation, field};
  }

  /** Bind a finite-element view to explicitly supplied frozen coefficients. */
  template <typename FERepresentation>
  FrozenField<std::decay_t<FERepresentation>>
  frozen_field(const FERepresentation                       &representation,
               std::shared_ptr<const ImmersXLA::MPI::Vector> coefficients)
  {
    AssertThrow(coefficients != nullptr,
                dealii::ExcMessage("Frozen FE coefficients must not be null."));
    return {representation, std::move(coefficients), {}};
  }

  /** Bind an imported field view, retaining its shared coefficient storage. */
  template <typename ImportedFieldView>
  auto
  frozen_field(const ImportedFieldView &field)
  {
    using FieldView = std::decay_t<ImportedFieldView>;
    using FieldType = std::decay_t<decltype(field.field())>;
    using FERepresentation =
      FiniteElementRepresentation<FieldType::dimension(),
                                  FieldType::spacedimension()>;
    auto       owner    = std::make_shared<const FieldView>(field);
    const auto fe_field = field.field();
    const auto representation =
      FERepresentation(field.triangulation(),
                       fe_field.dof_handler(),
                       fe_field.locally_owned_dofs(),
                       fe_field.locally_relevant_dofs(),
                       fe_field.constraints(),
                       fe_field.mapping(),
                       fe_field.extractor());
    return FrozenField<FERepresentation>{
      representation, field.coefficients_handle(), [owner]() {
        const auto current = owner->field();
        return FERepresentation(owner->triangulation(),
                                current.dof_handler(),
                                current.locally_owned_dofs(),
                                current.locally_relevant_dofs(),
                                current.constraints(),
                                current.mapping(),
                                current.extractor());
      }};
  }

  /** Bind a semantic value observable to an FE source and symbol. */
  template <typename Source>
  auto
  value(Source source, std::string symbol)
  {
    using FERepresentation =
      std::decay_t<decltype(source.current_representation())>;
    return ValueBinding<FERepresentation>{std::move(source), std::move(symbol)};
  }

  /** \cond legacy_expression_api */
  /** Bind a physical gradient component to an FE source and symbol. */
  template <typename Source>
  auto
  gradient(Source source, std::string symbol, const unsigned int component)
  {
    using FERepresentation =
      std::decay_t<decltype(source.current_representation())>;
    return GradientBinding<FERepresentation>{std::move(source),
                                             std::move(symbol),
                                             component};
  }
  /** \endcond */

  namespace detail
  {
    template <typename FERepresentation>
    inline dealii::UpdateFlags
    expression_update_flags(const ExpressionBinding<FERepresentation> &binding)
    {
      return std::visit(
        [](const auto &observable) {
          using Observable = std::decay_t<decltype(observable)>;
          if constexpr (std::is_same_v<Observable,
                                       GradientBinding<FERepresentation>>)
            return dealii::UpdateFlags(dealii::update_values |
                                       dealii::update_gradients);
          else
            return dealii::update_values;
        },
        binding);
    }

    template <typename FERepresentation>
    inline dealii::UpdateFlags
    expression_update_flags(
      const std::vector<ExpressionBinding<FERepresentation>> &bindings)
    {
      auto result = dealii::update_values;
      for (const auto &binding : bindings)
        result |= expression_update_flags(binding);
      return result;
    }

    template <typename FERepresentation>
    inline std::vector<FieldId>
    expression_dependencies(
      const std::vector<ExpressionBinding<FERepresentation>> &bindings)
    {
      std::vector<FieldId> result;
      for (const auto &binding : bindings)
        std::visit(
          [&result](const auto &observable) {
            std::visit(
              [&result](const auto &source) {
                using Source = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<Source,
                                             StateField<FERepresentation>>)
                  if (std::find(result.begin(), result.end(), source.field) ==
                      result.end())
                    result.push_back(source.field);
              },
              observable.source);
          },
          binding);
      return result;
    }

    template <typename FERepresentation>
    FERepresentation
    source_representation(const FEFieldSource<FERepresentation> &source)
    {
      return std::visit(
        [](const auto &field) -> FERepresentation {
          return field.current_representation();
        },
        source);
    }

    template <typename FERepresentation>
    inline bool
    same_representation(const FERepresentation &left,
                        const FERepresentation &right)
    {
      return &left.triangulation() == &right.triangulation() &&
             &left.dof_handler() == &right.dof_handler() &&
             &left.finite_element() == &right.finite_element() &&
             &left.mapping() == &right.mapping() &&
             &left.locally_owned_dofs() == &right.locally_owned_dofs() &&
             &left.locally_relevant_dofs() == &right.locally_relevant_dofs() &&
             &left.constraints() == &right.constraints() &&
             left.geometry_version() == right.geometry_version() &&
             left.extractor().get_name() == right.extractor().get_name();
    }

    template <typename FERepresentation>
    inline bool
    same_source(const FEFieldSource<FERepresentation> &left,
                const FEFieldSource<FERepresentation> &right)
    {
      return std::visit(
        [](const auto &left_field, const auto &right_field) {
          using Left  = std::decay_t<decltype(left_field)>;
          using Right = std::decay_t<decltype(right_field)>;
          if constexpr (!std::is_same_v<Left, Right>)
            return false;
          else if constexpr (std::is_same_v<Left, StateField<FERepresentation>>)
            return left_field.field == right_field.field &&
                   same_representation(left_field.current_representation(),
                                       right_field.current_representation());
          else
            return left_field.coefficients.get() ==
                     right_field.coefficients.get() &&
                   same_representation(left_field.current_representation(),
                                       right_field.current_representation());
        },
        left,
        right);
    }

    template <typename FERepresentation>
    inline std::vector<FEFieldSource<FERepresentation>>
    distinct_sources(
      const std::vector<ExpressionBinding<FERepresentation>> &bindings)
    {
      std::vector<FEFieldSource<FERepresentation>> result;
      for (const auto &binding : bindings)
        std::visit(
          [&result](const auto &observable) {
            const auto source = observable.source;
            const auto it =
              std::find_if(result.begin(),
                           result.end(),
                           [&source](const auto &candidate) {
                             return same_source(source, candidate);
                           });
            if (it == result.end())
              result.push_back(source);
          },
          binding);
      return result;
    }

    template <int spacedim,
              typename StateVectorType,
              typename QuantityVectorType>
    inline void
    assert_compatible_sampling(
      const RetainedSamplingPlan<spacedim, StateVectorType, QuantityVectorType>
        &candidate,
      const RetainedSamplingPlan<spacedim, StateVectorType, QuantityVectorType>
        &canonical)
    {
      AssertThrow(candidate.points().size() == canonical.points().size(),
                  dealii::ExcMessage(
                    "Expression sources must have the same number of "
                    "sampling points."));
      AssertThrow(candidate.locally_owned_points() ==
                      canonical.locally_owned_points() &&
                    candidate.locally_relevant_points() ==
                      canonical.locally_relevant_points(),
                  dealii::ExcMessage(
                    "Expression sources must share a sampling index space."));
      for (std::size_t q = 0; q < canonical.points().size(); ++q)
        AssertThrow(candidate.points()[q].point.distance(
                      canonical.points()[q].point) < 1e-12 &&
                      candidate.points()[q].source_entity_id ==
                        canonical.points()[q].source_entity_id &&
                      candidate.points()[q].representative_qpoint ==
                        canonical.points()[q].representative_qpoint,
                    dealii::ExcMessage(
                      "Expression sources must describe the same physical "
                      "sampling points."));
    }
  } // namespace detail

  /**
   * A symbolic pointwise transform of one or more retained observables.
   *
   * All bindings use one common retained sampling space. Internally, one
   * retained sampling plan is installed for each distinct FE source, with the
   * source's value/gradient requirements aggregated across its bindings.
   */
  template <int spacedim,
            typename StateVectorType    = ImmersXLA::MPI::Vector,
            typename QuantityVectorType = ImmersXLA::MPI::Vector,
            typename FERepresentation =
              FiniteElementRepresentation<1, spacedim>>
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

    using Binding = ExpressionBinding<FERepresentation>;
    using Source  = FEFieldSource<FERepresentation>;

    struct InstalledSource
    {
      Source       source;
      SamplingPlan sampling;
    };

    struct InstalledBinding
    {
      std::size_t  source_index;
      std::string  symbol;
      bool         gradient;
      unsigned int component;
    };

    ExpressionRepresentation(const std::vector<Binding>      &bindings,
                             const std::vector<Source>       &sources,
                             const std::vector<SamplingPlan> &samplings,
                             SymbolicExpressionKernel         kernel,
                             RepresentationDomain             domain =
                               RepresentationDomain(spacedim,
                                                    spacedim,
                                                    "retained-fe-sampling"))
      : domain_(std::move(domain))
      , kernel_(std::make_shared<SymbolicExpressionKernel>(std::move(kernel)))
    {
      AssertThrow(!bindings.empty(),
                  dealii::ExcMessage(
                    "An expression representation needs at least one "
                    "binding."));
      AssertThrow(kernel_->n_independent_symbols() == bindings.size(),
                  dealii::ExcMessage(
                    "The number of expression symbols must match the number "
                    "of field bindings."));

      AssertThrow(sources.size() == samplings.size(),
                  dealii::ExcMessage(
                    "Each expression source needs one sampling plan."));
      AssertThrow(!sources.empty(),
                  dealii::ExcMessage(
                    "An expression needs at least one sampling plan."));
      dependencies_ = detail::expression_dependencies(bindings);
      for (std::size_t i = 1; i < samplings.size(); ++i)
        detail::assert_compatible_sampling(samplings[i], samplings[0]);
      for (std::size_t source_index = 0; source_index < sources.size();
           ++source_index)
        {
          AssertThrow(
            std::none_of(sources.begin(),
                         sources.begin() + source_index,
                         [&sources, source_index](const auto &candidate) {
                           return detail::same_source(sources[source_index],
                                                      candidate);
                         }),
            dealii::ExcMessage("Expression sources must be distinct."));
          sources_.push_back({sources[source_index], samplings[source_index]});
        }
      bindings_.reserve(bindings.size());
      for (const auto &binding : bindings)
        {
          std::visit(
            [&](const auto &observable) {
              using Observable = std::decay_t<decltype(observable)>;
              const bool is_gradient =
                std::is_same_v<Observable, GradientBinding<FERepresentation>>;
              unsigned int component = 0;
              if constexpr (is_gradient)
                {
                  component = observable.component;
                  AssertThrow(component < spacedim,
                              dealii::ExcMessage(
                                "A gradient binding has an invalid "
                                "component."));
                  const auto source_index = [&] {
                    const auto it =
                      std::find_if(sources.begin(),
                                   sources.end(),
                                   [&observable](const auto &candidate) {
                                     return detail::same_source(
                                       observable.source, candidate);
                                   });
                    AssertThrow(it != sources.end(),
                                dealii::ExcMessage(
                                  "Expression binding references an unknown "
                                  "source."));
                    return static_cast<std::size_t>(it - sources.begin());
                  }();
                  AssertThrow(
                    (samplings[source_index].update_flags() &
                     dealii::update_gradients) != 0,
                    dealii::ExcMessage(
                      "A gradient binding requires a source plan with "
                      "update_gradients."));
                  bindings_.push_back(
                    {source_index, observable.symbol, is_gradient, component});
                }
              else
                {
                  const auto source_index = [&] {
                    const auto it =
                      std::find_if(sources.begin(),
                                   sources.end(),
                                   [&observable](const auto &candidate) {
                                     return detail::same_source(
                                       observable.source, candidate);
                                   });
                    AssertThrow(it != sources.end(),
                                dealii::ExcMessage(
                                  "Expression binding references an unknown "
                                  "source."));
                    return static_cast<std::size_t>(it - sources.begin());
                  }();
                  bindings_.push_back(
                    {source_index, observable.symbol, is_gradient, component});
                }
            },
            binding);
        }
    }

    ExpressionRepresentation(const std::vector<Binding> &bindings,
                             const SamplingPlan         &sampling,
                             SymbolicExpressionKernel    kernel,
                             RepresentationDomain        domain =
                               RepresentationDomain(spacedim,
                                                    spacedim,
                                                    "retained-fe-sampling"))
      : ExpressionRepresentation(
          bindings,
          detail::distinct_sources(bindings),
          std::vector<SamplingPlan>(detail::distinct_sources(bindings).size(),
                                    sampling),
          std::move(kernel),
          std::move(domain))
    {}

    ExpressionRepresentation(
      const std::vector<Binding>          &bindings,
      const std::vector<Source>           &sources,
      const std::vector<SamplingPlan>     &samplings,
      const std::string                   &expression,
      const std::map<std::string, double> &constants = {},
      const RepresentationDomain           domain =
        RepresentationDomain(spacedim, spacedim, "retained-fe-sampling"))
      : ExpressionRepresentation(
          bindings,
          sources,
          samplings,
          [&] {
            SymbolicExpressionKernel kernel;
            kernel.initialize(expression, symbols(bindings), constants);
            return kernel;
          }(),
          domain)
    {}

    ExpressionRepresentation(
      const std::vector<Binding>          &bindings,
      const SamplingPlan                  &sampling,
      const std::string                   &expression,
      const std::map<std::string, double> &constants = {},
      const RepresentationDomain           domain =
        RepresentationDomain(spacedim, spacedim, "retained-fe-sampling"))
      : ExpressionRepresentation(bindings,
                                 sampling,
                                 make_kernel(expression,
                                             symbols(bindings),
                                             constants),
                                 domain)
    {}

    ExpressionRepresentation(
      const std::vector<Binding>          &bindings,
      const std::vector<SamplingPlan>     &samplings,
      const std::string                   &expression,
      const std::map<std::string, double> &constants = {},
      const RepresentationDomain           domain =
        RepresentationDomain(spacedim, spacedim, "retained-fe-sampling"))
      : ExpressionRepresentation(bindings,
                                 detail::distinct_sources(bindings),
                                 samplings,
                                 make_kernel(expression,
                                             symbols(bindings),
                                             constants),
                                 domain)
    {}

    const std::vector<FieldId> &
    dependencies() const
    {
      return dependencies_;
    }

    const SamplingPlan &
    sampling_plan() const
    {
      return sources_.front().sampling;
    }

    /** Return the retained plan for one installed FE source. */
    const SamplingPlan &
    sampling_plan_for_source(const std::size_t source) const
    {
      AssertIndexRange(source, sources_.size());
      return sources_[source].sampling;
    }

    const SamplingPlan &
    sampling_plan(const std::size_t binding) const
    {
      AssertIndexRange(binding, bindings_.size());
      return sources_[bindings_[binding].source_index].sampling;
    }

    /** Number of retained plans, one for each distinct FE source. */
    std::size_t
    n_sampling_sources() const
    {
      return sources_.size();
    }

    const SymbolicExpressionKernel &
    kernel() const
    {
      return *kernel_;
    }

    quantity_space_type
    quantity_space() const
    {
      return quantity_space_type(domain_);
    }

    const RepresentationDomain &
    domain() const
    {
      return domain_;
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
        if (std::holds_alternative<StateField<FERepresentation>>(
              sources_[bindings_[binding_index].source_index].source) &&
            std::get<StateField<FERepresentation>>(
              sources_[bindings_[binding_index].source_index].source)
                .field == field)
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
              make_diagonal_operator(
                sources_[bindings_[binding_index].source_index].sampling,
                diagonal) *
              sampling_operator_for(
                bindings_[binding_index],
                sources_[bindings_[binding_index].source_index].sampling,
                state);
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
          const auto source_vector = std::visit(
            [&context](const auto &source) -> const state_type & {
              using SourceType = std::decay_t<decltype(source)>;
              if constexpr (std::is_same_v<SourceType,
                                           StateField<FERepresentation>>)
                return context.state(source.field);
              else
                return *source.coefficients;
            },
            sources_[binding.source_index].source);
          const auto sampling_operator =
            sampling_operator_for(binding,
                                  sources_[binding.source_index].sampling,
                                  source_vector);
          value_type samples;
          sampling_operator.reinit_range_vector(samples, false);
          sampling_operator.vmult(samples, source_vector);
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
          for (std::size_t binding_index = 0; binding_index < samples.size();
               ++binding_index)
            values.push_back(
              samples[binding_index]
                     [sources_[bindings_[binding_index].source_index]
                        .sampling.point_index(q)]);
          result.push_back(kernel_->evaluate(sampling_plan().points()[q].point,
                                             context.time(),
                                             values));
        }
      return result;
    }

    Operator
    sampling_operator_for(const InstalledBinding &binding,
                          const SamplingPlan     &sampling,
                          const state_type       &state) const
    {
      if (binding.gradient)
        return sampling.gradient_linearize(state, binding.component);
      return sampling.linearize(state);
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

    std::vector<InstalledSource>              sources_;
    std::vector<InstalledBinding>             bindings_;
    std::vector<FieldId>                      dependencies_;
    RepresentationDomain                      domain_;
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
    const dealii::Quadrature<dim> &quadrature,
    const std::vector<ExpressionBinding<
      FiniteElementRepresentation<dim, spacedim, ValueType, Extractor>>>
                                        &bindings,
    const std::string                   &expression,
    const std::map<std::string, double> &constants = {})
    -> ExpressionRepresentation<
      spacedim,
      ImmersXLA::MPI::Vector,
      ImmersXLA::MPI::Vector,
      FiniteElementRepresentation<dim, spacedim, ValueType, Extractor>>
  {
    (void)representation;
    static_assert(std::is_same_v<ValueType, double>,
                  "Retained scalar expressions require a scalar FE view.");
    using FERepresentation =
      FiniteElementRepresentation<dim, spacedim, ValueType, Extractor>;
    using Expression = ExpressionRepresentation<spacedim,
                                                ImmersXLA::MPI::Vector,
                                                ImmersXLA::MPI::Vector,
                                                FERepresentation>;
    const auto canonical_sampling =
      make_retained_sampling_plan(representation, quadrature);
    const auto sources = detail::distinct_sources(bindings);
    std::vector<dealii::UpdateFlags> source_flags(sources.size(),
                                                  dealii::update_values);
    for (const auto &binding : bindings)
      {
        const auto source =
          std::visit([](const auto &observable) { return observable.source; },
                     binding);
        const auto source_index =
          std::find_if(sources.begin(),
                       sources.end(),
                       [&source](const auto &candidate) {
                         return detail::same_source(source, candidate);
                       });
        AssertThrow(source_index != sources.end(),
                    dealii::ExcMessage(
                      "Expression binding references an unknown source."));
        source_flags[source_index - sources.begin()] |=
          detail::expression_update_flags(binding);
      }
    std::vector<typename Expression::SamplingPlan> samplings;
    samplings.reserve(sources.size());
    for (std::size_t source_index = 0; source_index < sources.size();
         ++source_index)
      {
        const auto sampling =
          make_retained_sampling_plan(detail::source_representation(
                                        sources[source_index]),
                                      quadrature,
                                      source_flags[source_index]);
        detail::assert_compatible_sampling(sampling, canonical_sampling);
        samplings.push_back(sampling);
      }
    return Expression(bindings,
                      sources,
                      samplings,
                      expression,
                      constants,
                      RepresentationDomain(dim,
                                           spacedim,
                                           "retained-fe-sampling"));
  }

  /**
   * A scalar finite-element expression before its sampling geometry is known.
   *
   * This is deliberately a description rather than a Representation: it owns
   * no quadrature, retained sampling plan, sampled values, or coupling data.
   * The finite-element view is a small non-owning value type, so copying this
   * description preserves the same Problem-owned lifetime requirements as the
   * view itself.
   */
  template <typename FERepresentation>
  class FiniteElementExpression
  {
  public:
    using SourceRepresentation = FERepresentation;
    using Binding              = ExpressionBinding<FERepresentation>;

    static constexpr unsigned int support_dimension =
      FERepresentation::support_dimension;
    static constexpr unsigned int ambient_dimension =
      FERepresentation::ambient_dimension;

    FiniteElementExpression(const FERepresentation       &source,
                            std::vector<Binding>          bindings,
                            std::string                   expression,
                            std::map<std::string, double> constants = {})
      : source_(source)
      , bindings_(std::move(bindings))
      , expression_(std::move(expression))
      , constants_(std::move(constants))
    {
      AssertThrow(!bindings_.empty(),
                  dealii::ExcMessage(
                    "A finite-element expression needs at least one "
                    "binding."));
      dependencies_ = detail::expression_dependencies(bindings_);
    }

    template <typename Geometry>
    decltype(auto)
    lift(const Geometry &geometry) const
    {
      return detail::invoke_lift(*this, geometry, 0);
    }

    const FERepresentation &
    source_representation() const
    {
      return source_;
    }

    const std::vector<Binding> &
    bindings() const
    {
      return bindings_;
    }

    const std::string &
    expression() const
    {
      return expression_;
    }

    const std::map<std::string, double> &
    constants() const
    {
      return constants_;
    }

    const std::vector<FieldId> &
    dependencies() const
    {
      return dependencies_;
    }

    RepresentationDomain
    domain() const
    {
      return RepresentationDomain(support_dimension,
                                  ambient_dimension,
                                  "finite-element-expression");
    }

  private:
    FERepresentation              source_;
    std::vector<Binding>          bindings_;
    std::string                   expression_;
    std::map<std::string, double> constants_;
    std::vector<FieldId>          dependencies_;
  };

  /** Describe a scalar expression without selecting its sampling quadrature. */
  template <typename FERepresentation>
  auto
  make_fe_expression(const FERepresentation                          &source,
                     std::vector<ExpressionBinding<FERepresentation>> bindings,
                     std::string                   expression,
                     std::map<std::string, double> constants = {})
    -> FiniteElementExpression<FERepresentation>
  {
    return FiniteElementExpression<FERepresentation>(source,
                                                     std::move(bindings),
                                                     std::move(expression),
                                                     std::move(constants));
  }

  /** Lift an already sampled representation through retained lift points. */
  template <typename SourceRepresentation, typename GeometryLiftRepresentation>
  class TensorProductExpressionLiftRepresentation
  {
  public:
    static constexpr unsigned int support_dimension =
      GeometryLiftRepresentation::support_dimension;
    static constexpr unsigned int ambient_dimension =
      GeometryLiftRepresentation::ambient_dimension;
    static constexpr unsigned int representative_dimension =
      GeometryLiftRepresentation::representative_dimension;

    using value_type          = typename SourceRepresentation::value_type;
    using state_type          = typename SourceRepresentation::state_type;
    using quantity_space_type = QuantitySpace<value_type>;
    using Operator            = RepresentationOperator<value_type, state_type>;
    using ValueOperator       = RepresentationOperator<value_type, value_type>;
    using QuadraturePoint =
      typename GeometryLiftRepresentation::QuadraturePoint;

    TensorProductExpressionLiftRepresentation(
      SourceRepresentation       source,
      GeometryLiftRepresentation geometry)
      : source_(std::move(source))
      , geometry_(std::move(geometry))
    {
      AssertThrow(source_.sampling_plan().points().size() > 0,
                  dealii::ExcMessage(
                    "An expression lift requires representative points."));
      for (const auto &point : geometry_.lifted_points())
        AssertThrow(point.mode_values.size() == 1,
                    dealii::ExcMessage(
                      "Expression lifting currently supports the physical "
                      "mode-zero tensor-product lift."));

      const auto [offset, size] = dealii::Utilities::MPI::partial_and_total_sum(
        geometry_.lifted_points().size(), geometry_.mpi_communicator());
      lifted_owned_.set_size(size);
      lifted_owned_.add_range(offset,
                              offset + geometry_.lifted_points().size());
      lifted_owned_.compress();
      lifted_relevant_ = lifted_owned_;
      lifted_indices_.reserve(geometry_.lifted_points().size());
      for (std::size_t q = 0; q < geometry_.lifted_points().size(); ++q)
        lifted_indices_.push_back(offset + q);
    }

    const SourceRepresentation &
    source_representation() const
    {
      return source_;
    }

    const GeometryLiftRepresentation &
    geometry_representation() const
    {
      return geometry_;
    }

    const auto &
    lifted_points() const
    {
      return geometry_.lifted_points();
    }

    const dealii::IndexSet &
    locally_owned_points() const
    {
      return lifted_owned_;
    }

    const dealii::IndexSet &
    locally_relevant_points() const
    {
      return lifted_relevant_;
    }

    dealii::types::global_dof_index
    point_index(const std::size_t local_point) const
    {
      AssertIndexRange(local_point, lifted_indices_.size());
      return lifted_indices_[local_point];
    }

    const RepresentationDomain &
    domain() const
    {
      return geometry_.domain();
    }

    MPI_Comm
    mpi_communicator() const
    {
      return geometry_.mpi_communicator();
    }

    std::vector<QuadraturePoint>
    locally_owned_quadrature_points(
      const dealii::Quadrature<support_dimension> &quadrature) const
    {
      return geometry_.locally_owned_quadrature_points(quadrature);
    }

    const std::vector<FieldId> &
    dependencies() const
    {
      return source_.dependencies();
    }

    quantity_space_type
    quantity_space() const
    {
      return quantity_space_type(geometry_.domain());
    }

    value_type
    evaluate(const EvaluationContext<state_type> &context,
             const EvaluationRequest             &request = {}) const
    {
      return apply_values(source_.evaluate(context, request));
    }

    Operator
    linearize(const EvaluationContext<state_type> &context,
              const EvaluationRequest             &request = {}) const
    {
      return make_lift_operator() * source_.linearize(context, request);
    }

    Operator
    linearize(const EvaluationContext<state_type> &context,
              const FieldId                        field,
              const EvaluationRequest             &request = {}) const
    {
      return make_lift_operator() * source_.linearize(context, field, request);
    }

  private:
    value_type
    apply_values(const value_type &representative) const
    {
      value_type result;
      result.reinit(lifted_owned_,
                    lifted_relevant_,
                    geometry_.mpi_communicator());
      const auto source_indices = representative_indices();
      detail::apply_tensor_product_lift(geometry_.lifted_points(),
                                        lifted_indices_,
                                        source_indices,
                                        representative,
                                        result,
                                        false,
                                        false,
                                        lifted_owned_,
                                        lifted_relevant_,
                                        geometry_.mpi_communicator());
      return result;
    }

    ValueOperator
    make_lift_operator() const
    {
      const auto lifted_points   = geometry_.lifted_points();
      const auto lifted_indices  = lifted_indices_;
      const auto lifted_owned    = lifted_owned_;
      const auto lifted_relevant = lifted_relevant_;
      const auto source_plan     = source_.sampling_plan();
      const auto source_indices  = representative_indices();
      const auto source_owned    = source_plan.locally_owned_points();
      const auto source_relevant = source_plan.locally_relevant_points();
      const auto communicator    = geometry_.mpi_communicator();

      ValueOperator result;
      result.reinit_range_vector =
        [lifted_owned, lifted_relevant, communicator](value_type &vector,
                                                      const bool  omit) {
          vector.reinit(lifted_owned, lifted_relevant, communicator);
          if (!omit)
            vector = 0.;
        };
      result.reinit_domain_vector = [source_plan](value_type &vector,
                                                  const bool  omit) {
        vector.reinit(source_plan.locally_owned_points(),
                      source_plan.locally_relevant_points(),
                      source_plan.mpi_communicator());
        if (!omit)
          vector = 0.;
      };
      result.vmult = [lifted_points,
                      lifted_indices,
                      source_indices,
                      lifted_owned,
                      lifted_relevant,
                      communicator](value_type       &destination,
                                    const value_type &source) {
        detail::apply_tensor_product_lift(lifted_points,
                                          lifted_indices,
                                          source_indices,
                                          source,
                                          destination,
                                          false,
                                          false,
                                          lifted_owned,
                                          lifted_relevant,
                                          communicator);
      };
      result.vmult_add = [lifted_points,
                          lifted_indices,
                          source_indices,
                          lifted_owned,
                          lifted_relevant,
                          communicator](value_type       &destination,
                                        const value_type &source) {
        detail::apply_tensor_product_lift(lifted_points,
                                          lifted_indices,
                                          source_indices,
                                          source,
                                          destination,
                                          false,
                                          true,
                                          lifted_owned,
                                          lifted_relevant,
                                          communicator);
      };
      result.Tvmult = [lifted_points,
                       lifted_indices,
                       source_indices,
                       source_owned,
                       source_relevant,
                       communicator](value_type       &destination,
                                     const value_type &source) {
        detail::apply_tensor_product_lift(lifted_points,
                                          lifted_indices,
                                          source_indices,
                                          source,
                                          destination,
                                          true,
                                          false,
                                          source_owned,
                                          source_relevant,
                                          communicator);
      };
      result.Tvmult_add = [lifted_points,
                           lifted_indices,
                           source_indices,
                           source_owned,
                           source_relevant,
                           communicator](value_type       &destination,
                                         const value_type &source) {
        detail::apply_tensor_product_lift(lifted_points,
                                          lifted_indices,
                                          source_indices,
                                          source,
                                          destination,
                                          true,
                                          true,
                                          source_owned,
                                          source_relevant,
                                          communicator);
      };
      return result;
    }

    std::vector<std::vector<dealii::types::global_dof_index>>
    representative_indices() const
    {
      const auto &lifted_points = geometry_.lifted_points();
      std::vector<std::vector<dealii::types::global_dof_index>> result(
        lifted_points.size());
      for (std::size_t q = 0; q < lifted_points.size(); ++q)
        result[q].push_back(source_.sampling_plan().point_index(
          lifted_points[q].source_representative_qpoint));
      return result;
    }

    SourceRepresentation                         source_;
    GeometryLiftRepresentation                   geometry_;
    dealii::IndexSet                             lifted_owned_;
    dealii::IndexSet                             lifted_relevant_;
    std::vector<dealii::types::global_dof_index> lifted_indices_;
  };

  /** Compose a sampled expression with an existing tensor-product lift. */
  template <typename SourceRepresentation, typename GeometryLiftRepresentation>
  auto
  make_tensor_product_expression_lift(
    const SourceRepresentation       &source,
    const GeometryLiftRepresentation &geometry)
    -> TensorProductExpressionLiftRepresentation<SourceRepresentation,
                                                 GeometryLiftRepresentation>
  {
    return TensorProductExpressionLiftRepresentation<
      SourceRepresentation,
      GeometryLiftRepresentation>(source, geometry);
  }

  /** Concretize a deferred FE expression on the supplied quadrature. */
  template <typename FERepresentation>
  auto
  sample(
    const FiniteElementExpression<FERepresentation>               &expression,
    const dealii::Quadrature<FERepresentation::support_dimension> &quadrature)
    -> ExpressionRepresentation<FERepresentation::ambient_dimension,
                                ImmersXLA::MPI::Vector,
                                ImmersXLA::MPI::Vector,
                                FERepresentation>
  {
    return make_expression_representation(expression.source_representation(),
                                          quadrature,
                                          expression.bindings(),
                                          expression.expression(),
                                          expression.constants());
  }

  /**
   * Supply the representative quadrature when a deferred expression is
   * consumed by a tensor-product lift, then reuse the sampled lift adapter.
   */
  template <typename FERepresentation,
            int reduced_dim,
            int surface_dim,
            int spacedim,
            int n_components>
  auto
  make_lift(
    const FiniteElementExpression<FERepresentation> &expression,
    const TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>
      &lift)
  {
    static_assert(FERepresentation::support_dimension == reduced_dim,
                  "A tensor-product lift must use the FE source dimension.");
    static_assert(FERepresentation::ambient_dimension == spacedim,
                  "A tensor-product lift must use the FE source space.");

    auto geometry = make_lift(expression.source_representation(), lift);
    auto sampled =
      sample(expression, geometry.support().representative_quadrature());
    return make_tensor_product_expression_lift(sampled, geometry);
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

} // namespace ImmersX

#endif // immersx_expression_representation_h
