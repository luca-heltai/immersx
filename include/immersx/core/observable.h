// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception (the "License"); either version 3.0 of the
// License, or (at your option) any later version. The full text of the
// License is available in the LICENSE.md file at the top level of the
// ImmersX distribution.
//
// ---------------------------------------------------------------------

#ifndef immersx_observable_h
#define immersx_observable_h

#include <deal.II/base/exceptions.h>

#include <deal.II/fe/fe_update_flags.h>

#include <immersx/core/fe_space.h>

#include <algorithm>
#include <any>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace ImmersX
{
  template <typename SourceFieldType, typename Operation>
  class Observable;

  namespace detail
  {
    /** The identity operation on a deal.II FEValues view. */
    struct ValueOperation
    {
      static constexpr dealii::UpdateFlags update_flags = dealii::update_values;

      template <typename View>
      decltype(auto)
      operator()(const View        &view,
                 const unsigned int i,
                 const unsigned int q) const
      {
        return view.value(i, q);
      }
    };

    /** The first derivative operation supplied by a deal.II FEValues view. */
    struct GradientOperation
    {
      static constexpr dealii::UpdateFlags update_flags =
        dealii::update_gradients;

      template <typename View>
      decltype(auto)
      operator()(const View        &view,
                 const unsigned int i,
                 const unsigned int q) const
      {
        return view.gradient(i, q);
      }
    };

    /** The divergence operation supplied by a deal.II FEValues view. */
    struct DivergenceOperation
    {
      static constexpr dealii::UpdateFlags update_flags =
        dealii::update_gradients;

      template <typename View>
      decltype(auto)
      operator()(const View        &view,
                 const unsigned int i,
                 const unsigned int q) const
      {
        return view.divergence(i, q);
      }
    };

    /** The symmetric-gradient operation supplied by a deal.II FEValues view.
     */
    struct SymmetricGradientOperation
    {
      static constexpr dealii::UpdateFlags update_flags =
        dealii::update_gradients;

      template <typename View>
      decltype(auto)
      operator()(const View        &view,
                 const unsigned int i,
                 const unsigned int q) const
      {
        return view.symmetric_gradient(i, q);
      }
    };

    /** The curl operation supplied by a deal.II FEValues view. */
    struct CurlOperation
    {
      static constexpr dealii::UpdateFlags update_flags =
        dealii::update_gradients;

      template <typename View>
      decltype(auto)
      operator()(const View        &view,
                 const unsigned int i,
                 const unsigned int q) const
      {
        return view.curl(i, q);
      }
    };

    template <typename Type>
    struct is_field : std::false_type
    {};

    template <int dim, int spacedim, typename Extractor>
    struct is_field<Field<dim, spacedim, Extractor>> : std::true_type
    {};

    template <typename FieldType, typename Operation, typename = void>
    struct is_supported_operation : std::false_type
    {};

    template <typename FieldType, typename Operation>
    struct is_supported_operation<
      FieldType,
      Operation,
      std::void_t<decltype(std::declval<Operation>()(
        std::declval<const typename FieldType::view_type &>(),
        0u,
        0u))>> : std::true_type
    {};

    template <typename Kernel, int spacedim, typename = void>
    struct has_point_scalar_kernel_evaluation : std::false_type
    {};

    template <typename Kernel, int spacedim>
    struct has_point_scalar_kernel_evaluation<
      Kernel,
      spacedim,
      std::void_t<decltype(std::declval<const Kernel &>().evaluate(
        std::declval<const dealii::Point<spacedim> &>(),
        0.,
        std::declval<const std::vector<double> &>()))>> : std::true_type
    {};

    template <typename Kernel, typename = void>
    struct has_scalar_kernel_evaluation : std::false_type
    {};

    template <typename Kernel>
    struct has_scalar_kernel_evaluation<
      Kernel,
      std::void_t<decltype(std::declval<const Kernel &>().evaluate(
        std::declval<const std::vector<double> &>()))>> : std::true_type
    {};

    template <typename Kernel, typename... InputTypes>
    struct has_typed_kernel_evaluation
    {
    private:
      template <typename Type>
      static auto
      test(int) -> decltype(std::declval<const Type &>().evaluate(
                              std::declval<const InputTypes &>()...),
                            std::true_type{});

      template <typename>
      static auto
      test(...) -> std::false_type;

    public:
      static constexpr bool value = decltype(test<Kernel>(0))::value;
    };

    template <typename Type, typename = void>
    struct has_value_member : std::false_type
    {};

    template <typename Type>
    struct has_value_member<
      Type,
      std::void_t<decltype(std::declval<const Type &>().value)>>
      : std::true_type
    {};

    template <typename Type>
    decltype(auto)
    kernel_value(const Type &result)
    {
      if constexpr (has_value_member<Type>::value)
        return result.value;
      else
        return result;
    }

    template <typename Kernel, int spacedim>
    auto
    evaluate_scalar_kernel(const Kernel                  &kernel,
                           const dealii::Point<spacedim> &point,
                           const double                   time,
                           const std::vector<double>     &values)
    {
      if constexpr (has_point_scalar_kernel_evaluation<Kernel, spacedim>::value)
        return kernel.evaluate(point, time, values);
      else
        {
          static_assert(has_scalar_kernel_evaluation<Kernel>::value,
                        "A scalar kernel must provide evaluate(values) or "
                        "evaluate(point, time, values).");
          return kernel.evaluate(values);
        }
    }

    template <typename Type>
    Type
    zero_like()
    {
      return Type{};
    }

    template <typename Type>
    struct is_transformed_observable : std::false_type
    {};

    template <typename Type>
    struct is_linear_observable : std::false_type
    {};

    template <typename Type>
    struct is_lifted_observable : std::false_type
    {};
  } // namespace detail

  /** A typed FE expression: a Field, a deal.II view operation, and metadata.
   *
   * The expression's value type is the result of invoking its operation on
   * the source Field's deal.II FEValues view. This keeps FE tensor typing and
   * operation support in deal.II rather than maintaining a parallel algebra in
   * ImmersX.
   */
  template <typename SourceFieldType, typename Operation>
  class Observable
  {
  public:
    using source_field_type = SourceFieldType;
    using operation_type    = Operation;
    using view_type         = typename SourceFieldType::view_type;
    using value_type        = std::decay_t<
      decltype(std::declval<Operation>()(std::declval<const view_type &>(),
                                         0u,
                                         0u))>;

    explicit Observable(const SourceFieldType &source)
      : source_(source)
      , dependencies_{source.field_id()}
    {}

    template <typename VectorType>
    Observable(const SourceFieldType &source, const VectorType &frozen_values)
      : source_(source)
      , frozen_(frozen_values)
    {}

    template <typename OtherOperation>
    Observable(const Observable<SourceFieldType, OtherOperation> &other)
      : source_(other.source_)
      , dependencies_(other.dependencies_)
      , scale_(other.scale_)
      , frozen_(other.frozen_)
    {}

    const std::vector<FieldId> &
    dependencies() const
    {
      return dependencies_;
    }

    static constexpr unsigned int
    dimension()
    {
      return SourceFieldType::dimension();
    }

    static constexpr unsigned int
    space_dimension()
    {
      return dimension();
    }

    static constexpr unsigned int
    spacedimension()
    {
      return SourceFieldType::spacedimension();
    }

    static constexpr unsigned int
    spacedim()
    {
      return spacedimension();
    }

    FieldId
    source_field() const
    {
      return dependencies_.empty() ? FieldId() : dependencies_.front();
    }

    const SourceFieldType &
    source() const
    {
      return source_;
    }

    const SourceFieldType &
    source_for(const FieldId field) const
    {
      AssertThrow(source_field() == field,
                  dealii::ExcMessage(
                    "The requested field is not an Observable dependency."));
      return source_;
    }

    Observable
    with_id(const FieldId id) const
    {
      Observable result = *this;
      result.source_    = source_.with_id(id);
      return result;
    }

    Operation
    operation() const
    {
      return {};
    }

    static constexpr dealii::UpdateFlags
    update_flags()
    {
      return Operation::update_flags;
    }

    bool
    is_frozen() const
    {
      return frozen_.has_value();
    }

    template <typename VectorType>
    const VectorType &
    frozen_values() const
    {
      const auto *values = std::any_cast<VectorType>(&frozen_);
      AssertThrow(values != nullptr,
                  dealii::ExcMessage(
                    "The frozen observable does not contain the requested "
                    "vector type."));
      return *values;
    }

    double
    scale() const
    {
      return scale_;
    }

    static constexpr bool
    is_linear()
    {
      return true;
    }

    template <typename Evaluator>
    value_type
    evaluate_point(const dealii::Point<spacedim()> &point,
                   const double                     time,
                   Evaluator                      &&evaluator) const
    {
      return scale_ * evaluator(*this, point, time);
    }

    template <typename DerivativeEvaluator>
    value_type
    linearize_point(const FieldId                    field,
                    const dealii::Point<spacedim()> &point,
                    const double                     time,
                    DerivativeEvaluator            &&derivative_evaluator) const
    {
      if (is_frozen() || source_field() != field)
        return detail::zero_like<value_type>();
      return scale_ * derivative_evaluator(*this, field, point, time);
    }

    Observable
    scaled(const double coefficient) const
    {
      Observable result = *this;
      result.scale_ *= coefficient;
      return result;
    }

  private:
    template <typename, typename>
    friend class Observable;

    SourceFieldType      source_;
    std::vector<FieldId> dependencies_;
    double               scale_ = 1.;
    std::any             frozen_;
  };

  namespace detail
  {
    template <typename Type>
    struct is_observable : std::false_type
    {};

    template <typename SourceFieldType, typename Operation>
    struct is_observable<Observable<SourceFieldType, Operation>>
      : std::true_type
    {};

    template <typename SourceFieldType, typename Operation>
    struct is_linear_observable<Observable<SourceFieldType, Operation>>
      : std::true_type
    {};
  } // namespace detail

  /**
   * A state-independent FE expression representing a residual test row.
   *
   * Test expressions deliberately expose the same native deal.II operation
   * and source field as an Observable, but have no state dependencies.  This
   * keeps the trial/state side and the residual/test side distinct in the
   * weak-term API while allowing the same FEValuesViews operations on both.
   */
  template <typename SourceFieldType, typename Operation>
  class TestExpression
  {
  public:
    using source_field_type = SourceFieldType;
    using operation_type    = Operation;
    using view_type         = typename SourceFieldType::view_type;
    using value_type        = std::decay_t<
      decltype(std::declval<Operation>()(std::declval<const view_type &>(),
                                         0u,
                                         0u))>;

    explicit TestExpression(const SourceFieldType &source)
      : source_(source)
    {}

    template <typename OtherOperation>
    TestExpression(const TestExpression<SourceFieldType, OtherOperation> &other)
      : source_(other.source_)
      , scale_(other.scale_)
    {}

    TestExpression(const TestExpression &) = default;
    TestExpression(TestExpression &&)      = default;
    TestExpression &
    operator=(const TestExpression &) = default;
    TestExpression &
    operator=(TestExpression &&) = default;

    static constexpr unsigned int
    dimension()
    {
      return SourceFieldType::dimension();
    }

    static constexpr unsigned int
    space_dimension()
    {
      return dimension();
    }

    static constexpr unsigned int
    spacedimension()
    {
      return SourceFieldType::spacedimension();
    }

    static constexpr unsigned int
    spacedim()
    {
      return spacedimension();
    }

    const SourceFieldType &
    source() const
    {
      return source_;
    }

    const SourceFieldType &
    source_for(const FieldId field) const
    {
      AssertThrow(source_.field_id() == field,
                  dealii::ExcMessage(
                    "The requested field is not a test expression source."));
      return source_;
    }

    FieldId
    source_field() const
    {
      return source_.field_id();
    }

    const std::vector<FieldId> &
    dependencies() const
    {
      static const std::vector<FieldId> none;
      return none;
    }

    Operation
    operation() const
    {
      return {};
    }

    static constexpr dealii::UpdateFlags
    update_flags()
    {
      return Operation::update_flags;
    }

    static constexpr bool
    is_linear()
    {
      return true;
    }

    bool
    is_frozen() const
    {
      return false;
    }

    double
    scale() const
    {
      return scale_;
    }

    TestExpression
    scaled(const double coefficient) const
    {
      auto result = *this;
      result.scale_ *= coefficient;
      return result;
    }

    TestExpression
    with_id(const FieldId id) const
    {
      auto result    = *this;
      result.source_ = source_.with_id(id);
      return result;
    }

  private:
    template <typename, typename>
    friend class TestExpression;

    SourceFieldType source_;
    double          scale_ = 1.;
  };

  namespace detail
  {
    template <typename Type>
    struct is_test_expression : std::false_type
    {};

    template <typename SourceFieldType, typename Operation>
    struct is_test_expression<TestExpression<SourceFieldType, Operation>>
      : std::true_type
    {};
  } // namespace detail

  /// \cond IMMERSX_INTERNAL
  template <typename ObservableType, typename Evaluator>
  decltype(auto)
  evaluate_observable_input(
    const ObservableType                                  &observable,
    const dealii::Point<ObservableType::spacedimension()> &point,
    const double                                           time,
    Evaluator                                            &&evaluator);

  template <typename ObservableType,
            typename Evaluator,
            typename DerivativeEvaluator>
  decltype(auto)
  linearize_observable_input(
    const ObservableType                                  &observable,
    const FieldId                                          field,
    const dealii::Point<ObservableType::spacedimension()> &point,
    const double                                           time,
    Evaluator                                            &&evaluator,
    DerivativeEvaluator &&derivative_evaluator);

  namespace detail
  {
    template <typename Kernel, int spacedim, bool>
    struct scalar_kernel_value_type;

    template <typename Kernel, int spacedim>
    struct scalar_kernel_value_type<Kernel, spacedim, true>
    {
      using evaluation_type = decltype(std::declval<const Kernel &>().evaluate(
        std::declval<const dealii::Point<spacedim> &>(),
        0.,
        std::declval<const std::vector<double> &>()));
      using type            = std::decay_t<decltype(kernel_value(
        std::declval<const evaluation_type &>()))>;
    };

    template <typename Kernel, int spacedim>
    struct scalar_kernel_value_type<Kernel, spacedim, false>
    {
      using evaluation_type = decltype(std::declval<const Kernel &>().evaluate(
        std::declval<const std::vector<double> &>()));
      using type            = std::decay_t<decltype(kernel_value(
        std::declval<const evaluation_type &>()))>;
    };

    template <typename Kernel, typename Tuple>
    struct typed_kernel_value_type;

    template <typename Kernel, typename... InputTypes>
    struct typed_kernel_value_type<Kernel, std::tuple<InputTypes...>>
    {
      using evaluation_type = decltype(std::declval<const Kernel &>().evaluate(
        std::declval<const InputTypes &>()...));
      using type            = std::decay_t<decltype(kernel_value(
        std::declval<const evaluation_type &>()))>;
    };

    template <typename Kernel, typename Tuple, bool, int spacedim>
    struct transformed_value_type;

    template <typename Kernel, typename Tuple, int spacedim>
    struct transformed_value_type<Kernel, Tuple, true, spacedim>
    {
      using type = typename typed_kernel_value_type<Kernel, Tuple>::type;
    };

    template <typename Kernel, typename Tuple, int spacedim>
    struct transformed_value_type<Kernel, Tuple, false, spacedim>
    {
      using type = typename scalar_kernel_value_type<
        Kernel,
        spacedim,
        has_point_scalar_kernel_evaluation<Kernel, spacedim>::value>::type;
    };

    template <typename Kernel, typename Tuple>
    struct has_typed_kernel_linearization;

    template <typename Kernel, typename... InputTypes>
    struct has_typed_kernel_linearization<Kernel, std::tuple<InputTypes...>>
    {
    private:
      template <typename Type>
      static auto
      test(int) -> decltype(std::declval<const Type &>().linearize(
                              std::declval<const InputTypes &>()...,
                              std::declval<const InputTypes &>()...),
                            std::true_type{});

      template <typename>
      static auto
      test(...) -> std::false_type;

    public:
      static constexpr bool value = decltype(test<Kernel>(0))::value;
    };
  } // namespace detail

  /** A compile-time pointwise composition of one or more Observables. */
  template <typename Kernel, typename... InputObservables>
  class TransformedObservable
  {
    static_assert(sizeof...(InputObservables) > 0,
                  "A nonlinear Observable needs at least one input.");
    static_assert(
      (detail::is_observable<std::decay_t<InputObservables>>::value && ...),
      "All nonlinear Observable inputs must be Observables.");

    using FirstInput =
      std::tuple_element_t<0, std::tuple<std::decay_t<InputObservables>...>>;

  public:
    using source_field_type = typename FirstInput::source_field_type;
    using input_value_types =
      std::tuple<typename std::decay_t<InputObservables>::value_type...>;
    static constexpr bool has_typed_kernel =
      detail::has_typed_kernel_evaluation<
        Kernel,
        typename std::decay_t<InputObservables>::value_type...>::value &&
      !detail::has_point_scalar_kernel_evaluation<
        Kernel,
        source_field_type::spacedimension()>::value &&
      !detail::has_scalar_kernel_evaluation<Kernel>::value;
    using value_type = typename detail::transformed_value_type<
      Kernel,
      input_value_types,
      has_typed_kernel,
      source_field_type::spacedimension()>::type;
    static constexpr int space_dim = source_field_type::spacedimension();

    TransformedObservable(Kernel kernel, InputObservables... inputs)
      : kernel_(std::make_shared<const Kernel>(std::move(kernel)))
      , inputs_(std::move(inputs)...)
      , dependencies_(collect_dependencies())
    {}

    const std::vector<FieldId> &
    dependencies() const
    {
      return dependencies_;
    }

    static constexpr unsigned int
    dimension()
    {
      return source_field_type::dimension();
    }

    static constexpr unsigned int
    space_dimension()
    {
      return source_field_type::space_dimension();
    }

    static constexpr unsigned int
    spacedimension()
    {
      return source_field_type::spacedimension();
    }

    const source_field_type &
    source() const
    {
      return std::get<0>(inputs_).source();
    }

    const source_field_type &
    source_for(const FieldId field) const
    {
      const source_field_type *result = nullptr;
      std::apply(
        [&](const auto &...input) {
          (
            [&] {
              if (std::find(input.dependencies().begin(),
                            input.dependencies().end(),
                            field) != input.dependencies().end())
                result = &input.source_for(field);
            }(),
            ...);
        },
        inputs_);
      AssertThrow(result != nullptr,
                  dealii::ExcMessage(
                    "The requested field is not a nonlinear Observable "
                    "dependency."));
      return *result;
    }

    FieldId
    source_field() const
    {
      return dependencies_.empty() ? FieldId() : dependencies_.front();
    }

    static constexpr dealii::UpdateFlags
    update_flags()
    {
      return (InputObservables::update_flags() | ...);
    }

    bool
    is_frozen() const
    {
      return dependencies_.empty();
    }

    double
    scale() const
    {
      return scale_;
    }

    TransformedObservable
    scaled(const double coefficient) const
    {
      auto result = *this;
      result.scale_ *= coefficient;
      return result;
    }

    static constexpr bool
    is_linear()
    {
      return false;
    }

    const Kernel &
    kernel() const
    {
      return *kernel_;
    }

    template <typename Evaluator>
    value_type
    evaluate_point(const dealii::Point<space_dim> &point,
                   const double                    time,
                   Evaluator                     &&evaluator) const
    {
      const auto values = input_values(point, time, evaluator);
      return scale_ * evaluate_kernel(point, time, values);
    }

    template <typename Evaluator, typename DerivativeEvaluator>
    value_type
    linearize_point(const FieldId                   field,
                    const dealii::Point<space_dim> &point,
                    const double                    time,
                    Evaluator                     &&evaluator,
                    DerivativeEvaluator           &&derivative_evaluator) const
    {
      if (std::find(dependencies_.begin(), dependencies_.end(), field) ==
          dependencies_.end())
        return detail::zero_like<value_type>();

      const auto values = input_values(point, time, evaluator);
      const auto directions =
        input_directions(field, point, time, evaluator, derivative_evaluator);
      return scale_ * linearize_kernel(point, time, values, directions);
    }

  private:
    template <typename Tuple>
    value_type
    evaluate_kernel(const dealii::Point<space_dim> &point,
                    const double                    time,
                    const Tuple                    &values) const
    {
      if constexpr (has_typed_kernel)
        return std::apply(
          [this](const auto &...input) {
            return detail::kernel_value(kernel_->evaluate(input...));
          },
          values);
      else
        {
          static_assert(
            (std::is_arithmetic_v<
               typename std::decay_t<InputObservables>::value_type> &&
             ...),
            "A scalar vector kernel may only be used with scalar Observables; "
            "typed kernels must accept the native Observable value types.");
          std::vector<double> scalar_values;
          std::apply(
            [&scalar_values](const auto &...input) {
              scalar_values.reserve(sizeof...(input));
              (scalar_values.push_back(static_cast<double>(input)), ...);
            },
            values);
          return detail::kernel_value(detail::evaluate_scalar_kernel(
            *kernel_, point, time, scalar_values));
        }
    }

    template <typename Tuple>
    value_type
    linearize_kernel(const dealii::Point<space_dim> &point,
                     const double                    time,
                     const Tuple                    &values,
                     const Tuple                    &directions) const
    {
      if constexpr (detail::has_typed_kernel_linearization<
                      Kernel,
                      input_value_types>::value)
        {
          const auto all = std::tuple_cat(values, directions);
          return std::apply(
            [this](const auto &...input) {
              return kernel_->linearize(input...);
            },
            all);
        }
      else
        {
          static_assert(!has_typed_kernel,
                        "A typed C++ kernel must provide linearize(values..., "
                        "directions...) returning its native output type.");
          std::vector<double> scalar_values;
          std::vector<double> scalar_directions;
          std::apply(
            [&scalar_values](const auto &...input) {
              scalar_values.reserve(sizeof...(input));
              (scalar_values.push_back(static_cast<double>(input)), ...);
            },
            values);
          std::apply(
            [&scalar_directions](const auto &...input) {
              scalar_directions.reserve(sizeof...(input));
              (scalar_directions.push_back(static_cast<double>(input)), ...);
            },
            directions);
          const auto evaluation = detail::evaluate_scalar_kernel(*kernel_,
                                                                 point,
                                                                 time,
                                                                 scalar_values);
          value_type result     = detail::zero_like<value_type>();
          for (std::size_t i = 0; i < scalar_directions.size(); ++i)
            result += evaluation.derivatives[i] * scalar_directions[i];
          return result;
        }
    }

    std::vector<FieldId>
    collect_dependencies() const
    {
      std::vector<FieldId> result;
      std::apply(
        [&result](const auto &...input) {
          (
            [&result](const auto &observable) {
              for (const auto field : observable.dependencies())
                if (std::find(result.begin(), result.end(), field) ==
                    result.end())
                  result.push_back(field);
            }(input),
            ...);
        },
        inputs_);
      return result;
    }

    template <typename Evaluator>
    auto
    input_values(const dealii::Point<space_dim> &point,
                 const double                    time,
                 Evaluator                      &evaluator) const
    {
      return std::apply(
        [&](const auto &...input) {
          return std::make_tuple(
            evaluate_observable_input(input, point, time, evaluator)...);
        },
        inputs_);
    }

    template <typename Evaluator, typename DerivativeEvaluator>
    auto
    input_directions(const FieldId                   field,
                     const dealii::Point<space_dim> &point,
                     const double                    time,
                     Evaluator                      &evaluator,
                     DerivativeEvaluator            &derivative_evaluator) const
    {
      return std::apply(
        [&](const auto &...input) {
          return std::make_tuple(linearize_observable_input(
            input, field, point, time, evaluator, derivative_evaluator)...);
        },
        inputs_);
    }

    std::shared_ptr<const Kernel>                 kernel_;
    std::tuple<std::decay_t<InputObservables>...> inputs_;
    std::vector<FieldId>                          dependencies_;
    double                                        scale_ = 1.;
  };

  namespace detail
  {
    template <typename Kernel, typename... InputObservables>
    struct is_observable<TransformedObservable<Kernel, InputObservables...>>
      : std::true_type
    {};

    template <typename Kernel, typename... InputObservables>
    struct is_transformed_observable<
      TransformedObservable<Kernel, InputObservables...>> : std::true_type
    {};
  } // namespace detail

  template <typename ObservableType, typename Evaluator>
  decltype(auto)
  evaluate_observable_input(
    const ObservableType                                  &observable,
    const dealii::Point<ObservableType::spacedimension()> &point,
    const double                                           time,
    Evaluator                                            &&evaluator)
  {
    if constexpr (detail::is_transformed_observable<ObservableType>::value)
      return observable.evaluate_point(point,
                                       time,
                                       std::forward<Evaluator>(evaluator));
    else
      return evaluator(observable, point, time);
  }

  template <typename ObservableType,
            typename Evaluator,
            typename DerivativeEvaluator>
  decltype(auto)
  linearize_observable_input(
    const ObservableType                                  &observable,
    const FieldId                                          field,
    const dealii::Point<ObservableType::spacedimension()> &point,
    const double                                           time,
    Evaluator                                            &&evaluator,
    DerivativeEvaluator                                  &&derivative_evaluator)
  {
    if constexpr (detail::is_transformed_observable<ObservableType>::value)
      return observable.linearize_point(
        field, point, time, evaluator, derivative_evaluator);
    else
      return observable.linearize_point(field,
                                        point,
                                        time,
                                        std::forward<DerivativeEvaluator>(
                                          derivative_evaluator));
  }
  /// \endcond

  namespace detail
  {
    template <typename Tuple, std::size_t... I>
    auto
    make_transformed_observable(Tuple &&all, std::index_sequence<I...>)
    {
      constexpr std::size_t kernel_index = sizeof...(I);
      using Kernel = std::decay_t<decltype(std::get<kernel_index>(all))>;
      return TransformedObservable<Kernel,
                                   std::decay_t<decltype(std::get<I>(all))>...>(
        std::move(std::get<kernel_index>(all)), std::move(std::get<I>(all))...);
    }
  } // namespace detail

  template <typename First, typename... Rest>
  auto
  transform(First first, Rest... rest)
  {
    static_assert(sizeof...(Rest) > 0,
                  "transform requires at least one input and one kernel.");
    auto all = std::tuple<std::decay_t<First>, std::decay_t<Rest>...>(
      std::move(first), std::move(rest)...);
    return detail::make_transformed_observable(
      std::move(all), std::make_index_sequence<sizeof...(Rest)>{});
  }

  template <typename FieldType,
            std::enable_if_t<
              detail::is_field<FieldType>::value &&
                detail::is_supported_operation<std::decay_t<FieldType>,
                                               detail::ValueOperation>::value,
              int> = 0>
  auto
  value(const FieldType &field)
  {
    return Observable<std::decay_t<FieldType>, detail::ValueOperation>(field);
  }

  template <typename FieldType,
            std::enable_if_t<detail::is_field<FieldType>::value &&
                               detail::is_supported_operation<
                                 std::decay_t<FieldType>,
                                 detail::GradientOperation>::value,
                             int> = 0>
  auto
  gradient(const FieldType &field)
  {
    return Observable<std::decay_t<FieldType>, detail::GradientOperation>(
      field);
  }

  template <typename FieldType,
            std::enable_if_t<detail::is_field<FieldType>::value &&
                               detail::is_supported_operation<
                                 std::decay_t<FieldType>,
                                 detail::DivergenceOperation>::value,
                             int> = 0>
  auto
  divergence(const FieldType &field)
  {
    return Observable<std::decay_t<FieldType>, detail::DivergenceOperation>(
      field);
  }

  template <typename FieldType,
            std::enable_if_t<detail::is_field<FieldType>::value &&
                               detail::is_supported_operation<
                                 std::decay_t<FieldType>,
                                 detail::SymmetricGradientOperation>::value,
                             int> = 0>
  auto
  symmetric_gradient(const FieldType &field)
  {
    return Observable<std::decay_t<FieldType>,
                      detail::SymmetricGradientOperation>(field);
  }

  template <typename FieldType,
            std::enable_if_t<
              detail::is_field<FieldType>::value &&
                detail::is_supported_operation<std::decay_t<FieldType>,
                                               detail::CurlOperation>::value,
              int> = 0>
  auto
  curl(const FieldType &field)
  {
    return Observable<std::decay_t<FieldType>, detail::CurlOperation>(field);
  }

  /** Mark a Field as the residual row represented by a test function. */
  template <typename FieldType,
            std::enable_if_t<detail::is_field<FieldType>::value, int> = 0>
  auto
  test(const FieldType &field)
  {
    return TestExpression<std::decay_t<FieldType>, detail::ValueOperation>(
      field);
  }

  /** \cond */
  template <
    typename Type,
    std::enable_if_t<detail::is_observable<std::decay_t<Type>>::value, int> = 0>
  const std::decay_t<Type> &
  as_fe_expression(const Type &expression)
  {
    return expression;
  }

  template <
    typename Type,
    std::enable_if_t<detail::is_field<std::decay_t<Type>>::value, int> = 0>
  auto
  as_fe_expression(const Type &field)
  {
    return value(field);
  }
  /** \endcond */

  /** Construct a dependency-free expression from fixed FE coefficients. */
  template <typename FieldType, typename VectorType>
  auto
  frozen(const FieldType &field, const VectorType &values)
  {
    return Observable<std::decay_t<FieldType>, detail::ValueOperation>(field,
                                                                       values);
  }

  /** Re-evaluate a frozen or active FE quantity with another FE operation. */
  template <typename SourceFieldType, typename Operation>
  auto
  value(const Observable<SourceFieldType, Operation> &observable)
  {
    return Observable<SourceFieldType, detail::ValueOperation>(observable);
  }

  template <typename SourceFieldType, typename Operation>
  auto
  gradient(const Observable<SourceFieldType, Operation> &observable)
  {
    return Observable<SourceFieldType, detail::GradientOperation>(observable);
  }

  template <typename SourceFieldType, typename Operation>
  auto
  divergence(const Observable<SourceFieldType, Operation> &observable)
  {
    return Observable<SourceFieldType, detail::DivergenceOperation>(observable);
  }

  template <typename SourceFieldType, typename Operation>
  auto
  symmetric_gradient(const Observable<SourceFieldType, Operation> &observable)
  {
    return Observable<SourceFieldType, detail::SymmetricGradientOperation>(
      observable);
  }

  template <typename SourceFieldType, typename Operation>
  auto
  curl(const Observable<SourceFieldType, Operation> &observable)
  {
    return Observable<SourceFieldType, detail::CurlOperation>(observable);
  }

  template <typename SourceFieldType, typename Operation>
  auto
  value(const TestExpression<SourceFieldType, Operation> &expression)
  {
    return TestExpression<SourceFieldType, detail::ValueOperation>(expression);
  }

  template <typename SourceFieldType, typename Operation>
  auto
  gradient(const TestExpression<SourceFieldType, Operation> &expression)
  {
    return TestExpression<SourceFieldType, detail::GradientOperation>(
      expression);
  }

  template <typename SourceFieldType, typename Operation>
  auto
  divergence(const TestExpression<SourceFieldType, Operation> &expression)
  {
    return TestExpression<SourceFieldType, detail::DivergenceOperation>(
      expression);
  }

  template <typename SourceFieldType, typename Operation>
  auto
  symmetric_gradient(
    const TestExpression<SourceFieldType, Operation> &expression)
  {
    return TestExpression<SourceFieldType, detail::SymmetricGradientOperation>(
      expression);
  }

  template <typename SourceFieldType, typename Operation>
  auto
  curl(const TestExpression<SourceFieldType, Operation> &expression)
  {
    return TestExpression<SourceFieldType, detail::CurlOperation>(expression);
  }

  namespace detail
  {
    template <typename Type>
    inline constexpr bool is_expression_operand =
      is_field<std::decay_t<Type>>::value ||
      is_observable<std::decay_t<Type>>::value;

    struct AddKernel
    {
      template <typename Left, typename Right>
      auto
      evaluate(const Left &left, const Right &right) const
      {
        return left + right;
      }

      template <typename Left, typename Right>
      auto
      linearize(const Left  &left,
                const Right &right,
                const Left  &dleft,
                const Right &dright) const
      {
        (void)left;
        (void)right;
        return dleft + dright;
      }
    };

    struct SubtractKernel
    {
      template <typename Left, typename Right>
      auto
      evaluate(const Left &left, const Right &right) const
      {
        return left - right;
      }

      template <typename Left, typename Right>
      auto
      linearize(const Left  &left,
                const Right &right,
                const Left  &dleft,
                const Right &dright) const
      {
        (void)left;
        (void)right;
        return dleft - dright;
      }
    };

    struct ProductKernel
    {
      template <typename Left, typename Right>
      auto
      evaluate(const Left &left, const Right &right) const
      {
        return left * right;
      }

      template <typename Left, typename Right>
      auto
      linearize(const Left  &left,
                const Right &right,
                const Left  &dleft,
                const Right &dright) const
      {
        return dleft * right + left * dright;
      }
    };

    struct NegateKernel
    {
      template <typename Value>
      auto
      evaluate(const Value &value) const
      {
        return -value;
      }

      template <typename Value>
      auto
      linearize(const Value &value, const Value &direction) const
      {
        (void)value;
        return -direction;
      }
    };
  } // namespace detail

  template <typename Left,
            typename Right,
            std::enable_if_t<detail::is_expression_operand<Left> &&
                               detail::is_expression_operand<Right>,
                             int> = 0>
  auto
  operator+(Left left, Right right)
  {
    return transform(as_fe_expression(left),
                     as_fe_expression(right),
                     detail::AddKernel{});
  }

  template <typename Left,
            typename Right,
            std::enable_if_t<detail::is_expression_operand<Left> &&
                               detail::is_expression_operand<Right>,
                             int> = 0>
  auto
  operator-(Left left, Right right)
  {
    return transform(as_fe_expression(left),
                     as_fe_expression(right),
                     detail::SubtractKernel{});
  }

  template <
    typename Expression,
    std::enable_if_t<detail::is_expression_operand<Expression>, int> = 0>
  auto
  operator-(Expression expression)
  {
    return transform(as_fe_expression(expression), detail::NegateKernel{});
  }

  template <typename Left,
            typename Right,
            std::enable_if_t<detail::is_expression_operand<Left> &&
                               detail::is_expression_operand<Right>,
                             int> = 0>
  auto
  operator*(Left left, Right right)
  {
    return transform(as_fe_expression(left),
                     as_fe_expression(right),
                     detail::ProductKernel{});
  }

  template <
    typename Expression,
    std::enable_if_t<detail::is_expression_operand<Expression>, int> = 0>
  auto
  operator*(const double coefficient, Expression expression)
  {
    return as_fe_expression(expression).scaled(coefficient);
  }

  template <
    typename Expression,
    std::enable_if_t<detail::is_expression_operand<Expression>, int> = 0>
  auto
  operator*(Expression expression, const double coefficient)
  {
    return as_fe_expression(expression).scaled(coefficient);
  }

  template <
    typename Expression,
    std::enable_if_t<detail::is_expression_operand<Expression>, int> = 0>
  auto
  operator/(Expression expression, const double coefficient)
  {
    return as_fe_expression(expression).scaled(1. / coefficient);
  }

  template <typename SourceFieldType, typename Operation>
  auto
  operator*(const double                               coefficient,
            TestExpression<SourceFieldType, Operation> expression)
  {
    return expression.scaled(coefficient);
  }

  template <typename SourceFieldType, typename Operation>
  auto
  operator*(TestExpression<SourceFieldType, Operation> expression,
            const double                               coefficient)
  {
    return expression.scaled(coefficient);
  }
} // namespace ImmersX

#endif // immersx_observable_h
