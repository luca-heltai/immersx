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
    struct has_point_kernel_evaluation : std::false_type
    {};

    template <typename Kernel, int spacedim>
    struct has_point_kernel_evaluation<
      Kernel,
      spacedim,
      std::void_t<decltype(std::declval<const Kernel &>().evaluate(
        std::declval<const dealii::Point<spacedim> &>(),
        0.,
        std::declval<const std::vector<double> &>()))>> : std::true_type
    {};

    template <typename Kernel, typename = void>
    struct has_value_kernel_evaluation : std::false_type
    {};

    template <typename Kernel>
    struct has_value_kernel_evaluation<
      Kernel,
      std::void_t<decltype(std::declval<const Kernel &>().evaluate(
        std::declval<const std::vector<double> &>()))>> : std::true_type
    {};

    template <typename Kernel, int spacedim>
    auto
    evaluate_kernel(const Kernel                  &kernel,
                    const dealii::Point<spacedim> &point,
                    const double                   time,
                    const std::vector<double>     &values)
    {
      if constexpr (has_point_kernel_evaluation<Kernel, spacedim>::value)
        return kernel.evaluate(point, time, values);
      else
        {
          static_assert(has_value_kernel_evaluation<Kernel>::value,
                        "A nonlinear kernel must provide evaluate(values) or "
                        "evaluate(point, time, values).");
          return kernel.evaluate(values);
        }
    }

    template <typename Type>
    struct is_transformed_observable : std::false_type
    {};

    template <typename Type>
    struct is_linear_observable : std::false_type
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
    double
    linearize_point(const FieldId                    field,
                    const dealii::Point<spacedim()> &point,
                    const double                     time,
                    DerivativeEvaluator            &&derivative_evaluator) const
    {
      if (is_frozen() || source_field() != field)
        return 0.;
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
  double
  linearize_observable_input(
    const ObservableType                                  &observable,
    const FieldId                                          field,
    const dealii::Point<ObservableType::spacedimension()> &point,
    const double                                           time,
    Evaluator                                            &&evaluator,
    DerivativeEvaluator &&derivative_evaluator);

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
    using source_field_type        = typename FirstInput::source_field_type;
    using value_type               = double;
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
      return scale_ *
             detail::evaluate_kernel(*kernel_, point, time, values).value;
    }

    template <typename Evaluator, typename DerivativeEvaluator>
    double
    linearize_point(const FieldId                   field,
                    const dealii::Point<space_dim> &point,
                    const double                    time,
                    Evaluator                     &&evaluator,
                    DerivativeEvaluator           &&derivative_evaluator) const
    {
      const auto values = input_values(point, time, evaluator);
      const auto evaluation =
        detail::evaluate_kernel(*kernel_, point, time, values);
      return scale_ * linearize_inputs(field,
                                       point,
                                       time,
                                       evaluation.derivatives,
                                       evaluator,
                                       derivative_evaluator);
    }

  private:
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
    std::vector<double>
    input_values(const dealii::Point<space_dim> &point,
                 const double                    time,
                 Evaluator                      &evaluator) const
    {
      std::vector<double> result;
      result.reserve(sizeof...(InputObservables));
      std::apply(
        [&](const auto &...input) {
          (result.push_back(
             evaluate_observable_input(input, point, time, evaluator)),
           ...);
        },
        inputs_);
      return result;
    }

    template <typename Evaluator, typename DerivativeEvaluator>
    double
    linearize_inputs(const FieldId                   field,
                     const dealii::Point<space_dim> &point,
                     const double                    time,
                     const std::vector<double>      &kernel_derivatives,
                     Evaluator                      &evaluator,
                     DerivativeEvaluator            &derivative_evaluator) const
    {
      AssertDimension(kernel_derivatives.size(), sizeof...(InputObservables));
      double      result = 0.;
      std::size_t index  = 0;
      std::apply(
        [&](const auto &...input) {
          ((result +=
            kernel_derivatives[index++] *
            linearize_observable_input(
              input, field, point, time, evaluator, derivative_evaluator)),
           ...);
        },
        inputs_);
      return result;
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
  double
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
      return derivative_evaluator(observable, field, point, time);
  }

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
  Observable<SourceFieldType, Operation>
  operator*(const double                           coefficient,
            Observable<SourceFieldType, Operation> observable)
  {
    return observable.scaled(coefficient);
  }

  template <typename SourceFieldType, typename Operation>
  Observable<SourceFieldType, Operation>
  operator*(Observable<SourceFieldType, Operation> observable,
            const double                           coefficient)
  {
    return observable.scaled(coefficient);
  }
} // namespace ImmersX

#endif // immersx_observable_h
