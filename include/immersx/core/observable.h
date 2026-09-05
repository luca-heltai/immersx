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

#include <any>
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
  } // namespace detail

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
