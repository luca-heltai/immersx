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
#include <deal.II/base/tensor.h>

#include <immersx/core/fe_space.h>

#include <any>
#include <type_traits>
#include <utility>
#include <vector>

namespace ImmersX
{
  enum class ObservableOperation
  {
    value,
    gradient
  };

  /** Metadata for a typed, differentiable quantity derived from Fields.
   *
   * Observable deliberately describes only the mathematical quantity. It
   * does not expose point search, quadrature, evaluation contexts, or caches.
   * A frozen observable may retain caller-supplied coefficients, but has no
   * active Field dependency.
   */
  template <typename ValueType, typename SourceFieldType = void>
  class Observable
  {
  public:
    using value_type        = ValueType;
    using source_field_type = SourceFieldType;

    Observable(std::vector<FieldId>      dependencies,
               const unsigned int        dimension,
               const unsigned int        spacedim,
               const ObservableOperation operation)
      : dependencies_(std::move(dependencies))
      , dimension_(dimension)
      , spacedim_(spacedim)
      , operation_(operation)
    {}

    template <typename SourceField>
    Observable(std::vector<FieldId>      dependencies,
               const unsigned int        dimension,
               const unsigned int        spacedim,
               const ObservableOperation operation,
               const SourceField        &source)
      : dependencies_(std::move(dependencies))
      , dimension_(dimension)
      , spacedim_(spacedim)
      , operation_(operation)
      , source_(source)
    {}

    template <typename SourceField, typename VectorType>
    Observable(std::vector<FieldId>      dependencies,
               const unsigned int        dimension,
               const unsigned int        spacedim,
               const ObservableOperation operation,
               const SourceField        &source,
               const VectorType         &frozen_values)
      : dependencies_(std::move(dependencies))
      , dimension_(dimension)
      , spacedim_(spacedim)
      , operation_(operation)
      , source_(source)
      , frozen_(frozen_values)
    {}

    const std::vector<FieldId> &
    dependencies() const
    {
      return dependencies_;
    }

    unsigned int
    dimension() const
    {
      return dimension_;
    }

    unsigned int
    space_dimension() const
    {
      return dimension();
    }

    unsigned int
    spacedimension() const
    {
      return spacedim_;
    }

    unsigned int
    spacedim() const
    {
      return spacedimension();
    }

    FieldId
    source_field() const
    {
      AssertThrow(dependencies_.empty() || dependencies_.size() == 1,
                  dealii::ExcMessage(
                    "source_field() is only defined for one-field "
                    "observables."));
      return dependencies_.empty() ? FieldId() : dependencies_.front();
    }

    /** Return the typed source retained by an observable factory. */
    template <typename SourceField>
    const SourceField &
    source() const
    {
      const auto *source = std::any_cast<SourceField>(&source_);
      AssertThrow(source != nullptr,
                  dealii::ExcMessage(
                    "The observable does not contain the requested source "
                    "field type."));
      return *source;
    }

    /** Whether this observable is supplied by fixed, external values. */
    bool
    is_frozen() const
    {
      return frozen_.has_value();
    }

    /** Return the fixed values retained by a frozen observable. */
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

    ObservableOperation
    operation() const
    {
      return operation_;
    }

    /** Return the scalar coefficient applied to this observable. */
    double
    scale() const
    {
      return scale_;
    }

    /** Return a copy multiplied by a scalar coefficient. */
    Observable
    scaled(const double coefficient) const
    {
      Observable result = *this;
      result.scale_ *= coefficient;
      return result;
    }

    bool
    is_differentiable() const
    {
      return true;
    }

    bool
    is_linear() const
    {
      return true;
    }

  private:
    std::vector<FieldId> dependencies_;
    unsigned int         dimension_;
    unsigned int         spacedim_;
    ObservableOperation  operation_;
    double               scale_ = 1.;
    std::any             source_;
    std::any             frozen_;
  };

  template <typename FieldType>
  Observable<typename std::decay_t<FieldType>::value_type,
             std::decay_t<FieldType>>
  value(const FieldType &field)
  {
    using Field = std::decay_t<FieldType>;
    return Observable<typename Field::value_type, Field>(
      {field.field_id()},
      Field::dimension(),
      Field::spacedimension(),
      ObservableOperation::value,
      field);
  }

  template <typename FieldType>
  auto
  gradient(const FieldType &field)
  {
    using Field = std::decay_t<FieldType>;
    using GradientValue =
      std::conditional_t<std::is_same_v<typename Field::value_type, double>,
                         dealii::Tensor<1, Field::spacedimension()>,
                         dealii::Tensor<2, Field::spacedimension()>>;
    return Observable<GradientValue, Field>({field.field_id()},
                                            Field::dimension(),
                                            Field::spacedimension(),
                                            ObservableOperation::gradient,
                                            field);
  }

  /** Construct a dependency-free observable from fixed FE coefficients. */
  template <typename FieldType, typename VectorType>
  Observable<typename std::decay_t<FieldType>::value_type,
             std::decay_t<FieldType>>
  frozen(const FieldType &field, const VectorType &values)
  {
    using Field = std::decay_t<FieldType>;
    return Observable<typename Field::value_type, Field>(
      {},
      Field::dimension(),
      Field::spacedimension(),
      ObservableOperation::value,
      field,
      values);
  }

  /** Scale an observable while preserving its dependencies and source. */
  template <typename ValueType, typename SourceFieldType>
  Observable<ValueType, SourceFieldType>
  operator*(const double                           coefficient,
            Observable<ValueType, SourceFieldType> observable)
  {
    return observable.scaled(coefficient);
  }

  /** Scale an observable while preserving its dependencies and source. */
  template <typename ValueType, typename SourceFieldType>
  Observable<ValueType, SourceFieldType>
  operator*(Observable<ValueType, SourceFieldType> observable,
            const double                           coefficient)
  {
    return observable.scaled(coefficient);
  }
} // namespace ImmersX

#endif // immersx_observable_h
