// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception (the "License"); either version 3.0 of the
// License, or (at your option) any later version. The full text of the
// license can be found in the LICENSE.md file at the top level of the
// ImmersX distribution.
//
// ---------------------------------------------------------------------

#ifndef immersx_residual_h
#define immersx_residual_h

#include <deal.II/base/exceptions.h>

#include <vector>

#include "state.h"

namespace ImmersX
{
  /**
   * Additive access to externally owned residual vectors by semantic field.
   *
   * A residual contributor only needs the structural interface
   * `add_residual(context, residual)`; it does not need to know global block
   * indices or how another contributor stores its native operator.
   */
  template <typename VectorType>
  class ResidualAccumulator
  {
  public:
    explicit ResidualAccumulator(const StateLayout &layout)
      : layout_(layout)
      , fields_(layout.n_fields(), nullptr)
    {}

    /** Bind an externally owned residual vector to a semantic field. */
    ResidualAccumulator &
    bind(const FieldId field_id, VectorType &values)
    {
      validate_field(field_id);
      AssertThrow(fields_[field_id.value()] == nullptr,
                  dealii::ExcMessage(
                    "A residual field may only be bound once."));
      fields_[field_id.value()] = &values;
      return *this;
    }

    /** Return mutable additive access to one residual row block. */
    VectorType &
    field(const FieldId field_id)
    {
      validate_field(field_id);
      AssertThrow(fields_[field_id.value()] != nullptr,
                  dealii::ExcMessage("The requested residual field is not "
                                     "bound."));
      return *fields_[field_id.value()];
    }

    /** Return read-only access to one residual row block. */
    const VectorType &
    field(const FieldId field_id) const
    {
      validate_field(field_id);
      AssertThrow(fields_[field_id.value()] != nullptr,
                  dealii::ExcMessage("The requested residual field is not "
                                     "bound."));
      return *fields_[field_id.value()];
    }

    /** Add a contribution to one residual row block. */
    void
    add(const FieldId field_id, const VectorType &contribution)
    {
      field(field_id).add(contribution);
    }

  private:
    void
    validate_field(const FieldId field_id) const
    {
      AssertThrow(layout_.contains(field_id),
                  dealii::ExcMessage("Field identifier is not in the state "
                                     "layout."));
    }

    const StateLayout        &layout_;
    std::vector<VectorType *> fields_;
  };
} // namespace ImmersX

#endif // immersx_residual_h
