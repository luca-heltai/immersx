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

#ifndef immersx_state_h
#define immersx_state_h

#include <deal.II/base/exceptions.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "field.h"

namespace ImmersX
{
  /**
   * Read-only state access supplied by the caller of a residual evaluation.
   *
   * The requested time is part of the interface so a history- or
   * interpolation-backed implementation can provide a field at a time other
   * than the most recently stored state.
   */
  template <typename VectorType>
  class StateAccessor
  {
  public:
    virtual ~StateAccessor() = default;

    virtual const VectorType &
    field(FieldId field_id, double time) const = 0;
  };

  /**
   * Non-owning view of externally supplied field vectors at one evaluation
   * time. More general time-dependent access is provided by StateAccessor.
   */
  template <typename VectorType>
  class StateView : public StateAccessor<VectorType>
  {
  public:
    StateView(const StateLayout &layout, const double time)
      : layout_(layout)
      , time_(time)
      , fields_(layout.n_fields(), nullptr)
    {}

    /** Bind an externally owned vector to a semantic field. */
    StateView &
    bind(const FieldId field_id, const VectorType &values)
    {
      validate_field(field_id);
      AssertThrow(fields_[field_id.value()] == nullptr,
                  dealii::ExcMessage("A state field may only be bound once."));
      fields_[field_id.value()] = &values;
      return *this;
    }

    const VectorType &
    field(const FieldId field_id, const double requested_time) const override
    {
      validate_field(field_id);
      AssertThrow(requested_time == time_,
                  dealii::ExcMessage(
                    "StateView does not provide a different time; "
                    "use a history-backed StateAccessor instead."));
      AssertThrow(fields_[field_id.value()] != nullptr,
                  dealii::ExcMessage(
                    "The requested state field is not bound."));
      return *fields_[field_id.value()];
    }

    /** Return the time associated with this view. */
    double
    time() const
    {
      return time_;
    }

  private:
    void
    validate_field(const FieldId field_id) const
    {
      AssertThrow(layout_.contains(field_id),
                  dealii::ExcMessage("Field identifier is not in the state "
                                     "layout."));
    }

    const StateLayout              &layout_;
    double                          time_;
    std::vector<const VectorType *> fields_;
  };

  /** Minimal term-level selection placeholder for future IMEX treatment. */
  enum class TermTreatment
  {
    all,
    explicit_term,
    implicit_term
  };

  /**
   * Sparse term selection. Unmentioned terms have treatment `all`, leaving
   * contributors free to add their complete residual for now.
   */
  class TermSelection
  {
  public:
    TermSelection &
    set(const std::string &term, const TermTreatment treatment)
    {
      treatments_[term] = treatment;
      return *this;
    }

    TermTreatment
    treatment(const std::string &term) const
    {
      const auto it = treatments_.find(term);
      return it == treatments_.end() ? TermTreatment::all : it->second;
    }

    bool
    includes(const std::string &term, const TermTreatment requested) const
    {
      const auto selected = treatment(term);
      return selected == TermTreatment::all ||
             requested == TermTreatment::all || selected == requested;
    }

  private:
    std::map<std::string, TermTreatment> treatments_;
  };

  /** Inputs shared by all contributors during one residual evaluation. */
  template <typename VectorType>
  class EvaluationContext
  {
  public:
    EvaluationContext(
      const double                     time,
      const StateAccessor<VectorType> &state,
      const StateAccessor<VectorType> *state_derivative = nullptr,
      TermSelection                    terms            = {})
      : time_(time)
      , state_(state)
      , state_derivative_(state_derivative)
      , terms_(std::move(terms))
    {}

    double
    time() const
    {
      return time_;
    }

    const StateAccessor<VectorType> &
    state() const
    {
      return state_;
    }

    const StateAccessor<VectorType> *
    state_derivative() const
    {
      return state_derivative_;
    }

    bool
    has_state_derivative() const
    {
      return state_derivative_ != nullptr;
    }

    const TermSelection &
    terms() const
    {
      return terms_;
    }

  private:
    double                           time_;
    const StateAccessor<VectorType> &state_;
    const StateAccessor<VectorType> *state_derivative_;
    TermSelection                    terms_;
  };
} // namespace ImmersX

#endif // immersx_state_h
