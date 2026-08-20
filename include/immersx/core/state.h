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

#include <immersx/core/field.h>

#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace ImmersX
{
  /**
   * Read-only state access supplied by the caller of a residual evaluation.
   *
   * The requested time is part of the interface so a history- or
   * interpolation-backed implementation can provide a field at a time other
   * than the most recently stored state. Implementations and returned field
   * references are owned by the caller and must remain alive for the duration
   * of the evaluation that uses them.
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
   * time. More general time-dependent access is provided by StateAccessor. The
   * StateLayout and every bound vector must outlive the view.
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

    StateView &
    bind(const FieldId, VectorType &&) = delete;

    StateView &
    bind(const FieldId, const VectorType &&) = delete;

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
    static TermSelection
    all()
    {
      return TermSelection();
    }

    static TermSelection
    none()
    {
      TermSelection result;
      result.all_terms_ = false;
      return result;
    }

    static TermSelection
    only(const std::string &term)
    {
      TermSelection result = none();
      result.selected_terms_.insert(term);
      return result;
    }

    TermSelection &
    set(const std::string &term, const TermTreatment treatment)
    {
      treatments_[term] = treatment;
      if (!all_terms_)
        selected_terms_.insert(term);
      return *this;
    }

    TermSelection &
    include(const std::string &term)
    {
      all_terms_ = false;
      selected_terms_.insert(term);
      return *this;
    }

    TermSelection &
    exclude(const std::string &term)
    {
      all_terms_ = false;
      selected_terms_.erase(term);
      return *this;
    }

    bool
    contains(const std::string &term) const
    {
      return all_terms_ || selected_terms_.find(term) != selected_terms_.end();
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
      if (!contains(term))
        return false;
      const auto selected = treatment(term);
      return selected == TermTreatment::all ||
             requested == TermTreatment::all || selected == requested;
    }

  private:
    bool                                 all_terms_ = true;
    std::set<std::string>                selected_terms_;
    std::map<std::string, TermTreatment> treatments_;
  };

  /**
   * Inputs shared by all contributors during one residual evaluation.
   *
   * This is an evaluation-scoped view. It retains non-owning references to
   * the supplied state accessors and must not be stored by a contributor or
   * used after those accessors are destroyed.
   */
  template <typename VectorType>
  class EvaluationContext
  {
  public:
    using HistoryQuery = std::function<VectorType(HistoryGroupId, double)>;

    EvaluationContext(
      const double                     time,
      const StateAccessor<VectorType> &state,
      const StateAccessor<VectorType> *state_derivative = nullptr,
      TermSelection                    terms            = {},
      HistoryQuery                     history_query    = {})
      : time_(time)
      , state_(state)
      , state_derivative_(state_derivative)
      , terms_(std::move(terms))
      , history_query_(std::move(history_query))
    {}

    EvaluationContext(double,
                      StateAccessor<VectorType> &&,
                      const StateAccessor<VectorType> * = nullptr,
                      TermSelection                     = {},
                      HistoryQuery                      = {}) = delete;

    EvaluationContext(double,
                      const StateAccessor<VectorType> &&,
                      const StateAccessor<VectorType> * = nullptr,
                      TermSelection                     = {},
                      HistoryQuery                      = {}) = delete;

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

    bool
    has_history() const
    {
      return static_cast<bool>(history_query_);
    }

    VectorType
    historical_state(const HistoryGroupId group, const double time) const
    {
      AssertThrow(history_query_,
                  dealii::ExcMessage(
                    "No state history query is attached to this evaluation "
                    "context."));
      return history_query_(group, time);
    }

  private:
    double                           time_;
    const StateAccessor<VectorType> &state_;
    const StateAccessor<VectorType> *state_derivative_;
    TermSelection                    terms_;
    HistoryQuery                     history_query_;
  };

  /** Coefficients of the state and state-derivative parts of a Jacobian. */
  template <typename VectorType>
  class LinearizationContext
  {
  public:
    LinearizationContext(const EvaluationContext<VectorType> &evaluation,
                         const double                         state_weight,
                         const double                         derivative_weight)
      : evaluation_(evaluation)
      , state_weight_(state_weight)
      , derivative_weight_(derivative_weight)
    {}

    const EvaluationContext<VectorType> &
    evaluation() const
    {
      return evaluation_;
    }

    double
    state_weight() const
    {
      return state_weight_;
    }

    double
    derivative_weight() const
    {
      return derivative_weight_;
    }

  private:
    const EvaluationContext<VectorType> &evaluation_;
    double                               state_weight_;
    double                               derivative_weight_;
  };
} // namespace ImmersX

#endif // immersx_state_h
