// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_state_history_h
#define immersx_state_history_h

#include <deal.II/base/exceptions.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "time_residual.h"


/**
 * Default deterministic linear interpolation for deal.II-style vectors.
 *
 * A different policy can be supplied to StateHistory without changing its
 * storage or query interface.  This is the seam for higher-order
 * interpolation and controlled extrapolation.
 */
template <typename Value>
struct LinearHistoryInterpolation
{
  Value
  operator()(const Value &left, const Value &right, const double alpha) const
  {
    Value result = left;
    result *= (1. - alpha);
    result.add(alpha, right);
    return result;
  }
};


template <>
struct LinearHistoryInterpolation<double>
{
  double
  operator()(const double left, const double right, const double alpha) const
  {
    return (1. - alpha) * left + alpha * right;
  }
};


/** Accepted snapshots for one state or subsystem. */
template <typename Value,
          typename InterpolationPolicy = LinearHistoryInterpolation<Value>>
class StateHistory
{
public:
  struct Snapshot
  {
    double time;
    Value  state;
  };

  void
  accept(const double time, const Value &state)
  {
    AssertThrow(std::isfinite(time),
                dealii::ExcMessage("A history snapshot time must be finite."));
    if (!snapshots.empty())
      AssertThrow(time > snapshots.back().time,
                  dealii::ExcMessage(
                    "Accepted history snapshots must have strictly increasing "
                    "times."));

    snapshots.push_back({time, state});
  }

  bool
  empty() const
  {
    return snapshots.empty();
  }

  std::size_t
  size() const
  {
    return snapshots.size();
  }

  double
  first_time() const
  {
    AssertThrow(!empty(),
                dealii::ExcMessage("Cannot query an empty state history."));
    return snapshots.front().time;
  }

  double
  last_time() const
  {
    AssertThrow(!empty(),
                dealii::ExcMessage("Cannot query an empty state history."));
    return snapshots.back().time;
  }

  const Value &
  latest() const
  {
    AssertThrow(!empty(),
                dealii::ExcMessage("Cannot query an empty state history."));
    return snapshots.back().state;
  }

  /**
   * Return an accepted state or its deterministic linear interpolation.
   *
   * Queries outside the accepted interval are rejected.  Extrapolation is a
   * separate policy decision and is intentionally not hidden in this first
   * prototype.
   */
  Value
  at(const double time) const
  {
    AssertThrow(!empty(),
                dealii::ExcMessage("Cannot query an empty state history."));
    AssertThrow(time >= first_time() && time <= last_time(),
                dealii::ExcMessage("State history query is outside the "
                                   "accepted time interval."));

    const auto right =
      std::lower_bound(snapshots.begin(),
                       snapshots.end(),
                       time,
                       [](const Snapshot &snapshot, const double value) {
                         return snapshot.time < value;
                       });

    if (right == snapshots.begin() || right->time == time)
      return right->state;

    const auto  &left_snapshot = *(right - 1);
    const double alpha =
      (time - left_snapshot.time) / (right->time - left_snapshot.time);
    return InterpolationPolicy()(left_snapshot.state, right->state, alpha);
  }

  Value
  operator()(const double time) const
  {
    return at(time);
  }

  const std::vector<Snapshot> &
  snapshots_view() const
  {
    return snapshots;
  }

private:
  std::vector<Snapshot> snapshots;
};


/**
 * Type-erased registry of independent histories.
 *
 * Each subsystem has its own accepted time grid.  The registry does not know
 * whether a value is a field, a block vector, or a subsystem state; that
 * choice stays with the caller and can later be replaced by a StateLayout
 * adapter.
 */
template <typename Value,
          typename InterpolationPolicy = LinearHistoryInterpolation<Value>>
class StateHistoryRegistry
{
public:
  using History = StateHistory<Value, InterpolationPolicy>;

  void
  accept(const SubsystemId subsystem, const double time, const Value &state)
  {
    histories[subsystem].accept(time, state);
  }

  bool
  has_history(const SubsystemId subsystem) const
  {
    return histories.find(subsystem) != histories.end();
  }

  const History &
  history(const SubsystemId subsystem) const
  {
    const auto it = histories.find(subsystem);
    AssertThrow(it != histories.end(),
                dealii::ExcMessage("Unknown subsystem history."));
    return it->second;
  }

  Value
  at(const SubsystemId subsystem, const double time) const
  {
    return history(subsystem).at(time);
  }

  std::function<Value(SubsystemId, double)>
  query() const
  {
    return [this](const SubsystemId subsystem, const double time) {
      return at(subsystem, time);
    };
  }

private:
  std::map<SubsystemId, History> histories;
};

#endif // immersx_state_history_h
