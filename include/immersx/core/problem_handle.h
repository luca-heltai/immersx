// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_problem_handle_h
#define immersx_problem_handle_h

#include <immersx/core/field.h>

#include <utility>

namespace ImmersX
{
  namespace detail
  {
    template <typename Quantity, typename Geometry>
    auto
    invoke_lift(const Quantity &quantity, const Geometry &geometry, int)
      -> decltype(make_lift(quantity, geometry))
    {
      return make_lift(quantity, geometry);
    }

    template <typename Quantity, typename Geometry>
    auto
    invoke_lift(const Quantity &quantity, const Geometry &geometry, long)
      -> decltype(geometry.create(quantity))
    {
      return geometry.create(quantity);
    }

    template <typename Quantity, typename Geometry>
    auto
    invoke_lift(const Quantity &quantity, const Geometry &geometry, char)
      -> decltype(lift(quantity, geometry))
    {
      return lift(quantity, geometry);
    }
  } // namespace detail

  /**
   * Semantic view returned when a Problem is added to an execution adapter.
   *
   * The handle owns only the contributor's small field-descriptor value and a
   * non-owning pointer to the adapter.  It does not own state, vectors, or a
   * second Problem.
   */
  template <typename Adapter, typename Fields>
  class ProblemHandle
  {
  public:
    using adapter_type = Adapter;
    using fields_type  = Fields;

    ProblemHandle(Adapter &adapter, Fields fields)
      : adapter_(&adapter)
      , fields_(std::move(fields))
    {}

    const Fields &
    fields() const
    {
      return fields_;
    }

    Adapter &
    adapter() const
    {
      return *adapter_;
    }

  private:
    Adapter *adapter_;
    Fields   fields_;
  };

} // namespace ImmersX

#endif // immersx_problem_handle_h
