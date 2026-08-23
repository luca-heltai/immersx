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
  class FieldComponentView;

  namespace detail
  {
    /*
     * These helpers deliberately use unqualified dependent calls.  The
     * observable, geometry, and coupling descriptors therefore provide their
     * behavior through ADL (or through the small member fallbacks below),
     * without a physics-facing base class in ImmersX.
     */
    template <typename ProblemHandle, typename Observable>
    auto
    invoke_observe(const ProblemHandle &handle,
                   const Observable    &observable,
                   int) -> decltype(make_representation(handle, observable))
    {
      return make_representation(handle, observable);
    }

    template <typename ProblemHandle, typename Observable>
    auto
    invoke_observe(const ProblemHandle &handle,
                   const Observable    &observable,
                   long) -> decltype(observable.create(handle))
    {
      return observable.create(handle);
    }

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

    template <typename Quantity, typename Target, typename Coupling>
    auto
    invoke_coupling(const Quantity &quantity,
                    const Target   &target,
                    const Coupling &coupling,
                    int)
      -> decltype(make_interaction(quantity, target, coupling))
    {
      return make_interaction(quantity, target, coupling);
    }

    template <typename Quantity, typename Target, typename Coupling>
    auto
    invoke_coupling(const Quantity &quantity,
                    const Target   &target,
                    const Coupling &coupling,
                    long) -> decltype(coupling.create(quantity, target))
    {
      return coupling.create(quantity, target);
    }

    template <typename Quantity, typename Target, typename Coupling>
    auto
    invoke_coupling(const Quantity &quantity,
                    const Target   &target,
                    const Coupling &coupling,
                    char) -> decltype(couple(quantity, target, coupling))
    {
      return couple(quantity, target, coupling);
    }
  } // namespace detail

  /**
   * Semantic view returned when a Problem is added to an execution adapter.
   *
   * The handle owns only the contributor's small field-descriptor value and a
   * non-owning pointer to the adapter.  It does not own state, vectors, or a
   * second Problem.  Physics modules extend it with `make_representation`
   * (or an Observable::create member) and use `fields()` to resolve their
   * semantic fields.
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

    auto
    observe(const FieldId id) const
    {
      return adapter_->observe(id);
    }

    auto
    observe(const FieldComponentView &view) const
    {
      return adapter_->observe(view);
    }

    template <typename Observable>
    decltype(auto)
    observe(const Observable &observable) const
    {
      return detail::invoke_observe(*this, observable, 0);
    }

  private:
    Adapter *adapter_;
    Fields   fields_;
  };

} // namespace ImmersX

#endif // immersx_problem_handle_h
