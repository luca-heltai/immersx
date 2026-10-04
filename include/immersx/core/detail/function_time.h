// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_detail_function_time_h
#define immersx_detail_function_time_h

#include <type_traits>
#include <utility>

namespace ImmersX::detail
{
  template <typename FunctionType, typename = void>
  struct has_set_time : std::false_type
  {};

  template <typename FunctionType>
  struct has_set_time<
    FunctionType,
    std::void_t<decltype(std::declval<FunctionType &>().set_time(0.))>>
    : std::true_type
  {};

  template <typename FunctionType>
  void
  set_function_time(const FunctionType &function, const double time)
  {
    using MutableFunction = std::remove_const_t<FunctionType>;
    if constexpr (has_set_time<MutableFunction>::value)
      const_cast<MutableFunction &>(function).set_time(time);
    else
      (void)time;
  }
} // namespace ImmersX::detail

#endif // immersx_detail_function_time_h
