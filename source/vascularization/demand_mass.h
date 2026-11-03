// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_demand_mass_h
#define immersx_internal_demand_mass_h
#include <deal.II/base/exceptions.h>

#include <boost/multiprecision/cpp_int.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
namespace ImmersX::Vascularization::internal
{
  // Every positive binary64 mass is an integer multiple of 2^-1074. Exact
  // addition/subtraction preserves a tiny remaining territory when a much
  // larger contribution is reassigned. No extra geometric work is needed.
  inline boost::multiprecision::cpp_int
  demand_mass_units(double mass)
  {
    static_assert(std::numeric_limits<double>::is_iec559 &&
                    std::numeric_limits<double>::digits == 53 &&
                    std::numeric_limits<double>::min_exponent == -1021,
                  "Exact demand accumulation requires IEEE binary64.");
    AssertThrow(std::isfinite(mass) && mass >= 0.,
                dealii::ExcMessage(
                  "Quadrature mass must be finite and nonnegative."));
    int                            exponent = 0;
    const double                   fraction = std::frexp(mass, &exponent);
    boost::multiprecision::cpp_int result =
      static_cast<std::uint64_t>(std::ldexp(fraction, 53));
    const int shift = exponent + 1021;
    if (shift >= 0)
      result <<= shift;
    else
      result >>= -shift;
    return result;
  }
  inline double
  physical_demand_mass(const boost::multiprecision::cpp_int &units)
  {
    AssertThrow(units >= 0,
                dealii::ExcMessage(
                  "Integrated demand cannot have negative exact mass."));
    if (units == 0)
      return 0.;
    const unsigned int             bits    = boost::multiprecision::msb(units);
    const unsigned int             shift   = bits > 52 ? bits - 52 : 0;
    boost::multiprecision::cpp_int rounded = units >> shift;
    if (shift > 0)
      {
        const boost::multiprecision::cpp_int remainder =
          units - (rounded << shift);
        const boost::multiprecision::cpp_int half =
          boost::multiprecision::cpp_int(1) << (shift - 1);
        if (remainder > half || (remainder == half && bit_test(rounded, 0)))
          ++rounded;
      }
    return std::ldexp(rounded.convert_to<double>(),
                      static_cast<int>(shift) - 1074);
  }
} // namespace ImmersX::Vascularization::internal
#endif
