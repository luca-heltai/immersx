#include "modulated_parsed_function.h"

#include <deal.II/base/numbers.h>
#include <deal.II/base/patterns.h>

#include <cmath>

template <int spacedim>
ModulatedParsedFunction<spacedim>::ModulatedParsedFunction(
  const std::string &section_name,
  const unsigned int n_components)
  : ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>(section_name,
                                                                n_components)
{
  this->declare_parameters_call_back.connect([this]() {
    this->prm.declare_entry(
      "Function expression",
      (spacedim == 2 ? "0; 0" : "0; 0; 0"),
      Patterns::List(Patterns::Anything(), spacedim, spacedim, ";"));
    this->prm.add_parameter("Modulation frequency", modulation_frequency);
    this->prm.add_parameter("Phase shift", phase_shift);
  });
}

template <int spacedim>
double
ModulatedParsedFunction<spacedim>::scale(const double time) const
{
  return (modulation_frequency == 0.0) ?
           1.0 :
           std::sin(numbers::PI * 2.0 * modulation_frequency * time +
                    phase_shift);
}

template <int spacedim>
bool
ModulatedParsedFunction<spacedim>::has_zero_modulation() const
{
  return std::abs(modulation_frequency) == 0.0;
}

template class ModulatedParsedFunction<2>;
template class ModulatedParsedFunction<3>;
