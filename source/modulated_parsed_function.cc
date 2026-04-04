#include "modulated_parsed_function.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/patterns.h>

#include <cmath>

template <int spacedim>
ModulatedParsedFunction<spacedim>::ModulatedParsedFunction(
  const std::string &section_name,
  const unsigned int n_components)
  : ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>(section_name,
                                                                n_components)
  , n_components(n_components)
  , function_expression(spacedim == 2 ? "0; 0" : "0; 0; 0")
  , variable_names(spacedim == 1 ? "x,t" :
                   spacedim == 2 ? "x,y,t" :
                                   "x,y,z,t")
{}

template <int spacedim>
void
ModulatedParsedFunction<spacedim>::declare_parameters(ParameterHandler &prm)
{
  Functions::ParsedFunction<spacedim>::declare_parameters(prm,
                                                          n_components,
                                                          function_expression);
  prm.add_parameter("Modulation frequency", modulation_frequency);
  prm.add_parameter("Phase shift", phase_shift);
}

template <int spacedim>
void
ModulatedParsedFunction<spacedim>::parse_parameters(ParameterHandler &prm)
{
  try
    {
      function_constants  = prm.get("Function constants");
      function_expression = prm.get("Function expression");
      variable_names      = prm.get("Variable names");
      Functions::ParsedFunction<spacedim>::parse_parameters(prm);
    }
  catch (const dealii::ExceptionBase &)
    {
      // Dynamically created acceptors are instantiated during the first
      // parameter pass and only declare their entries on the second pass.
      // Ignore the intermediate parse and let the second pass populate them.
    }
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
void
ModulatedParsedFunction<spacedim>::copy_configuration_from(
  const ModulatedParsedFunction<spacedim> &other)
{
  modulation_frequency = other.modulation_frequency;
  phase_shift          = other.phase_shift;
  function_constants   = other.function_constants;
  function_expression  = other.function_expression;
  variable_names       = other.variable_names;
  parse_stored_function();
}

template <int spacedim>
void
ModulatedParsedFunction<spacedim>::parse_stored_function()
{
  ParameterHandler prm;
  Functions::ParsedFunction<spacedim>::declare_parameters(prm,
                                                          n_components,
                                                          function_expression);
  prm.set("Function constants", function_constants);
  prm.set("Function expression", function_expression);
  prm.set("Variable names", variable_names);
  Functions::ParsedFunction<spacedim>::parse_parameters(prm);
}

template class ModulatedParsedFunction<1>;
template class ModulatedParsedFunction<2>;
template class ModulatedParsedFunction<3>;
