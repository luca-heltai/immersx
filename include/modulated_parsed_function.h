#ifndef reduced_lagrange_modulated_parsed_function_h
#define reduced_lagrange_modulated_parsed_function_h

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>

using namespace dealii;

template <int spacedim>
class ModulatedParsedFunction
  : public ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
{
public:
  ModulatedParsedFunction(const std::string &section_name,
                          const unsigned int n_components = 1);

  double
  scale(const double time) const;

  bool
  has_zero_modulation() const;

  double modulation_frequency = 0.0;
  double phase_shift          = 0.0;
};

#endif
