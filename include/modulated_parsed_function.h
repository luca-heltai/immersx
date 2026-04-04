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

  void
  declare_parameters(ParameterHandler &prm) override;

  void
  parse_parameters(ParameterHandler &prm) override;

  double
  scale(const double time) const;

  void
  copy_configuration_from(const ModulatedParsedFunction<spacedim> &other);

  double modulation_frequency = 0.0;
  double phase_shift          = 0.0;

private:
  void
  parse_stored_function();

  unsigned int n_components;
  std::string  function_constants;
  std::string  function_expression;
  std::string  variable_names;
};

#endif
