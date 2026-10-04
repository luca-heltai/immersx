#ifndef immersx_modulated_parsed_function_h
#define immersx_modulated_parsed_function_h

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>

#include <memory>

namespace ImmersX
{
  using namespace dealii;

  template <int spacedim>
  class ModulatedParsedFunction
    : public ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>,
      public std::enable_shared_from_this<ModulatedParsedFunction<spacedim>>
  {
  public:
    using FunctionHandle = std::shared_ptr<const dealii::Function<spacedim>>;

    ModulatedParsedFunction(const std::string &section_name,
                            const unsigned int n_components = 1);

    void
    declare_parameters(ParameterHandler &prm) override;

    void
    parse_parameters(ParameterHandler &prm) override;

    double
    scale(const double time) const;

    FunctionHandle
    function_handle() const;

    void
    copy_configuration_from(const ModulatedParsedFunction<spacedim> &other);

    void
    set_fallback_configuration_source(
      const ModulatedParsedFunction<spacedim> *source);

    double modulation_frequency = 0.0;
    double phase_shift          = 0.0;

  private:
    void
    parse_stored_function();

    void
    inherit_missing_configuration_from(
      const ModulatedParsedFunction<spacedim> &source);

    void
    reset_explicit_flags();

    unsigned int                             n_components;
    std::string                              function_constants;
    std::string                              function_expression;
    std::string                              variable_names;
    const ModulatedParsedFunction<spacedim> *fallback_source = nullptr;
    bool function_constants_explicitly_set                   = false;
    bool function_expression_explicitly_set                  = false;
    bool variable_names_explicitly_set                       = false;
    bool modulation_frequency_explicitly_set                 = false;
    bool phase_shift_explicitly_set                          = false;
  };

} // namespace ImmersX

#endif
