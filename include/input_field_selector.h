#ifndef immersx_input_field_selector_h
#define immersx_input_field_selector_h

#include <map>
#include <string>
#include <vector>

#include "vtk_utils.h"

/** A scalar symbolic alias bound to one VTK array component. */
struct InputFieldBinding
{
  std::string  symbol_name;
  unsigned int field_index   = 0;
  unsigned int vtk_component = 0;
  unsigned int fe_component  = 0;
};

/** Parse and resolve the Representative domain Input file fields selector. */
class InputFieldSelector
{
public:
  /**
   * Resolve a comma-separated selector against a previously built catalogue.
   * The catalog order is preserved, including for wildcard expansion.
   */
  static std::vector<InputFieldBinding>
  resolve(const std::string              &selector,
          const VTKFieldCatalog          &catalog,
          const std::vector<std::string> &reserved_names = {});
};

#endif
