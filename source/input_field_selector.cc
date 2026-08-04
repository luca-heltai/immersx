#include "input_field_selector.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace
{
  std::string trim(const std::string &value)
  {
    const auto first = value.find_first_not_of(" \t\n\r");
    if (first == std::string::npos)
      return {};
    const auto last = value.find_last_not_of(" \t\n\r");
    return value.substr(first, last - first + 1);
  }

  bool valid_identifier(const std::string &value)
  {
    static const std::regex pattern("[A-Za-z_][A-Za-z0-9_]*");
    return std::regex_match(value, pattern);
  }

  std::string association_name(const VTKFieldAssociation association)
  {
    return association == VTKFieldAssociation::point_data ? "point" : "cell";
  }

  struct ParsedSelector
  {
    std::string symbol;
    std::string name;
    std::string association;
    unsigned int component = std::numeric_limits<unsigned int>::max();
  };

  ParsedSelector parse_selector(const std::string &raw)
  {
    ParsedSelector parsed;
    const auto equal = raw.find('=');
    const std::string target =
      equal == std::string::npos ? raw : trim(raw.substr(equal + 1));
    parsed.symbol = equal == std::string::npos ? target : trim(raw.substr(0, equal));
    if (parsed.symbol.empty() || !valid_identifier(parsed.symbol))
      throw std::runtime_error("Invalid Input file fields alias '" + parsed.symbol + "'.");

    std::string qualified = target;
    const auto colon = qualified.find(':');
    if (colon != std::string::npos)
      {
        parsed.association = trim(qualified.substr(0, colon));
        if (parsed.association != "point" && parsed.association != "cell")
          throw std::runtime_error("Unknown VTK field association '" + parsed.association + "'.");
        qualified = trim(qualified.substr(colon + 1));
      }
    const auto open = qualified.find('[');
    if (open != std::string::npos)
      {
        if (qualified.back() != ']' || qualified.find('[', open + 1) != std::string::npos)
          throw std::runtime_error("Invalid VTK component selector '" + target + "'.");
        const std::string index = qualified.substr(open + 1, qualified.size() - open - 2);
        if (index.empty() || !std::all_of(index.begin(), index.end(), ::isdigit))
          throw std::runtime_error("Invalid VTK component index in '" + target + "'.");
        parsed.component = static_cast<unsigned int>(std::stoul(index));
        qualified = trim(qualified.substr(0, open));
      }
    parsed.name = qualified;
    if (parsed.name.empty())
      throw std::runtime_error("Missing VTK field name in selector '" + target + "'.");
    return parsed;
  }
}

std::vector<InputFieldBinding>
InputFieldSelector::resolve(const std::string              &selector,
                           const VTKFieldCatalog          &catalog,
                           const std::vector<std::string> &reserved_names)
{
  std::vector<InputFieldBinding> bindings;
  std::set<std::string> reserved = {"x", "y", "z", "t", "pi", "E"};
  reserved.insert(reserved_names.begin(), reserved_names.end());
  std::set<std::string> aliases;

  std::vector<std::string> entries;
  std::stringstream stream(selector);
  std::string entry;
  while (std::getline(stream, entry, ','))
    if (!(entry = trim(entry)).empty())
      entries.push_back(entry);
  if (entries.empty())
    return bindings;
  if (std::find(entries.begin(), entries.end(), "*") != entries.end())
    {
      if (entries.size() != 1)
        throw std::runtime_error("Wildcard '*' cannot be combined with other Input file fields selectors.");
      for (unsigned int field = 0; field < catalog.size(); ++field)
        for (unsigned int component = 0; component < catalog[field].n_components; ++component)
          {
            const std::string alias = catalog[field].n_components == 1 ?
                                        catalog[field].vtk_name :
                                        catalog[field].vtk_name + "_" + std::to_string(component);
            if (!valid_identifier(alias))
              throw std::runtime_error("Wildcard generated invalid symbolic alias '" +
                                       alias + "' from VTK field '" +
                                       catalog[field].vtk_name + "'.");
            if (reserved.count(alias) || !aliases.insert(alias).second)
              throw std::runtime_error("Wildcard generated reserved or duplicate alias '" + alias + "'.");
            bindings.push_back({alias, field, component,
                                catalog[field].first_fe_component + component});
          }
      return bindings;
    }

  for (const auto &raw : entries)
    {
      const ParsedSelector parsed = parse_selector(raw);
      if (reserved.count(parsed.symbol) || !aliases.insert(parsed.symbol).second)
        throw std::runtime_error("Duplicate or reserved Input file fields alias '" + parsed.symbol + "'.");
      std::vector<unsigned int> matches;
      for (unsigned int field = 0; field < catalog.size(); ++field)
        if (catalog[field].vtk_name == parsed.name &&
            (parsed.association.empty() ||
             association_name(catalog[field].association) == parsed.association))
          matches.push_back(field);
      if (matches.empty())
        throw std::runtime_error("Requested VTK field '" + parsed.name + "' was not found.");
      if (matches.size() > 1)
        throw std::runtime_error("VTK field '" + parsed.name + "' is ambiguous; use point: or cell:.");
      const auto field = matches.front();
      if (parsed.component == std::numeric_limits<unsigned int>::max())
        {
          if (catalog[field].n_components != 1)
            throw std::runtime_error("Vector VTK field '" + parsed.name + "' requires an explicit component.");
          bindings.push_back({parsed.symbol, field, 0, catalog[field].first_fe_component});
        }
      else
        {
          if (parsed.component >= catalog[field].n_components)
            throw std::runtime_error("Component index " + std::to_string(parsed.component) +
                                     " is out of range for VTK field '" + parsed.name + "'.");
          bindings.push_back({parsed.symbol, field, parsed.component,
                              catalog[field].first_fe_component + parsed.component});
        }
    }
  return bindings;
}
