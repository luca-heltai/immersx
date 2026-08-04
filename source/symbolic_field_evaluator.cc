#include "symbolic_field_evaluator.h"

#include <cctype>
#include <cmath>
#include <regex>
#include <set>
#include <sstream>

#ifdef DEAL_II_WITH_SYMENGINE
#  include <deal.II/differentiation/sd.h>
#endif

#ifdef DEAL_II_WITH_SYMENGINE
namespace
{
  namespace SD = dealii::Differentiation::SD;
  struct SymEngineData
  {
    std::vector<SD::Expression>   symbols;
    std::vector<SD::Expression>   expressions;
    std::map<std::string, double> constants;
    std::vector<double>           substitution_values;
    SD::BatchOptimizer<double>    optimizer;
  };
} // namespace
#endif

struct SymbolicFieldEvaluator::Impl
{
  std::vector<std::string>      expressions;
  std::vector<std::string>      symbols;
  std::map<std::string, double> constants;
#ifdef DEAL_II_WITH_SYMENGINE
  std::shared_ptr<SymEngineData> symengine_data;
#endif
};

#ifdef DEAL_II_WITH_SYMENGINE
namespace
{
  bool
  is_builtin(const std::string &name)
  {
    static const std::set<std::string> builtins = {"sin",
                                                   "cos",
                                                   "tan",
                                                   "asin",
                                                   "acos",
                                                   "atan",
                                                   "sinh",
                                                   "cosh",
                                                   "tanh",
                                                   "exp",
                                                   "log",
                                                   "log10",
                                                   "sqrt",
                                                   "abs",
                                                   "min",
                                                   "max",
                                                   "floor",
                                                   "ceil",
                                                   "pow"};
    return builtins.count(name) != 0;
  }

  bool
  is_numeric_exponent(const std::string &text,
                      const std::size_t  position,
                      const std::size_t  length)
  {
    if (length != 1 || (text[position] != 'e' && text[position] != 'E') ||
        position == 0 ||
        !std::isdigit(static_cast<unsigned char>(text[position - 1])))
      return false;
    std::size_t next = position + length;
    if (next < text.size() && (text[next] == '+' || text[next] == '-'))
      ++next;
    return next < text.size() &&
           std::isdigit(static_cast<unsigned char>(text[next]));
  }
} // namespace
#endif

SymbolicFieldEvaluator::SymbolicFieldEvaluator()
  : impl(std::make_unique<Impl>())
{}

SymbolicFieldEvaluator::~SymbolicFieldEvaluator() = default;
SymbolicFieldEvaluator::SymbolicFieldEvaluator(
  SymbolicFieldEvaluator &&) noexcept = default;
SymbolicFieldEvaluator &
SymbolicFieldEvaluator::operator=(SymbolicFieldEvaluator &&) noexcept = default;

void
SymbolicFieldEvaluator::initialize(
  const std::vector<std::string>      &expression_strings,
  const std::vector<std::string>      &field_symbols,
  const std::map<std::string, double> &constants)
{
  impl              = std::make_unique<Impl>();
  impl->expressions = expression_strings;
  impl->constants   = constants;
  impl->symbols     = {"x", "y", "z", "t"};
  impl->symbols.insert(impl->symbols.end(),
                       field_symbols.begin(),
                       field_symbols.end());
  std::set<std::string> seen_symbols;
  for (const auto &name : impl->symbols)
    if (!seen_symbols.insert(name).second)
      throw std::runtime_error("Duplicate symbolic identifier '" + name + "'.");
  for (const auto &[name, value] : constants)
    {
      (void)value;
      if (!seen_symbols.insert(name).second)
        throw std::runtime_error(
          "Constant collides with symbolic identifier '" + name + "'.");
      impl->symbols.push_back(name);
    }
  if (expression_strings.empty())
    return;
#ifndef DEAL_II_WITH_SYMENGINE
  throw std::runtime_error(
    "Symbolic expressions require deal.II built with SymEngine; symbolic evaluation is unavailable.");
#else
  for (const auto &name : impl->symbols)
    if (name.empty() ||
        !std::regex_match(name, std::regex("[A-Za-z_][A-Za-z0-9_]*")))
      throw std::runtime_error("Invalid symbolic identifier '" + name + "'.");
  std::set<std::string> allowed(impl->symbols.begin(), impl->symbols.end());
  for (const auto &text : expression_strings)
    {
      static const std::regex token_pattern("[A-Za-z_][A-Za-z0-9_]*");
      for (std::sregex_iterator it(text.begin(), text.end(), token_pattern),
           end;
           it != end;
           ++it)
        if (!allowed.count(it->str()) && !is_builtin(it->str()) &&
            !is_numeric_exponent(text,
                                 static_cast<std::size_t>(it->position()),
                                 static_cast<std::size_t>(it->length())))
          throw std::runtime_error(
            "Unknown symbol '" + it->str() + "' in expression '" + text +
            "'. Allowed symbols: x, y, z, t and registered fields/constants.");
    }

  auto data       = std::make_shared<SymEngineData>();
  data->constants = constants;
  for (const auto &name : impl->symbols)
    data->symbols.emplace_back(name);
  data->substitution_values.resize(data->symbols.size());
  std::string current_expression;
  try
    {
      for (const auto &text : expression_strings)
        {
          current_expression = text;
          data->expressions.emplace_back(text, true);
        }
      data->optimizer.register_symbols(data->symbols);
      data->optimizer.register_functions(data->expressions);
      data->optimizer.optimize();
    }
  catch (const std::exception &error)
    {
      throw std::runtime_error("Failed to parse symbolic expression '" +
                               current_expression + "': " + error.what());
    }
  impl->symengine_data = std::move(data);
#endif
}

unsigned int
SymbolicFieldEvaluator::n_outputs() const
{
  return static_cast<unsigned int>(impl ? impl->expressions.size() : 0);
}

std::vector<double>
SymbolicFieldEvaluator::evaluate(const std::vector<double> &coordinates,
                                 const double               time,
                                 const std::vector<double> &field_values) const
{
  std::vector<double> result(n_outputs());
  evaluate_into(coordinates, time, field_values, result);
  return result;
}

void
SymbolicFieldEvaluator::evaluate_into(const std::vector<double> &coordinates,
                                      const double               time,
                                      const std::vector<double> &field_values,
                                      std::vector<double> &output_values) const
{
  if (!impl || impl->expressions.empty())
    {
      if (!output_values.empty())
        throw std::runtime_error(
          "Symbolic evaluator output buffer must be empty for no expressions.");
      return;
    }
  if (output_values.size() != impl->expressions.size())
    throw std::runtime_error(
      "Symbolic evaluator output buffer has wrong size for " +
      std::to_string(impl->expressions.size()) + " expressions.");
#ifndef DEAL_II_WITH_SYMENGINE
  (void)coordinates;
  (void)time;
  (void)field_values;
  throw std::runtime_error(
    "Symbolic evaluation is unavailable without SymEngine.");
#else
  const auto expected_fields =
    impl->symbols.size() - 4 - impl->constants.size();
  if (field_values.size() != expected_fields)
    throw std::runtime_error(
      "Symbolic evaluator received " + std::to_string(field_values.size()) +
      " field values, expected " + std::to_string(expected_fields) + " for " +
      std::to_string(impl->expressions.size()) + " expression(s).");
  auto &values = impl->symengine_data->substitution_values;
  std::fill(values.begin(), values.end(), 0.0);
  for (unsigned int d = 0; d < std::min<unsigned int>(3, coordinates.size());
       ++d)
    values[d] = coordinates[d];
  values[3] = time;
  std::copy(field_values.begin(), field_values.end(), values.begin() + 4);
  unsigned int offset = 4 + static_cast<unsigned int>(field_values.size());
  for (const auto &[name, value] : impl->constants)
    {
      (void)name;
      values[offset++] = value;
    }
  try
    {
      impl->symengine_data->optimizer.substitute(impl->symengine_data->symbols,
                                                 values);
      const auto &result = impl->symengine_data->optimizer.evaluate();
      for (unsigned int i = 0; i < result.size(); ++i)
        {
          if (!std::isfinite(result[i]))
            throw std::runtime_error("expression " + std::to_string(i) +
                                     " evaluated to a non-finite value");
          output_values[i] = result[i];
        }
    }
  catch (const std::exception &error)
    {
      throw std::runtime_error("Failed evaluating symbolic expression set (" +
                               std::to_string(impl->expressions.size()) +
                               " expressions): " + error.what());
    }
#endif
}

bool
SymbolicFieldEvaluator::available()
{
#ifdef DEAL_II_WITH_SYMENGINE
  return true;
#else
  return false;
#endif
}
