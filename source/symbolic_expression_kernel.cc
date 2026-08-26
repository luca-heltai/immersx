#include <immersx/core/symbolic_expression_kernel.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "symbolic_expression_support.h"

#ifdef DEAL_II_WITH_SYMENGINE
namespace ImmersX
{
  namespace SD = dealii::Differentiation::SD;
}
#endif

namespace ImmersX
{
  struct SymbolicExpressionKernel::Impl
  {
    std::string                   expression;
    std::vector<std::string>      independent_symbols;
    std::vector<std::string>      symbol_names;
    std::map<std::string, double> constants;
#ifdef DEAL_II_WITH_SYMENGINE
    std::shared_ptr<internal::SymbolicOptimizerData> optimizer_data;
#endif
  };

  SymbolicExpressionKernel::SymbolicExpressionKernel()
    : impl(std::make_unique<Impl>())
  {}

  SymbolicExpressionKernel::~SymbolicExpressionKernel() = default;
  SymbolicExpressionKernel::SymbolicExpressionKernel(
    SymbolicExpressionKernel &&) noexcept = default;
  SymbolicExpressionKernel &
  SymbolicExpressionKernel::operator=(SymbolicExpressionKernel &&) noexcept =
    default;

  void
  SymbolicExpressionKernel::initialize(
    const std::string                   &expression,
    const std::vector<std::string>      &independent_symbols,
    const std::map<std::string, double> &constants)
  {
    auto new_impl                 = std::make_unique<Impl>();
    new_impl->expression          = expression;
    new_impl->independent_symbols = independent_symbols;
    new_impl->constants           = constants;
    new_impl->symbol_names        = {"x", "y", "z", "t"};
    new_impl->symbol_names.insert(new_impl->symbol_names.end(),
                                  independent_symbols.begin(),
                                  independent_symbols.end());
    for (const auto &[name, value] : constants)
      {
        (void)value;
        new_impl->symbol_names.push_back(name);
      }

    internal::validate_symbolic_inputs(new_impl->symbol_names, {expression});
#ifndef DEAL_II_WITH_SYMENGINE
    throw std::runtime_error(
      "Symbolic expressions require deal.II built with SymEngine; symbolic evaluation is unavailable.");
#else
    SD::Expression output;
    try
      {
        output = SD::Expression(expression, true);
      }
    catch (const std::exception &error)
      {
        throw std::runtime_error("Failed to parse symbolic expression '" +
                                 expression + "': " + error.what());
      }

    std::vector<SD::Expression> expressions = {output};
    expressions.reserve(1 + independent_symbols.size());
    for (const auto &name : independent_symbols)
      expressions.push_back(SD::differentiate(output, SD::Expression(name)));
    new_impl->optimizer_data =
      internal::make_symbolic_optimizer(new_impl->symbol_names, expressions);
    impl         = std::move(new_impl);
#endif
  }

  SymbolicExpressionKernel::Evaluation
  SymbolicExpressionKernel::evaluate(
    const std::vector<double> &coordinates,
    const double               time,
    const std::vector<double> &independent_values) const
  {
    if (!impl || impl->expression.empty())
      throw std::runtime_error(
        "Symbolic expression kernel is not initialized.");
    if (independent_values.size() != impl->independent_symbols.size())
      throw std::runtime_error(
        "Symbolic expression kernel received " +
        std::to_string(independent_values.size()) +
        " independent values, expected " +
        std::to_string(impl->independent_symbols.size()) + ".");
#ifndef DEAL_II_WITH_SYMENGINE
    (void)coordinates;
    (void)time;
    throw std::runtime_error(
      "Symbolic evaluation is unavailable without SymEngine.");
#else
    auto &values = impl->optimizer_data->substitution_values;
    std::fill(values.begin(), values.end(), 0.0);
    for (std::size_t d = 0; d < std::min<std::size_t>(3, coordinates.size());
         ++d)
      values[d] = coordinates[d];
    values[3] = time;
    std::copy(independent_values.begin(),
              independent_values.end(),
              values.begin() + 4);
    std::size_t offset = 4 + independent_values.size();
    for (const auto &[name, value] : impl->constants)
      {
        (void)name;
        values[offset++] = value;
      }

    try
      {
        impl->optimizer_data->optimizer.substitute(
          impl->optimizer_data->symbols, values);
        const auto &evaluated = impl->optimizer_data->optimizer.evaluate();
        for (const double value : evaluated)
          if (!std::isfinite(value))
            throw std::runtime_error(
              "symbolic expression evaluated to a non-finite value");

        SymbolicExpressionKernel::Evaluation result;
        result.value = evaluated[0];
        result.derivatives.assign(evaluated.begin() + 1, evaluated.end());
        return result;
      }
    catch (const std::exception &error)
      {
        throw std::runtime_error("Failed evaluating symbolic expression: " +
                                 std::string(error.what()));
      }
#endif
  }

  unsigned int
  SymbolicExpressionKernel::n_independent_symbols() const
  {
    return static_cast<unsigned int>(impl ? impl->independent_symbols.size() :
                                            0);
  }

  bool
  SymbolicExpressionKernel::available()
  {
#ifdef DEAL_II_WITH_SYMENGINE
    return true;
#else
    return false;
#endif
  }
} // namespace ImmersX
