#include <immersx/core/symbolic_field_evaluator.h>

#include <algorithm>
#include <set>

#include "symbolic_expression_support.h"

namespace ImmersX
{

  struct SymbolicFieldEvaluator::Impl
  {
    std::vector<std::string>      expressions;
    std::vector<std::string>      symbols;
    std::map<std::string, double> constants;
#ifdef DEAL_II_WITH_SYMENGINE
    std::shared_ptr<internal::SymbolicOptimizerData> symengine_data;
#endif
  };

  SymbolicFieldEvaluator::SymbolicFieldEvaluator()
    : impl(std::make_unique<Impl>())
  {}

  SymbolicFieldEvaluator::~SymbolicFieldEvaluator() = default;
  SymbolicFieldEvaluator::SymbolicFieldEvaluator(
    SymbolicFieldEvaluator &&) noexcept = default;
  SymbolicFieldEvaluator &
  SymbolicFieldEvaluator::operator=(SymbolicFieldEvaluator &&) noexcept =
    default;

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
    for (const auto &[name, value] : constants)
      {
        (void)value;
        impl->symbols.push_back(name);
      }
    if (expression_strings.empty())
      {
        std::set<std::string> seen_symbols;
        for (const auto &name : impl->symbols)
          if (!seen_symbols.insert(name).second)
            throw std::runtime_error("Duplicate symbolic identifier '" + name +
                                     "'.");
        return;
      }
    internal::validate_symbolic_inputs(impl->symbols, expression_strings);
#ifndef DEAL_II_WITH_SYMENGINE
    throw std::runtime_error(
      "Symbolic expressions require deal.II built with SymEngine; symbolic evaluation is unavailable.");
#else
    impl->symengine_data =
      internal::make_symbolic_optimizer(impl->symbols, expression_strings);
#endif
  }

  unsigned int
  SymbolicFieldEvaluator::n_outputs() const
  {
    return static_cast<unsigned int>(impl ? impl->expressions.size() : 0);
  }

  std::vector<double>
  SymbolicFieldEvaluator::evaluate(
    const std::vector<double> &coordinates,
    const double               time,
    const std::vector<double> &field_values) const
  {
    std::vector<double> result(n_outputs());
    evaluate_into(coordinates, time, field_values, result);
    return result;
  }

  void
  SymbolicFieldEvaluator::evaluate_into(
    const std::vector<double> &coordinates,
    const double               time,
    const std::vector<double> &field_values,
    std::vector<double>       &output_values) const
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
        impl->symengine_data->optimizer.substitute(
          impl->symengine_data->symbols, values);
        const auto result =
          internal::evaluate_symbolic_expressions(*impl->symengine_data);
        std::copy(result.begin(), result.end(), output_values.begin());
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
} // namespace ImmersX
