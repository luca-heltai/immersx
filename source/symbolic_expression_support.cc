#include "symbolic_expression_support.h"

#include <cctype>
#include <regex>
#include <set>
#include <stdexcept>
#include <utility>

namespace ImmersX
{
  namespace internal
  {
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

    void
    validate_symbolic_inputs(const std::vector<std::string> &symbol_names,
                             const std::vector<std::string> &expressions)
    {
      static const std::regex identifier_pattern("[A-Za-z_][A-Za-z0-9_]*");
      static const std::regex token_pattern("[A-Za-z_][A-Za-z0-9_]*");

      std::set<std::string> seen_symbols;
      for (const auto &name : symbol_names)
        {
          if (name.empty() || !std::regex_match(name, identifier_pattern))
            throw std::runtime_error("Invalid symbolic identifier '" + name +
                                     "'.");
          if (!seen_symbols.insert(name).second)
            throw std::runtime_error("Duplicate symbolic identifier '" + name +
                                     "'.");
        }

      std::set<std::string> allowed(seen_symbols.begin(), seen_symbols.end());
      for (const auto &text : expressions)
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
              "'. Allowed symbols are the registered symbolic variables, "
              "x, y, z, and t.");
    }

#ifdef DEAL_II_WITH_SYMENGINE
    namespace
    {
      std::shared_ptr<SymbolicOptimizerData>
      make_optimizer_data(const std::vector<std::string> &symbol_names,
                          std::vector<SD::Expression>     expressions,
                          const std::string              &parse_context)
      {
        auto data         = std::make_shared<SymbolicOptimizerData>();
        data->expressions = std::move(expressions);
        for (const auto &name : symbol_names)
          data->symbols.emplace_back(name);
        data->substitution_values.resize(data->symbols.size());
        try
          {
            data->optimizer.register_symbols(data->symbols);
            data->optimizer.register_functions(data->expressions);
            data->optimizer.optimize();
          }
        catch (const std::exception &error)
          {
            throw std::runtime_error("Failed to prepare symbolic expression '" +
                                     parse_context + "': " + error.what());
          }
        return data;
      }
    } // namespace

    std::shared_ptr<SymbolicOptimizerData>
    make_symbolic_optimizer(const std::vector<std::string> &symbol_names,
                            const std::vector<std::string> &expressions)
    {
      std::vector<SD::Expression> parsed_expressions;
      parsed_expressions.reserve(expressions.size());
      std::string current_expression;
      try
        {
          for (const auto &text : expressions)
            {
              current_expression = text;
              parsed_expressions.emplace_back(text, true);
            }
        }
      catch (const std::exception &error)
        {
          throw std::runtime_error("Failed to parse symbolic expression '" +
                                   current_expression + "': " + error.what());
        }
      return make_optimizer_data(symbol_names,
                                 std::move(parsed_expressions),
                                 current_expression);
    }

    std::shared_ptr<SymbolicOptimizerData>
    make_symbolic_optimizer(const std::vector<std::string>    &symbol_names,
                            const std::vector<SD::Expression> &expressions)
    {
      return make_optimizer_data(symbol_names, expressions, "expression set");
    }
#endif
  } // namespace internal
} // namespace ImmersX
