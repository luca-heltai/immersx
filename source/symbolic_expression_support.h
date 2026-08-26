#ifndef immersx_symbolic_expression_support_h
#define immersx_symbolic_expression_support_h

#include <deal.II/base/config.h>

#include <memory>
#include <string>
#include <vector>

#ifdef DEAL_II_WITH_SYMENGINE
#  include <deal.II/differentiation/sd.h>
#endif

namespace ImmersX
{
  namespace internal
  {
    void
    validate_symbolic_inputs(const std::vector<std::string> &symbol_names,
                             const std::vector<std::string> &expressions);

#ifdef DEAL_II_WITH_SYMENGINE
    namespace SD = dealii::Differentiation::SD;

    struct SymbolicOptimizerData
    {
      std::vector<SD::Expression> symbols;
      std::vector<SD::Expression> expressions;
      std::vector<double>         substitution_values;
      SD::BatchOptimizer<double>  optimizer;
    };

    std::shared_ptr<SymbolicOptimizerData>
    make_symbolic_optimizer(const std::vector<std::string> &symbol_names,
                            const std::vector<std::string> &expressions);

    std::shared_ptr<SymbolicOptimizerData>
    make_symbolic_optimizer(const std::vector<std::string>    &symbol_names,
                            const std::vector<SD::Expression> &expressions);
#endif
  } // namespace internal
} // namespace ImmersX

#endif
