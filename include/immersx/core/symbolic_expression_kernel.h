#ifndef immersx_symbolic_expression_kernel_h
#define immersx_symbolic_expression_kernel_h

#include <deal.II/base/point.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ImmersX
{
  /**
   * Parse-once pointwise evaluator for one scalar symbolic expression.
   *
   * The independent symbols are ordinary scalar variables. They have no
   * finite-element or field semantics; those semantics belong to a future
   * sampling layer. The reserved coordinate symbols are @c x, @c y, @c z,
   * and @c t, with the same convention as SymbolicFieldEvaluator.
   *
   * When deal.II is built without SymEngine, the interface remains available
   * but initialization of a non-empty expression reports a clear runtime
   * error. Instances are not thread-safe because BatchOptimizer caches
   * substitutions.
   */
  class SymbolicExpressionKernel
  {
  public:
    /** @cond INTERNAL */
    /** Value and partial derivatives returned by one pointwise evaluation. */
    struct Evaluation
    {
      double              value;
      std::vector<double> derivatives;
    };
    /** @endcond */

    SymbolicExpressionKernel();
    ~SymbolicExpressionKernel();
    SymbolicExpressionKernel(SymbolicExpressionKernel &&) noexcept;
    SymbolicExpressionKernel &
    operator=(SymbolicExpressionKernel &&) noexcept;
    SymbolicExpressionKernel(const SymbolicExpressionKernel &) = delete;
    SymbolicExpressionKernel &
    operator=(const SymbolicExpressionKernel &) = delete;

    /** Initialize from an expression and an ordered list of independent
     * symbols. */
    void
    initialize(const std::string                   &expression,
               const std::vector<std::string>      &independent_symbols,
               const std::map<std::string, double> &constants = {});

    /** Evaluate the value and all partial derivatives at one point. */
    Evaluation
    evaluate(const std::vector<double> &coordinates,
             double                     time,
             const std::vector<double> &independent_values) const;

    template <int spacedim>
    Evaluation
    evaluate(const dealii::Point<spacedim> &point,
             double                         time,
             const std::vector<double>     &independent_values) const
    {
      std::vector<double> coordinates(spacedim);
      for (unsigned int d = 0; d < spacedim; ++d)
        coordinates[d] = point[d];
      return evaluate(coordinates, time, independent_values);
    }

    unsigned int
    n_independent_symbols() const;

    static bool
    available();

  private:
    /** @cond INTERNAL */
    struct Impl;
    /** @endcond */
    std::unique_ptr<Impl> impl;
  };

} // namespace ImmersX

#endif
