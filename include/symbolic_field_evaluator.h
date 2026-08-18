#ifndef immersx_symbolic_field_evaluator_h
#define immersx_symbolic_field_evaluator_h

#include <deal.II/base/array_view.h>
#include <deal.II/base/point.h>

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Parse-once evaluator for scalar expressions involving coordinates, time,
 * constants, and selected reduced-field aliases.
 *
 * The SymEngine-enabled implementation uses deal.II's SD::BatchOptimizer.
 * Builds without SymEngine retain this interface and report a clear runtime
 * error when a non-empty expression set is requested. Instances are not
 * thread-safe because BatchOptimizer caches substitutions; use one instance
 * per assembly scratch object when evaluating concurrently.
 */
class SymbolicFieldEvaluator
{
public:
  SymbolicFieldEvaluator();
  ~SymbolicFieldEvaluator();
  SymbolicFieldEvaluator(SymbolicFieldEvaluator &&) noexcept;
  SymbolicFieldEvaluator &
  operator=(SymbolicFieldEvaluator &&) noexcept;
  SymbolicFieldEvaluator(const SymbolicFieldEvaluator &) = delete;
  SymbolicFieldEvaluator &
  operator=(const SymbolicFieldEvaluator &) = delete;

  void
  initialize(const std::vector<std::string>      &expression_strings,
             const std::vector<std::string>      &field_symbols,
             const std::map<std::string, double> &constants = {});

  unsigned int
  n_outputs() const;

  std::vector<double>
  evaluate(const std::vector<double> &coordinates,
           double                     time,
           const std::vector<double> &field_values) const;

  /** Evaluate into caller-owned storage to avoid per-point result allocations.
   */
  void
  evaluate_into(const std::vector<double> &coordinates,
                double                     time,
                const std::vector<double> &field_values,
                std::vector<double>       &output_values) const;

  template <int spacedim>
  std::vector<double>
  evaluate(const dealii::Point<spacedim> &point,
           double                         time,
           const std::vector<double>     &field_values) const
  {
    std::vector<double> coordinates(spacedim);
    for (unsigned int d = 0; d < spacedim; ++d)
      coordinates[d] = point[d];
    return evaluate(coordinates, time, field_values);
  }

  template <int spacedim>
  void
  evaluate_into(const dealii::Point<spacedim> &point,
                double                         time,
                const std::vector<double>     &field_values,
                std::vector<double>           &output_values) const
  {
    std::vector<double> coordinates(spacedim);
    for (unsigned int d = 0; d < spacedim; ++d)
      coordinates[d] = point[d];
    evaluate_into(coordinates, time, field_values, output_values);
  }

  template <int spacedim>
  void
  evaluate(const dealii::Point<spacedim>  &point,
           double                          time,
           dealii::ArrayView<const double> field_values,
           dealii::ArrayView<double>       output_values) const
  {
    std::vector<double> values(field_values.begin(), field_values.end());
    if (output_values.size() != n_outputs())
      throw std::runtime_error(
        "Symbolic evaluator output view has wrong size.");
    std::vector<double> result(output_values.size());
    evaluate_into(point, time, values, result);
    std::copy(result.begin(), result.end(), output_values.begin());
  }

  static bool
  available();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

#endif
