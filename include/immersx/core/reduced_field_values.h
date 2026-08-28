#ifndef immersx_reduced_field_values_h
#define immersx_reduced_field_values_h

#include <deal.II/base/array_view.h>
#include <deal.II/base/point.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <immersx/core/input_field_selector.h>
#include <immersx/core/symbolic_field_evaluator.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ImmersX
{
  /**
   * Extracts selected imported property components in symbolic-binding order.
   *
   * The FEValues object, extractors, and scratch buffers are all constructed
   * once.  Calling extract() only reinitializes FEValues and copies numeric
   * values; it never performs a field-name lookup.
   */
  template <int dim, int spacedim>
  class ReducedFieldValues
  {
  public:
    using CellIterator =
      typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator;

    ReducedFieldValues(
      const dealii::DoFHandler<dim, spacedim>                  &dof_handler,
      const dealii::Quadrature<dim>                            &quadrature,
      const dealii::LinearAlgebra::distributed::Vector<double> &properties,
      const std::vector<InputFieldBinding>                     &bindings)
      : properties(properties)
      , fe_values(dof_handler.get_fe(), quadrature, dealii::update_values)
      , n_quadrature_points(quadrature.size())
    {
      std::vector<unsigned int> unique_components;
      component_slots.reserve(bindings.size());
      for (const auto &binding : bindings)
        {
          const auto   found = std::find(unique_components.begin(),
                                       unique_components.end(),
                                       binding.fe_component);
          unsigned int slot  = 0;
          if (found == unique_components.end())
            {
              slot = static_cast<unsigned int>(unique_components.size());
              unique_components.push_back(binding.fe_component);
              extractors.emplace_back(binding.fe_component);
              component_values.emplace_back(n_quadrature_points);
            }
          else
            slot = static_cast<unsigned int>(found - unique_components.begin());
          component_slots.push_back(slot);
        }
    }

    /** Fill values[q*bindings.size() + binding] for one property cell. */
    void
    extract(const CellIterator &cell, std::vector<double> &values)
    {
      if (values.size() != n_quadrature_points * component_slots.size())
        throw std::runtime_error("Reduced field value buffer has wrong size.");
      if (component_slots.empty())
        return;

      fe_values.reinit(cell);
      for (unsigned int slot = 0; slot < extractors.size(); ++slot)
        fe_values[extractors[slot]].get_function_values(properties,
                                                        component_values[slot]);

      for (unsigned int q = 0; q < n_quadrature_points; ++q)
        for (unsigned int binding = 0; binding < component_slots.size();
             ++binding)
          values[q * component_slots.size() + binding] =
            component_values[component_slots[binding]][q];
    }

  private:
    const dealii::LinearAlgebra::distributed::Vector<double> &properties;
    dealii::FEValues<dim, spacedim>                           fe_values;
    unsigned int                                    n_quadrature_points;
    std::vector<dealii::FEValuesExtractors::Scalar> extractors;
    std::vector<unsigned int>                       component_slots;
    std::vector<std::vector<double>>                component_values;
  };

  /** Evaluate a scalar thickness expression for one reduced cell. */
  template <int dim, int spacedim>
  void
  evaluate_thickness_values(
    const SymbolicFieldEvaluator &evaluator,
    const std::string            &expression,
    const typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator
                                               &cell,
    const std::vector<dealii::Point<spacedim>> &qpoints,
    const std::vector<double>                  &field_values,
    const double                                constant_thickness,
    const double                                time,
    std::vector<double>                        &thickness_values)
  {
    if (thickness_values.size() != qpoints.size())
      throw std::runtime_error("Thickness value buffer has wrong size.");
    const unsigned int n_fields =
      qpoints.empty() ?
        0 :
        static_cast<unsigned int>(field_values.size() / qpoints.size());
    if (qpoints.size() * n_fields != field_values.size())
      throw std::runtime_error(
        "Reduced field values do not match quadrature size.");

    std::vector<double> fields(n_fields);
    std::vector<double> result(1);
    for (unsigned int q = 0; q < qpoints.size(); ++q)
      {
        double value = constant_thickness;
        if (evaluator.n_outputs() != 0)
          {
            std::copy(field_values.begin() + q * n_fields,
                      field_values.begin() + (q + 1) * n_fields,
                      fields.begin());
            evaluator.evaluate_into(qpoints[q], time, fields, result);
            value = result[0];
          }
        if (!std::isfinite(value) || value <= 0.)
          {
            std::ostringstream diagnostic;
            diagnostic
              << "Invalid thickness in expression '" << expression
              << "' at cell " << cell->global_active_cell_index()
              << ", quadrature point " << q
              << ": value must be finite and positive; input values = [";
            for (unsigned int field = 0; field < n_fields; ++field)
              {
                if (field != 0)
                  diagnostic << ", ";
                diagnostic << field_values[q * n_fields + field];
              }
            diagnostic << "].";
            throw std::runtime_error(diagnostic.str());
          }
        thickness_values[q] = value;
      }
  }

} // namespace ImmersX

#endif
