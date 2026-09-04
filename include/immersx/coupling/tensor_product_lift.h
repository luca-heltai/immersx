// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_tensor_product_lift_h
#define immersx_tensor_product_lift_h

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_selector.h>

#include <immersx/coupling/reference_cross_section.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ImmersX
{
  template <int surface_dim, int spacedim>
  struct TensorProductLiftPoint;

  namespace detail
  {
    inline std::string
    normalize_lift_subsection(const std::string &subsection)
    {
      if (subsection.empty())
        return "/Tensor product lift/";

      std::string result = subsection;
      if (result.front() != '/')
        result.insert(result.begin(), '/');
      if (result.back() != '/')
        result.push_back('/');
      return result;
    }

    /**
     * Build the iterated reduced-domain quadrature rule shared by the legacy
     * TensorProductSpace and the modern tensor-product lift.
     */
    template <int dim>
    dealii::Quadrature<dim>
    make_tensor_product_quadrature(const std::string &quadrature_type,
                                   const unsigned int n_points,
                                   const unsigned int repetitions)
    {
      AssertThrow(repetitions > 0,
                  dealii::ExcMessage(
                    "Tensor-product quadrature repetitions must be positive."));
      return dealii::QIterated<dim>(
        dealii::QuadratureSelector<1>(quadrature_type, n_points), repetitions);
    }

    /**
     * Expand one representative quadrature point into the lifted cross-section
     * points, weights, and selected-mode values.
     *
     * Both the legacy TensorProductSpace and the modern TensorProductLift
     * compute the actual lifting through this function, so the transformed
     * cross-section quadrature, weight scaling, section measure, selected-mode
     * lookup, and representative/section indexing are defined once.
     */
    template <int surface_dim, int spacedim, typename Section>
    std::vector<TensorProductLiftPoint<surface_dim, spacedim>>
    transform_representative_point(
      const Section                     &section,
      const std::vector<unsigned int>   &selected_modes,
      const dealii::Point<spacedim>     &origin,
      const dealii::Tensor<1, spacedim> &tangent,
      const double                       representative_weight,
      const double                       thickness,
      const unsigned int                 representative_qpoint,
      const std::vector<double>         &representative_basis)
    {
      const auto transformed =
        section.get_transformed_quadrature(origin, tangent, thickness);
      std::vector<TensorProductLiftPoint<surface_dim, spacedim>> result;
      result.reserve(transformed.size());
      for (unsigned int section_q = 0; section_q < transformed.size();
           ++section_q)
        {
          TensorProductLiftPoint<surface_dim, spacedim> point;
          point.point                = transformed.point(section_q);
          point.representative_point = origin;
          point.weight = transformed.weight(section_q) * representative_weight;
          point.representative_qpoint = representative_qpoint;
          point.section_qpoint        = section_q;
          point.selected_modes        = selected_modes;
          point.mode_values =
            section.get_transformed_mode_values(section_q, tangent);

          if (!representative_basis.empty())
            {
              point.tensor_product_basis_values.resize(
                representative_basis.size() * point.mode_values.size());
              unsigned int index = 0;
              for (const auto mode_value : point.mode_values)
                for (const auto representative_value : representative_basis)
                  point.tensor_product_basis_values[index++] =
                    representative_value * mode_value;
            }
          result.emplace_back(std::move(point));
        }
      return result;
    }
  } // namespace detail

  /**
   * Source-side thickness evaluation contract for the modern lift.
   *
   * Provides the thickness at one representative point, given the evaluation
   * time and any named source properties the source can resolve. The lift
   * never owns imported fields: a source installs a provider that evaluates
   * its own properties.
   */
  template <int spacedim>
  using SourceThicknessEvaluator =
    std::function<double(const dealii::Point<spacedim> &point,
                         double                         time,
                         const std::vector<double>     &source_properties)>;

  /** Lifting-only configuration for a tensor-product representation. */
  template <int reduced_dim,
            int surface_dim,
            int spacedim,
            int n_components = 1>
  struct TensorProductLiftParameters : public dealii::ParameterAcceptor
  {
    static_assert(surface_dim > reduced_dim,
                  "A tensor-product lift must increase support dimension.");
    static constexpr int cross_section_dim = surface_dim - reduced_dim;

    explicit TensorProductLiftParameters(
      const std::string &subsection = "/Tensor product lift/")
      : ParameterAcceptor(detail::normalize_lift_subsection(subsection))
      , section(detail::normalize_lift_subsection(subsection) +
                "Cross section/")
    {
      add_parameter("Thickness", thickness);
      add_parameter("Constant thickness", constant_thickness);

      enter_subsection("Representative quadrature");
      add_parameter("Type", representative_quadrature_type);
      add_parameter("Number of points", representative_n_q_points);
      add_parameter("Number of repetitions", representative_n_repetitions);
      leave_subsection();

      add_parameter("Representative quadrature type",
                    representative_quadrature_type,
                    "Deprecated flat alias for the nested setting.",
                    this->prm,
                    Patterns::Selection(
                      dealii::QuadratureSelector<1>::get_quadrature_names()));
    }

    /** Thickness expression; a numeric string is the modern default path. */
    std::string thickness = "0.01";

    /** Fallback used when the thickness expression is constant. */
    double constant_thickness = 0.01;

    std::string  representative_quadrature_type = "gauss";
    unsigned int representative_n_q_points      = 0;
    unsigned int representative_n_repetitions   = 1;

    ReferenceCrossSectionParameters<cross_section_dim, spacedim, n_components>
      section;

    const TensorProductLiftParameters &
    parameters() const
    {
      return *this;
    }
  };

  /** User-facing descriptor for a parameterized tensor-product lift. */
  template <int reduced_dim,
            int surface_dim,
            int spacedim,
            int n_components = 1>
  class TensorProductLift : public TensorProductLiftParameters<reduced_dim,
                                                               surface_dim,
                                                               spacedim,
                                                               n_components>
  {
  public:
    using Parameters = TensorProductLiftParameters<reduced_dim,
                                                   surface_dim,
                                                   spacedim,
                                                   n_components>;

    explicit TensorProductLift(
      const std::string &subsection = "/Tensor product lift/")
      : Parameters(subsection)
    {}

    const Parameters &
    parameters() const
    {
      return *this;
    }
  };

  /** A point generated by the lifting-side tensor-product support. */
  template <int surface_dim, int spacedim>
  struct TensorProductLiftPoint
  {
    dealii::Point<spacedim>          point;
    dealii::Point<spacedim>          representative_point;
    double                           weight = 0.;
    dealii::types::global_cell_index source_entity_id =
      dealii::numbers::invalid_unsigned_int;
    std::uint64_t stable_id = std::numeric_limits<std::uint64_t>::max();
    /** Index in the source-owned quadrature vector used for evaluation. */
    unsigned int representative_qpoint = 0;
    /** Quadrature slot within the source representative entity. */
    unsigned int source_representative_qpoint =
      dealii::numbers::invalid_unsigned_int;
    unsigned int              section_qpoint = 0;
    std::vector<unsigned int> selected_modes;
    /**
     * Selected mode values. Scalar modes remain scalar; vector mode values
     * with n_components == spacedim are expressed in the physical frame
     * associated with the representative tangent.
     */
    std::vector<double>                          mode_values;
    std::vector<double>                          tensor_product_basis_values;
    std::vector<dealii::types::global_dof_index> source_dof_indices;
    std::vector<double>                          source_basis_values;
  };

  /** Shared lifting-side geometry, quadrature, and modal support. */
  template <int reduced_dim,
            int surface_dim,
            int spacedim,
            int n_components = 1>
  class TensorProductLiftSupport
  {
  public:
    static constexpr int cross_section_dim = surface_dim - reduced_dim;
    using Parameters = TensorProductLiftParameters<reduced_dim,
                                                   surface_dim,
                                                   spacedim,
                                                   n_components>;
    using Section =
      ReferenceCrossSection<cross_section_dim, spacedim, n_components>;
    using Point = TensorProductLiftPoint<surface_dim, spacedim>;

    explicit TensorProductLiftSupport(const Parameters &parameters)
      : parameters_(parameters)
      , section_(std::make_shared<Section>(parameters_.section))
      , representative_quadrature_(make_representative_quadrature())
    {
      AssertThrow(parameters_.representative_n_repetitions > 0,
                  dealii::ExcMessage(
                    "Representative quadrature repetitions must be positive."));
      AssertThrow(parameters_.section.n_quadrature_repetitions > 0,
                  dealii::ExcMessage("Cross-section quadrature repetitions "
                                     "must be positive."));
    }

    const Section &
    reference_cross_section() const
    {
      return *section_;
    }

    const dealii::Quadrature<reduced_dim> &
    representative_quadrature() const
    {
      return representative_quadrature_;
    }

    double
    section_measure(const double scale = 1.) const
    {
      return section_->measure(scale);
    }

    const std::vector<unsigned int> &
    selected_modes() const
    {
      return parameters_.section.selected_coefficients;
    }

    /** Install a source-side provider for symbolic thickness expressions. */
    void
    set_thickness_evaluator(SourceThicknessEvaluator<spacedim> evaluator)
    {
      thickness_evaluator_ = std::move(evaluator);
    }

    std::vector<Point>
    transform(const dealii::Point<spacedim>     &origin,
              const dealii::Tensor<1, spacedim> &tangent,
              const double                       representative_weight,
              const double                       thickness,
              const unsigned int                 representative_qpoint,
              const std::vector<double> &representative_basis = {}) const
    {
      return detail::transform_representative_point<surface_dim, spacedim>(
        *section_,
        parameters_.section.selected_coefficients,
        origin,
        tangent,
        representative_weight,
        thickness,
        representative_qpoint,
        representative_basis);
    }

    /** Apply the common reference-section transformation to legacy users. */
    static dealii::Quadrature<spacedim>
    transform_section(const Section                     &section,
                      const dealii::Point<spacedim>     &origin,
                      const dealii::Tensor<1, spacedim> &tangent,
                      const double                       thickness)
    {
      return section.get_transformed_quadrature(origin, tangent, thickness);
    }

    /**
     * Evaluate a numeric modern thickness expression.
     *
     * Symbolic expressions require a source-side evaluator; use the
     * point-dependent overload once one is installed.
     */
    double
    thickness() const
    {
      AssertThrow(has_constant_thickness(),
                  dealii::ExcMessage(
                    "The modern tensor-product lift thickness expression '" +
                    parameters_.thickness +
                    "' requires a source-property evaluator; install one with "
                    "set_thickness_evaluator() or use a constant thickness."));
      return constant_thickness_value();
    }

    /**
     * Evaluate the thickness at one representative point.
     *
     * Constant expressions return the parsed value; symbolic expressions are
     * delegated to the installed source-side evaluator.
     */
    double
    thickness(const dealii::Point<spacedim> &point, const double time) const
    {
      if (has_constant_thickness())
        return constant_thickness_value();
      AssertThrow(thickness_evaluator_,
                  dealii::ExcMessage(
                    "The modern tensor-product lift thickness expression '" +
                    parameters_.thickness +
                    "' requires a source-property evaluator; install one with "
                    "set_thickness_evaluator() or use a constant thickness."));
      const double value = thickness_evaluator_(point, time, {});
      AssertThrow(std::isfinite(value) && value > 0.,
                  dealii::ExcMessage(
                    "Lift thickness must be finite and positive."));
      return value;
    }

  private:
    bool
    has_constant_thickness() const
    {
      parse_constant_thickness();
      return constant_is_parsed_;
    }

    double
    constant_thickness_value() const
    {
      parse_constant_thickness();
      AssertThrow(constant_is_parsed_, dealii::ExcInternalError());
      return constant_thickness_value_;
    }

    void
    parse_constant_thickness() const
    {
      if (constant_thickness_parsed_)
        return;
      constant_thickness_parsed_ = true;
      std::size_t consumed       = 0;
      try
        {
          constant_thickness_value_ =
            std::stod(parameters_.thickness, &consumed);
        }
      catch (const std::exception &)
        {
          return;
        }
      constant_is_parsed_ = consumed == parameters_.thickness.size() &&
                            std::isfinite(constant_thickness_value_) &&
                            constant_thickness_value_ > 0.;
    }

    dealii::Quadrature<reduced_dim>
    make_representative_quadrature() const
    {
      const unsigned int n_points =
        parameters_.representative_n_q_points == 0 ?
          2 * parameters_.section.inclusion_degree + 1 :
          parameters_.representative_n_q_points;
      return detail::make_tensor_product_quadrature<reduced_dim>(
        parameters_.representative_quadrature_type,
        n_points,
        parameters_.representative_n_repetitions);
    }

    const Parameters                  &parameters_;
    std::shared_ptr<Section>           section_;
    dealii::Quadrature<reduced_dim>    representative_quadrature_;
    SourceThicknessEvaluator<spacedim> thickness_evaluator_;
    mutable bool                       constant_thickness_parsed_ = false;
    mutable bool                       constant_is_parsed_        = false;
    mutable double                     constant_thickness_value_  = 0.;
  };
} // namespace ImmersX

#endif // immersx_tensor_product_lift_h
