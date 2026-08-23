// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_lifting_h
#define immersx_lifting_h

#include <deal.II/base/point.h>

#include <immersx/core/representation.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace ImmersX
{
  /** A geometry-only map from target points to source parameters. */
  class ParametricGeometryMap
  {
  public:
    using Point        = dealii::Point<3>;
    using ParameterMap = std::function<double(const Point &)>;

    ParametricGeometryMap(std::vector<double>        source_parameters,
                          const RepresentationDomain target_domain)
      : ParametricGeometryMap(std::move(source_parameters),
                              target_domain,
                              default_source_parameter)
    {}

    ParametricGeometryMap(std::vector<double>        source_parameters,
                          const RepresentationDomain target_domain,
                          ParameterMap               source_parameter)
      : source_parameters_(std::move(source_parameters))
      , target_domain_(target_domain)
      , source_parameter_(std::move(source_parameter))
    {
      AssertThrow(!source_parameters_.empty(),
                  dealii::ExcMessage(
                    "A geometry map requires source parameters."));
      AssertThrow(
        std::is_sorted(source_parameters_.begin(), source_parameters_.end()),
        dealii::ExcMessage("Geometry-map source parameters must be sorted."));
      AssertThrow(std::adjacent_find(source_parameters_.begin(),
                                     source_parameters_.end()) ==
                    source_parameters_.end(),
                  dealii::ExcMessage(
                    "Geometry-map source parameters must be distinct."));
    }

    const RepresentationDomain &
    domain() const
    {
      return target_domain_;
    }

    const std::vector<double> &
    source_parameters() const
    {
      return source_parameters_;
    }

    double
    source_parameter(const Point &target_point) const
    {
      return source_parameter_(target_point);
    }

  private:
    static double
    default_source_parameter(const Point &point)
    {
      return point[0];
    }

    std::vector<double>  source_parameters_;
    RepresentationDomain target_domain_;
    ParameterMap         source_parameter_;
  };

  /**
   * A primal-to-primal transfer between physical quantity value spaces.
   *
   * This object knows vector storage because it implements the transfer, but
   * it knows no Fields, Problems, residuals, or test functions.
   */
  template <typename SourceVectorType, typename TargetVectorType>
  class ValueTransfer
  {
  public:
    using Operator = RepresentationOperator<TargetVectorType, SourceVectorType>;

    struct Weight
    {
      double       lower_weight;
      double       upper_weight;
      unsigned int lower_index;
      unsigned int upper_index;
    };

    ValueTransfer(const ParametricGeometryMap &geometry,
                  const TargetVectorType      &target_prototype,
                  const EvaluationRequest     &default_request = {})
      : source_size_(geometry.source_parameters().size())
      , target_prototype_(&target_prototype)
      , geometry_(&geometry)
      , default_request_(default_request)
    {}

    TargetVectorType
    apply(const SourceVectorType &source) const
    {
      return apply(source, default_request_);
    }

    TargetVectorType
    apply(const SourceVectorType  &source,
          const EvaluationRequest &request) const
    {
      TargetVectorType result;
      result.reinit(*target_prototype_);
      linearize(request).vmult(result, source);
      return result;
    }

    Operator
    linearize() const
    {
      return linearize(default_request_);
    }

    Operator
    linearize(const EvaluationRequest &request) const
    {
      const auto *target_prototype = target_prototype_;
      const auto  source_size      = source_size_;
      const auto  weights          = interpolation_weights(request);

      Operator result;
      result.reinit_range_vector = [target_prototype](TargetVectorType &vector,
                                                      const bool        omit) {
        vector.reinit(*target_prototype, omit);
      };
      result.reinit_domain_vector = [source_size](SourceVectorType &vector,
                                                  const bool        omit) {
        vector.reinit(source_size);
        if (!omit)
          vector = 0.;
      };
      result.vmult = [weights](TargetVectorType       &destination,
                               const SourceVectorType &source) {
        for (unsigned int i = 0; i < weights.size(); ++i)
          destination[i] =
            weights[i].lower_weight * source[weights[i].lower_index] +
            weights[i].upper_weight * source[weights[i].upper_index];
      };
      result.vmult_add = [weights](TargetVectorType       &destination,
                                   const SourceVectorType &source) {
        for (unsigned int i = 0; i < weights.size(); ++i)
          destination[i] +=
            weights[i].lower_weight * source[weights[i].lower_index] +
            weights[i].upper_weight * source[weights[i].upper_index];
      };
      result.Tvmult = [weights](SourceVectorType       &destination,
                                const TargetVectorType &source) {
        destination = 0.;
        for (unsigned int i = 0; i < weights.size(); ++i)
          {
            destination[weights[i].lower_index] +=
              weights[i].lower_weight * source[i];
            destination[weights[i].upper_index] +=
              weights[i].upper_weight * source[i];
          }
      };
      result.Tvmult_add = [weights](SourceVectorType       &destination,
                                    const TargetVectorType &source) {
        for (unsigned int i = 0; i < weights.size(); ++i)
          {
            destination[weights[i].lower_index] +=
              weights[i].lower_weight * source[i];
            destination[weights[i].upper_index] +=
              weights[i].upper_weight * source[i];
          }
      };
      return result;
    }

  private:
    std::vector<Weight>
    interpolation_weights(const EvaluationRequest &request) const
    {
      const auto         &parameters = geometry_->source_parameters();
      std::vector<Weight> weights;
      weights.reserve(request.points.size());

      AssertDimension(request.points.size(), target_prototype_->size());

      for (const auto &point : request.points)
        {
          const double parameter = geometry_->source_parameter(point);
          const auto   upper =
            std::lower_bound(parameters.begin(), parameters.end(), parameter);

          if (upper == parameters.begin())
            weights.push_back({1., 0., 0u, 0u});
          else if (upper == parameters.end())
            {
              const auto last = parameters.size() - 1;
              weights.push_back({1.,
                                 0.,
                                 static_cast<unsigned int>(last),
                                 static_cast<unsigned int>(last)});
            }
          else
            {
              const auto upper_index =
                static_cast<unsigned int>(upper - parameters.begin());
              const auto   lower_index = upper_index - 1;
              const double interval =
                parameters[upper_index] - parameters[lower_index];
              const double upper_weight =
                (parameter - parameters[lower_index]) / interval;
              weights.push_back(
                {1. - upper_weight, upper_weight, lower_index, upper_index});
            }
        }

      return weights;
    }

    std::size_t                  source_size_;
    const TargetVectorType      *target_prototype_;
    const ParametricGeometryMap *geometry_;
    EvaluationRequest            default_request_;
  };

  /** A Representation obtained by lifting another Representation through a
   * geometry map and its value transfer. */
  template <typename SourceRepresentation,
            typename GeometryMap,
            typename TargetVectorType =
              typename SourceRepresentation::value_type>
  class LiftingRepresentation
  {
  public:
    using source_value_type   = typename SourceRepresentation::value_type;
    using value_type          = TargetVectorType;
    using state_type          = typename SourceRepresentation::state_type;
    using quantity_space_type = QuantitySpace<value_type>;
    using Operator            = RepresentationOperator<value_type, state_type>;
    using ValueTransfer =
      ImmersX::ValueTransfer<source_value_type, TargetVectorType>;

    LiftingRepresentation(const SourceRepresentation &source,
                          const GeometryMap          &geometry,
                          const TargetVectorType     &target_prototype,
                          const EvaluationRequest    &default_request = {})
      : source_(source)
      , geometry_(geometry)
      , transfer_(geometry_, target_prototype, default_request)
    {}

    FieldId
    source() const
    {
      return source_.source();
    }

    const RepresentationDomain &
    domain() const
    {
      return geometry_.domain();
    }

    quantity_space_type
    quantity_space() const
    {
      return quantity_space_type(domain());
    }

    template <typename Geometry>
    decltype(auto)
    lift(const Geometry &geometry) const
    {
      return detail::invoke_lift(*this, geometry, 0);
    }

    const SourceRepresentation &
    source_representation() const
    {
      return source_;
    }

    const GeometryMap &
    geometry() const
    {
      return geometry_;
    }

    const ValueTransfer &
    value_transfer() const
    {
      return transfer_;
    }

    value_type
    evaluate(const EvaluationContext<source_value_type> &context,
             const EvaluationRequest                    &request = {}) const
    {
      const auto &source_value = source_.evaluate(context, request);
      return request.points.empty() ? transfer_.apply(source_value) :
                                      transfer_.apply(source_value, request);
    }

    Operator
    linearize(const EvaluationContext<source_value_type> &context,
              const EvaluationRequest                    &request = {}) const
    {
      return (request.points.empty() ? transfer_.linearize() :
                                       transfer_.linearize(request)) *
             source_.linearize(context, request);
    }

  private:
    SourceRepresentation source_;
    GeometryMap          geometry_;
    ValueTransfer        transfer_;
  };

  template <typename SourceRepresentation,
            typename GeometryMap,
            typename TargetVectorType>
  LiftingRepresentation<SourceRepresentation, GeometryMap, TargetVectorType>
  lift(const SourceRepresentation &source,
       const GeometryMap          &geometry,
       const TargetVectorType     &target_prototype,
       const EvaluationRequest    &default_request = {})
  {
    return LiftingRepresentation<SourceRepresentation,
                                 GeometryMap,
                                 TargetVectorType>(source,
                                                   geometry,
                                                   target_prototype,
                                                   default_request);
  }
} // namespace ImmersX

#endif // immersx_lifting_h
