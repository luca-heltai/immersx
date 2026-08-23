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
  /**
   * A geometry-only map from target evaluation points to source parameters.
   *
   * The map owns no vectors, Fields, Representations, or residual data. A
   * source parameter may occur at any number of target points, which is the
   * natural direction for line-to-surface pullbacks.
   */
  class ParametricGeometryMap
  {
  public:
    using Point        = dealii::Point<3>;
    using ParameterMap = std::function<double(const Point &)>;

    ParametricGeometryMap(std::vector<double>    source_parameters,
                          std::vector<Point>     target_points,
                          const EvaluationDomain target_domain)
      : ParametricGeometryMap(std::move(source_parameters),
                              std::move(target_points),
                              target_domain,
                              default_source_parameter)
    {}

    ParametricGeometryMap(std::vector<double>    source_parameters,
                          std::vector<Point>     target_points,
                          const EvaluationDomain target_domain,
                          ParameterMap           source_parameter)
      : source_parameters_(std::move(source_parameters))
      , target_points_(std::move(target_points))
      , target_domain_(target_domain)
      , source_parameter_(std::move(source_parameter))
    {
      AssertDimension(target_domain_.evaluation_points.size(),
                      target_points_.size());
      AssertThrow(!source_parameters_.empty() || target_points_.empty(),
                  dealii::ExcMessage(
                    "A non-empty target requires source parameters."));
      AssertThrow(
        std::is_sorted(source_parameters_.begin(), source_parameters_.end()),
        dealii::ExcMessage("Geometry-map source parameters must be sorted."));
      AssertThrow(std::adjacent_find(source_parameters_.begin(),
                                     source_parameters_.end()) ==
                    source_parameters_.end(),
                  dealii::ExcMessage(
                    "Geometry-map source parameters must be distinct."));
    }

    const EvaluationDomain &
    domain() const
    {
      return target_domain_;
    }

    const std::vector<double> &
    source_parameters() const
    {
      return source_parameters_;
    }

    const std::vector<Point> &
    target_points() const
    {
      return target_points_;
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

    std::vector<double> source_parameters_;
    std::vector<Point>  target_points_;
    EvaluationDomain    target_domain_;
    ParameterMap        source_parameter_;
  };

  /**
   * The algebraic value transfer associated with a ParametricGeometryMap.
   *
   * This object is deliberately separate from the map: it knows vector types
   * and constructs the LinearOperator, while the map knows only domains and
   * coordinates.
   */
  template <typename SourceVectorType, typename TargetVectorType>
  class ParametricValueTransfer
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

    ParametricValueTransfer(const ParametricGeometryMap &geometry,
                            const TargetVectorType      &target_prototype)
      : source_size_(geometry.source_parameters().size())
      , target_prototype_(&target_prototype)
      , weights_(interpolation_weights(geometry))
    {}

    TargetVectorType
    apply(const SourceVectorType &source) const
    {
      TargetVectorType result;
      result.reinit(*target_prototype_);
      linearize().vmult(result, source);
      return result;
    }

    Operator
    linearize() const
    {
      const auto *target_prototype = target_prototype_;
      const auto  source_size      = source_size_;
      const auto  weights          = weights_;

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
    interpolation_weights(const ParametricGeometryMap &geometry) const
    {
      const auto         &parameters = geometry.source_parameters();
      std::vector<Weight> weights;
      weights.reserve(geometry.target_points().size());

      for (const auto &point : geometry.target_points())
        {
          const double parameter = geometry.source_parameter(point);
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

    std::size_t             source_size_;
    const TargetVectorType *target_prototype_;
    std::vector<Weight>     weights_;
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
    using source_value_type = typename SourceRepresentation::value_type;
    using value_type        = TargetVectorType;
    using Operator = RepresentationOperator<value_type, source_value_type>;
    using ValueTransfer =
      ParametricValueTransfer<source_value_type, TargetVectorType>;

    LiftingRepresentation(const SourceRepresentation &source,
                          const GeometryMap          &geometry,
                          const TargetVectorType     &target_prototype)
      : source_(source)
      , geometry_(geometry)
      , transfer_(geometry_, target_prototype)
    {}

    FieldId
    source() const
    {
      return source_.source();
    }

    const EvaluationDomain &
    domain() const
    {
      return geometry_.domain();
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
    evaluate(const EvaluationContext<source_value_type> &context) const
    {
      return transfer_.apply(source_.evaluate(context));
    }

    Operator
    linearize(const EvaluationContext<source_value_type> &context) const
    {
      return transfer_.linearize() * source_.linearize(context);
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
       const TargetVectorType     &target_prototype)
  {
    return LiftingRepresentation<SourceRepresentation,
                                 GeometryMap,
                                 TargetVectorType>(source,
                                                   geometry,
                                                   target_prototype);
  }
} // namespace ImmersX

#endif // immersx_lifting_h
