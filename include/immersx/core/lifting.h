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
#include <functional>
#include <utility>
#include <vector>

namespace ImmersX
{
  /**
   * A small geometry-owned interpolation from source parameters to target
   * evaluation points.
   *
   * This is an analytical lifting boundary rather than a finite-element
   * assembly object.  The source vector contains values at
   * `source_parameters`; target values are obtained by piecewise-linear
   * interpolation after the supplied geometry map extracts a source
   * parameter from each target point.  The target points and the transfer
   * action belong to the geometry, not to a Problem or an Interaction.
   */
  template <typename SourceVectorType, typename TargetVectorType>
  class ParametricLiftingGeometry
  {
  public:
    using Point        = dealii::Point<3>;
    using ParameterMap = std::function<double(const Point &)>;
    using Operator = RepresentationOperator<TargetVectorType, SourceVectorType>;
    using source_value_type = SourceVectorType;
    using target_value_type = TargetVectorType;

    ParametricLiftingGeometry(
      const SourceVectorType &source_prototype,
      const TargetVectorType &target_prototype,
      std::vector<double>     source_parameters,
      std::vector<Point>      target_points,
      const EvaluationDomain  target_domain,
      ParameterMap            source_parameter =
        [](const Point &point) { return point[0]; })
      : source_prototype_(&source_prototype)
      , target_prototype_(&target_prototype)
      , source_parameters_(std::move(source_parameters))
      , target_points_(std::move(target_points))
      , target_domain_(target_domain)
      , source_parameter_(std::move(source_parameter))
    {
      AssertDimension(source_parameters_.size(), source_prototype.size());
      AssertDimension(target_points_.size(), target_prototype.size());
      AssertDimension(target_domain_.evaluation_points.size(),
                      target_points_.size());
      AssertThrow(
        std::is_sorted(source_parameters_.begin(), source_parameters_.end()),
        dealii::ExcMessage("Lifting source parameters must be sorted."));
    }

    const EvaluationDomain &
    domain() const
    {
      return target_domain_;
    }

    const std::vector<Point> &
    target_points() const
    {
      return target_points_;
    }

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
      const auto *source_prototype = source_prototype_;
      const auto *target_prototype = target_prototype_;
      const auto  weights          = interpolation_weights();

      Operator result;
      result.reinit_range_vector = [target_prototype](TargetVectorType &vector,
                                                      const bool        omit) {
        vector.reinit(*target_prototype, omit);
      };
      result.reinit_domain_vector = [source_prototype](SourceVectorType &vector,
                                                       const bool        omit) {
        vector.reinit(*source_prototype, omit);
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
    struct Weight
    {
      double       lower_weight;
      double       upper_weight;
      unsigned int lower_index;
      unsigned int upper_index;
    };

    std::vector<Weight>
    interpolation_weights() const
    {
      std::vector<Weight> weights;
      weights.reserve(target_points_.size());

      for (const auto &point : target_points_)
        {
          const double parameter = source_parameter_(point);
          const auto   upper     = std::lower_bound(source_parameters_.begin(),
                                              source_parameters_.end(),
                                              parameter);

          if (upper == source_parameters_.begin())
            weights.push_back({1., 0., 0u, 0u});
          else if (upper == source_parameters_.end())
            {
              const auto last = source_parameters_.size() - 1;
              weights.push_back({1.,
                                 0.,
                                 static_cast<unsigned int>(last),
                                 static_cast<unsigned int>(last)});
            }
          else
            {
              const auto upper_index =
                static_cast<unsigned int>(upper - source_parameters_.begin());
              const auto   lower_index = upper_index - 1;
              const double interval    = source_parameters_[upper_index] -
                                      source_parameters_[lower_index];
              const double upper_weight =
                (parameter - source_parameters_[lower_index]) / interval;
              weights.push_back(
                {1. - upper_weight, upper_weight, lower_index, upper_index});
            }
        }

      return weights;
    }

    const SourceVectorType *source_prototype_;
    const TargetVectorType *target_prototype_;
    std::vector<double>     source_parameters_;
    std::vector<Point>      target_points_;
    EvaluationDomain        target_domain_;
    ParameterMap            source_parameter_;
  };

  /** A Representation obtained by evaluating another Representation on a new
   * geometric support. */
  template <typename SourceRepresentation, typename Geometry>
  class LiftedRepresentation
  {
  public:
    using source_value_type = typename SourceRepresentation::value_type;
    using value_type        = typename Geometry::target_value_type;
    using Operator = RepresentationOperator<value_type, source_value_type>;

    LiftedRepresentation(const SourceRepresentation &source,
                         const Geometry             &geometry)
      : source_(source)
      , geometry_(geometry)
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

    const Geometry &
    geometry() const
    {
      return geometry_;
    }

    value_type
    evaluate(const EvaluationContext<source_value_type> &context) const
    {
      return geometry_.apply(source_.evaluate(context));
    }

    Operator
    linearize(const EvaluationContext<source_value_type> &context) const
    {
      return geometry_.linearize() * source_.linearize(context);
    }

  private:
    SourceRepresentation source_;
    Geometry             geometry_;
  };

  template <typename SourceRepresentation, typename Geometry>
  LiftedRepresentation<SourceRepresentation, Geometry>
  lift(const SourceRepresentation &source, const Geometry &geometry)
  {
    return LiftedRepresentation<SourceRepresentation, Geometry>(source,
                                                                geometry);
  }
} // namespace ImmersX

#endif // immersx_lifting_h
