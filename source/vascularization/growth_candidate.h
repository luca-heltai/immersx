// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_candidate_h
#define immersx_internal_growth_candidate_h
#include <immersx/vascularization/growth_work.h>

#include <functional>
#include <limits>

#include "growth_geometry.h"
namespace ImmersX::Vascularization
{
  struct Parameters;
  namespace internal
  {
    struct GrowthPathBranch
    {
      Edge         edge;
      double       demand = 0., viscosity = 0.;
      unsigned int owner = dealii::numbers::invalid_unsigned_int;
      std::array<dealii::Point<3>, 2> port_normals;
      std::array<unsigned int, 2>     node_owners;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &edge &demand &viscosity &owner &port_normals &node_owners;
      }
    };
    // Physical demand and moments relative to this subtree demand.
    struct GrowthSubtreePhysiology
    {
      long double demand = 0., volume_length = 0., resistance_length = 0.;
      long double minimum_ratio = 1.;
    };
    struct GrowthCandidatePhysiology
    {
      std::vector<GrowthSubtreePhysiology>               path;
      std::vector<std::map<Id, GrowthSubtreePhysiology>> children;
      long double other_volume = 0., other_power_coefficient = 0.;
      long double final_system_demand = 0.;
      double      reference_volume    = 0.;
      // Accepted power / Q^2, evaluated without dividing underflowed SI power.
      long double reference_resistance = 0.;
    };
    // Only the selected branch and its actual ancestors, ordered towards the
    // physical root. Root moments include ALL branches of that rooted tree.
    struct GrowthCandidateContext
    {
      std::vector<GrowthPathBranch> path;
      System                        system = System::arterial;
      Id root = invalid_id, next_node = invalid_id, next_vessel = invalid_id;
      double root_radius = 0., root_demand = 0., minimum_demand = 0.;
      double volume_sum = 0., dissipation_sum = 0.;
      double total_demand = 0., demand_scale = 1., murray_exponent = 3.;
      double forest_volume = 0., forest_power_numerator = 0.;
      double minimum_scale = 1.;
      std::optional<GrowthCandidatePhysiology> physiology;
    };
    // Trial obstacle radii for a single insertion. Baseline obstacle radii
    // must already describe the same old-demand preview as the context.
    struct GrowthCandidateRadii
    {
      Id                   root               = invalid_id;
      long double          sibling_scale      = 1.;
      bool                 physical_precision = false;
      std::map<Id, double> path_radii;
      double
      operator()(Id vessel, Id obstacle_root, double baseline_radius) const
      {
        if (obstacle_root != root)
          return baseline_radius;
        const auto found = path_radii.find(vessel);
        if (found != path_radii.end())
          return found->second;
        if (physical_precision)
          return static_cast<long double>(baseline_radius) * sibling_scale;
        return baseline_radius * static_cast<double>(sibling_scale);
      }
    };
    std::optional<GrowthCandidateRadii>
    candidate_radii(const GrowthCandidateContext &context,
                    double                        new_demand,
                    GrowthWork                   *work = nullptr);
    struct GrowthCandidate
    {
      Id                vessel = invalid_id;
      dealii::Point<3>  junction, terminal;
      std::vector<Id>   ancestors;
      long double       volume_sum = 0., dissipation_sum = 0.;
      double            cost = std::numeric_limits<double>::infinity();
      double            terminal_demand = 1.;
      std::vector<Edge> changed;
    };
    // The predicate supplies the caller's locally available geometry checks.
    // Distributed callers still require batch certification and joint-radius
    // checks before adopting a result. Evaluation never changes physical data.
    using CandidateGeometry = std::function<bool(const std::vector<Edge> &)>;
    std::optional<GrowthCandidate>
    propose_growth(const GrowthCandidateContext &context,
                   const Parameters             &parameters,
                   const dealii::Point<3>       &junction,
                   const dealii::Point<3>       &terminal,
                   double                        ceiling,
                   double                        terminal_demand,
                   const CandidateGeometry      &geometry,
                   GrowthWork                   *work = nullptr);
    std::optional<GrowthCandidate>
    optimize_growth(const GrowthCandidateContext &context,
                    const Parameters             &parameters,
                    const dealii::Point<3>       &terminal,
                    double                        terminal_demand,
                    const CandidateGeometry      &geometry,
                    GrowthWork                   *work = nullptr);
  } // namespace internal
} // namespace ImmersX::Vascularization
#endif
