// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_demand_state_h
#define immersx_internal_growth_demand_state_h
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_int/serialize.hpp>
#include <boost/serialization/array.hpp>

#include <immersx/vascularization/growth_work.h>

#include "demand_mass.h"
#include "forest_state.h"
#include "growth_demand.h"
#include "growth_geometry.h"
#include "growth_geometry_measure.h"

namespace ImmersX::Vascularization::internal
{
  struct GrowthCandidatePaths;
  class GrowthExecution;
  class GrowthDemandJournal;
  using GrowthRootSumKey = std::pair<unsigned int, Id>;
  // Physical summaries belong to GrowthDemandSummary, not to the topology
  // retained from the independent bootstrap reconstruction.
  struct GrowthBranchTopology
  {
    Id           parent = invalid_id, root = invalid_id, terminal = invalid_id;
    unsigned int parent_owner   = dealii::numbers::invalid_unsigned_int;
    unsigned int terminal_owner = dealii::numbers::invalid_unsigned_int;
    std::map<Id, unsigned int> children;
    double                     root_radius = 0.;

    template <class Archive>
    void
    serialize(Archive &ar, unsigned int)
    {
      ar &parent &root &terminal &parent_owner &terminal_owner &children
        &root_radius;
    }
  };

  // Moments are relative to this subtree's own demand. Changing a root or
  // system normalization therefore does not rewrite its descendants.
  struct GrowthDemandSummary
  {
    boost::multiprecision::cpp_int mass, minimum_leaf_mass;
    long double volume_length = 0., resistance_length = 0., minimum_ratio = 1.;
    long double minimum_scaled_ratio      = 1.;
    long double minimum_resistance_length = 0., maximum_resistance_length = 0.;
    std::uint64_t leaves = 0;
    template <class Archive>
    void
    serialize(Archive &ar, unsigned int)
    {
      ar &mass &minimum_leaf_mass &volume_length &resistance_length
        &minimum_ratio;
      ar &minimum_scaled_ratio &minimum_resistance_length
        &maximum_resistance_length &leaves;
    }
  };

  struct GrowthDemandPreview
  {
    ValidationReport validation;
    std::uint64_t    snapshot_revision = 0;
    // Per system: absolute demand, SI volume, physical-demand power
    // coefficient.
    std::array<long double, 9>           totals{};
    std::optional<std::array<double, 2>> canonical_total_demand;
    // Separate portal control preserves the existing arterial/venous input.
    std::optional<double> canonical_portal_demand;
    long double
    total_demand(unsigned int system) const
    {
      AssertIndexRange(system, 3);
      if (system < 2 && canonical_total_demand)
        return (*canonical_total_demand)[system];
      else if (system == 2 && canonical_portal_demand)
        return *canonical_portal_demand;
      return totals[3 * system];
    }
    // Nine scalar controls per rank; no topology or per-terminal data.
    std::vector<std::array<long double, 9>> rank_totals;
    // Sparse derived nodes in the accepted root-sum tree.
    std::map<GrowthRootSumKey, std::array<long double, 9>> root_sums;
    // Only the union of updated terminal-to-root paths, on branch owners.
    std::map<Id, GrowthDemandSummary>               owned;
    std::map<Id, double>                            leaf_demands;
    std::map<Id, std::map<Id, GrowthDemandSummary>> child_summaries;
    std::uint64_t marked_branches = 0, recomputed_branches = 0;
  };

  struct GrowthDemandRootView
  {
    GrowthDemandSummary        summary;
    std::array<long double, 9> other_local{};
    double                     radius = 0., gamma = 3.;
    unsigned int               system = 0;
    template <class Archive>
    void
    serialize(Archive &ar, unsigned int)
    {
      ar &summary &other_local &radius &gamma &system;
    }
  };

  struct GrowthDemandEdge
  {
    Edge        edge;
    Id          root   = invalid_id;
    long double demand = 0., root_demand = 0.;
    template <class Archive>
    void
    serialize(Archive &ar, unsigned int)
    {
      ar &edge &root &demand &root_demand;
    }
  };
  struct GrowthDemandEdges
  {
    ValidationReport               validation;
    std::map<Id, GrowthDemandEdge> records;
  };

  class GrowthDemandState
  {
  public:
    // One full bootstrap from an independently validated physical forest.
    GrowthDemandState(const NetworkData &,
                      const ForestState &,
                      const Parameters &,
                      std::uint64_t,
                      MPI_Comm,
                      const GrowthGeometryMeasures *geometry = nullptr);

    const ValidationReport &
    validation() const;

    const GrowthDemandSummary &summary(Id) const;
    const GrowthDemandSummary                             &
    summary(Id, const GrowthDemandPreview &) const;
    const GrowthBranchTopology &topology(Id) const;
    std::map<Id, GrowthDemandSummary>
    children(Id, const GrowthDemandPreview &) const;
    GrowthDemandRootView
    root_view(Id, const GrowthDemandPreview &) const;

    const std::array<long double, 9> &
    totals() const;

    GrowthDemandPreview
    preview(const GrowthDemandUpdate &) const;

    // Resolve only requested vessels and their current root controls.
    GrowthDemandEdges
    physical_edges(const NetworkData &,
                   const GrowthDemandPreview &,
                   const std::map<Id, unsigned int> &) const;

    // One collective overlay for one globally ordered tissue proposal. Only
    // its producer requests paths; owners must not use different previews.
    GrowthCandidatePaths
    candidate_paths(const NetworkData &,
                    const GrowthDemandPreview &,
                    const std::map<Id, unsigned int> &,
                    Id          next_node,
                    Id          next_vessel,
                    double      new_demand,
                    double      minimum_scale = 1.,
                    GrowthWork *work          = nullptr) const;

  private:
    friend class GrowthExecution;
    friend class GrowthDemandJournal;

    std::map<Id, GrowthDemandRootView>
    root_controls(const std::set<Id> &,
                  const GrowthDemandPreview &,
                  ValidationReport &) const;
    GrowthDemandPreview
    route_demands(const GrowthDemandUpdate &) const;
    GrowthDemandPreview
    preview_branches(GrowthDemandPreview, const std::set<Id> &) const;

    struct Branch
    {
      GrowthBranchTopology topology;
      double       length = 0., viscosity = 0., gamma = 3., leaf_demand = 0.;
      unsigned int system = 0;
      GrowthGeometryMeasure             geometry;
      bool                              with_resistance = false;
      std::map<Id, GrowthDemandSummary> children;
      GrowthDemandSummary               value;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &topology &length &viscosity &gamma &leaf_demand &system;
        ar &geometry.normalized_length &geometry.radius_scale;
        ar &geometry.clearance_scale  &geometry.complete;
        ar &with_resistance &children &value;
      }
    };
    struct DirectoryEntry
    {
      Id           branch = invalid_id;
      unsigned int owner  = dealii::numbers::invalid_unsigned_int;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &branch &owner;
      }
    };
    static GrowthDemandSummary
    aggregate(const Branch &,
              double,
              const std::map<Id, GrowthDemandSummary> &);

    static long double
    demand_value(const GrowthDemandSummary &);

    MPI_Comm                     communicator;
    std::uint64_t                revision = 0, branch_count = 0;
    ValidationReport             report;
    std::map<Id, Branch>         branches;
    std::map<Id, DirectoryEntry> directory;
    std::map<Id, unsigned int>   root_directory;
    std::set<Id>                 owned_roots;
    std::map<GrowthRootSumKey, std::array<long double, 9>> root_sums;
    std::array<long double, 9>                             accepted_totals{};
    std::array<double, 3>                                  accepted_volume{};
    std::array<long double, 3> accepted_resistance{};
    std::array<long double, 9>
    root_value(Id, const GrowthDemandSummary &) const;
  };
} // namespace ImmersX::Vascularization::internal
#endif
