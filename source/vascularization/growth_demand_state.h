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

namespace ImmersX::Vascularization::internal
{
  struct GrowthCandidatePaths;
  // Moments are relative to this subtree's own demand. Changing a root or
  // system normalization therefore does not rewrite its descendants.
  struct GrowthDemandSummary
  {
    boost::multiprecision::cpp_int mass;
    long double volume_length = 0., resistance_length = 0., minimum_ratio = 1.;
    std::uint64_t leaves = 0;
    template <class Archive>
    void
    serialize(Archive &ar, unsigned int)
    {
      ar &mass &volume_length &resistance_length &minimum_ratio &leaves;
    }
  };

  struct GrowthDemandPreview
  {
    ValidationReport validation;
    std::uint64_t    snapshot_revision = 0;
    // Per system: absolute demand, SI volume, physical-demand power
    // coefficient.
    std::array<long double, 6>           totals{};
    std::optional<std::array<double, 2>> canonical_total_demand;
    // Six scalar controls per rank; no topology or per-terminal data.
    std::vector<std::array<long double, 6>> rank_totals;
    // Sparse derived nodes in the accepted root-sum tree.
    std::map<std::size_t, std::array<long double, 6>> root_sums;
    // Only the union of updated terminal-to-root paths, on branch owners.
    std::map<Id, GrowthDemandSummary>               owned;
    std::map<Id, double>                            leaf_demands;
    std::map<Id, std::map<Id, GrowthDemandSummary>> child_summaries;
    std::uint64_t marked_branches = 0, recomputed_branches = 0;
  };

  struct GrowthDemandRootView
  {
    GrowthDemandSummary        summary;
    std::array<long double, 6> other_local{};
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
                      MPI_Comm);

    const ValidationReport &
    validation() const;

    const GrowthDemandSummary &summary(Id) const;
    const GrowthDemandSummary                     &
    summary(Id, const GrowthDemandPreview &) const;
    const ForestBranch &topology(Id) const;
    std::map<Id, GrowthDemandSummary>
    children(Id, const GrowthDemandPreview &) const;
    GrowthDemandRootView
    root_view(Id, const GrowthDemandPreview &) const;

    const std::array<long double, 6> &
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
    struct Branch
    {
      ForestBranch topology;
      double       length = 0., viscosity = 0., gamma = 3., leaf_demand = 0.;
      unsigned int system = 0;
      std::map<Id, GrowthDemandSummary> children;
      GrowthDemandSummary               value;
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

    MPI_Comm                                communicator;
    std::uint64_t                           revision = 0, branch_count = 0;
    ValidationReport                        report;
    std::map<Id, Branch>                    branches;
    std::map<Id, DirectoryEntry>            directory;
    std::map<Id, unsigned int>              root_directory;
    std::map<Id, std::size_t>               root_positions;
    std::vector<std::array<long double, 6>> root_sums;
    std::array<long double, 6>              accepted_totals{};
    std::array<double, 2>                   accepted_volume{};
    std::array<long double, 2>              accepted_resistance{};
    std::size_t                             root_capacity = 1;
    std::array<long double, 6>
    root_value(Id, const GrowthDemandSummary &) const;
  };
} // namespace ImmersX::Vascularization::internal
#endif
