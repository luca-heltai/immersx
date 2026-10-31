// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_execution_h
#define immersx_internal_growth_execution_h
#include "growth_demand_journal.h"
#include "growth_owned_rows.h"
namespace ImmersX::Vascularization::internal
{
  struct GrowthExecutionOutput
  {
    NetworkData          network;
    std::set<Id>         mutable_owned;
    std::map<Id, double> fixed_owned_roots;
  };
  struct GrowthExecutionResult
  {
    bool                  committed = false;
    ValidationReport      validation;
    GrowthGeometryChanges geometry_changes;
    std::array<double, 2> volume{}, power{}, total_demand{};
    double                redistribution_seconds = 0., physiology_seconds = 0.,
           certification_seconds = 0.;
  };
  class GrowthExecution
  {
  public:
    using ObstacleLeaf = std::pair<dealii::BoundingBox<3>, Id>;
    using ObstacleTree = dealii::RTree<ObstacleLeaf>;
    // This private view borrows base rows and indices until the next stage.
    // Physical radii must be read through the demand-state sparse accessors.
    class AcceptedSnapshot
    {
    public:
      bool
      current() const noexcept;
      std::uint64_t
      revision() const;
      const NetworkData &
      network() const;
      const GrowthDemandState &
      demand() const;
      const GrowthIndices &
      indices() const;
      const ObstacleTree &
      obstacles() const;
      std::uint64_t
      owned_mutable_count() const;
      std::uint64_t
      mutable_count() const;
      std::uint64_t
      owned_terminal_count() const;
      std::uint64_t
      terminal_count() const;

    private:
      friend class GrowthExecution;
      AcceptedSnapshot(const GrowthExecution &, std::uint64_t, std::uint64_t);
      const GrowthExecution *owner;
      std::uint64_t          snapshot_revision, view_epoch;
    };
    GrowthExecution(const NetworkData &,
                    const ForestState &,
                    const Parameters &,
                    std::uint64_t,
                    BackgroundGeometry &,
                    const GrowthGeometryMeasures &initial_owned_measures,
                    GrowthGeometryMeasureFunction,
                    GrowthGeometryReleaseFunction,
                    double physical_ghost_halo,
                    MPI_Comm,
                    GrowthWork * = nullptr);
    ~GrowthExecution();
    GrowthExecution(const GrowthExecution &) = delete;
    GrowthExecution &
    operator=(const GrowthExecution &) = delete;
    AcceptedSnapshot
    accepted_snapshot() const;
    const ValidationReport &
    validation() const noexcept;
    GrowthExecutionResult
    commit(const std::vector<GrowthSplit> &,
           const GrowthDemandJointUpdate &,
           const GrowthDemandCallbacks & = {});
    // Explicit O(N) SI/output operation, never part of an ordinary proposal.
    GrowthExecutionOutput
    materialize() const;

  private:
    using Mass = boost::multiprecision::cpp_int;
    struct Certificate
    {
      Edge   cap;
      Id     root = invalid_id;
      Mass   mass, lower_root_mass, upper_root_mass;
      double clearance_scale = 0.;
    };
    GrowthDemandState             demand_state;
    GrowthOwnedRows               rows;
    GrowthDemandJournal           journal;
    GrowthIndices                 spatial;
    ObstacleTree                  obstacle_index;
    const Parameters             &parameters;
    BackgroundGeometry           &geometry;
    GrowthGeometryMeasureFunction measure;
    GrowthGeometryReleaseFunction release;
    MPI_Comm                      communicator;
    GrowthWork                   *work;
    double           ghost_halo, clearance_bound = 0., old_clearance_bound = 0.;
    std::uint64_t    view_epoch = 0, owned_leaves = 0, leaves = 0;
    bool             staging = false;
    ValidationReport report;
    std::map<Id, Certificate>                  certificates;
    std::map<Id, GrowthDemandRootView>         controls;
    std::map<Id, std::map<Mass, std::set<Id>>> thresholds, shrink_thresholds;
    std::map<Id, std::size_t>                  root_members;
    std::set<Id> root_members_changed, index_touched;
    std::map<Id, std::optional<Certificate>>          certificates_before;
    std::map<Id, std::optional<GrowthDemandRootView>> controls_before;
    GrowthGeometrySegments release_old, release_trial;
    bool
    phase(ValidationReport &, bool, const char *) const;
    GrowthDemandPreview
    accepted_preview() const;
    bool
    refresh_controls(const GrowthOwnershipDelta &,
                     const GrowthDemandPreview &,
                     ValidationReport &);
    bool
    physical_bounds(const GrowthDemandPreview &, ValidationReport &) const;
    double
    radius(Id, const GrowthDemandPreview &) const;
    double
    radius_at(const Mass &, const GrowthDemandRootView &, const Mass &) const;
    dealii::BoundingBox<3>
    certificate_box(const Certificate &) const;
    Certificate
    certificate_at(Id,
                   const NetworkData &,
                   const GrowthDemandPreview &,
                   double,
                   const Mass &) const;
    void
    replace_certificate(Id, std::optional<Certificate>, bool remember = true);
    bool
    certify(const GrowthOwnershipDelta &,
            const GrowthDemandPreview &,
            const GrowthGeometryMeasures &,
            ValidationReport &);
    void
    discard() noexcept;
    void
    adopt();
  };
} // namespace ImmersX::Vascularization::internal
#endif
