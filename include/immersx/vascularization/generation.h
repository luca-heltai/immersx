// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_generation_h
#define immersx_vascularization_generation_h

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <immersx/vascularization/discretization.h>
#include <immersx/vascularization/growth_work.h>
#include <immersx/vascularization/mesh_scale.h>
#include <immersx/vascularization/metrics.h>
#include <immersx/vascularization/network.h>
#include <immersx/vascularization/reference_flow.h>

#include <functional>

namespace ImmersX::Vascularization
{
  struct CalibrationParameters
  {
    void
    validate() const;

    /** Zero disables the volume target. */
    double volume_fraction    = 0.;
    double relative_tolerance = .05;
    /** Minimum service coverage: arterial, dual for A/V, or triple for A/P/V.
     */
    double       minimum_coverage  = 0.;
    unsigned int minimum_terminals = 1;
    /** Zero derives the maximum from the scaled initial count (or explicit
     * minimum). */
    unsigned int maximum_terminals = 0;
    unsigned int attempt_budget    = 12;
  };

  struct TissueIntegrationParameters
  {
    void
                 validate() const;
    unsigned int quadrature_order     = 2;
    unsigned int samples_per_terminal = 64;
    unsigned int maximum_points       = 4000000;
    unsigned int refinement_budget    = 8;
  };

  enum class CalibrationStatus
  {
    not_requested,
    converged,
    incompatible,
    exhausted
  };

  struct CalibrationAttempt
  {
    unsigned int     terminal_count = 0;
    TissueMetrics    tissue;
    ValidationReport generation_validation;
    std::uint64_t    sampled_points = 0;
  };

  struct CalibrationReport
  {
    CalibrationStatus               status = CalibrationStatus::not_requested;
    std::vector<CalibrationAttempt> attempts;
    /** Includes samples from discarded attempts. */
    std::uint64_t sampled_points = 0;
  };

  struct CompanionParameters
  {
    bool             enabled = false;
    std::vector<Id>  selected_terminals;
    dealii::Point<3> seed_position, seed_inward_port_normal;
    double           seed_radius     = 0.;
    double           murray_exponent = 3.;
    /** Separation from the arterial endpoint in the configured length scale. */
    double terminal_separation = .25;
    /** Two-system venous bed resistance in Pa s/m^3; zero derives beds from
     * flow. Keep zero for portal controls, which use three compartment
     * resistances. */
    double microvascular_resistance = 0.;
  };

  using VenousParameters = CompanionParameters;

  struct Parameters : dealii::ParameterAcceptor
  {
    explicit Parameters(const std::string &section = "/Vascularization/");
    void
    validate(bool creating_seed = true) const;

    TissueIntegrationParameters tissue_integration;
    DiscretizationParameters    discretization;
    CoverageParameters          coverage;
    CalibrationParameters       calibration;
    VenousParameters            venous;
    CompanionParameters         portal;
    /** Fraction of total new-tissue flow supplied by the portal network. */
    double portal_flow_fraction = .5;
    /** Star-arm resistances in arterial, portal, venous order; zero derives
     * them. */
    std::array<double, 3> compartment_resistances{};
    /** In network-resistance mode, multiply supplied arms by the mean new
     * territory demand divided by the local demand. Equal demands retain the
     * supplied per-bed resistances; otherwise conductance scales with demand.
     */
    bool scale_compartment_resistances_by_demand = false;
    /** Explicit port pressures, Robin data and optional loads on unpaired input
     * terminals.
     */
    ReferenceConditions reference_conditions;
    ReferenceParameters reference_parameters;
    std::vector<Id>     selected_terminals;
    dealii::Point<3>    seed_position;
    /** Zero selects a deterministic direction into the containing cell. */
    dealii::Point<3> seed_direction;
    /** Zero retains a closed seed; otherwise declares a planar inlet section.
     */
    dealii::Point<3> seed_inward_port_normal;
    double           seed_radius = 0.;
    double           seed_length = 1.;
    double           viscosity   = 0.;
    double           clearance   = 0.;
    /** Enforce geometric growth heuristics and wall/vessel admissibility.
     * Disable to defer geometric repair while retaining positive finite
     * hydraulic data, graph ownership and topology, and reference-flow checks.
     */
    bool         enforce_geometry         = true;
    unsigned int geometry_depth           = 18;
    unsigned int geometry_work            = 100000;
    unsigned int terminal_count           = 1;
    unsigned int random_seed              = 1;
    unsigned int sample_budget            = 2000;
    unsigned int candidate_count          = 12;
    unsigned int optimization_steps       = 10;
    double       arterial_murray_exponent = 3.;
    double       minimum_radius           = 0.;
    double       minimum_length           = .1;
    double       minimum_angle            = 15.;
    double       dissipation_weight       = 0.;
    double       density_beta             = 3.;
    std::string  terminal_weight          = "1";
    double       terminal_weight_bound    = 1.;
    /** Relative tissue-demand density in x,y,z, integrated with physical JxW.
     */
    std::string tissue_demand = "1";
    /** Total new-tissue venous drainage in m^3/s; zero leaves flow unspecified.
     * Portal mode splits this total between arterial and portal supply.
     */
    double reference_flow = 0.;
    /** Final reference flow policy: tissue demand matches each new terminal;
     * network resistance checks only per-system totals, using supplied positive
     * bed resistances and the explicit reference boundary conditions.
     * Growth still uses demand-weighted design flows in either mode.
     */
    std::string reference_flow_distribution = "tissue demand";
    /** Multiply the initial count by one, global active cells or used vertices.
     * Calibration limits remain absolute terminal counts. */
    std::string terminal_count_multiplier_factor = "background cell count";
    /** Reference for configured lengths; physical network/input data stay SI.
     */
    std::string length_scale = "local mesh size";
  };

  struct Territory
  {
    Id                          id = invalid_id;
    dealii::Point<3>            position;
    dealii::types::subdomain_id owner  = 0;
    double                      demand = 1.;
  };

  /** Local measurements for batches starting in one power-of-two leaf-count
   * range. No graph data or extra reductions are collected. Counts of samples,
   * proposals and accepted insertions belong to their producers; vessel counts
   * include only owned mutable vessels of the active system, never ghosts.
   * Timings are local wall seconds, including communication waits. Commit
   * subphases are included in commit_seconds, not added to it. Initialization
   * and finalization are outside the accumulated batch total_seconds.
   */
  struct GrowthStatistics
  {
    GrowthWork    work, initialization_work, finalization_work;
    System        system              = System::arterial;
    unsigned int  requested_terminals = 0;
    std::uint64_t first_terminals = 0, last_terminals = 0;
    // Global accepted/peak cardinalities: use max, not sum, across ranks.
    std::uint64_t demand_quadrature_points      = 0,
                  demand_quadrature_peak_points = 0;
    std::uint64_t batches = 0, sampled_points = 0, proposed_insertions = 0,
                  accepted_insertions = 0, committed_batches = 0,
                  rejected_batches = 0, active_writer_batches = 0;
    std::uint64_t owned_vessels_first = 0, owned_vessels_last = 0,
                  owned_vessels_maximum = 0;
    double initialization_seconds = 0., sampling_seconds = 0.,
           search_seconds = 0., optimization_seconds = 0.,
           selection_seconds = 0., commit_seconds = 0.,
           redistribution_seconds = 0., physiology_seconds = 0.,
           certification_seconds = 0., finalization_seconds = 0.,
           total_seconds = 0.;
  };

  /** Accepted-state progress of one native forest. Counts are global. */
  struct GenerationProgress
  {
    System        system              = System::arterial;
    unsigned int  requested_terminals = 0;
    std::uint64_t accepted_terminals = 0, batches = 0;
  };

  struct GenerationAction
  {
    bool checkpoint = false, stop = false;
  };

  /** Optional synchronous controls at completed batch boundaries.
   *
   * Supply or omit each callback consistently on every MPI rank. poll must not
   * perform collective operations: its requests are combined collectively.
   * checkpoint is called on every rank with ownership of a materialized copy
   * of the last accepted state. It may use collective output, but must follow
   * the same collective sequence on every rank. The bool marks a controlled
   * stop; all such snapshots remain incomplete, diagnostic and non-resumable.
   * Empty controls retain the ordinary path without additional collectives or
   * accepted-state materializations.
   */
  struct GenerationControl
  {
    std::function<GenerationAction(const GenerationProgress &)> poll;
    std::function<void(const GenerationProgress &, NetworkData, bool)>
      checkpoint;
  };

  struct GenerationResult
  {
    MeshLengthScale length_scale;
    unsigned int    base_terminal_count = 0, initial_terminal_count = 0,
                 requested_terminal_count = 0, effective_terminal_count = 0;
    std::map<Id, ScaledVesselMeasure> normalized_geometry;
    NetworkData                       network;
    std::vector<Territory>            territories;
    std::vector<TerminalPair>         terminal_pairs;
    ValidationReport                  validation;
    DiscretizationResult              discretization;
    TissueMetrics                     tissue;
    /** Includes completed batch attempts from every calibration trial. */
    std::vector<GrowthStatistics> growth_statistics;
    CalibrationReport             calibration;
    /** Actual coupled Poiseuille solution, evaluated with venous drainage. */
    ReferenceResult reference;
    /** Demand-weighted design m^3/s by NEW Terminal id when reference_flow>0.
     * These are final constraints only for the tissue demand policy. */
    std::map<Id, double> target_terminal_flows;
    /** Desired m^3/s by NEW attachment Vessel id, using the same demand split.
     */
    std::map<Id, double> target_attachment_flows;
    /**
     * Consumed input terminal id -> new attachment vessel id, recorded on the
     * consumed terminal's native node owner. The vessel may belong elsewhere.
     */
    std::map<Id, Id>             attachment_roots;
    std::uint64_t                sampled_points      = 0;
    unsigned int                 accepted_insertions = 0;
    std::vector<TerminalTriplet> terminal_triplets;
    /** A cooperative stop always returns an invalid, non-adoptable result. */
    bool controlled_stop = false;
  };

  /**
   * Grow a CCO tree from an initial seed without changing the input. The
   * collective path starts an arterial tree from an empty input on its native
   * background owner and grows it across partitions. Imported arterial
   * attachments are selected and started collectively at their native node
   * owners. Venous growth starts collectively from real outlets or selected
   * input venous leaves, with exclusive common-territory assignments. A new
   * venous port is reserved before arterial growth. Terminal-count calibration
   * uses global volume and arterial or dual coverage, with the same count
   * decisions on all processes.
   * All paths use a degree-one mapping. Arterial
   * endpoints define the common territories; venous endpoints are separated
   * from them by the configured distance. All forests use the same CCO engine,
   * and all new vessels are checked against every system. Portal growth targets
   * the same territories, with its own separation and supply fraction. Every
   * input node and vessel stays fixed; selected tissue terminals become
   * attachment identities. Every new vessel wall, including rounded ends and
   * clearance, must lie inside the prepared background. An explicit inward port
   * normal permits a planar surface section: the envelope is cut at that plane,
   * and the opposite junction/end must clear it by the full radius plus
   * clearance. The normal must match an actual planar boundary face and point
   * into the domain. Imported exterior portions are preserved. Ambiguous
   * boundary proximity at the finite geometry budget is rejected
   * conservatively. Optional volume/coverage targets trigger a bounded,
   * deterministic search over terminal counts without monotonicity pruning.
   * A failed calibration returns an invalid result, including its best valid
   * trial when available, for diagnostics only; it must not be adopted.
   */
  GenerationResult
  generate(const dealii::Triangulation<3> &background,
           const dealii::Mapping<3>       &mapping,
           const NetworkData              &initial,
           const Parameters               &parameters,
           MPI_Comm                        communicator);

  /** Generate with optional accepted-state snapshot and cooperative controls.
   * Callbacks are invoked synchronously and never retained after return.
   * A stop request finishes the last accepted state, returns controlled_stop,
   * and bypasses further calibration trials; it does not impose a hard limit
   * on a batch or finalization already in progress.
   */
  GenerationResult
  generate(const dealii::Triangulation<3> &background,
           const dealii::Mapping<3>       &mapping,
           const NetworkData              &initial,
           const Parameters               &parameters,
           MPI_Comm                        communicator,
           const GenerationControl        &control);
  /**
   * Complete a network with supplied physical territory and bed metadata.
   * Each input record is supplied once globally; territories belong to their
   * arterial endpoint owner. Untouched records and resistances are retained.
   * The two endpoints of an existing bed must be extended together. Its bed
   * and territory are then replaced by the newly generated territories.
   * Remaining input beds enter both the design and final reference solves.
   * reference_flow specifies the total through NEW terminals only. In a valid
   * result, territories and pairs include untouched input records, while
   * desired-flow maps still refer only to new terminals and attachments. Input
   * beds require venous.enabled. Invalid input metadata throws collectively;
   * unsuccessful growth or incompatible reference flow returns an invalid
   * result.
   */
  GenerationResult
  generate(const dealii::Triangulation<3>  &background,
           const dealii::Mapping<3>        &mapping,
           const NetworkData               &initial,
           const Parameters                &parameters,
           MPI_Comm                         communicator,
           const std::vector<Territory>    &initial_territories,
           const std::vector<TerminalPair> &initial_pairs);

  /** Complete mixed two- and three-terminal compartments. All endpoints of a
   * compartment must be extended together; untouched compartments are retained.
   */
  GenerationResult
  generate(const dealii::Triangulation<3>     &background,
           const dealii::Mapping<3>           &mapping,
           const NetworkData                  &initial,
           const Parameters                   &parameters,
           MPI_Comm                            communicator,
           const std::vector<Territory>       &initial_territories,
           const std::vector<TerminalPair>    &initial_pairs,
           const std::vector<TerminalTriplet> &initial_triplets);

  /**
   * Continue portal/venous growth from a materialized physical checkpoint,
   * retaining every original node/vessel, untouched terminal and completed
   * artery. The original registry must contain no beds, and generated
   * components must start at its real non-port tissue terminals.
   * Checkpoint-only vessel ids identify generated descendants; imported records
   * with generated provenance remain frozen. The active phase is portal or
   * venous; starting at venous requires a complete portal forest when portal
   * growth is enabled.
   *
   * New arterial terminal identities, positions and demands define the common
   * territories. Their absolute count is preserved and must lie within the
   * configured terminal limits. Existing companion leaves count toward that
   * same total, never as a request for additional terminals. Every common
   * territory must finish with one terminal of each enabled system.
   *
   * Records are owned once globally, with ordinary ghosts. Physical bootstrap,
   * pending claims, geometry and final reference validation are collective.
   * Invalid or unsuccessful continuation returns an invalid diagnostic result;
   * malformed controls throw collectively. No arterial count calibration is
   * restarted. Search counters/caches and RNG streams start afresh: this is not
   * a bit-identical restart of an interrupted sampling sequence.
   */
  GenerationResult
  resume_companions(const dealii::Triangulation<3> &background,
                    const dealii::Mapping<3>       &mapping,
                    const NetworkData              &original,
                    const NetworkData              &checkpoint,
                    const Parameters               &parameters,
                    MPI_Comm                        communicator,
                    System                          active_companion);
} // namespace ImmersX::Vascularization
#endif
