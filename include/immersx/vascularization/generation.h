// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_generation_h
#define immersx_vascularization_generation_h

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <immersx/vascularization/discretization.h>
#include <immersx/vascularization/metrics.h>
#include <immersx/vascularization/network.h>
#include <immersx/vascularization/reference_flow.h>

namespace ImmersX::Vascularization
{
  struct CalibrationParameters
  {
    void
    validate() const;

    /** Zero disables the volume target. */
    double volume_fraction    = 0.;
    double relative_tolerance = .05;
    /** Minimum service coverage: arterial, or dual in arterio-venous mode. */
    double       minimum_coverage  = 0.;
    unsigned int minimum_terminals = 1;
    unsigned int maximum_terminals = 10000;
    unsigned int attempt_budget    = 12;
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

  struct VenousParameters
  {
    bool             enabled = false;
    std::vector<Id>  selected_terminals;
    dealii::Point<3> seed_position, seed_inward_port_normal;
    double           seed_radius     = 0.;
    double           murray_exponent = 3.;
    /** Metres between the arterial endpoint and its venous partner. */
    double terminal_separation = 0.;
    /** Pa s/m^3 for each bed; zero derives individual beds from demand flow. */
    double microvascular_resistance = 0.;
  };

  struct Parameters : dealii::ParameterAcceptor
  {
    explicit Parameters(const std::string &section = "/Vascularization/");
    void
    validate(bool creating_seed = true) const;

    DiscretizationParameters discretization;
    CoverageParameters       coverage;
    CalibrationParameters    calibration;
    VenousParameters         venous;
    /** Explicit port pressures and optional loads on unpaired input terminals.
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
    double           seed_radius              = 0.;
    double           seed_length              = 0.;
    double           viscosity                = 0.;
    double           clearance                = 0.;
    unsigned int     geometry_depth           = 18;
    unsigned int     geometry_work            = 100000;
    unsigned int     terminal_count           = 1;
    unsigned int     random_seed              = 1;
    unsigned int     sample_budget            = 2000;
    unsigned int     candidate_count          = 12;
    unsigned int     optimization_steps       = 10;
    double           arterial_murray_exponent = 3.;
    double           minimum_radius           = 0.;
    double           minimum_length           = 0.;
    double           minimum_angle            = 15.;
    double           dissipation_weight       = 0.;
    double           density_beta             = 0.;
    std::string      terminal_weight          = "1";
    double           terminal_weight_bound    = 1.;
    /** Positive dimensionless tissue-demand expression in x,y,z. */
    std::string tissue_demand = "1";
    /** Total m^3/s through NEW tissue terminals; zero leaves flow unspecified.
     */
    double reference_flow = 0.;
  };

  struct Territory
  {
    Id                          id = invalid_id;
    dealii::Point<3>            position;
    dealii::types::subdomain_id owner  = 0;
    double                      demand = 1.;
  };

  struct GenerationResult
  {
    NetworkData               network;
    std::vector<Territory>    territories;
    std::vector<TerminalPair> terminal_pairs;
    ValidationReport          validation;
    DiscretizationResult      discretization;
    TissueMetrics             tissue;
    CalibrationReport         calibration;
    /** Actual paired Poiseuille solution, evaluated in arterio-venous mode. */
    ReferenceResult reference;
    /** Desired m^3/s by NEW Terminal id, populated when reference_flow>0. */
    std::map<Id, double> target_terminal_flows;
    /** Desired m^3/s by NEW attachment Vessel id, using the same demand split.
     */
    std::map<Id, double> target_attachment_flows;
    /**
     * Consumed input terminal id -> new attachment vessel id, recorded on the
     * consumed terminal's native node owner. The vessel may belong elsewhere.
     */
    std::map<Id, Id> attachment_roots;
    std::uint64_t    sampled_points      = 0;
    unsigned int     accepted_insertions = 0;
  };

  /**
   * Grow a CCO tree from an initial seed without changing the input. The
   * collective path starts an arterial tree from an empty input on its native
   * background owner and grows it across partitions. Imported arterial
   * attachments are selected and started collectively at their native node
   * owners. Arterial terminal-count calibration uses global metrics and the
   * same count decisions on all processes. Venous startup currently requires
   * one process.
   * Both paths use a degree-one mapping. Arterial
   * endpoints define the common territories; venous endpoints are separated
   * from them by the configured distance. Both forests use the same CCO engine,
   * and all new vessels are checked against both systems. Every input node and
   * vessel stays fixed; selected tissue terminals become attachment identities.
   * Every new vessel wall, including rounded ends and clearance, must lie
   * inside the prepared background. An explicit inward port normal permits a
   * planar surface section: the envelope is cut at that plane, and the opposite
   * junction/end must clear it by the full radius plus clearance. The normal
   * must match an actual planar boundary face and point into the domain.
   * Imported exterior portions are preserved. Ambiguous
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
} // namespace ImmersX::Vascularization
#endif
