// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_generation_h
#define immersx_vascularization_generation_h

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <immersx/vascularization/network.h>

namespace ImmersX::Vascularization
{
  struct Parameters : dealii::ParameterAcceptor
  {
    explicit Parameters(const std::string &section = "/Vascularization/");
    void
    validate() const;

    dealii::Point<3> seed_position;
    /** Zero selects a deterministic direction into the containing cell. */
    dealii::Point<3> seed_direction;
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
  };

  struct Territory
  {
    Id                          id = invalid_id;
    dealii::Point<3>            position;
    dealii::types::subdomain_id owner  = 0;
    double                      demand = 1.;
  };

  struct TerminalPair
  {
    Id     territory  = invalid_id;
    Id     arterial   = invalid_id;
    Id     venous     = invalid_id;
    double resistance = 0.;
  };

  struct GenerationResult
  {
    NetworkData               network;
    std::vector<Territory>    territories;
    std::vector<TerminalPair> terminal_pairs;
    ValidationReport          validation;
    std::uint64_t             sampled_points      = 0;
    unsigned int              accepted_insertions = 0;
  };

  /**
   * Grow a CCO tree from an initial seed without changing the input. The
   * current serial path accepts an empty network and a degree-one mapping. The
   * entire swept vessel, including its rounded ends and clearance, must lie
   * inside the prepared background. Ambiguous boundary proximity at the finite
   * geometry budget is rejected conservatively.
   */
  GenerationResult
  generate(const dealii::Triangulation<3> &background,
           const dealii::Mapping<3>       &mapping,
           const NetworkData              &initial,
           const Parameters               &parameters,
           MPI_Comm                        communicator);
} // namespace ImmersX::Vascularization
#endif
