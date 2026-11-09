// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_global_generation_h
#define immersx_vascularization_global_generation_h
#include <immersx/vascularization/generation.h>
#include <immersx/vascularization/global_generation_report.h>
#include <immersx/vascularization/global_geometry.h>

#include <functional>
namespace ImmersX::Vascularization
{
  /** Bounded global-forest search controls, separate from physical parameters.
   */
  struct GlobalGenerationParameters : dealii::ParameterAcceptor
  {
    explicit GlobalGenerationParameters(
      const std::string &section = "/Vascularization/Global forest/");
    void
                 validate() const;
    unsigned int initial_terminals = 16, topology_trials = 64, random_seed = 1,
                 pool_attempt_budget    = 100000;
    std::uint64_t maximum_group_records = 200000;
    /** Relative to the initially optimized batch objective; not a paper
     * constant. */
    double initial_temperature = .01, cooling_factor = .95;
    /** Zero geometry minimum uses the existing minimum_length/mesh-scale
     * policy; zero maximum uses the global background bounding-box diagonal.
     * The resolved bounds must be positive and ordered.
     */
    GlobalGeometryParameters geometry;
    /** Collective on all ranks, with no filesystem policy in the generator.
     * Drafts/intermediate stages are never a complete valid GenerationResult.
     */
    std::function<void(const GenerationResult &, GlobalStage)> stage_callback;
  };

  /** Sample one common pool before topology construction, grow three
   * independent forests in doubled prefixes and repair their geometry jointly.
   * Original physical records and fixed leaf IDs/positions are retained.
   * Temporary per-attachment-group topology views have an explicit record
   * budget; physical data return to native distributed ownership.
   *
   * A successful result requires the complete target, joint geometry,
   * native reference/discretization and calibration gates. Finite exhaustion
   * returns an invalid diagnostic result. SA is bounded and makes no global
   * optimum claim. All calls/callbacks are collective, including empty ranks.
   */
  GenerationResult
  generate_global(const dealii::Triangulation<3>   &background,
                  const dealii::Mapping<3>         &mapping,
                  const NetworkData                &original,
                  const Parameters                 &parameters,
                  const GlobalGenerationParameters &controls,
                  MPI_Comm                          communicator);
} // namespace ImmersX::Vascularization
#endif
