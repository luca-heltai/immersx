// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_application_h
#define immersx_vascularization_application_h
#include <immersx/vascularization/generation.h>

namespace ImmersX::Vascularization
{
  /** Configuration of the standalone driver. All lengths and physical data
   * use SI units. The domain is prepared before generation and never adapted
   * in response to growth. Library callers supply absolute archive/output
   * paths; the executable resolves relative paths against its parameter file.
   */
  struct ApplicationParameters : dealii::ParameterAcceptor
  {
    explicit ApplicationParameters(
      const std::string &section = "/Vascularization/");
    Parameters       generation;
    std::string      mode  = "generate", input_archive, output_directory;
    std::string      shape = "cube", refinement_focus = "none";
    dealii::Point<3> cube_origin;
    double           cube_side = 1.;
    dealii::Point<3> sphere_center{.5, .5, .5};
    double           sphere_radius       = .5;
    unsigned int     uniform_refinements = 2, edge_axis = 2;
    std::vector<unsigned int> edge_sides{0, 0};
    std::vector<double>       refinement_fractions{.5, .25, .125};
    bool                      evaluate_reference = true;
    /** Total m^3/s through ALL unpaired terminals in an arterial load run.
     * Zero uses only the explicit reference boundary data. */
    double loaded_reference_flow = 0.;
    bool   save_archive = true, write_tables = true, write_meshes = true;
  };

  /** Run generate, complete or load mode collectively, including empty ranks.
   * Write per-rank physical/numerical/background CSV tables and optional native
   * distributed line/background VTU meshes. A new output directory is required.
   * Only valid results are archived. Invalid generation/reference results get
   * a summary and per-rank diagnostics and must not be adopted by the caller.
   * Configuration and filesystem errors throw collectively.
   */
  GenerationResult
  run(const ApplicationParameters &parameters, MPI_Comm communicator);
} // namespace ImmersX::Vascularization
#endif
